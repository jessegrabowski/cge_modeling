import functools as ft
import inspect
import logging
import math
import os.path
import re
import warnings

from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from numbers import Integral
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import xarray as xr

from better_optimize import minimize, root
from better_optimize.constants import minimize_method, root_method
from better_optimize.utilities import ToggleableProgress
from cloudpickle import cloudpickle
from IPython.display import Latex
from pytensor.compile.mode import Mode, get_mode
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.basic import xtensor_from_tensor
from scipy.optimize import OptimizeResult

from cge_modeling.base.context import ModelObjectVariable, XTensorEquation, model_stack
from cge_modeling.base.function_wrappers import (
    select_free_variable_entries,
    wrap_compiled_function_for_scipy,
)
from cge_modeling.base.primitives import Parameter, Variable
from cge_modeling.base.utilities import (
    _replace_dim_marker_with_dim_name,
    create_final_param_dict,
    ensure_input_is_sequence,
    expand_obj_by_indices,
    flat_array_to_variable_dict,
    get_method_defaults,
    infer_object_shape_from_coords,
    make_flat_array_return_mask,
    make_indexed_name,
    unpack_equation_name,
    variable_dict_to_flat_array,
)
from cge_modeling.compile.constants import CompiledFunctions
from cge_modeling.solvers.sparse_root import sparse_root
from cge_modeling.tools.latex import equation_to_latex
from cge_modeling.tools.output_tools import (
    datatree_to_value_dict,
    display_table,
    list_of_array_to_idata,
    optimizer_result_to_idata,
)
from cge_modeling.tools.progress import (
    MINIMIZE_STATISTICS,
    ROOT_STATISTICS,
    simulation_progress,
)

_log = logging.getLogger(__name__)


def _parse_compile_kwarg(
    functions_to_compile: CompiledFunctions
    | Sequence[CompiledFunctions]
    | Literal[True]
    | None = None,
) -> Sequence[CompiledFunctions] | None:
    if functions_to_compile is None:
        return None
    if functions_to_compile is True or functions_to_compile == "all":
        return ["root", "minimize", "euler"]
    if isinstance(functions_to_compile, str):
        return [functions_to_compile]
    return functions_to_compile


def _judge_equilibrium(
    res: OptimizeResult, squared_error: float, equilibrium_tol: float
) -> OptimizeResult:
    """Set ``res.success`` from the sum of squared residuals, noting in ``res.message`` when this overrules the solver."""
    # A solver can stop on a flat objective well short of an equilibrium, or fail to make progress at one, so
    # success is judged on the residuals themselves.
    is_equilibrium = squared_error < equilibrium_tol
    if bool(res.success) != is_equilibrium:
        res.message = (
            f"{res.message} Sum of squared residuals is {squared_error:.3e}, "
            f"{'below' if is_equilibrium else 'above'} equilibrium_tol={equilibrium_tol:g}."
        )
    res.success = is_equilibrium
    return res


def _final_state(tree: xr.DataTree) -> dict[str, np.ndarray]:
    """Return the values of every variable and parameter at the last step of a simulation group."""
    if "step" in tree["variables"].dims:
        tree = tree.isel(step=-1)
    return datatree_to_value_dict(tree)


class CGEModel:
    """
    A computable general equilibrium model: named variables, parameters and equations over labeled dimensions.

    Declare the model's objects with :func:`~cge_modeling.model.Variable`, :func:`~cge_modeling.model.Parameter` and
    :func:`~cge_modeling.model.Equation` inside a ``with model:`` block, or by passing ``model=model``. Variables are
    solved for, and parameters are held fixed. A model can be solved only when it is square, meaning it has as many
    variable entries as equation entries once every object is unpacked along its dims.

    Parameters
    ----------
    coords : dict of str to list of str, optional
        Labels of each dimension, keyed by dimension name, such as ``{"sector": ["Agriculture", "Industry"]}``.
        Default is None, which gives a model with no dimensions.

    Examples
    --------
    Declare a two-sector model in which each sector's output is its labor productivity times its labor:

    .. code-block:: python

        import cge_modeling as cm

        with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
            Y = cm.Variable("Y", dims="sector", description="Output of each sector")
            L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
            A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
            cm.Equation("Production", Y, A * L)

        model.n_variables, model.n_equations  # (2, 2)
    """

    def __init__(self, coords: dict[str, list[str]] | None = None):
        self.coords: dict[str, list[str]] = {} if coords is None else coords

        self._variables: dict[str, Variable] = {}
        self._parameters: dict[str, Parameter] = {}

        self._unpacked_variables: dict[str, dict[str, Any]] = {}
        self._unpacked_parameters: dict[str, dict[str, Any]] = {}
        self._unpacked_equations: dict[str, None] = {}

        self._model_objects: dict[str, ModelObjectVariable] = {}
        self._xtensor_equations: dict[str, XTensorEquation] = {}
        self._context_tokens: list = []

        self._compiled: Sequence[CompiledFunctions] = []
        self._compile_settings: dict[str, Any] = {}
        self.mode: Mode | str | None = None

        self.f_system: Callable | None = None
        self.f_resid: Callable | None = None
        self.f_grad: Callable | None = None
        self.f_hess: Callable | None = None
        self.f_hessp: Callable | None = None
        self.f_jac: Callable | None = None
        self.f_euler: Callable | None = None
        self.f_sparse_system_and_jac: Callable | None = None

    @property
    def n_variables(self) -> int:
        """The number of variable entries once every variable is unpacked along its dims."""
        return len(self.unpacked_variable_names)

    @property
    def n_parameters(self) -> int:
        """The number of parameter entries once every parameter is unpacked along its dims."""
        return len(self.unpacked_parameter_names)

    @property
    def n_equations(self) -> int:
        """The number of equation entries once every equation is unpacked along its dims."""
        return len(self.unpacked_equation_names)

    def __enter__(self) -> "CGEModel":
        self._context_tokens.append(model_stack.set((*model_stack.get(), self)))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        model_stack.reset(self._context_tokens.pop())

    def __getitem__(self, name: str) -> ModelObjectVariable:
        """Return the variable or parameter named ``name`` that was registered in a ``with`` block."""
        try:
            return self._model_objects[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not a variable or parameter registered in a `with` block on this model."
            ) from None

    def label_positions(self, dim: str, labels: Any) -> int | list[int]:
        """
        Map one label, or a list of labels, of ``dim`` to positions along it.

        Parameters
        ----------
        dim : str
            Name of a dimension in :attr:`coords`.
        labels : str or sequence of str
            One label of ``dim``, or several.

        Returns
        -------
        positions : int or list of int
            The position of ``labels`` along ``dim``, or a list of positions when ``labels`` is a sequence.

        Examples
        --------
        Find where a sector sits along the ``sector`` dimension, for indexing a flat array of results:

        .. code-block:: python

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            model.label_positions("sector", "Industry")  # 1
            model.label_positions("sector", ["Industry", "Agriculture"])  # [1, 0]
        """
        if dim not in self.coords:
            raise ValueError(f"Unknown dimension {dim!r}. Model dims are {list(self.coords)}.")

        dim_labels = list(self.coords[dim])

        def position(label):
            if label not in dim_labels:
                raise KeyError(f"{label!r} is not a label of dimension {dim!r}: {dim_labels}.")
            return dim_labels.index(label)

        if isinstance(labels, str) or not isinstance(labels, Sequence | np.ndarray | pd.Index):
            return position(labels)
        return [position(label) for label in labels]

    def __copy__(self) -> "CGEModel":
        return self.copy()

    def __deepcopy__(self, memo) -> "CGEModel":
        return self.copy()

    def copy(self) -> "CGEModel":
        """
        Return an uncompiled copy of the model, with its own variables, parameters and equations.

        The copy's objects are retrieved by name, as ``copy["name"]``. Changes to the copy, such as new equations or
        :meth:`reassign_roles`, leave this model unchanged.

        Returns
        -------
        copy : CGEModel
            An uncompiled model with the same coords, variables, parameters and equations.

        Examples
        --------
        Extend a copy with a new equation while the original keeps its own:

        .. code-block:: python

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            extended = model.copy()
            with extended:
                cm.Variable("wage_bill", description="Total labor employed")
                cm.Equation("Wage bill", extended["wage_bill"], extended["L"].sum("sector"))

            extended.n_equations, model.n_equations  # (3, 2)
        """
        clone = CGEModel(coords={dim: list(labels) for dim, labels in self.coords.items()})

        replacements = {}
        for obj in [*self.variables, *self.parameters]:
            cloned_object = clone._register_model_object(
                name=obj.name,
                dims=obj.dims,
                kind="variable" if isinstance(obj, Variable) else "parameter",
                description=obj.description,
                latex_name=obj.latex_name,
                extend_subscript=obj.extend_subscript,
            )
            replacements[self._model_objects[obj.name].owner.inputs[0]] = (
                cloned_object.owner.inputs[0]
            )

        for equation in self.equations:
            lhs, rhs = graph_replace(
                [equation.lhs, equation.rhs], replace=replacements, strict=False
            )
            clone._register_model_equation(name=equation.name, lhs=lhs, rhs=rhs)

        return clone

    def reassign_roles(
        self, *, to_parameters: str | Sequence[str], to_variables: str | Sequence[str]
    ) -> "CGEModel":
        """
        Turn some variables into parameters and some parameters into variables, keeping the equations.

        The compiled functions are discarded, so the model has to be compiled again before it is solved. The model
        is unchanged if any name is invalid.

        Parameters
        ----------
        to_parameters : str or sequence of str
            Names of variables that become parameters.
        to_variables : str or sequence of str
            Names of parameters that become variables.

        Returns
        -------
        model : CGEModel
            This model, with the new roles.

        Examples
        --------
        Calibrate productivity from observed output by making ``A`` a variable and ``Y`` a parameter:

        .. code-block:: python

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            model.reassign_roles(to_parameters="Y", to_variables="A")
            model.variable_names, model.parameter_names  # (['A'], ['L', 'Y'])
        """
        to_parameters = list(ensure_input_is_sequence(to_parameters))
        to_variables = list(ensure_input_is_sequence(to_variables))

        name_counts = Counter(to_parameters + to_variables)
        repeated = sorted(name for name, count in name_counts.items() if count > 1)
        if repeated:
            raise ValueError(
                f"Each name can be reassigned once, and {repeated} appear more than once."
            )
        not_variables = [name for name in to_parameters if name not in self.variable_names]
        if not_variables:
            raise ValueError(
                f"Only variables can become parameters, and {not_variables} are not variables of the model."
            )
        not_parameters = [name for name in to_variables if name not in self.parameter_names]
        if not_parameters:
            raise ValueError(
                f"Only parameters can become variables, and {not_parameters} are not parameters of the model."
            )
        n_to_parameters = len(self._unpack_objects(self.get_variables(to_parameters)))
        n_to_variables = len(self._unpack_objects(self.get_parameters(to_variables)))
        if n_to_parameters != n_to_variables:
            raise ValueError(
                f"The model would not be square: {to_parameters} have {n_to_parameters} entries, but "
                f"{to_variables} have {n_to_variables}."
            )

        for name in to_parameters:
            self._move_object(name, kind="parameter")
        for name in to_variables:
            self._move_object(name, kind="variable")
        self._clear_compiled_functions()

        return self

    def _move_object(self, name: str, kind: Literal["variable", "parameter"]) -> None:
        source = "parameters" if kind == "variable" else "variables"
        obj = getattr(self, f"_{source}").pop(name)
        for unpacked in self._unpack_objects([obj]):
            del getattr(self, f"_unpacked_{source}")[make_indexed_name(unpacked)]

        object_class = Variable if kind == "variable" else Parameter
        moved = object_class(
            name=obj.name,
            dims=obj.dims,
            description=obj.description,
            latex_name=obj.latex_name,
            extend_subscript=obj.extend_subscript,
        )
        self._store_object(moved)

    def _store_object(self, obj: Variable | Parameter) -> None:
        group = "variables" if isinstance(obj, Variable) else "parameters"
        getattr(self, f"_{group}")[obj.name] = obj
        getattr(self, f"_unpacked_{group}").update(
            {
                make_indexed_name(unpacked): {"modelobj": unpacked}
                for unpacked in self._unpack_objects([obj])
            }
        )

    def _clear_compiled_functions(self) -> None:
        self._compiled = []
        self.mode = None
        self.f_system = None
        self.f_resid = None
        self.f_grad = None
        self.f_hess = None
        self.f_hessp = None
        self.f_jac = None
        self.f_euler = None
        self.f_sparse_system_and_jac = None

    def _register_model_object(
        self,
        name: str,
        dims: str | list[str] | tuple[str, ...] | None,
        kind: Literal["variable", "parameter"],
        description: str | None,
        latex_name: str | None,
        extend_subscript: bool | int,
    ) -> ModelObjectVariable:
        if name in self.variable_names + self.parameter_names:
            raise ValueError(f"The model already has a variable or parameter named {name!r}.")
        if name in self.coords:
            raise ValueError(f"Name {name!r} conflicts with an existing dimension name.")

        object_class = Variable if kind == "variable" else Parameter
        obj = object_class(
            name=name,
            dims=dims,
            description=description,
            latex_name=latex_name,
            extend_subscript=extend_subscript,
        )

        normalized_dims = cast(tuple[str, ...], obj.dims)
        unknown_dims = [dim for dim in normalized_dims if dim not in self.coords]
        if unknown_dims:
            raise ValueError(
                f"{name!r} has dims {unknown_dims} that are not in the model coords {list(self.coords)}."
            )

        tensor = pt.tensor(name, shape=infer_object_shape_from_coords(obj, self.coords))
        xtensor = xtensor_from_tensor(tensor, dims=normalized_dims, name=name)
        # pytensor builds the output variable from its type, so the subclass is attached afterward. It adds methods
        # only, and graph clones made during compilation are plain XTensorVariables, which is all they need to be.
        xtensor.__class__ = ModelObjectVariable

        self._store_object(obj)
        self._model_objects[name] = xtensor
        return xtensor

    def _register_model_equation(self, name: str, lhs: Any, rhs: Any) -> XTensorEquation:
        if name in self.equation_names:
            raise ValueError(f"The model already has an equation named {name!r}.")
        for side, value in (("lhs", lhs), ("rhs", rhs)):
            if isinstance(value, bool):
                raise TypeError(
                    f"{side} of equation {name!r} is a bool. `==` does not build equations; pass the two "
                    "sides separately, as Equation(name, lhs, rhs)."
                )

        lhs, rhs = as_xtensor(lhs), as_xtensor(rhs)
        residual = lhs - rhs

        unknown_dims = [dim for dim in residual.type.dims if dim not in self.coords]
        if unknown_dims:
            raise ValueError(
                f"Equation {name!r} has dims {unknown_dims} that are not in the model coords {list(self.coords)}."
            )

        model_tensors = {obj.owner.inputs[0] for obj in self._model_objects.values()}
        inputs = list(explicit_graph_inputs([residual]))
        if not inputs:
            raise ValueError(f"Equation {name!r} uses no variables or parameters of the model.")
        foreign_inputs = [variable for variable in inputs if variable not in model_tensors]
        if foreign_inputs:
            raise ValueError(
                f"Equation {name!r} uses inputs that are not objects of this model: {foreign_inputs}."
            )

        residual = residual.transpose(*[dim for dim in self.coords if dim in residual.type.dims])

        element_names = unpack_equation_name(name, residual.type.dims, self.coords)
        duplicates = [element for element in element_names if element in self._unpacked_equations]
        if duplicates:
            raise ValueError(
                f"Equation {name!r} repeats existing equation element names {duplicates}."
            )

        equation = XTensorEquation(name=name, lhs=lhs, rhs=rhs, residual=residual)
        self._xtensor_equations[name] = equation
        self._unpacked_equations.update(dict.fromkeys(element_names))
        return equation

    def _compiled_function(
        self,
        name: Literal["f_system", "f_jac", "f_resid", "f_grad", "f_hess", "f_hessp", "f_euler"],
    ) -> Callable[..., Any]:
        function = getattr(self, name)
        if function is None:
            raise ValueError(f"{name} has not been compiled.")
        return function

    def compile(
        self,
        mode: str | None = None,
        functions_to_compile: CompiledFunctions | Sequence[CompiledFunctions] | None = "all",
        use_scan_euler: bool = False,
        use_rk4: bool = False,
        use_sparse_matrices: bool = False,
    ) -> "CGEModel":
        """
        Compile the functions the solvers use: the system residuals, and the Jacobian, loss or Euler step each solver
        needs.

        Parameters
        ----------
        mode : str, optional
            PyTensor compilation mode. Default is PyTensor's default mode.
        functions_to_compile : {"root", "minimize", "euler", "all"} or sequence of them, optional
            Solver functions to compile, on top of the system residuals, which are always compiled. Default is
            ``"all"``.
        use_scan_euler : bool, optional
            Build the Euler approximation as a single scan graph. Default is False.
        use_rk4 : bool, optional
            Take fourth-order Runge-Kutta steps in the Euler approximation. Default is False.
        use_sparse_matrices : bool, optional
            Build the Jacobian as a sparse matrix and find roots with ``sparse_root``. Default is False.

        Returns
        -------
        model : CGEModel
            This model, with its compiled functions set.

        Examples
        --------
        Compile only the root-finding functions, which is all :meth:`simulate` needs when it skips the Euler approximation:

        .. code-block:: python

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            model.compile(functions_to_compile="root")
        """
        # cge_modeling.compile.pytensor imports this module, so it can only be imported here.
        from cge_modeling.compile.pytensor import compile_pytensor_cge_functions

        self.check_initialization()
        _log.info(
            f"Compiling a model with {self.n_equations} equations, {self.n_variables} variables, "
            f"{self.n_parameters} parameters"
        )

        compile_list = list(_parse_compile_kwarg(functions_to_compile) or [])

        # TODO: 'inplace_elemwise_optimizer' was triggering an infinite loop during graph rewrites, but that's probably
        #  a bug in PyTensor. Remove this once the bug is fixed.
        compile_mode = get_mode(mode).excluding("inplace_elemwise_optimizer")

        (
            self.f_system,
            self.f_jac,
            self.f_resid,
            self.f_grad,
            self.f_hess,
            self.f_hessp,
            self.f_euler,
            self.f_sparse_system_and_jac,
        ) = compile_pytensor_cge_functions(
            cge_model=self,
            functions_to_compile=compile_list,
            mode=compile_mode,
            use_scan_euler=use_scan_euler,
            use_rk4=use_rk4,
            use_sparse_matrices=use_sparse_matrices,
        )
        self._compiled = compile_list
        self._compile_settings = {
            "mode": mode,
            "use_scan_euler": use_scan_euler,
            "use_rk4": use_rk4,
            "use_sparse_matrices": use_sparse_matrices,
        }
        self.mode = compile_mode

        return self

    def _compile_missing(
        self, needed: set[CompiledFunctions], compile_kwargs: dict[str, Any] | None
    ) -> None:
        """Compile the ``needed`` functions the model lacks, or recompile with ``compile_kwargs`` when given."""
        if compile_kwargs is None and self.f_system is not None and needed <= set(self._compiled):
            return
        settings = {**self._compile_settings, **(compile_kwargs or {})}
        requested = _parse_compile_kwarg(settings.pop("functions_to_compile", None)) or []
        # Compiling replaces every function, so the ones compiled earlier are compiled again with the rest.
        functions: list[CompiledFunctions] = sorted({*self._compiled, *needed, *requested})
        self.compile(**settings, functions_to_compile=functions)

    def check_initialization(self) -> None:
        """
        Check that the model is square, with as many variable entries as equation entries.

        Examples
        --------
        Check a model before compiling it. A model with a missing equation fails here:

        .. code-block:: python

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            model.check_initialization()
        """
        if self.n_variables != self.n_equations:
            raise ValueError(
                f"Found {self.n_variables} variables but {self.n_equations} equations. "
                "System is not square. Check your model equations."
            )

    @property
    def variable_names(self) -> list[str]:
        """The names of the model's variables, in declaration order."""
        return list(self._variables.keys())

    @property
    def variables(self) -> list[Variable]:
        """The model's variables, in declaration order."""
        return list(self._variables.values())

    @property
    def unpacked_variable_names(self) -> list[str]:
        """The names of the variable entries, one per label combination, such as ``"Y_Agriculture"``."""
        return list(self._unpacked_variables.keys())

    @property
    def unpacked_variables(self) -> list[Variable]:
        """The variable entries, one per label combination."""
        return [
            self._unpacked_variables[var_name]["modelobj"]
            for var_name in self.unpacked_variable_names
        ]

    @property
    def parameter_names(self) -> list[str]:
        """The names of the model's parameters, in declaration order."""
        return list(self._parameters.keys())

    @property
    def parameters(self) -> list[Parameter]:
        """The model's parameters, in declaration order."""
        return list(self._parameters.values())

    @property
    def unpacked_parameter_names(self) -> list[str]:
        """The names of the parameter entries, one per label combination, such as ``"A_Industry"``."""
        return list(self._unpacked_parameters.keys())

    @property
    def unpacked_parameters(self) -> list[Parameter]:
        """The parameter entries, one per label combination."""
        return [
            self._unpacked_parameters[param_name]["modelobj"]
            for param_name in self.unpacked_parameter_names
        ]

    @property
    def equation_names(self) -> list[str]:
        """The names of the model's equations, in declaration order."""
        return list(self._xtensor_equations)

    @property
    def equations(self) -> list[XTensorEquation]:
        """The equations :func:`~cge_modeling.model.Equation` registered on the model, in declaration order."""
        return list(self._xtensor_equations.values())

    @property
    def unpacked_equation_names(self) -> list[str]:
        """The names of the equation entries, one per label combination of each equation's dims."""
        return list(self._unpacked_equations.keys())

    def _unpack_objects(self, objs: Sequence[Variable | Parameter]):
        """
        Convert a single object with dimension index to a list of objects with coordinate indices.

        Parameters
        ----------
        objs : Variable | Parameter
            The object to be unpacked.

        Returns
        -------
        list[Variable | Parameter]
            A list of objects with coordinate indices.
        """

        expanded_objects = []
        for obj in objs:
            expanded_objs = expand_obj_by_indices(
                obj, self.coords, dims=None, on_unused_dim="ignore"
            )
            expanded_objects.extend(expanded_objs)

        return expanded_objects

    def _get_object(self, obj_name: str, group: Literal["variables", "parameters"]):
        return getattr(self, f"_{group}")[obj_name]

    def get_parameter(self, param_name: str) -> Parameter:
        """
        Return the parameter named ``param_name``.

        Examples
        --------
        Read a parameter's dims and description:

        .. code-block:: python

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            model.get_parameter("A").dims  # ('sector',)
        """
        return self._get_object(param_name, "parameters")

    def get_parameters(self, param_names: str | list[str] | None = None) -> list[Parameter]:
        """
        Return the parameters named in ``param_names``.

        Parameters
        ----------
        param_names : str or list of str, optional
            Names of the parameters to return. Default is None, which returns every parameter.

        Returns
        -------
        parameters : list of Parameter
            The requested parameters, in the order of ``param_names``.

        Examples
        --------
        List every parameter's name:

        .. code-block:: python

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            [parameter.name for parameter in model.get_parameters()]  # ['L', 'A']
        """
        if param_names is None:
            param_names = self.parameter_names
        return [self.get_parameter(name) for name in ensure_input_is_sequence(param_names)]

    def get_variable(self, var_name: str) -> Variable:
        """
        Return the variable named ``var_name``.

        Examples
        --------
        Read a variable's description:

        .. code-block:: python

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            model.get_variable("Y").description  # 'Output of each sector'
        """
        return self._get_object(var_name, "variables")

    def get_variables(self, var_names: str | list[str] | None = None) -> list[Variable]:
        """
        Return the variables named in ``var_names``.

        Parameters
        ----------
        var_names : str or list of str, optional
            Names of the variables to return. Default is None, which returns every variable.

        Returns
        -------
        variables : list of Variable
            The requested variables, in the order of ``var_names``.

        Examples
        --------
        List every variable's name:

        .. code-block:: python

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            [variable.name for variable in model.get_variables()]  # ['Y']
        """
        if var_names is None:
            var_names = self.variable_names
        return [self.get_variable(name) for name in ensure_input_is_sequence(var_names)]

    def get(
        self, names: str | list[str] | None = None
    ) -> Variable | Parameter | list[Variable | Parameter]:
        """
        Return variables or parameters by name.

        Parameters
        ----------
        names : str or list of str, optional
            Names of variables or parameters. Default is None, which returns every parameter and variable.

        Returns
        -------
        out : Variable, Parameter, or list of Variable and Parameter
            The requested object when one name is given, otherwise the list of requested objects.

        Examples
        --------
        Fetch a variable and a parameter in one call:

        .. code-block:: python

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            output, productivity = model.get(["Y", "A"])
        """
        if names is None:
            names = self.parameter_names + self.variable_names
        if isinstance(names, str):
            names = [names]

        out: list[Variable | Parameter] = []
        for name in names:
            if name in self.variable_names:
                out.append(self._variables[name])
            elif name in self.parameter_names:
                out.append(self._parameters[name])
            else:
                raise ValueError(f'Requested name "{name}" is not a known parameter or variable.')
        if len(out) == 1:
            return out[0]

        return out

    def get_shape(self, name: str) -> Iterator[int]:
        """
        Yield the length of each dim of the variable or parameter named ``name``.

        Examples
        --------
        Find the shape an array of values for ``Y`` must have:

        .. code-block:: python

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            tuple(model.get_shape("Y"))  # (2,)
        """
        obj = self.get(name)
        return (len(self.coords[dim]) for dim in obj.dims)

    def summary(
        self,
        variables: Literal["all", "variables", "parameters"] | list[str] = "all",
    ) -> None:
        """
        Display a table of the symbols and descriptions of model objects.

        Parameters
        ----------
        variables : {"all", "variables", "parameters"} or list of str, optional
            The objects to list. Default is ``"all"``, which lists every variable and parameter.

        Examples
        --------
        Show every object of a model with its description:

        .. code-block:: python

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            model.summary()
        """

        if variables == "all":
            variables = [
                x.name
                for value in ["variables", "parameters"]
                for x in getattr(self, f"get_{value}")()
            ]

        if variables in ["variables", "parameters"]:
            variables = [x.name for x in getattr(self, f"get_{variables}")()]

        info_dict: dict[str, Any] = {"Symbol": [], "Description": []}

        for var_name in variables:
            item: Variable | Parameter
            if var_name in self.parameter_names:
                item = self._parameters[var_name]
            elif var_name in self.variable_names:
                item = self._variables[var_name]
            else:
                raise ValueError(f"{var_name} is not a variable or parameter of the model.")
            info_dict["Symbol"].append(item)
            info_dict["Description"].append(item.description)

        info_dict["Description"] = [
            _replace_dim_marker_with_dim_name(desc) for desc in info_dict["Description"]
        ]

        display_table(info_dict)

    def equation_table(self) -> None:
        """
        Display a table of the model's equations, numbered, with their names and LaTeX forms.

        Examples
        --------
        Typeset the equations of a model:

        .. code-block:: python

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            model.equation_table()
        """
        info_dict: dict[str, list[Any]] = {"": [], "Name": [], "Equation": []}
        for number, equation in enumerate(self._xtensor_equations.values(), start=1):
            info_dict[""].append(number)
            info_dict["Name"].append(_replace_dim_marker_with_dim_name(equation.name))
            info_dict["Equation"].append(Latex(equation_to_latex(equation, model=self)))

        display_table(info_dict, column_widths=["4%", "22%", "74%"])

    def calibrate(
        self,
        data: Mapping[str, float | np.ndarray] | None = None,
        max_iter: int = 100,
        initial_values: Mapping[str, float | np.ndarray] | None = None,
    ) -> xr.DataTree:
        """
        Solve for every variable and parameter value that ``data`` does not give.

        Parameters
        ----------
        data : dict of str to float or ndarray, optional
            Known values, keyed by variable or parameter name.
        max_iter : int, optional
            Iteration limit of the solver. Default is 100.
        initial_values : dict of str to float or ndarray, optional
            Starting values for unknowns, keyed by name.

        Returns
        -------
        state : xr.DataTree
            The given and solved values in ``variables`` and ``parameters`` children, which :meth:`simulate` accepts
            as its ``initial_state``. The root holds ``converged``, which is True when every equation holds.

        Examples
        --------
        Calibrate productivity from observed output and labor:

        .. code-block:: python

            import numpy as np

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            state = model.calibrate({"Y": np.array([100.0, 200.0]), "L": np.array([50.0, 40.0])})
            state["parameters"]["A"]  # [2., 5.]
        """
        # cge_modeling.tools.calibration imports the compile modules, which import this module.
        from cge_modeling.tools.calibration import (
            calibrate_entries,
            compile_calibration_system,
            residual_is_converged,
        )

        data = {} if data is None else data
        initial_values = {} if initial_values is None else initial_values

        system = compile_calibration_system(self)
        entries, residual = calibrate_entries(
            system=system, data=data, initial_values=initial_values, max_iter=max_iter
        )
        values = flat_array_to_variable_dict(entries, self.variables + self.parameters, self.coords)

        def group(objects: list[Variable] | list[Parameter]) -> xr.Dataset:
            return xr.Dataset(
                {obj.name: (obj.dims, values[obj.name]) for obj in objects}, coords=self.coords
            )

        return xr.DataTree.from_dict(
            {
                "/": xr.Dataset({"converged": residual_is_converged(residual)}),
                "variables": group(self.variables),
                "parameters": group(self.parameters),
            }
        )

    def _step_statistics(
        self, optimizer_mode: Literal["root", "minimize"]
    ) -> Callable[[np.ndarray, np.ndarray], dict[str, float]]:
        """Return a function that computes, at a model state, the statistics the given optimizer mode reports."""
        if optimizer_mode == "minimize":
            f_resid = self._compiled_function("f_resid")
            f_grad = self._compiled_function("f_grad")

            def minimize_statistics(x, theta):
                return {
                    "f_value": float(np.asarray(f_resid(x, theta))),
                    "grad_norm": float(np.linalg.norm(f_grad(x, theta))),
                }

            return minimize_statistics

        f_system = self._compiled_function("f_system")

        def root_statistics(x, theta):
            if self.f_sparse_system_and_jac is not None:
                residuals, jacobian = self.f_sparse_system_and_jac(x, theta)
            else:
                residuals = np.asarray(f_system(x, theta))
                jacobian = None if self.f_jac is None else np.asarray(self.f_jac(x, theta))

            statistics = {"f_value": float(np.linalg.norm(residuals))}
            if jacobian is not None:
                statistics["grad_norm"] = float(np.linalg.norm(jacobian.T @ residuals))
            return statistics

        return root_statistics

    def _statistic_headings(self, optimizer_mode: Literal["root", "minimize"]) -> dict[str, str]:
        if optimizer_mode == "minimize":
            return MINIMIZE_STATISTICS
        has_jacobian = self.f_jac is not None or self.f_sparse_system_and_jac is not None
        return ROOT_STATISTICS if has_jacobian else {"f_value": ROOT_STATISTICS["f_value"]}

    def _euler_trajectory(
        self,
        x0: np.ndarray,
        theta0: np.ndarray,
        theta_final: np.ndarray,
        n_steps: int,
        progress: ToggleableProgress,
        optimizer_mode: Literal["root", "minimize"],
    ) -> list[np.ndarray]:
        f_euler = self._compiled_function("f_euler")

        # The scan Euler approximation is one compiled graph with no Python loop to report progress from.
        if not isinstance(f_euler, ft.partial):
            return f_euler(x0, theta0, theta_final, n_steps)

        return f_euler(
            x0,
            theta0,
            theta_final,
            n_steps,
            progress=progress,
            step_statistics=self._step_statistics(optimizer_mode),
        )

    def _solve_with_euler_approximation(
        self,
        data,
        theta_final,
        n_steps,
        progressbar: bool | ToggleableProgress = True,
        optimizer_mode: Literal["root", "minimize"] = "root",
        solve_every: int | None = None,
        solve_last_step: bool = False,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        x0, theta0 = variable_dict_to_flat_array(data, self.variables, self.parameters)
        progress = (
            simulation_progress(self._statistic_headings(optimizer_mode), enabled=progressbar)
            if isinstance(progressbar, bool)
            else progressbar
        )

        if solve_every is None:
            trajectory = self._euler_trajectory(
                x0, theta0, theta_final, n_steps, progress, optimizer_mode
            )
            return list_of_array_to_idata(trajectory, self)

        return self._solve_with_euler_checkpoints(
            x0,
            theta0,
            theta_final,
            n_steps=n_steps,
            solve_every=solve_every,
            solve_last_step=solve_last_step,
            progress=progress,
            optimizer_mode=optimizer_mode,
            optimizer_kwargs={} if optimizer_kwargs is None else optimizer_kwargs,
        )

    def _solve_with_euler_checkpoints(
        self,
        x0: np.ndarray,
        theta0: np.ndarray,
        theta_final: np.ndarray,
        n_steps: int,
        solve_every: int,
        solve_last_step: bool,
        progress: ToggleableProgress,
        optimizer_mode: Literal["root", "minimize"],
        optimizer_kwargs: dict[str, Any],
    ) -> xr.DataTree:
        """Run the Euler approximation in segments of ``solve_every`` steps, solving exactly at the end of each one."""
        solve = self._solve_with_root if optimizer_mode == "root" else self._solve_with_minimize
        objects = self.variables + self.parameters
        segment_ends = [*range(solve_every, n_steps, solve_every), n_steps]
        checkpoint_steps = segment_ends if solve_last_step else segment_ends[:-1]
        segments: list[list[np.ndarray]] = []
        converged: list[bool] = []
        worst_residual = 0.0
        x, start = x0, 0

        with progress:
            checkpoint_row = progress.add_task("Checkpoints", total=len(checkpoint_steps))
            for end in segment_ends:
                theta_start = theta0 + (theta_final - theta0) * start / n_steps
                theta_end = theta0 + (theta_final - theta0) * end / n_steps
                segment = [
                    np.array(values)
                    for values in self._euler_trajectory(
                        x, theta_start, theta_end, end - start, progress, optimizer_mode
                    )
                ]
                # Each segment after the first starts at the previous checkpoint, which the segment before it ends on.
                segments.append([values[1:] for values in segment] if segments else segment)

                if end == n_steps and not solve_last_step:
                    break

                euler_point = {obj.name: values[-1] for obj, values in zip(objects, segments[-1])}
                res = solve(euler_point, theta_end, progressbar=progress, **optimizer_kwargs)
                converged.append(bool(res.success))
                if res.success:
                    x = res.x
                else:
                    x, _ = variable_dict_to_flat_array(euler_point, self.variables, self.parameters)

                # The row holds the parameters the point was solved at, not the sum of the Euler steps toward them.
                checkpoint = flat_array_to_variable_dict(np.r_[x, theta_end], objects, self.coords)
                for obj, values in zip(objects, segments[-1]):
                    values[-1] = checkpoint[obj.name]

                if res.fun is None:
                    residual = np.inf
                elif optimizer_mode == "root":
                    residual = float(np.linalg.norm(res.fun))
                else:
                    residual = float(res.fun)
                # A NaN residual stays the worst one, because no later comparison with it is true.
                if math.isnan(residual) or residual > worst_residual:
                    worst_residual = residual
                n_failed = converged.count(False)
                progress.update(
                    checkpoint_row,
                    advance=1,
                    description=f"Checkpoints ({n_failed} failed)" if n_failed else "Checkpoints",
                    f_value=worst_residual,
                )
                start = end

        failed_steps = [step for step, success in zip(checkpoint_steps, converged) if not success]
        if failed_steps:
            warnings.warn(
                f"The optimizer did not converge at Euler steps {', '.join(map(str, failed_steps))}. The Euler path "
                f"continued from the unsolved point after each of them."
            )

        trajectory = [np.concatenate(parts) for parts in zip(*segments)]
        idata = list_of_array_to_idata(trajectory, self)
        return xr.DataTree.from_dict(
            {
                "/": xr.Dataset(
                    {"converged": ("checkpoint", np.array(converged, dtype=bool))},
                    coords={"checkpoint": np.array(checkpoint_steps, dtype=int)},
                ),
                "variables": idata["variables"].to_dataset(),
                "parameters": idata["parameters"].to_dataset(),
            }
        )

    def _solve_with_root(
        self,
        data,
        theta_final,
        use_jac=True,
        progressbar: bool | ToggleableProgress = True,
        verbose: bool = True,
        equilibrium_tol: float = 1e-6,
        **optimizer_kwargs,
    ):
        method: root_method | None = optimizer_kwargs.pop("method", None)

        use_hess = optimizer_kwargs.pop("use_hess", None)
        _ = optimizer_kwargs.pop("use_hessp", None)

        if use_hess:
            _log.warning(
                'The "use_hess" argument is not used by the root optimizer and will be ignored.'
            )
        if use_jac and "root" not in self._compiled:
            raise ValueError("Jacobian has not been compiled.")
        if progressbar is True:
            progressbar = simulation_progress(self._statistic_headings("root"))

        x0, _theta0 = variable_dict_to_flat_array(data, self.variables, self.parameters)

        if use_jac and self.f_sparse_system_and_jac is not None:
            if method is not None:
                raise ValueError(
                    "This model was compiled with use_sparse_matrices=True, so roots are found by sparse_root. "
                    "Pass a sparse_root strategy as `solver=` instead of `method=`."
                )
            res = sparse_root(
                self.f_sparse_system_and_jac,
                x0,
                args=(theta_final,),
                progressbar=progressbar,
                **optimizer_kwargs,
            )
            squared_error = np.inf if res.fun is None else float(np.sum(res.fun**2))
            return _judge_equilibrium(res, squared_error, equilibrium_tol)

        f_system = wrap_compiled_function_for_scipy(self._compiled_function("f_system"))
        f_jac = (
            wrap_compiled_function_for_scipy(self._compiled_function("f_jac")) if use_jac else None
        )

        res = root(
            f=f_system,
            x0=x0,
            args=(theta_final,),
            jac=f_jac,
            progressbar=progressbar,
            verbose=verbose,
            method="hybr" if method is None else method,
            **optimizer_kwargs,
        )

        return _judge_equilibrium(res, float(np.sum(np.asarray(res.fun) ** 2)), equilibrium_tol)

    def _solve_with_minimize(
        self,
        data: dict[str, np.ndarray],
        theta_final: np.ndarray,
        use_jac: bool = True,
        use_hess: bool = False,
        use_hessp: bool = True,
        progressbar: bool | ToggleableProgress = True,
        verbose: bool = True,
        fixed_values: dict[str, np.ndarray] | None = None,
        equilibrium_tol: float = 1e-6,
        **optimizer_kwargs,
    ):
        if "minimize" not in self._compiled:
            raise ValueError("Minimization functions have not been compiled.")
        if progressbar is True:
            progressbar = simulation_progress(self._statistic_headings("minimize"))

        fixed_values = {} if fixed_values is None else fixed_values

        method: minimize_method = optimizer_kwargs.pop("method", "trust-ncg")
        use_jac, use_hess, use_hessp = get_method_defaults(use_jac, use_hess, use_hessp, method)

        if not use_jac and use_hess:
            use_hess = False
            _log.warning("use_hess is ignored when use_jac=False. Setting use_hess=False.")

        free_mask = make_flat_array_return_mask(
            np.zeros(self.n_variables), self.variables, list(fixed_values), self.coords
        )
        x_full, _theta0 = variable_dict_to_flat_array(
            {**data, **fixed_values}, self.variables, self.parameters
        )

        def wrap(name, variable_axes, include_p=False):
            f = wrap_compiled_function_for_scipy(
                self._compiled_function(name),
                include_p=include_p,
                free_mask=free_mask,
                x_full=x_full,
            )
            return select_free_variable_entries(f, free_mask, variable_axes)

        f_resid = wrap("f_resid", variable_axes=())
        f_grad = wrap("f_grad", variable_axes=(0,))
        f_hess = wrap("f_hess", variable_axes=(0, 1))
        f_hessp = wrap("f_hessp", variable_axes=(0,), include_p=True)

        x0 = x_full[free_mask]

        res = minimize(
            f=f_resid,
            x0=x0,
            args=(theta_final,),
            jac=f_grad if use_jac else None,
            hess=f_hess if use_hess else None,
            hessp=f_hessp if use_hessp else None,
            progressbar=progressbar,
            verbose=verbose,
            method=method,
            **optimizer_kwargs,
        )

        # The objective is half the sum of squared residuals.
        return _judge_equilibrium(res, 2 * float(res.fun), equilibrium_tol)

    def generate_SAM(
        self,
        param_dict: dict[str, np.ndarray],
        initial_variable_guess: dict[str, np.ndarray],
        solve_method: Literal["root", "minimize", "euler"] = "root",
        fixed_values: dict[str, np.ndarray] | None = None,
        use_jac: bool = True,
        use_hess: bool = True,
        n_steps: int = 100,
        **solver_kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Solve for the variable values implied by a set of parameter values, which together form a balanced SAM.

        Market clearing makes every solution a closed system of transactions, but a model usually has infinitely many,
        so the solver returns the one nearest ``initial_variable_guess``. The guess pins down the level of the economy.

        Parameters
        ----------
        param_dict : dict of str to ndarray
            Value of every parameter, keyed by name.
        initial_variable_guess : dict of str to ndarray
            Starting value of every variable, keyed by name.
        solve_method : {"root", "minimize", "euler"}, optional
            How to solve the system. ``"euler"`` assumes the starting point is already an equilibrium, so it is only
            useful for testing. Default is ``"root"``.
        fixed_values : dict of str to ndarray, optional
            Values at which to hold some variables while solving for the rest. Requires ``solve_method="minimize"``,
            because fixing variables leaves more equations than unknowns. Default is None.
        use_jac : bool, optional
            Use the Jacobian of the system with the root and minimize methods. Default is True.
        use_hess : bool, optional
            Use the Hessian of the loss with the minimize method. Default is True.
        n_steps : int, optional
            Number of steps of the Euler method. Default is 100.
        **solver_kwargs
            Passed to the solver of ``solve_method``, such as ``progressbar=False``.

        Returns
        -------
        variable_dict : dict of str to ndarray
            The solved value of every variable, including the fixed ones, keyed by name.

        Examples
        --------
        Solve for the output that a given productivity and labor imply:

        .. code-block:: python

            import numpy as np

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            model.compile(functions_to_compile="root")
            model.generate_SAM(param_dict={"A": np.array([2.0, 5.0]), "L": np.array([50.0, 40.0])},
                               initial_variable_guess={"Y": np.ones(2)},
                               progressbar=False)
        """
        fixed_values = {} if fixed_values is None else fixed_values

        unknown_names = set(fixed_values) - set(self.variable_names)
        if unknown_names:
            raise ValueError(
                f"fixed_values contains names that are not model variables: {unknown_names}"
            )
        if fixed_values and solve_method != "minimize":
            raise ValueError(
                "fixed_values is only supported with solve_method='minimize', because fixing variables leaves "
                "more equations than unknowns."
            )

        SOLVER_FACTORY = {
            "root": ft.partial(self._solve_with_root, use_jac=use_jac),
            "minimize": ft.partial(
                self._solve_with_minimize,
                use_jac=use_jac,
                use_hess=use_hess,
                fixed_values=fixed_values,
            ),
            "euler": ft.partial(self._solve_with_euler_approximation, n_steps=n_steps),
        }

        if solve_method not in SOLVER_FACTORY:
            raise ValueError(
                f"Unknown method {solve_method}. Must be one of {list(SOLVER_FACTORY.keys())}"
            )

        joint_dict = {**param_dict, **initial_variable_guess, **fixed_values}
        free_variables = [var for var in self.variables if var.name not in fixed_values]
        _, flat_params = variable_dict_to_flat_array(joint_dict, free_variables, self.parameters)

        f_solver = SOLVER_FACTORY[solve_method]
        res = f_solver(
            data=joint_dict,
            theta_final=flat_params,
            **solver_kwargs,
        )

        if not solve_method == "euler":
            if not res.success:
                warnings.warn(
                    "Solver did not converge. Results do not represent a valid SAM, and are returned for "
                    "diagnostic purposes only"
                )
            result_dict = {
                **flat_array_to_variable_dict(res.x, free_variables, self.coords),
                **fixed_values,
            }
        else:
            result_dict = res

        return result_dict

    def simulate(
        self,
        initial_state: dict[str, Any] | xr.DataTree,
        final_values: dict[str, float | np.ndarray] | None = None,
        final_delta: dict[str, float | np.ndarray] | None = None,
        final_delta_pct: dict[str, float | np.ndarray] | None = None,
        use_euler_approximation: bool = True,
        use_optimizer: bool = True,
        optimizer_mode: Literal["root", "minimize"] = "root",
        n_iter_euler=10_000,
        solve_every: int | None = None,
        compile_kwargs=None,
        save_results: bool = False,
        overwrite: bool = False,
        output_path: str = "results/",
        simulation_name: str | None = None,
        verbose: bool = True,
        **optimizer_kwargs,
    ):
        """
        Simulate a shock: move some parameters from their values in ``initial_state`` to new ones, and solve for the
        equilibrium at the new values.

        Parameters
        ----------
        initial_state : dict of str to ndarray or xr.DataTree
            An equilibrium of the model: the value of every variable and parameter, keyed by name. The state that
            :meth:`calibrate` returns, the result of an earlier call to ``simulate``, or one of its groups such as
            ``result["euler"]``, is also accepted, and the simulation starts from its final state. A ``DataTree`` state
            is checked, and a ValueError is raised if it is not an equilibrium.
        final_values : dict of str to ndarray, optional
            Final values of the shocked parameters. Exactly one of ``final_values``, ``final_delta`` and
            ``final_delta_pct`` must be given.
        final_delta : dict of str to ndarray, optional
            Changes added to the initial values of the shocked parameters.
        final_delta_pct : dict of str to ndarray, optional
            Factors that multiply the initial values of the shocked parameters, so that 1.1 is a 10% increase.
        use_euler_approximation : bool, optional
            Trace the path to the new equilibrium with the Euler approximation. When ``use_optimizer`` is also True,
            the last Euler step is the optimizer's starting point. Default is True.
        use_optimizer : bool, optional
            Solve for the new equilibrium with a numerical optimizer. Default is True.
        optimizer_mode : {"root", "minimize"}, optional
            Find a root of the system, or minimize its squared residuals. Default is ``"root"``.
        n_iter_euler : int, optional
            Number of Euler steps. Default is 10,000.
        solve_every : int, optional
            Solve for the equilibrium exactly every ``solve_every`` Euler steps with the optimizer of
            ``optimizer_mode``, and continue the Euler approximation from the solution, so the error of the path does
            not grow along it. The Euler result then has a ``converged`` variable along a ``checkpoint`` dimension,
            whose coordinate holds the step of each solve. Default is None, which never solves along the path.
        compile_kwargs : dict, optional
            Passed to :meth:`compile`, which then runs on every call. Default is None, which compiles the functions
            this call needs and the model lacks, with the settings of the model's last compile.
        save_results : bool, optional
            Pickle the result to ``output_path``. Default is False.
        overwrite : bool, optional
            Overwrite a saved result with the same name. When False and one exists, it is loaded and returned instead
            of simulating. Default is False.
        output_path : str, optional
            Directory of saved results. Default is ``"results/"``.
        simulation_name : str, optional
            File name of the saved result. Default is None, which builds one from the date and the solver settings.
        verbose : bool, optional
            Log status messages. Default is True.
        **optimizer_kwargs
            Passed to the optimizer, such as ``method``, ``use_jac`` or ``progressbar=False``.

        Returns
        -------
        result : dict
            ``"euler"``, the Euler path as an ``xr.DataTree`` with ``variables`` and ``parameters`` children indexed by
            ``step``, when the Euler approximation ran. ``"optimizer"``, the initial and solved states as an
            ``xr.DataTree``, and ``"optim_res"``, the optimizer's raw result, when the optimizer ran.

        Examples
        --------
        Raise productivity in both sectors by 10% and solve for the new output:

        .. code-block:: python

            import numpy as np

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            values = {"Y": np.array([100.0, 200.0]), "L": np.array([50.0, 40.0]), "A": np.array([2.0, 5.0])}

            result = model.simulate(initial_state=values,
                                    final_delta_pct={"A": 1.1},
                                    use_euler_approximation=False,
                                    progressbar=False)
            result["optimizer"]["variables"]["Y"]

        Trace the path between the two equilibria with the Euler approximation, solving exactly every 25 steps:

        .. code-block:: python

            import numpy as np

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)
            values = {"Y": np.array([100.0, 200.0]), "L": np.array([50.0, 40.0]), "A": np.array([2.0, 5.0])}

            result = model.simulate(initial_state=values,
                                    final_delta_pct={"A": 1.1},
                                    n_iter_euler=100,
                                    solve_every=25,
                                    progressbar=False)
            result["euler"]["variables"]["Y"].sel(sector="Industry")
        """
        if optimizer_mode not in ("root", "minimize"):
            raise ValueError(f"Unknown optimizer mode {optimizer_mode}")
        if solve_every is not None and (
            not isinstance(solve_every, Integral)
            or isinstance(solve_every, bool)
            or solve_every < 1
        ):
            raise ValueError(f"solve_every must be a positive number of steps, got {solve_every}.")

        solves_with_optimizer = use_optimizer or (
            use_euler_approximation and solve_every is not None
        )
        if simulation_name is None:
            today_str = datetime.now().strftime("%Y-%m-%d")
            simulation_name = f"Simulation_{today_str}"
            if use_euler_approximation:
                checkpoints = "" if solve_every is None else f",solve_every={solve_every}"
                simulation_name += f"_Euler[steps={n_iter_euler}{checkpoints}]"
            if solves_with_optimizer:
                simulation_name += f"_{optimizer_mode}[{','.join([f'{k}={v}' for k, v in optimizer_kwargs.items()])}]"
        if not overwrite and os.path.exists(os.path.join(output_path, simulation_name + ".pkl")):
            res = read_results(output_path, simulation_name, verbose=verbose)
            return res

        needed: set[CompiledFunctions] = {"euler"} if use_euler_approximation else set()
        if solves_with_optimizer:
            needed.add(optimizer_mode)
        self._compile_missing(needed, compile_kwargs)

        from_tree = isinstance(initial_state, xr.DataTree) or (
            isinstance(initial_state, dict)
            and any(isinstance(value, xr.DataTree) for value in initial_state.values())
        )
        x0_var_param: dict[str, Any]
        if from_tree:
            if isinstance(initial_state, xr.DataTree):
                final_group = initial_state
            else:
                simulation = cast(dict[str, xr.DataTree], initial_state)
                final_group = (
                    simulation["optimizer"] if "optimizer" in simulation else simulation["euler"]
                )
            x0_var_param = _final_state(final_group)
        elif isinstance(initial_state, dict):
            x0_var_param = deepcopy(initial_state)
        elif isinstance(initial_state, xr.Dataset):
            x0_var_param = {x.name: initial_state.variables[x.name].values for x in self.variables}
            x0_var_param.update(
                {x.name: initial_state.parameters[x.name].values for x in self.parameters}
            )
        else:
            raise ValueError(
                f"initial_state must be a dict of initial values or a DataTree, found {type(initial_state)}"
            )

        all_model_objs = [*self.variable_names, *self.parameter_names]

        if not all(x in x0_var_param for x in all_model_objs):
            missing_objs = [x for x in all_model_objs if x not in x0_var_param]
            raise ValueError(
                f"initial_state must contain values for all variables and parameters in the model. Did not find values "
                f"for {', '.join(missing_objs)}"
            )

        if any(x not in all_model_objs for x in x0_var_param):
            unknown_objs = [x for x in x0_var_param if x not in all_model_objs]
            raise ValueError(
                f"initial_state contains values for variables or parameters not in the model: {', '.join(unknown_objs)}"
            )

        if from_tree:
            x0, theta0 = variable_dict_to_flat_array(x0_var_param, self.variables, self.parameters)
            squared_error = float(
                np.sum(np.asarray(self._compiled_function("f_system")(x0, theta0)) ** 2)
            )
            equilibrium_tol = optimizer_kwargs.get("equilibrium_tol", 1e-6)
            if squared_error >= equilibrium_tol:
                raise ValueError(
                    f"initial_state is not an equilibrium: its sum of squared residuals is {squared_error:.3e}, above "
                    f"equilibrium_tol={equilibrium_tol:g}."
                )

        initial_values = deepcopy(x0_var_param)
        final_param_dict = create_final_param_dict(
            x0_var_param, final_values, final_delta, final_delta_pct
        )

        _, theta_simulation = variable_dict_to_flat_array(
            final_param_dict, self.variables, self.parameters
        )

        return_values = {}

        statistics_mode = optimizer_mode if solves_with_optimizer else "root"
        progress = simulation_progress(
            self._statistic_headings(statistics_mode),
            enabled=optimizer_kwargs.pop("progressbar", True),
        )
        with progress:
            if use_euler_approximation:
                idata_euler = self._solve_with_euler_approximation(
                    x0_var_param,
                    theta_simulation,
                    n_steps=n_iter_euler,
                    progressbar=progress,
                    optimizer_mode=statistics_mode,
                    solve_every=solve_every,
                    solve_last_step=not use_optimizer,
                    # The root solver warns that it ignores use_hess, which would repeat at every checkpoint.
                    optimizer_kwargs={
                        key: value
                        for key, value in optimizer_kwargs.items()
                        if not (statistics_mode == "root" and key == "use_hess")
                    },
                )
                return_values["euler"] = idata_euler
                final_step = idata_euler["variables"].sizes["step"] - 1
                for group in ("variables", "parameters"):
                    idata_euler[group] = idata_euler[group].dataset.dropna(dim="step")
                last_step = min(
                    idata_euler["variables"].coords["step"].max().item(),
                    idata_euler["parameters"].coords["step"].max().item(),
                )
                if last_step != final_step:
                    _log.warning(
                        f"NaNs were encountered in the euler approximation. Backing up to step {last_step} "
                        f"(the last non-NaN step) for the optimizer started point."
                    )

                x0_var_param = datatree_to_value_dict(idata_euler.isel(step=last_step))

            if use_optimizer:
                if optimizer_mode == "root":
                    use_jac = optimizer_kwargs.pop("use_jac", True)
                    res = self._solve_with_root(
                        x0_var_param,
                        theta_simulation,
                        use_jac=use_jac,
                        progressbar=progress,
                        **optimizer_kwargs,
                    )
                else:
                    use_jac = optimizer_kwargs.pop("use_jac", True)
                    use_hess = optimizer_kwargs.pop("use_hess", True)
                    res = self._solve_with_minimize(
                        x0_var_param,
                        theta_simulation,
                        use_jac=use_jac,
                        use_hess=use_hess,
                        progressbar=progress,
                        **optimizer_kwargs,
                    )

                if not res.success:
                    warnings.warn(
                        "Optimizer did not converge. Results do not represent a valid equlibriuim, and are "
                        "returned for diagnostic purposes only"
                    )

                idata_optim = optimizer_result_to_idata(res, theta_simulation, initial_values, self)
                return_values["optimizer"] = idata_optim
                return_values["optim_res"] = res

        if save_results:
            write_results(
                return_values,
                output_path=output_path,
                filename=simulation_name,
                overwrite=overwrite,
            )

        return return_values

    def check_for_equilibrium(
        self,
        data: dict[str, float | np.ndarray] | xr.DataTree | xr.Dataset | OptimizeResult,
        tol: float = 1e-6,
    ) -> None:
        """
        Log whether a state of the model is an equilibrium, and the residual of every equation that misses it.

        Parameters
        ----------
        data : dict of str to ndarray, xr.DataTree, xr.Dataset, or OptimizeResult
            The value of every variable and parameter, keyed by name, or a result that holds them. A ``DataTree`` is
            read at its last step.
        tol : float, optional
            Largest sum of squared residuals that counts as an equilibrium. Default is 1e-6.

        Examples
        --------
        Check that a calibrated state solves the model:

        .. code-block:: python

            import numpy as np

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            values = {"Y": np.array([100.0, 200.0]), "L": np.array([50.0, 40.0]), "A": np.array([2.0, 5.0])}

            model.compile(functions_to_compile="root")
            model.check_for_equilibrium(values)  # Equilibrium found! Total squared error: 0.000000
        """

        if isinstance(data, xr.DataTree):
            if not {"variables", "parameters"} <= set(data.children):
                raise ValueError("DataTree must contain variables and parameters children")

            data = _final_state(data)

        elif isinstance(data, xr.Dataset):
            data = {x.name: data[x.name].values for x in self.variables + self.parameters}

        elif isinstance(data, OptimizeResult):
            variables = flat_array_to_variable_dict(data.x, self.variables, self.coords)
            parameters = flat_array_to_variable_dict(data.args, self.parameters, self.coords)
            data = {**variables, **parameters}

        x, theta = variable_dict_to_flat_array(data, self.variables, self.parameters)
        errors = np.asarray(self._compiled_function("f_system")(x, theta))
        sse = (errors**2).sum()

        if sse < tol:
            _log.info(f"Equilibrium found! Total squared error: {sse:0.6f}")
        else:
            _log.info(f"Equilibrium not found. Total squared error: {sse:0.6f}")
            longest_name = max(len(x) for x in self.unpacked_equation_names)

            _log.info("\n")
            _log.info(f"{'Equation':<{longest_name + 10}}" + "Residual")
            _log.info("=" * 100)

            for eq_name, resid in zip(self.unpacked_equation_names, errors):
                is_negative = resid < 0
                pad = "" if is_negative else " "
                if abs(resid) > tol:
                    _log.info(f"{eq_name:<{longest_name + 10}}{pad}{resid:0.6f}")

    def check_shape(self, data: Any, name: str) -> np.ndarray:
        """
        Convert ``data`` to an array for the variable or parameter ``name``, printing a message if its shape is wrong.

        Parameters
        ----------
        data : array_like
            Values for ``name``. A one-element array for a scalar object is squeezed to a scalar.
        name : str
            Name of a variable or parameter.

        Returns
        -------
        data : ndarray
            ``data`` as an array.

        Examples
        --------
        Convert a list of sector outputs to an array:

        .. code-block:: python

            import numpy as np

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            model.check_shape([100.0, 200.0], "Y")  # array([100., 200.])
        """
        obj = self.get(name)
        dims = obj.dims
        shape = tuple([len(self.coords[i]) for i in dims])
        data = np.array(data)

        if data.shape in [(1,), ()] and shape == ():
            data = data.squeeze()
        if data.shape != shape:
            print(f"{name} has shape {data.shape}, expected {shape}")
        return data

    def finalize_calibration(self) -> dict[str, np.ndarray]:
        """
        Collect the calibrated value of every variable and parameter from the calling scope's local names.

        Call it at the end of a calibration function that assigns each model object to a local variable of the same
        name. Each value is converted with :meth:`check_shape`, and every object without a local value is printed.

        Returns
        -------
        values : dict of str to ndarray
            The value of every variable and parameter found, keyed by name.

        Examples
        --------
        Calibrate productivity in a function whose local names match the model's objects:

        .. code-block:: python

            import numpy as np

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            def calibrate(model, Y, L):
                A = Y / L
                return model.finalize_calibration()


            values = calibrate(model, Y=np.array([100.0, 200.0]), L=np.array([50.0, 40.0]))
            values["A"]  # array([2., 5.])
        """
        d = {}
        caller_locals = inspect.currentframe().f_back.f_locals

        for obj in sorted(self.variables) + sorted(self.parameters):
            if obj.name != "_" and obj.name in caller_locals:
                data = caller_locals[obj.name]
                if isinstance(data, pd.DataFrame | pd.Series):
                    data = data.values
                d[obj.name] = self.check_shape(data, obj.name)
            else:
                print(f"{obj.name} not found in model variables")
        return d

    def initialize_parameter(self, name: str, value: float = 5.0) -> np.ndarray:
        """
        Return an array of ``value`` with the shape of the variable or parameter ``name``.

        Parameters
        ----------
        name : str
            Name of a variable or parameter.
        value : float, optional
            The fill value. Default is 5.0.

        Returns
        -------
        values : ndarray
            An array of ``value`` with one axis per dim of ``name``.

        Examples
        --------
        Start every sector's productivity at one:

        .. code-block:: python

            import numpy as np

            import cge_modeling as cm

            with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
                Y = cm.Variable("Y", dims="sector", description="Output of each sector")
                L = cm.Parameter("L", dims="sector", description="Labor employed in each sector")
                A = cm.Parameter("A", dims="sector", description="Labor productivity of each sector")
                cm.Equation("Production", Y, A * L)

            model.initialize_parameter("A", value=1.0)  # array([1., 1.])
        """
        dims = self.get(name).dims
        size = [len(self.coords[dim]) for dim in dims]
        return np.full(size, value)


def write_results(results, output_path, filename=None, overwrite=False, verbose=True):
    # Check if the output path exists, create it if it doesn't
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Use today's date as the filename if filename is None
    if filename is None:
        filename = datetime.today().strftime("%Y-%m-%d") + ".pkl"
    if not filename.endswith(".pkl"):
        filename = filename + ".pkl"

    base_filename, ext = os.path.splitext(filename)
    new_filename = base_filename + ext
    counter = 1

    if not overwrite:
        while os.path.exists(os.path.join(output_path, new_filename)):
            new_filename = f"{base_filename}({counter}){ext}"
            counter += 1
    else:
        if os.path.exists(os.path.join(output_path, new_filename)) and verbose:
            _log.warning(
                f"Overwriting existing file found at {os.path.join(output_path, new_filename)}"
            )

    with open(os.path.join(output_path, new_filename), "wb") as f:
        cloudpickle.dump(results, f)

    if verbose:
        _log.info(f"Results saved to {os.path.join(output_path, new_filename)}")


def read_results(output_path, filename, verbose=True):
    ext = "pkl"
    if not filename.endswith(ext):
        base_filename = filename
        filename += f".{ext}"
    else:
        base_filename = filename.replace(ext, "")

    max_suffix = 0
    pattern = re.compile(rf"^{re.escape(base_filename)}\((\d+)\)\.{ext}$")

    for fname in os.listdir(output_path):
        match = pattern.match(fname)
        if match:
            suffix = int(match.group(1))
            if suffix > max_suffix:
                max_suffix = suffix

    if max_suffix > 0:
        filename = f"{base_filename}({max_suffix}).{ext}"

    with open(os.path.join(output_path, filename), "rb") as f:
        results = cloudpickle.load(f)

    if verbose:
        _log.info(f"Results loaded from {os.path.join(output_path, filename)}")
    return results
