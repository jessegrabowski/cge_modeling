import functools as ft
import inspect
import logging
import os.path
import re
import warnings

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import xarray as xr

from better_optimize import minimize, root
from better_optimize.constants import minimize_method, root_method
from cloudpickle import cloudpickle
from IPython.display import Latex
from pytensor.compile.mode import Mode, get_mode
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.basic import xtensor_from_tensor
from scipy.optimize import OptimizeResult

from cge_modeling.base.context import ModelObjectVariable, XTensorEquation, model_stack
from cge_modeling.base.function_wrappers import (
    select_free_variable_entries,
    wrap_compiled_function_for_scipy,
)
from cge_modeling.base.primitives import Parameter, Result, Variable
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
    display_latex_table,
    list_of_array_to_idata,
    optimizer_result_to_idata,
)

_log = logging.getLogger(__name__)


def _parse_compile_kwarg(
    compile: CompiledFunctions | Sequence[CompiledFunctions] | Literal[True] | None = None,
) -> Sequence[CompiledFunctions] | None:
    if compile is None:
        return None
    if compile is True or compile == "all":
        return ["root", "minimize", "euler"]
    if isinstance(compile, str):
        return [compile]
    return compile


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


class CGEModel:
    def __init__(self, coords: dict[str, list[str]] | None = None):
        """
        Main interface for interacting with a CGE model.

        Parameters
        ----------
        coords: dict[str, Sequence[str]], optional
            A dictionary of dimension names and labels associated with each dimension. For example, if certain objects in
            the model are indexed by "sector", then coords should contain a key "sector" with a list of sector labels as
            values: {"sector": ["Agriculture", "Industry", "Service"]}.

        Notes
        -----
        Computable General Equilibrium models are used to study the change in economic equilibria following shifts in model
        parameters. The purpose of this class is to provide an organized framework for declaring and studying this class
        of models.

        A model is built from pytensor xtensor expressions: create it with ``coords``, then declare variables,
        parameters, and equations with :mod:`cge_modeling.model` inside ``with model:`` or with ``model=model``.
        Distinction between variables and parameters is enforced to check that the model is square (there are as many
        equations as there are variables), and to determine which objects to solve for during simulation.

        Variables and parameters are declared with abstract indices, called dims, which ``coords`` makes concrete. When
        considering the "squareness" of a model, the "unpacked" dimensions of the variables and equations are
        considered. Therefore, it may not be the case that the number of variables declared is equal to the number of
        equations declared.
        """
        self.coords: dict[str, list[str]] = {} if coords is None else coords

        self._variables: dict[str, Variable] = {}
        self._parameters: dict[str, Parameter] = {}

        self._unpacked_variables: dict[str, dict[str, Any]] = {}
        self._unpacked_parameters: dict[str, dict[str, Any]] = {}
        self._unpacked_equations: dict[str, None] = {}

        self._model_objects: dict[str, ModelObjectVariable] = {}
        self._xtensor_equations: dict[str, XTensorEquation] = {}
        self._context_tokens: list = []

        self.scenarios: dict[str, Result] = {}

        self._compiled: Sequence[str] = []
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
        return len(self.unpacked_variable_names)

    @property
    def n_parameters(self) -> int:
        return len(self.unpacked_parameter_names)

    @property
    def n_equations(self) -> int:
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
        """Map one label, or a list of labels, of ``dim`` to positions along it."""
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

        group = f"{kind}s"
        getattr(self, f"_{group}")[name] = obj
        getattr(self, f"_unpacked_{group}").update(
            {
                make_indexed_name(unpacked): {"modelobj": unpacked}
                for unpacked in self._unpack_objects([obj])
            }
        )
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
        self.mode = compile_mode

        return self

    def check_initialization(self):
        if self.n_variables != self.n_equations:
            raise ValueError(
                f"Found {self.n_variables} variables but {self.n_equations} equations. "
                "System is not square. Check your model equations."
            )

    @property
    def variable_names(self) -> list[str]:
        return list(self._variables.keys())

    @property
    def variables(self) -> list[Variable]:
        return list(self._variables.values())

    @property
    def unpacked_variable_names(self) -> list[str]:
        return list(self._unpacked_variables.keys())

    @property
    def unpacked_variables(self) -> list[Variable]:
        return [
            self._unpacked_variables[var_name]["modelobj"]
            for var_name in self.unpacked_variable_names
        ]

    @property
    def parameter_names(self) -> list[str]:
        return list(self._parameters.keys())

    @property
    def parameters(self) -> list[Parameter]:
        return list(self._parameters.values())

    @property
    def unpacked_parameter_names(self) -> list[str]:
        return list(self._unpacked_parameters.keys())

    @property
    def unpacked_parameters(self) -> list[Parameter]:
        return [
            self._unpacked_parameters[param_name]["modelobj"]
            for param_name in self.unpacked_parameter_names
        ]

    @property
    def equation_names(self) -> list[str]:
        return list(self._xtensor_equations)

    @property
    def equations(self) -> list[XTensorEquation]:
        """The records :func:`cge_modeling.model.Equation` registered on the model."""
        return list(self._xtensor_equations.values())

    @property
    def unpacked_equation_names(self):
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

    def get_parameter(self, param_name: str):
        return self._get_object(param_name, "parameters")

    def get_parameters(self, param_names: str | list[str] | None = None):
        if param_names is None:
            param_names = self.parameter_names
        return [self.get_parameter(name) for name in ensure_input_is_sequence(param_names)]

    def get_variable(self, var_name: str):
        return self._get_object(var_name, "variables")

    def get_variables(self, var_names: str | list[str] | None = None):
        if var_names is None:
            var_names = self.variable_names
        return [self.get_variable(name) for name in ensure_input_is_sequence(var_names)]

    def get(
        self, names: str | list[str] | None = None
    ) -> Variable | Parameter | list[Variable | Parameter]:
        """
        Retrieve model objects (variables or parameters) from the model by name.

        Parameters
        ----------
        names: str or list of str, optional
            Variable or parameter names to retrieve from the model. If None, return all model variables and parameters.

        Returns
        -------
        out: Variable, Parameter, or list of Variable or Parameter
            The requested model object when one name matches, otherwise the list of requested objects.
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

    def get_shape(self, name):
        obj = self.get(name)
        return (len(self.coords[dim]) for dim in obj.dims)

    def summary(
        self,
        variables: Literal["all", "variables", "parameters"] | list[str] = "all",
        results: Result | list[Result] | None = None,
    ) -> None:
        """Display a table of the symbols and descriptions of ``variables``, with columns from each of ``results``."""
        results = [] if results is None else results
        if isinstance(results, Result):
            results = [results]

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

        for result in results:
            result_df = result.to_frame()
            initial, final = result_df.loc[variables, :].values.T
            info_dict[result.name] = {"Initial": initial, "Final": final}

        display_latex_table(info_dict)

    def equation_table(self) -> None:
        """Display a table of the model's equations, numbered, with their names and LaTeX forms."""
        info_dict: dict[str, list[Any]] = {"": [], "Name": [], "Equation": []}
        for number, equation in enumerate(self._xtensor_equations.values(), start=1):
            info_dict[""].append(number)
            info_dict["Name"].append(_replace_dim_marker_with_dim_name(equation.name))
            info_dict["Equation"].append(Latex(equation_to_latex(equation, model=self)))

        display_latex_table(info_dict)

    def calibrate(
        self,
        data: Mapping[str, float | np.ndarray] | None = None,
        max_iter: int = 100,
        name: str = "initial_calibration",
        initial_values: Mapping[str, float | np.ndarray] | None = None,
    ) -> Result:
        """
        Solve for every variable and parameter value that ``data`` does not give.

        Parameters
        ----------
        data : dict of str to float or ndarray, optional
            Known values, keyed by variable or parameter name.
        max_iter : int, optional
            Iteration limit of the solver. Default is 100.
        name : str, optional
            Name of the returned result. Default is ``"initial_calibration"``.
        initial_values : dict of str to float or ndarray, optional
            Starting values for unknowns, keyed by name.

        Returns
        -------
        result : Result
            The given and solved values, with ``success`` set when every equation holds.
        """
        # cge_modeling.tools.calibration imports the compile modules, which import this module.
        from cge_modeling.tools.calibration import (
            calibrate_entries,
            compile_calibration_system,
            flatten_entries,
            residual_is_converged,
        )

        data = {} if data is None else data
        initial_values = {} if initial_values is None else initial_values

        system = compile_calibration_system(self)
        fitted_values, residual = calibrate_entries(
            system=system, data=data, initial_values=initial_values, max_iter=max_iter
        )
        initial_values_flat, _ = flatten_entries(system=system, values=data, fill=np.nan)

        return Result(
            name=name,
            success=residual_is_converged(residual),
            variables=self.unpacked_variable_names,
            parameters=self.unpacked_parameter_names,
            initial_values=initial_values_flat,
            fitted_values=fitted_values,
        )

    def _solve_with_euler_approximation(self, data, theta_final, n_steps):
        x0, theta0 = variable_dict_to_flat_array(data, self.variables, self.parameters)
        result = self._compiled_function("f_euler")(x0, theta0, theta_final, n_steps)
        return list_of_array_to_idata(result, self)

    def _solve_with_root(
        self,
        data,
        theta_final,
        use_jac=True,
        progressbar=True,
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
        progressbar: bool = True,
        verbose: bool = True,
        fixed_values: dict[str, np.ndarray] | None = None,
        equilibrium_tol: float = 1e-6,
        **optimizer_kwargs,
    ):
        if "minimize" not in self._compiled:
            raise ValueError("Minimization functions have not been compiled.")

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
        Generate a Social Accounting Matrix (SAM) from the model parameters.

        Parameters
        ----------
        param_dict: dict[str, np.ndarray]
            A dictionary of parameter values. The keys should be the names of the parameters, and the values should be
            numpy arrays of the same shape as the parameter.

        initial_variable_guess: dict[str, np.ndarray]
            A dictionary of initial values for the model variables. The keys should be the names of the variables, and
            the values should be numpy arrays of the same shape as the variable.

        fixed_values: dict[str, np.ndarray], optional
            Values at which to hold a subset of model variables while solving for the rest. The keys are variable
            names, and the values are numpy arrays of the same shape as the variable. Only supported with
            solve_method='minimize', because fixing variables leaves more equations than unknowns.

        solve_method: str
            The method to use to solve the model. One of 'root', 'minimize', or 'euler'. Defaults to 'root'.

            Note that Euler is not recommended for generating a SAM, because it assumes the initial point is itself a
            SAM (in which case you don't need this function). It is included for testing purposes only.

        use_jac: bool
            Whether to use the Jacobian of the system of equations when solving with the root or minimize methods.
            Defaults to True.

        use_hess: bool
            Whether to use the Hessian of the loss function when solving with the minimize method. Ignored if method
            is not 'minimize'. Defaults to True.

        n_steps: int
            The number of steps to use when solving with the Euler method. Ignored if method is not 'euler'. Defaults
            to 100.

        **solver_kwargs: dict
            Additional keyword arguments to pass to the solver, either scipy.optimize.root or scipy.optimize.minimize,
            depending on the chosen method. See those docstrings for details.

        Returns
        -------
        variable_dict: dict[str, np.ndarray]
            A dictionary of variable values, including any fixed values. The keys are the names of the variables, and
            the values are numpy arrays of the same shape as the variable.

        Notes
        -----
        This function generates a "SAM" by solving for the values of the model variables that are implied by the
        provided parameter values. This mapping is valid by construction, because the structure of a correctly specified
        CGE model will ensure all markets clear, resulting in a closed system of economic transactions between model
        agents.

        Initial variables need to be provided in order to pin down the **level** of the economy. In general, an infinite
        number of solutions exist for a given set of parameter values. What is returned is just one solution, which is
        close to the provided initial values.
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
        initial_state: dict[str, float | np.ndarray],
        final_values: dict[str, float | np.ndarray] | None = None,
        final_delta: dict[str, float | np.ndarray] | None = None,
        final_delta_pct: dict[str, float | np.ndarray] | None = None,
        use_euler_approximation: bool = True,
        use_optimizer: bool = True,
        optimizer_mode: Literal["root", "minimize"] = "root",
        n_iter_euler=10_000,
        compile_kwargs=None,
        save_results: bool = False,
        overwrite: bool = False,
        output_path: str = "results/",
        simulation_name: str | None = None,
        verbose: bool = True,
        **optimizer_kwargs,
    ):
        """
        Simulate a shock to the model economy by computing the equilibrium state of the economy at the provided
        post-shock values.

        Parameters
        ----------
        initial_state: dict[str, np.ndarray]
            A dictionary of initial values for the model variables and parameters. The keys should be the names of the
            variables and parameters, and the values should be numpy arrays of the same shape as the variable or
            parameter.

            .. warning:: The initial state **must** represent a model equilibrium. This is checked before simulation
               begins, and an error is raised if the initial state is not an equilibrium.

        final_values: dict[str, np.ndarray], optional
            A dictionary of exact final values for a subset of model parameters. The keys should be the names of the
            parameters to be changed, and the values should be numpy arrays of the same shape as the parameter.

            Exactly one of final_values, final_delta, or final_delta_pct must be provided.

        final_delta: dict[str, np.ndarray], optional
            A dictionary of changes to be applied to a subset of the model parameters. Changes are added to the initial
            values, so that positive values increase the parameter value and negative values decrease the parameter.
            The keys should be the names of the parameters to be changed, and the values should be numpy arrays of the
            same shape as the parameter.

            Exactly one of final_values, final_delta, or final_delta_pct must be provided.

        final_delta_pct: dict[str, np.ndarray], optional
            A dictionary of percentage changes to be applied to a subset of the model parameters. Values provided are
            multipled by the initial values, so that values greater than 1.00 represent precentage increases, and values
            less than 1.00 represent percentage decreases. The keys should be the names of the parameters to be changed,
            and the values should be numpy arrays of the same shape as the parameter.

            Exactly one of final_values, final_delta, or final_delta_pct must be provided.

        use_euler_approximation: bool, optional
            Whether to use the Euler approximation to compute the final state of the economy. Defaults to True. If
            use_optimizer is also true, the Euler approximation will be used as the initial guess for the optimizer.

        use_optimizer: bool, optional
            Whether to use a numerical optimizer to compute the final state of the economy. Defaults to True. If
            use_euler_approximation is also true, the Euler approximation will be used as the initial guess for the
            optimizer.

        optimizer_mode: str, optional
            The type of optimizer to use. One of 'root' or 'minimize'. Ignored if use_optimizer is False. Defaults to
            'root'.

        n_iter_euler: int
            The number of iterations to use when computing the Euler approximation. Defaults to 10,000. Ignored if
            use_euler_approximation is False.

        compile_kwargs: dict, optional
            Additional keyword arguments to pass to the compile method. See the docstring for compile for details.

        optimizer_kwargs: dict, optional
            Additional keyword arguments to pass to the optimizer. See the docstring for scipy.optimize.root or
            scipy.optimize.minimize for details.

        save_results: bool, optional
            If True, result object will be pickled and saved. Defaults to True.

        output_path: str, optional
            The path to save the result object. Defaults to 'results'.

        simulation_name: str, optional
            The name of the simulation. Defaults to '{date}-{time}'. Used to name the saved result object.

        verbose: bool, default True
            If True, display status messages about the simulation.

        Returns
        -------
        result: Result or InferenceData
            A Result object containing the initial and final values of the model variables and parameters. If
            use_euler_approximation is True, an InferenceData object containing the Euler approximation will be returned
            instead.
        """
        if simulation_name is None:
            today_str = datetime.now().strftime("%Y-%m-%d")
            simulation_name = f"Simulation_{today_str}"
            if use_euler_approximation:
                simulation_name += f"_Euler[steps={n_iter_euler}]"
            if use_optimizer:
                simulation_name += f"_{optimizer_mode}[{','.join([f'{k}={v}' for k, v in optimizer_kwargs.items()])}]"
        if not overwrite and os.path.exists(os.path.join(output_path, simulation_name + ".pkl")):
            res = read_results(output_path, simulation_name, verbose=verbose)
            return res

        if not self._compiled:
            if compile_kwargs is None:
                to_compile = []
                if use_euler_approximation:
                    to_compile.append("euler")
                if use_optimizer:
                    to_compile.append(optimizer_mode)

                compile_kwargs = {"functions_to_compile": to_compile}

            self.compile(**compile_kwargs)

        if isinstance(initial_state, Result):
            x0_var_param = deepcopy(initial_state.to_dict()["fitted"])
        elif isinstance(initial_state, dict):
            x0_var_param = deepcopy(initial_state)
        elif isinstance(initial_state, xr.Dataset):
            x0_var_param = {x.name: initial_state.variables[x.name].values for x in self.variables}
            x0_var_param.update(
                {x.name: initial_state.parameters[x.name].values for x in self.parameters}
            )
        else:
            raise ValueError(
                f"initial_state must be a Result or a dict of initial values, found {type(initial_state)}"
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

        final_param_dict = create_final_param_dict(
            x0_var_param, final_values, final_delta, final_delta_pct
        )

        _, theta_simulation = variable_dict_to_flat_array(
            final_param_dict, self.variables, self.parameters
        )

        return_values = {}

        if use_euler_approximation:
            idata_euler = self._solve_with_euler_approximation(
                x0_var_param, theta_simulation, n_steps=n_iter_euler
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
                    x0_var_param, theta_simulation, use_jac=use_jac, **optimizer_kwargs
                )
            elif optimizer_mode == "minimize":
                use_jac = optimizer_kwargs.pop("use_jac", True)
                use_hess = optimizer_kwargs.pop("use_hess", True)
                res = self._solve_with_minimize(
                    x0_var_param,
                    theta_simulation,
                    use_jac=use_jac,
                    use_hess=use_hess,
                    **optimizer_kwargs,
                )
            else:
                raise ValueError(f"Unknown optimizer mode {optimizer_mode}")

            if not res.success:
                warnings.warn(
                    "Optimizer did not converge. Results do not represent a valid equlibriuim, and are "
                    "returned for diagnostic purposes only"
                )

            idata_optim = optimizer_result_to_idata(res, theta_simulation, initial_state, self)
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

    def print_residuals(self, res):
        n_vars = len(self.unpacked_variable_names)
        endog, exog = res.fitted_values[:n_vars], res.fitted_values[n_vars:]
        errors = self.f_system(endog, exog)
        for eq, val in zip(self.unpacked_equation_names, errors):
            _log.info(f"{eq:<75}: {val:<10.3f}")

    def check_for_equilibrium(
        self,
        data: dict[str, float | np.ndarray] | xr.DataTree | xr.Dataset | OptimizeResult,
        tol=1e-6,
        print_equations=True,
    ):
        """
        Verify if a given state of the model is an equilibrium.

        Parameters
        ----------
        data: dict, xr.DataTree, or xr.Dataset
            The state of the model to check. Must contain values for all variables and parameters in the model.
        tol: float
            The tolerance for the squared error of the system of equations. Defaults to 1e-6.
        print_equations: bool
            If true, prints the symbolic equation along with the name and residuls for each equations that exceeds
            the tolerance.

        Notes
        -----
        The data argument is expected to be either a dictionary of variable and parameter values.
        """

        if isinstance(data, xr.DataTree):
            if not {"variables", "parameters"} <= set(data.children):
                raise ValueError("DataTree must contain variables and parameters children")

            if "step" in data["variables"].dims:
                data = data.isel(step=-1)
            data = datatree_to_value_dict(data)

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

    def check_shape(self, data, name):
        obj = self.get(name)
        dims = obj.dims
        shape = tuple([len(self.coords[i]) for i in dims])
        data = np.array(data)

        if data.shape in [(1,), ()] and shape == ():
            data = data.squeeze()
        if data.shape != shape:
            print(f"{name} has shape {data.shape}, expected {shape}")
        return data

    def finalize_calibration(self):
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

    def initialize_parameter(self, name, value=5.0):
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
