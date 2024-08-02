import functools as ft
import inspect
import logging
import os.path
import re
import time
import warnings

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from typing import Literal, cast

import numba as nb
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import sympy as sp
import xarray as xr

from arviz import InferenceData
from cloudpickle import cloudpickle
from fastprogress import progress_bar
from numba_progress import ProgressBar as NumbaProgressBar
from scipy import optimize
from scipy.optimize import OptimizeResult

from cge_modeling.base.primitives import (
    Equation,
    Parameter,
    Result,
    Variable,
    _SympyEquation,
)
from cge_modeling.base.utilities import (
    CostFuncWrapper,
    _optimzer_early_stopping_wrapper,
    _replace_dim_marker_with_dim_name,
    _validate_input,
    ensure_input_is_sequence,
    flat_array_to_variable_dict,
    infer_object_shape_from_coords,
    unpack_equation_strings,
    variable_dict_to_flat_array,
    wrap_fixed_values,
    wrap_pytensor_func_for_scipy,
)
from cge_modeling.pytensorf.compile import (
    compile_cge_model_to_pytensor,
    compile_euler_approximation_function,
    euler_approximation,
    jax_euler_step,
    jax_loss_grad_hessp,
    pytensor_euler_step,
    pytensor_objects_from_CGEModel,
)
from cge_modeling.tools.numba_tools import euler_approx, numba_lambdify
from cge_modeling.tools.output_tools import (
    display_latex_table,
    list_of_array_to_idata,
    optimizer_result_to_idata,
)
from cge_modeling.tools.pytensor_tools import (
    make_jacobian,
    rewrite_pregrad,
)
from cge_modeling.tools.sympy_tools import (
    enumerate_indexbase,
    expand_obj_by_indices,
    find_equation_dims,
    indexed_var_to_symbol,
    indexed_variables_to_sub_dict,
    make_indexbase_sub_dict,
    make_indexed_name,
    sub_all_eqs,
    substitute_reduce_ops,
)

ValidGroups = Literal[
    "variables",
    "parameters",
    "equations",
    "_unpacked_variables",
    "_unpacked_parameters",
    "_unpacked_equations",
]

CompiledFunctions = Literal["root", "mininimze", "euler", "all"]

_log = logging.getLogger(__name__)


class CGEModel:
    def __init__(
        self,
        coords: dict[str, list[str, ...]] | None = None,
        variables: list[Variable] | dict[str, Variable] | None = None,
        parameters: list[Parameter] | dict[str, Parameter] | None = None,
        equations: list[Equation] | dict[str, Equation] | None = None,
        numeraire: Variable | None = None,
        parse_equations_to_sympy: bool = True,
        apply_sympy_simplify: bool = False,
        backend: Literal["pytensor", "numba"] | None = "pytensor",
        mode: str | None = None,
        inverse_method: str = "solve",
        use_sparse_matrices: bool = False,
        compile: list[CompiledFunctions] | None = "all",
        use_scan_euler: bool = False,
    ):
        """
        Main interface for interacting with the a CGE model.

        Parameters
        ----------
        coords: dict[str, Sequence[str]]
            A dictionary of dimension names and labels associated with each dimension. For example, if certain objects in
            the model are indexed by "sector", then coords should contain a key "sector" with a list of sector labels as
            values: {"sector": ["Agriculture", "Industry", "Service"]}.

        variables: list of Variables
            A list of endogenous variables associated with the model. These will be solved for during simulation.

        parameters: list of Parameters
            A list of parameters associated with the model. These are fixed during simulation. Note that exogenous variables
            should be included as parameters.

        equations: list of Equations
            A list of equations associated with the model. These define relationships between the variables and parameters.

        numeraire: Variable, optional
            The numéraire variable. If supplied, the numéraire will be removed from the model, and all instances of the
            numéraire in model equations will be set to 1.
            NOTE: It is not currently recommended to use this feature. Instead, add an equation to the model
                  fixing the numéraire variable to 1.

        parse_equations_to_sympy: bool, default
            If True, the equations will be parsed to sympy expressions. This is required for the model to be compiled to
            the "numba" backend. If False, the equations will be converted to symbolic pytensor expressions.
            This is required for the model to be compiled to the "pytensor" backend. Defaults to True.

        apply_sympy_simplify: bool, default False
            If True, `.simplify()` will be called elementwise on the

        backend: str, optional
            Computational backend to compile the model to. One of "pytensor" or "numba". Defaults to "numba".

        mode: str, optional
            Compilation mode for the pytensor backend. One of None, "JAX", or "NUMBA". Defaults to None. Note that this
            argument is ignored if the backend is not "pytensor".

        inverse_method: str, optional
            Method to use to compute the inverse of the Jacobian matrix. One of "solve", "pinv", or "SGD".
            Defaults to "solve". Ignored if the backend is not "pytensor".

        use_sparse_matrices: bool, optional
            Whether to use sparse matrices to represent matrices associated with the system (e.g. the Jacobian matrix).
            Defaults to False. Ignored if the backend is not "pytensor".

        compile: bool, optional
            Whether to compile the model during initialization. Defaults to True.


        Notes
        -----
        Computable General Equilibrium models are used to study the change in economic equilibria following shifts in model
        parameters. The purpose of this class is to provide an organized framework for declaring and studying this class
        of models.

        At it's most basic, a CGE model is comprised of three types of objects: variables, parameters, and equations. Each
        of these should declared separately using the appropriate object type. For example, to declare a variable, use the
        Variable class. Distinction between variables and parameters is enforced to check that the model is square (there
        are as many equations as there are variables), and to determine which objects to solve for during simulation.

        Another important job of a model is to hold information about the dimensions of the model. Variables and parameters
        are declared with abstract indices, called dims. This is meant to match the notation used in the literature. For
        example, a variable representing the demand for capital in sector i would be written as ..math :: K_{d,i}. To
        represent this in a CGE model, we would declare a Variable with name "K_d" and dims "i". The dims on the Variable
        are arbitrary, but become concrete when placed in the context of a model. Therefore, the model must be supplied with
        a dictionary of dimension names and labels, called "coordinates" in packages for multi-dimensional array handing
        (e.g. xArray). For example, if the model will recieve data with dimension i, then the coords dictionary should
        provide a mapping from "i" to a list of labels for the dimension i. For example, {"i": ["Agriculture", "Industry",
        "Service"]}.

        When considering the "squareness" of a model, the "unpacked" dimensions of the variables and equations are
        considered. Therefore, it may not be the case that the number of variables declared is equal to the number of
        equations declared.

        """

        self.numeraire = None
        self.coords: dict[str, list[str, ...]] = {} if coords is None else coords
        self.parse_equations_to_sympy = parse_equations_to_sympy
        self.sparse = use_sparse_matrices

        self._sympy_cache = {}
        self._pytensor_cache = {}
        self._symbolic_coords = {sp.Idx(k): v for k, v in self.coords.items()}

        self._variables = {}
        self._parameters = {}
        self._equations = {}

        self._unpacked_variables = {}
        self._unpacked_parameters = {}
        self._unpacked_equations = {}

        _log.info("Initializing variables")
        self._initialize_group(variables, "variables")
        if numeraire is not None:
            if numeraire not in self.variable_names:
                raise ValueError("Requested numeraire not found among supplied variables")
            self.numeraire = self._variables[numeraire]

        _log.info("Initializing parameters")
        self._initialize_group(parameters, "parameters")

        _log.info("Initializing equations")
        self._initialize_group(equations, "equations")

        if self.parse_equations_to_sympy:
            _log.info("Expanding reduction Ops (Sum, Prod)")
            self._simplify_unpacked_sympy_representation()

        if numeraire:
            del self._variables[numeraire]

        self.n_equations = len(self.unpacked_equation_names)
        self.n_variables = len(self.unpacked_variable_names)
        self.n_parameters = len(self.unpacked_parameter_names)

        self.check_initialization()
        _log.info(
            f"Initial pre-processing complete, found {self.n_equations} equations, {self.n_variables} variables, "
            f"{self.n_parameters} parameters"
        )

        self.scenarios: dict[str, Result] = {}

        self._compiled: list[str] = []
        self._compile_backend: Literal["pytensor", "numba"] | None = None

        self.f_system: Callable | None = None
        self.f_resid: Callable | None = None
        self.f_grad: Callable | None = None
        self.f_hess: Callable | None = None
        self.f_hessp: Callable | None = None
        self.f_jac: Callable | None = None
        self.f_dX: Callable | None = None

        self.symbolic_solution_matrices = {}

        if compile is not None:
            if compile in ["all", True]:
                compile = ["root", "minimize", "euler"]
            else:
                compile = [compile] if not isinstance(compile, list) else compile

            compile = cast(list[CompiledFunctions], compile)
            msg = f"Beginning compilation using {backend} backend"
            if backend == "pytensor":
                text_mode = "C (FAST_RUN)" if mode is None else mode
                msg += f" using {text_mode} mode"

            _log.info(msg)
            self._compile(
                backend=backend,
                mode=mode,
                functions_to_compile=compile,
                use_scan_euler=use_scan_euler,
            )
            self.mode = mode

        else:
            _log.info("Model equations parsed and loaded. Skipping model compilation")

    def check_initialization(self):
        if self.n_variables != self.n_equations:
            raise ValueError(
                f"Found {self.n_variables} variables but {self.n_equations} equations. "
                "System is not square. Check your model equations."
            )

    def _initialize_group(self, objects, group_name):
        if objects is None:
            return

        add_func = getattr(self, f"add_{group_name}")
        unpack_func = getattr(self, f"_unpack_{group_name}")

        add_func(objects)
        unpack_func(objects)

    def _simplify_unpacked_sympy_representation(self, n_jobs=-1):
        equations = [eq._eq for eq in self.unpacked_equations]
        variables = [x.to_sympy() for x in self.unpacked_variables]
        parameters = [x.to_sympy() for x in self.unpacked_parameters]

        # Remove indices from equations, variables, and parameters
        remove_indices_subdict = indexed_variables_to_sub_dict(
            self.unpacked_variables + self.unpacked_parameters
        )

        _log.info("Post-processing sympy equations")
        equations = sub_all_eqs(equations, remove_indices_subdict, n_jobs)

        _log.info("Post-processing sympy variables")
        variables = sub_all_eqs(variables, remove_indices_subdict, n_jobs)

        _log.info("Post-processing sympy parameters")
        parameters = sub_all_eqs(parameters, remove_indices_subdict, n_jobs)

        for group, simplified_symbols in zip(
            ["parameters", "variables", "equations"], [parameters, variables, equations]
        ):
            for name, symbol in zip(
                getattr(self, f"unpacked_{group[:-1]}_names"), simplified_symbols
            ):
                getattr(self, f"_unpacked_{group}")[name]["symbol"] = symbol

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
    def unpacked_variable_symbols(self) -> list[sp.Symbol]:
        return [
            self._unpacked_variables[var_name]["symbol"]
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
    def unpacked_parameter_symbols(self) -> list[sp.Symbol]:
        return [
            self._unpacked_parameters[param_name]["symbol"]
            for param_name in self.unpacked_parameter_names
        ]

    @property
    def equation_names(self) -> list[str]:
        return list(self._equations.keys())

    @property
    def equations(self) -> list[Equation]:
        return list(self._equations.values())

    @property
    def unpacked_equation_names(self):
        return list(self._unpacked_equations.keys())

    @property
    def unpacked_equation_symbols(self):
        if self.parse_equations_to_sympy:
            return [
                self._unpacked_equations[eq_name]["symbol"]
                for eq_name in self.unpacked_equation_names
            ]
        else:
            raise NotImplementedError(
                "Symbolic equations not available when sympy parsing is disabled"
            )

    @property
    def unpacked_equations(self):
        if self.parse_equations_to_sympy:
            return [
                self._unpacked_equations[eq_name]["modelobj"]
                for eq_name in self.unpacked_equation_names
            ]
        else:
            raise NotImplementedError("Cannot unpack equations when sympy parsing is disabled")

    def _add_object(
        self,
        obj: Variable | Parameter | Equation,
        group: ValidGroups,
        overwrite: bool = False,
    ):
        obj_dict = getattr(self, f"_{group}")
        obj_names = getattr(self, f"{group[:-1]}_names")
        if obj.name in obj_names and not overwrite:
            raise ValueError(
                f"Cannot add {obj.name}; a {group} of this name already exists. Pass overwrite=True to "
                f"allow existing {group}s to be overwritten."
            )
        obj_dict.update({obj["name"]: obj})

    def _unpack_objects(self, objs: list[Variable | Parameter | Equation]):
        """
        Convert a single object with dimension index to a list of objects with coordinate indices.

        Parameters
        ----------
        objs : Variable | Parameter | Equation
            The object to be unpacked.

        Returns
        -------
        list[Variable | Parameter | Equation]
            A list of objects with coordinate indices.
        """

        expanded_objects = []
        for obj in objs:
            expanded_objs = expand_obj_by_indices(
                obj, self.coords, dims=None, on_unused_dim="ignore"
            )
            expanded_objects.extend(expanded_objs)

        return expanded_objects

    def _add_objects(
        self,
        obj_list: list[Variable | Parameter | Equation],
        group: ValidGroups,
        overwrite: bool = False,
    ):
        for obj in obj_list:
            self._add_object(obj, group, overwrite)

    def add_parameter(self, parameter: Parameter, overwrite=False):
        _validate_input(parameter, Parameter)
        self._add_object(parameter, "parameters", overwrite)

    def add_parameters(self, parameters: list[Parameter], overwrite=False):
        parameters = ensure_input_is_sequence(parameters)
        [_validate_input(parameter, Parameter) for parameter in parameters]
        self._add_objects(parameters, "parameters", overwrite)

    def _unpack_parameters(self, parameters: list[Parameter]):
        unpacked_params = self._unpack_objects(parameters)
        unpacked_names = [make_indexed_name(param) for param in unpacked_params]
        param_dicts = [{"modelobj": param} for param in unpacked_params]
        self._unpacked_parameters = dict(zip(unpacked_names, param_dicts))

    def add_variable(self, variable: Variable, overwrite=False):
        _validate_input(variable, Variable)
        self._add_object(variable, "variables", overwrite)

    def add_variables(self, variables: list[Variable], overwrite=False):
        variables = ensure_input_is_sequence(variables)
        [_validate_input(variable, Variable) for variable in variables]
        self._add_objects(variables, "variables", overwrite)

    def _unpack_variables(self, variables: list[Variable]):
        unpacked_vars = self._unpack_objects(variables)
        unpacked_names = [make_indexed_name(param) for param in unpacked_vars]
        param_dicts = [{"modelobj": param} for param in unpacked_vars]
        self._unpacked_variables = dict(zip(unpacked_names, param_dicts))

    def add_sympy_equation(self, equation: Equation, overwrite: bool = False):
        _validate_input(equation, Equation)
        var_dict = {k: v.to_sympy() for k, v in self._variables.items()}
        fancy_var_dict = {k: v.to_sympy(use_latex_name=True) for k, v in self._variables.items()}

        param_dict = {k: v.to_sympy() for k, v in self._parameters.items()}
        fancy_param_dict = {k: v.to_sympy(use_latex_name=True) for k, v in self._parameters.items()}

        str_dim_to_symbol = dict(zip(self.coords.keys(), self._symbolic_coords.keys()))

        local_dict = var_dict | param_dict | str_dim_to_symbol
        fancy_dict = fancy_var_dict | fancy_param_dict | str_dim_to_symbol

        try:
            sympy_eq = sp.parse_expr(
                equation.equation, local_dict=local_dict, transformations="all"
            )
            fancy_eq = sp.parse_expr(
                equation.equation, local_dict=fancy_dict, transformations="all"
            )
        except Exception as e:
            raise ValueError(
                f"""Could not parse equation "{equation.name}":\n{equation.equation}\n\nEncountered the """
                f"following error:\n{e}"
            )

        if self.numeraire:
            x = self.numeraire.to_sympy()
            sympy_eq = sympy_eq.subs({x: 1})

        # Standardize equation
        try:
            standard_eq = substitute_reduce_ops(sympy_eq.lhs - sympy_eq.rhs, self.coords)
        except Exception as e:
            raise ValueError(
                f"""Could not standardize equation "{equation.name}":\n{sympy_eq}\n\nEncountered the """
                f"following error:\n{e}"
            )

        eq_id = equation.eq_id
        if eq_id is None:
            eq_id = len(self.equations) + 1

        new_eq = _SympyEquation(
            name=equation.name,
            equation=equation.equation,
            symbolic_eq=sympy_eq,
            _eq=standard_eq,
            _fancy_eq=fancy_eq,
            dims=find_equation_dims(standard_eq, list(str_dim_to_symbol.values())),
            eq_id=eq_id,
        )

        self._add_object(new_eq, "equations", overwrite)

    def add_pytensor_equation(self, equation: Equation, overwrite: bool = False) -> None:
        self._add_object(equation, "equations", overwrite)

    def add_equations(self, equations: list[Equation], overwrite: bool = False):
        equations = ensure_input_is_sequence(equations)
        [_validate_input(equation, Equation) for equation in equations]
        add_function = (
            self.add_sympy_equation if self.parse_equations_to_sympy else self.add_pytensor_equation
        )
        for equation in equations:
            add_function(equation, overwrite)  # type: ignore

    def _unpack_equations(self, *args):
        if self.equations is None:
            raise ValueError("Cannot unpack equations before they are added to the model.")
        if self.parse_equations_to_sympy:
            unpacked_eqs = self._unpack_objects(self.equations)
            unpacked_names = [eq.name for eq in unpacked_eqs]
            param_dicts = [{"modelobj": eq} for eq in unpacked_eqs]
            self._unpacked_equations = dict(zip(unpacked_names, param_dicts))
        else:
            unpacked_eq_names = unpack_equation_strings(self.equations, self.coords)
            self._unpacked_equations = dict.fromkeys(unpacked_eq_names, None)

    def _get_object(self, obj_name: str, group: Literal["variables", "parameters"]):
        return getattr(self, f"_{group}")[obj_name]

    def get_parameter(self, param_name: str):
        return self._get_object(param_name, "parameters")

    def get_parameters(self, param_names: list[str] | None = None):
        if param_names is None:
            param_names = self.parameter_names
        param_names = ensure_input_is_sequence(param_names)
        return [self.get_parameter(name) for name in param_names]

    def get_variable(self, var_name: str):
        return self._get_object(var_name, "variables")

    def get_variables(self, var_names: list[str] | None = None):
        if var_names is None:
            var_names = self.variable_names
        var_names = ensure_input_is_sequence(var_names)
        return [self.get_variable(name) for name in var_names]

    def get(self, names: str | list[str, ...] | None = None) -> list[Variable | Parameter]:
        """
        Retrieve a list of model objects (variables or parameters) from the model by name.

        Parameters
        ----------
        names: str or list of str, optional
            Variable or parameter names to retrieve from the model. If None, return all model variables and parameters.

        Returns
        -------
        out: list of Variable or Parameter
            The requested model objects.
        """
        if names is None:
            names = self.parameter_names + self.variable_names
        if isinstance(names, str):
            names = [names]

        out = []
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

    def get_symbol(self, names: list[str] | None = None):
        if names is None:
            names = self.unpacked_parameter_names + self.unpacked_parameter_symbols
        if isinstance(names, str):
            names = [names]

        out = []
        for name in names:
            if name in self.unpacked_parameter_names:
                out.append(self._unpacked_parameters[name]["symbol"])
            elif name in self.unpacked_variable_names:
                out.append(self._unpacked_variables[name]["symbol"])
            else:
                raise ValueError(
                    f'Requested name "{name}" is not a known parameter or variable. Did you append the '
                    f"indices?"
                )

        if len(out) == 1:
            return out[0]

        return out

    def _euler_wrapper(self, f_dX, *, theta_final, n_steps, mode=None, **data):
        x0 = np.concatenate([np.atleast_1d(data[x]).ravel() for x in self.variable_names], axis=0)
        theta0 = np.concatenate(
            [np.atleast_1d(data[x]).ravel() for x in self.parameter_names], axis=0
        )

        with NumbaProgressBar(total=n_steps) as progress:
            result = euler_approx(f_dX, x0, theta0, theta_final, n_steps, progress)
        # Decompose the result back to a list of numpy arrays
        shapes = [
            infer_object_shape_from_coords(x, self.coords) for x in self.variables + self.parameters
        ]
        out = []
        cursor = 0
        for shape in shapes:
            s = int(np.prod(shape))
            out.append(result[:, cursor : cursor + s].reshape(-1, *shape))
            cursor += s

        return out

    def get_unpacked_parent_object(self, name):
        if name in self.unpacked_variable_names:
            return self._unpacked_variables[name]["modelobj"]
        elif name in self.unpacked_parameter_names:
            return self._unpacked_parameters[name]["modelobj"]
        else:
            raise ValueError(f"Object {name} not found in model")

    def _compile_numba(self, functions_to_compile: list[CompiledFunctions]):
        equations = self.unpacked_equation_symbols
        variables = self.unpacked_variable_symbols
        parameters = self.unpacked_parameter_symbols

        # Always compile the system -- used to check residuals
        self.f_system = numba_lambdify(
            variables, sp.Matrix(equations), parameters, ravel_outputs=True
        )

        if "root" in functions_to_compile:
            # Functions used by optimize.root to directly solve the system of equations
            # Compute Jacobian of the nonlinear system
            jac = sp.Matrix(equations).jacobian(variables)
            self.symbolic_solution_matrices["jacobian"] = jac

            self.f_jac = numba_lambdify(variables, jac, parameters)

        if "minimize" in functions_to_compile:
            # Symbolically compute loss function and derivatives
            resid = sum(eq**2 for eq in equations)
            grad = sp.Matrix([resid.diff(x) for x in variables])
            hess = grad.jacobian(variables)

            self.symbolic_solution_matrices["loss_function"] = resid
            self.symbolic_solution_matrices["gradient"] = grad
            self.symbolic_solution_matrices["hessian"] = hess

            # Functions used by optimize.minimize to solve the system of equations by minimizing the loss function
            # Save these as member functions so they can be reused by optimizers
            self.f_resid = numba_lambdify(variables, resid, parameters)
            self.f_grad = numba_lambdify(variables, grad, parameters, ravel_outputs=True)
            self.f_hess = numba_lambdify(variables, hess, parameters)

        if "euler" in functions_to_compile:
            # Compile the one-step linear approximation function used by the iterative Euler approximation
            # We don't need to save this because it's only used internally by the euler_approx function
            f_dX = numba_linearize_cge_func(self)

            self.f_euler = ft.partial(self._euler_wrapper, f_dX)

        self._compiled = functions_to_compile

    def _compile_pytensor(
        self,
        mode,
        functions_to_compile: list[CompiledFunctions],
        sparse: bool = False,
        use_scan_euler: bool = True,
    ):
        _log.info("Compiling model equations into pytensor graph")
        (variables, parameters), (system, jac, B) = compile_cge_model_to_pytensor(
            self, sparse=sparse
        )
        inputs = variables + parameters
        text_mode = "C" if mode is None else mode
        _log.info(f"Compiling CGE equations into {text_mode} function")

        # Always compile the system function, so we can compute residuals from calibration
        f_system = pytensor.function(
            inputs=inputs, outputs=system, mode=mode, on_unused_input="raise"
        )
        f_system.trust_inputs = True
        self.f_system = f_system

        if "root" in functions_to_compile:
            _log.info(f"Compiling Jacobian equations into {text_mode} function")

            # It's possible that certain variables/parameters don't influence the jacobian at all (for example if they
            # enter as additive constants in the equations). In this case, we still want to be able to pass them as
            # inputs, but they will be unused.
            f_jac = pytensor.function(
                inputs=inputs, outputs=jac, mode=mode, on_unused_input="ignore"
            )
            f_jac.trust_inputs = True

            if mode in ["JAX", "NUMBA"]:
                _f_system = f_system.copy()
                _f_jac = f_jac.copy()

                def f_system(*args, **kwargs):
                    [x] = _f_system.vm.jit_fn(*args, **kwargs)
                    return np.asarray(x)

                def f_jac(*args, **kwargs):
                    [x] = _f_jac.vm.jit_fn(*args, **kwargs)
                    return np.asarray(x)

                self.f_system = f_system
                self.f_jac = f_jac

            else:
                self.f_system = f_system
                self.f_jac = f_jac

        if "minimize" in functions_to_compile:
            f_resid, f_grad, f_hess, f_hessp = None, None, None, None

            if mode == "JAX":
                f_resid, f_grad, f_hessp = jax_loss_grad_hessp(system, variables, parameters)
            else:
                _log.info("Computing sum of squared errors")
                resid = (system**2).mean()
                rewrite_pregrad(resid)

                _log.info("Computing SSE gradient")
                grad = pytensor.grad(resid, variables)
                grad = pt.specify_shape(
                    pt.concatenate([pt.atleast_1d(eq).ravel() for eq in grad]),
                    self.n_variables,
                )

                # View grad as the function, compute jvp instead  (hessp)
                # _log.info("Computing SSE Jacobian")
                # hessp, p = make_jacobian(grad, variables, return_jvp=True)
                hess = make_jacobian(grad, variables)

                _log.info(
                    f"Compiling SSE functions (sse, gradient, jacobian) into {text_mode} function"
                )

                # f_resid_grad_hess = pytensor.function(inputs=inputs, outputs=[resid, grad, hess], mode=mode)
                # f_resid_grad_hess.trust_inputs = True
                #
                # def f_resid(*args, **kwargs):
                #     return f_resid_grad_hess(*args, **kwargs)[0]
                #
                # def f_grad(*args, **kwargs):
                #     return f_resid_grad_hess(*args, **kwargs)[1]
                #
                # def f_hess(*args, **kwargs):
                #     return f_resid_grad_hess(*args, **kwargs)[2]

                f_resid = pytensor.function(inputs=inputs, outputs=resid, mode=mode)
                f_resid.trust_inputs = True

                f_grad = pytensor.function(inputs=inputs, outputs=grad, mode=mode)
                f_grad.trust_inputs = True

                # f_hessp = pytensor.function(inputs=inputs, outputs=hessp, mode=mode)
                # f_hessp.trust_inputs = True

                f_hess = pytensor.function(inputs=inputs, outputs=hess, mode=mode)
                f_hess.trust_inputs = True

                # f_resid_grad = pytensor.function(inputs=inputs, outputs=[resid, grad], mode=mode)
                # f_resid_grad.trust_inputs = True

            self.f_resid = f_resid
            self.f_grad = f_grad
            self.f_hess = f_hess
            self.f_hessp = f_hessp
            # self.f_resid_grad = f_resid_grad

        if "euler" in functions_to_compile:
            _log.info("Compiling euler approximation function")
            if use_scan_euler:

                def __pytensor_euler_helper(*args, n_steps=100, **kwargs):
                    if n_steps == self.__last_n_steps:
                        _log.info("Re-using compiled euler approximation")
                        return self.__compiled_f_euler(*args, **kwargs)
                    else:
                        _log.info(
                            f"n_steps has changed ({n_steps} != {self.__last_n_steps}). Compiling euler approximation"
                        )
                        self.__last_n_steps = n_steps
                        self.__compiled_f_euler = compile_euler_approximation_function(
                            jac,
                            B,
                            variables,
                            parameters,
                            n_steps=n_steps,
                            mode=self.mode,
                        )

                        return self.__compiled_f_euler(*args, **kwargs)

                self.__last_n_steps = 0
                self.f_euler = __pytensor_euler_helper

            else:
                if mode == "JAX":
                    f_step = jax_euler_step(system, variables, parameters)
                else:
                    inputs, outputs = pytensor_euler_step(system, jac, B, variables, parameters)
                    # inputs = get_required_inputs(outputs)
                    f_step = pytensor.function(inputs, outputs, mode=mode)
                    f_step.trust_inputs = True
                    self.f_step = f_step

                def f_euler(theta_final, n_steps, progressbar=True, update_freq=1, **data):
                    current_params = {k: v for k, v in data.items() if k in self.parameter_names}
                    current_variables = {k: v for k, v in data.items() if k in self.variable_names}
                    theta_final = flat_array_to_variable_dict(
                        theta_final, self.parameters, self.coords
                    )
                    theta_final = {f"{k}_final": v for k, v in theta_final.items()}
                    theta_initial = {
                        f"{k}_initial": v for k, v in data.items() if k in self.parameter_names
                    }

                    results = {
                        k: np.empty((n_steps, *getattr(data[k], "shape", ())))
                        for k in self.variable_names + self.parameter_names
                    }
                    progress = progress_bar(range(n_steps), total=n_steps, display=progressbar)
                    progress.update(0)
                    desc = "{:,.2f} {}, f = {:,.5g}"

                    for t in progress:
                        start = time.time()
                        current_step = f_step(
                            **current_params,
                            **current_variables,
                            **theta_final,
                            **theta_initial,
                            n_steps=n_steps,
                        )

                        current_variable_vals = current_step[: len(self.variable_names)]
                        current_parameter_vals = current_step[len(self.variable_names) :]

                        # current_variable_vals = flat_current_step[
                        #     : len(self.variable_names)
                        # ]
                        # current_parameter_vals = flat_current_step[
                        #     len(self.variable_names) :
                        # ]

                        current_variables = {
                            k: current_variable_vals[i] for i, k in enumerate(self.variable_names)
                        }
                        current_params = {
                            k: current_parameter_vals[i] for i, k in enumerate(self.parameter_names)
                        }

                        errors = self.f_system(**current_variables, **current_params)

                        rmse = (errors**2).sum() / min(1, int(np.prod(errors.shape)))
                        end = time.time()
                        elapsed = end - start
                        unit = "s/it"
                        if elapsed < 1:
                            # if elapsed is exactly zero just put a big number
                            elapsed = 1 / elapsed if elapsed != 0 else 1000000
                            unit = "it/s"

                        progress.comment = desc.format(elapsed, unit, rmse)

                        for k in data.keys():
                            results[k][t] = (
                                current_variables[k]
                                if k in self.variable_names
                                else current_params[k]
                            )

                    errors = self.f_system(**current_variables, **current_params)
                    rmse = (errors**2).sum() / min(1, int(np.prod(errors.shape)))
                    progress.update_bar(n_steps)
                    progress.comment = desc.format(elapsed, unit, rmse)

                    return list(results.values())

                self.f_euler = f_euler

        self._compiled = functions_to_compile

    def _compile(
        self,
        backend: Literal["pytensor", "numba"] | None = "pytensor",
        mode=None,
        functions_to_compile: list[CompiledFunctions] = "all",
        use_scan_euler: bool = True,
    ):
        """
        Compile the model to a backend.

        Parameters
        ----------
        backend: str
            The backend to compile to. One of 'pytensor' or 'numba'.
        mode: str
            Pytensor compile mode. Ignored if mode is not 'pytensor'.
        functions_to_compile: list of str
            A list of functions to compile. Valid choices are 'root', 'minimum', 'euler', or 'all'. Defaults to 'all'.
        use_scan_euler: bool
            Whether to directly compile the loop over iterations of the euler approximation using pytensor's scan Op.
            If False, the inner function will be compiled, but the outer loop will be done in Python. Defaults to True.
            Ignored if backend is not 'pytensor'.

        Returns
        -------
        None
        """
        if functions_to_compile is None or not functions_to_compile:
            _log.info("Skipping model compilation")
            return

        if functions_to_compile in ["all", True]:
            functions_to_compile = ["root", "minimize", "euler"]
        functions_to_compile = (
            [functions_to_compile]
            if not isinstance(functions_to_compile, list)
            else functions_to_compile
        )
        functions_to_compile = cast(list[CompiledFunctions], functions_to_compile)

        if all([fn in self._compiled for fn in functions_to_compile]):
            _log.info(f"Model already compiled with {functions_to_compile} functions")
            return

        else:
            functions_to_compile = [fn for fn in functions_to_compile if fn not in self._compiled]
            _log.info(f"Compiling model with {functions_to_compile} functions")

        if backend == "numba":
            self._compile_numba(functions_to_compile=functions_to_compile)
        elif backend == "pytensor":
            self._compile_pytensor(
                mode=mode,
                functions_to_compile=functions_to_compile,
                use_scan_euler=use_scan_euler,
                sparse=self.sparse,
            )
        else:
            raise NotImplementedError(
                f'Only "numba" and "pytensor" backends are supported, got {backend}'
            )

        self._compile_backend = backend

    def summary(
        self,
        variables: Literal["all", "variables", "parameters"] | list[str] = "all",
        results: Result | list[Result] | None = None,
        expand_indices: bool = False,
        index_labels: dict[str, list[str]] | None = None,
    ):
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

        info_dict = {"Symbol": [], "Description": []}

        for var_name in variables:
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

    def equation_table(self, display=True):
        eq_dict = [eq.to_dict() for eq in self.equations]
        info_dict = {"": [], "Name": [], "Equation": []}
        for i, d in enumerate(eq_dict):
            info_dict[""].append(i + 1)
            info_dict["Name"].append(_replace_dim_marker_with_dim_name(d["name"]))
            info_dict["Equation"].append(d["fancy_eq"])

        display_latex_table(info_dict)

    def calibrate(self, data=None, max_iter=100, name="initial_calibration"):
        if not self._compiled:
            self._compile()

        equations = self.unpacked_equation_symbols
        symbolic_data = {self.get_symbol(k): v for k, v in data.items()}

        all_model_objects = self.unpacked_variable_symbols + self.unpacked_parameter_symbols

        calibrated_system = recursive_solve_symbolic(equations, symbolic_data, max_iter)
        initial_values = np.array(
            [data.get(x.name, np.nan) for x in all_model_objects], dtype="float64"
        )
        fitted_values = np.array(
            [calibrated_system.get(x, np.nan) for x in all_model_objects],
            dtype="float64",
        )

        x0, theta0 = (
            fitted_values[: len(self.unpacked_variable_names)],
            fitted_values[len(self.unpacked_variable_names) :],
        )

        success = self.f_resid(x0, theta0) < 1e-6

        result = Result(
            name=name,
            success=success,
            variables=self.unpacked_variable_names,
            parameters=self.unpacked_parameter_names,
            initial_values=initial_values,
            fitted_values=fitted_values,
        )

        return result

    def _solve_with_euler_approximation(self, data, theta_final, n_steps):
        result = self.f_euler(**data, theta_final=theta_final, n_steps=n_steps)
        return list_of_array_to_idata(result, self)

    def _euler_approximation_Op(self, n_steps=100):
        if not self._compile_backend == "pytensor":
            raise ValueError('Can only create an fgraph when mode is "pytensor"')

        flat_equations, variables, parameters, _ = pytensor_objects_from_CGEModel(self)
        theta_final, result = euler_approximation(
            flat_equations, variables, parameters, n_steps=n_steps
        )
        theta_final.name = "theta_final"

        inputs = variables + parameters + [theta_final]

        euler_op = pytensor.compile.builders.OpFromGraph(inputs, result, inline=True)
        return euler_op

    def _solve_with_root(
        self,
        data,
        theta_final,
        use_jac=True,
        fixed_values=None,
        progressbar=True,
        **optimizer_kwargs,
    ):
        f_system = self.f_system
        f_jac = self.f_jac
        free_variables = self.variables

        use_hess = optimizer_kwargs.pop("use_hess", None)
        method = optimizer_kwargs.pop("method", "hybr")
        tol = optimizer_kwargs.pop("tol", 1e-6)
        _ = optimizer_kwargs.pop("use_hessp", None)

        if use_hess:
            _log.warning(
                'The "use_hess" argument is not used by the root optimizer and will be ignored.'
            )
        if use_jac and "root" not in self._compiled:
            raise ValueError("Jacobian has not been compiled.")

        if fixed_values is not None:
            if any(x in self.parameter_names for x in fixed_values.keys()):
                raise ValueError(
                    "The following parameters were requested to be fixed via the fixed_values argument, "
                    "but parameters are always fixed and should not be included in the fixed_values dict: "
                    f"{[x for x in fixed_values.keys() if x in self.parameter_names]}"
                )
            free_variables = [x for x in self.variables if x.name not in fixed_values]
            f_system = wrap_fixed_values(f_system, fixed_values, self.variables, self.coords)
            f_jac = wrap_fixed_values(f_jac, fixed_values, self.variables, self.coords)

        if self._compile_backend == "pytensor":
            parameters = self.parameters
            coords = self.coords

            f_system = wrap_pytensor_func_for_scipy(f_system, free_variables, parameters, coords)
            f_jac = wrap_pytensor_func_for_scipy(f_jac, free_variables, parameters, coords)

        x0, theta0 = variable_dict_to_flat_array(data, free_variables, self.parameters)
        maxeval = optimizer_kwargs.pop("niter", 5000)
        objective = CostFuncWrapper(
            maxeval=maxeval,
            f=f_system,
            args=theta_final,
            f_jac=f_jac if use_jac else None,
            progressbar=progressbar,
            tol=tol,
            backend=self._compile_backend,
        )

        f_optim = ft.partial(
            optimize.root,
            objective,
            x0=x0,
            method=method,
            jac=use_jac or None,
            # callback=objective.callback, # Callbacks don't work with root
            **optimizer_kwargs,
        )
        res = _optimzer_early_stopping_wrapper(f_optim)
        return res

    def _solve_with_minimize(
        self,
        data: dict[str, np.ndarray],
        theta_final: dict[str, np.ndarray],
        use_jac: bool = True,
        use_hess: bool = False,
        use_hessp: bool = True,
        progressbar: bool = True,
        fixed_values: dict[str, int | float | np.ndarray] | None = None,
        **optimizer_kwargs,
    ):
        if "minimize" not in self._compiled:
            raise ValueError("Minimization functions have not been compiled.")

        f_resid = self.f_resid
        f_grad = self.f_grad
        f_hess = self.f_hess
        f_hessp = self.f_hessp
        free_variables = self.variables

        if not use_jac and use_hess:
            use_hess = False
            _log.warning("use_hess is ignored when use_jac=False. Setting use_hess=False.")

        if fixed_values is not None:
            if any(x in self.parameter_names for x in fixed_values.keys()):
                raise ValueError(
                    "The following parameters were requested to be fixed via the fixed_values argument, "
                    "but parameters are always fixed and should not be included in the fixed_values dict: "
                    f"{[x for x in fixed_values.keys() if x in self.parameter_names]}"
                )

            free_variables = [x for x in self.variables if x.name not in fixed_values]
            f_resid = wrap_fixed_values(f_resid, fixed_values, self.variables, self.coords)
            f_grad = wrap_fixed_values(f_grad, fixed_values, self.variables, self.coords)
            f_hess = wrap_fixed_values(f_hess, fixed_values, self.variables, self.coords)

        if self._compile_backend == "pytensor" and self.mode != "JAX":
            parameters = self.parameters
            coords = self.coords

            f_resid = wrap_pytensor_func_for_scipy(f_resid, free_variables, parameters, coords)
            f_grad = wrap_pytensor_func_for_scipy(f_grad, free_variables, parameters, coords)
            f_hess = wrap_pytensor_func_for_scipy(f_hess, free_variables, parameters, coords)

        x0, theta0 = variable_dict_to_flat_array(data, free_variables, self.parameters)
        maxeval = optimizer_kwargs.pop("niter", 5000)
        tol = optimizer_kwargs.pop("tol", 1e-6)

        objective = CostFuncWrapper(
            maxeval=maxeval,
            f=f_resid,
            args=theta_final,
            f_jac=f_grad if use_jac else None,
            f_hess=f_hess if use_hess else None,
            progressbar=progressbar,
            tol=tol,
            backend=self._compile_backend,
        )

        f_optim = ft.partial(
            optimize.minimize,
            objective,
            x0,
            jac=use_jac or None,
            hess=objective.f_hess if use_hess else None,
            hessp=None if not use_hessp else lambda x, p: f_hessp(x, p, theta_final),
            callback=objective.callback,
            **optimizer_kwargs,
        )

        res = _optimzer_early_stopping_wrapper(f_optim)
        return res

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
            A dictionary of exact values for a subset of model variables. The keys should be the names of the variables
            to be changed, and the values should be numpy arrays of the same shape as the variable.

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
            A dictionary of variable values. The keys are the names of the variables, and the values are numpy arrays
            of the same shape as the variable.

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
        SOLVER_FACTORY = {
            "root": ft.partial(self._solve_with_root, use_jac=use_jac),
            "minimize": ft.partial(self._solve_with_minimize, use_jac=use_jac, use_hess=use_hess),
            "euler": ft.partial(self._solve_with_euler_approximation, n_steps=n_steps),
        }

        if solve_method not in SOLVER_FACTORY:
            raise ValueError(
                f"Unknown method {solve_method}. Must be one of {list(SOLVER_FACTORY.keys())}"
            )

        joint_dict = {**param_dict, **initial_variable_guess}
        free_variables = [var for var in self.variables if var.name not in fixed_values]
        _, flat_params = variable_dict_to_flat_array(joint_dict, free_variables, self.parameters)

        f_solver = SOLVER_FACTORY[solve_method]
        res = f_solver(
            data=joint_dict,
            theta_final=flat_params,
            fixed_values=fixed_values,
            **solver_kwargs,
        )

        if not solve_method == "euler":
            if not res.success:
                warnings.warn(
                    "Solver did not converge. Results do not represent a valid SAM, and are returned for "
                    "diagnostic purposes only"
                )

            result_dict = flat_array_to_variable_dict(res.x, free_variables, self.coords)
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

            .. warning:: The inital state **must** represent a model equlibrium! This will be checked automatically
            before simulation begins, and an error will be raised if the initial state does not represent an equlibrium.

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
                simulation_name += f'_{optimizer_mode}[{",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])}]'
        if not overwrite and os.path.exists(os.path.join(output_path, simulation_name + ".pkl")):
            res = read_results(output_path, simulation_name)
            return res

        if not self._compiled:
            if compile_kwargs is None:
                to_compile = []
                if use_euler_approximation:
                    to_compile.append("euler")
                if use_optimizer:
                    to_compile.append(optimizer_mode)

                compile_kwargs = {"functions_to_compile": to_compile}

            self._compile(**compile_kwargs)

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

        if not all(x in x0_var_param for x in self.variable_names + self.parameter_names):
            raise ValueError(
                f"initial_state must contain values for all variables and parameters in the model. Found {initial_state.keys()}"
            )

        final_param_dict = deepcopy(x0_var_param)
        if final_values is not None:
            final_param_dict.update(final_values)
        elif final_delta is not None:
            for k, v in final_delta.items():
                final_param_dict[k] += v
        elif final_delta_pct is not None:
            for k, v in final_delta_pct.items():
                final_param_dict[k] *= v
        else:
            raise ValueError()

        _, theta_simulation = variable_dict_to_flat_array(
            final_param_dict, self.variables, self.parameters
        )

        return_values = {}

        if use_euler_approximation:
            idata_euler = self._solve_with_euler_approximation(
                x0_var_param, theta_simulation, n_steps=n_iter_euler
            )
            return_values["euler"] = idata_euler
            x0_var_param = (
                idata_euler.isel(step=-1).to_dict()["variables"]
                | idata_euler.isel(step=-1).to_dict()["parameters"]
            )

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
        data: dict[str, float | np.ndarray] | InferenceData | xr.Dataset | OptimizeResult,
        tol=1e-6,
        print_equations=True,
    ):
        """
        Verify if a given state of the model is an equilibrium.

        Parameters
        ----------
        data: dict, InferenceData, or xr.Dataset
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

        if isinstance(data, InferenceData):
            if not ("variables" in data.groups() and "parameters" in data.groups()):
                raise ValueError("InferenceData must contain variables and parameters groups")

            if "step" in data["variables"].dims:
                data = (
                    data.isel(step=-1).to_dict()["variables"]
                    | data.isel(step=-1).to_dict()["parameters"]
                )
            else:
                data = data.to_dict()["variables"] | data.to_dict()["parameters"]

        elif isinstance(data, xr.Dataset):
            data = {x.name: data[x.name].values for x in self.variables + self.parameters}

        elif isinstance(data, OptimizeResult):
            variables = flat_array_to_variable_dict(data.x, self.variables, self.coords)
            parameters = flat_array_to_variable_dict(data.args, self.parameters, self.coords)
            data = {**variables, **parameters}

        if self._compile_backend == "pytensor":
            errors = self.f_system(**data)
        else:
            var_inputs, param_inputs = variable_dict_to_flat_array(
                data, self.variables, self.parameters
            )
            errors = self.f_system(var_inputs, param_inputs)
        sse = (errors**2).sum()

        if sse < tol:
            _log.info(f"Equilibrium found! Total squared error: {sse:0.6f}")
        else:
            _log.info(f"Equilibrium not found. Total squared error: {sse:0.6f}")
            longest_name = max(len(x) for x in self.unpacked_equation_names)

            _log.info("\n")
            _log.info(f'{"Equation":<{longest_name + 10}}' + "Residual")
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


def write_results(results, output_path, filename=None, overwrite=False):
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
        if os.path.exists(os.path.join(output_path, new_filename)):
            _log.warning(
                f"Overwriting existing file found at {os.path.join(output_path, new_filename)}"
            )

    with open(os.path.join(output_path, new_filename), "wb") as f:
        cloudpickle.dump(results, f)

    _log.info(f"Results saved to {os.path.join(output_path, new_filename)}")


def read_results(output_path, filename):
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

    _log.info(f"Results loaded from {os.path.join(output_path, filename)}")
    return results


def recursive_solve_symbolic(equations, known_values=None, max_iter=100):
    """
    Solve a system of symbolic equations iteratively, given known initial values

    Parameters
    ----------
    equations : list of Sympy expressions
        List of symbolic equations to be solved.
    known_values : dict of symbol, float; optional
        Dictionary of known initial values for symbols (default is an empty dictionary).
    max_iter : int, optional
        Maximum number of iterations (default is 100).

    Returns
    -------
    known_values : dict of symbol, float; optional
        Dictionary of solved values for symbols.
    """

    if known_values is None:
        known_values = {}
    unsolved = equations.copy()

    for _ in range(max_iter):
        new_solution_found = False
        simplified_equations = [sp.simplify(eq.subs(known_values)) for eq in unsolved]
        remove = []
        for i, eq in enumerate(simplified_equations):
            unknowns = [var for var in eq.free_symbols]
            if len(unknowns) == 1:
                unknown = unknowns[0]
                solution = sp.solve(eq, unknown)

                if isinstance(solution, list):
                    if len(solution) == 0:
                        solution = sp.core.numbers.Zero()
                    else:
                        solution = solution[0]

                known_values[unknown] = solution.subs(known_values).evalf()
                new_solution_found = True
                remove.append(eq)
            elif len(unknowns) == 0:
                remove.append(eq)
        for eq in remove:
            simplified_equations.remove(eq)
        unsolved = simplified_equations.copy()

        # Break if the system is solved, or if we're stuck
        if len(known_values) == len(equations):
            break
        if not new_solution_found:
            break

    if len(unsolved) > 0:
        msg = "The following equations were not solvable given the provided initial values:\n"
        msg += "\n".join([str(eq) for eq in unsolved])
        raise ValueError(msg)

    return known_values


def expand_compact_system(
    compact_equations,
    compact_variables,
    compact_params,
    index_dict,
    numeraire_dict=None,
    check_square=True,
):
    if numeraire_dict is None:
        numeraire_dict = {}
        numeraires = []
    else:
        numeraires = list(numeraire_dict.keys())

    index_symbols = list(index_dict.keys())
    index_dicts = [{k: v for k, v in enumerate(index_dict[idx])} for idx in index_symbols]

    compact_equations = [eq.doit() for eq in compact_equations]
    variables = enumerate_indexbase(
        compact_variables, index_symbols, index_dicts, expand_using="index"
    )
    named_variables = enumerate_indexbase(
        compact_variables, index_symbols, index_dicts, expand_using="name"
    )
    named_variables = [indexed_var_to_symbol(x) for x in named_variables]

    parameters = enumerate_indexbase(
        compact_params, index_symbols, index_dicts, expand_using="index"
    )
    named_parameters = enumerate_indexbase(
        compact_params, index_symbols, index_dicts, expand_using="name"
    )
    named_parameters = [indexed_var_to_symbol(x) for x in named_parameters]

    idx_equations = enumerate_indexbase(
        compact_equations, index_symbols, index_dicts, expand_using="index"
    )

    var_sub_dict = make_indexbase_sub_dict(variables)
    param_sub_dict = make_indexbase_sub_dict(parameters)

    named_var_sub_dict = make_indexbase_sub_dict(named_variables)
    named_param_sub_dict = make_indexbase_sub_dict(named_parameters)

    idx_var_to_named_var = dict(zip(var_sub_dict.values(), named_var_sub_dict.values()))
    idx_param_to_named_param = dict(zip(param_sub_dict.values(), named_param_sub_dict.values()))

    numeraires = [idx_var_to_named_var.get(var_sub_dict.get(x)) for x in numeraires]
    numeraire_dict = {k: v for k, v in zip(numeraires, numeraire_dict.values())}

    equations = sub_all_eqs(
        sub_all_eqs(idx_equations, var_sub_dict | param_sub_dict),
        idx_var_to_named_var | idx_param_to_named_param,
    )
    equations = sub_all_eqs(equations, numeraire_dict)

    [named_variables.remove(x) for x in numeraires]

    n_eq = len(equations)
    n_vars = len(named_variables)

    if check_square:
        if n_eq != n_vars:
            names = [x.name for x in numeraires]
            msg = "After expanding index sets"
            if len(names) > 0:
                msg += f' and removing {", ".join(names)},'
            msg += f" system is not square. Found {n_eq} equations and {n_vars} variables."
            raise ValueError(msg)

    return equations, named_variables, named_parameters


def compile_cge_to_numba(
    compact_equations,
    compact_variables,
    compact_params,
    index_dict,
    numeraire_dict=None,
):
    equations, variables, parameters = expand_compact_system(
        compact_equations, compact_variables, compact_params, index_dict, numeraire_dict
    )

    resid = sum(eq**2 for eq in equations)
    grad = sp.Matrix([resid.diff(x) for x in variables])
    jac = sp.Matrix(equations).jacobian(variables)
    hess = grad.jacobian(variables)

    f_system = numba_lambdify(variables, sp.Matrix(equations), parameters, ravel_outputs=True)
    f_resid = numba_lambdify(variables, resid, parameters)
    f_grad = numba_lambdify(variables, grad, parameters, ravel_outputs=True)
    f_hess = numba_lambdify(variables, hess, parameters)
    f_jac = numba_lambdify(variables, jac, parameters)

    return (f_resid, f_grad, f_hess), (f_system, f_jac), (variables, parameters)


def numba_linearize_cge_func(model: CGEModel):
    # equations, variables, parameters = expand_compact_system(
    #     compact_equations, compact_variables, compact_params, index_dict
    # )

    variables = model.unpacked_variable_symbols
    parameters = model.unpacked_parameter_symbols
    equations = model.unpacked_equation_symbols

    if "Jacobian" in model.symbolic_solution_matrices:
        A_mat = model.symbolic_solution_matrices["Jacobian"].copy()
    else:
        A_mat = sp.Matrix([[eq.diff(x) for x in variables] for eq in equations])

    B_mat = sp.Matrix([[eq.diff(x) for x in parameters] for eq in equations])
    model.symbolic_solution_matrices["B_matirx"] = B_mat

    sub_dict = {x: sp.Symbol(f"{x.name}_0", **x.assumptions0) for x in variables + parameters}

    A_sub = A_mat.subs(sub_dict)
    Bv = B_mat.subs(sub_dict) @ sp.Matrix([[x] for x in parameters])

    nb_A_sub = numba_lambdify(exog_vars=parameters, expr=A_sub, endog_vars=list(sub_dict.values()))
    nb_B_sub = numba_lambdify(exog_vars=parameters, expr=Bv, endog_vars=list(sub_dict.values()))

    @nb.njit
    def f_dX(endog, exog):
        A = nb_A_sub(endog, exog)
        B = nb_B_sub(endog, exog)

        return -np.linalg.solve(A, np.identity(A.shape[0])) @ B

    return f_dX
