import functools as ft
from typing import Callable, Literal, Optional, Sequence, Union

import numba as nb
import numpy as np
import pytensor
import pytensor.tensor as pt
import sympy as sp
from scipy import optimize

from cge_modeling.base.primitives import (
    Equation,
    Parameter,
    Result,
    Variable,
    _SympyEquation,
)
from cge_modeling.base.utilities import (
    _replace_dim_marker_with_dim_name,
    _validate_input,
    ensure_input_is_sequence,
    infer_object_shape_from_coords,
    variable_dict_to_flat_array,
    wrap_pytensor_func_for_scipy,
)
from cge_modeling.pytensorf.compile import (
    compile_cge_model_to_pytensor,
    euler_approximation_from_CGEModel,
    pytensor_objects_from_CGEModel,
)
from cge_modeling.tools.numba_tools import euler_approx, numba_lambdify
from cge_modeling.tools.output_tools import display_latex_table, list_of_array_to_idata
from cge_modeling.tools.pytensor_tools import make_jacobian
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


class CGEModel:
    def __init__(
        self,
        coords: Optional[dict[str, Sequence[str]]] = None,
        variables: Optional[Union[list[Variable], dict[str, Variable]]] = None,
        parameters: Optional[Union[list[Parameter], dict[str, Parameter]]] = None,
        equations: Optional[Union[list[Equation], dict[str, Equation]]] = None,
        numeraire: Optional[Variable] = None,
        parse_equations_to_sympy: bool = True,
    ):
        self.numeraire = None
        self.coords = {} if coords is None else coords
        self.parse_equations_to_sympy = parse_equations_to_sympy
        self._symbolic_coords = {sp.Idx(k): v for k, v in self.coords.items()}

        self._variables = {}
        self._parameters = {}
        self._equations = {}

        self._unpacked_variables = {}
        self._unpacked_parameters = {}
        self._unpacked_equations = {}

        self._initialize_group(variables, "variables")
        if numeraire is not None:
            if numeraire not in self.variable_names:
                raise ValueError("Requested numeraire not found among supplied variables")
            self.numeraire = self._variables[numeraire]

        self._initialize_group(parameters, "parameters")
        self._initialize_group(equations, "equations")

        if self.parse_equations_to_sympy:
            self._simplify_unpacked_sympy_representation()

        if numeraire:
            del self._variables[numeraire]

        # self.n_equations = len(self.unpacked_equation_names)
        self.n_variables = len(self.unpacked_variable_names)
        self.n_parameters = len(self.unpacked_parameter_names)

        self.scenarios: dict[str, Result] = {}

        self._compiled: bool = False
        self._compile_backend: Optional[Literal["pytensor", "numba"]] = None

        self.f_system: Optional[Callable] = None
        self.f_resid: Optional[Callable] = None
        self.f_grad: Optional[Callable] = None
        self.f_hess: Optional[Callable] = None
        self.f_jac: Optional[Callable] = None
        self.f_dX: Optional[Callable] = None

    def _initialize_group(self, objects, group_name):
        if objects is None:
            return

        add_func = getattr(self, f"add_{group_name}")
        unpack_func = getattr(self, f"_unpack_{group_name}")

        add_func(objects)
        unpack_func(objects)

    def _simplify_unpacked_sympy_representation(self):
        equations = [eq._eq for eq in self.unpacked_equations]
        variables = [x.to_sympy() for x in self.unpacked_variables]
        parameters = [x.to_sympy() for x in self.unpacked_parameters]

        # Remove indices from equations, variables, and parameters
        remove_indices_subdict = indexed_variables_to_sub_dict(
            self.unpacked_variables + self.unpacked_parameters
        )
        equations = sub_all_eqs(equations, remove_indices_subdict)
        variables = sub_all_eqs(variables, remove_indices_subdict)
        parameters = sub_all_eqs(parameters, remove_indices_subdict)

        for group, simplified_symbols in zip(
            ["parameters", "variables", "equations"], [parameters, variables, equations]
        ):
            for name, symbol in zip(
                getattr(self, f"unpacked_{group[:-1]}_names"), simplified_symbols
            ):
                getattr(self, f"_unpacked_{group}")[name]["symbol"] = symbol

    @property
    def variable_names(self):
        return list(self._variables.keys())

    @property
    def variables(self):
        return list(self._variables.values())

    @property
    def unpacked_variable_names(self):
        return list(self._unpacked_variables.keys())

    @property
    def unpacked_variables(self):
        return [
            self._unpacked_variables[var_name]["modelobj"]
            for var_name in self.unpacked_variable_names
        ]

    @property
    def unpacked_variable_symbols(self):
        return [
            self._unpacked_variables[var_name]["symbol"]
            for var_name in self.unpacked_variable_names
        ]

    @property
    def parameter_names(self):
        return list(self._parameters.keys())

    @property
    def parameters(self):
        return list(self._parameters.values())

    @property
    def unpacked_parameter_names(self):
        return list(self._unpacked_parameters.keys())

    @property
    def unpacked_parameters(self):
        return [
            self._unpacked_parameters[param_name]["modelobj"]
            for param_name in self.unpacked_parameter_names
        ]

    @property
    def unpacked_parameter_symbols(self):
        return [
            self._unpacked_parameters[param_name]["symbol"]
            for param_name in self.unpacked_parameter_names
        ]

    @property
    def equation_names(self):
        return list(self._equations.keys())

    @property
    def equations(self):
        return list(self._equations.values())

    @property
    def unpacked_equation_names(self):
        return list(self._unpacked_equations.keys())

    @property
    def unpacked_equation_symbols(self):
        return [
            self._unpacked_equations[eq_name]["symbol"] for eq_name in self.unpacked_equation_names
        ]

    @property
    def unpacked_equations(self):
        return [
            self._unpacked_equations[eq_name]["modelobj"]
            for eq_name in self.unpacked_equation_names
        ]

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
            if isinstance(obj, Equation) and not self.parse_equations_to_sympy:
                expanded_objects.append(obj)
                continue

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

    def add_equation(self, equation: Equation, overwrite: bool = False):
        _validate_input(equation, Equation)
        var_dict = {k: v.to_sympy() for k, v in self._variables.items()}
        fancy_var_dict = {k: v.to_sympy(use_latex_name=True) for k, v in self._variables.items()}

        param_dict = {k: v.to_sympy() for k, v in self._parameters.items()}
        fancy_param_dict = {k: v.to_sympy(use_latex_name=True) for k, v in self._parameters.items()}

        str_dim_to_symbol = dict(zip(self.coords.keys(), self._symbolic_coords.keys()))

        local_dict = var_dict | param_dict | str_dim_to_symbol
        fancy_dict = fancy_var_dict | fancy_param_dict | str_dim_to_symbol
        if self.parse_equations_to_sympy:
            # TODO: Should i call substitute_reduce_ops here to remove the sum/product over dummy indices in the equation
            #  lists? Downside: it will make very long expressions if the dim labels are long.
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
        else:
            new_eq = equation

        self._add_object(new_eq, "equations", overwrite)

    def add_equations(self, equations: list[Equation], overwrite: bool = False):
        equations = ensure_input_is_sequence(equations)
        [_validate_input(equation, Equation) for equation in equations]
        for equation in equations:
            self.add_equation(equation, overwrite)

    def _unpack_equations(self, *args):
        if self.equations is None:
            raise ValueError("Cannot unpack equations before they are added to the model.")
        unpacked_eqs = self._unpack_objects(self.equations)
        unpacked_names = [eq.name for eq in unpacked_eqs]
        param_dicts = [{"modelobj": param} for param in unpacked_eqs]
        self._unpacked_equations = dict(zip(unpacked_names, param_dicts))

    def _get_object(self, obj_name: str, group: Literal["variables", "parameters"]):
        return getattr(self, f"_{group}")[obj_name]

    def get_parameter(self, param_name: str):
        return self._get_object(param_name, "parameters")

    def get_parameters(self, param_names: Optional[list[str]] = None):
        if param_names is None:
            param_names = self.parameter_names
        param_names = ensure_input_is_sequence(param_names)
        return [self.get_parameter(name) for name in param_names]

    def get_variable(self, var_name: str):
        return self._get_object(var_name, "variables")

    def get_variables(self, var_names: Optional[list[str]] = None):
        if var_names is None:
            var_names = self.variable_names
        var_names = ensure_input_is_sequence(var_names)
        return [self.get_variable(name) for name in var_names]

    def get(self, names: Optional[list[str]] = None):
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

    def get_symbol(self, names: Optional[list[str]] = None):
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

    def _compile_numba(self):
        equations = self.unpacked_equation_symbols
        variables = self.unpacked_variable_symbols
        parameters = self.unpacked_parameter_symbols

        # Symbolically compute loss function and derivatives
        resid = sum(eq**2 for eq in equations)
        grad = sp.Matrix([resid.diff(x) for x in variables])
        hess = grad.jacobian(variables)

        # Compute Jacobian of the nonlinear system
        jac = sp.Matrix(equations).jacobian(variables)

        # Functions used by optimize.root to directly solve the system of equations
        self.f_system = numba_lambdify(
            variables, sp.Matrix(equations), parameters, ravel_outputs=True
        )
        self.f_jac = numba_lambdify(variables, jac, parameters)

        # Functions used by optimize.minimize to solve the system of equations by minimizing the loss function
        # Save these as member functions so they can be reused by optimizers
        self.f_resid = numba_lambdify(variables, resid, parameters)
        self.f_grad = numba_lambdify(variables, grad, parameters, ravel_outputs=True)
        self.f_hess = numba_lambdify(variables, hess, parameters)

        # Compile the one-step linear approximation function used by the iterative Euler approximation
        # We don't need to save this because it's only used internally by the euler_approx function
        f_dX = numba_linearize_cge_func(equations, variables, parameters)

        def euler_wrapper(*, theta_final, n_steps, **data):
            x0 = np.array([data[x] for x in self.variable_names])
            theta0 = np.array([data[x] for x in self.parameter_names])

            result = euler_approx(f_dX, x0, theta0, theta_final, n_steps)
            # Decompose the result back to a list of numpy arrays
            shapes = [
                infer_object_shape_from_coords(x, self.coords)
                for x in self.variables + self.parameters
            ]
            out = []
            cursor = 0
            for shape in shapes:
                s = int(np.prod(shape))
                out.append(result[:, cursor : cursor + s].reshape(-1, *shape))
                cursor += s

            return out

        self.f_euler = euler_wrapper
        self._compiled = True

    def __pytensor_euler_helper(self, *args, n_steps=100, **kwargs):
        if n_steps == self.__last_n_steps:
            return self.__compiled_f_euler(*args, **kwargs)
        else:
            self.__last_n_steps = n_steps
            self.__compiled_f_euler = euler_approximation_from_CGEModel(self, n_steps=n_steps)
            return self.__compiled_f_euler(*args, **kwargs)

    def _compile_pytensor(self, mode, inverse_method="solve"):
        (variables, parameters), (system, jac, jac_inv, B) = compile_cge_model_to_pytensor(
            self, inverse_method=inverse_method
        )
        inputs = variables + parameters
        self.f_system = pytensor.function(inputs=inputs, outputs=system, mode=mode)
        self.f_jac = pytensor.function(inputs=inputs, outputs=jac, mode=mode)

        resid = (system**2).sum()
        grad = pytensor.grad(resid, variables)
        grad = pt.specify_shape(
            pt.concatenate([pt.atleast_1d(eq).ravel() for eq in grad]), self.n_variables
        )
        hess = make_jacobian(grad, variables)

        f_root = pytensor.function(inputs=inputs, outputs=[resid, grad, hess], mode=mode)

        def f_resid(**kwargs):
            return f_root(**kwargs)[0]

        def f_grad(**kwargs):
            return f_root(**kwargs)[1]

        def f_hess(**kwargs):
            return f_root(**kwargs)[2]

        self.f_resid = f_resid
        self.f_grad = f_grad
        self.f_hess = f_hess

        self.__last_n_steps = 0
        self.f_euler = self.__pytensor_euler_helper
        self._compiled = True

    def _compile(
        self,
        backend: Optional[Literal["pytensor", "numba"]] = "pytensor",
        mode=None,
        inverse_method="solve",
    ):
        """
        Compile the model to a backend.

        Parameters
        ----------
        backend: str
            The backend to compile to. One of 'pytensor' or 'numba'.
        mode: str
            Pytensor compile mode. Ignored if mode is not 'pytensor'.
        inverse_method: str
            The method to use to compute the inverse of the Jacobian. One of "solve", "pinv", or "svd".
            Defaults to "solve". Ignored if mode is not 'pytensor'

        Returns
        -------
        None
        """
        if backend == "numba":
            self._compile_numba()
        elif backend == "pytensor":
            self._compile_pytensor(mode=mode, inverse_method=inverse_method)
        else:
            raise NotImplementedError(
                f'Only "numba" and "pytensor" backends are supported, got {backend}'
            )

        self._compile_backend = backend

    def summary(
        self,
        variables: Union[Literal["all", "variables", "parameters"], list[str]] = "all",
        results: Optional[Union[Result, list[Result]]] = None,
        expand_indices: bool = False,
        index_labels: Optional[dict[str, list[str]]] = None,
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
            [calibrated_system.get(x, np.nan) for x in all_model_objects], dtype="float64"
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

    def _solve_with_root(self, data, theta_final, use_jac=True, **optimizer_kwargs):
        if self._compile_backend == "numba":
            f_system = self.f_system
            f_jac = self.f_jac
        elif self._compile_backend == "pytensor":
            variables = self.variables
            parameters = self.parameters
            coords = self.coords

            f_system = wrap_pytensor_func_for_scipy(self.f_system, variables, parameters, coords)
            f_jac = wrap_pytensor_func_for_scipy(self.f_jac, variables, parameters, coords)
        else:
            raise ValueError(
                "Model must be compiled to a computational backend before it can be solved."
            )

        x0, theta0 = variable_dict_to_flat_array(data, self)

        res = optimize.root(
            f_system,
            x0,
            jac=f_jac if use_jac else None,
            args=theta_final,
            **optimizer_kwargs,
        )

        return res

    def _solve_with_minimize(
        self, data, theta_final, use_jac=True, use_hess=True, **optimizer_kwargs
    ):
        if self._compile_backend == "numba":
            f_resid = self.f_resid
            f_grad = self.f_grad
            f_hess = self.f_hess
        elif self._compile_backend == "pytensor":
            variables = self.variables
            parameters = self.parameters
            coords = self.coords

            f_resid = wrap_pytensor_func_for_scipy(self.f_resid, variables, parameters, coords)
            f_grad = wrap_pytensor_func_for_scipy(self.f_grad, variables, parameters, coords)
            f_hess = wrap_pytensor_func_for_scipy(self.f_hess, variables, parameters, coords)

        else:
            raise ValueError(
                "Model must be compiled to a computational backend before it can be solved."
            )

        x0, theta0 = variable_dict_to_flat_array(data, self)

        res = optimize.minimize(
            f_resid,
            x0,
            jac=f_grad if use_jac else None,
            hess=f_hess if use_hess else None,
            args=theta_final,
            **optimizer_kwargs,
        )

        return res

    def simulate(
        self,
        initial_state,
        final_values=None,
        final_delta=None,
        final_delta_pct=None,
        use_euler_approximation=True,
        use_root_solver=True,
        n_iter_euler=10_000,
        name=None,
        compile_kwargs=None,
        **optimizer_kwargs,
    ):
        """
        Simulate a shock to the model economy by computing the equilibrium state of the economy at the provided
        post-shock values.

        Parameters
        ----------
        initial_state: dict[str, np.arary]
            A dictionary of initial values for the model variables and parameters. The keys should be the names of the
            variables and parameters, and the values should be numpy arrays of the same shape as the variable or
            parameter.

        final_values:
        final_delta
        final_delta_pct
        n_iter_euler
        name
        compile_kwargs
        optimizer_kwargs

        Returns
        -------

        """
        if not self._compiled:
            if compile_kwargs is None:
                compile_kwargs = {}
            self._compile(**compile_kwargs)

        if isinstance(initial_state, Result):
            x0_var_param = initial_state.to_dict()["fitted"].copy()
        elif isinstance(initial_state, dict):
            x0_var_param = initial_state.copy()
        else:
            raise ValueError(
                f"initial_state must be a Result or a dict of initial values, found {type(initial_state)}"
            )

        x0, theta0 = state_dict_to_input_arrays(
            x0_var_param, self.unpacked_variable_names, self.unpacked_parameter_names
        )

        if final_values is not None:
            x0_var_param.update(final_values)
        elif final_delta is not None:
            for k, v in final_delta.items():
                x0_var_param[k] += v
        elif final_delta_pct is not None:
            for k, v in final_delta_pct.items():
                x0_var_param[k] *= v
        else:
            raise ValueError()

        x0, theta_simulation = state_dict_to_input_arrays(
            x0_var_param, self.unpacked_variable_names, self.unpacked_parameter_names
        )
        euler_result = euler_approx(self.f_dX, x0, theta0, theta_simulation, n_iter_euler)
        x0_improved = euler_result[: len(self.unpacked_variable_names)]

        res = optimize.minimize(
            self.f_resid,
            x0_improved,
            jac=self.f_grad,
            hess=self.f_hess,
            args=theta_simulation,
            **optimizer_kwargs,
        )

        result = Result(
            name=name,
            success=res.success,
            variables=self.unpacked_variable_names,
            parameters=self.unpacked_parameter_names,
            initial_values=initial_state.fitted_values,
            fitted_values=np.r_[res.x, theta_simulation],
        )

        return result

    def print_residuals(self, res):
        n_vars = len(self.unpacked_variable_names)
        endog, exog = res.fitted_values[:n_vars], res.fitted_values[n_vars:]
        errors = self.f_system(endog, exog)
        for eq, val in zip(self.unpacked_equation_names, errors):
            print(f"{eq:<75}: {val:<10.3f}")

    def check_for_equilibrium(self, data, tol=1e-6):
        errors = self.f_system(**data)
        sse = (errors**2).sum()

        if sse < tol:
            print(f"Equilibrium found! Total squared error: {sse:0.6f}")
        else:
            print(f"Equilibrium not found. Total squared error: {sse:0.6f}")


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
            msg = f"After expanding index sets"
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


def numba_linearize_cge_func(equations, variables, parameters):
    # equations, variables, parameters = expand_compact_system(
    #     compact_equations, compact_variables, compact_params, index_dict
    # )

    A_mat = sp.Matrix([[eq.diff(x) for x in variables] for eq in equations])
    B_mat = sp.Matrix([[eq.diff(x) for x in parameters] for eq in equations])

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


def state_dict_to_input_arrays(state_dict, variable_names, param_names):
    x = np.array([state_dict[k] for k in variable_names], dtype=float)
    theta = np.array([state_dict[x] for x in param_names], dtype=float)

    return x, theta
