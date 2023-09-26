from abc import ABC
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Union

import numba as nb
import numpy as np
import sympy as sp

from cge_modeling.numba_tools import numba_lambdify
from cge_modeling.output_tools import display_info_as_table
from cge_modeling.sympy_tools import (
    enumerate_indexbase,
    indexed_var_to_symbol,
    make_indexbase_sub_dict,
    sub_all_eqs,
)


def _validate_input(obj, cls):
    if not isinstance(obj, cls):
        raise ValueError(f"Expected instance of type {cls.__name__}, found {type(obj).__name__}")


def ensure_input_is_sequence(x):
    if not isinstance(x, (list, tuple)):
        x = [x]
    return x


@dataclass(slots=True, order=True, frozen=True)
class ModelObject(ABC):
    name: str
    index: tuple[str] | str = ()
    latex_name: Optional[str] = None
    description: Optional[str] = None
    real: bool = True
    positive: bool = True
    assumptions: dict = field(default_factory=dict)

    def __post_init__(self):
        self.assumptions["real"] = self.real
        self.assumptions["positive"] = self.positive

        object.__setattr__(self, "sort_index", self.name)
        object.__setattr__(self, "index", self._initialize_index())
        object.__setattr__(self, "latex_name", self._initialize_latex_name())
        object.__setattr__(self, "description", self._initialize_description())

    def _make_index_string(self):
        idx_str = ""
        if len(self.index) > 0:
            idx_str = ",".join([x for x in self.index])

        return idx_str

    def _initialize_index(self):
        index = self.index
        if index is None:
            return ()

        # Case 1: Index is empty (valid), or a tuple of strings (valid)
        if isinstance(index, tuple):
            if len(index) == 0:
                return index
            elif all([isinstance(x, str) for x in index]):
                return index
            else:
                raise ValueError(f"index must be a string or a tuple of strings, found {index}")

        # Case 2: Index is a string. Convert to a tuple
        elif isinstance(index, str):
            idxs = index.split(",")
            return tuple(idx.strip() for idx in idxs)

        else:
            raise ValueError(f"index must be a string or a tuple of strings, found {index}")

    def _initialize_latex_name(self):
        if self.latex_name is not None:
            return self.latex_name
        idx_str = self._make_index_string()
        *base, subscript = self.name.split("_")
        if len(base) == 0:
            base, subscript = subscript, base
        if len(subscript) > 0:
            idx_str = f"{subscript},{idx_str}"

        latex_name = "_".join(base)
        if len(idx_str) > 0:
            latex_name = latex_name + "_{" + idx_str + "}"
        return latex_name

    def _initialize_description(self):
        if self.description is not None:
            return self.description
        return f"{self.latex_name}, Positive = {self.positive}, Real = {self.real}"

    def __getitem__(self, item: str):
        return getattr(self, item)


@dataclass(order=True, frozen=True)
class Variable(ModelObject):
    pass


@dataclass(order=True, frozen=True)
class Parameter(ModelObject):
    pass


class CGEModel:
    def __init__(
        self,
        coords: Optional[dict[str, Sequence[str]]] = None,
        variables: Optional[dict[str, Variable]] = None,
        parameters: Optional[dict[str, Parameter]] = None,
    ):

        self.coords = {} if coords is None else coords

        self._variables = {} if variables is None else variables
        self._parameters = {} if parameters is None else parameters

    @property
    def variable_names(self):
        return list(self._variables.keys())

    @property
    def variables(self):
        return list(self._variables.values())

    @property
    def parameter_names(self):
        return list(self._parameters.keys())

    @property
    def parameters(self):
        return list(self._parameters.values())

    def _add_object(
        self,
        obj: Variable | Parameter,
        group: Literal["variables", "parameters"],
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

    def _add_objects(
        self,
        obj_list: list[Variable | Parameter],
        group: Literal["variables", "parameters"],
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

    def add_variable(self, variable: Variable, overwrite=False):
        _validate_input(variable, Variable)
        self._add_object(variable, "variables", overwrite)

    def add_variables(self, variables: list[Variable], overwrite=False):
        variables = ensure_input_is_sequence(variables)
        [_validate_input(variable, Variable) for variable in variables]
        self._add_objects(variables, "variables", overwrite)

    def _get_object(self, obj_name: str, group: Literal["variables", "parameters"]):
        print(getattr(self, f"_{group}"))
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

    def get_any(self, names: Optional[list[str]] = None):
        if names is None:
            names = self.parameter_names + self.variable_names

        out = []
        for name in names:
            if name in self.variable_names:
                out.append(self._variables[name])
            elif name in self.parameter_names:
                out.append(self._parameters[name])
            else:
                raise ValueError(f'Requested name "{name}" is not a known parameter or variable.')

    def print_table(
        self,
        variables: Union[Literal["variables", "parameters"], list[str]],
        values: Optional[np.ndarray] = None,
        value_headers: Optional[list[str]] = None,
        expand_indices: bool = False,
        index_labels: Optional[dict[str, list[str]]] = None,
    ):

        if variables in ["variables", "parameters"]:
            variables = getattr(self, f"get_{variables}")

        var_info_list = []
        display_info_as_table()


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
                if solution:
                    known_values[unknown] = solution[0].subs(known_values).evalf()
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


def numba_linearize_cge_func(compact_equations, compact_variables, compact_params, index_dict):
    equations, variables, parameters = expand_compact_system(
        compact_equations, compact_variables, compact_params, index_dict
    )

    A_mat = sp.Matrix([[eq.diff(x) for x in variables] for eq in equations])
    B_mat = sp.Matrix([[eq.diff(x) for x in parameters] for eq in equations])

    sub_dict = {x: sp.Symbol(f"{x.name}_0", **x._assumptions0) for x in variables + parameters}

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
