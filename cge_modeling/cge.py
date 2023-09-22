from cge_modeling.numba_tools import numba_lambdify
from cge_modeling.sympy_tools import enumerate_indexbase, indexed_var_to_symbol, make_indexbase_sub_dict, sub_all_eqs
import sympy as sp
import numba as nb
import numpy as np


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
        msg = 'The following equations were not solvable given the provided initial values:\n'
        msg += '\n'.join([str(eq) for eq in unsolved])
        raise ValueError(msg)

    return known_values


def expand_compact_system(compact_equations, compact_variables, compact_params, index_dict, numeraire_dict=None,
                          check_square=True):
    if numeraire_dict is None:
        numeraire_dict = {}
        numeraires = []
    else:
        numeraires = list(numeraire_dict.keys())

    index_symbols = list(index_dict.keys())
    index_dicts = [{k: v for k, v in enumerate(index_dict[idx])} for idx in index_symbols]

    variables = enumerate_indexbase(compact_variables, index_symbols, index_dicts, expand_using='index')
    named_variables = enumerate_indexbase(compact_variables, index_symbols, index_dicts, expand_using='name')
    named_variables = [indexed_var_to_symbol(x) for x in named_variables]

    parameters = enumerate_indexbase(compact_params, index_symbols, index_dicts, expand_using='index')
    named_parameters = enumerate_indexbase(compact_params, index_symbols, index_dicts, expand_using='name')
    named_parameters = [indexed_var_to_symbol(x) for x in named_parameters]

    idx_equations = enumerate_indexbase(compact_equations, index_symbols, index_dicts, expand_using='index')

    var_sub_dict = make_indexbase_sub_dict(variables)
    param_sub_dict = make_indexbase_sub_dict(parameters)

    named_var_sub_dict = make_indexbase_sub_dict(named_variables)
    named_param_sub_dict = make_indexbase_sub_dict(named_parameters)

    idx_var_to_named_var = dict(zip(var_sub_dict.values(), named_var_sub_dict.values()))
    idx_param_to_named_param = dict(zip(param_sub_dict.values(), named_param_sub_dict.values()))

    numeraires = [idx_var_to_named_var.get(var_sub_dict.get(x)) for x in numeraires]
    numeraire_dict = {k: v for k, v in zip(numeraires, numeraire_dict.values())}

    equations = sub_all_eqs(sub_all_eqs(idx_equations, var_sub_dict | param_sub_dict),
                            idx_var_to_named_var | idx_param_to_named_param)
    equations = sub_all_eqs(equations, numeraire_dict)

    [named_variables.remove(x) for x in numeraires]

    n_eq = len(equations)
    n_vars = len(named_variables)

    if check_square:
        if n_eq != n_vars:
            names = [x.name for x in numeraires]
            msg = f'After expanding index sets'
            if len(names) > 0:
                msg += f' and removing {", ".join(names)},'
            msg += f' system is not square. Found {n_eq} equations and {n_vars} variables.'
            raise ValueError(msg)

    return equations, named_variables, named_parameters


def compile_cge_to_numba(compact_equations,
                         compact_variables,
                         compact_params,
                         index_dict,
                         numeraire_dict=None):
    equations, variables, parameters = expand_compact_system(compact_equations, compact_variables, compact_params,
                                                             index_dict, numeraire_dict)

    resid = sum([eq ** 2 for eq in equations])
    grad = sp.Matrix([resid.diff(x) for x in variables])
    jac = sp.Matrix(equations).jacobian(variables)
    hess = grad.jacobian(variables)

    f_system = numba_lambdify(variables, sp.Matrix(equations), parameters, ravel_outputs=True)
    f_resid = numba_lambdify(variables, resid, parameters)
    f_grad = numba_lambdify(variables, grad, parameters, ravel_outputs=True)
    f_hess = numba_lambdify(variables, hess, parameters)
    f_jac = numba_lambdify(variables, jac, parameters)

    return (f_resid, f_grad, f_hess), (f_system, f_jac), (variables, parameters)


def numba_linearize_cge_func(compact_equations,
                             compact_variables,
                             compact_params,
                             index_dict):
    equations, variables, parameters = expand_compact_system(compact_equations, compact_variables,
                                                             compact_params, index_dict)

    A_mat = sp.Matrix([[eq.diff(x) for x in variables] for eq in equations])
    B_mat = sp.Matrix([[eq.diff(x) for x in parameters] for eq in equations])

    sub_dict = {x: sp.Symbol(f'{x.name}_0', **x._assumptions0) for x in variables + parameters}

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
