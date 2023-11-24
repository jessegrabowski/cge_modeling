import re
from typing import Literal, Sequence, Union, cast

import pytensor
import pytensor.tensor as pt
import sympy as sp
from sympytensor import as_tensor

from cge_modeling.base.primitives import Parameter, Variable
from cge_modeling.sympy_tools import sub_all_eqs


def object_to_pytensor(obj: Union[Parameter, Variable], coords: dict[str, Sequence]):
    """
    Convert a CGE model object to a PyTensor object.

    Parameters
    ----------
    obj: Union[Parameter, Variable]
        The CGE model object to convert
    coords: dict[str, Sequence]
        A dictionary of coordinate names and values known by the CGE model. Used to determine the shapes and dimensions
        of the returned PyTensor object

    Returns
    -------
    pytensor_obj: pytensor.tensor.TensorVariable
        The PyTensor object representing the CGE model object

    Notes
    -----
    Distinction between Parameters and Variables is lost in the conversion to symbolic tensors. It is up to the user to
    maintain this distinction.

    Examples
    --------
    .. code-block:: python

        from cge_modeling.base.primitives import Parameter
        from cge_modeling.pytensorf.compile import object_to_pytensor

        K_d = Parameter(name='K_d', dims='i', description='Capital demand')
        coords = {'i': [0, 1, 2]}
        pt_K_d = object_to_pytensor(K_d, coords)
        pt_K_d.eval({pt_K_d: [1, 2, 3]})
        # Out: array([1., 2., 3.])
    """

    name = obj.name
    dims = obj.dims

    shape = tuple(list(map(lambda dim: len(coords[dim]), dims)))

    return pt.tensor(name, shape=shape)


def make_dummy_sub_dict(cge_model):
    """
    Create a dictionary of dummy symbols to replace indexed variables and parameters in a CGE model.

    Parameters
    ----------
    cge_model: CGEModel
        The CGE model object to create the dictionary for

    Returns
    -------
    dummy_dict: dict
        A dictionary with keys as model variables and values as dummy symbols.

    Notes
    -----
    This function is needed because Sympy cannot print indexed variables and parameters. We need to replace them
    with non-indexed dummies before printing. Dimension information will be contained in the pytensor tensors.

    """
    var_dict = {
        x.to_sympy(): sp.Symbol(x.base_name)
        for x in cge_model.variables
        if isinstance(x.to_sympy(), sp.Indexed)
    }
    param_dict = {
        x.to_sympy(): sp.Symbol(x.base_name)
        for x in cge_model.parameters
        if isinstance(x.to_sympy(), sp.Indexed)
    }

    return {**var_dict, **param_dict}


def replace_indexed_variables(equations, sub_dict):
    """
    Descriptive alias for sub_all_eqs
    """
    sp_equations = [eq._eq for eq in equations]
    return sub_all_eqs(sp_equations, sub_dict)


def make_printer_cache(variables: list[pt.TensorLike], parameters: list[pt.TensorLike]) -> dict:
    """
    Create a cache of PyTensor functions for printing sympy equations to pytensor.

    Parameters
    ----------
    cge_model: CGEModel
        CGEModel object
    variables: list[pytensor.tensor.TensorVariable]
        List of cge_model variables, converted to symbolic pytensor tensors
    parameters: list[pytensor.tensor.TensorVariable]
        List of cge_model parameters, converted to symbolic pytensor tensors

    Returns
    -------
    cache: dict[tuple[str, sp.Symbol, tuple, str, tuple], pt.TensorLike]
        A cache of variables used to print equations to pytensor
    """

    def make_key(name) -> tuple:
        return name, sp.Symbol, (), "floatX", ()

    cache = {make_key(var.name): var for var in variables + parameters}
    return cast(dict, cache)


def normalize_eq(eq: str) -> str:
    """
    Normalize an equation y = f(x) to the form y - f(x) = 0

    Parameters
    ----------
    eq: str
        A string representing an equation

    Returns
    -------
    normalized_eq: str
        A string representing the same equation in normalized form
    """

    lhs, rhs = re.split("(?<!axis)=", eq, 1)
    return f"{lhs} - ({rhs})"


def wrap_in_ravel(eq: str) -> str:
    return f"({eq}).ravel()"


def compile_cge_model_to_pytensor(
    cge_model, inverse_method: Literal["solve", "pinv", "svd"] = "solve", return_B_matrix=False
) -> tuple[pt.Op, pt.Op, pt.Op, pt.Op]:
    """
    Compile a CGE model to a PyTensor function.

    Parameters
    ----------
    cge_model: CGEModel
        The CGE model object to compile

    Returns
    -------
    f_model: pt.Op
        A PyTensor Op representing computation of the residuals of model equations given model variables and
        parameters as inputs

    f_jac: pt.Op
        A PyTensor Op representing computation of the Jacobian of model equations given model variables and
        parameters as inputs

    f_jac_inv: pt.Op
        A PyTensor Op representing computation of the inverse of the Jacobian of model equations given model
        variables and parameters as inputs

    f_B: pt.Op
        A PyTensor Op representing computation of the B matrix (derivative of equations with respect to parameters).
        Used in the Euler step method to find a new equilibrium after a change of parameters.

    Notes
    -----

    """
    n_eq = len(cge_model.unpacked_variable_names)
    variables = [object_to_pytensor(var, cge_model.coords) for var in cge_model.variables]
    parameters = [object_to_pytensor(param, cge_model.coords) for param in cge_model.parameters]
    inputs = variables + parameters

    cache = make_printer_cache(variables, parameters)

    if cge_model.parse_equations_to_sympy:
        remove_index_subs = make_dummy_sub_dict(cge_model)
        substituted_equations = replace_indexed_variables(cge_model.equations, remove_index_subs)
        pytensor_equations = [as_tensor(eq, cache=cache) for eq in substituted_equations]
    else:
        cache = {k[0]: v for k, v in cache.items()}
        pytensor_equations = [eval(normalize_eq(eq.equation), cache) for eq in cge_model.equations]

    flat_equations = pt.concatenate([eq.ravel() for eq in pytensor_equations], axis=-1)
    flat_equations = pt.specify_shape(flat_equations, (n_eq,))
    flat_equations.name = "equations"

    jac = pytensor.gradient.jacobian(flat_equations, variables)
    B_matirx_columns = pytensor.gradient.jacobian(flat_equations, parameters)

    # Reshape to (n_eq, -1) so that the variables end up on the columns and the equations on the rows of the Jacobian
    jac = pt.concatenate([pt.atleast_2d(x).reshape((n_eq, -1)) for x in jac], axis=-1)
    jac = pt.specify_shape(jac, (n_eq, n_eq))
    jac.name = "jacobian"

    B_matrix = pt.concatenate(
        [pt.atleast_2d(x).reshape((n_eq, -1)) for x in B_matirx_columns], axis=-1
    )
    B_matrix = pt.specify_shape(B_matrix, (n_eq, len(parameters)))
    B_matrix.name = "B_matrix"

    if inverse_method == "pinv":
        jac_inv = pt.linalg.pinv(jac)
    elif inverse_method == "solve":
        jac_inv = pt.linalg.solve(jac, pt.identity_like(jac), check_finite=False)
    elif inverse_method == "svd":
        U, S, V = pt.linalg.svd(jac)
        S_inv = pt.where(pt.gt(S, 1e-8), 1 / S, 0)
        jac_inv = V @ pt.diag(S_inv) @ U.T
    else:
        raise ValueError(
            f'Invalid inverse method {inverse_method}, expected one of "pinv", "solve", "svd"'
        )

    jac_inv = pt.specify_shape(jac_inv, (n_eq, n_eq))
    jac_inv.name = "inverse_jacobian"

    f_model = pytensor.compile.builders.OpFromGraph(inputs, outputs=[flat_equations], inline=True)
    f_jac = pytensor.compile.builders.OpFromGraph(inputs, outputs=[jac], inline=True)
    f_jac_inv = pytensor.compile.builders.OpFromGraph(inputs, outputs=[jac_inv], inline=True)
    f_B = pytensor.compile.builders.OpFromGraph(inputs, outputs=[B_matrix], inline=True)

    ret_vals = (f_model, f_jac, f_jac_inv)
    if return_B_matrix:
        ret_vals += (f_B,)

    return ret_vals


def compile_euler_step_function(x0, theta_0, f_jac_inv, f_B):
    x0_shared = pytensor.shared(x0, name="current_state")
    A_inv = f_jac_inv(*x0_shared, *theta_0)
    B = f_B(*x0_shared, *theta_0)

    euler_step = A_inv @ B @ theta_0

    return x0_shared, euler_step


#     sub_dict = {x: sp.Symbol(f"{x.name}_0", **x.assumptions0) for x in variables + parameters}
#
#     A_sub = A_mat.subs(sub_dict)
#     Bv = B_mat.subs(sub_dict) @ sp.Matrix([[x] for x in parameters])
#
#     nb_A_sub = numba_lambdify(exog_vars=parameters, expr=A_sub, endog_vars=list(sub_dict.values()))
#     nb_B_sub = numba_lambdify(exog_vars=parameters, expr=Bv, endog_vars=list(sub_dict.values()))
#
#     @nb.njit
#     def f_dX(endog, exog):
#         A = nb_A_sub(endog, exog)
#         B = nb_B_sub(endog, exog)
#
#         return -np.linalg.solve(A, np.identity(A.shape[0])) @ B
#
#     return f_dX
#
# def compile_euler_step():
