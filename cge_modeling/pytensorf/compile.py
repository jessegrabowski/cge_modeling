from typing import Literal, cast

import pytensor
import pytensor.tensor as pt
from sympytensor import as_tensor

from cge_modeling.parsing import normalize_eq
from cge_modeling.tools.pytensor_tools import (
    at_least_list,
    clone_and_rename,
    flatten_equations,
    make_jacobian,
    make_printer_cache,
    object_to_pytensor,
)
from cge_modeling.tools.sympy_tools import (
    make_dummy_sub_dict,
    replace_indexed_variables,
)


def pytensor_objects_from_CGEModel(cge_model):
    n_eq = len(cge_model.unpacked_variable_names)
    variables = [object_to_pytensor(var, cge_model.coords) for var in cge_model.variables]
    parameters = [object_to_pytensor(param, cge_model.coords) for param in cge_model.parameters]

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

    return flat_equations, variables, parameters


def compile_cge_model_to_pytensor(
    cge_model, inverse_method: Literal["solve", "pinv", "svd"] = "solve"
) -> tuple[pt.Op, pt.Op, pt.Op, pt.Op]:
    """
    Compile a CGE model to a PyTensor function.

    Parameters
    ----------
    cge_model: CGEModel
        The CGE model object to compile
    inverse_method: str, optional
        The method to use to compute the inverse of the Jacobian. One of "solve", "pinv", or "svd". Defaults to "solve".
        Note that if svd is chosen, gradients for autodiff will not be available.

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
    Notes
    -----

    """
    flat_equations, variables, parameters = pytensor_objects_from_CGEModel(cge_model)
    n_eq = flat_equations.type.shape[0]

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

    inputs = variables + parameters
    f_model = pytensor.compile.builders.OpFromGraph(inputs, outputs=[flat_equations], inline=True)
    f_jac = pytensor.compile.builders.OpFromGraph(inputs, outputs=[jac], inline=True)
    f_jac_inv = pytensor.compile.builders.OpFromGraph(inputs, outputs=[jac_inv], inline=True)

    ret_vals = (f_model, f_jac, f_jac_inv)
    return ret_vals


def euler_approximation(
    system: list[pt.TensorLike],
    variables: list[pt.TensorLike],
    parameters: list[pt.TensorLike],
    n_steps: int = 100,
):
    """
    Find the solution to a system of equations using the Euler approximation method.

    The Euler approximation method is a simple method for finding the solution to a system of equations. It is
    an extension of a first order Taylor approximation, improving on the linearized approximation by decomposing
    the move from the initial (known) solution, around which the system is linearized, to the final (unknown) point
    into a series of small steps. At each step, linear approximation is re-computed around the previous step. By this
    method, the solution to the system can be traced out to arbitrary precision by iteratively taking small steps.

    Parameters
    ----------
    system: list of pytensor.tensor.TensorLike
        A vector of equations of the form y(x, theta) = 0, where x are the variables and theta are the parameters
    variables: list of pytensor.tensor.TensorVariable
        A list of pytensor variables representing the variables in the system of equations
    parameters: list of pytensor.tensor.TensorVariable
        A list of pytensor variables representing the parameters in the system of equations
    n_steps: int
        The number of steps to take in the Euler approximation

    Returns
    -------
    theta_final: pytensor.tensor.TensorVariable
        A symbolic tensor representing the full trajectory of parameter updates used to compute the solution to the
        system of equations at the final point, theta_final[-1].

    result: pytensor.tensor.TensorVariable
        The values of the variables at each step of the Euler approximation, corresponding to linear approximation
        around the point (x[i-1], theta_final[i-1]) at each step i.

    """
    n_eq = len(system)
    flat_system = flatten_equations(system)

    x_list = cast(list, at_least_list(variables))
    theta_list = cast(list, at_least_list(parameters))
    theta_final = pt.stack([clone_and_rename(x) for x in theta_list])

    theta0 = flatten_equations(theta_list)

    dtheta = theta_final - theta0
    step_size = dtheta / n_steps

    A = make_jacobian(flat_system, x_list, n_eq)
    A_inv = pt.linalg.solve(A, pt.identity_like(A), check_finite=False)
    A_inv.name = "A_inv"

    B = make_jacobian(flat_system, theta_list, n_eq)
    Bv = B @ pt.atleast_1d(step_size)
    Bv.name = "Bv"

    step = -A_inv @ Bv
    f_step = pytensor.compile.builders.OpFromGraph(
        x_list + theta_list + [step_size], [step], inline=True
    )

    def step_func(x_prev, theta_prev, step_size):
        step = f_step(*x_prev, *theta_prev, step_size)

        x = x_prev + step
        theta = theta_prev + step_size

        return x, theta

    result, updates = pytensor.scan(
        step_func,
        outputs_info=[pt.stack(x_list), pt.stack(theta_list)],
        non_sequences=[step_size],
        n_steps=n_steps,
    )

    x_trajectory = pt.concatenate([pt.stack(x_list)[None], result[0]], axis=0)
    theta_trajectory = pt.concatenate([pt.stack(theta_list)[None], result[1]], axis=0)

    return theta_final, [x_trajectory, theta_trajectory]


def compile_euler_approximation_function(equations, variables, parameters, n_steps=100, mode=None):
    theta_final, result = euler_approximation(equations, variables, parameters, n_steps=n_steps)

    inputs = variables + parameters + [theta_final]

    f_euler = pytensor.function(inputs, result, mode=mode)
    return f_euler


def euler_approximation_from_CGEModel(cge_model, n_steps=100, mode=None):
    n_eq, flat_equations, variables, parameters = pytensor_objects_from_CGEModel(cge_model)
    return compile_euler_approximation_function(
        flat_equations, variables, parameters, n_steps=n_steps, mode=mode
    )
