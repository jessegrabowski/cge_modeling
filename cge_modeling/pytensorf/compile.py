from typing import Literal, cast

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph import FunctionGraph
from pytensor.graph.rewriting.basic import (
    SubstitutionNodeRewriter,
    WalkingGraphRewriter,
)
from sympytensor import as_tensor

from cge_modeling.tools.parsing import normalize_eq
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
    n_variables = len(variables)

    cache = make_printer_cache(variables, parameters)

    if cge_model.parse_equations_to_sympy:
        remove_index_subs = make_dummy_sub_dict(cge_model)
        substituted_equations = replace_indexed_variables(cge_model.equations, remove_index_subs)
        pytensor_equations = [as_tensor(eq, cache=cache) for eq in substituted_equations]
    else:
        cache = {k[0]: v for k, v in cache.items()}
        pytensor_equations = [eval(normalize_eq(eq.equation), cache) for eq in cge_model.equations]

    flat_equations = flatten_equations(pytensor_equations)
    flat_equations = pt.specify_shape(flat_equations, (n_eq,))
    flat_equations.name = "equations"

    # JAX currently throws an error when trying to compute the gradient of a Prod op with zeros in the input.
    # We shouldn't ever have this case anyway, so we can manually replace all Prod Ops with ones that have the
    # no_zeros_in_input flag set to True.
    # TODO: Fix this upstream in pytensor then remove all this
    default_prod_op = pt.math.Prod(dtype=pytensor.config.floatX, acc_dtype=pytensor.config.floatX)
    new_prod_op = pt.math.Prod(
        dtype=pytensor.config.floatX, acc_dtype=pytensor.config.floatX, no_zeros_in_input=True
    )

    local_add_no_zeros = SubstitutionNodeRewriter(default_prod_op, new_prod_op)
    add_no_zeros = WalkingGraphRewriter(local_add_no_zeros)
    fg = FunctionGraph(variables + parameters, outputs=[flat_equations])
    add_no_zeros.rewrite(fg)
    flat_equations = fg.outputs[0]
    variables, parameters = fg.inputs[:n_variables], fg.inputs[n_variables:]

    return flat_equations, variables, parameters


def compile_cge_model_to_pytensor(
    cge_model, inverse_method: Literal["solve", "pinv", "svd"] = "solve"
) -> tuple[tuple[list, list], tuple[pt.TensorLike, pt.TensorLike, pt.TensorLike, pt.TensorLike]]:
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
    inputs: tuple
        variables: list[pytensor.tensor.TensorVariable] of length cge_model.n_variables
            A list of pytensor variables representing the variables in the system of equations. The order of the
            variables is the same as the in which the user supplied them to the CGE model.

        parameters: list[pytensor.tensor.TensorVariable] of length cge_model.n_parameters
            A list of pytensor variables representing the parameters in the system of equations. The order of the
            parameters is the same as the in which the user supplied them to the CGE model.

    outputs: tuple
        flat_equations: pytensor.tensor.TensorVariable
            A single pytensor vector representing the equations in the system of equations. The length of the vector is
            equal to the number of *unrolled* equations in the system of equations.

        jac: pytensor.tensor.TensorVariable
            A pytensor matrix representing the Jacobian of the system of equations. The shape of the matrix is
            (n_eq, n_eq), where n_eq is the number of *unrolled* equations in the system of equations.

        jac_inv: pytensor.tensor.TensorVariable
            A pytensor matrix representing the inverse of the Jacobian of the system of equations.

        B: pytensor.tensor.TensorVariable
            A pytensor matrix representing the Jacobian of the system of equations with respect to the parameters.
            The shape of the matrix is (n_eq, n_params), where n_eq is the number of *unrolled* equations in the system
            of equations and n_params is the number of *unrolled* parameters in the system of equations.
    """
    flat_equations, variables, parameters = pytensor_objects_from_CGEModel(cge_model)
    n_eq = flat_equations.type.shape[0]

    jac = make_jacobian(flat_equations, variables)
    jac.name = "jacobian"

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

    B = make_jacobian(flat_equations, parameters)
    B.name = "B"

    inputs = (variables, parameters)
    outputs = (flat_equations, jac, jac_inv, B)

    return inputs, outputs


def compile_cge_model_to_pytensor_Op(
    cge_model, inverse_method: Literal["solve", "pinv", "svd"] = "solve"
) -> tuple[pt.Op, pt.Op, pt.Op]:
    """
    Compile a CGE model to a PyTensor Ops.

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
    In general, it shouldn't be necessary to use this function. Most downstream computation can directly use the graph
    generated by compile_cge_model_to_pytensor. In the case of optimization, however, the function and its Jacobian
    need to be "anonymous". This function exists to facilitate that use case.
    """

    inputs, outputs = compile_cge_model_to_pytensor(cge_model, inverse_method=inverse_method)
    flat_equations, jac, jac_inv, B = outputs
    variables, parameters = inputs
    inputs = variables + parameters

    f_model = pytensor.compile.builders.OpFromGraph(inputs, outputs=[flat_equations], inline=True)
    f_jac = pytensor.compile.builders.OpFromGraph(inputs, outputs=[jac], inline=True)
    f_jac_inv = pytensor.compile.builders.OpFromGraph(inputs, outputs=[jac_inv], inline=True)

    return f_model, f_jac, f_jac_inv


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
    x_list = cast(list, at_least_list(variables))
    theta_list = cast(list, at_least_list(parameters))
    theta_final = pt.concatenate(
        [pt.atleast_1d(clone_and_rename(x)).flatten() for x in theta_list], axis=-1
    )

    theta0 = flatten_equations(theta_list)

    dtheta = theta_final - theta0
    step_size = dtheta / n_steps

    A = make_jacobian(system, x_list)
    A_inv = pt.linalg.solve(A, pt.identity_like(A), check_finite=False)
    A_inv.name = "A_inv"

    B = make_jacobian(system, theta_list)
    Bv = B @ pt.atleast_1d(step_size)
    Bv.name = "Bv"

    step = -A_inv @ Bv
    f_step = pytensor.compile.builders.OpFromGraph(
        x_list + theta_list + [step_size], [step], inline=True
    )

    x_args = len(x_list)
    x_shapes = [x.type.shape for x in x_list]
    theta_shapes = [theta.type.shape for theta in theta_list]

    def step_func(*args):
        x_prev = args[:x_args]
        theta_prev = args[x_args:-1]
        step_size = args[-1]

        step = f_step(*x_prev, *theta_prev, step_size)
        x_next = []
        cursor = 0

        for x, shape in zip(x_prev, x_shapes):
            s = int(np.prod(shape))
            delta_var = step[cursor : cursor + s].reshape(shape)
            x_next.append(x + delta_var)
            cursor += s
        assert len(x_next) == len(x_prev)

        theta_next = []
        cursor = 0
        for theta, shape in zip(theta_prev, theta_shapes):
            s = int(np.prod(shape))
            delta_theta = step_size[cursor : cursor + s].reshape(shape)
            theta_next.append(theta + delta_theta)
            cursor += s
        assert len(theta_next) == len(theta_prev)

        return x_next + theta_next

    result, updates = pytensor.scan(
        step_func,
        outputs_info=x_list + theta_list,
        non_sequences=[step_size],
        n_steps=n_steps,
    )

    final_result = []
    for i, x in enumerate(x_list + theta_list):
        x_with_initial = pt.concatenate([pt.atleast_Nd(x, n=result[i].ndim), result[i]], axis=0)
        final_result.append(x_with_initial)

    return theta_final, final_result


def compile_euler_approximation_function(equations, variables, parameters, n_steps=100, mode=None):
    theta_final, result = euler_approximation(equations, variables, parameters, n_steps=n_steps)
    theta_final.name = "theta_final"
    inputs = variables + parameters + [theta_final]

    f_euler = pytensor.function(inputs, result, mode=mode)
    return f_euler


def euler_approximation_from_CGEModel(cge_model, n_steps=100, mode=None):
    flat_equations, variables, parameters = pytensor_objects_from_CGEModel(cge_model)
    return compile_euler_approximation_function(
        flat_equations, variables, parameters, n_steps=n_steps, mode=mode
    )
