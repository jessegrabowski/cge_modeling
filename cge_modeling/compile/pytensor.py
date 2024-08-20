import logging

from functools import partial
from typing import cast

import numpy as np
import pytensor

from pytensor import tensor as pt
from pytensor.compile import Function
from pytensor.compile.builders import OpFromGraph
from pytensor.graph.basic import explicit_graph_inputs

from cge_modeling import CGEModel
from cge_modeling.base.function_wrappers import return_array_from_jax_wrapper
from cge_modeling.base.utilities import flat_array_to_variable_dict
from cge_modeling.compile.constants import CompiledFunctions
from cge_modeling.compile.pytensor_tools import (
    clone_and_rename,
    flat_tensor_to_ragged_list,
    flatten_equations,
    make_printer_cache,
    object_to_pytensor,
    rewrite_pregrad,
)
from cge_modeling.tools.parsing import normalize_eq
from cge_modeling.tools.utils import at_least_list, initialize_progress_bar

_log = logging.getLogger(__name__)


def vars_and_params_to_pytensor(
    cge_model: CGEModel,
) -> tuple[list[pt.TensorVariable], list[pt.TensorVariable], dict]:
    variables = cge_model.variables
    parameters = cge_model.parameters
    coords = cge_model.coords

    variables = [object_to_pytensor(var, coords) for var in variables]
    parameters = [object_to_pytensor(param, coords) for param in parameters]

    cache = make_printer_cache(variables, parameters)

    return variables, parameters, cache


def cge_primitives_to_pytensor(
    cge_model: CGEModel, verbose: bool = True
) -> tuple[pt.TensorVariable, list[pt.TensorVariable], list[pt.TensorVariable], tuple[dict, dict]]:
    variables, parameters, cache = vars_and_params_to_pytensor(cge_model)
    n_eq = cge_model.n_equations

    if verbose:
        _log.info("Converting equations to pytensor graphs")

    cache = {k[0]: v for k, v in cache.items()}
    cache.update({"pt": pt})

    # Copy the cache to call with eval, otherwise python will add a bunch of global environment variables to it
    eval_cache = cache.copy()
    pytensor_equations = [eval(normalize_eq(eq.equation), eval_cache) for eq in cge_model.equations]

    flat_equations = flatten_equations(pytensor_equations)
    flat_equations = pt.specify_shape(flat_equations, (n_eq,))
    flat_equations.name = "equations"

    rewrite_pregrad(flat_equations)

    return flat_equations, variables, parameters, (cache, {})


def make_jacobian(
    system: pt.TensorLike,
    x: list[pt.TensorLike],
) -> pt.TensorVariable:
    """
    Make a Jacobian matrix from a system of equations and a list of variables.

    Parameters
    ----------
    system: pytensor.tensor.TensorVariable
        A vector of equations
    x: list[pytensor.tensor.TensorVariable]
        A list of variables

    Returns
    -------
    jac: pytensor.tensor.TensorVariable
        The Jacobian matrix of the system of equations with respect to the variables

    Notes
    -----
    The Jacobian of the system of equations is the matrix of partial derivatives of each equation with respect to each
    variable. The rows of the matrix correspond to the equations in the system, while the columns correspond to the
    variables. This function computes the Jacobian matrix from a vector of equations and a list of variables.
    """
    n_eq = system.type.shape[0]
    n_vars = int(np.sum([np.prod(var.type.shape) for var in x]))

    rewrite_pregrad(system)
    column_list = pytensor.gradient.jacobian(system, x)

    jac = pt.concatenate([pt.atleast_2d(x).reshape((n_eq, -1)) for x in column_list], axis=-1)
    jac = pt.specify_shape(jac, shape=(n_eq, n_vars))

    return jac


def make_jvp(
    grad: pt.TensorVariable, x: list[pt.TensorVariable], p: list[pt.TensorVariable] | None = None
) -> tuple[pt.TensorVariable, list[pt.TensorVariable]]:
    """
    Compute the jacobian-vector product between a system of equations and a vector of variables.

    Parameters
    ----------
    grad: pytensor.tensor.TensorVariable
        Vector of partial derivatives of a loss function with respect to input variables
    x: list[pytensor.tensor.TensorVariable]
        A list of variables

    Returns
    -------
    jvp: pytensor.tensor.TensorVariable
        The jacobian-vector product of the original loss function
    p: list of pytensor.tensor.TensorVariable
        The symbolic variables representing each component of ``p``, the point vector where the JVP is evaluated.
    """
    n_eq = grad.type.shape[0]
    rewrite_pregrad(grad)

    if p is None:
        p_vars = [var.type(name=f"{var.name}_point") for var in x]
        p = pt.concatenate([v.ravel() for v in p_vars], axis=0)

    jvp_chunks = pt.grad(pt.sum(grad * p), x)
    jvp = pt.concatenate([x.ravel() for x in jvp_chunks], axis=-1)
    jvp = pt.specify_shape(jvp, shape=(n_eq,))

    return jvp, p_vars


def validate_pytensor_parsing_result(flat_equations, variables, parameters, cache) -> None:
    if not all([x in cache.values() for x in variables + parameters]):
        missing_inputs = set(variables + parameters) - set(cache.values())
        raise ValueError(
            f"Cannot compile pytensor graph because the following inputs are missing: {missing_inputs}"
        )

    required_inputs = list(explicit_graph_inputs(flat_equations))
    missing_inputs = set(required_inputs) - set(variables + parameters)
    extra_inputs = set(variables + parameters) - set(required_inputs)

    if missing_inputs:
        raise ValueError(
            f"Cannot compile pytensor graph because the following inputs are missing: {missing_inputs}"
        )
    if extra_inputs:
        raise ValueError(
            f"Cannot compile pytensor graph because the following inputs were provided but are not required: {extra_inputs}"
        )


def compile_pytensor_jacobian_function(
    system: pt.TensorVariable,
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    mode: str | None = None,
) -> tuple[pt.TensorVariable, Function]:
    jac = make_jacobian(system, variables)
    jac.name = "jacobian"

    # It's possible that certain variables/parameters don't influence the jacobian at all (for example if they
    # enter as additive constants in the equations). In this case, we still want to be able to pass them as
    # inputs, but they will be unused.
    f_jac = pytensor.function(
        inputs=[*variables, *parameters], outputs=jac, mode=mode, on_unused_input="ignore"
    )
    f_jac.trust_inputs = True

    return jac, f_jac


def compile_pytensor_error_function(
    system: pt.TensorVariable,
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    mode: str | None = None,
) -> tuple[pt.TensorVariable, Function]:
    squared_loss = 0.5 * (system**2).sum()
    squared_loss.name = "squared_loss"
    rewrite_pregrad(squared_loss)

    f_loss = pytensor.function(inputs=[*variables, *parameters], outputs=squared_loss, mode=mode)
    f_loss.trust_inputs = True

    return squared_loss, f_loss


def compile_pytensor_gradient_function(
    loss: pt.TensorVariable,
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    n_variables: int,
    mode: str | None = None,
) -> tuple[pt.TensorVariable, Function]:
    grad = pytensor.grad(loss, variables)
    grad = pt.specify_shape(pt.concatenate([pt.atleast_1d(eq).ravel() for eq in grad]), n_variables)
    rewrite_pregrad(grad)

    f_grad = pytensor.function(
        inputs=[*variables, *parameters], outputs=grad, mode=mode, on_unused_input="ignore"
    )
    f_grad.trust_inputs = True

    return grad, f_grad


def compile_pytensor_hess_function(
    grad: pt.TensorVariable,
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    mode: str | None = None,
) -> tuple[pt.TensorVariable, Function]:
    hess = make_jacobian(grad, variables)
    hess.name = "hessian"

    f_hess = pytensor.function(
        inputs=[*variables, *parameters], outputs=hess, mode=mode, on_unused_input="ignore"
    )
    f_hess.trust_inputs = True

    return hess, f_hess


def compile_pytensor_hessp_function(
    grad: pt.TensorVariable,
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    mode: str | None = None,
) -> tuple[pt.TensorVariable, Function]:
    hessp, p_vars = make_jvp(grad, variables)
    hessp.name = "hessp"

    f_hessp = pytensor.function(inputs=[*variables, *parameters, *p_vars], outputs=hessp, mode=mode)
    f_hessp.trust_inputs = True

    return hessp, f_hessp


def _make_euler_matrices(
    system, variables, parameters, jacobian=None, grad=None
) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    if jacobian is None:
        A = make_jacobian(system, variables)
        A.name = "A"
    else:
        A = jacobian

    B = make_jacobian(system, parameters)
    B.name = "B"

    return A, B


def pytensor_euler_step(
    system, variables, parameters, jacobian=None, grad=None
) -> tuple[list[pt.TensorVariable], list[pt.TensorVariable]]:
    A, B = _make_euler_matrices(system, variables, parameters, jacobian)

    x_list = at_least_list(variables)
    x_shapes = [x.type.shape for x in x_list]

    theta_list = at_least_list(parameters)

    n_steps = pt.iscalar("n_steps")
    theta_final = [x.clone(name=f"{x.name}_final") for x in theta_list]
    theta0 = [x.clone(name=f"{x.name}_initial") for x in theta_list]

    step_size = [(x - y) / n_steps for x, y in zip(theta_final, theta0)]

    try:
        Bv = pytensor.gradient.Rop(system, parameters, step_size)
    except NotImplementedError:
        _log.warning(
            "Tried to compute jvp of model with respect to parameters but failed; an Op in your model does not "
            "have an implemented Rop method. Falling back to direct computation via jacobian. This is slow and "
            "memory intensive for large models."
        )
        v = flatten_equations(step_size)
        Bv = B @ v

    step = -pt.linalg.solve(A, Bv)
    step.name = "euler_step"

    delta_x = flat_tensor_to_ragged_list(step, x_shapes)
    x_next = [x + dx for x, dx in zip(x_list, delta_x)]

    theta_next = [theta + dtheta for theta, dtheta in zip(theta_list, step_size)]

    inputs = x_list + theta_list + theta0 + theta_final + [n_steps]
    outputs = x_next + theta_next

    return inputs, outputs


def get_step_statistics(
    current_variables, current_params, f_system=None, f_grad=None
) -> dict[str, float]:
    value_dict = {}
    if f_system is not None:
        errors = f_system(**current_variables, **current_params)
        rmse = (errors**2).sum() / min(1, int(np.prod(errors.shape)))
        value_dict["RMSE"] = rmse
    if f_grad is not None:
        grad_norm = np.linalg.norm(f_grad(**current_variables, **current_params))
        value_dict["grad_norm"] = grad_norm

    return value_dict


def pytensor_euler_function_with_python_loop(
    cge_model, f_step, f_system=None, f_grad=None, *, theta_final, n_steps, **data
) -> list[np.ndarray]:
    parameter_names = cge_model.parameter_names
    variable_names = cge_model.variable_names
    n_variables = len(variable_names)

    parameters = cge_model.parameters
    coords = cge_model.coords

    current_params = {k: v for k, v in data.items() if k in parameter_names}
    current_variables = {k: v for k, v in data.items() if k in variable_names}
    theta_final = flat_array_to_variable_dict(theta_final, parameters, coords)

    theta_final = {f"{k}_final": v for k, v in theta_final.items()}
    theta_initial = {f"{k}_initial": v for k, v in data.items() if k in parameter_names}

    results = {
        k: np.empty((n_steps, *getattr(data[k], "shape", ())))
        for k in variable_names + parameter_names
    }

    stat_string = None
    if f_system is not None:
        stat_string = "f = {task.fields[RMSE]:,.5g}"
    if f_grad is not None:
        stat_string = "f = {task.fields[RMSE]:,.5g}, ||grad|| = {task.fields[grad_norm]:,.5g}"

    with initialize_progress_bar(stat_string) as progress:
        for t in range(n_steps):
            current_step = f_step(
                **current_params,
                **current_variables,
                **theta_final,
                **theta_initial,
                n_steps=n_steps,
            )

            current_variable_vals = current_step[:n_variables]
            current_parameter_vals = current_step[n_variables:]

            current_variables = {k: current_variable_vals[i] for i, k in enumerate(variable_names)}
            current_params = {k: current_parameter_vals[i] for i, k in enumerate(parameter_names)}

            for k in data.keys():
                results[k][t] = current_variables[k] if k in variable_names else current_params[k]

            value_dict = get_step_statistics(current_variables, current_params, f_system, f_grad)

            if t == 0:
                task = progress.add_task("[green]Euler Approximation", total=n_steps, **value_dict)
            else:
                progress.update(task, advance=1, **value_dict)

    value_dict = get_step_statistics(current_variables, current_params, f_system, f_grad)
    progress.update(task, completed=n_steps, **value_dict)

    return list(results.values())


def symbolic_euler_approximation(
    system: pt.TensorVariable,
    variables: list[pt.Variable],
    parameters: list[pt.Variable],
    jacobian: pt.TensorVariable | None = None,
    grad: pt.TensorVariable | None = None,
) -> tuple[pt.TensorVariable, pt.TensorVariable, list[pt.TensorVariable]]:
    """
    Find the solution to a system of equations using the Euler approximation method.

    The Euler approximation method is a simple method for finding the solution to a system of equations. It is
    an extension of a first order Taylor approximation, improving on the linearized approximation by decomposing
    the move from the initial (known) solution, around which the system is linearized, to the final (unknown) point
    into a series of small steps. At each step, linear approximation is re-computed around the previous step. By this
    method, the solution to the system can be traced out to arbitrary precision by iteratively taking small steps.

    Parameters
    ----------
    system: TensorVariable
        Vector of model equations, normalized to the form f(x) = 0
    variables: list of pytensor.tensor.TensorVariable
        A list of pytensor variables representing the variables in the system of equations
    parameters: list of pytensor.tensor.TensorVariable
        A list of pytensor variables representing the parameters in the system of equations
    jacobian: TensorVariable, optional
        A matrix of partial derivatives df(x) / dx. If not provided, it will be computed from the the system and
        variables.
    n_steps: int
        The number of steps to take in the Euler approximation

    Returns
    -------
    theta_final: pytensor.tensor.TensorVariable
        A symbolic tensor representing the full trajectory of parameter updates used to compute the solution to the
        system of equations at the final point, theta_final[-1].

    n_steps: TensorVariable
        Symbolic scalar representing the number of steps in the euler approximation

    result: list of pytensor.tensor.TensorVariable
        The values of the variables at each step of the Euler approximation, corresponding to linear approximation
        around the point (x[i-1], theta_final[i-1]) at each step i.
    """
    A, B = _make_euler_matrices(system, variables, parameters, jacobian)

    x_list = cast(list, at_least_list(variables))
    theta_list = cast(list, at_least_list(parameters))

    theta_initial = [clone_and_rename(x, suffix="_initial") for x in theta_list]
    theta_final = [clone_and_rename(x, suffix="_final") for x in theta_list]

    theta_final_vec = pt.concatenate([pt.atleast_1d(x).flatten() for x in theta_final], axis=-1)
    theta_initial_vec = pt.concatenate([pt.atleast_1d(x).flatten() for x in theta_initial], axis=-1)

    n_steps = pt.scalar("n_steps", dtype="int32")

    step_size = (theta_final_vec - theta_initial_vec) / n_steps

    Bv = B @ pt.atleast_1d(step_size)
    Bv.name = "Bv"

    step = -pt.linalg.solve(A, Bv, assume_a="gen", check_finite=False)
    rewrite_pregrad(step)

    f_step = OpFromGraph(
        inputs=x_list + theta_list + theta_initial + theta_final + [n_steps],
        outputs=[step, step_size],
        inline=True,
    )

    x_args = len(x_list)
    theta_args = len(theta_list)
    x_shapes = [x.type.shape for x in x_list]
    theta_shapes = [theta.type.shape for theta in theta_list]

    def step_func(*args):
        x_prev = args[:x_args]
        theta_prev = args[x_args : x_args + theta_args]
        theta_initial = args[x_args + theta_args : x_args + 2 * theta_args]
        theta_final = args[x_args + 2 * theta_args : -1]
        n_steps = args[-1]

        step, step_size = f_step(*x_prev, *theta_prev, *theta_initial, *theta_final, n_steps)
        delta_x = flat_tensor_to_ragged_list(step, x_shapes)
        x_next = [x + dx for x, dx in zip(x_prev, delta_x)]

        delta_theta = flat_tensor_to_ragged_list(step_size, theta_shapes)
        theta_next = [theta + dtheta for theta, dtheta in zip(theta_prev, delta_theta)]

        assert len(theta_next) == len(theta_prev)
        return x_next + theta_next

    result, updates = pytensor.scan(
        step_func,
        outputs_info=x_list + theta_list,
        non_sequences=theta_list + theta_final + [n_steps],
        n_steps=n_steps,
        strict=True,
    )

    final_result = []
    for i, x in enumerate(x_list + theta_list):
        x_with_initial = cast(
            pt.TensorVariable,
            pt.concatenate([pt.atleast_Nd(x, n=result[i].ndim), result[i]], axis=0),
        )
        final_result.append(x_with_initial)

    return theta_final, n_steps, final_result


def compile_pytensor_cge_functions(
    cge_model: CGEModel,
    functions_to_compile: list[CompiledFunctions],
    mode: str | None = None,
    use_scan_for_euler: bool = False,
    *args,
    **kwargs,
) -> tuple[Function | None, ...]:
    system, variables, parameters, (cache, unpacked_cache) = cge_primitives_to_pytensor(cge_model)
    validate_pytensor_parsing_result(system, variables, parameters, cache)

    # Always compile the system -- used to check residuals
    f_system = pytensor.function(
        inputs=[*variables, *parameters], outputs=system, mode=mode, on_unused_input="raise"
    )
    f_system.trust_inputs = True
    if mode == "JAX":
        # This function needs to return an array, because the loss function displayed by minimize specifically checks
        # for an array. If it returns a jnp.Array, the check fails and an error is raised.
        f_system = return_array_from_jax_wrapper(f_system)

    # Optional functions
    f_jac = None
    f_resid = None
    f_grad = None
    f_hess = None
    f_hessp = None
    f_euler = None

    # jac and grad can be reused if both root/minimize and euler are requested
    jac = None
    grad = None

    if "root" in functions_to_compile:
        jac, f_jac = compile_pytensor_jacobian_function(system, variables, parameters, mode=mode)

    if "minimize" in functions_to_compile:
        squared_loss, f_resid = compile_pytensor_error_function(
            system, variables, parameters, mode=mode
        )
        grad, f_grad = compile_pytensor_gradient_function(
            squared_loss, variables, parameters, cge_model.n_variables, mode=mode
        )
        hess, f_hess = compile_pytensor_hess_function(grad, variables, parameters, mode=mode)
        hessp, f_hessp = compile_pytensor_hessp_function(grad, variables, parameters, mode=mode)

    if "euler" in functions_to_compile:
        if not use_scan_for_euler:
            inputs, outputs = pytensor_euler_step(
                system, variables, parameters, jacobian=jac, grad=grad
            )

            f_step = pytensor.function(inputs, outputs, mode=mode)
            f_step.trust_inputs = True

            f_euler = partial(
                pytensor_euler_function_with_python_loop,
                cge_model=cge_model,
                f_step=f_step,
                f_system=f_system,
                f_grad=f_grad,
            )
        else:
            theta_final, n_steps, euler_output = symbolic_euler_approximation(
                system, variables, parameters, jacobian=jac, grad=grad
            )
            f_euler = pytensor.function(
                inputs=[*variables, *parameters, *theta_final], outputs=euler_output, mode=mode
            )
            f_euler.trust_inputs = True

    return f_system, f_jac, f_resid, f_grad, f_hess, f_hessp, f_euler


def compile_cge_model_to_pytensor_Op(cge_model) -> tuple[pt.Op, pt.Op]:
    """
    Compile a CGE model to a PyTensor Ops.

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

    Notes
    -----
    In general, it shouldn't be necessary to use this function. Most downstream computation can directly use the graph
    generated by compile_cge_model_to_pytensor. In the case of optimization, however, the function and its Jacobian
    need to be "anonymous". This function exists to facilitate that use case.
    """

    system, variables, parameters, _ = cge_primitives_to_pytensor(cge_model)
    jac = make_jacobian(system, variables)

    inputs = list(variables) + list(parameters)

    f_model = OpFromGraph(inputs, outputs=[system], inline=True)
    f_jac = OpFromGraph(inputs, outputs=[jac], inline=True)

    return f_model, f_jac
