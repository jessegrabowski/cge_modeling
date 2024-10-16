from typing import cast

import numpy as np
import pytensor

from pytensor import tensor as pt
from pytensor.compile.builders import OpFromGraph

from cge_modeling.base.cge import CGEModel
from cge_modeling.base.utilities import flat_array_to_variable_dict
from cge_modeling.compile.pytensor_tools import (
    clone_and_rename,
    flat_tensor_to_ragged_list,
    make_jacobian,
    rewrite_pregrad,
)
from cge_modeling.tools.utils import at_least_list, initialize_progress_bar


def pytensor_euler_function_with_python_loop(
    cge_model: CGEModel, f_step, f_system=None, f_grad=None, *, theta_final, n_steps, **data
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

    progress = initialize_progress_bar(stat_string)
    value_dict = get_step_statistics(current_variables, current_params, f_system, f_grad)
    task = progress.add_task("[green]Euler Approximation", total=n_steps, **value_dict)

    with progress:
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
            progress.update(task, advance=1, **value_dict)

        value_dict = get_step_statistics(current_variables, current_params, f_system, f_grad)
        progress.update(task, completed=n_steps, refresh=True, **value_dict)

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
