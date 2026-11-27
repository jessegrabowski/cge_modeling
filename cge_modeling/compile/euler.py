from collections.abc import Callable
from typing import cast

import numpy as np
import pytensor

from better_optimize.utilities import ToggleableProgress
from pytensor import tensor as pt
from pytensor.compile.builders import OpFromGraph

from cge_modeling.base.cge import CGEModel
from cge_modeling.base.utilities import infer_object_shape_from_coords
from cge_modeling.compile.pytensor_tools import (
    clone_and_rename,
    flat_tensor_to_ragged_list,
    make_jacobian,
    rewrite_pregrad,
)
from cge_modeling.tools.utils import at_least_list


def _no_statistics(x: np.ndarray, theta: np.ndarray) -> dict[str, float]:
    return {}


def pytensor_euler_function_with_python_loop(
    x: np.ndarray,
    theta: np.ndarray,
    theta_final: np.ndarray,
    n_steps: int,
    *,
    cge_model: CGEModel,
    f_step,
    progress: ToggleableProgress,
    step_statistics: Callable[[np.ndarray, np.ndarray], dict[str, float]],
) -> list[np.ndarray]:
    trajectory = np.empty((n_steps + 1, x.shape[0] + theta.shape[0]))
    trajectory[0] = np.concatenate([x, theta])
    theta_initial = theta

    if not progress.is_enabled:
        step_statistics = _no_statistics

    with progress:
        task = progress.add_task("Euler", total=n_steps, **step_statistics(x, theta))
        for t in range(1, n_steps + 1):
            x, theta = f_step(x, theta, theta_initial, theta_final, n_steps)
            trajectory[t] = np.concatenate([x, theta])
            progress.update(task, advance=1, **step_statistics(x, theta))

        progress.update(task, completed=n_steps, refresh=True)

    results = []
    cursor = 0
    for obj in cge_model.variables + cge_model.parameters:
        shape = infer_object_shape_from_coords(obj, cge_model.coords)
        size = int(np.prod(shape))
        results.append(trajectory[:, cursor : cursor + size].reshape(n_steps + 1, *shape))
        cursor += size

    return results


def symbolic_euler_approximation(
    system: pt.TensorVariable,
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    jacobian: pt.TensorVariable | None = None,
    use_rk4: bool = False,
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
    theta_initial_vec = pt.concatenate([pt.atleast_1d(x).flatten() for x in theta_initial], axis=-1)
    theta_final = pt.tensor(
        name="theta_final", shape=theta_initial_vec.type.shape, dtype=theta_initial_vec.type.dtype
    )

    n_steps = pt.scalar("n_steps", dtype="int32")

    step_size = (theta_final - theta_initial_vec) / n_steps

    Bv = B @ pt.atleast_1d(step_size)
    Bv.name = "Bv"

    step = -pt.linalg.solve(A, Bv, assume_a="gen", check_finite=False)
    step = rewrite_pregrad(step)

    f_step = OpFromGraph(
        inputs=x_list + theta_list + theta_initial + [theta_final, n_steps],
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
        theta_final = args[-2]
        n_steps = args[-1]

        if use_rk4:
            dx1, step_size = f_step(*x_prev, *theta_prev, *theta_initial, theta_final, n_steps)
            delta_x1 = flat_tensor_to_ragged_list(dx1, x_shapes)
            delta_theta = flat_tensor_to_ragged_list(step_size, theta_shapes)

            dx2, _ = f_step(
                *[x + 0.5 * dx for x, dx in zip(x_prev, delta_x1)],
                *[theta + 0.5 * d_theta for theta, d_theta in zip(theta_prev, delta_theta)],
                *theta_initial,
                theta_final,
                n_steps,
            )
            delta_x2 = flat_tensor_to_ragged_list(dx2, x_shapes)

            dx3, _ = f_step(
                *[x + 0.5 * dx for x, dx in zip(x_prev, delta_x2)],
                *[theta + 0.5 * d_theta for theta, d_theta in zip(theta_prev, delta_theta)],
                *theta_initial,
                theta_final,
                n_steps,
            )
            delta_x3 = flat_tensor_to_ragged_list(dx3, x_shapes)

            dx4, _ = f_step(
                *[x + dx for x, dx in zip(x_prev, delta_x3)],
                *[theta + d_theta for theta, d_theta in zip(theta_prev, delta_theta)],
                *theta_initial,
                theta_final,
                n_steps,
            )
            delta_x4 = flat_tensor_to_ragged_list(dx4, x_shapes)

            delta_x = [
                (dx1 + 2 * dx2 + 2 * dx3 + dx4) / 6
                for dx1, dx2, dx3, dx4 in zip(delta_x1, delta_x2, delta_x3, delta_x4)
            ]

        else:
            step, step_size = f_step(*x_prev, *theta_prev, *theta_initial, theta_final, n_steps)
            delta_x = flat_tensor_to_ragged_list(step, x_shapes)
            delta_theta = flat_tensor_to_ragged_list(step_size, theta_shapes)

        x_next = [x + dx for x, dx in zip(x_prev, delta_x)]
        theta_next = [theta + dtheta for theta, dtheta in zip(theta_prev, delta_theta)]

        assert len(theta_next) == len(theta_prev)
        return x_next + theta_next

    result, _updates = pytensor.scan(
        step_func,
        outputs_info=x_list + theta_list,
        non_sequences=[*theta_list, theta_final, n_steps],
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
