import logging

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import pytensor

from better_optimize.utilities import ToggleableProgress
from pytensor import tensor as pt
from pytensor.compile.builders import OpFromGraph
from pytensor.graph.replace import graph_replace

from cge_modeling.compile.constants import StepMethod
from cge_modeling.compile.pytensor_tools import make_jacobian, pushforward

_log = logging.getLogger(__name__)

Tangent = Callable[[Any, Any, Any], Any]


def euler_step(f_tangent: Tangent, x: Any, theta: Any, theta_step: Any) -> Any:
    """Move ``x`` along the tangent at the start of the step."""
    return x + f_tangent(x, theta, theta_step)


def rk4_step(f_tangent: Tangent, x: Any, theta: Any, theta_step: Any) -> Any:
    """Move ``x`` by the classical fourth-order Runge-Kutta average of four tangents across the step."""
    half_step = theta_step / 2
    k1 = f_tangent(x, theta, theta_step)
    k2 = f_tangent(x + k1 / 2, theta + half_step, theta_step)
    k3 = f_tangent(x + k2 / 2, theta + half_step, theta_step)
    k4 = f_tangent(x + k3, theta + theta_step, theta_step)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


STEP_FUNCTIONS: dict[StepMethod, Callable[[Tangent, Any, Any, Any], Any]] = {
    "euler": euler_step,
    "rk4": rk4_step,
}


def split_trajectory(trajectory: Any, shapes: Sequence[tuple[int, ...]]) -> list[Any]:
    """Split the columns of a ``(n_points, n_values)`` trajectory into one array per object, with its own shape."""
    results = []
    cursor = 0
    for shape in shapes:
        size = int(np.prod(shape))
        results.append(trajectory[:, cursor : cursor + size].reshape((trajectory.shape[0], *shape)))
        cursor += size
    return results


def path_tangent(
    system: pt.TensorVariable,
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    vectorize_jacobian: bool,
    jacobian: pt.TensorVariable | None = None,
) -> tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]:
    r"""
    Build the change in the variables that keeps the system at zero when the parameters move by a small step.

    Differentiating :math:`f(x, \theta) = 0` along a parameter step :math:`\Delta\theta` gives
    :math:`A \, \Delta x = -B \, \Delta\theta`, where :math:`A` and :math:`B` are the Jacobians of :math:`f` with
    respect to :math:`x` and :math:`\theta`. :math:`B \, \Delta\theta` is computed as one pushforward, so :math:`B` is
    never formed.

    Parameters
    ----------
    system : TensorVariable
        Flat vector of model residuals.
    variables : list of TensorVariable
        Variables of the system, packed into ``x``.
    parameters : list of TensorVariable
        Parameters of the system, packed into ``theta``.
    vectorize_jacobian : bool
        Build the Jacobians here with one batched backward pass, as
        :func:`~cge_modeling.compile.pytensor_tools.make_jacobian` does with ``vectorize``.
    jacobian : TensorVariable, optional
        The Jacobian of ``system`` with respect to ``variables``, if already built. Built here otherwise.

    Returns
    -------
    x : TensorVariable
        Flat vector of the variables.
    theta : TensorVariable
        Flat vector of the parameters.
    theta_step : TensorVariable
        Flat vector of the parameter step.
    x_step : TensorVariable
        The solution :math:`\Delta x`, as a graph of ``x``, ``theta`` and ``theta_step``.
    """
    variable_shapes = [variable.type.shape for variable in variables]
    parameter_shapes = [parameter.type.shape for parameter in parameters]
    n_variables = sum(int(np.prod(shape)) for shape in variable_shapes)
    n_parameters = sum(int(np.prod(shape)) for shape in parameter_shapes)

    x = pt.vector("x", shape=(n_variables,))
    theta = pt.vector("theta", shape=(n_parameters,))
    theta_step = pt.vector("theta_step", shape=(n_parameters,))

    try:
        [b_times_step] = pushforward(
            [system], parameters, pt.unpack(theta_step, packed_shapes=parameter_shapes)
        )
    except NotImplementedError:
        _log.warning(
            "An Op in the model has no pushforward, so the path step forms the full Jacobian with respect to the "
            "parameters. This is slow and memory intensive for large models."
        )
        b_times_step = make_jacobian(system, parameters, vectorize=vectorize_jacobian) @ theta_step

    A = (
        make_jacobian(system, variables, vectorize=vectorize_jacobian)
        if jacobian is None
        else jacobian
    )
    x_step = -pt.linalg.solve(A, b_times_step, assume_a="gen", check_finite=False)

    replacements = {
        **dict(zip(variables, pt.unpack(x, packed_shapes=variable_shapes), strict=True)),
        **dict(zip(parameters, pt.unpack(theta, packed_shapes=parameter_shapes), strict=True)),
    }
    x_step = cast(pt.TensorVariable, graph_replace(x_step, replacements, strict=False))
    x_step.name = "x_step"

    return x, theta, theta_step, x_step


def _no_statistics(x: np.ndarray, theta: np.ndarray) -> dict[str, float]:
    return {}


def follow_path_with_python_loop(
    x: np.ndarray,
    theta: np.ndarray,
    theta_final: np.ndarray,
    n_steps: int,
    *,
    f_tangent: Tangent,
    step_method: StepMethod,
    shapes: Sequence[tuple[int, ...]],
    progress: ToggleableProgress,
    step_statistics: Callable[[np.ndarray, np.ndarray], dict[str, float]],
) -> list[np.ndarray]:
    """
    Follow the equilibrium from ``(x, theta)`` to ``theta_final`` in ``n_steps`` equal parameter steps.

    Returns
    -------
    trajectory : list of ndarray
        The path of each variable, then each parameter, with shape ``(n_steps + 1, *shape)``.
    """
    take_step = STEP_FUNCTIONS[step_method]
    theta_initial = theta
    theta_change = theta_final - theta_initial
    theta_step = theta_change / n_steps

    trajectory = np.empty((n_steps + 1, x.shape[0] + theta.shape[0]))
    trajectory[0] = np.concatenate([x, theta])

    if not progress.is_enabled:
        step_statistics = _no_statistics

    with progress:
        task = progress.add_task(
            f"Path ({step_method})", total=n_steps, **step_statistics(x, theta)
        )
        for t in range(1, n_steps + 1):
            x = take_step(f_tangent, x, theta, theta_step)
            theta = theta_initial + theta_change * (t / n_steps)
            trajectory[t] = np.concatenate([x, theta])
            progress.update(task, advance=1, **step_statistics(x, theta))

        progress.update(task, completed=n_steps, refresh=True)

    return split_trajectory(trajectory, shapes)


def symbolic_path(
    system: pt.TensorVariable,
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    step_method: StepMethod,
    vectorize_jacobian: bool,
    jacobian: pt.TensorVariable | None = None,
) -> tuple[list[pt.TensorVariable], list[pt.TensorVariable]]:
    """
    Build the path from an equilibrium to new parameters as one scan graph.

    The path splits the move from the initial parameters to ``theta_final`` into ``n_steps`` equal steps. Each step
    moves the variables by :func:`euler_step` or :func:`rk4_step`, using the tangent from :func:`path_tangent`.

    Parameters
    ----------
    system : TensorVariable
        Flat vector of model residuals.
    variables : list of TensorVariable
        Variables of the system.
    parameters : list of TensorVariable
        Parameters of the system.
    step_method : {"euler", "rk4"}
        How each step moves the variables.
    vectorize_jacobian : bool
        Passed to :func:`path_tangent`.
    jacobian : TensorVariable, optional
        The Jacobian of ``system`` with respect to ``variables``, if already built.

    Returns
    -------
    inputs : list of TensorVariable
        The flat initial variables ``x``, the flat initial parameters ``theta``, ``theta_final`` and ``n_steps``.
    trajectory : list of TensorVariable
        The path of each variable, then each parameter, with shape ``(n_steps + 1, *shape)``.
    """
    x, theta, theta_step, x_step = path_tangent(
        system, variables, parameters, vectorize_jacobian=vectorize_jacobian, jacobian=jacobian
    )
    f_tangent = OpFromGraph(inputs=[x, theta, theta_step], outputs=[x_step], inline=True)
    take_step = STEP_FUNCTIONS[step_method]

    x0 = x.clone(name="x0")
    theta0 = theta.clone(name="theta0")
    theta_final = theta.clone(name="theta_final")
    n_steps = pt.scalar("n_steps", dtype="int32")

    def step(t, x_previous, theta_previous, theta_initial, theta_final, n_steps):
        theta_change = theta_final - theta_initial
        x_next = take_step(f_tangent, x_previous, theta_previous, theta_change / n_steps)
        theta_next = theta_initial + theta_change * (t / n_steps)
        return x_next, theta_next

    x_path, theta_path = pytensor.scan(
        step,
        sequences=[pt.arange(1, n_steps + 1)],
        outputs_info=[x0, theta0],
        non_sequences=[theta0, theta_final, n_steps],
        strict=True,
        return_updates=False,
    )

    trajectory = pt.concatenate(
        [pt.concatenate([x0, theta0])[None], pt.concatenate([x_path, theta_path], axis=1)],
        axis=0,
    )
    shapes = [obj.type.shape for obj in variables + parameters]

    return [x0, theta0, theta_final, n_steps], split_trajectory(trajectory, shapes)
