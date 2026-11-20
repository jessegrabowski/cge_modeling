import logging

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, cast

import pytensor

from pytensor import tensor as pt
from pytensor.compile import Function
from pytensor.graph import rewrite_graph

from cge_modeling import CGEModel
from cge_modeling.compile.constants import CompiledFunctions
from cge_modeling.compile.euler import (
    _make_euler_matrices,
    pytensor_euler_function_with_python_loop,
    symbolic_euler_approximation,
)
from cge_modeling.compile.pytensor_tools import (
    compile_with_packed_inputs,
    flat_tensor_to_ragged_list,
    flatten_equations,
    make_jacobian,
    pushforward,
    rewrite_pregrad,
)
from cge_modeling.tools.utils import at_least_list

_log = logging.getLogger(__name__)


def xtensor_cge_primitives_to_pytensor(
    cge_model: CGEModel,
) -> tuple[pt.TensorVariable, list[pt.TensorVariable], list[pt.TensorVariable]]:
    """
    Collect the system and its inputs from a model built in a ``with`` block.

    Returns
    -------
    system: TensorVariable
        Every equation's residual, flattened in registration order and lowered from xtensor to tensor operations.
    variables: list of TensorVariable
        The tensor behind each model variable, in ``cge_model.variables`` order.
    parameters: list of TensorVariable
        The tensor behind each model parameter, in ``cge_model.parameters`` order.
    """
    variables = [cge_model[obj.name].owner.inputs[0] for obj in cge_model.variables]
    parameters = [cge_model[obj.name].owner.inputs[0] for obj in cge_model.parameters]

    residuals = [equation.residual.values for equation in cge_model._xtensor_equations.values()]
    system = cast(
        pt.TensorVariable, rewrite_graph(flatten_equations(residuals), include=("lower_xtensor",))
    )
    system.name = "equations"

    return system, variables, parameters


def make_jvp(
    grad: pt.TensorVariable, x: list[pt.TensorVariable], p: pt.TensorVariable
) -> pt.TensorVariable:
    """
    Compute the jacobian-vector product between a system of equations and a vector of variables.

    Parameters
    ----------
    grad: pytensor.tensor.TensorVariable
        Vector of partial derivatives of a loss function with respect to input variables
    x: list[pytensor.tensor.TensorVariable]
        A list of variables
    p: pytensor.tensor.TensorVariable
        Flat vector where the JVP is evaluated, ordered like the raveled and concatenated ``x``.

    Returns
    -------
    jvp: pytensor.tensor.TensorVariable
        The jacobian-vector product of the original loss function
    """
    n_eq = grad.type.shape[0]
    rewrite_pregrad(grad)

    jvp_chunks = pt.grad(pt.sum(grad * p), x)
    jvp = pt.concatenate([x.ravel() for x in jvp_chunks], axis=-1)
    jvp = pt.specify_shape(jvp, shape=(n_eq,))

    return jvp


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
    f_jac = compile_with_packed_inputs(
        [("x", variables), ("theta", parameters)], jac, mode=mode, on_unused_input="ignore"
    )

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

    f_loss = compile_with_packed_inputs(
        [("x", variables), ("theta", parameters)], squared_loss, mode=mode
    )

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

    f_grad = compile_with_packed_inputs(
        [("x", variables), ("theta", parameters)], grad, mode=mode, on_unused_input="ignore"
    )

    return grad, f_grad


def compile_pytensor_hess_function(
    grad: pt.TensorVariable,
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    mode: str | None = None,
) -> tuple[pt.TensorVariable, Function]:
    hess = make_jacobian(grad, variables)
    hess.name = "hessian"

    f_hess = compile_with_packed_inputs(
        [("x", variables), ("theta", parameters)], hess, mode=mode, on_unused_input="ignore"
    )

    return hess, f_hess


def compile_pytensor_hessp_function(
    grad: pt.TensorVariable,
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    mode: str | None = None,
) -> tuple[pt.TensorVariable, Function]:
    p = pt.tensor("p", shape=grad.type.shape, dtype=grad.type.dtype)
    hessp = make_jvp(grad, variables, p)
    hessp.name = "hessp"

    f_hessp = compile_with_packed_inputs(
        [("x", variables), p, ("theta", parameters)], hessp, mode=mode
    )

    return hessp, f_hessp


def pytensor_euler_step(
    system, variables, parameters, jacobian=None, grad=None, use_rk4=False
) -> tuple[list, list[pt.TensorVariable]]:
    if use_rk4:
        raise NotImplementedError("RK4 not yet implemented for Pytensor")

    A, B = _make_euler_matrices(system, variables, parameters, jacobian)

    x_list = at_least_list(variables)
    x_shapes = [x.type.shape for x in x_list]

    theta_list = at_least_list(parameters)

    n_steps = pt.iscalar("n_steps")
    theta_final = [x.clone(name=f"{x.name}_final") for x in theta_list]
    theta0 = [x.clone(name=f"{x.name}_initial") for x in theta_list]

    step_size = [(x - y) / n_steps for x, y in zip(theta_final, theta0)]

    try:
        [Bv] = pushforward([system], parameters, step_size)
    except NotImplementedError:
        _log.warning(
            "Tried to compute jvp of model with respect to parameters but failed; an Op in your model does not "
            "have an implemented pushforward. Falling back to direct computation via jacobian. This is slow and "
            "memory intensive for large models."
        )
        v = flatten_equations(step_size)
        Bv = B @ v

    step = -pt.linalg.solve(A, Bv)
    step.name = "euler_step"

    delta_x = flat_tensor_to_ragged_list(step, x_shapes)
    x_next = [x + dx for x, dx in zip(x_list, delta_x)]

    theta_next = [theta + dtheta for theta, dtheta in zip(theta_list, step_size)]

    inputs = [
        ("x", x_list),
        ("theta", theta_list),
        ("theta_initial", theta0),
        ("theta_final", theta_final),
        n_steps,
    ]
    outputs = [flatten_equations(x_next), flatten_equations(theta_next)]

    return inputs, outputs


def compile_pytensor_cge_functions(
    cge_model: CGEModel,
    functions_to_compile: Sequence[CompiledFunctions],
    mode: str | None = None,
    use_scan_euler: bool = False,
    use_rk4: bool = False,
) -> tuple[Callable[..., Any] | None, ...]:
    system, variables, parameters = xtensor_cge_primitives_to_pytensor(cge_model)

    # Always compile the system -- used to check residuals
    f_system = compile_with_packed_inputs(
        [("x", variables), ("theta", parameters)], system, mode=mode
    )

    # Optional functions
    f_jac = None
    f_resid = None
    f_grad = None
    f_hess = None
    f_hessp = None
    f_euler: Callable[..., Any] | None = None

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
        _hess, f_hess = compile_pytensor_hess_function(grad, variables, parameters, mode=mode)
        _hessp, f_hessp = compile_pytensor_hessp_function(grad, variables, parameters, mode=mode)

    if "euler" in functions_to_compile:
        if not use_scan_euler:
            inputs, outputs = pytensor_euler_step(
                system, variables, parameters, jacobian=jac, grad=grad, use_rk4=use_rk4
            )

            f_step = compile_with_packed_inputs(inputs, outputs, mode=mode)

            f_euler = partial(
                pytensor_euler_function_with_python_loop,
                cge_model=cge_model,
                f_step=f_step,
                f_system=f_system,
                f_grad=f_grad,
            )
        else:
            theta_final, n_steps, euler_output = symbolic_euler_approximation(
                system, variables, parameters, jacobian=jac, use_rk4=use_rk4
            )
            f_euler = compile_with_packed_inputs(
                [("x", variables), ("theta", parameters), theta_final, n_steps],
                euler_output,
                mode=mode,
            )

    return f_system, f_jac, f_resid, f_grad, f_hess, f_hessp, f_euler
