import logging

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, cast

import pytensor

from pytensor import tensor as pt
from pytensor.compile import Function
from pytensor.compile.builders import OpFromGraph
from pytensor.graph import rewrite_graph
from pytensor.graph.basic import explicit_graph_inputs

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
    make_printer_cache,
    object_to_pytensor,
    rewrite_pregrad,
)
from cge_modeling.tools.parsing import normalize_eq
from cge_modeling.tools.utils import at_least_list

_log = logging.getLogger(__name__)


def vars_and_params_to_pytensor(
    cge_model: CGEModel,
) -> tuple[list[pt.TensorVariable], list[pt.TensorVariable], dict]:
    coords = cge_model.coords

    variables = [object_to_pytensor(var, coords) for var in cge_model.variables]
    parameters = [object_to_pytensor(param, coords) for param in cge_model.parameters]

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
    *args,
    **kwargs,
) -> tuple[Callable[..., Any] | None, ...]:
    if cge_model._model_objects:
        system, variables, parameters = xtensor_cge_primitives_to_pytensor(cge_model)
    else:
        system, variables, parameters, (cache, _unpacked_cache) = cge_primitives_to_pytensor(
            cge_model
        )
        validate_pytensor_parsing_result(system, variables, parameters, cache)

    _log.info(f"Compiling model to Pytensor using {mode} mode")

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
