import logging

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, cast

import pytensor

from pytensor import tensor as pt
from pytensor.compile import Function
from pytensor.graph import rewrite_graph

from cge_modeling import CGEModel
from cge_modeling.compile.constants import CompiledFunctions, StepMethod
from cge_modeling.compile.path import (
    STEP_FUNCTIONS,
    follow_path_with_python_loop,
    path_tangent,
    symbolic_path,
)
from cge_modeling.compile.pytensor_tools import (
    compile_with_packed_inputs,
    flatten_equations,
    make_jacobian,
    rewrite_pregrad,
)
from cge_modeling.compile.sparse import (
    compile_sparse_path_tangent,
    compile_sparse_system_and_jacobian,
    compressed_jacobian,
)

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
    grad = rewrite_pregrad(grad)

    jvp_chunks = cast(list[pt.TensorVariable], pt.grad(pt.sum(grad * p), x))
    jvp = pt.concatenate([chunk.ravel() for chunk in jvp_chunks], axis=-1)
    jvp = pt.specify_shape(jvp, shape=(n_eq,))

    return jvp


def compile_pytensor_jacobian_function(
    system: pt.TensorVariable,
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    vectorize_jacobian: bool,
    mode: str | None = None,
) -> tuple[pt.TensorVariable, Function]:
    jac = make_jacobian(system, variables, vectorize=vectorize_jacobian)
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
    squared_loss = rewrite_pregrad(squared_loss)

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
    grad_parts = cast(list[pt.TensorVariable], pytensor.grad(loss, variables))
    grad = pt.specify_shape(
        pt.concatenate([pt.atleast_1d(part).ravel() for part in grad_parts]), n_variables
    )
    grad = rewrite_pregrad(grad)

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
    # Vectorizing the gradient graph drops the static length of canonicalized Split pieces, so the result fails at
    # runtime (https://github.com/pymc-devs/pytensor/issues/2732). The Hessian keeps the scan over rows until then.
    hess = make_jacobian(grad, variables, vectorize=False)
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


def compile_pytensor_cge_functions(
    cge_model: CGEModel,
    functions_to_compile: Sequence[CompiledFunctions],
    step_method: StepMethod,
    vectorize_jacobian: bool,
    mode: str | None = None,
    use_scan_path: bool = False,
    use_sparse_matrices: bool = False,
) -> tuple[Callable[..., Any] | None, ...]:
    if step_method not in STEP_FUNCTIONS:
        raise ValueError(
            f"Unknown step_method {step_method!r}. Use one of {sorted(STEP_FUNCTIONS)}."
        )
    if use_sparse_matrices and use_scan_path and "path" in functions_to_compile:
        raise ValueError(
            "use_sparse_matrices=True requires the Python-loop path; set use_scan_path=False."
        )

    system, variables, parameters = xtensor_cge_primitives_to_pytensor(cge_model)
    if use_sparse_matrices:
        compressed, layout = compressed_jacobian(system, variables, parameters)

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
    f_path: Callable[..., Any] | None = None
    f_sparse_system_and_jac = None

    # The Jacobian is reused by the path if both root and path are requested
    jac = None

    if "root" in functions_to_compile and use_sparse_matrices:
        f_sparse_system_and_jac = compile_sparse_system_and_jacobian(
            system, variables, parameters, compressed=compressed, layout=layout, mode=mode
        )
    elif "root" in functions_to_compile:
        jac, f_jac = compile_pytensor_jacobian_function(
            system, variables, parameters, vectorize_jacobian=vectorize_jacobian, mode=mode
        )

    if "minimize" in functions_to_compile:
        squared_loss, f_resid = compile_pytensor_error_function(
            system, variables, parameters, mode=mode
        )
        grad, f_grad = compile_pytensor_gradient_function(
            squared_loss, variables, parameters, cge_model.n_variables, mode=mode
        )
        _hess, f_hess = compile_pytensor_hess_function(grad, variables, parameters, mode=mode)
        _hessp, f_hessp = compile_pytensor_hessp_function(grad, variables, parameters, mode=mode)

    if "path" in functions_to_compile and use_scan_path:
        path_inputs, trajectory = symbolic_path(
            system,
            variables,
            parameters,
            step_method=step_method,
            vectorize_jacobian=vectorize_jacobian,
            jacobian=jac,
        )
        f_path = pytensor.function(path_inputs, trajectory, mode=mode)
    elif "path" in functions_to_compile:
        if use_sparse_matrices:
            f_tangent = compile_sparse_path_tangent(
                system, variables, parameters, compressed=compressed, layout=layout, mode=mode
            )
        else:
            *tangent_inputs, x_step = path_tangent(
                system, variables, parameters, vectorize_jacobian=vectorize_jacobian, jacobian=jac
            )
            f_tangent = pytensor.function(tangent_inputs, x_step, mode=mode)

        f_path = partial(
            follow_path_with_python_loop,
            f_tangent=f_tangent,
            step_method=step_method,
            shapes=[obj.type.shape for obj in variables + parameters],
        )

    return f_system, f_jac, f_resid, f_grad, f_hess, f_hessp, f_path, f_sparse_system_and_jac
