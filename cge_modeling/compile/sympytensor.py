import logging

from functools import partial
from typing import Literal, cast

import pytensor
import sympy as sp

from pytensor import tensor as pt
from pytensor.compile import Function
from sympytensor import as_tensor

from cge_modeling import CGEModel
from cge_modeling.base.function_wrappers import return_array_from_jax_wrapper
from cge_modeling.compile.constants import CompiledFunctions
from cge_modeling.compile.euler import (
    pytensor_euler_function_with_python_loop,
    symbolic_euler_approximation,
)
from cge_modeling.compile.pytensor import (
    pytensor_euler_step,
    validate_pytensor_parsing_result,
    vars_and_params_to_pytensor,
)
from cge_modeling.compile.pytensor_tools import (
    make_unpacked_to_packed_map,
    rewrite_pregrad,
    unpacked_graph_to_packed_graph,
)
from cge_modeling.compile.sympy import make_sympy_gradient, make_sympy_hessp, make_sympy_jacobian
from cge_modeling.tools.sympy_tools import sparse_jacobian

_log = logging.getLogger(__name__)


def sympy_cge_primitives_to_pytensor(
    cge_model: CGEModel,
) -> tuple[
    pt.TensorVariable, list[pt.TensorVariable], list[pt.TensorVariable], tuple[dict, dict, dict]
]:
    variables, parameters, cache = vars_and_params_to_pytensor(cge_model)
    unpacked_cache = {}

    # This will be a flat list of equations with a unique variable for each unpacked object
    # We want to rewrite this graph so that it will accept the packed array inputs (held in the cache dictionary)
    pytensor_equations = [
        as_tensor(eq, cache=unpacked_cache) for eq in cge_model.unpacked_equation_symbols
    ]
    n_eq = len(pytensor_equations)

    # We need to replace the indexed variables with the unpacked variables
    unpacked_to_indexed_dict = make_unpacked_to_packed_map(cge_model, cache, unpacked_cache)
    pytensor_equations = unpacked_graph_to_packed_graph(
        pytensor_equations, unpacked_to_indexed_dict
    )

    flat_equations = pt.stack(pytensor_equations)
    flat_equations = pt.specify_shape(flat_equations, (n_eq,))
    flat_equations.name = "equations"

    rewrite_pregrad(flat_equations)

    return flat_equations, variables, parameters, (cache, unpacked_cache, unpacked_to_indexed_dict)


def make_sympytensor_jacobian(
    cge_model: CGEModel,
    cache,
    unpacked_cache,
    equations=None,
    wrt: Literal["variables", "parameters"] = "variables",
    sparse=False,
) -> pt.TensorVariable:
    if equations is None:
        equations = cge_model.unpacked_equation_symbols

    variables = cge_model.unpacked_variable_symbols
    parameters = cge_model.unpacked_parameter_symbols

    wrt_symbols = variables if wrt == "variables" else parameters
    jac = (
        sp.Matrix(equations).jacobian(wrt_symbols)
        if not sparse
        else sparse_jacobian(equations, wrt_symbols)
    )

    unpacked_jac = as_tensor(jac, cache=unpacked_cache)
    packed_jac = unpacked_graph_to_packed_graph(unpacked_jac, cge_model, cache, unpacked_cache)

    return packed_jac


def make_sympytensor_jvp(
    cge_model: CGEModel,
    cache,
    unpacked_cache,
    grad,
    wrt: Literal["variables", "parameters"] = "variables",
    sparse=False,
) -> tuple[pt.TensorVariable, list[pt.TensorVariable]]:
    """
    Compute the jacobian-vector product between a system of equations and a vector of variables.

    Parameters
    ----------
    cge_model: CGEModel
        The CGE model
    cache: dict

    unpacked_cache: dict

    grad: pt.TensorVariable
        The gradient tensor of the loss function.
    wrt: Literal["variables", "parameters"], optional
        Specifies whether to compute the JVP with respect to variables or parameters. Default is "variables".
    sparse: bool, optional
        If True, use sparse matrices for the computation. Default is False.

    Returns
    -------
    jvp: pytensor.tensor.TensorVariable
        The jacobian-vector product of the original loss function
    p: list of pytensor.tensor.TensorVariable
        The symbolic variables representing each component of ``p``, the point vector where the JVP is evaluated.
    """
    variables = cge_model.unpacked_variable_symbols
    parameters = cge_model.unpacked_parameter_symbols

    wrt_symbols = variables if wrt == "variables" else parameters
    jvp, p_vars = make_sympy_hessp(grad, wrt_symbols)

    unpacked_jvp = as_tensor(jvp, cache=unpacked_cache)
    packed_jvp = unpacked_graph_to_packed_graph(unpacked_jvp, cge_model, cache, unpacked_cache)

    return packed_jvp, p_vars


def compile_sympytensor_jacobian(
    sp_equations: sp.Matrix | list[sp.Expr],
    sp_wrt: list[sp.Symbol],
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    cache: dict,
    unpacked_to_indexed_dict: dict,
    sparse: bool = False,
    mode: str | None = None,
) -> tuple[sp.Matrix, pt.TensorVariable, Function]:
    f_make_jac = make_sympy_jacobian if not sparse else sparse_jacobian
    sp_jac = f_make_jac(sp_equations, sp_wrt)
    unpacked_jac = as_tensor(sp_jac, cache=cache)

    jac = unpacked_graph_to_packed_graph(unpacked_jac, unpacked_to_indexed_dict)
    rewrite_pregrad(jac)

    f_jac = pytensor.function(
        inputs=[*variables, *parameters], outputs=jac, on_unused_input="ignore", mode=mode
    )
    f_jac.trust_inputs = True

    return sp_jac, jac, f_jac


def compile_sympytensor_error_function(
    sp_equations: list[sp.Expr],
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    cache: dict,
    unpacked_to_indexed_dict: dict,
    mode: str | None = None,
) -> tuple[sp.Expr, pt.TensorVariable, Function]:
    sp_squared_loss = cast(sp.Expr, 0.5 * sum(eq**2 for eq in sp_equations))
    squared_loss = as_tensor(sp_squared_loss, cache=cache)
    squared_loss = unpacked_graph_to_packed_graph(squared_loss, unpacked_to_indexed_dict)
    squared_loss.name = "squared_loss"
    rewrite_pregrad(squared_loss)

    f_loss = pytensor.function(inputs=[*variables, *parameters], outputs=squared_loss, mode=mode)
    f_loss.trust_inputs = True

    return sp_squared_loss, squared_loss, f_loss


def compile_sympytensor_gradient_function(
    sp_loss: sp.Expr,
    sp_variables: list[sp.Symbol],
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    cache: dict,
    unpacked_to_indexed_dict: dict,
    mode: str | None = None,
) -> tuple[sp.Matrix, pt.TensorVariable, Function]:
    sp_grad = make_sympy_gradient(sp_loss, sp_variables)
    grad = as_tensor(sp_grad, cache=cache)
    grad = unpacked_graph_to_packed_graph(grad, unpacked_to_indexed_dict).ravel()
    grad.name = "gradient"

    rewrite_pregrad(grad)

    f_grad = pytensor.function(
        inputs=[*variables, *parameters], outputs=grad, mode=mode, on_unused_input="ignore"
    )
    f_grad.trust_inputs = True

    return sp_grad, grad, f_grad


def compile_sympytensor_hess_function(
    sp_grad: sp.Matrix,
    sp_wrt: list[sp.Symbol],
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    cache: dict,
    unpacked_to_indexed_dict: dict,
    sparse: bool = False,
    mode: str | None = None,
) -> tuple[sp.Expr, pt.TensorVariable, Function]:
    f_make_hess = make_sympy_jacobian if not sparse else sparse_jacobian
    sp_hess = f_make_hess(sp_grad, sp_wrt)
    unpacked_hess = as_tensor(sp_hess, cache=cache)
    hess = unpacked_graph_to_packed_graph(unpacked_hess, unpacked_to_indexed_dict)
    rewrite_pregrad(hess)

    f_hess = pytensor.function(
        inputs=[*variables, *parameters], outputs=hess, mode=mode, on_unused_input="ignore"
    )
    f_hess.trust_inputs = True

    return sp_hess, hess, f_hess


def compile_sympytensor_hessp_function(
    sp_grad: sp.Matrix,
    sp_wrt: list[sp.Symbol],
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    cache: dict,
    unpacked_to_indexed_dict: dict,
    mode: str | None = None,
) -> tuple[sp.Matrix, pt.TensorVariable, Function]:
    sp_hessp, p_vars = make_sympy_hessp(sp_grad, sp_wrt)
    unpacked_hessp = as_tensor(sp_hessp, cache=cache)
    hessp = unpacked_graph_to_packed_graph(unpacked_hessp, unpacked_to_indexed_dict)
    hessp.name = "hessp"

    p_vars_pt = next(x for x in cache.values() if x.name == "hess_point")
    point_vars = [var.type(name=f"{var.name}_point") for var in variables]
    p = pt.concatenate([v.ravel() for v in point_vars], axis=0)

    hessp = pytensor.clone_replace(hessp, {p_vars_pt: p})

    f_hessp = pytensor.function(
        inputs=[*variables, *parameters, *point_vars],
        outputs=hessp.ravel(),
        mode=mode,
        on_unused_input="ignore",
    )
    f_hessp.trust_inputs = True

    return sp_hessp, hessp, f_hessp


def compile_sympytensor_cge_functions(
    cge_model: CGEModel,
    functions_to_compile: list[CompiledFunctions],
    mode: str | None = None,
    use_scan_for_euler: bool = False,
    use_sparse_matrices: bool = False,
    *args,
    **kwargs,
) -> tuple[Function | None, ...]:
    system, variables, parameters, caches = sympy_cge_primitives_to_pytensor(cge_model)
    sp_equations, sp_variables, _sp_parameteres = (
        cge_model.unpacked_equation_symbols,
        cge_model.unpacked_variable_symbols,
        cge_model.unpacked_parameter_symbols,
    )

    cache, unpacked_cache, unpacked_to_indexed_dict = caches
    validate_pytensor_parsing_result(system, variables, parameters, cache)

    # Always compile the system -- used to check residuals
    f_system = pytensor.function(
        inputs=[*variables, *parameters], outputs=system, mode=mode, on_unused_input="raise"
    )
    f_system.trust_inputs = True
    if mode == "JAX":
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
        sp_jac, jac, f_jac = compile_sympytensor_jacobian(
            sp_equations,
            sp_variables,
            variables,
            parameters,
            unpacked_cache,
            unpacked_to_indexed_dict,
            sparse=use_sparse_matrices,
            mode=mode,
        )

    if "minimize" in functions_to_compile:
        sp_squared_loss, squared_loss, f_resid = compile_sympytensor_error_function(
            sp_equations, variables, parameters, unpacked_cache, unpacked_to_indexed_dict, mode=mode
        )

        sp_grad, grad, f_grad = compile_sympytensor_gradient_function(
            sp_squared_loss,
            sp_variables,
            variables,
            parameters,
            unpacked_cache,
            unpacked_to_indexed_dict,
            mode=mode,
        )

        sp_hess, hess, f_hess = compile_sympytensor_hess_function(
            sp_grad,
            sp_variables,
            variables,
            parameters,
            unpacked_cache,
            unpacked_to_indexed_dict,
            sparse=use_sparse_matrices,
            mode=mode,
        )

        sp_hessp, hessp, f_hessp = compile_sympytensor_hessp_function(
            sp_grad,
            sp_variables,
            variables,
            parameters,
            unpacked_cache,
            unpacked_to_indexed_dict,
            mode=mode,
        )

    if "euler" in functions_to_compile:
        if not use_scan_for_euler:
            inputs, outputs = pytensor_euler_step(
                system, variables, parameters, jacobian=None, grad=None
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
                system, variables, parameters, jacobian=None
            )
            f_euler = pytensor.function(
                inputs=[*variables, *parameters, *theta_final], outputs=euler_output, mode=mode
            )
            f_euler.trust_inputs = True

    return f_system, f_jac, f_resid, f_grad, f_hess, f_hessp, f_euler
