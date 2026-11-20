import logging

from collections.abc import Sequence
from typing import cast

import numpy as np
import pytensor

from pytensor import tensor as pt
from pytensor.compile import Function, Mode
from pytensor.gradient import pullback
from pytensor.graph import node_rewriter, rewrite_graph
from pytensor.graph.basic import Variable as GraphVariable
from pytensor.graph.replace import graph_replace
from pytensor.graph.rewriting.basic import in2out
from pytensor.tensor.math import Prod
from pytensor.tensor.rewriting.basic import register_stabilize

_log = logging.getLogger(__name__)


@register_stabilize
@node_rewriter([Prod])
def prod_to_no_zero_prod(fgraph, node):
    """
    JAX doesn't support gradient computation in the case where there are zeros in the product. We're allowed to promise
    there will never be zeros, which should always be the case for CGE models. This rewrite makes this promse for any
    product Ops that are in the graph.

    Note that this only affects product reduction Ops, it's not the same as a multiplication.
    """
    if isinstance(node.op, Prod) and not node.op.no_zeros_in_input:
        (x,) = node.inputs
        new_op = Prod(dtype=node.op.dtype, acc_dtype=node.op.dtype, no_zeros_in_input=True)
        return [new_op(x)]


def rewrite_pregrad(graph: pt.TensorVariable) -> pt.TensorVariable:
    """Return ``graph`` with the canonicalize and stabilize rewrites applied, which are safe before differentiating."""
    return cast(pt.TensorVariable, rewrite_graph(graph, include=("canonicalize", "stabilize")))


def clone_and_rename(x: pt.TensorVariable, suffix: str = "_next") -> pt.TensorVariable:
    """
    Clone a pytensor tensor and rename it by appending a suffix to its name.

    Cloning variables is useful for creating intermediate "dummy" variables that can be replaced by graph rewrites,
    mostly commonly in vectorization.

    Parameters
    ----------
    x: pytensor.tensor.TensorVariable
        The pytensor variable to clone
    suffix: str
        The suffix to append to the cloned variable's name

    Returns
    -------
    x_new: pytensor.tensor.TensorVariable
        The cloned variable with the new name
    """
    x_new = x.clone()
    x_new.name = f"{x.name}{suffix}"
    return x_new


def flatten_equations(eqs: Sequence[pt.TensorLike]) -> pt.TensorVariable:
    """
    Flatten a list of pytensor vectors, each representing equations in a system of equations, into a single vector.

    Parameters
    ----------
    eqs: list[pytensor.tensor.TensorVariable]
        A list of pytensor vectors, each representing equations in a system of equations

    Returns
    -------
    flat_equations: pytensor.tensor.TensorVariable
        A single pytensor vector representing the equations in the system of equations

    Notes
    -----
    In the context of CGE modeling, each equation in the model can be thought of as a vector (or matrix) of equations,
    with each dimension corresponding to a single variable index. This perspective is useful for efficient vectorized
    computation of each "block" of the economic model. However, when solving for the equilibrium, we need to flatten
    these multidimensional expressions into a single vector to compute the Jacobian of the system. This function
    performs that flattening operation.
    """
    expr_len = 0
    for eq in eqs:
        if isinstance(eq, pt.TensorVariable):
            expr_len += int(np.prod(eq.type.shape))
        else:
            expr_len += int(np.prod(np.atleast_1d(eq).size))

    # expr_len = int(np.sum([np.prod(eq.type.shape) for eq in eqs]))
    flat_expr = pt.concatenate(
        [pt.atleast_1d(pt.cast(eq, pytensor.config.floatX)).ravel() for eq in eqs],
        axis=-1,
    )
    flat_expr = pt.specify_shape(flat_expr, shape=(expr_len,))
    return flat_expr


def flat_tensor_to_ragged_list(tensor, shapes):
    out = []
    cursor = 0

    for shape in shapes:
        s = int(np.prod(shape))
        x = tensor[cursor : cursor + s].reshape(shape)
        out.append(x)
        cursor += s

    return out


def compile_with_packed_inputs(
    inputs: Sequence[pt.TensorVariable | tuple[str, Sequence[pt.TensorVariable]]],
    outputs: pt.TensorVariable | Sequence[pt.TensorVariable],
    mode: str | Mode | None = None,
    on_unused_input: str = "raise",
) -> Function:
    """
    Compile a function whose grouped inputs are each packed into one flat vector.

    Parameters
    ----------
    inputs: sequence of TensorVariable or tuple of (str, sequence of TensorVariable)
        Inputs of the compiled function, in order. A ``(name, parts)`` pair becomes a single flat vector input called
        ``name``, holding every element of ``parts`` raveled in C order and concatenated. A bare TensorVariable is
        passed through as its own input.
    outputs: TensorVariable or sequence of TensorVariable
        Outputs of the compiled function, as graphs of the unpacked parts.
    mode: str or Mode, optional
        PyTensor compile mode.
    on_unused_input: str
        Passed to :func:`pytensor.function`. Defaults to ``"raise"``.

    Returns
    -------
    f: Function
        The compiled function.
    """
    function_inputs = []
    replacements: dict[GraphVariable, GraphVariable] = {}

    for item in inputs:
        if isinstance(item, tuple):
            name, parts = item
            shapes = [part.type.shape for part in parts]
            packed = pt.tensor(
                name,
                shape=(sum(int(np.prod(shape)) for shape in shapes),),
                dtype=pytensor.scalar.upcast(*(part.type.dtype for part in parts)),
            )
            replacements.update(zip(parts, pt.unpack(packed, packed_shapes=shapes), strict=True))
            function_inputs.append(packed)
        else:
            function_inputs.append(item)

    packed_outputs = graph_replace(outputs, replacements, strict=False)

    return pytensor.function(
        inputs=function_inputs,
        outputs=packed_outputs,
        mode=mode,
        on_unused_input=on_unused_input,
    )


def pushforward(
    f: Sequence[pt.TensorVariable],
    wrt: Sequence[pt.TensorVariable],
    tangents: Sequence[pt.TensorVariable],
) -> list[pt.TensorVariable]:
    """
    Compute the Jacobian of ``f`` times ``tangents`` as the gradient of a pullback with respect to its cotangents.

    The pullback of ``f`` is linear in the output cotangents, so differentiating its inner product with ``tangents``
    with respect to those cotangents gives the Jacobian-vector product. Writing the inner product as one scalar sums
    the terms of inputs that share a cotangent. Product reductions in ``f`` are first marked free of zeros, as
    :func:`prod_to_no_zero_prod` does at compile time, because the zero-handling pullback of ``Prod`` has no gradient.

    Parameters
    ----------
    f : sequence of TensorVariable
        Outputs to differentiate.
    wrt : sequence of TensorVariable
        Inputs to differentiate with respect to.
    tangents : sequence of TensorVariable
        One tangent per entry of ``wrt``, with its shape.

    Returns
    -------
    jvp : list of TensorVariable
        The Jacobian-vector product of each output.
    """
    outputs = cast(
        list[pt.TensorVariable],
        rewrite_graph(list(f), include=(), custom_rewrite=in2out(prod_to_no_zero_prod), clone=True),
    )
    cotangents = [output.zeros_like(dtype=pytensor.config.floatX) for output in outputs]
    input_cotangents = cast(
        list[pt.TensorVariable],
        pullback(outputs, list(wrt), cotangents, return_disconnected="zero"),
    )

    inner_product = pt.add(
        *(
            pt.sum(input_cotangent * tangent)
            for input_cotangent, tangent in zip(input_cotangents, tangents, strict=True)
        )
    )
    jvp = pullback(
        inner_product, cotangents, pt.ones_like(inner_product), disconnected_inputs="ignore"
    )
    return cast(list[pt.TensorVariable], list(cast(Sequence[pt.TensorVariable], jvp)))


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

    system = rewrite_pregrad(system)
    column_list = pytensor.gradient.jacobian(system, x)

    jac = pt.concatenate([pt.atleast_2d(x).reshape((n_eq, -1)) for x in column_list], axis=-1)
    jac = pt.specify_shape(jac, shape=(n_eq, n_vars))

    return jac
