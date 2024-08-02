from pytensor.tensor.math import Prod
from pytensor.tensor.rewriting.basic import node_rewriter, register_stabilize


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
        print("hi :)")
        (x,) = node.inputs
        new_op = Prod(dtype=node.op.dtype, acc_dtype=node.op.dtype, no_zeros_in_input=True)
        return [new_op(x)]
