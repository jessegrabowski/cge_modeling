import pytensor
import pytensor.tensor as pt


def test_prod_to_no_zero_prod():
    x = pt.dvector("x")
    z = pt.prod(x)

    assert not z.owner.op.no_zeros_in_input
    f = pytensor.function([x], z)
    assert any([isinstance(node.op, pt.math.Prod) for node in f.maker.fgraph.toposort()])

    for node in f.maker.fgraph.toposort():
        if isinstance(node.op, pt.math.Prod):
            assert node.op.no_zeros_in_input
            break
