import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from cge_modeling.compile.pytensor_tools import compile_with_packed_inputs, pushforward


def test_prod_to_no_zero_prod_rewrite():
    import pytensor
    import pytensor.tensor as pt

    x = pt.dvector("x")
    z = pt.prod(x)

    assert not z.owner.op.no_zeros_in_input
    f = pytensor.function([x], z)
    assert any([isinstance(node.op, pt.math.Prod) for node in f.maker.fgraph.toposort()])

    for node in f.maker.fgraph.toposort():
        if isinstance(node.op, pt.math.Prod):
            assert node.op.no_zeros_in_input
            break


def test_compile_with_packed_inputs_unpacks_groups_in_row_major_order():
    scalar = pt.dscalar("scalar")
    matrix = pt.dmatrix("matrix", shape=(2, 2))
    weight = pt.dscalar("weight")
    n = pt.iscalar("n")

    f = compile_with_packed_inputs(
        [("x", [scalar, matrix]), weight, n],
        [scalar + n, weight * matrix[0, 1]],
        mode="FAST_RUN",
    )
    scalar_value, weighted_entry = f(np.array([10.0, 1.0, 2.0, 3.0, 4.0]), 3.0, 5)

    assert [variable.name for variable in f.maker.fgraph.inputs] == ["x", "weight", "n"]
    assert scalar_value == 15.0
    assert weighted_entry == 6.0


@pytest.mark.parametrize(
    "build_outputs",
    [
        lambda a, b: [a + b],
        lambda a, b: [a - b],
        lambda a, b: [b - (a + b) * a],
        lambda a, b: [a * b + b, a - b],
        lambda a, b: [pt.prod(a * b, keepdims=True) + a],
    ],
    ids=["add", "sub", "nested", "two_outputs", "product_reduction"],
)
def test_pushforward_matches_the_jacobian_when_inputs_share_a_cotangent(build_outputs):
    a, b = pt.dvector("a", shape=(2,)), pt.dvector("b", shape=(2,))
    tangent_a, tangent_b = pt.dvector("tangent_a", shape=(2,)), pt.dvector("tangent_b", shape=(2,))
    outputs = build_outputs(a, b)
    jvp = pushforward(outputs, [a, b], [tangent_a, tangent_b])
    expected = [
        pt.jacobian(output, a) @ tangent_a + pt.jacobian(output, b) @ tangent_b
        for output in outputs
    ]
    f = pytensor.function([a, b, tangent_a, tangent_b], [*jvp, *expected], on_unused_input="ignore")

    values = f(
        np.array([1.5, -2.0]), np.array([0.5, 3.0]), np.array([1.0, 2.0]), np.array([-1.0, 4.0])
    )

    for computed, reference in zip(values[: len(outputs)], values[len(outputs) :], strict=True):
        np.testing.assert_allclose(computed, reference)
