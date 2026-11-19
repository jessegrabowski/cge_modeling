import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.compile.builders import OpFromGraph

from cge_modeling.sparsity.prototype import jacobian_sparsity_vjp
from tests.sparsity.conftest import detected_and_numeric_patterns

a = pt.matrix("a")
rhs = pt.vector("rhs")
x = pt.vector("x")
inner = pt.matrix("inner")
wrapped = OpFromGraph([inner], [pt.sin(inner * np.eye(4))])(a)
equations = pt.stack(
    [
        x[0] ** 2 + pt.sin(x[1]),
        x[1] ** 3,
        pt.exp(x[2]) + x[0],
        pt.exp(x[3]) + 3 * x[0],
        x[4] ** 3 + x[1] * x[4],
    ]
)
rosenbrock = pt.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

CASES = {
    "slice": ([a[1:3, ::2]], [a]),
    "join/split": ([pt.concatenate([a, rhs[None]])], [a, rhs]),
    "inner graph": ([wrapped], [a]),
    "nonlinear system": ([equations], [x]),
    "Rosenbrock Hessian": ([pt.grad(rosenbrock, x)], [x]),
}


@pytest.fixture(scope="module")
def patterns():
    rng = np.random.default_rng(13)
    values = [rng.uniform(0.5, 1.5, shape) for shape in ((4, 4), (4,), (8,))]
    with pytensor.config.change_flags(mode="FAST_COMPILE"):
        reference = detected_and_numeric_patterns(CASES, [a, rhs, x], values)
    function = pytensor.function(
        [a, rhs, x],
        [jacobian_sparsity_vjp(outputs, wrt) for outputs, wrt in CASES.values()],
        on_unused_input="ignore",
    )
    return {
        title: (detected, reference[title][1])
        for title, detected in zip(CASES, function(*values), strict=True)
    }


@pytest.mark.parametrize("title", CASES)
def test_advanced_patterns(patterns, title):
    detected, numeric = patterns[title]
    np.testing.assert_array_equal(detected, numeric)
