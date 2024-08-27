import numpy as np
import pytensor
import pytensor.tensor as pt

from cge_modeling.compile.euler import symbolic_euler_approximation
from cge_modeling.compile.pytensor_tools import make_jacobian


def test_euler_approximation_1d():
    x = pt.dscalar("x")
    y = pt.dscalar("y")
    eq = pt.atleast_1d(y - pt.cos(x) + 1)

    x0_final, n_steps, result = symbolic_euler_approximation(eq, variables=[y], parameters=[x])
    f = pytensor.function([x, y, x0_final, n_steps], result)
    y_values, x_values = f(0, 0, 10.0, 10_000)
    true = -np.cos(np.array([10])) + 1
    np.testing.assert_allclose(-true, y_values[-1], atol=1e-3)


def test_euler_approximation_2d():
    variables = v1, v2 = [pt.dscalar(name) for name in ["v1", "v2"]]
    parameters = v3 = pt.dscalar("v3")
    inputs = [*variables, parameters]

    equations = pt.stack([v1**2 * v3 - 1, v1 + v2 - 2])

    theta_final, n_steps, result = symbolic_euler_approximation(
        equations, variables=variables, parameters=[parameters]
    )

    f = pytensor.function([*inputs, theta_final, n_steps], result)

    def f_analytic(v3):
        v1 = 1 / np.sqrt(v3)
        v2 = 2 - v1
        return np.array([v1, v2])

    initial_point = [1.0, 1.0]
    v3_initial = 1.0
    v3_final = 2.0

    analytic_solution = np.array(f_analytic(v3_final))
    *x, theta = f(*initial_point, v3_initial, v3_final, 10_000)
    np.testing.assert_allclose(x[0][-1], analytic_solution[0], atol=1e-3)
    np.testing.assert_allclose(x[1][-1], analytic_solution[1], atol=1e-3)


def test_euler_approximation_step():
    pass
