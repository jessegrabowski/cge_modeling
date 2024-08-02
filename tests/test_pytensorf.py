import numpy as np
import pytensor
import pytensor.tensor as pt

from cge_modeling.pytensorf.compile import euler_approximation
from cge_modeling.tools.pytensor_tools import make_jacobian


def test_euler_approximation_1d():
    x = pt.dscalar("x")
    y = pt.dscalar("y")
    eq = pt.atleast_1d(y - pt.cos(x) + 1)
    n_steps = pt.iscalar("n_steps")

    A = make_jacobian(eq, [y])
    A_inv = 1 / A
    B = make_jacobian(eq, [x])

    x0_final, result = euler_approximation(A_inv, B, variables=[y], parameters=[x], n_steps=n_steps)
    f = pytensor.function([x, y, x0_final, n_steps], result)
    y_values, x_values = f(0, 0, np.array([10.0]), 10_000)
    true = -np.cos(np.array([10])) + 1
    np.testing.assert_allclose(-true, y_values[-1], atol=1e-3)


def test_euler_approximation_2d():
    variables = v1, v2 = [pt.dscalar(name) for name in ["v1", "v2"]]
    parameters = v3 = pt.dscalar("v3")
    n_steps = pt.iscalar("n_steps")
    inputs = [*variables, parameters]

    equations = pt.stack([v1**2 * v3 - 1, v1 + v2 - 2])
    A = make_jacobian(equations, variables)
    B = make_jacobian(equations, [parameters])

    theta_final, result = euler_approximation(
        A, B, variables=variables, parameters=[parameters], n_steps=n_steps
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
    *x, theta = f(*initial_point, v3_initial, np.array([v3_final]), 10_000)
    np.testing.assert_allclose(x[0][-1], analytic_solution[0], atol=1e-3)
    np.testing.assert_allclose(x[1][-1], analytic_solution[1], atol=1e-3)


def test_euler_approximation_step():
    pass
