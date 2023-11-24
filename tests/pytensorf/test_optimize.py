import numpy as np
import pytensor
import pytensor.tensor as pt
from numpy.testing import assert_allclose

from cge_modeling.pytensorf.compile import compile_cge_model_to_pytensor
from cge_modeling.pytensorf.optimize import root
from tests.utilities.models import load_model_1

FLOATX = pytensor.config.floatX


def test_simple_root():
    x = pt.dscalar("x")
    y = pt.dscalar("y")
    eq1 = x**2 + y - 1
    eq2 = x - y**2 + 1
    system = pt.stack([eq1, eq2])
    jac = pt.stack(pytensor.gradient.jacobian(system, [x, y]))
    jac_inv = pt.linalg.solve(jac, pt.identity_like(jac))

    f = pytensor.compile.builders.OpFromGraph([x, y], [system])
    f_jac_inv = pytensor.compile.builders.OpFromGraph([x, y], [jac_inv])

    x_val = 1.0
    y_val = 1.0

    solution, converged, step_size, n_steps = root(f, f_jac_inv, [x_val, y_val])

    _root = solution[-1].eval()
    assert_allclose(_root, [0.0, 1.0], atol=1e-8)


def test_small_model():
    variables = Y, C, L_d, K_d, P, r, resid = list(
        map(pt.dscalar, ["Y", "C", "L_d", "K_d", "P", "r", "resid"])
    )
    params = K_s, L_s, A, alpha, w = list(map(pt.dscalar, ["K_s", "L_s", "A", "alpha", "w"]))

    def f_model(*args):
        Y, C, L_d, K_d, P, r, resid, *params = args
        K_s, L_s, A, alpha, w = params

        equations = pt.stack(
            [
                Y - A * K_d**alpha * L_d ** (1 - alpha),
                r * K_d - alpha * Y * P,
                w * L_d - (1 - alpha) * Y * P,
                Y - C,
                P * C - w * L_s - r * K_s,
                K_d - K_s,
                L_d - L_s + resid,
            ]
        )

        return equations

    equations = f_model(*variables, *params)
    jac = pt.stack(pytensor.gradient.jacobian(equations, variables)).T
    jac_inv = pt.linalg.solve(jac, pt.identity_like(jac), check_finite=False)

    f_jac = pytensor.compile.builders.OpFromGraph(variables + params, outputs=[jac], inline=True)
    f_jac_inv = pytensor.compile.builders.OpFromGraph(
        variables + params, outputs=[jac_inv], inline=True
    )

    x0 = np.array([11000, 11000, 7000, 4000, 1, 1, 0], dtype=FLOATX)
    param_vals = np.array([4000, 7000, 2, 0.33, 1], dtype=FLOATX)

    root_history, converged, step_size, n_steps = root(f_model, f_jac_inv, x0=x0, exog=param_vals)

    assert converged[-1].eval()
    expected_root = np.array(
        [1.16392629e04, 1.16392629e04, 7000.0, 4000.0, 0.897630824, 0.861940299, 0.0]
    )
    root_eval = root_history[-1].eval()

    # Check optimizer converges to the scipy result
    assert_allclose(root_eval, expected_root, atol=1e-8)

    # Check Y = C
    assert_allclose(root_eval[0], root_eval[1], atol=1e-8)

    # Check residuals are zero at the root
    assert_allclose(f_model(*root_eval, *param_vals).eval(), np.zeros(7), atol=1e-8)


def test_small_model_from_compile():
    mod = load_model_1(parse_equations_to_sympy=False)
    (f_model, f_jac, f_jac_inv) = compile_cge_model_to_pytensor(mod, inverse_method="solve")
    data = {
        "Y": 11000,
        "C": 11000,
        "income": 11000,
        "L_d": 7000,
        "K_d": 4000,
        "P": 1,
        "r": 1,
        "resid": 0,
    }
    param_data = {"K_s": 4000, "L_s": 7000, "A": 2, "alpha": 0.33, "w": 1}
    expected_roots = {
        "Y": 11639.2629,
        "C": 11639.2629,
        "income": 10447.761194,
        "L_d": 7000.0,
        "K_d": 4000.0,
        "P": 0.897630824,
        "r": 0.861940299,
        "resid": 0.0,
    }
    x0 = np.array([data[var] for var in mod.variable_names], dtype=FLOATX)
    params = np.array([param_data[var] for var in mod.parameter_names], dtype=FLOATX)

    root_history, converged, step_size, n_steps = root(
        f_model, f_jac_inv, x0=x0, exog=params, tol=1e-8, max_iter=500
    )
    root_eval = root_history[-1].eval()
    with np.printoptions(precision=10):
        print(root_eval)
    assert converged[-1].eval()

    # Check optimizer converges to the scipy result
    assert_allclose(
        root_eval,
        np.array([expected_roots[var] for var in mod.variable_names], dtype=FLOATX),
        atol=1e-8,
    )

    # Check Y = C
    Y_idx = mod.variable_names.index("Y")
    C_idx = mod.variable_names.index("C")
    assert_allclose(root_eval[Y_idx], root_eval[C_idx], atol=1e-8)

    # Check residuals are zero at the root
    assert_allclose(f_model(*root_eval, *params).eval(), np.zeros(8), atol=1e-8)
