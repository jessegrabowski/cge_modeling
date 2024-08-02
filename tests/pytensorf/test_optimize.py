import numpy as np
import pytensor
import pytensor.tensor as pt

from numpy.testing import assert_allclose

from cge_modeling.base.utilities import flat_array_to_variable_dict
from cge_modeling.pytensorf.compile import (
    compile_cge_model_to_pytensor_Op,
)
from cge_modeling.pytensorf.optimize import root
from tests.utilities.models import (
    calibrate_model_2,
    load_model_1,
    load_model_2,
    model_2_data,
)

FLOATX = pytensor.config.floatX


def _postprocess_root_return(root_histories: list[pt.TensorVariable]) -> np.ndarray:
    final_symbolic = pt.concatenate(
        [pt.atleast_1d(root_history[-1]).ravel() for root_history in root_histories]
    )
    final_roots = final_symbolic.eval()
    return final_roots


def test_simple_root():
    x = pt.dscalar("x")
    y = pt.dscalar("y")
    eq1 = x**2 + y - 1
    eq2 = x - y**2 + 1
    system = pt.stack([eq1, eq2])
    jac = pt.stack(pytensor.gradient.jacobian(system, [x, y]))

    f = pytensor.compile.builders.OpFromGraph([x, y], [system])
    f_jac = pytensor.compile.builders.OpFromGraph([x, y], [jac])

    x_val = 1.0
    y_val = 1.0

    root_histories, converged, step_size, n_steps = root(f, f_jac, {"x": x_val, "y": y_val})
    final_root = _postprocess_root_return(root_histories)

    assert_allclose(final_root, [0.0, 1.0], atol=1e-8)


def test_small_model():
    def f_model(Y, C, L_d, K_d, P, r, resid, K_s, L_s, A, alpha, w):
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

    variables = list(map(pt.dscalar, ["Y", "C", "L_d", "K_d", "P", "r", "resid"]))
    params = list(map(pt.dscalar, ["K_s", "L_s", "A", "alpha", "w"]))

    equations = f_model(*variables, *params)
    jac = pt.stack(pytensor.gradient.jacobian(equations, variables)).T
    f_jac = pytensor.compile.builders.OpFromGraph(variables + params, outputs=[jac], inline=True)

    x0 = {"Y": 11000, "C": 11000, "L_d": 7000, "K_d": 4000, "r": 1, "P": 1, "resid": 0}
    param_vals = {"K_s": 4000, "L_s": 7000, "A": 2, "alpha": 0.33, "w": 1}

    root_histories, converged, step_size, n_steps = root(
        f_model, f_jac, initial_data=x0, parameters=param_vals
    )
    root_eval = _postprocess_root_return(root_histories)

    assert converged[-1].eval()
    expected_root = np.array(
        [1.16392629e04, 1.16392629e04, 7000.0, 4000.0, 0.897630824, 0.861940299, 0.0]
    )
    f_model_compiled = pytensor.function(
        inputs=variables + params, outputs=equations, mode="FAST_COMPILE"
    )

    # Check optimizer converges to the scipy result
    assert_allclose(root_eval, expected_root, atol=1e-8)

    # Check Y = C
    assert_allclose(root_eval[0], root_eval[1], atol=1e-8)

    # Check residuals are zero at the root
    assert_allclose(f_model_compiled(*root_eval, **param_vals).ravel(), np.zeros(7), atol=1e-8)


def test_small_model_from_compile():
    mod = load_model_1(parse_equations_to_sympy=False, backend="pytensor", compile=False)
    f_model, f_jac = compile_cge_model_to_pytensor_Op(mod)
    data = {
        "Y": 11000.0,
        "C": 11000.0,
        "income": 11000.0,
        "L_d": 7000.0,
        "K_d": 4000.0,
        "P": 1.0,
        "r": 1.0,
        "resid": 0.0,
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

    sorted_data = {k: data[k] for k in mod.variable_names}
    sorted_params = {k: param_data[k] for k in mod.parameter_names}

    root_histories, converged, step_size, n_steps = root(
        f_model,
        f_jac,
        initial_data=sorted_data,
        parameters=sorted_params,
        tol=1e-8,
        max_iter=500,
    )

    root_eval = _postprocess_root_return(root_histories)
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
    params = np.concatenate([np.atleast_1d(param_data[param]) for param in mod.parameter_names])
    assert_allclose(f_model(*root_eval, *params).eval(), np.zeros(8), atol=1e-8)


def test_sector_model_from_compile():
    mod = load_model_2(parse_equations_to_sympy=False, mode="FAST_COMPILE", backend="pytensor")
    calib_dict = calibrate_model_2(**model_2_data)

    x0 = {var.name: calib_dict[var.name] for var in mod.variables}
    params = {param.name: calib_dict[param.name] for param in mod.parameters}

    f_model, f_jac = compile_cge_model_to_pytensor_Op(mod)
    root_history, converged, step_size, n_steps = root(
        f_model, f_jac, initial_data=x0, parameters=params, tol=1e-8, max_iter=500
    )

    root_eval = _postprocess_root_return(root_history)
    root_point = flat_array_to_variable_dict(root_eval, mod.variables, mod.coords)

    # Check residuals are zero at the root
    assert_allclose(
        mod.f_system(**root_point, **params).ravel(),
        np.zeros(mod.n_variables),
        atol=1e-8,
    )
