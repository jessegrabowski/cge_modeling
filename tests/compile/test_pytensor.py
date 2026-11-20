import numpy as np

from cge_modeling import compile_model
from cge_modeling.base.utilities import variable_dict_to_flat_array
from tests.utilities.models import (
    build_model_1_in_context,
    calibrate_model_1,
    load_and_cache_model,
    load_model_1,
    model_1_data,
)


def test_context_built_model_compiles_to_the_string_model_functions():
    context_model = build_model_1_in_context()
    compile_model(context_model, functions_to_compile=("root", "minimize"))
    string_model = load_and_cache_model(model_id=1, functions_to_compile=("root", "minimize"))

    calibrated = calibrate_model_1(**model_1_data)
    x, theta = variable_dict_to_flat_array(
        calibrated, string_model.variables, string_model.parameters
    )
    x_perturbed = x * (1 + 0.05 * np.random.default_rng(0).standard_normal(x.shape))
    point = np.ones_like(x)

    for name in ["f_system", "f_jac", "f_resid", "f_grad", "f_hess"]:
        np.testing.assert_allclose(
            getattr(context_model, name)(x_perturbed, theta),
            getattr(string_model, name)(x_perturbed, theta),
            atol=1e-10,
            err_msg=name,
        )
    np.testing.assert_allclose(
        context_model.f_hessp(x_perturbed, point, theta),
        string_model.f_hessp(x_perturbed, point, theta),
        atol=1e-10,
    )


def test_context_built_model_scan_euler_matches_the_string_model():
    context_model = build_model_1_in_context()
    compile_model(context_model, functions_to_compile=("euler",), use_scan_euler=True)
    string_model = load_model_1(functions_to_compile=("euler",), use_scan_euler=True)
    calibrated = calibrate_model_1(**model_1_data)
    _, theta_final = variable_dict_to_flat_array(
        {**calibrated, "L_s": calibrated["L_s"] * 1.5},
        string_model.variables,
        string_model.parameters,
    )

    context_result = context_model._solve_with_euler_approximation(
        calibrated, theta_final, n_steps=50
    )
    string_result = string_model._solve_with_euler_approximation(
        calibrated, theta_final, n_steps=50
    )

    np.testing.assert_allclose(
        context_result["variables"].dataset.isel(step=-1).to_array().values,
        string_result["variables"].dataset.isel(step=-1).to_array().values,
        rtol=1e-10,
    )
