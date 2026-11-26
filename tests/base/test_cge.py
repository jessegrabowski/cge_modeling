import logging

from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from scipy.differentiate import jacobian
from scipy.optimize import OptimizeResult

from cge_modeling import model as cm
from cge_modeling.base.cge import CGEModel, _judge_equilibrium, _parse_compile_kwarg
from cge_modeling.base.utilities import flat_array_to_variable_dict, variable_dict_to_flat_array
from tests.utilities.models import (
    build_model_1,
    calibrate_model_1,
    calibrate_model_2,
    load_and_cache_model,
    load_model_1,
    model_1_data,
    model_2_data,
)


def numeric_jacobian(function, x, *args):
    """Differentiate ``function`` at ``x`` with :func:`scipy.differentiate.jacobian`, evaluating one point at a time."""

    def evaluate_points(points):
        flat_points = points.reshape(points.shape[0], -1)
        values = np.stack(
            [function(flat_points[:, column], *args) for column in range(flat_points.shape[1])],
            axis=-1,
        )
        return values.reshape(values.shape[:1] + points.shape[1:])

    return jacobian(evaluate_points, x, tolerances={"atol": 1e-8, "rtol": 1e-6})


@pytest.mark.parametrize(
    "model_id, calibrate_model, data",
    [(1, calibrate_model_1, model_1_data), (2, calibrate_model_2, model_2_data)],
    ids=["simple_model", "3-goods simple"],
)
def test_model_gradients(model_id, calibrate_model, data):
    mod = load_and_cache_model(model_id, mode="FAST_RUN", functions_to_compile=("root", "minimize"))
    x, theta = variable_dict_to_flat_array(calibrate_model(**data), mod.variables, mod.parameters)
    rng = np.random.default_rng(0)
    x = x * rng.uniform(0.9, 1.1, size=x.shape)
    p = rng.normal(size=x.shape)

    system_jacobian = numeric_jacobian(mod.f_system, x, theta)
    gradient_jacobian = numeric_jacobian(mod.f_grad, x, theta)

    for name, numeric, analytic in [
        ("system", system_jacobian, np.asarray(mod.f_jac(x, theta))),
        ("gradient", gradient_jacobian, np.asarray(mod.f_hess(x, theta))),
    ]:
        failed = ~numeric.success
        assert not failed.any(), (
            f"The {name} Jacobian did not converge at {failed.sum()} entries, "
            f"indices {np.argwhere(failed)[:10].tolist()}, "
            f"error estimates {numeric.error[failed][:10]}, "
            f"numeric values {numeric.df[failed][:10]}, "
            f"analytic values {analytic[failed][:10]}, "
            f"status {numeric.status[failed][:10]}"
        )
    np.testing.assert_allclose(mod.f_jac(x, theta), system_jacobian.df, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(mod.f_hess(x, theta), gradient_jacobian.df, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        mod.f_hessp(x, p, theta).ravel(), mod.f_hess(x, theta) @ p, rtol=1e-10, atol=1e-8
    )


@pytest.mark.parametrize(
    "model_id, calibrate_model, data",
    [
        (1, calibrate_model_1, model_1_data),
        # (2, calibrate_model_2, model_2_data),
    ],
    ids=["simple_model"],  # , "3-goods simple"],
)
@pytest.mark.parametrize(
    "method, solver_kwargs, functions_to_compile",
    [
        (
            "_solve_with_minimize",
            {
                "method": "trust-exact",
                "use_hess": True,
                "use_hessp": False,
                "maxiter": 10_000,
                "tol": 1e-16,
            },
            "minimize",
        ),
        (
            "_solve_with_minimize",
            {
                "method": "trust-ncg",
                "use_hess": False,
                "use_hessp": True,
                "maxiter": 10_000,
                "tol": 1e-16,
            },
            "minimize",
        ),
        (
            "_solve_with_root",
            {"method": "hybr", "use_jac": True, "maxiter": 10_000, "tol": 1e-16},
            "root",
        ),
        ("_solve_with_euler_approximation", {"n_steps": 500}, "euler"),
    ],
    ids=["minimize_hess", "minimize_hessp", "root", "euler"],
)
def test_jax_mode_agrees_with_default_mode(
    model_id: int,
    calibrate_model: Callable,
    data: dict,
    method: str,
    solver_kwargs: dict,
    functions_to_compile: str,
):
    pytest.importorskip("jax")
    model_default = load_and_cache_model(
        model_id=model_id, functions_to_compile=functions_to_compile
    )
    model_jax = load_and_cache_model(
        model_id=model_id, mode="JAX", functions_to_compile=functions_to_compile
    )

    def solver_agreement_checks(results: list, names: list):
        solutions = []
        for res, name in zip(results, names):
            if hasattr(res, "success"):
                assert res.success or np.allclose(res.fun, 0.0, rtol=1e-8, atol=1e-8), (
                    f"{name} solver failed to converge"
                )

            solution = res.x if hasattr(res, "x") else res
            assert not np.all(np.isnan(solution)), f"{name} solver returned NaNs"
            assert np.all(np.isfinite(solution)), f"{name} solver returned Infs"
            solutions.append(solution)

        np.testing.assert_allclose(
            np.around(solutions[0], 4),
            np.around(solutions[1], 4),
            atol=1e-5,
            rtol=1e-5,
            err_msg="Solvers disagree",
        )

    calibrated_data = calibrate_model(**data)
    x, theta = variable_dict_to_flat_array(
        calibrated_data, model_default.variables, model_default.parameters
    )
    np.testing.assert_allclose(model_default.f_system(x, theta), 0, atol=1e-8)
    np.testing.assert_allclose(model_jax.f_system(x, theta), 0, atol=1e-8)

    labor_increase = calibrated_data.copy()
    labor_increase["L_s"] = 10_000

    _, theta_labor_increase = variable_dict_to_flat_array(
        labor_increase, model_default.variables, model_default.parameters
    )

    res_default = getattr(model_default, method)(
        calibrated_data, theta_labor_increase, **solver_kwargs
    )
    res_jax = getattr(model_jax, method)(calibrated_data, theta_labor_increase, **solver_kwargs)

    if method == "_solve_with_euler_approximation":
        res_default = res_default["parameters"].dataset.isel(step=-1).to_array().values
        res_jax = res_jax["parameters"].dataset.isel(step=-1).to_array().values

    solver_agreement_checks([res_default, res_jax], ["Default mode", "JAX mode"])


@pytest.mark.parametrize(
    "gtol, expected_success",
    [(1e10, False), (1e-10, True)],
    ids=["stops_before_first_step", "reaches_equilibrium"],
)
def test_minimize_success_reflects_equilibrium_residuals(gtol, expected_success):
    model = load_and_cache_model(model_id=2, functions_to_compile="minimize")
    calibrated_data = calibrate_model_2(**model_2_data)
    shocked_data = {**calibrated_data, "L_s": calibrated_data["L_s"] * 1.5}
    _, theta_final = variable_dict_to_flat_array(shocked_data, model.variables, model.parameters)

    result = model._solve_with_minimize(
        calibrated_data,
        theta_final,
        method="trust-ncg",
        progressbar=False,
        verbose=False,
        gtol=gtol,
    )

    assert bool(result.success) is expected_success
    assert "equilibrium_tol" in result.message


def test_root_success_reflects_equilibrium_residuals_when_hybr_stops_at_one():
    model = load_and_cache_model(model_id=1)
    calibrated_data = calibrate_model_1(**model_1_data)

    result = model.simulate(
        calibrated_data, final_delta_pct={"L_s": 0.1}, n_iter_euler=20, progressbar=False
    )

    assert result["optim_res"].success


@pytest.mark.parametrize(
    "solver_success, squared_error, expected_note",
    [(False, 1e-20, "below equilibrium_tol=1e-06"), (True, 1.0, "above equilibrium_tol=1e-06")],
    ids=["solver_stalled_at_equilibrium", "solver_stopped_short"],
)
def test_judge_equilibrium_overrules_the_solver_from_the_residuals(
    solver_success, squared_error, expected_note
):
    result = OptimizeResult(success=solver_success, message="Solver message.")

    judged = _judge_equilibrium(result, squared_error, equilibrium_tol=1e-6)

    assert judged.success is not solver_success
    assert judged.message.startswith("Solver message.")
    assert expected_note in judged.message


def test_judge_equilibrium_keeps_the_message_when_it_agrees_with_the_solver():
    result = OptimizeResult(success=True, message="Solver message.")

    judged = _judge_equilibrium(result, 1e-20, equilibrium_tol=1e-6)

    assert judged.success
    assert judged.message == "Solver message."


def test_root_reports_failure_when_it_stops_short_of_equilibrium():
    model = load_and_cache_model(model_id=1, functions_to_compile="root")
    calibrated_data = calibrate_model_1(**model_1_data)
    shocked_data = {**calibrated_data, "L_s": calibrated_data["L_s"] * 1.5}
    _, theta_final = variable_dict_to_flat_array(shocked_data, model.variables, model.parameters)

    result = model._solve_with_root(
        calibrated_data, theta_final, progressbar=False, verbose=False, options={"maxfev": 2}
    )

    assert not result.success


def test_scan_euler_matches_python_loop_euler():
    scan_model = load_model_1(functions_to_compile=("euler",), use_scan_euler=True)
    loop_model = load_model_1(functions_to_compile=("euler",))
    calibrated_data = calibrate_model_1(**model_1_data)
    _, theta_final = variable_dict_to_flat_array(
        {**calibrated_data, "L_s": 10_000}, scan_model.variables, scan_model.parameters
    )

    scan_result = scan_model._solve_with_euler_approximation(
        calibrated_data, theta_final, n_steps=500
    )
    loop_result = loop_model._solve_with_euler_approximation(
        calibrated_data, theta_final, n_steps=500
    )

    np.testing.assert_allclose(
        scan_result["variables"].dataset.isel(step=-1).to_array().values,
        loop_result["variables"].dataset.isel(step=-1).to_array().values,
        rtol=1e-10,
        atol=1e-8,
    )


def test_simulate_with_scan_euler_reports_no_nans(caplog):
    mod = load_model_1(functions_to_compile=("euler", "root"), use_scan_euler=True)
    initial_state = calibrate_model_1(**model_1_data)

    with caplog.at_level(logging.WARNING, logger="cge_modeling"):
        res = mod.simulate(initial_state, final_delta_pct={"L_s": 0.5}, n_iter_euler=100)

    assert res["optim_res"].success
    assert "NaNs were encountered" not in caplog.text


def test_generate_SAM():
    mod = load_and_cache_model(model_id=1, functions_to_compile="minimize")
    param_dict = {
        "alpha": 0.75,
        "A": 2.0,
        "L_s": 1000,
        "K_s": 5000,
        "w": 1.0,
    }
    initial_guess = {
        "C": 10000,
        "Y": 10000,
        "income": 10000,
        "K_d": 5000,
        "L_d": 1000,
        "r": 1.0,
        "P": 1.0,
        "resid": 0.0,
    }
    # fixed_values = {"r": 1.0, "P": 1.0, "resid": 0.0}

    mod.generate_SAM(
        param_dict=param_dict,
        initial_variable_guess=initial_guess,
        # fixed_values=fixed_values,
        solve_method="minimize",
        method="nelder-mead",
        use_jac=False,
        use_hess=False,
    )


def test_generate_SAM_with_fixed_value_matches_free_solution():
    mod = load_and_cache_model(model_id=1, functions_to_compile="minimize")
    param_dict = {"alpha": 0.75, "A": 2.0, "L_s": 1000, "K_s": 5000, "w": 1.0}
    initial_guess = {
        "C": 10000,
        "Y": 10000,
        "income": 10000,
        "K_d": 5000,
        "L_d": 1000,
        "r": 1.0,
        "P": 1.0,
        "resid": 0.0,
    }
    solver_kwargs = {
        "solve_method": "minimize",
        "method": "trust-ncg",
        "progressbar": False,
        "verbose": False,
    }

    free_solution = mod.generate_SAM(
        param_dict=param_dict, initial_variable_guess=initial_guess, **solver_kwargs
    )

    # Y is the first variable, so mapping the solution onto the free variables by position would
    # shift every other variable by one.
    fixed_solution = mod.generate_SAM(
        param_dict=param_dict,
        initial_variable_guess=initial_guess,
        fixed_values={"Y": free_solution["Y"]},
        **solver_kwargs,
    )

    for name in mod.variable_names:
        np.testing.assert_allclose(fixed_solution[name], free_solution[name], rtol=1e-6, atol=1e-4)


def test_generate_SAM_rejects_fixed_values_outside_minimize():
    mod = load_and_cache_model(model_id=1, functions_to_compile="minimize")

    with pytest.raises(ValueError, match="only supported with solve_method='minimize'"):
        mod.generate_SAM(
            param_dict={}, initial_variable_guess={}, fixed_values={"Y": 1.0}, solve_method="root"
        )


def test_generate_SAM_rejects_unknown_fixed_value_names():
    mod = load_and_cache_model(model_id=1, functions_to_compile="minimize")

    with pytest.raises(ValueError, match=r"not model variables: \{'alpha'\}"):
        mod.generate_SAM(
            param_dict={},
            initial_variable_guess={},
            fixed_values={"alpha": 0.5},
            solve_method="minimize",
        )


def test_invalid_simulation_raises():
    mod = load_and_cache_model(model_id=1, functions_to_compile=("root", "euler"))
    initial_state = calibrate_model_1(**model_1_data)

    with pytest.raises(
        ValueError, match="initial_state must be a Result or a dict of initial values"
    ):
        mod.simulate([1, 2, 3, 4, 5], final_values={"L_s": 1500})

    with pytest.raises(
        ValueError,
        match=r"initial_state must contain values for all variables and parameters in the "
        r"model\. Did not find values for alpha",
    ):
        bad_init = initial_state.copy()
        del bad_init["alpha"]

        mod.simulate(bad_init, final_values={"L_s": 1500})

    with pytest.raises(
        ValueError,
        match="initial_state contains values for variables or parameters not in the model: lol",
    ):
        bad_init = initial_state.copy()
        bad_init["lol"] = 3
        mod.simulate(bad_init, final_values={"L_s": 1500})

    with pytest.raises(ValueError, match="No parameters to update!"):
        mod.simulate(initial_state)

    with pytest.raises(ValueError, match="Arguments K_s are repeated among"):
        mod.simulate(
            initial_state, final_values={"L_s": 1500, "K_s": 5000}, final_delta_pct={"K_s": 0.5}
        )


@pytest.mark.parametrize(
    "scenario_kwargs, expected_result",
    [
        ({"final_values": {"L_s": 1500}}, {"L_s": lambda x: 1500}),
        ({"final_delta_pct": {"L_s": 0.5}}, {"L_s": lambda x: x * 0.5}),
        ({"final_delta": {"L_s": -500}}, {"L_s": lambda x: x - 500}),
        (
            {"final_values": {"L_s": 1500}, "final_delta_pct": {"K_s": 0.5}},
            {"L_s": lambda x: 1500, "K_s": lambda x: x * 0.5},
        ),
    ],
    ids=["final_values", "final_delta_pct", "final_delta", "combined"],
)
def test_simulate(scenario_kwargs, expected_result):
    mod = load_and_cache_model(model_id=1)
    initial_state = calibrate_model_1(**model_1_data)

    res = mod.simulate(initial_state, **scenario_kwargs)

    for key in expected_result:
        final_param = res["optimizer"]["parameters"][key].isel(step=-1)
        assert final_param == expected_result[key](initial_state[key])


def test_simulate_compiles_an_uncompiled_model():
    mod = build_model_1()
    initial_state = calibrate_model_1(**model_1_data)

    res = mod.simulate(initial_state, final_values={"L_s": 1500}, n_iter_euler=100)

    assert list(mod._compiled) == ["euler", "root"]
    assert res["optimizer"]["parameters"]["L_s"].isel(step=-1) == 1500


def test_check_for_equilibrium_reads_simulation_datatree(caplog):
    mod = load_and_cache_model(model_id=1)
    initial_state = calibrate_model_1(**model_1_data)
    res = mod.simulate(initial_state, final_delta_pct={"L_s": 0.5})

    with caplog.at_level(logging.INFO, logger="cge_modeling"):
        mod.check_for_equilibrium(res["optimizer"])

    assert "Equilibrium found!" in caplog.text


def test_check_for_equilibrium_flags_mismatched_state(caplog):
    mod = load_and_cache_model(model_id=1)
    initial_state = calibrate_model_1(**model_1_data)
    optimizer = mod.simulate(initial_state, final_delta_pct={"L_s": 0.5})["optimizer"]

    pre_shock_variables_post_shock_parameters = xr.DataTree.from_dict(
        {
            "variables": optimizer["variables"].dataset.isel(step=0),
            "parameters": optimizer["parameters"].dataset.isel(step=-1),
        }
    )

    with caplog.at_level(logging.INFO, logger="cge_modeling"):
        mod.check_for_equilibrium(pre_shock_variables_post_shock_parameters)

    assert "Equilibrium not found" in caplog.text


def test_check_for_equilibrium_requires_variables_and_parameters():
    mod = load_and_cache_model(model_id=1)
    variables_only = xr.DataTree.from_dict({"variables": xr.Dataset()})

    with pytest.raises(ValueError, match="DataTree must contain variables and parameters children"):
        mod.check_for_equilibrium(variables_only)


@pytest.mark.parametrize(
    "compile_arg, expected_output",
    [
        (None, None),
        ("all", ["root", "minimize", "euler"]),
        (True, ["root", "minimize", "euler"]),
        ("root", ["root"]),
        (["root", "minimize"], ["root", "minimize"]),
    ],
    ids=["none", "all_str", "true_bool", "single_str", "list_str"],
)
def test_parse_compile_kwarg(compile_arg, expected_output):
    assert _parse_compile_kwarg(compile_arg) == expected_output


@pytest.mark.parametrize(
    "functions_to_compile, expected_compiled_list",
    [
        ("all", ["root", "minimize", "euler"]),
        (("root",), ["root"]),
        (("minimize",), ["minimize"]),
        (("euler",), ["euler"]),
        (None, []),
    ],
    ids=["all", "root_only", "minimize_only", "euler_only", "none"],
)
def test_compile_builds_only_the_requested_solver_functions(
    functions_to_compile, expected_compiled_list
):
    model = load_and_cache_model(
        model_id=1, mode="FAST_RUN", functions_to_compile=functions_to_compile
    )

    if expected_compiled_list:
        assert model._compiled
    else:
        assert not model._compiled

    # f_system is always compiled
    assert model.f_system is not None

    compiled_root = "root" in expected_compiled_list
    compiled_minimize = "minimize" in expected_compiled_list
    compiled_euler = "euler" in expected_compiled_list

    assert (model.f_jac is not None) == compiled_root
    for func in [model.f_resid, model.f_grad, model.f_hess, model.f_hessp]:
        assert (func is not None) == compiled_minimize
    assert (model.f_euler is not None) == compiled_euler


def check_function_signature(func, expected_input_names):
    input_names = [x.name for x in func.maker.fgraph.inputs]
    assert input_names == expected_input_names


def test_compiled_functions_take_packed_inputs():
    model = load_and_cache_model(model_id=1, mode="FAST_RUN", functions_to_compile="all")

    for func in [model.f_system, model.f_jac, model.f_resid, model.f_grad, model.f_hess]:
        check_function_signature(func, ["x", "theta"])
    check_function_signature(model.f_hessp, ["x", "p", "theta"])
    check_function_signature(
        model.f_euler.keywords["f_step"], ["x", "theta", "theta_initial", "theta_final", "n_steps"]
    )


def test_compile_rejects_a_non_square_context_built_model():
    with CGEModel(coords={"i": ["a", "b"]}) as model:
        P = cm.Variable("P", dims="i")
        cm.Equation("first price", P.sel(i="a"), 1.0)

    with pytest.raises(ValueError, match="Found 2 variables but 1 equations"):
        model.compile()


def test_compile_logs_the_model_size_once(caplog):
    model = load_and_cache_model(model_id=1, functions_to_compile=None)

    with caplog.at_level(logging.INFO, logger="cge_modeling"):
        caplog.clear()
        model.compile(functions_to_compile=None)

    size_messages = [r.message for r in caplog.records if "Compiling a model with" in r.message]
    assert len(size_messages) == 1


def test_simulate_compiles_only_the_functions_it_uses():
    model = build_model_1()

    model.simulate(
        calibrate_model_1(**model_1_data),
        final_delta_pct={"L_s": 0.1},
        use_euler_approximation=False,
        progressbar=False,
    )

    assert model._compiled == ["root"]
    assert model.f_jac is not None
    assert model.f_euler is None
    assert model.f_grad is None


def test_root_step_statistics_report_the_residual_norm_and_the_merit_gradient():
    model = load_and_cache_model(model_id=1, mode="FAST_RUN", functions_to_compile="all")
    x, theta = variable_dict_to_flat_array(
        calibrate_model_1(**model_1_data), model.variables, model.parameters
    )
    x = x * np.random.default_rng(0).uniform(0.9, 1.1, size=x.shape)
    residuals = np.asarray(model.f_system(x, theta))
    jacobian = np.asarray(model.f_jac(x, theta))

    statistics = model._step_statistics("root")(x, theta)

    np.testing.assert_allclose(statistics["f_value"], np.linalg.norm(residuals), rtol=1e-10)
    np.testing.assert_allclose(
        statistics["grad_norm"], np.linalg.norm(jacobian.T @ residuals), rtol=1e-10
    )


def test_root_step_statistics_from_a_sparse_jacobian_match_the_dense_ones():
    dense = load_and_cache_model(model_id=1, mode="FAST_RUN", functions_to_compile="all")
    sparse = build_model_1().compile(functions_to_compile=["root"], use_sparse_matrices=True)
    x, theta = variable_dict_to_flat_array(
        calibrate_model_1(**model_1_data), dense.variables, dense.parameters
    )
    x = x * np.random.default_rng(0).uniform(0.9, 1.1, size=x.shape)

    dense_statistics = dense._step_statistics("root")(x, theta)
    sparse_statistics = sparse._step_statistics("root")(x, theta)

    assert sparse_statistics == pytest.approx(dense_statistics, rel=1e-10)


def test_minimize_step_statistics_report_the_loss_and_its_gradient_norm():
    model = load_and_cache_model(model_id=1, mode="FAST_RUN", functions_to_compile="all")
    x, theta = variable_dict_to_flat_array(
        calibrate_model_1(**model_1_data), model.variables, model.parameters
    )
    x = x * np.random.default_rng(0).uniform(0.9, 1.1, size=x.shape)
    residuals = np.asarray(model.f_system(x, theta))

    statistics = model._step_statistics("minimize")(x, theta)

    np.testing.assert_allclose(statistics["f_value"], 0.5 * np.sum(residuals**2), rtol=1e-10)
    np.testing.assert_allclose(
        statistics["grad_norm"], np.linalg.norm(model.f_grad(x, theta)), rtol=1e-10
    )


def test_root_solve_with_a_bare_progressbar_draws_the_shared_progress_table(capsys):
    model = load_and_cache_model(model_id=1, mode="FAST_RUN", functions_to_compile=("root",))
    data = calibrate_model_1(**model_1_data)
    _, theta = variable_dict_to_flat_array(data, model.variables, model.parameters)

    model._solve_with_root(data, theta, progressbar=True, verbose=False)

    output = capsys.readouterr().err
    assert "Finding Roots" in output
    assert "Step" in output
