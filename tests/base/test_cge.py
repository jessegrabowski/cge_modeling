import logging

from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from pytensor.scan.op import Scan
from scipy.differentiate import jacobian
from scipy.optimize import OptimizeResult

from cge_modeling import model as cm
from cge_modeling.base.cge import CGEModel, _judge_equilibrium, _parse_compile_kwarg
from cge_modeling.base.utilities import flat_array_to_variable_dict, variable_dict_to_flat_array
from cge_modeling.tools.output_tools import datatree_to_value_dict
from tests.utilities.models import (
    build_model_1,
    calibrate_model_1,
    calibrate_model_2,
    load_and_cache_model,
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

    return jacobian(evaluate_points, x, tolerances={"atol": 1e-8})


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
        ("system Jacobian", system_jacobian, np.asarray(mod.f_jac(x, theta))),
        ("loss Hessian", gradient_jacobian, np.asarray(mod.f_hess(x, theta))),
    ]:
        # The gradient reaches 1e7 here, so finite differences cannot always meet scipy's absolute tolerance and
        # report non-convergence on some CPUs. Their own error estimate bounds what they can certify instead.
        bound = 10 * numeric.error + 1e-6 * np.abs(analytic) + 1e-8
        np.testing.assert_array_less(np.abs(analytic - numeric.df), bound, err_msg=name)
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
        ("_solve_with_path", {"n_steps": 500}, "path"),
    ],
    ids=["minimize_hess", "minimize_hessp", "root", "path"],
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

    if method == "_solve_with_path":
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
        calibrated_data, final_delta_pct={"L_s": 0.1}, n_path_steps=20, progressbar=False
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


def test_scan_path_matches_python_loop_path():
    scan_model = load_and_cache_model(1, functions_to_compile=("path",), use_scan_path=True)
    loop_model = load_and_cache_model(1, functions_to_compile=("path",))
    calibrated_data = calibrate_model_1(**model_1_data)
    _, theta_final = variable_dict_to_flat_array(
        {**calibrated_data, "L_s": 10_000}, scan_model.variables, scan_model.parameters
    )

    scan_result = scan_model._solve_with_path(calibrated_data, theta_final, n_steps=500)
    loop_result = loop_model._solve_with_path(calibrated_data, theta_final, n_steps=500)

    scan_path = scan_result["variables"].dataset.to_array().values
    loop_path = loop_result["variables"].dataset.to_array().values
    initial_point = np.array([calibrated_data[name] for name in loop_model.variable_names])

    np.testing.assert_allclose(scan_path, loop_path, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(loop_path[:, 0], initial_point)


def test_simulate_with_scan_path_reports_no_nans(caplog):
    mod = load_and_cache_model(1, functions_to_compile=("path", "root"), use_scan_path=True)
    initial_state = calibrate_model_1(**model_1_data)

    with caplog.at_level(logging.WARNING, logger="cge_modeling"):
        res = mod.simulate(initial_state, final_delta_pct={"L_s": 0.5}, n_path_steps=100)

    assert res["optim_res"].success
    assert "NaNs were encountered" not in caplog.text


def equilibrium_residual_norms(mod, path, steps):
    """Return the norm of the model's residuals at each of ``steps`` of a path result."""
    return np.array(
        [
            np.linalg.norm(
                mod.f_system(
                    *variable_dict_to_flat_array(
                        datatree_to_value_dict(path.isel(step=step)), mod.variables, mod.parameters
                    )
                )
            )
            for step in steps
        ]
    )


@pytest.mark.parametrize("use_scan_path", [False, True], ids=["loop", "scan"])
def test_solve_every_puts_the_path_back_on_the_equilibrium_at_each_checkpoint(use_scan_path):
    mod = load_and_cache_model(
        1, functions_to_compile=("path", "root"), use_scan_path=use_scan_path
    )
    initial_state = calibrate_model_1(**model_1_data)
    scenario = {"final_delta_pct": {"L_s": 3.0}, "n_path_steps": 40, "use_optimizer": False}

    plain = mod.simulate(initial_state, progressbar=False, **scenario)["path"]
    checked = mod.simulate(initial_state, solve_every=10, progressbar=False, **scenario)["path"]
    checkpoints = checked["converged"]["checkpoint"].values

    assert checked["converged"].all()
    np.testing.assert_array_equal(checkpoints, [10, 20, 30, 40])
    assert checked["variables"].sizes["step"] == plain["variables"].sizes["step"] == 41
    np.testing.assert_allclose(
        equilibrium_residual_norms(mod, checked, checkpoints), 0.0, atol=1e-8
    )
    assert equilibrium_residual_norms(mod, plain, [40])[0] > 1e-6


@pytest.mark.parametrize(
    "first_run, select_start",
    [
        ({}, lambda result: result),
        ({}, lambda result: result["optimizer"]),
        ({"use_optimizer": False, "solve_every": 25}, lambda result: result["path"]),
        ({"use_path_approximation": False}, lambda result: result),
    ],
    ids=["result", "optimizer_group", "solved_path_group", "optimizer_only_result"],
)
def test_simulate_continues_from_an_earlier_simulation(first_run, select_start):
    mod = load_and_cache_model(1, functions_to_compile=("path", "root"))
    initial_state = calibrate_model_1(**model_1_data)
    labor_supply = initial_state["L_s"]

    def final_variables(result):
        return result["optimizer"]["variables"].dataset.isel(step=-1).to_array().values

    first = mod.simulate(
        initial_state,
        final_values={"L_s": 0.5 * labor_supply},
        n_path_steps=50,
        progressbar=False,
        **first_run,
    )
    continued = mod.simulate(
        select_start(first),
        final_values={"L_s": 0.25 * labor_supply},
        n_path_steps=50,
        progressbar=False,
    )
    direct = mod.simulate(
        initial_state,
        final_values={"L_s": 0.25 * labor_supply},
        n_path_steps=100,
        progressbar=False,
    )

    np.testing.assert_allclose(
        continued["path"]["parameters"].dataset["L_s"].isel(step=0), 0.5 * labor_supply
    )
    np.testing.assert_allclose(
        final_variables(continued), final_variables(direct), rtol=1e-8, atol=1e-8
    )


@pytest.mark.parametrize(
    "select_start",
    [lambda result: result, lambda result: result["path"]],
    ids=["result", "path_group"],
)
def test_simulate_rejects_an_earlier_simulation_that_ended_off_the_equilibrium(select_start):
    mod = load_and_cache_model(1, functions_to_compile=("path", "root"))
    initial_state = calibrate_model_1(**model_1_data)

    unsolved = mod.simulate(
        initial_state,
        final_delta_pct={"L_s": 3.0},
        n_path_steps=5,
        use_optimizer=False,
        progressbar=False,
    )

    with pytest.raises(ValueError, match="not an equilibrium"):
        mod.simulate(select_start(unsolved), final_delta_pct={"L_s": 0.5}, progressbar=False)


def test_solve_every_with_minimize_solves_each_checkpoint():
    mod = load_and_cache_model(1, functions_to_compile=("path", "minimize"))
    initial_state = calibrate_model_1(**model_1_data)

    path = mod.simulate(
        initial_state,
        final_delta_pct={"L_s": 3.0},
        n_path_steps=40,
        solve_every=10,
        use_optimizer=False,
        optimizer_mode="minimize",
        tol=1e-14,
        progressbar=False,
    )["path"]
    checkpoints = path["converged"]["checkpoint"].values

    assert path["converged"].all()
    np.testing.assert_allclose(equilibrium_residual_norms(mod, path, checkpoints), 0.0, atol=1e-6)


def test_solve_every_leaves_the_last_step_to_the_optimizer_when_it_runs():
    mod = load_and_cache_model(1, functions_to_compile=("path", "root"))
    initial_state = calibrate_model_1(**model_1_data)
    scenario = {"final_delta_pct": {"L_s": 3.0}, "n_path_steps": 40, "progressbar": False}

    plain = mod.simulate(initial_state, **scenario)
    checked = mod.simulate(initial_state, solve_every=40, **scenario)

    assert checked["path"]["converged"].size == 0
    xr.testing.assert_allclose(
        checked["path"]["variables"].to_dataset(inherit=False),
        plain["path"]["variables"].to_dataset(inherit=False),
        rtol=1e-12,
    )
    assert checked["optim_res"].success


def test_solve_every_stores_the_parameters_each_checkpoint_was_solved_at():
    mod = load_and_cache_model(1, functions_to_compile=("path", "root"))
    initial_state = calibrate_model_1(**model_1_data)
    labor_supply = initial_state["L_s"]

    path = mod.simulate(
        initial_state,
        final_values={"L_s": 1.1 * labor_supply},
        n_path_steps=30,
        solve_every=7,
        use_optimizer=False,
        progressbar=False,
    )["path"]
    checkpoints = path["converged"]["checkpoint"].values

    # Summing seven path steps of this size misses these values in the last digits.
    np.testing.assert_array_equal(checkpoints, [7, 14, 21, 28, 30])
    np.testing.assert_array_equal(
        path["parameters"].dataset["L_s"].isel(step=checkpoints),
        labor_supply + (1.1 * labor_supply - labor_supply) * checkpoints / 30,
    )


def test_solve_every_continues_from_the_path_point_after_a_failed_checkpoint():
    mod = load_and_cache_model(1, functions_to_compile=("path", "root"))
    initial_state = calibrate_model_1(**model_1_data)
    scenario = {"final_delta_pct": {"L_s": 1.5}, "n_path_steps": 30, "use_optimizer": False}

    plain = mod.simulate(initial_state, progressbar=False, **scenario)["path"]
    with pytest.warns(UserWarning, match="did not converge at path steps 10, 20, 30"):
        failed = mod.simulate(
            initial_state, solve_every=10, equilibrium_tol=0.0, progressbar=False, **scenario
        )["path"]

    np.testing.assert_array_equal(failed["converged"], [False, False, False])
    xr.testing.assert_allclose(
        failed["variables"].to_dataset(inherit=False),
        plain["variables"].to_dataset(inherit=False),
        rtol=1e-10,
    )


def test_solve_every_accepts_a_numpy_integer():
    mod = load_and_cache_model(1, functions_to_compile=("path", "root"))
    initial_state = calibrate_model_1(**model_1_data)

    path = mod.simulate(
        initial_state,
        final_delta_pct={"L_s": 1.5},
        n_path_steps=20,
        solve_every=np.int64(10),
        use_optimizer=False,
        progressbar=False,
    )["path"]

    np.testing.assert_array_equal(path["converged"]["checkpoint"], [10, 20])


def test_solve_every_with_root_warns_about_use_hess_only_for_the_final_solve(caplog):
    mod = load_and_cache_model(1, functions_to_compile=("path", "root"))
    initial_state = calibrate_model_1(**model_1_data)

    with caplog.at_level(logging.WARNING, logger="cge_modeling"):
        mod.simulate(
            initial_state,
            final_delta_pct={"L_s": 1.5},
            n_path_steps=40,
            solve_every=10,
            use_hess=True,
            progressbar=False,
        )

    assert caplog.text.count('The "use_hess" argument is not used') == 1


def test_saved_simulation_names_record_checkpoints_and_the_optimizer_that_solves_them(tmp_path):
    mod = load_and_cache_model(1, functions_to_compile=("path", "root"))
    initial_state = calibrate_model_1(**model_1_data)
    scenario = {
        "final_delta_pct": {"L_s": 1.5},
        "n_path_steps": 20,
        "progressbar": False,
        "save_results": True,
        "output_path": str(tmp_path),
    }

    mod.simulate(initial_state, **scenario)
    mod.simulate(initial_state, solve_every=10, use_optimizer=False, **scenario)
    names = sorted(path.name for path in tmp_path.iterdir())

    assert len(names) == 2
    assert any("_Path[rk4,steps=20]_root[" in name for name in names)
    assert any("_Path[rk4,steps=20,solve_every=10]_root[" in name for name in names)


def test_simulate_takes_euler_steps_when_compile_kwargs_ask_for_them(tmp_path):
    mod = build_model_1()
    initial_state = calibrate_model_1(**model_1_data)

    result = mod.simulate(
        initial_state,
        final_delta_pct={"L_s": 1.5},
        n_path_steps=20,
        use_optimizer=False,
        compile_kwargs={"functions_to_compile": ["path"], "step_method": "euler"},
        progressbar=False,
        save_results=True,
        output_path=str(tmp_path),
    )

    assert mod.f_path.keywords["step_method"] == "euler"
    assert result["path"]["variables"].sizes["step"] == 21
    assert [
        name
        for name in (path.name for path in tmp_path.iterdir())
        if "_Path[euler,steps=20]" in name
    ]


def test_compile_rejects_an_unknown_step_method():
    with pytest.raises(ValueError, match="Unknown step_method 'RK4'"):
        build_model_1().compile(functions_to_compile=["path"], step_method="RK4")


def test_path_approximation_rejects_an_unknown_keyword_argument():
    mod = load_and_cache_model(1, functions_to_compile=("path",))
    initial_state = calibrate_model_1(**model_1_data)
    _, theta_final = variable_dict_to_flat_array(initial_state, mod.variables, mod.parameters)

    with pytest.raises(TypeError, match="not_an_option"):
        mod._solve_with_path(
            initial_state, theta_final, n_steps=10, progressbar=False, not_an_option=1
        )


@pytest.mark.parametrize(
    "options, message",
    [
        ({"solve_every": 0}, "solve_every must be a positive number of steps"),
        ({"solve_every": 2.5}, "solve_every must be a positive number of steps"),
        ({"solve_every": True}, "solve_every must be a positive number of steps"),
        ({"optimizer_mode": "Root"}, "Unknown optimizer mode Root"),
        ({"n_path_steps": 0}, "n_path_steps must be a positive number of steps"),
    ],
    ids=["zero_steps", "fractional_steps", "boolean_steps", "unknown_mode", "zero_path_steps"],
)
def test_simulate_rejects_invalid_checkpoint_options_before_compiling(options, message):
    mod = build_model_1()
    initial_state = calibrate_model_1(**model_1_data)

    with pytest.raises(ValueError, match=message):
        mod.simulate(
            initial_state,
            final_delta_pct={"L_s": 1.5},
            use_optimizer=False,
            **{"solve_every": 10, **options},
        )

    assert not mod._compiled


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
    mod = load_and_cache_model(model_id=1, functions_to_compile=("root", "path"))
    initial_state = calibrate_model_1(**model_1_data)

    with pytest.raises(
        ValueError, match="initial_state must be a dict of initial values or a DataTree"
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

    res = mod.simulate(initial_state, final_values={"L_s": 1500}, n_path_steps=100)

    assert list(mod._compiled) == ["path", "root"]
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
        ("all", ["root", "minimize", "path"]),
        (True, ["root", "minimize", "path"]),
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
        ("all", ["root", "minimize", "path"]),
        (("root",), ["root"]),
        (("minimize",), ["minimize"]),
        (("path",), ["path"]),
        (None, []),
    ],
    ids=["all", "root_only", "minimize_only", "path_only", "none"],
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
    compiled_path = "path" in expected_compiled_list

    assert (model.f_jac is not None) == compiled_root
    for func in [model.f_resid, model.f_grad, model.f_hess, model.f_hessp]:
        assert (func is not None) == compiled_minimize
    assert (model.f_path is not None) == compiled_path


def check_function_signature(func, expected_input_names):
    input_names = [x.name for x in func.maker.fgraph.inputs]
    assert input_names == expected_input_names


def test_compiled_functions_take_packed_inputs():
    model = load_and_cache_model(model_id=1, mode="FAST_RUN", functions_to_compile="all")

    for func in [model.f_system, model.f_jac, model.f_resid, model.f_grad, model.f_hess]:
        check_function_signature(func, ["x", "theta"])
    check_function_signature(model.f_hessp, ["x", "p", "theta"])
    check_function_signature(model.f_path.keywords["f_tangent"], ["x", "theta", "theta_step"])


def test_vectorized_and_scanned_jacobians_agree():
    vectorized = load_and_cache_model(model_id=1, functions_to_compile="all")
    scanned = load_and_cache_model(model_id=1, functions_to_compile="all", vectorize_jacobian=False)
    x, theta = variable_dict_to_flat_array(
        calibrate_model_1(**model_1_data), vectorized.variables, vectorized.parameters
    )

    np.testing.assert_allclose(vectorized.f_jac(x, theta), scanned.f_jac(x, theta))
    np.testing.assert_allclose(
        vectorized.f_path.keywords["f_tangent"](x, theta, 0.01 * theta),
        scanned.f_path.keywords["f_tangent"](x, theta, 0.01 * theta),
        atol=1e-10,
    )
    assert not any(isinstance(node.op, Scan) for node in vectorized.f_jac.maker.fgraph.apply_nodes)
    assert any(isinstance(node.op, Scan) for node in scanned.f_jac.maker.fgraph.apply_nodes)


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
        use_path_approximation=False,
        progressbar=False,
    )

    assert model._compiled == ["root"]
    assert model.f_jac is not None
    assert model.f_path is None
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
