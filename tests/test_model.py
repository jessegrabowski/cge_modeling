import copy
import logging

import numpy as np
import pytest
import xarray.testing as xr_testing

from pytensor.graph import rewrite_graph
from pytensor.graph.traversal import explicit_graph_inputs

from cge_modeling import CGEModel
from cge_modeling import model as cm
from cge_modeling.base.utilities import variable_dict_to_flat_array
from tests.utilities.models import (
    build_model_1,
    build_model_2,
    build_model_3,
    build_model_4,
    build_model_5,
    calibrate_model_1,
    calibrate_model_2,
    calibrate_model_3,
    calibrate_model_4,
    calibrate_model_5,
    load_and_cache_model,
    model_1_data,
    model_2_data,
    model_3_data,
    model_4_data,
    model_5_data,
)


def evaluate(xtensor, values):
    """Evaluate ``xtensor`` given values for model objects, which lowering replaces by their tensors."""
    lowered = rewrite_graph(xtensor.values, include=("lower_xtensor",))
    return lowered.eval({obj.owner.inputs[0]: value for obj, value in values.items()})


@pytest.mark.parametrize(
    "build, calibrate, data",
    [
        (build_model_1, calibrate_model_1, model_1_data),
        (build_model_2, calibrate_model_2, model_2_data),
    ],
    ids=["model_1", "model_2"],
)
def test_context_built_model_residuals_vanish_at_the_calibrated_point(build, calibrate, data):
    model = build()
    calibrated = calibrate(**data)

    for equation in model.equations:
        lowered = rewrite_graph(equation.residual.values, include=("lower_xtensor",))
        inputs = {
            tensor: np.asarray(calibrated[tensor.name], dtype=float)
            for tensor in explicit_graph_inputs([lowered])
        }
        np.testing.assert_allclose(lowered.eval(inputs), 0.0, atol=1e-8, err_msg=equation.name)


@pytest.mark.parametrize(
    "build, calibrate, data",
    [
        (build_model_1, calibrate_model_1, model_1_data),
        (build_model_2, calibrate_model_2, model_2_data),
    ],
    ids=["model_1", "model_2"],
)
@pytest.mark.parametrize(
    "optimizer_mode, optimizer_kwargs",
    [("root", {}), ("minimize", {"tol": 1e-14})],
    ids=["root", "minimize"],
)
def test_labor_supply_shock_reaches_an_equilibrium(
    build, calibrate, data, optimizer_mode, optimizer_kwargs
):
    model = build()
    calibrated = calibrate(**data)

    result = model.simulate(
        calibrated,
        final_delta_pct={"L_s": 0.5},
        n_iter_euler=100,
        optimizer_mode=optimizer_mode,
        verbose=False,
        **optimizer_kwargs,
    )
    final = {
        name: values.values
        for group in ("variables", "parameters")
        for name, values in result["optimizer"][group].dataset.isel(step=-1).data_vars.items()
    }
    x, theta = variable_dict_to_flat_array(final, model.variables, model.parameters)

    assert result["optim_res"].success
    np.testing.assert_allclose(final["L_s"], 0.5 * calibrated["L_s"])
    np.testing.assert_allclose(model.f_system(x, theta), 0.0, atol=1e-8)


@pytest.mark.parametrize(
    "optimizer_mode, optimizer_kwargs",
    [("root", {}), ("minimize", {"tol": 1e-14})],
    ids=["root", "minimize"],
)
def test_sweep_solves_each_value_like_simulate(optimizer_mode, optimizer_kwargs):
    model = build_model_1()
    calibrated = calibrate_model_1(**model_1_data)
    labor_supply = float(calibrated["L_s"])

    sweep = model.sweep(
        calibrated,
        parameter="L_s",
        values=[1.5 * labor_supply, 0.5 * labor_supply, labor_supply, 0.75 * labor_supply],
        optimizer_mode=optimizer_mode,
        progressbar=False,
        **optimizer_kwargs,
    )
    simulated = model.simulate(
        calibrated,
        final_delta_pct={"L_s": 0.5},
        optimizer_mode=optimizer_mode,
        progressbar=False,
        verbose=False,
        **optimizer_kwargs,
    )["optimizer"]["variables"].dataset.isel(step=-1)

    np.testing.assert_allclose(sweep["L_s"], [0.5, 0.75, 1.0, 1.5] * np.array(labor_supply))
    assert sweep["converged"].all()
    for name in model.variable_names:
        np.testing.assert_allclose(
            sweep["variables"][name].sel(L_s=labor_supply), calibrated[name], atol=1e-8
        )
        np.testing.assert_allclose(
            sweep["variables"][name].sel(L_s=0.5 * labor_supply), simulated[name], atol=1e-6
        )


def test_sweep_keeps_a_point_that_fails_and_solves_the_others():
    model = build_model_1()
    calibrated = calibrate_model_1(**model_1_data)
    labor_supply = float(calibrated["L_s"])
    feasible = [0.5 * labor_supply, 1.5 * labor_supply]

    with pytest.warns(UserWarning, match=f"did not converge at L_s={-labor_supply:g}\\."):
        sweep = model.sweep(
            calibrated,
            parameter="L_s",
            values=[-labor_supply, *feasible],
            progressbar=False,
        )
    reference = model.sweep(calibrated, parameter="L_s", values=feasible, progressbar=False)

    np.testing.assert_array_equal(sweep["converged"], [False, True, True])
    xr_testing.assert_allclose(
        sweep["variables"].to_dataset().sel(L_s=feasible), reference["variables"].to_dataset()
    )


@pytest.mark.parametrize(
    "build, name, message",
    [
        (build_model_1, "Y", "Y is not a parameter of the model"),
        (build_model_2, "alpha", "Only scalar parameters can be swept"),
    ],
    ids=["variable", "indexed_parameter"],
)
def test_sweep_rejects_names_that_are_not_scalar_parameters(build, name, message):
    model = build()

    with pytest.raises(ValueError, match=message):
        model.sweep({}, parameter=name, values=[1.0], progressbar=False)


def test_value_form_demands_keep_the_walrasian_residual_at_zero_along_the_euler_path():
    context_model = build_model_1()
    calibrated = calibrate_model_1(**model_1_data)
    scenario = {
        "final_delta_pct": {"L_s": 0.5},
        "n_iter_euler": 25,
        "optimizer_mode": "root",
        "verbose": False,
    }

    context_euler = context_model.simulate(calibrated, **scenario)["euler"]["variables"].dataset

    np.testing.assert_allclose(context_euler["resid"], 0.0, atol=1e-9)


def test_model_3_is_square_and_holds_at_the_calibrated_point():
    model = build_model_3()
    calibrated = calibrate_model_3(**model_3_data)

    assert model.n_variables == model.n_equations == 39
    np.testing.assert_allclose(calibrated["gamma"].sum() + calibrated["gamma_E"].sum(), 1.0)
    for equation in model.equations:
        lowered = rewrite_graph(equation.residual.values, include=("lower_xtensor",))
        inputs = {
            tensor: np.asarray(calibrated[tensor.name], dtype=float)
            for tensor in explicit_graph_inputs([lowered])
        }
        np.testing.assert_allclose(lowered.eval(inputs), 0.0, atol=1e-10, err_msg=equation.name)


def test_model_3_labor_supply_shock_clears_every_market():
    model = build_model_3()
    calibrated = calibrate_model_3(**model_3_data)

    result = model.simulate(
        calibrated, final_delta_pct={"L_s": 1.1}, n_iter_euler=200, verbose=False
    )
    root = result["optimizer"]["variables"].dataset.isel(step=-1, drop=True)
    euler = result["euler"]["variables"].dataset.isel(step=-1, drop=True)

    assert result["optim_res"].success
    np.testing.assert_allclose(float(root["resid"]), 0.0, atol=1e-8)
    np.testing.assert_allclose(root["P"].sel(i="agriculture"), 1.0)
    np.testing.assert_allclose(
        float(root["L_d"].sum() + root["L_d_E"].sum()), 1.1 * calibrated["L_s"]
    )
    assert float(root["w"]) < 1.0 < float(root["r"])
    xr_testing.assert_allclose(euler.drop_vars("resid"), root.drop_vars("resid"), rtol=1e-4)


@pytest.mark.parametrize("closure", ["savings_driven", "fixed_investment"])
def test_model_4_is_square_and_holds_at_the_calibrated_point(closure):
    model = build_model_4(closure=closure)
    calibrated = calibrate_model_4(**model_4_data)

    assert model.n_variables == model.n_equations == 19
    for equation in model.equations:
        lowered = rewrite_graph(equation.residual.values, include=("lower_xtensor",))
        inputs = {
            tensor: np.asarray(calibrated[tensor.name], dtype=float)
            for tensor in explicit_graph_inputs([lowered])
        }
        np.testing.assert_allclose(lowered.eval(inputs), 0.0, atol=1e-12, err_msg=equation.name)


def test_model_4_calibration_reproduces_the_published_base_year():
    calibrated = calibrate_model_4(**model_4_data)
    published = {
        "at": 2.22,
        "bt": 0.77,
        "aq": 1.97,
        "bq": 0.38,
        "E": 0.33,
        "M": 0.50,
        "Ds": 0.67,
        "Qs": 1.18,
        "Tax": 0.20,
        "Y": 1.13,
        "S": 0.27,
        "Cn": 0.83,
        "Pt": 1.08,
        "ts": 0.08,
        "Z": 0.25,
        "Sg": -0.01,
    }

    for name, value in published.items():
        np.testing.assert_allclose(calibrated[name], value, atol=0.005, err_msg=name)


def test_model_4_doubling_the_exchange_rate_doubles_prices_and_leaves_quantities():
    model = build_model_4(closure="savings_driven")
    calibrated = calibrate_model_4(**model_4_data)

    result = model.simulate(
        calibrated, final_values={"Er": 2.0}, n_iter_euler=10, optimizer_mode="root", verbose=False
    )
    root = result["optimizer"]["variables"].dataset.isel(step=-1, drop=True)

    assert result["optim_res"].success
    for name in ["E", "M", "Ds", "Dd", "Qs", "Qd", "Cn", "Z"]:
        np.testing.assert_allclose(float(root[name]), calibrated[name], rtol=1e-8, err_msg=name)
    for name in ["Pe", "Pm", "Pd", "Pt", "Px", "Pq", "Tax", "Y", "S", "Sg"]:
        np.testing.assert_allclose(float(root[name]), 2 * calibrated[name], rtol=1e-8, err_msg=name)


def test_model_4_tariff_cut_reproduces_the_published_sales_tax_increase():
    model = build_model_4(closure="fixed_investment")
    calibrated = calibrate_model_4(**model_4_data)
    published = {
        "E": 0.33,
        "M": 0.51,
        "Ds": 0.67,
        "Dd": 0.67,
        "Qs": 1.18,
        "Qd": 1.18,
        "Tax": 0.19,
        "Y": 1.10,  # Table 6.9 prints 0.10; its Cur/Base ratio of 0.97 on a base of 1.13 gives 1.10.
        "S": 0.26,
        "Cn": 0.83,
        "Pm": 0.93,
        "Pe": 1.00,
        "Pt": 1.05,
        "Pq": 0.95,
        "Px": 0.97,
        "Pd": 0.96,
        "ts": 0.11,
        "Sg": -0.01,
    }

    result = model.simulate(
        calibrated, final_values={"tm": 0.05}, n_iter_euler=50, optimizer_mode="root", verbose=False
    )
    root = result["optimizer"]["variables"].dataset.isel(step=-1, drop=True)

    assert result["optim_res"].success
    for name, value in published.items():
        np.testing.assert_allclose(float(root[name]), value, atol=0.005, err_msg=name)


def residuals_at_random_point(model):
    rng = np.random.default_rng(seed=0)
    point = {
        name: rng.uniform(0.5, 1.5) for name in sorted(model.variable_names + model.parameter_names)
    }
    residuals = {}
    for equation in model.equations:
        lowered = rewrite_graph(equation.residual.values, include=("lower_xtensor",))
        residuals[equation.name] = lowered.eval(
            {tensor: point[tensor.name] for tensor in explicit_graph_inputs([lowered])}
        )
    return residuals


def test_copy_is_an_independent_uncompiled_model_with_the_same_equations():
    model = build_model_4(closure="savings_driven")
    expected = residuals_at_random_point(model)

    for clone in [model.copy(), copy.copy(model), copy.deepcopy(model)]:
        assert clone["Z"] is not model["Z"]
        assert clone.variable_names == model.variable_names
        assert clone.parameter_names == model.parameter_names
        assert clone.f_system is None

        actual = residuals_at_random_point(clone)
        for name in model.equation_names:
            np.testing.assert_allclose(actual[name], expected[name], err_msg=name)

    with model.copy() as clone:
        cm.Variable("extra")
    assert "extra" not in model.variable_names


def test_reassign_roles_matches_the_model_built_with_those_roles():
    savings_driven = build_model_4(closure="savings_driven")
    fixed_investment = build_model_4(closure="fixed_investment")

    swapped = savings_driven.copy().reassign_roles(to_parameters=["Z"], to_variables=["ts"])

    assert sorted(swapped.variable_names) == sorted(fixed_investment.variable_names)
    assert sorted(swapped.parameter_names) == sorted(fixed_investment.parameter_names)
    assert "Z" in savings_driven.variable_names

    expected = residuals_at_random_point(fixed_investment)
    actual = residuals_at_random_point(swapped)
    for name in fixed_investment.equation_names:
        np.testing.assert_allclose(actual[name], expected[name], err_msg=name)


def test_reassign_roles_discards_the_compiled_functions():
    with CGEModel() as model:
        x = cm.Variable("x")
        a = cm.Parameter("a")
        cm.Equation("x equation", x, a)
    model.compile(functions_to_compile="root")

    model.reassign_roles(to_parameters=["x"], to_variables=["a"])

    assert model.f_system is None
    assert model.variable_names == ["a"]
    assert model.parameter_names == ["x"]


@pytest.mark.parametrize(
    "to_parameters, to_variables, message",
    [
        (["ts"], [], r"Only variables can become parameters, and \['ts'\] are not variables"),
        ([], ["Z"], r"Only parameters can become variables, and \['Z'\] are not parameters"),
        (["Z"], [], r"would not be square: \['Z'\] have 1 entries, but \[\] have 0"),
        (["Z", "Z"], ["ts", "tm"], r"reassigned once, and \['Z'\] appear more than once"),
    ],
    ids=["parameter_to_parameter", "variable_to_variable", "unequal_entries", "repeated_name"],
)
def test_reassign_roles_rejects_an_invalid_reassignment_and_leaves_the_model_unchanged(
    to_parameters, to_variables, message
):
    model = build_model_4(closure="savings_driven")
    variable_names = model.variable_names

    with pytest.raises(ValueError, match=message):
        model.reassign_roles(to_parameters=to_parameters, to_variables=to_variables)
    assert model.variable_names == variable_names


def test_reassign_roles_accepts_single_names():
    model = build_model_4(closure="savings_driven")

    model.reassign_roles(to_parameters="Z", to_variables="ts")

    assert "ts" in model.variable_names
    assert "Z" in model.parameter_names


def test_reassigned_indexed_parameter_is_solved_for():
    with CGEModel(coords={"i": ["a", "b"]}) as model:
        C = cm.Variable("C", dims="i")
        S = cm.Variable("S")
        Y = cm.Parameter("Y")
        s = cm.Parameter("s")
        g = cm.Parameter("g", dims="i")
        cm.Equation("Saving", S, s * Y)
        cm.Equation("Consumption", C, g * (1 - s) * Y)
    benchmark = {
        "C": np.array([3.0, 6.0]),
        "S": 1.0,
        "Y": 10.0,
        "s": 0.1,
        "g": np.array([1 / 3, 2 / 3]),
    }

    model.reassign_roles(to_parameters="C", to_variables="g").compile()
    result = model.simulate(
        initial_state=benchmark, final_values={"C": np.array([4.5, 4.5])}, progressbar=False
    )

    assert result["optim_res"].success
    np.testing.assert_allclose(result["optimizer"]["variables"]["g"].isel(step=-1), [0.5, 0.5])


def test_reassign_roles_counts_the_entries_of_indexed_objects():
    with CGEModel(coords={"i": ["a", "b"]}) as model:
        S = cm.Variable("S")
        g = cm.Parameter("g", dims="i")
        cm.Equation("Saving", S, g.sum())

    with pytest.raises(ValueError, match=r"\['S'\] have 1 entries, but \['g'\] have 2"):
        model.reassign_roles(to_parameters="S", to_variables="g")


def test_model_5_is_square_and_holds_at_the_calibrated_point():
    model = build_model_5()
    calibrated = calibrate_model_5(**model_5_data)

    assert model.n_variables == model.n_equations == 59
    for equation in model.equations:
        lowered = rewrite_graph(equation.residual.values, include=("lower_xtensor",))
        inputs = {
            tensor: np.asarray(calibrated[tensor.name], dtype=float)
            for tensor in explicit_graph_inputs([lowered])
        }
        np.testing.assert_allclose(lowered.eval(inputs), 0.0, atol=1e-12, err_msg=equation.name)


def test_model_5_calibration_reproduces_the_published_parameters():
    calibrated = calibrate_model_5(**model_5_data)
    manufactures_productivity = calibrated["A"][1] / calibrated["psi_VA"][1]
    primaries_productivity = calibrated["A"][0] / calibrated["psi_VA"][0]

    np.testing.assert_allclose(calibrated["a"][0, 1], 4 / 35)
    np.testing.assert_allclose(calibrated["t"][1], 1 / 35)
    np.testing.assert_allclose(calibrated["alpha"][1], 5 / 7)
    np.testing.assert_allclose(calibrated["tau"], 2 / 46)
    np.testing.assert_allclose(calibrated["theta"][0], 2 / 51)
    np.testing.assert_allclose(calibrated["theta_G"], 4 / 51)
    np.testing.assert_allclose(manufactures_productivity, 35 * 10 ** (-5 / 7) * 4 ** (-2 / 7))
    np.testing.assert_allclose(primaries_productivity, 8 * 4 ** (-4 / 5))


def test_model_5_abolishing_the_manufactures_tax_clears_every_market():
    model = build_model_5()
    calibrated = calibrate_model_5(**model_5_data)
    tax_rates = calibrated["t"] * np.array([1.0, 0.0, 1.0])

    result = model.simulate(
        calibrated,
        final_values={"t": tax_rates},
        n_iter_euler=20,
        optimizer_mode="root",
        verbose=False,
    )
    root = result["optimizer"]["variables"].dataset.isel(step=-1, drop=True)
    euler = result["euler"]["variables"].dataset.isel(step=-1, drop=True)
    P = root["P"]
    price_index = float(
        (calibrated["theta"] * P).sum()
        + calibrated["theta_M"] * root["P_M"]
        + calibrated["theta_I"] * root["P_I"]
        + calibrated["theta_G"] * root["P_G"]
    )

    assert result["optim_res"].success
    np.testing.assert_allclose(float(root["resid"]), 0.0, atol=1e-10)
    np.testing.assert_allclose(float(euler["resid"]), 0.0, atol=1e-10)
    np.testing.assert_allclose(price_index, 1.0)
    np.testing.assert_allclose(float(root["K_d"].sum()), calibrated["K_s"])
    assert float(P.sel(i="manufactures")) < min(
        float(P.sel(i="primaries")), float(P.sel(i="services"))
    )
    assert float(root["T"]) < calibrated["T"]


@pytest.mark.parametrize(
    "build, calibrate, data, unknown",
    [
        (
            build_model_1,
            calibrate_model_1,
            model_1_data,
            ["Y", "L_d", "K_d", "C", "income", "resid", "alpha", "A"],
        ),
        (
            build_model_2,
            calibrate_model_2,
            model_2_data,
            ["U", "psi_VA", "psi_VC", "psi_X", "alpha", "A", "gamma"],
        ),
        (
            build_model_3,
            calibrate_model_3,
            model_3_data,
            [
                "psi_VA",
                "psi_VC",
                "psi_X",
                "psi_X_EN",
                "alpha",
                "A",
                "psi_VA_E",
                "psi_VC_E",
                "psi_X_E",
                "alpha_E",
                "A_E",
                "gamma",
                "gamma_E",
            ],
        ),
        (
            lambda: build_model_4(closure="savings_driven"),
            calibrate_model_4,
            model_4_data,
            ["wm", "we", "sy", "B", "at", "bt", "aq", "bq"],
        ),
    ],
    ids=["model_1", "model_2", "model_3", "model_4"],
)
def test_calibrate_recovers_the_hand_written_calibration(build, calibrate, data, unknown):
    model = build()
    expected = calibrate(**data)
    given = {
        name: expected[name]
        for name in [*model.variable_names, *model.parameter_names]
        if name not in unknown
    }

    result = model.calibrate(data=given)

    assert result.success
    np.testing.assert_allclose(
        result.fitted_values,
        np.concatenate(variable_dict_to_flat_array(expected, model.variables, model.parameters)),
        rtol=1e-10,
        atol=1e-12,
    )


def test_calibrate_flags_data_that_disagree_with_the_model():
    model = build_model_1()
    expected = calibrate_model_1(**model_1_data)

    result = model.calibrate(data={**model_1_data, "w": 1.0, "C": 1.1 * expected["C"]})

    assert not result.success


def test_calibrate_without_data_names_the_entries_no_equation_determines():
    model = build_model_1()

    with pytest.raises(ValueError, match="No equation determines"):
        model.calibrate()


def test_equation_table_shows_the_latex_of_context_built_equations(monkeypatch):
    shown = {}
    monkeypatch.setattr("cge_modeling.base.cge.display_latex_table", shown.update)
    with CGEModel(coords={"i": ["a", "b"]}) as model:
        P = cm.Variable("P", dims="i")
        P_bar = cm.Parameter("P_bar", latex_name=r"\bar{P}")
        cm.Equation("Price of <dim:i>", P, P_bar)

    model.equation_table()

    assert shown[""] == [1]
    assert shown["Name"] == ["Price of i"]
    assert [equation.data for equation in shown["Equation"]] == [r"P_{i} = \bar{P}"]


def test_summary_shows_the_symbols_and_descriptions_of_the_requested_objects(monkeypatch):
    shown = {}
    monkeypatch.setattr("cge_modeling.base.cge.display_latex_table", shown.update)
    with CGEModel(coords={"i": ["a", "b"]}) as model:
        P = cm.Variable("P", dims="i", description="Price of <dim:i>")
        P_bar = cm.Parameter("P_bar", description="Reference price")
        cm.Equation("Price of <dim:i>", P, P_bar)

    model.summary("variables")

    assert [symbol.name for symbol in shown["Symbol"]] == ["P"]
    assert shown["Description"] == ["Price of i"]


def test_check_for_equilibrium_names_the_failing_elements_of_a_context_built_model(caplog):
    with CGEModel(coords={"i": ["Agriculture", "Industry"]}) as model:
        P = cm.Variable("P", dims="i")
        P_bar = cm.Parameter("P_bar")
        cm.Equation("Price of <dim:i>", P, P_bar)
    model.compile(functions_to_compile=("root",))

    with caplog.at_level(logging.INFO, logger="cge_modeling"):
        model.check_for_equilibrium({"P": np.array([1.0, 3.0]), "P_bar": 1.0})

    assert "Equilibrium not found" in caplog.text
    assert "Price of Industry" in caplog.text
    assert "Price of Agriculture" not in caplog.text


def test_equation_repeating_an_existing_element_name_raises():
    with pytest.raises(ValueError, match=r"repeats existing equation element names \['Price a'\]"):
        with CGEModel(coords={"i": ["a", "b"]}):
            P = cm.Variable("P", dims="i")
            cm.Equation("Price <dim:i>", P, 1.0)
            cm.Equation("Price a", P.sel(i="a"), 1.0)


def test_equation_residual_is_lhs_minus_rhs_in_coords_dim_order():
    with CGEModel(coords={"i": ["a", "b"], "j": ["x", "y", "z"]}):
        w = cm.Variable("w", dims="j")
        m = cm.Parameter("m", dims=("i", "j"))
        broadcast = cm.Equation("broadcast", w, m)

    m_values = np.arange(6.0).reshape(2, 3)
    w_values = np.array([10.0, 20.0, 30.0])
    residual = evaluate(broadcast.residual, {w: w_values, m: m_values})

    assert broadcast.residual.type.dims == ("i", "j")
    np.testing.assert_allclose(residual, w_values - m_values)


def test_equation_built_with_double_equals_raises():
    with pytest.raises(TypeError, match="`==` does not build equations"):
        with CGEModel(coords={"i": ["a", "b"]}):
            P = cm.Variable("P", dims="i")
            Q = cm.Variable("Q", dims="i")
            cm.Equation("clearing", P * 2 == Q, 0.0)


def test_object_registration_needs_an_active_model():
    with pytest.raises(RuntimeError, match="No active model"):
        cm.Variable("Y")


@pytest.mark.parametrize(
    "register, message",
    [
        (lambda: cm.Parameter("Y"), "already has a variable or parameter named 'Y'"),
        (lambda: cm.Variable("Z", dims="k"), r"dims \['k'\] that are not in the model coords"),
        (lambda: cm.Variable("i"), "'i' conflicts with an existing dimension name"),
    ],
    ids=["duplicate_name", "unknown_dim", "dim_name"],
)
def test_invalid_object_registration_raises(register, message):
    with pytest.raises(ValueError, match=message):
        with CGEModel(coords={"i": ["a"]}):
            cm.Variable("Y")
            register()


@pytest.mark.parametrize(
    "lhs, rhs, message",
    [
        (
            lambda P: P.rename(i="k"),
            lambda P: 1.0,
            r"dims \['k'\] that are not in the model coords",
        ),
        (lambda P: 1.0, lambda P: 2.0, "uses no variables or parameters of the model"),
    ],
    ids=["unknown_dim", "no_model_objects"],
)
def test_invalid_equation_raises(lhs, rhs, message):
    with pytest.raises(ValueError, match=message):
        with CGEModel(coords={"i": ["a", "b"]}):
            P = cm.Variable("P", dims="i")
            cm.Equation("bad", lhs(P), rhs(P))


def test_equation_using_another_models_object_raises():
    with pytest.raises(ValueError, match="uses inputs that are not objects of this model"):
        with CGEModel():
            outer_variable = cm.Variable("outer_variable")
            with CGEModel():
                inner_variable = cm.Variable("inner_variable")
                cm.Equation("inner", inner_variable, outer_variable)


def test_model_can_be_built_across_several_blocks():
    model = CGEModel(coords={"i": ["a", "b"]})

    with model:
        P = cm.Variable("P", dims="i")
    with model:
        cm.Equation("prices", P, 1.0)

    assert model.n_variables == model.n_equations == 2


def test_objects_register_on_an_explicit_model_outside_a_block():
    model = CGEModel(coords={"i": ["a", "b"]})

    P = cm.Variable("P", dims="i", model=model)
    cm.Equation("prices", P, 1.0, model=model)

    assert model.variable_names == ["P"]
    assert model.equation_names == ["prices"]


def test_registered_objects_are_retrievable_by_name():
    with CGEModel(coords={"i": ["a", "b"]}) as model:
        P = cm.Variable("P", dims="i")

    assert model["P"] is P
    with pytest.raises(KeyError, match="'Q' is not a variable or parameter registered"):
        model["Q"]


def test_reentering_a_model_inside_its_own_block_keeps_it_active():
    with CGEModel() as model:
        with model:
            x = cm.Variable("x")
        cm.Equation("x equation", x, 1.0)

    assert model.n_variables == model.n_equations == 1
    assert model.equation_names == ["x equation"]


def test_nested_contexts_register_on_the_innermost_model():
    with CGEModel() as outer:
        with CGEModel() as inner:
            cm.Variable("inner_variable")
        cm.Variable("outer_variable")

    assert inner.variable_names == ["inner_variable"]
    assert outer.variable_names == ["outer_variable"]
