import numpy as np
import pytest

from pytensor.graph import rewrite_graph
from pytensor.graph.traversal import explicit_graph_inputs

from cge_modeling import CGEModel, compile_model
from cge_modeling import model as cm
from cge_modeling.base.utilities import variable_dict_to_flat_array
from cge_modeling.production_functions import (
    CES,
    CESValue,
    CobbDouglas,
    DixitStiglitz,
    DixitStiglitzValue,
    Leontief,
)


def lowered(xtensor):
    return rewrite_graph(xtensor.values, include=("lower_xtensor",))


def tensor_of(model_object):
    return model_object.owner.inputs[0]


def evaluate(xtensor, objects, point):
    """Evaluate ``xtensor`` at ``point``, a mapping from object names to values."""
    graph = lowered(xtensor)
    inputs = set(explicit_graph_inputs([graph]))
    values = {tensor_of(objects[name]): value for name, value in point.items()}
    return graph.eval({tensor: value for tensor, value in values.items() if tensor in inputs})


def evaluate_residual(equation, objects, point):
    return evaluate(equation.residual, objects, point)


def cobb_douglas_block(factor_shares):
    with CGEModel(coords={"i": ["a", "b"]}) as model:
        Y = cm.Variable("Y", dims="i")
        K = cm.Variable("K", dims="i")
        L = cm.Variable("L", dims="i")
        P = cm.Variable("P", dims="i")
        r = cm.Parameter("r")
        w = cm.Parameter("w")
        alpha = cm.Parameter("alpha", dims="i")
        A = cm.Parameter("A", dims="i")
        equations = CobbDouglas(
            "Production of <dim:i>",
            output=Y,
            output_price=P,
            factors=[K, L],
            factor_prices=[r, w],
            factor_shares=factor_shares(alpha),
            tfp=A,
        )
    return (
        model,
        equations,
        {"Y": Y, "K": K, "L": L, "P": P, "r": r, "w": w, "alpha": alpha, "A": A},
    )


def cobb_douglas_equilibrium_point():
    K, alpha, A, r, w = np.array([2.0, 3.0]), np.array([0.3, 0.6]), np.array([1.7, 1.7]), 1.2, 0.8
    L = K * (1 - alpha) * r / (alpha * w)
    Y = A * K**alpha * L ** (1 - alpha)
    P = r * K / (alpha * Y)
    return {"Y": Y, "K": K, "L": L, "P": P, "r": r, "w": w, "alpha": alpha, "A": A}


def test_cobb_douglas_registers_production_then_one_demand_per_factor():
    model, equations, objects = cobb_douglas_block(lambda alpha: [alpha])

    assert [equation.name for equation in equations] == [
        "Production of <dim:i>",
        "Production of <dim:i>: demand for K",
        "Production of <dim:i>: demand for L",
    ]
    assert model.unpacked_equation_names == [
        "Production of a",
        "Production of b",
        "Production of a: demand for K",
        "Production of b: demand for K",
        "Production of a: demand for L",
        "Production of b: demand for L",
    ]

    point = cobb_douglas_equilibrium_point()
    for equation in equations:
        np.testing.assert_allclose(
            evaluate_residual(equation, objects, point), 0.0, atol=1e-12, err_msg=equation.name
        )


def test_missing_last_share_is_one_minus_the_others():
    _, implied, implied_objects = cobb_douglas_block(lambda alpha: [alpha])
    _, explicit, explicit_objects = cobb_douglas_block(lambda alpha: [alpha, 1 - alpha])

    point = cobb_douglas_equilibrium_point()
    point["P"] = point["P"] * 1.1
    for implied_equation, explicit_equation in zip(implied, explicit, strict=True):
        np.testing.assert_allclose(
            evaluate_residual(implied_equation, implied_objects, point),
            evaluate_residual(explicit_equation, explicit_objects, point),
        )


@pytest.mark.parametrize(
    "factor_prices, factor_shares, message",
    [
        (lambda r, w: [r], lambda alpha: [alpha], "one price per factor"),
        (
            lambda r, w: [r, w],
            lambda alpha: [],
            "Expected 2 factor shares, or 1 with the last implied",
        ),
    ],
    ids=["missing_price", "missing_shares"],
)
def test_cobb_douglas_rejects_mismatched_inputs(factor_prices, factor_shares, message):
    with pytest.raises(ValueError, match=message):
        with CGEModel():
            Y, K, L, P = (cm.Variable(name) for name in ("Y", "K", "L", "P"))
            r, w, alpha = (cm.Parameter(name) for name in ("r", "w", "alpha"))
            CobbDouglas(
                "Production",
                output=Y,
                output_price=P,
                factors=[K, L],
                factor_prices=factor_prices(r, w),
                factor_shares=factor_shares(alpha),
                tfp=1.0,
            )


def test_model_built_with_cobb_douglas_solves_to_the_analytic_equilibrium():
    with CGEModel() as model:
        Y, K, L, P = (cm.Variable(name) for name in ("Y", "K", "L", "P"))
        r, w, alpha, A, K_s = (cm.Parameter(name) for name in ("r", "w", "alpha", "A", "K_s"))
        CobbDouglas(
            "Production",
            output=Y,
            output_price=P,
            factors=[K, L],
            factor_prices=[r, w],
            factor_shares=[alpha],
            tfp=A,
        )
        cm.Equation("Capital supply", K, K_s)
    compile_model(model, backend="pytensor", functions_to_compile=("root",))

    parameters = {"r": 1.2, "w": 0.8, "alpha": 0.3, "A": 1.5, "K_s": 2.0}
    K_star = parameters["K_s"]
    L_star = (
        (1 - parameters["alpha"])
        * parameters["r"]
        * K_star
        / (parameters["alpha"] * parameters["w"])
    )
    Y_star = parameters["A"] * K_star ** parameters["alpha"] * L_star ** (1 - parameters["alpha"])
    P_star = parameters["r"] * K_star / (parameters["alpha"] * Y_star)
    start = {"Y": 1.0, "K": 1.0, "L": 1.0, "P": 1.0, **parameters}
    _, theta = variable_dict_to_flat_array(start, model.variables, model.parameters)

    result = model._solve_with_root(start, theta, progressbar=False, verbose=False)

    assert result.success
    np.testing.assert_allclose(result.x, [Y_star, K_star, L_star, P_star], rtol=1e-8)


def ces_block(helper=CES):
    with CGEModel(coords={"i": ["a", "b"]}) as model:
        Y = cm.Variable("Y", dims="i")
        K = cm.Variable("K", dims="i")
        L = cm.Variable("L", dims="i")
        P = cm.Variable("P", dims="i")
        r = cm.Parameter("r")
        w = cm.Parameter("w")
        alpha = cm.Parameter("alpha", dims="i")
        A = cm.Parameter("A", dims="i")
        epsilon = cm.Parameter("epsilon", dims="i")
        equations = helper(
            "Value added in <dim:i>",
            output=Y,
            output_price=P,
            factors=[K, L],
            factor_prices=[r, w],
            factor_shares=[alpha],
            tfp=A,
            epsilon=epsilon,
        )
    objects = {"Y": Y, "K": K, "L": L, "P": P, "r": r, "w": w, "alpha": alpha, "A": A}
    return model, equations, {**objects, "epsilon": epsilon}


def ces_unit_cost_point():
    alpha, A, epsilon = np.array([0.3, 0.6]), np.array([1.5, 2.0]), np.array([0.7, 1.8])
    r, w, Y = 1.2, 0.8, np.array([3.0, 5.0])
    shares = np.stack([alpha, 1 - alpha])
    prices = np.array([r, w])[:, None]
    P = (shares**epsilon * prices ** (1 - epsilon)).sum(axis=0) ** (1 / (1 - epsilon)) / A
    K, L = Y / A * (shares * P * A / prices) ** epsilon
    return {
        "Y": Y,
        "K": K,
        "L": L,
        "P": P,
        "r": r,
        "w": w,
        "alpha": alpha,
        "A": A,
        "epsilon": epsilon,
    }


@pytest.mark.parametrize("helper", [CES, CESValue], ids=["technology", "value"])
def test_ces_blocks_hold_when_the_output_price_is_the_unit_cost(helper):
    model, equations, objects = ces_block(helper)
    point = ces_unit_cost_point()

    assert model.unpacked_equation_names[:2] == ["Value added in a", "Value added in b"]
    for equation in equations:
        np.testing.assert_allclose(
            evaluate_residual(equation, objects, point), 0.0, atol=1e-12, err_msg=equation.name
        )


def test_ces_value_zero_profit_fails_off_the_unit_cost():
    _, equations, objects = ces_block(CESValue)
    point = ces_unit_cost_point()
    point["P"] = point["P"] * 1.1

    assert np.all(np.abs(evaluate_residual(equations[0], objects, point)) > 1e-3)


def dixit_stiglitz_block(helper):
    with CGEModel(coords={"v": ["a", "b", "c"]}) as model:
        Y = cm.Variable("Y")
        X = cm.Variable("X", dims="v")
        P_Y = cm.Variable("P_Y")
        P_X = cm.Parameter("P_X", dims="v")
        alpha = cm.Parameter("alpha", dims="v")
        A = cm.Parameter("A")
        epsilon = cm.Parameter("epsilon")
        equations = helper(
            "Composite good",
            output=Y,
            output_price=P_Y,
            factor=X,
            factor_price=P_X,
            factor_dim="v",
            epsilon=epsilon,
            tfp=A,
            factor_share=alpha,
        )
    return (
        model,
        equations,
        {"Y": Y, "X": X, "P_Y": P_Y, "P_X": P_X, "alpha": alpha, "A": A, "epsilon": epsilon},
    )


def dixit_stiglitz_unit_cost_point():
    alpha, A, epsilon, P_X, Y = np.array([0.2, 0.5, 0.3]), 1.4, 2.5, np.array([1.0, 1.5, 0.7]), 4.0
    P_Y = (alpha**epsilon * P_X ** (1 - epsilon)).sum() ** (1 / (1 - epsilon)) / A
    X = Y / A * (alpha * P_Y * A / P_X) ** epsilon
    return {"Y": Y, "X": X, "P_Y": P_Y, "P_X": P_X, "alpha": alpha, "A": A, "epsilon": epsilon}


@pytest.mark.parametrize("helper", [DixitStiglitz, DixitStiglitzValue], ids=["technology", "value"])
def test_dixit_stiglitz_blocks_hold_when_the_output_price_is_the_unit_cost(helper):
    model, equations, objects = dixit_stiglitz_block(helper)
    point = dixit_stiglitz_unit_cost_point()

    assert model.unpacked_equation_names == [
        "Composite good",
        "Composite good: demand for X [v=a]",
        "Composite good: demand for X [v=b]",
        "Composite good: demand for X [v=c]",
    ]
    for equation in equations:
        np.testing.assert_allclose(
            evaluate_residual(equation, objects, point), 0.0, atol=1e-12, err_msg=equation.name
        )


def test_leontief_holds_when_the_output_price_is_the_unit_cost():
    with CGEModel() as model:
        Y, X_1, X_2, P_Y = (cm.Variable(name) for name in ("Y", "X_1", "X_2", "P_Y"))
        P_1, P_2, a_1, a_2 = (cm.Parameter(name) for name in ("P_1", "P_2", "a_1", "a_2"))
        equations = Leontief(
            "Bundle",
            output=Y,
            output_price=P_Y,
            factors=[X_1, X_2],
            factor_prices=[P_1, P_2],
            factor_shares=[a_1, a_2],
        )
    objects = {
        "Y": Y,
        "X_1": X_1,
        "X_2": X_2,
        "P_Y": P_Y,
        "P_1": P_1,
        "P_2": P_2,
        "a_1": a_1,
        "a_2": a_2,
    }
    point = {"Y": 4.0, "P_1": 1.3, "P_2": 0.6, "a_1": 0.25, "a_2": 0.4}
    point |= {
        "X_1": point["a_1"] * point["Y"],
        "X_2": point["a_2"] * point["Y"],
        "P_Y": point["a_1"] * point["P_1"] + point["a_2"] * point["P_2"],
    }

    assert model.equation_names == ["Bundle", "Bundle: demand for X_1", "Bundle: demand for X_2"]
    for equation in equations:
        np.testing.assert_allclose(
            evaluate_residual(equation, objects, point), 0.0, atol=1e-12, err_msg=equation.name
        )


def test_leontief_zero_profit_fails_off_the_unit_cost():
    with CGEModel():
        Y, X_1, X_2, P_Y = (cm.Variable(name) for name in ("Y", "X_1", "X_2", "P_Y"))
        P_1, P_2, a_1, a_2 = (cm.Parameter(name) for name in ("P_1", "P_2", "a_1", "a_2"))
        zero_profit, *_ = Leontief(
            "Bundle",
            output=Y,
            output_price=P_Y,
            factors=[X_1, X_2],
            factor_prices=[P_1, P_2],
            factor_shares=[a_1, a_2],
        )
    objects = {"Y": Y, "X_1": X_1, "X_2": X_2, "P_Y": P_Y, "P_1": P_1, "P_2": P_2}
    point = {"Y": 4.0, "X_1": 1.0, "X_2": 1.6, "P_1": 1.3, "P_2": 0.6}
    unit_cost = 0.25 * point["P_1"] + 0.4 * point["P_2"]

    assert abs(evaluate_residual(zero_profit, objects, point | {"P_Y": unit_cost * 1.1})) > 1e-3


def test_leontief_over_an_input_output_matrix_reproduces_the_model_2_bundle():
    sectors = ["0", "1", "2"]
    with CGEModel(coords={"i": sectors, "j": sectors}):
        X = cm.Variable("X", dims=("i", "j"))
        VC = cm.Variable("VC", dims="i")
        P_VC = cm.Variable("P_VC", dims="i")
        P = cm.Parameter("P", dims="i")
        psi_X = cm.Parameter("psi_X", dims=("i", "j"))
        zero_profit, demand = Leontief(
            "Intermediate bundle",
            output=VC.rename(i="j"),
            output_price=P_VC.rename(i="j"),
            factors=[X],
            factor_prices=[P],
            factor_shares=[psi_X],
            factor_dim="i",
        )
    objects = {"X": X, "VC": VC, "P_VC": P_VC, "P": P, "psi_X": psi_X}
    rng = np.random.default_rng(2)
    point = {
        "X": rng.uniform(1, 2, (3, 3)),
        "VC": rng.uniform(1, 2, 3),
        "P_VC": rng.uniform(1, 2, 3),
        "P": rng.uniform(1, 2, 3),
        "psi_X": rng.uniform(0.1, 0.5, (3, 3)),
    }
    inline_zero_profit = VC * P_VC - (P * X).sum("i").rename(j="i")
    inline_demand = X - psi_X * VC.rename(i="j")

    for equation, inline in [(zero_profit, inline_zero_profit), (demand, inline_demand)]:
        np.testing.assert_allclose(
            evaluate_residual(equation, objects, point),
            evaluate(inline, objects, point),
            err_msg=equation.name,
        )


def test_leontief_sums_each_factor_over_its_own_input_dim():
    with CGEModel(coords={"i": ["a", "b"], "j": ["a", "b"], "k": ["energy"]}):
        X = cm.Variable("X", dims=("i", "j"))
        X_E = cm.Variable("X_E", dims=("k", "j"))
        VC = cm.Variable("VC", dims="j")
        P_VC = cm.Variable("P_VC", dims="j")
        P = cm.Parameter("P", dims="i")
        P_E = cm.Parameter("P_E", dims="k")
        psi_X = cm.Parameter("psi_X", dims=("i", "j"))
        psi_X_E = cm.Parameter("psi_X_E", dims=("k", "j"))
        zero_profit, *_ = Leontief(
            "Intermediate bundle",
            output=VC,
            output_price=P_VC,
            factors=[X, X_E],
            factor_prices=[P, P_E],
            factor_shares=[psi_X, psi_X_E],
            factor_dim=["i", "k"],
        )
    objects = {"X": X, "X_E": X_E, "VC": VC, "P_VC": P_VC, "P": P, "P_E": P_E}
    rng = np.random.default_rng(3)
    point = {
        "X": rng.uniform(1, 2, (2, 2)),
        "X_E": rng.uniform(1, 2, (1, 2)),
        "VC": rng.uniform(1, 2, 2),
        "P_VC": rng.uniform(1, 2, 2),
        "P": rng.uniform(1, 2, 2),
        "P_E": rng.uniform(1, 2, 1),
    }
    expected = point["P_VC"] * point["VC"] - (
        (point["P"][:, None] * point["X"]).sum(axis=0)
        + (point["P_E"][:, None] * point["X_E"]).sum(axis=0)
    )

    np.testing.assert_allclose(evaluate_residual(zero_profit, objects, point), expected)


def test_leontief_requires_one_factor_dim_per_factor():
    with pytest.raises(ValueError, match="Expected one factor_dim per factor"):
        with CGEModel(coords={"i": ["a"], "j": ["a"]}):
            X = cm.Variable("X", dims=("i", "j"))
            VC = cm.Variable("VC", dims="j")
            P_VC = cm.Variable("P_VC", dims="j")
            P = cm.Parameter("P", dims="i")
            psi_X = cm.Parameter("psi_X", dims=("i", "j"))
            Leontief(
                "Intermediate bundle",
                output=VC,
                output_price=P_VC,
                factors=[X],
                factor_prices=[P],
                factor_shares=[psi_X],
                factor_dim=["i", "j"],
            )


def test_leontief_requires_one_share_per_factor():
    with pytest.raises(
        ValueError, match="Expected one share per factor, found 2 factors and 1 shares"
    ):
        with CGEModel():
            Y, X_1, X_2, P_Y = (cm.Variable(name) for name in ("Y", "X_1", "X_2", "P_Y"))
            P_1, P_2, a_1 = (cm.Parameter(name) for name in ("P_1", "P_2", "a_1"))
            Leontief(
                "Bundle",
                output=Y,
                output_price=P_Y,
                factors=[X_1, X_2],
                factor_prices=[P_1, P_2],
                factor_shares=[a_1],
            )


def test_helpers_reject_factors_that_are_not_model_objects():
    with pytest.raises(ValueError, match=r"Factors at positions \[0\] have no name"):
        with CGEModel(coords={"i": ["a", "b"], "j": ["a", "b"]}):
            X = cm.Variable("X", dims=("i", "j"))
            VC = cm.Variable("VC", dims="i")
            P_VC = cm.Variable("P_VC", dims="i")
            P = cm.Parameter("P", dims="j")
            psi_X = cm.Parameter("psi_X", dims=("i", "j"))
            Leontief(
                "Bundle",
                output=VC,
                output_price=P_VC,
                factors=[X.rename(i="j", j="i")],
                factor_prices=[P],
                factor_shares=[psi_X.rename(i="j", j="i")],
                factor_dim="j",
            )


def test_a_name_clash_leaves_the_model_unchanged():
    with CGEModel() as model:
        Y, K, L, P = (cm.Variable(name) for name in ("Y", "K", "L", "P"))
        r, w, alpha = (cm.Parameter(name) for name in ("r", "w", "alpha"))
        cm.Equation("Production: demand for L", L, 1.0)

        with pytest.raises(ValueError, match="already taken or repeated"):
            CobbDouglas(
                "Production",
                output=Y,
                output_price=P,
                factors=[K, L],
                factor_prices=[r, w],
                factor_shares=[alpha],
                tfp=1.0,
            )

    assert model.equation_names == ["Production: demand for L"]


def test_helpers_register_on_an_explicit_model_in_order():
    model = CGEModel()
    Y, K, L, P = (cm.Variable(name, model=model) for name in ("Y", "K", "L", "P"))
    r, w, alpha = (cm.Parameter(name, model=model) for name in ("r", "w", "alpha"))

    equations = CobbDouglas(
        "Production",
        output=Y,
        output_price=P,
        factors=[K, L],
        factor_prices=[r, w],
        factor_shares=[alpha],
        tfp=1.0,
        model=model,
    )

    assert list(equations) == model.equations
    assert [equation.name for equation in model.equations] == model.equation_names


def test_cobb_douglas_with_one_factor_and_no_share_uses_a_share_of_one():
    with CGEModel():
        Y, K, P = (cm.Variable(name) for name in ("Y", "K", "P"))
        r, A = (cm.Parameter(name) for name in ("r", "A"))
        production, demand = CobbDouglas(
            "Production",
            output=Y,
            output_price=P,
            factors=[K],
            factor_prices=[r],
            factor_shares=[],
            tfp=A,
        )
    objects = {"Y": Y, "K": K, "P": P, "r": r, "A": A}
    point = {"Y": 6.0, "K": 3.0, "A": 2.0, "P": 1.5, "r": 3.0}

    np.testing.assert_allclose(evaluate_residual(production, objects, point), 0.0)
    np.testing.assert_allclose(evaluate_residual(demand, objects, point), 0.0)


@pytest.mark.parametrize(
    "factor_dim, message",
    [
        (None, r"has dims \['i'\] that the output does not; pass them as its factor_dim"),
        ("k", "Factor X has no dim 'k' to sum over"),
    ],
    ids=["forgotten", "wrong_dim"],
)
def test_leontief_rejects_factor_costs_with_dims_the_output_lacks(factor_dim, message):
    with pytest.raises(ValueError, match=message):
        with CGEModel(coords={"i": ["a", "b"], "j": ["a", "b"], "k": ["energy"]}):
            X = cm.Variable("X", dims=("i", "j"))
            VC = cm.Variable("VC", dims="j")
            P_VC = cm.Variable("P_VC", dims="j")
            P = cm.Parameter("P", dims="i")
            psi_X = cm.Parameter("psi_X", dims=("i", "j"))
            Leontief(
                "Bundle",
                output=VC,
                output_price=P_VC,
                factors=[X],
                factor_prices=[P],
                factor_shares=[psi_X],
                factor_dim=factor_dim,
            )


def test_dixit_stiglitz_rejects_an_output_indexed_by_the_varieties():
    with pytest.raises(ValueError, match="output has the factor dim 'v'"):
        with CGEModel(coords={"v": ["a", "b"]}):
            Y = cm.Variable("Y", dims="v")
            X = cm.Variable("X", dims="v")
            P_Y = cm.Variable("P_Y")
            P_X = cm.Parameter("P_X", dims="v")
            DixitStiglitz(
                "Composite good",
                output=Y,
                output_price=P_Y,
                factor=X,
                factor_price=P_X,
                factor_dim="v",
                factor_share=1.0,
                tfp=1.0,
                epsilon=2.0,
            )
