from itertools import chain
from typing import Literal, cast

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import sympy as sp

from cge_modeling.production_functions import (
    BACKEND_TYPE,
    CES,
    _add_second_alpha,
    _check_pairwise_lengths_match,
    dixit_stiglitz,
    leontief,
    unpack_string_inputs,
)


@pytest.mark.parametrize(
    "dims, coords, expected",
    [
        (None, None, (["L"], ["w"])),
        ("i", {"i": ["A", "B", "C"]}, (["L_A", "L_B", "L_C"], ["w_A", "w_B", "w_C"])),
        (
            ["i", "j"],
            {"i": ["A", "B", "C"], "j": ["D", "E", "F"]},
            (
                ["L_A_D", "L_A_E", "L_A_F", "L_B_D", "L_B_E", "L_B_F", "L_C_D", "L_C_E", "L_C_F"],
                ["w_A_D", "w_A_E", "w_A_F", "w_B_D", "w_B_E", "w_B_F", "w_C_D", "w_C_E", "w_C_F"],
            ),
        ),
    ],
)
def test_unpack_string_inputs(dims, coords, expected):
    factors = "L"
    factor_prices = "w"
    factors, factor_prices = unpack_string_inputs(factors, factor_prices, dims=dims, coords=coords)

    assert factors == expected[0]
    assert factor_prices == expected[1]


def test_add_second_alpha():
    factors = ["L", "K"]
    alpha = _add_second_alpha("alpha", factors)
    assert alpha == ["alpha", "1 - alpha"]


def test_add_second_alpha_doesnt_add():
    # Case 1: single factor
    factors = ["L"]
    alpha = _add_second_alpha("alpha", factors)
    assert alpha == ["alpha"]

    # Case 2: too many factors
    factors = ["L", "K", "M"]
    alpha = _add_second_alpha("alpha", factors)
    assert alpha == ["alpha"]


def test_check_pairwise_lengths():
    factors = ["K"]
    factor_prices = ["r"]
    alpha = ["alpha", "1 - alpha"]
    with pytest.raises(ValueError, match="Lengths of factors and alpha do not match"):
        _check_pairwise_lengths_match(
            ["factors", "factor_prices", "alpha"], [factors, factor_prices, alpha]
        )


@pytest.mark.parametrize("alpha", ["alpha", ["alpha", "1 - alpha"]], ids=["single", "double"])
def test_CES(alpha):
    factors = ["L", "K"]
    factor_prices = ["w", "r"]
    output = "Y"
    output_price = "P"
    A = "A"
    epsilon = "epsilon"

    eq_production, *(L_demand, K_demand) = CES(
        factors=factors,
        factor_prices=factor_prices,
        output=output,
        output_price=output_price,
        TFP=A,
        factor_shares=alpha,
        epsilon=epsilon,
    )

    assert eq_production == (
        "Y = A * ((alpha) * L ** ((epsilon - 1) / epsilon) + (1 - alpha) * K ** ((epsilon - 1) / "
        "epsilon)) ** (epsilon / (epsilon - 1))"
    )
    assert L_demand == "L = Y / A * ((alpha) * P * A / w) ** epsilon"
    assert K_demand == "K = Y / A * ((1 - alpha) * P * A / r) ** epsilon"


def test_computation_of_CES():
    factors = ["L", "K"]
    factor_prices = ["w", "r"]
    output = "Y"
    output_price = "P"
    A = "A"
    epsilon = "epsilon"

    eq_production, *(L_demand, K_demand) = CES(
        factors=factors,
        factor_prices=factor_prices,
        output=output,
        output_price=output_price,
        TFP=A,
        factor_shares="alpha",
        epsilon=epsilon,
    )
    inputs = ["L", "K", "w", "r", "Y", "P", "A", "epsilon", "alpha"]

    eq = sp.parse_expr(eq_production, transformations="all")
    f_eq = sp.lambdify(inputs, eq.rhs)

    def CES_Y(L, K, A, alpha, epsilon):
        return A * (
            (alpha) * L ** ((epsilon - 1) / epsilon) + (1 - alpha) * K ** ((epsilon - 1) / epsilon)
        ) ** (epsilon / (epsilon - 1))

    sympy_out = f_eq(L=10, K=8, w=1, r=1, P=1, A=1, alpha=0.5, epsilon=3, Y=1)
    exact = CES_Y(L=10, K=8, A=1, alpha=0.5, epsilon=3)

    np.testing.assert_allclose(sympy_out, exact)

    def CES_demand(factor_price, output, output_price, TFP, factor_share, epsilon):
        return output / TFP * ((factor_share) * output_price * TFP / factor_price) ** epsilon

    for demand, a_val in zip([L_demand, K_demand], [0.3, 0.7]):
        eq = sp.parse_expr(demand, transformations="all")
        f_eq = sp.lambdify(inputs, eq.rhs)
        sympy_out = f_eq(L=10, K=8, w=1, r=1, P=1, A=2, alpha=0.3, epsilon=3, Y=1)
        exact = CES_demand(1, 1, 1, 2, a_val, 3)
        np.testing.assert_allclose(sympy_out, exact)


@pytest.mark.parametrize("A", ["A", None], ids=["A", "no_A"])
@pytest.mark.parametrize("alpha", ["alpha", None], ids=["alpha", "no_alpha"])
@pytest.mark.parametrize("backend", ["numba", "pytensor"], ids=["numba", "pytensor"])
def test_dixit_stiglitz(A, alpha, backend):
    factors = "X"
    factor_prices = "P_X"
    output = "Y"
    output_price = "P_Y"
    epsilon = "epsilon"

    coords = {"i": ["A", "B", "C"]}
    dims = "i"

    eq_production, X_demand = dixit_stiglitz(
        factors=factors,
        factor_prices=factor_prices,
        output=output,
        output_price=output_price,
        TFP=A,
        factor_shares=alpha,
        epsilon=epsilon,
        dims=dims,
        coords=coords,
        backend=cast(BACKEND_TYPE, backend),
    )

    alpha_str = "alpha * " if alpha is not None else ""
    if backend == "numba":
        prod_expected_inner = f"Sum({alpha_str}X ** ((epsilon - 1) / epsilon), ({dims}, 0, 2)) ** (epsilon / (epsilon - 1))"
    elif backend == "pytensor":
        prod_expected_inner = (
            f"({alpha_str}X ** ((epsilon - 1) / epsilon)).sum() ** (epsilon / (epsilon - 1))"
        )
    else:
        assert False

    A_str = "A * " if A is not None else ""
    prod_expected = f"Y = {A_str}{prod_expected_inner}"
    X_demand_expected = f"X = Y / {A_str}({A_str}{alpha_str}P_Y / P_X) ** epsilon"

    assert eq_production == prod_expected
    assert X_demand == X_demand_expected


@pytest.mark.parametrize("backend", ["numba", "pytensor"], ids=["numba", "pytensor"])
def test_dixit_stiglitz_computation(backend):
    def dx_Y(X, A, alpha, epsilon):
        return A * (alpha * X ** ((epsilon - 1) / epsilon)).sum() ** (epsilon / (epsilon - 1))

    def dx_X(Y, A, alpha, P_Y, P_X, epsilon):
        return Y / A * ((alpha * A * P_Y) / P_X) ** epsilon

    factors = "X"
    factor_prices = "P_X"
    output = "Y"
    output_price = "P_Y"
    epsilon = "epsilon"
    dims = "f"
    coords = {"f": ["A", "B", "C", "D", "E", "F", "G"]}
    eq_production, X_demand = dixit_stiglitz(
        factors=factors,
        factor_prices=factor_prices,
        output=output,
        output_price=output_price,
        TFP="A",
        factor_shares="alpha",
        epsilon=epsilon,
        dims=dims,
        coords=coords,
        backend=cast(BACKEND_TYPE, backend),
    )

    inputs = ["X", "P_X", "Y", "P_Y", "A", "alpha", "epsilon"]
    f = sp.Idx("f")

    local_dict = {
        "X": sp.IndexedBase("X")[f],
        "P_X": sp.IndexedBase("P_X")[f],
        "alpha": sp.IndexedBase("alpha")[f],
        "Y": sp.Symbol("Y"),
        "P_Y": sp.Symbol("P_Y"),
        "A": sp.Symbol("A"),
        "epsilon": sp.Symbol("epsilon"),
        "f": f,
    }

    if backend == "numba":
        eq = sp.parse_expr(eq_production, transformations="all", local_dict=local_dict)
        f_eq = sp.lambdify(inputs, eq.rhs)
        k = len(coords["f"])

        sympy_out = f_eq(
            X=np.arange(k), P_X=np.ones(k), Y=1, P_Y=1, A=1, alpha=np.full(k, 1 / k), epsilon=3
        )

        exact = dx_Y(X=np.arange(k), A=1, alpha=np.full(k, 1 / k), epsilon=3)
        np.testing.assert_allclose(sympy_out, exact)

        # Need to remove the index so the lambdify will vectorize
        local_dict.update(
            {
                "X": sp.IndexedBase("X"),
                "P_X": sp.IndexedBase("P_X"),
                "alpha": sp.IndexedBase("alpha"),
            }
        )

        eq = sp.parse_expr(X_demand, transformations="all", local_dict=local_dict)
        f_eq = sp.lambdify(inputs, eq.rhs)

        sympy_out = f_eq(
            X=np.arange(k), P_X=np.ones(k), Y=1, P_Y=1, A=1, alpha=np.full(k, 1 / k), epsilon=3
        )
        exact = dx_X(Y=1, A=1, alpha=np.full(k, 1 / k), P_Y=1, P_X=np.ones(k), epsilon=3)
        np.testing.assert_allclose(sympy_out, exact)

    elif backend == "pytensor":

        def compile_Y():
            X = pt.dvector("X")
            P_X = pt.dvector("P_X")
            alpha = pt.dvector("alpha")
            P_Y = pt.dscalar("P_Y")
            A = pt.dscalar("A")
            epsilon = pt.dscalar("epsilon")

            exec(eq_production)
            return locals()["Y"], [X, P_X, P_Y, alpha, A, epsilon]

        Y, inputs = compile_Y()
        f_Y = pytensor.function(inputs, Y, on_unused_input="ignore", mode="FAST_COMPILE")
        expected = dx_Y(X=np.arange(7), A=1, alpha=np.full(7, 1 / 7), epsilon=3)
        np.testing.assert_allclose(
            f_Y(np.arange(7), np.ones(7), 1, np.full(7, 1 / 7), 1, 3), expected
        )

        def compile_X():
            P_X = pt.dvector("P_X")
            Y = pt.dscalar("Y")
            P_Y = pt.dscalar("P_Y")
            A = pt.dscalar("A")
            alpha = pt.dvector("alpha")
            epsilon = pt.dscalar("epsilon")

            exec(X_demand)
            return locals()["X"], [Y, P_Y, P_X, A, alpha, epsilon]

        X, inputs = compile_X()
        f_X = pytensor.function(inputs, X, on_unused_input="ignore", mode="FAST_COMPILE")
        expected = dx_X(Y=1, A=1, alpha=np.full(7, 1 / 7), P_Y=1, P_X=np.ones(7), epsilon=3)
        np.testing.assert_allclose(f_X(1, 1, np.ones(7), 1, np.full(7, 1 / 7), 3), expected)


@pytest.mark.parametrize("backend", ["numba", "pytensor"], ids=["numba", "pytensor"])
def test_2d_leontief(backend: BACKEND_TYPE):
    factors = ["X"]
    factor_prices = ["P_X"]
    output = "Y"
    output_price = "P_Y"
    factor_shares = ["phi_X"]

    coords = {"i": ["A", "B", "C"], "j": ["A", "B", "C"]}
    dims = ["i", "j"]

    zero_profit, *factor_demands = leontief(
        factors=factors,
        factor_prices=factor_prices,
        output=output,
        output_price=output_price,
        factor_shares=factor_shares,
        dims=dims,
        coords=coords,
        backend=backend,
    )

    if backend == "numba":
        assert zero_profit == (
            f"{output_price} * {output} = Sum({factor_prices[0]}.subs("
            + "{"
            + f"{dims[0]}:{dims[1]}"
            + "}) "
            f"* {factors[0]}"
            f".subs([({dims[0]}, a), ({dims[1]}, {dims[0]}), (a, {dims[1]})]), "
            f"({dims[1]}, 0, {len(coords[dims[1]]) - 1}))"
        )

    elif backend == "pytensor":
        assert (
            zero_profit
            == f"{output_price} * {output} = ({factor_prices[0]}[:, None] * {factors[0]}).sum(axis=0).ravel()"
        )
        assert factor_demands[0] == f"{factors[0]} = {factor_shares[0]} * {output}[None]"
    else:
        assert False


def test_1d_leontief():
    factors = ["VA", "VC"]
    factor_prices = ["P_VA", "P_VC"]
    output = "Y"
    output_price = "P_Y"
    factor_shares = ["phi_VA", "phi_VC"]

    coords = {"i": ["A", "B", "C"], "j": ["A", "B", "C"]}
    dims = "i"

    zero_profit, *factor_demands = leontief(
        factors=factors,
        factor_prices=factor_prices,
        output=output,
        output_price=output_price,
        factor_shares=factor_shares,
        dims=dims,
        coords=coords,
        backend="numba",
    )

    assert zero_profit == "P_Y * Y = P_VA * VA + P_VC * VC"
    for i, demand in enumerate(factor_demands):
        assert demand == f"{factors[i]} = {output} * {factor_shares[i]}"


@pytest.mark.parametrize("backend", ["numba", "pytensor"], ids=["numba", "pytensor"])
def test_leonteif_1d_computation(backend):
    def leontief_Y(VA, VC, P_VA, P_VC, P_Y):
        return (P_VA * VA + P_VC * VC) / P_Y

    def leontief_VA(Y, phi_VA, *args, **kwargs):
        return Y * phi_VA

    def leontief_VC(Y, phi_VC, *args, **kwargs):
        return Y * phi_VC

    factors = ["VA", "VC"]
    factor_prices = ["P_VA", "P_VC"]
    output = "Y"
    output_price = "P_Y"
    factor_shares = ["phi_VA", "phi_VC"]

    coords = {"i": ["A", "B", "C"], "j": ["A", "B", "C"]}
    dims = "i"

    zero_profit, *factor_demands = leontief(
        factors=factors,
        factor_prices=factor_prices,
        output=output,
        output_price=output_price,
        factor_shares=factor_shares,
        dims=dims,
        coords=coords,
        backend=cast(BACKEND_TYPE, backend),
    )

    inputs = ["VA", "VC", "P_VA", "P_VC", "Y", "P_Y", "phi_VA", "phi_VC"]
    i = sp.Idx("i")
    local_dict = {
        "VA": sp.IndexedBase("VA"),
        "VC": sp.IndexedBase("VC"),
        "P_VA": sp.IndexedBase("P_VA"),
        "P_VC": sp.IndexedBase("P_VC"),
        "Y": sp.Symbol("Y"),
        "P_Y": sp.Symbol("P_Y"),
        "phi_VA": sp.IndexedBase("phi_VA"),
        "phi_VC": sp.IndexedBase("phi_VC"),
    }

    if backend == "numba":
        eq = sp.parse_expr(zero_profit, transformations="all", local_dict=local_dict)
        f_eq = sp.lambdify(inputs, eq.rhs)
        k = len(coords["i"])

        sympy_out = f_eq(
            VA=np.arange(k),
            VC=np.arange(k),
            P_VA=np.ones(k),
            P_VC=np.ones(k),
            Y=1,
            P_Y=1,
            phi_VA=np.full(k, 1 / k),
            phi_VC=np.full(k, 1 / k),
        )

        exact = leontief_Y(
            VA=np.arange(k), VC=np.arange(k), P_VA=np.ones(k), P_VC=np.ones(k), P_Y=1
        )
        np.testing.assert_allclose(sympy_out, exact)

        for factor_demand, f_expected in zip(factor_demands, [leontief_VA, leontief_VC]):
            eq = sp.parse_expr(factor_demand, transformations="all", local_dict=local_dict)
            f_eq = sp.lambdify(inputs, eq.rhs)
            sympy_out = f_eq(
                VA=np.arange(k),
                VC=np.arange(k),
                P_VA=np.ones(k),
                P_VC=np.ones(k),
                Y=1,
                P_Y=1,
                phi_VA=np.full(k, 1 / k),
                phi_VC=np.full(k, 1 / k),
            )
            exact = f_expected(Y=1, phi_VA=np.full(k, 1 / k), phi_VC=np.full(k, 1 / k))
            np.testing.assert_allclose(sympy_out, exact)

    elif backend == "pytensor":

        def compile_Y():
            VA = pt.dvector("VA")
            VC = pt.dvector("VC")
            P_VA = pt.dvector("P_VA")
            P_VC = pt.dvector("P_VC")
            P_Y = pt.dscalar("P_Y")

            exec(zero_profit)
            return locals()["Y"], [VA, VC, P_VA, P_VC, P_Y]

        Y, inputs = compile_Y()
        f_Y = pytensor.function(inputs, Y, on_unused_input="ignore", mode="FAST_COMPILE")
        expected = leontief_Y(
            VA=np.arange(7), VC=np.arange(7), P_VA=np.ones(7), P_VC=np.ones(7), P_Y=1
        )
        np.testing.assert_allclose(
            f_Y(np.arange(7), np.arange(7), np.ones(7), np.ones(7), 1), expected
        )

        def compile_VA():
            Y = pt.dscalar("Y")
            phi_VA = pt.dvector("phi_VA")

            exec(factor_demands[0])
            return locals()["VA"], [Y, phi_VA]

        VA, inputs = compile_VA()
        f_VA = pytensor.function(inputs, VA, on_unused_input="ignore", mode="FAST_COMPILE")
        expected = leontief_VA(Y=1, phi_VA=np.full(7, 1 / 7))
        np.testing.assert_allclose(f_VA(1, np.full(7, 1 / 7)), expected)

        def compile_VC():
            Y = pt.dscalar("Y")
            phi_VC = pt.dvector("phi_VC")

            exec(factor_demands[1])
            return locals()["VC"], [Y, phi_VC]

        VC, inputs = compile_VC()
        f_VC = pytensor.function(inputs, VC, on_unused_input="ignore", mode="FAST_COMPILE")
        expected = leontief_VC(Y=1, phi_VC=np.full(7, 1 / 7))
        np.testing.assert_allclose(f_VC(1, np.full(7, 1 / 7)), expected)


@pytest.mark.parametrize("backend", ["numba", "pytensor"], ids=["numba", "pytensor"])
@pytest.mark.parametrize(
    "dims, coords",
    [
        (("i", "j"), {"i": ["A", "B", "C"], "j": ["A", "B", "C"]}),
        (("i", "j"), {"i": ["A", "B", "C"], "j": ["D", "E"]}),
    ],
    ids=["square", "rectangular"],
)
def test_2d_leontief_computation(backend, dims, coords):
    def leontief_Y(X, P_X, P_Y):
        return (P_X[:, None] * X).sum(axis=0) / P_Y

    def leontief_X(Y, phi_X, *args, **kwargs):
        return Y * phi_X

    factors = ["X"]
    factor_prices = ["P_X"]
    output = "Y"
    output_price = "P_Y"
    factor_shares = ["phi_X"]

    zero_profit, *factor_demands = leontief(
        factors=factors,
        factor_prices=factor_prices,
        output=output,
        output_price=output_price,
        factor_shares=factor_shares,
        dims=dims,
        coords=coords,
        backend=backend,
    )

    inputs = ["X", "P_X", "Y", "P_Y", "phi_X"]
    i = sp.Idx("i")
    j = sp.Idx("j")

    local_dict = {
        "X": sp.IndexedBase("X")[i, j],
        "P_X": sp.IndexedBase("P_X")[i],
        "Y": sp.Symbol("Y"),
        "P_Y": sp.Symbol("P_Y"),
        "phi_X": sp.IndexedBase("phi_X")[i, j],
        "i": i,
        "j": j,
    }

    if backend == "numba":
        print(zero_profit)
        eq = sp.parse_expr(zero_profit, transformations="all", local_dict=local_dict)
        print(eq)
        f_eq = sp.lambdify(inputs + ["i"], eq.rhs)
        k = len(coords["i"])
        l = len(coords["j"])

        X_prices = np.random.normal(size=k) ** 2
        phi_X_vals = np.random.dirichlet(np.ones(l), size=(k,)).T
        X_val = np.arange(k * l).reshape((k, l))

        sympy_out = f_eq(X=X_val, P_X=X_prices, Y=1, P_Y=1, phi_X=phi_X_vals, i=np.arange(k))

        exact = leontief_Y(X=X_val, P_X=X_prices, P_Y=1)
        np.testing.assert_allclose(sympy_out, exact)

    #     for factor_demand, f_expected in zip(factor_demands, [leontief_X]):
    #         eq = sp.parse_expr(factor_demand, transformations="all", local_dict=local_dict)
    #         f_eq = sp.lambdify(inputs, eq
    #
