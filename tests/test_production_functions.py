from typing import Literal, cast

import pytest

from cge_modeling.production_functions import (
    BACKEND_TYPE,
    CES,
    _add_second_alpha,
    _check_pairwise_lengths_match,
    dixit_stiglitz,
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
        prod_expected_inner = f"Sum({alpha_str}X ** ((epsilon - 1) / epsilon), (i, 0, 2)) ** (epsilon / (epsilon - 1))"
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
