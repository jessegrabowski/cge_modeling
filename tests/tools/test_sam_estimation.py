import warnings

from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from cge_modeling.tools.sam_estimation import (
    KL,
    LeastSquares,
    Pearson,
    ReverseKL,
    SymbolicSAM,
    cells_forced_to_zero,
    linear_constraints,
)

UNBALANCED_SAM = Path(__file__).parents[1] / "data" / "unbalanced_sam.csv"

# CLARABEL's tolerances apply after fit divides every cell by the total of the absolute cells, which leaves the cells of
# these SAMs accurate to about 1e-4 relative.
SOLVER_RTOL = 1e-3
TIGHT_CLARABEL = {"tol_gap_abs": 1e-12, "tol_gap_rel": 1e-12, "tol_feas": 1e-12}


def load_unbalanced_sam() -> pd.DataFrame:
    return pd.read_csv(UNBALANCED_SAM, index_col=[0, 1], header=[0, 1]).fillna(0.0)


def assert_balanced(sam: pd.DataFrame) -> None:
    np.testing.assert_allclose(sam.sum(axis=0).to_numpy(), sam.sum(axis=1).to_numpy(), atol=1e-6)


def test_constraint_rows():
    prior = pd.DataFrame(
        [[0.0, 5.0, 2.0], [1.0, 0.0, 3.0], [4.0, 6.0, 0.0]], index=list("abc"), columns=list("abc")
    )
    sam = SymbolicSAM(prior)

    linear = linear_constraints(
        sam,
        [
            sam.sel(row="a").sum() == 10.0,
            sam.sel(row="c", col="a") == 2.0 * sam.sel(row="b", col="c"),
            sam.sel(row=["b", "c"], col="a").sum() <= 4.0,
            sam.sel(col="c").sum() >= 1.0,
        ],
    )

    np.testing.assert_allclose(linear.equality_operator @ sam.prior_values, [7.0, -2.0])
    np.testing.assert_allclose(linear.equality_values, [10.0, 0.0])
    np.testing.assert_allclose(linear.inequality_operator @ sam.prior_values, [5.0, -5.0])
    np.testing.assert_allclose(linear.inequality_values, [4.0, -1.0])


def test_sum_of_single_cell_is_the_cell():
    sam = SymbolicSAM(pd.DataFrame([[0.0, 2.0], [3.0, 0.0]], index=["a", "b"], columns=["a", "b"]))
    cell = sam.sel(row="a", col="b")

    summed = linear_constraints(sam, [cell.sum() == 5.0, cell.sum().sum() == 5.0])
    direct = linear_constraints(sam, [cell == 5.0])

    np.testing.assert_array_equal(
        summed.equality_operator.toarray(), np.tile(direct.equality_operator.toarray(), (2, 1))
    )
    np.testing.assert_array_equal(summed.equality_values, [5.0, 5.0])


def test_expression_broadcasts_over_missing_dim():
    prior = pd.DataFrame(
        [[0.0, 5.0, 2.0], [1.0, 0.0, 3.0], [4.0, 6.0, 0.0]], index=list("abc"), columns=list("abc")
    )
    sam = SymbolicSAM(prior)

    # Every account's income equals the income of a, which repeats over the row dim.
    linear = linear_constraints(sam, [sam.sum("col") == sam.sel(row="a").sum()])

    np.testing.assert_allclose(
        linear.equality_operator @ sam.prior_values, [0.0, 4.0 - 7.0, 10.0 - 7.0]
    )
    np.testing.assert_allclose(linear.equality_values, [0.0, 0.0, 0.0])


def test_expression_on_the_left_broadcasts():
    prior = pd.DataFrame(
        [[0.0, 5.0, 2.0], [1.0, 0.0, 3.0], [4.0, 6.0, 0.0]], index=list("abc"), columns=list("abc")
    )
    sam = SymbolicSAM(prior)

    linear = linear_constraints(sam, [sam.sel(row="a").sum() == sam.sum("col")])

    np.testing.assert_allclose(
        linear.equality_operator @ sam.prior_values, [0.0, 7.0 - 4.0, 7.0 - 10.0]
    )


def test_identity_operator_is_sparse():
    # Each off-diagonal cell enters its row account's income and its column account's spending, and a diagonal cell
    # cancels out.
    prior = pd.DataFrame(
        [[1.0, 5.0, 2.0], [1.0, 0.0, 3.0], [4.0, 6.0, 0.0]], index=list("abc"), columns=list("abc")
    )
    sam = SymbolicSAM(prior)

    balance = sam.sum("col") == sam.sum("row").rename({"col": "row"})
    operator = linear_constraints(sam, [balance]).equality_operator

    assert operator.shape == (3, sam.n_cells)
    assert operator.nnz == 2 * (sam.n_cells - 1)


def test_transposed_expressions_align_by_dim_name():
    prior = pd.DataFrame(
        [[0.0, 5.0, 2.0], [1.0, 0.0, 3.0], [4.0, 6.0, 0.0]], index=list("abc"), columns=list("abc")
    )
    sam = SymbolicSAM(prior)

    # Swapping the dim names makes position (i, j) of the right side the cell (j, i).
    linear = linear_constraints(sam, [sam.sel() == sam.sel().rename({"row": "col", "col": "row"})])

    np.testing.assert_allclose(
        linear.equality_operator @ sam.prior_values, (prior - prior.T).to_numpy().ravel()
    )


def test_series_constraint_rows():
    prior = pd.DataFrame(
        [[0.0, 5.0, 2.0], [1.0, 0.0, 3.0], [4.0, 6.0, 0.0]], index=list("abc"), columns=list("abc")
    )
    sam = SymbolicSAM(prior)
    known_income = pd.Series({"c": 11.0, "a": 8.0, "b": np.nan})

    linear = linear_constraints(sam, [sam.sum("col") == known_income])

    np.testing.assert_allclose(linear.equality_operator @ sam.prior_values, [7.0, 10.0])
    np.testing.assert_allclose(linear.equality_values, [8.0, 11.0])

    reflected = linear_constraints(sam, [known_income == sam.sum("col")])
    np.testing.assert_allclose(
        reflected.equality_operator.toarray(), linear.equality_operator.toarray()
    )
    np.testing.assert_allclose(reflected.equality_values, linear.equality_values)


def test_block_constraint_rows():
    prior = pd.DataFrame(
        [[0.0, 5.0, 2.0], [1.0, 0.0, 3.0], [4.0, 6.0, 0.0]], index=list("abc"), columns=list("abc")
    )
    sam = SymbolicSAM(prior)
    block = sam.sel(row=["a", "b"], col=["b", "c"])
    known = pd.DataFrame({"b": [6.0, np.nan], "c": [2.5, 3.5]}, index=["a", "b"])
    known_transposed = xr.DataArray(
        known.to_numpy().T, dims=["col", "row"], coords={"col": known.columns, "row": known.index}
    )

    from_frame = linear_constraints(sam, [block == known])
    from_array = linear_constraints(sam, [block == known_transposed])

    np.testing.assert_allclose(from_frame.equality_operator @ sam.prior_values, [5.0, 2.0, 3.0])
    np.testing.assert_allclose(from_frame.equality_values, [6.0, 2.5, 3.5])
    np.testing.assert_allclose(
        from_array.equality_operator.toarray(), from_frame.equality_operator.toarray()
    )
    np.testing.assert_allclose(from_array.equality_values, from_frame.equality_values)


def test_dataarray_broadcasts():
    prior = pd.DataFrame(
        [[0.0, 5.0, 2.0], [1.0, 0.0, 3.0], [4.0, 6.0, 0.0]], index=list("abc"), columns=list("abc")
    )
    sam = SymbolicSAM(prior)
    column_caps = xr.DataArray([10.0, 20.0], dims=["col"], coords={"col": ["b", "c"]})

    linear = linear_constraints(sam, [sam.sel(col=["b", "c"]) <= column_caps])

    np.testing.assert_allclose(
        linear.inequality_operator @ sam.prior_values, [5.0, 2.0, 0.0, 3.0, 6.0, 0.0]
    )
    np.testing.assert_allclose(linear.inequality_values, [10.0, 20.0] * 3)


def test_sel_multiindex_group():
    prior = load_unbalanced_sam()
    sam = SymbolicSAM(prior)

    selected = sam.sel(row="Factor", col=("Institution", "Household"))

    assert selected.dims == ("row",)
    assert selected.labels["row"].equals(
        prior.loc["Factor"].index.map(lambda name: ("Factor", name))
    )


def test_named_dims():
    values = [[0.0, 5.0, 2.0], [1.0, 0.0, 3.0], [4.0, 6.0, 0.0]]
    frame = pd.DataFrame(
        values,
        index=pd.Index(list("abc"), name="receiver"),
        columns=pd.Index(list("abc"), name="payer"),
    )
    array = xr.DataArray(
        values, dims=["receiver", "payer"], coords={"receiver": list("abc"), "payer": list("abc")}
    )

    for prior in (frame, array):
        sam = SymbolicSAM(prior)
        linear = linear_constraints(sam, [sam.sel(receiver="a", payer=["b", "c"]).sum() == 7.0])

        assert sam.dims == ("receiver", "payer")
        np.testing.assert_allclose(linear.equality_operator @ sam.prior_values, [7.0])
        assert_balanced(sam.fit().sam)


def test_index_column_round_trip():
    values = np.array([[0.0, 5.0, 2.0], [1.0, 0.0, 3.0], [4.0, 6.0, 0.0]])
    accounts = ["a", "b", "c"]
    columns = {"account": accounts} | {account: values[:, j] for j, account in enumerate(accounts)}
    expected = SymbolicSAM(pd.DataFrame(values, index=accounts, columns=accounts)).fit().sam

    for prior in (pl.DataFrame(columns), pd.DataFrame(columns)):
        sam = SymbolicSAM(prior, index_column="account")
        estimate = sam.fit().sam

        assert sam.dims == ("account", "col")
        assert type(estimate) is type(prior)
        assert list(estimate.columns) == ["account", *accounts]
        assert list(estimate["account"]) == accounts
        np.testing.assert_allclose(np.asarray(estimate[accounts]), expected.to_numpy(), rtol=1e-12)


def test_dataarray_round_trip():
    accounts = pd.MultiIndex.from_tuples(
        [("Factor", "a"), ("Institution", "b"), ("Institution", "c")]
    )
    coordinates = xr.Coordinates.from_pandas_multiindex(accounts, "receiver").merge(
        xr.Coordinates.from_pandas_multiindex(accounts, "payer")
    )
    prior = xr.DataArray(
        [[0.0, 5.0, 2.0], [1.0, 0.0, 3.0], [4.0, 6.0, 0.0]],
        dims=["receiver", "payer"],
        coords=coordinates.coords,
    )

    estimate = SymbolicSAM(prior).fit().sam

    assert isinstance(estimate, xr.DataArray)
    assert estimate.dims == prior.dims
    assert estimate.indexes["receiver"].equals(accounts)
    np.testing.assert_allclose(
        estimate.sum("payer").to_numpy(), estimate.sum("receiver").to_numpy(), atol=1e-9
    )


def test_sel_partial_keys_of_a_three_level_index():
    accounts = pd.MultiIndex.from_tuples(
        [
            ("Factor", "Labor", "Skilled"),
            ("Factor", "Labor", "Unskilled"),
            ("Factor", "Capital", "Capital"),
            ("Institution", "Household", "Rural"),
        ]
    )
    sam = SymbolicSAM(pd.DataFrame(1.0 - np.eye(4), index=accounts, columns=accounts))

    assert sam.sel(row="Factor").labels["row"].equals(accounts[:3])
    assert sam.sel(row=("Factor", "Labor")).labels["row"].equals(accounts[:2])
    assert (
        sam.sel(row=[("Factor", "Labor"), "Institution"]).labels["row"].equals(accounts[[0, 1, 3]])
    )
    assert sam.sel(row=("Factor", "Labor", "Skilled")).dims == ("col",)


def test_dataarray_sel_multiindex_group():
    accounts = pd.MultiIndex.from_tuples(
        [("Factor", "a"), ("Institution", "b"), ("Institution", "c")]
    )
    coordinates = xr.Coordinates.from_pandas_multiindex(accounts, "receiver").merge(
        xr.Coordinates.from_pandas_multiindex(accounts, "payer")
    )
    prior = xr.DataArray(
        [[0.0, 5.0, 2.0], [1.0, 0.0, 3.0], [4.0, 6.0, 0.0]],
        dims=["receiver", "payer"],
        coords=coordinates.coords,
    )
    sam = SymbolicSAM(prior)

    selected = sam.sel(receiver="Institution", payer=("Factor", "a"))
    linear = linear_constraints(sam, [selected.sum() == 9.0])

    assert selected.dims == ("receiver",)
    assert selected.labels["receiver"].equals(accounts[1:])
    np.testing.assert_allclose(linear.equality_operator @ sam.prior_values, [5.0])


def test_dataarray_multiindex_constraint_rows():
    accounts = pd.MultiIndex.from_tuples(
        [("Factor", "a"), ("Institution", "b"), ("Institution", "c")]
    )
    prior = pd.DataFrame(
        [[0.0, 5.0, 2.0], [1.0, 0.0, 3.0], [4.0, 6.0, 0.0]], index=accounts, columns=accounts
    )
    sam = SymbolicSAM(prior)
    institutions = accounts[1:]
    coordinates = xr.Coordinates.from_pandas_multiindex(institutions, "row").merge(
        xr.Coordinates.from_pandas_multiindex(institutions, "col")
    )
    known = xr.DataArray(
        [[np.nan, 3.5], [6.5, np.nan]], dims=["row", "col"], coords=coordinates.coords
    )

    linear = linear_constraints(sam, [sam.sel(row="Institution", col="Institution") == known])

    np.testing.assert_allclose(linear.equality_operator @ sam.prior_values, [3.0, 6.0])
    np.testing.assert_allclose(linear.equality_values, [3.5, 6.5])


@pytest.mark.parametrize(
    "build, message",
    [
        (
            lambda: SymbolicSAM(
                pd.DataFrame(
                    np.eye(2),
                    index=pd.Index([0, 1], name="account"),
                    columns=pd.Index([0, 1], name="account"),
                )
            ),
            "both named 'account'",
        ),
        (
            lambda: SymbolicSAM(pd.DataFrame([[0.0, 1.0], [1.0, 0.0]])).sel(receiver=0),
            "Unknown dims",
        ),
        (
            lambda: SymbolicSAM(
                pl.DataFrame({"account": ["a", "b"], "a": [0.0, 1.0], "b": [1.0, 0.0]})
            ),
            "polars.DataFrame prior has no index",
        ),
        (
            lambda: SymbolicSAM(
                pl.DataFrame({"account": ["a", "b"], "a": [0.0, 1.0], "b": [1.0, 0.0]}),
                index_column="accounts",
            ),
            "'accounts' is not a column",
        ),
        (
            lambda: SymbolicSAM(
                xr.DataArray([[0.0, 1.0], [1.0, 0.0]], dims=["row", "col"]), index_column="row"
            ),
            "index_column applies to a frame prior",
        ),
    ],
    ids=[
        "axes_with_the_same_name",
        "unknown_dim_in_sel",
        "frame_without_an_index",
        "index_column_not_in_frame",
        "index_column_with_dataarray",
    ],
)
def test_invalid_prior_raises(build, message):
    with pytest.raises(ValueError, match=message):
        build()


@pytest.mark.parametrize(
    "build, error, message",
    [
        (lambda sam: sam.sum("col") - sam.sum("row"), ValueError, "outer product"),
        (lambda sam: sam.sel(row=0, col=1) * sam.sel(row=1, col=0), TypeError, "linear"),
        (lambda sam: sam.sel(row=0).sum() < 1.0, TypeError, "Strict"),
        (lambda sam: bool(sam.sel(row=0).sum() == 1.0), TypeError, "no truth value"),
        (
            lambda sam: sam.sum("col") == pd.Series({0: 1.0, 5: 2.0}),
            ValueError,
            r"not in dim 'row': \[5\]",
        ),
        (lambda sam: np.ones(2) + sam.sel(row=0), TypeError, "unlabeled array"),
        (lambda sam: sam.sel(row=0).sum("row"), ValueError, "Cannot sum over 'row'"),
        (
            lambda sam: sam.sel().rename({"account": "row"}),
            ValueError,
            r"Cannot rename \['account'\]",
        ),
        (lambda sam: sam.sel().rename({"row": "col"}), ValueError, "repeated dims"),
        (lambda sam: sam.sum("col") == pd.DataFrame(np.ones((2, 2))), ValueError, "index is 'row'"),
        (
            lambda sam: sam.sel() == pd.DataFrame(np.ones((2, 2)), columns=[0, 7]),
            ValueError,
            r"not in dim 'col': \[7\]",
        ),
        (
            lambda sam: sam.sum("col") == xr.DataArray(np.ones((2, 2)), dims=["row", "col"]),
            ValueError,
            "outer product",
        ),
        (
            lambda sam: xr.DataArray(np.ones(2), dims=["col"]) <= sam.sel(row=0),
            TypeError,
            "to the right of the expression",
        ),
        (
            lambda sam: (
                sam.sel(row=[0, 1]).sum("col")
                == sam.sel(col=[1, 0]).sum("row").rename({"col": "row"})
            ),
            ValueError,
            r"accounts on dim 'row' differ: \[0, 1\] and \[1, 0\]",
        ),
    ],
    ids=[
        "mismatched_dims",
        "product_of_cells",
        "strict_inequality",
        "constraint_as_condition",
        "unknown_series_label",
        "array_operand",
        "sum_over_missing_dim",
        "rename_unknown_dim",
        "rename_onto_existing_dim",
        "dataframe_against_one_dim",
        "unknown_dataframe_column_label",
        "dataarray_with_extra_dim",
        "dataarray_on_the_left",
        "expressions_with_reordered_labels",
    ],
)
def test_invalid_expression_raises(build, error, message):
    sam = SymbolicSAM(pd.DataFrame([[0.0, 1.0], [1.0, 0.0]]))

    with pytest.raises(error, match=message):
        build(sam)


def test_subject_to():
    sam = SymbolicSAM(pd.DataFrame([[0.0, 1.0], [1.0, 0.0]]))
    other = SymbolicSAM(pd.DataFrame([[0.0, 1.0], [1.0, 0.0]]))

    sam.subject_to(cap=sam.sel(row=0).sum() <= 2.0)

    assert list(sam.constraints) == ["cap"]
    with pytest.raises(ValueError, match="already exists"):
        sam.subject_to(cap=sam.sel(row=0).sum() <= 3.0)
    with pytest.raises(TypeError, match="this SymbolicSAM"):
        sam.subject_to(foreign=other.sel(row=0).sum() <= 3.0)
    with pytest.raises(ValueError, match="reserved"):
        sam.subject_to(**{"accounting identity": sam.sel(row=0).sum() <= 3.0})


def test_cells_forced_to_zero():
    prior = pd.DataFrame(
        [[0.0, 2.0, 1.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], index=list("abc"), columns=list("abc")
    )
    assert cells_forced_to_zero(SymbolicSAM(prior)) == [("a", "c")]

    prior.loc["c", "a"] = -1.0
    assert cells_forced_to_zero(SymbolicSAM(prior)) == [("a", "c"), ("c", "a")]

    prior.loc["c", "a"] = 1.0
    assert cells_forced_to_zero(SymbolicSAM(prior)) == []


def test_fit_raises_on_forced_zeros():
    sam = SymbolicSAM(pd.DataFrame([[0.0, 1.0], [0.0, 0.0]]))

    with pytest.raises(ValueError, match=r"lie on no cycle.*\(0, 1\)"):
        sam.fit()


def test_infeasible_fit_raises():
    sam = SymbolicSAM(pd.DataFrame([[0.0, 2.0], [1.0, 0.0]], index=["a", "b"], columns=["a", "b"]))
    sam.subject_to(income=sam.sel(row="a").sum() == 5.0, spending=sam.sel(col="a").sum() == 7.0)

    with pytest.raises(ValueError, match="no solution: solver status is 'infeasible'"):
        sam.fit()


@pytest.mark.parametrize(
    "constraint",
    [lambda cell: cell == 3.0, lambda cell: cell >= 1.0],
    ids=["equal to nonzero", "at least positive"],
)
def test_unsatisfiable_constraint_on_zero_prior_cell_raises(constraint):
    sam = SymbolicSAM(pd.DataFrame([[0.0, 2.0], [2.0, 0.0]], index=["a", "b"], columns=["a", "b"]))
    sam.subject_to(diagonal=constraint(sam.sel(row="a", col="a")))

    with pytest.raises(ValueError, match="'diagonal' involves no nonzero cell"):
        sam.fit()


@pytest.mark.parametrize(
    "constraint",
    [lambda cell: cell == 0.0, lambda cell: cell <= 1.0],
    ids=["equal to zero", "at most positive"],
)
def test_satisfied_constraint_on_zero_prior_cell_is_ignored(constraint):
    prior = pd.DataFrame([[0.0, 2.0], [2.0, 0.0]], index=["a", "b"], columns=["a", "b"])
    sam = SymbolicSAM(prior)
    sam.subject_to(diagonal=constraint(sam.sel(row="a", col="a")))

    np.testing.assert_allclose(sam.fit().sam, prior, rtol=SOLVER_RTOL)


def test_balanced_sam_fixed_point():
    prior = pd.DataFrame([[0.0, 3.0, 0.0], [2.0, 0.0, 0.0], [1.0, -1.0, 0.0]])

    estimate = SymbolicSAM(prior).fit()

    np.testing.assert_allclose(estimate.sam, prior, rtol=1e-9)
    assert estimate.objective == pytest.approx(0.0, abs=1e-12)


def test_balanced_sam_tiny_cells():
    prior = pd.DataFrame([[0.0, 5000.0, 1e-7], [5000.0, 0.0, 0.0], [1e-7, 0.0, 0.0]])

    estimate = SymbolicSAM(prior).fit()

    np.testing.assert_allclose(estimate.sam, prior, rtol=1e-9)


def test_negative_cells():
    prior = pd.DataFrame([[0.0, 5.0, 0.0], [-2.0, -1.0, 3.0], [4.0, 0.0, 0.0]])

    estimate = SymbolicSAM(prior).fit().sam

    assert_balanced(estimate)
    assert estimate.loc[0, 1] > 0
    assert estimate.loc[1, 0] < 0
    assert estimate.loc[1, 1] == pytest.approx(-1.0, rel=1e-12)


def test_account_scaling_structure():
    sam = SymbolicSAM(load_unbalanced_sam())

    estimate = sam.fit()

    assert_balanced(estimate.sam)

    # The KKT conditions give |x_ij| = |z_ij| * exp(u_j - u_i) for positive cells and exp(u_i - u_j) for negative ones,
    # with no common factor, so the log ratios are fit exactly by account potentials.
    cell_values = estimate.sam.to_numpy()[sam.rows, sam.columns]
    log_ratios = np.log(cell_values / sam.prior_values)
    potentials = np.zeros((sam.n_cells, sam.n_accounts))
    cells = np.arange(sam.n_cells)
    potentials[cells, sam.columns] += sam.signs
    potentials[cells, sam.rows] -= sam.signs
    fitted, *_ = np.linalg.lstsq(potentials, log_ratios, rcond=None)

    np.testing.assert_allclose(potentials @ fitted, log_ratios, atol=1e-9)


def test_dual_matches_clarabel():
    sam = SymbolicSAM(load_unbalanced_sam())

    dual = sam.fit(solver="dual")
    clarabel = sam.fit(solver=cp.CLARABEL, **TIGHT_CLARABEL)

    assert dual.status == "optimal"
    np.testing.assert_allclose(dual.sam, clarabel.sam, rtol=1e-6, atol=1e-9)
    assert dual.objective == pytest.approx(clarabel.objective, rel=1e-8)


def test_default_solver():
    prior = load_unbalanced_sam()
    unconstrained = SymbolicSAM(prior)
    constrained = SymbolicSAM(prior)
    constrained.subject_to(institution_income=constrained.sel(row="Institution").sum() >= 0.0)

    pd.testing.assert_frame_equal(unconstrained.fit().sam, unconstrained.fit(solver="dual").sam)
    pd.testing.assert_frame_equal(constrained.fit().sam, constrained.fit(solver=cp.CLARABEL).sam)


@pytest.mark.parametrize(
    "constrained, objective",
    [(True, None), (False, KL(weights=2.0)), (False, LeastSquares())],
    ids=["constraints", "weighted KL", "least squares"],
)
def test_dual_rejects_other_problems(constrained, objective):
    sam = SymbolicSAM(pd.DataFrame([[0.0, 1.0], [1.0, 0.0]]))
    if constrained:
        sam.subject_to(cap=sam.sel(row=0).sum() <= 2.0)

    with pytest.raises(ValueError, match="only the unweighted KL objective"):
        sam.fit(objective=objective, solver="dual")


def test_dual_with_disconnected_accounts_and_diagonal():
    # Accounts a and b trade only with each other, as do c and d, and a also pays itself.
    prior = pd.DataFrame(
        [[1.0, 2.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0], [0.0, 0.0, 1.0, 0.0]],
        index=list("abcd"),
        columns=list("abcd"),
    )
    sam = SymbolicSAM(prior)

    dual = sam.fit(solver="dual")
    clarabel = sam.fit(solver=cp.CLARABEL, **TIGHT_CLARABEL)

    assert dual.status == "optimal"
    assert dual.sam.loc["a", "a"] == pytest.approx(1.0, rel=1e-12)
    np.testing.assert_allclose(dual.sam, clarabel.sam, rtol=1e-6, atol=1e-9)


def test_dual_balances_an_account_with_tiny_flows():
    # Account a trades a millionth of what the others do. Every account must still balance to the dual tolerance of
    # its own turnover, including a.
    values = np.random.default_rng(1).uniform(100.0, 200.0, size=(4, 4))
    np.fill_diagonal(values, 0.0)
    values[0, :] *= 1e-6
    values[:, 0] *= 1e-6
    prior = pd.DataFrame(values, index=list("abcd"), columns=list("abcd"))

    estimate = SymbolicSAM(prior).fit(solver="dual")
    balanced = estimate.sam.to_numpy()
    relative_imbalance = np.abs(balanced.sum(axis=1) - balanced.sum(axis=0)) / (
        balanced.sum(axis=1) + balanced.sum(axis=0)
    )

    assert estimate.status == "optimal"
    assert relative_imbalance.max() < 1e-12


def test_dual_tol_reaches_the_solver():
    sam = SymbolicSAM(load_unbalanced_sam())

    with pytest.warns(UserWarning, match="optimal_inaccurate"):
        loose = sam.fit(solver="dual", tol=1e-3)

    assert loose.status == "optimal_inaccurate"
    assert sam.fit(solver="dual").status == "optimal"


def test_dual_with_no_account_to_scale(caplog):
    # Only diagonal cells, so no account's scale changes anything and the solver has nothing to do.
    prior = pd.DataFrame(np.diag([2.0, 3.0]), index=["a", "b"], columns=["a", "b"])

    estimate = SymbolicSAM(prior).fit(solver="dual")

    assert estimate.status == "optimal"
    np.testing.assert_array_equal(estimate.sam, prior)
    assert caplog.records == []


def test_dual_solver_kwargs():
    sam = SymbolicSAM(load_unbalanced_sam())

    with pytest.warns(
        UserWarning, match="dual solver reported: Did not converge after 1 iterations"
    ):
        estimate = sam.fit(solver="dual", maxiter=1)

    assert estimate.status == "optimal_inaccurate"


def test_known_totals_and_fixed_cells():
    prior = load_unbalanced_sam()
    negative_row, negative_column = np.argwhere(prior.to_numpy() < 0)[0]
    account = prior.index[negative_row]
    fixed_row, fixed_column = (
        prior.index[0],
        prior.columns[np.flatnonzero(prior.iloc[0].to_numpy())[0]],
    )
    sam = SymbolicSAM(prior)

    sam.subject_to(
        income=sam.sel(row=account).sum() == 1.05 * prior.loc[account].sum(),
        fixed=sam.sel(row=fixed_row, col=fixed_column) == prior.loc[fixed_row, fixed_column],
    )
    estimate = sam.fit().sam

    assert_balanced(estimate)
    assert estimate.loc[account].sum() == pytest.approx(1.05 * prior.loc[account].sum(), rel=1e-7)
    assert estimate.loc[fixed_row, fixed_column] == pytest.approx(
        prior.loc[fixed_row, fixed_column], rel=1e-7
    )
    assert estimate.iloc[negative_row, negative_column] < 0


def test_inequality_binding():
    prior = load_unbalanced_sam()
    unconstrained = SymbolicSAM(prior).fit().sam
    cap = 0.95 * unconstrained.loc["Factor"].to_numpy().sum()

    capped_sam = SymbolicSAM(prior)
    capped_sam.subject_to(factor_cap=capped_sam.sel(row="Factor").sum() <= cap)
    loose_sam = SymbolicSAM(prior)
    loose_sam.subject_to(factor_cap=loose_sam.sel(row="Factor").sum() <= 2.0 * cap)

    capped = capped_sam.fit().sam
    loose = loose_sam.fit().sam

    assert capped.loc["Factor"].to_numpy().sum() == pytest.approx(cap, rel=1e-7)
    np.testing.assert_allclose(loose, unconstrained, rtol=SOLVER_RTOL, atol=1e-6)


def test_gras_published_example():
    # Temurshoev, Miller and Bouwmeester (2013), Tables 1 to 3: a 3 x 4 block updated to known row
    # and column totals. The block sits in a square SAM with a closing account, Totals, that is
    # fixed at those totals, so the accounting identity imposes the GRAS row and column constraints.
    rows = ["Goods", "Services", "Net taxes"]
    columns = ["Goods use", "Services use", "Consumption", "Net exports"]
    row_totals = pd.Series([15.0, 26.0, -1.0], index=rows)
    column_totals = pd.Series([9.0, 16.0, 17.0, -2.0], index=columns)

    accounts = [*rows, *columns, "Totals"]
    prior = pd.DataFrame(0.0, index=accounts, columns=accounts)
    prior.loc[rows, columns] = [[7.0, 3.0, 5.0, -3.0], [2.0, 9.0, 8.0, 0.0], [-2.0, 0.0, 2.0, 0.0]]
    prior.loc["Totals", rows] = row_totals
    prior.loc[columns, "Totals"] = column_totals

    sam = SymbolicSAM(prior)
    sam.subject_to(
        row_totals=sam.sel(row="Totals", col=rows) == row_totals,
        column_totals=sam.sel(row=columns, col="Totals") == column_totals,
    )
    estimate = sam.fit(**TIGHT_CLARABEL).sam

    published = [
        [8.424, 3.375, 5.200, -2.000],
        [3.001, 12.625, 10.374, 0.0],
        [-2.425, 0.0, 1.425, 0.0],
    ]
    np.testing.assert_allclose(estimate.loc[rows, columns], published, atol=5e-4)


@pytest.mark.parametrize(
    "objective",
    [KL(weights=2.0), LeastSquares(), LeastSquares(sigma=0.5), ReverseKL(), Pearson()],
    ids=["weighted KL", "least squares", "least squares sigma", "reverse KL", "Pearson"],
)
def test_objective_fixed_point(objective):
    prior = pd.DataFrame([[0.0, 3.0, 0.0], [2.0, 0.0, 0.0], [1.0, -1.0, 0.0]])

    estimate = SymbolicSAM(prior).fit(objective=objective)

    np.testing.assert_allclose(estimate.sam, prior, rtol=SOLVER_RTOL)
    assert estimate.objective == pytest.approx(0.0, abs=1e-6)


def test_least_squares_closed_form():
    # With equality constraints C x = c alone and no cell pushed to zero, weighted least squares has the closed form
    # z - V C' (C V C')^+ (C z - c) with V = diag(sigma^2) (Stone, Champernowne and Meade 1942).
    sam = SymbolicSAM(load_unbalanced_sam())
    prior_cells = sam.prior_values
    sigma = np.abs(prior_cells) * np.linspace(0.5, 2.0, sam.n_cells)

    balance = sam.sum("col") == sam.sum("row").rename({"col": "row"})
    linear = linear_constraints(sam, [balance])
    operator = linear.equality_operator.toarray()
    variance = np.diag(sigma**2)
    gain = variance @ operator.T @ np.linalg.pinv(operator @ variance @ operator.T)
    closed_form = prior_cells - gain @ (operator @ prior_cells - linear.equality_values)
    assert (np.sign(closed_form) == np.sign(prior_cells)).all()

    sigma_grid = np.ones(sam.prior.shape)
    sigma_grid[sam.rows, sam.columns] = sigma
    estimate = sam.fit(objective=LeastSquares(sigma=sigma_grid), **TIGHT_CLARABEL)
    estimate_cells = estimate.sam.to_numpy()[sam.rows, sam.columns]

    np.testing.assert_allclose(estimate_cells, closed_form, rtol=1e-8)
    assert estimate.objective == pytest.approx(
        0.5 * np.sum(((closed_form - prior_cells) / sigma) ** 2), rel=1e-8
    )


@pytest.mark.parametrize(
    "objective_for",
    [lambda weight: KL(weights=1.0 / weight), lambda weight: LeastSquares(sigma=weight)],
    ids=["KL weights", "least squares sigma"],
)
def test_reliable_cell_moves_less(objective_for):
    prior = load_unbalanced_sam()
    sam = SymbolicSAM(prior)
    rest_of_world = ("Other", "Rest of World")
    cell = (rest_of_world, ("Commodity", "Agriculture"))

    # Labels in reverse order check that a DataFrame aligns by label.
    loose = pd.DataFrame(1.0, index=prior.index[::-1], columns=prior.columns[::-1])
    tight = loose.copy()
    tight.loc[cell] = 1e-3
    relative_change = {
        name: abs(
            sam.fit(objective=objective_for(weight), **TIGHT_CLARABEL).sam.loc[cell]
            / prior.loc[cell]
            - 1.0
        )
        for name, weight in {"loose": loose, "tight": tight}.items()
    }

    assert relative_change["tight"] < 1e-2 * relative_change["loose"]


def test_objectives_agree_near_balance():
    # KL, reverse KL and Pearson are all proportional to sum (x - z)^2 / |z| to second order, so near balance their
    # estimates differ by a fraction of the adjustment of order the perturbation, here about 3e-4.
    balanced = SymbolicSAM(load_unbalanced_sam()).fit().sam
    perturbation = np.random.default_rng(0).uniform(-1e-3, 1e-3, size=balanced.shape)
    sam = SymbolicSAM(balanced * (1.0 + perturbation))
    tolerances = {"tol_gap_abs": 1e-11, "tol_gap_rel": 1e-11, "tol_feas": 1e-11}

    kl_estimate = sam.fit().sam.to_numpy()
    adjustment = np.abs(kl_estimate - sam.prior.to_numpy()).max()

    for objective in [ReverseKL(), Pearson()]:
        estimate = sam.fit(objective=objective, **tolerances).sam.to_numpy()
        assert np.abs(estimate - kl_estimate).max() < 1e-2 * adjustment


def test_dataarray_sigma_aligns_by_label():
    prior = pd.DataFrame(
        [[0.0, 5.0, 2.0], [1.0, 0.0, 3.0], [4.0, 6.0, 0.0]], index=list("abc"), columns=list("abc")
    )
    sam = SymbolicSAM(prior)
    sigma = pd.DataFrame(
        [[1.0, 0.5, 2.0], [3.0, 1.0, 0.2], [1.5, 4.0, 1.0]], index=list("cba"), columns=list("bca")
    )
    sigma_array = xr.DataArray(
        sigma.to_numpy(), dims=["row", "col"], coords={"row": sigma.index, "col": sigma.columns}
    )

    from_frame = sam.fit(objective=LeastSquares(sigma=sigma), **TIGHT_CLARABEL).sam
    from_array = sam.fit(objective=LeastSquares(sigma=sigma_array), **TIGHT_CLARABEL).sam

    np.testing.assert_allclose(from_array, from_frame, rtol=1e-9)


@pytest.mark.parametrize(
    "weights, message",
    [(np.ones((2, 3)), "has shape"), (0.0, "positive and finite"), (np.nan, "positive and finite")],
    ids=["wrong shape", "zero", "nan"],
)
def test_invalid_weights_raise(weights, message):
    sam = SymbolicSAM(pd.DataFrame([[0.0, 1.0], [1.0, 0.0]]))

    with pytest.raises(ValueError, match=message):
        sam.fit(objective=KL(weights=weights))


def test_normal_measurement_closed_form():
    # A normal prior on the cells, the exact identity C x = c, and a normal measurement h' x = y with standard deviation
    # r give the estimate solving [[V^-1 + h h' / r^2, C'], [C, 0]] [x; lambda] = [V^-1 z + h y / r^2; c].
    sam = SymbolicSAM(load_unbalanced_sam())
    prior_cells = sam.prior_values
    sigma = np.abs(prior_cells)
    household_income, error = 9893.0, 50.0

    measurement = sam.sel(row=("Institution", "Household")).sum() == household_income
    sam.subject_to(household_income=measurement.normal(sigma=error))
    estimate = sam.fit(objective=LeastSquares(), **TIGHT_CLARABEL)

    balance = sam.sum("col") == sam.sum("row").rename({"col": "row"})
    linear = linear_constraints(sam, [balance])
    identity = linear.equality_operator.toarray()
    loading = linear_constraints(sam, [measurement]).equality_operator.toarray()[0]
    precision = np.diag(1.0 / sigma**2) + np.outer(loading, loading) / error**2
    system = np.block([[precision, identity.T], [identity, np.zeros((identity.shape[0],) * 2)]])
    right_side = np.concatenate(
        [prior_cells / sigma**2 + loading * household_income / error**2, linear.equality_values]
    )
    closed_form = np.linalg.lstsq(system, right_side, rcond=None)[0][: sam.n_cells]

    estimate_cells = estimate.sam.to_numpy()[sam.rows, sam.columns]
    expected_objective = (
        0.5 * np.sum(((closed_form - prior_cells) / sigma) ** 2)
        + 0.5 * ((loading @ closed_form - household_income) / error) ** 2
    )
    np.testing.assert_allclose(estimate_cells, closed_form, rtol=1e-7)
    assert estimate.objective == pytest.approx(expected_objective, rel=1e-7)


@pytest.mark.parametrize(
    "with_error",
    [
        lambda measurement, error: measurement.normal(sigma=error),
        lambda measurement, error: measurement.laplace(scale=error),
        lambda measurement, error: measurement.huber(sigma=error, threshold=1.0),
    ],
    ids=["normal", "laplace", "huber"],
)
def test_measurement_limits(with_error):
    # A precise measurement acts like the exact constraint, a vague one like no constraint, and one in between lands
    # between the two.
    prior = load_unbalanced_sam()
    household = ("Institution", "Household")

    def fit_household_income(error):
        sam = SymbolicSAM(prior)
        sam.subject_to(household_income=with_error(sam.sel(row=household).sum() == 9893.0, error))
        return sam.fit(objective=Pearson(dispersion=1.0), **TIGHT_CLARABEL).sam.loc[household].sum()

    unconstrained_income = (
        SymbolicSAM(prior).fit(objective=Pearson(), **TIGHT_CLARABEL).sam.loc[household].sum()
    )

    assert fit_household_income(1e-3) == pytest.approx(9893.0, abs=1e-2)
    assert fit_household_income(1e9) == pytest.approx(unconstrained_income, rel=1e-4)
    assert unconstrained_income < fit_household_income(100.0) < 9893.0


def test_huber_between_normal_and_laplace():
    # With a threshold far beyond every residual, Huber is the normal penalty. With a threshold k far below them, it is
    # k |r| / sigma up to a constant, the Laplace penalty with scale sigma / k.
    prior = load_unbalanced_sam()
    household = ("Institution", "Household")

    def fit_with(with_error):
        sam = SymbolicSAM(prior)
        sam.subject_to(household_income=with_error(sam.sel(row=household).sum() == 9893.0))
        return sam.fit(objective=Pearson(dispersion=1.0), **TIGHT_CLARABEL).sam

    np.testing.assert_allclose(
        fit_with(lambda measurement: measurement.huber(sigma=100.0, threshold=1e6)),
        fit_with(lambda measurement: measurement.normal(sigma=100.0)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        fit_with(lambda measurement: measurement.huber(sigma=100.0, threshold=1e-3)),
        fit_with(lambda measurement: measurement.laplace(scale=1e5)),
        rtol=1e-6,
    )


def test_measurement_scale_aligns_by_label():
    prior = load_unbalanced_sam()
    agriculture, industry, service = (
        ("Activity", name) for name in ["Agriculture", "Industry", "Service"]
    )
    totals = pd.Series(
        [5500.0, 9800.0, np.nan], index=pd.MultiIndex.from_tuples([agriculture, industry, service])
    )
    errors = pd.Series([200.0, 50.0], index=pd.MultiIndex.from_tuples([industry, agriculture]))

    together = SymbolicSAM(prior)
    together.subject_to(output=(together.sum("col") == totals).normal(sigma=errors))
    separate = SymbolicSAM(prior)
    separate.subject_to(
        agriculture=(separate.sel(row=agriculture).sum() == 5500.0).normal(sigma=50.0),
        industry=(separate.sel(row=industry).sum() == 9800.0).normal(sigma=200.0),
    )

    np.testing.assert_allclose(
        together.fit(objective=Pearson(dispersion=1.0), **TIGHT_CLARABEL).sam,
        separate.fit(objective=Pearson(dispersion=1.0), **TIGHT_CLARABEL).sam,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "objective_in",
    [
        lambda unit: KL(dispersion=2.0 * unit),
        lambda unit: ReverseKL(dispersion=2.0 * unit),
        lambda unit: Pearson(dispersion=2.0 * unit),
        lambda unit: LeastSquares(sigma=np.abs(load_unbalanced_sam().to_numpy()) * unit + unit),
    ],
    ids=["KL", "reverse KL", "Pearson", "least squares"],
)
def test_estimate_is_unit_invariant(objective_in):
    # Expressing every number in thousands, including the dispersion and the standard deviations, must give the same
    # SAM in thousands.
    prior = load_unbalanced_sam()
    household = ("Institution", "Household")

    def fit_in(unit):
        sam = SymbolicSAM(prior * unit)
        measured = sam.sel(row=household).sum() == 9893.0 * unit
        sam.subject_to(household_income=measured.normal(sigma=50.0 * unit))
        return sam.fit(objective=objective_in(unit), **TIGHT_CLARABEL).sam / unit

    np.testing.assert_allclose(fit_in(1e-3), fit_in(1.0), rtol=1e-6)


@pytest.mark.parametrize(
    "objective, measured, warns",
    [
        (KL(), True, True),
        (Pearson(), True, True),
        (KL(), False, False),
        (KL(dispersion=1.0), True, False),
        (LeastSquares(), True, False),
    ],
    ids=["KL measured", "Pearson measured", "KL exact", "KL with dispersion", "least squares"],
)
def test_default_dispersion_warns_with_measurements(objective, measured, warns):
    sam = SymbolicSAM(pd.DataFrame([[0.0, 2.0], [1.0, 0.0]], index=["a", "b"], columns=["a", "b"]))
    income = sam.sel(row="a").sum() == 1.5
    sam.subject_to(income=income.normal(sigma=0.1) if measured else income)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sam.fit(objective=objective)

    assert any("Pass dispersion" in str(warning.message) for warning in caught) == warns


@pytest.mark.parametrize("dispersion", [0.0, -1.0, np.inf], ids=["zero", "negative", "infinite"])
def test_invalid_dispersion_raises(dispersion):
    with pytest.raises(ValueError, match="dispersion must be positive and finite"):
        KL(dispersion=dispersion)


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda sam: (sam.sel(row=0).sum() <= 2.0).normal(sigma=1.0), "Only an equality"),
        (
            lambda sam: (sam.sel(row=0).sum() == 2.0).normal(sigma=1.0).laplace(scale=1.0),
            "already has",
        ),
        (
            lambda sam: (sam.sel(row=0).sum() == 2.0).huber(sigma=1.0, threshold=0.0),
            "threshold must be positive",
        ),
    ],
    ids=["inequality", "two penalties", "huber threshold"],
)
def test_invalid_measurement_raises(build, message):
    sam = SymbolicSAM(pd.DataFrame([[0.0, 1.0], [1.0, 0.0]]))

    with pytest.raises(ValueError, match=message):
        build(sam)


def test_nonpositive_measurement_scale_raises():
    sam = SymbolicSAM(pd.DataFrame([[0.0, 1.0], [1.0, 0.0]]))
    sam.subject_to(income=(sam.sel(row=0).sum() == 2.0).normal(sigma=0.0))

    with pytest.raises(ValueError, match="positive and finite"):
        sam.fit(objective=KL(dispersion=1.0))


@pytest.mark.parametrize("measured", [False, True], ids=["identity only", "normal measurement"])
def test_least_squares_covariance(measured):
    # Normal cells x ~ N(z, V) observed through y = h' x + e with e ~ N(0, r^2) have covariance
    # V1 = V - V h (h' V h + r^2)^-1 h' V, and conditioning on the identity C x = c gives
    # S = V1 - V1 C' (C V1 C')^+ C V1 (Stone, Champernowne and Meade 1942, for the identity alone).
    sam = SymbolicSAM(load_unbalanced_sam())
    variance = np.diag(np.abs(sam.prior_values) ** 2)
    household_income = sam.sel(row=("Institution", "Household")).sum() == 9893.0
    if measured:
        sam.subject_to(household_income=household_income.normal(sigma=50.0))
    estimate = sam.fit(objective=LeastSquares(), **TIGHT_CLARABEL)

    if measured:
        loading = linear_constraints(sam, [household_income]).equality_operator.toarray()[0]
        gain = variance @ loading / (loading @ variance @ loading + 50.0**2)
        variance = variance - np.outer(gain, loading @ variance)
    balance = sam.sum("col") == sam.sum("row").rename({"col": "row"})
    identity = linear_constraints(sam, [balance]).equality_operator.toarray()
    expected = (
        variance
        - variance
        @ identity.T
        @ np.linalg.pinv(identity @ variance @ identity.T)
        @ identity
        @ variance
    )

    covariance = estimate.covariance()

    np.testing.assert_allclose(
        covariance.to_numpy(), expected, rtol=1e-6, atol=1e-6 * expected.max()
    )
    assert covariance.index[0] == (sam.prior.index[sam.rows[0]], sam.prior.columns[sam.columns[0]])


def test_covariance_with_small_cells():
    # Cells of 5 next to cells of 1e7 are far from zero relative to their own size. The identity makes each cell equal
    # its mirror, so a pair of independent normal cells with standard deviation sigma becomes one value with variance
    # sigma^2 / 2, shared by both cells of the pair. The cells are (0, 1), (0, 2), (1, 0) and (2, 0).
    prior = pd.DataFrame([[0.0, 1e7, 5.0], [1e7, 0.0, 0.0], [5.0, 0.0, 0.0]])
    estimate = SymbolicSAM(prior).fit(objective=LeastSquares(), **TIGHT_CLARABEL)

    expected = np.zeros((4, 4))
    expected[np.ix_([0, 2], [0, 2])] = 1e7**2 / 2.0
    expected[np.ix_([1, 3], [1, 3])] = 5.0**2 / 2.0

    np.testing.assert_allclose(estimate.covariance().to_numpy(), expected, rtol=1e-6, atol=1e-2)


def test_covariance_of_disconnected_accounts():
    # Accounts a and b trade only with each other, as do c and d, so each pair of mirrored cells is one value with
    # variance sigma^2 / 2 and the two groups are independent. The cells are (a, b), (b, a), (c, d) and (d, c).
    prior = pd.DataFrame(
        [[0.0, 4.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 6.0], [0.0, 0.0, 6.0, 0.0]],
        index=list("abcd"),
        columns=list("abcd"),
    )
    estimate = SymbolicSAM(prior).fit(objective=LeastSquares(), **TIGHT_CLARABEL)

    expected = np.zeros((4, 4))
    expected[np.ix_([0, 1], [0, 1])] = 4.0**2 / 2.0
    expected[np.ix_([2, 3], [2, 3])] = 6.0**2 / 2.0

    np.testing.assert_allclose(estimate.covariance().to_numpy(), expected, rtol=1e-6, atol=1e-6)


def test_covariance_without_equalities():
    # Diagonal cells enter no account's imbalance, so nothing conditions them and each keeps its prior variance.
    prior = pd.DataFrame(np.diag([2.0, 3.0]), index=["a", "b"], columns=["a", "b"])
    estimate = SymbolicSAM(prior).fit(objective=LeastSquares(), solver=cp.CLARABEL)

    np.testing.assert_allclose(estimate.covariance().to_numpy(), np.diag([4.0, 9.0]), rtol=1e-6)


def test_covariance_of_selected_cells():
    sam = SymbolicSAM(load_unbalanced_sam())
    household = ("Institution", "Household")
    sam.subject_to(household_income=(sam.sel(row=household).sum() == 9893.0).normal(sigma=50.0))
    estimate = sam.fit(objective=LeastSquares(), **TIGHT_CLARABEL)
    cells = [
        (household, ("Factor", "Labor")),
        (("Commodity", "Service"), household),
        (household, ("Factor", "Unskilled Labor")),
    ]

    selected = estimate.covariance(cells=cells)
    full = estimate.covariance()
    positions = [list(full.index).index(cell) for cell in cells]

    assert list(selected.index) == cells
    np.testing.assert_allclose(
        selected.to_numpy(), full.to_numpy()[np.ix_(positions, positions)], rtol=1e-12
    )


@pytest.mark.parametrize(
    "cells, message",
    [
        ([("a", "z")], "labels not in the prior"),
        ([("a", "a")], "zero in the prior"),
    ],
    ids=["unknown account", "zero cell"],
)
def test_covariance_rejects_cells(cells, message):
    sam = SymbolicSAM(pd.DataFrame([[0.0, 2.0], [2.0, 0.0]], index=["a", "b"], columns=["a", "b"]))
    estimate = sam.fit(objective=LeastSquares(), solver=cp.CLARABEL)

    with pytest.raises(ValueError, match=message):
        estimate.covariance(cells=cells)


@pytest.mark.parametrize(
    "objective, constraint, message",
    [
        (KL(), None, "only for the LeastSquares objective"),
        (
            LeastSquares(),
            lambda sam: (sam.sel(row=0).sum() == 4.0).laplace(scale=1.0),
            "every measurement error is normal",
        ),
        (LeastSquares(), lambda sam: sam.sel(row=0, col=1) <= 2.0, "An inequality binds"),
        (LeastSquares(), lambda sam: sam.sel(row=0, col=1) == 0.0, "A cell sits at zero"),
        (
            LeastSquares(),
            lambda sam: sam.sum("col") == sam.sum("row").rename({"col": "row"}),
            "linearly dependent",
        ),
    ],
    ids=[
        "KL",
        "laplace measurement",
        "binding inequality",
        "cell at zero",
        "identity imposed twice",
    ],
)
def test_covariance_raises(objective, constraint, message):
    # The prior is balanced, so the cap of 2 on its cell of 3 binds and the fixed zero holds exactly.
    sam = SymbolicSAM(pd.DataFrame([[0.0, 3.0, 1.0], [2.0, 0.0, 1.0], [2.0, 1.0, 0.0]]))
    if constraint is not None:
        sam.subject_to(data=constraint(sam))
    estimate = sam.fit(objective=objective, solver=cp.CLARABEL)

    with pytest.raises(ValueError, match=message):
        estimate.covariance()


@pytest.mark.parametrize(
    "objective, with_error",
    [
        (KL(), None),
        (LeastSquares(), lambda measurement: measurement.normal(sigma=50.0)),
        (Pearson(dispersion=20.0), lambda measurement: measurement.laplace(scale=50.0)),
        (
            ReverseKL(dispersion=0.5),
            lambda measurement: measurement.huber(sigma=50.0, threshold=1.0),
        ),
    ],
    ids=["KL", "least squares, normal", "Pearson, laplace", "reverse KL, huber"],
)
def test_objective_value_at_estimate(objective, with_error):
    sam = SymbolicSAM(load_unbalanced_sam())
    if with_error is not None:
        sam.subject_to(
            household_income=with_error(sam.sel(row=("Institution", "Household")).sum() == 9893.0)
        )
    estimate = sam.fit(objective=objective)

    assert sam.objective_value(estimate.sam, objective=objective) == pytest.approx(
        estimate.objective, rel=1e-6
    )


def test_objective_value_at_prior():
    # Every objective is zero at the prior, so what remains is the measurement penalty alone.
    prior = load_unbalanced_sam()
    household = ("Institution", "Household")
    sam = SymbolicSAM(prior)
    sam.subject_to(household_income=(sam.sel(row=household).sum() == 9893.0).normal(sigma=50.0))

    expected = 0.5 * ((prior.loc[household].sum() - 9893.0) / 50.0) ** 2

    assert sam.objective_value(prior, objective=LeastSquares()) == pytest.approx(
        expected, rel=1e-12
    )


@pytest.mark.parametrize(
    "as_prior, index_column",
    [
        (lambda frame: frame, None),
        (
            lambda frame: xr.DataArray(
                frame.to_numpy(),
                dims=["row", "col"],
                coords={"row": frame.index, "col": frame.columns},
            ),
            None,
        ),
        (
            lambda frame: pl.DataFrame(
                {"account": list(frame.index), **{name: frame[name].to_numpy() for name in frame}}
            ),
            "account",
        ),
    ],
    ids=["pandas", "xarray", "polars"],
)
def test_objective_value_takes_prior_type(as_prior, index_column):
    frame = pd.DataFrame([[0.0, 2.0], [1.0, 0.0]], index=["a", "b"], columns=["a", "b"])
    sam = SymbolicSAM(as_prior(frame), index_column=index_column)
    candidate = sam.like_prior(sam.prior_values * 1.5)

    expected = np.sum(3.0 * np.log(1.5) - 3.0 + 2.0) + np.sum(1.5 * np.log(1.5) - 1.5 + 1.0)

    assert sam.objective_value(candidate) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "candidate, message",
    [
        (
            pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], index=["b", "a"], columns=["b", "a"]),
            "account labels",
        ),
        (
            pd.DataFrame([[1.0, 1.0], [1.0, 0.0]], index=["a", "b"], columns=["a", "b"]),
            "where the prior is zero",
        ),
        (
            pd.DataFrame([[0.0, -1.0], [1.0, 0.0]], index=["a", "b"], columns=["a", "b"]),
            "opposite sign",
        ),
        (
            pd.DataFrame([[np.nan, 1.0], [1.0, 0.0]], index=["a", "b"], columns=["a", "b"]),
            "must be finite",
        ),
        (
            pd.DataFrame([[0.0, np.nan], [1.0, 0.0]], index=["a", "b"], columns=["a", "b"]),
            "must be finite",
        ),
    ],
    ids=["labels", "new cell", "sign", "NaN in a zero cell", "NaN in a prior cell"],
)
def test_objective_value_rejects_candidate(candidate, message):
    sam = SymbolicSAM(pd.DataFrame([[0.0, 2.0], [1.0, 0.0]], index=["a", "b"], columns=["a", "b"]))

    with pytest.raises(ValueError, match=message):
        sam.objective_value(candidate)


@pytest.mark.parametrize(
    "objective, formula",
    [
        (KL(), lambda x, z: np.sum(x * np.log(x / z) - x + z)),
        (ReverseKL(), lambda x, z: np.sum(z * np.log(z / x) - z + x)),
        (Pearson(), lambda x, z: 0.5 * np.sum((x - z) ** 2 / z)),
        (KL(dispersion=4.0), lambda x, z: np.sum(x * np.log(x / z) - x + z) / 4.0),
        (LeastSquares(sigma=25.0), lambda x, z: 0.5 * np.sum(((x - z) / 25.0) ** 2)),
    ],
    ids=["KL", "reverse KL", "Pearson", "KL with dispersion", "least squares"],
)
def test_objective_value_matches_formula(objective, formula):
    # Away from the prior, where every objective is nonzero, checked against each formula on the cell magnitudes.
    prior = load_unbalanced_sam()
    sam = SymbolicSAM(prior)
    candidate = prior * 1.1

    prior_magnitudes = np.abs(sam.prior_values)
    expected = formula(1.1 * prior_magnitudes, prior_magnitudes)

    assert sam.objective_value(candidate, objective=objective) == pytest.approx(expected, rel=1e-10)
