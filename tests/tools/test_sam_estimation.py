from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from cge_modeling.tools.sam_estimation import SymbolicSAM, cells_forced_to_zero, linear_constraints

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
    ],
    ids=[
        "mismatched_dims",
        "product_of_cells",
        "strict_inequality",
        "constraint_as_condition",
        "unknown_series_label",
        "array_operand",
        "dataframe_against_one_dim",
        "unknown_dataframe_column_label",
        "dataarray_with_extra_dim",
        "dataarray_on_the_left",
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


def test_infeasible_fit_warns():
    sam = SymbolicSAM(pd.DataFrame([[0.0, 2.0], [1.0, 0.0]], index=["a", "b"], columns=["a", "b"]))
    sam.subject_to(income=sam.sel(row="a").sum() == 5.0, spending=sam.sel(col="a").sum() == 7.0)

    with pytest.warns(UserWarning, match="infeasible"):
        estimate = sam.fit()

    assert estimate.status == "infeasible"
    assert np.isnan([estimate.sam.loc["a", "b"], estimate.sam.loc["b", "a"]]).all()


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


def test_dual_rejects_constraints():
    sam = SymbolicSAM(pd.DataFrame([[0.0, 1.0], [1.0, 0.0]]))
    sam.subject_to(cap=sam.sel(row=0).sum() <= 2.0)

    with pytest.raises(ValueError, match="only the accounting identity"):
        sam.fit(solver="dual")


def test_dual_solver_kwargs():
    sam = SymbolicSAM(load_unbalanced_sam())

    with pytest.warns(UserWarning, match="optimal_inaccurate"):
        estimate = sam.fit(solver="dual", options={"maxiter": 1})

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
