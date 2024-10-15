import numpy as np
import pandas as pd
import pytest

from cge_modeling.tools.rebalance import SAMTransformer, balance_SAM


@pytest.mark.parametrize("index", ["simple", "multiindex"])
def test_SAM_transformer(index):
    data = np.array([[0, 0, 3], [0, 0, 0], [0, -1, 0]])

    if index == "simple":
        df = pd.DataFrame(data)
    else:
        idx = pd.MultiIndex.from_tuples([("A", 1), ("B", 2), ("C", 3)])
        df = pd.DataFrame(data, index=idx, columns=idx)

    transformer = SAMTransformer()
    transformer.fit(df)
    assert transformer.scale == 2

    if index == "simple":
        assert transformer.negative_locs == [(2, 1)]
    else:
        assert transformer.negative_locs == [(("C", 3), ("B", 2))]

    transformed = transformer.transform(df)
    assert np.all(transformed == np.array([[0, 0, 1.5], [0, 0, 0.5], [0, 0, 0]]))

    inverse = transformer.inverse_transform(transformed)
    assert np.all(inverse == np.array([[0, 0, 3], [0, 0, 0], [0, -1, 0]]))


@pytest.mark.parametrize("backend, method", [("cvxpy", "CLARABEL"), ("scipy", "SLSQP")])
def test_cross_entropy_rebalance(backend, method):
    df = pd.read_csv("tests/data/unbalanced_sam.csv", index_col=[0, 1], header=[0, 1]).fillna(0.0)
    if backend == "scipy":
        minimizer_kwargs = {"maxiter": 5000}
    else:
        minimizer_kwargs = {"delta": 1e-4}

    df2, optim_res, success = balance_SAM(
        df, use_cvxpy=backend == "cvxpy", method=method, **minimizer_kwargs
    )
    assert success
    assert np.allclose(df2.sum(axis=0), df2.sum(axis=1))
