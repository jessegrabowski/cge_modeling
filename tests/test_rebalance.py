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


def test_cross_entropy_rebalance():
    df = pd.read_csv("tests/data/unbalanced_sam.csv", index_col=[0, 1], header=[0, 1])
    df2, optim_res = balance_SAM(df, maxiter=5000)
    assert optim_res.success
    assert np.allclose(df2.sum(axis=0), df2.sum(axis=1))
