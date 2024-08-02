import numpy as np
import pandas as pd

from cge_modeling.tools.rebalance import SAMTransformer, balance_SAM


def test_SAM_transformer():
    data = np.array([[0, 0, 1], [0, 0, 0], [0, -1, 0]])

    df = pd.DataFrame(data)
    transformer = SAMTransformer()
    transformer.fit(df)
    assert transformer.scale == 2
    assert np.all(
        transformer.negative_mask
        == np.array([[False, False, False], [False, False, False], [False, True, False]])
    )

    transformed = transformer.transform(df)
    assert np.all(transformed == np.array([[0, 0, 0.5], [0, 0, 0.5], [0, 0, 0]]))

    inverse = transformer.inverse_transform(transformed)
    assert np.all(inverse == np.array([[0, 0, 1], [0, 0, 0], [0, -1, 0]]))


def test_cross_entropy_rebalance():
    df = pd.read_csv("tests/data/unbalanced_sam.csv", index_col=[0, 1], header=[0, 1])
    df2, optim_res = balance_SAM(df)
    assert optim_res.success
    assert np.allclose(df2.sum(axis=0), df2.sum(axis=1))
