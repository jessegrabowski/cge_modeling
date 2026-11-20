import numpy as np

from cge_modeling.compile.sparse import SparseJacobianLayout, color_jacobian_columns


def test_coloring_never_shares_a_color_within_a_row_of_a_tridiagonal_pattern():
    n = 30
    pattern = np.eye(n, dtype=bool) | np.eye(n, k=1, dtype=bool) | np.eye(n, k=-1, dtype=bool)

    colors = color_jacobian_columns(pattern)

    for row in pattern:
        row_colors = colors[row]
        assert len(set(row_colors)) == len(row_colors)
    assert colors.max() + 1 == 3


def test_layout_assembles_the_jacobian_from_its_compressed_columns():
    rng = np.random.default_rng(0)
    pattern = rng.uniform(size=(12, 15)) < 0.2
    jacobian = np.where(pattern, rng.normal(size=pattern.shape), 0.0)
    colors = color_jacobian_columns(pattern)
    seeds = (colors[None, :] == np.arange(colors.max() + 1)[:, None]).astype(float)

    assembled = SparseJacobianLayout.from_pattern(pattern, colors).assemble(seeds @ jacobian.T)

    np.testing.assert_array_equal(assembled.toarray(), jacobian)
