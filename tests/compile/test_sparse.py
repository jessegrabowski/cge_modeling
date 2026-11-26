import numpy as np
import pytest

from cge_modeling.base.utilities import variable_dict_to_flat_array
from cge_modeling.compile.sparse import SparseJacobianLayout, color_jacobian_columns
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


@pytest.mark.parametrize(
    "build, calibrated_values",
    [
        (build_model_1, lambda: calibrate_model_1(**model_1_data)),
        (build_model_2, lambda: calibrate_model_2(**model_2_data)),
        (build_model_3, lambda: calibrate_model_3(**model_3_data)),
        (
            lambda: build_model_4(closure="savings_driven"),
            lambda: calibrate_model_4(**model_4_data),
        ),
        (build_model_5, lambda: calibrate_model_5(**model_5_data)),
    ],
    ids=["model_1", "model_2", "model_3", "model_4", "model_5"],
)
def test_sparse_jacobian_matches_the_dense_jacobian(build, calibrated_values):
    dense = build().compile(functions_to_compile=["root"])
    sparse = build().compile(functions_to_compile=["root"], use_sparse_matrices=True)
    x, theta = variable_dict_to_flat_array(calibrated_values(), dense.variables, dense.parameters)
    x = x * np.random.default_rng(0).uniform(0.9, 1.1, size=x.shape)

    residual, jacobian = sparse.f_sparse_system_and_jac(x, theta)

    np.testing.assert_allclose(residual, dense.f_system(x, theta))
    np.testing.assert_allclose(jacobian.toarray(), dense.f_jac(x, theta), rtol=1e-10, atol=1e-10)
    assert jacobian.nnz < jacobian.shape[0] * jacobian.shape[1]


@pytest.mark.parametrize(
    "model_id, build, calibrated_values",
    [
        (1, build_model_1, lambda: calibrate_model_1(**model_1_data)),
        (2, build_model_2, lambda: calibrate_model_2(**model_2_data)),
    ],
    ids=["model_1", "model_2"],
)
def test_sparse_simulation_matches_the_dense_simulation(model_id, build, calibrated_values):
    dense = load_and_cache_model(model_id)
    sparse = build().compile(functions_to_compile=["root", "euler"], use_sparse_matrices=True)
    initial_state = calibrated_values()
    shock = {"L_s": 0.1}

    dense_result = dense.simulate(
        initial_state, final_delta_pct=shock, n_iter_euler=20, progressbar=False
    )
    sparse_result = sparse.simulate(
        initial_state, final_delta_pct=shock, n_iter_euler=20, progressbar=False
    )

    assert sparse_result["optim_res"].success
    for stage in ("euler", "optimizer"):
        for name, values in dense_result[stage]["variables"].dataset.data_vars.items():
            np.testing.assert_allclose(
                sparse_result[stage]["variables"][name].values, values.values, rtol=1e-8, atol=1e-8
            )


def test_sparse_matrices_reject_the_scan_euler_approximation():
    with pytest.raises(ValueError, match="use_scan_euler=False"):
        build_model_1().compile(
            functions_to_compile=["euler"],
            use_scan_euler=True,
            use_sparse_matrices=True,
        )


def test_sparse_matrices_reject_rk4_euler_steps():
    with pytest.raises(NotImplementedError, match="RK4"):
        build_model_1().compile(
            functions_to_compile=["euler"], use_rk4=True, use_sparse_matrices=True
        )


@pytest.mark.parametrize("method", ["lm", "hybr"])
def test_sparse_root_rejects_any_scipy_method(method):
    model = build_model_1().compile(functions_to_compile=["root"], use_sparse_matrices=True)
    initial_state = calibrate_model_1(**model_1_data)

    with pytest.raises(ValueError, match="solver="):
        model.simulate(
            initial_state,
            final_delta_pct={"L_s": 0.1},
            use_euler_approximation=False,
            method=method,
            progressbar=False,
        )
