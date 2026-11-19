import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from cge_modeling.compile.pytensor import xtensor_cge_primitives_to_pytensor
from cge_modeling.sparsity import jacobian_sparsity_vjp
from tests.utilities import models


def nonzero_point_near(calibrated, tensors, rng):
    """
    Perturb each calibrated value by up to 10% and replace zeros with positive values, so every share stays in
    its domain and no entry of the Jacobian is hidden by a zero coefficient.
    """
    point = []
    for tensor in tensors:
        value = np.broadcast_to(np.asarray(calibrated[tensor.name], dtype=float), tensor.type.shape)
        value = np.where(value == 0, rng.uniform(0.1, 0.5, size=value.shape), value)
        point.append(value * rng.uniform(0.9, 1.1, size=value.shape))
    return point


@pytest.mark.parametrize(
    "build, calibrate, data",
    [
        (models.build_model_1_in_context, models.calibrate_model_1, models.model_1_data),
        (models.build_model_2_in_context, models.calibrate_model_2, models.model_2_data),
        (models.build_model_3_in_context, models.calibrate_model_3, models.model_3_data),
        (
            lambda: models.build_model_4_in_context(closure="savings_driven"),
            models.calibrate_model_4,
            models.model_4_data,
        ),
        (models.build_model_5_in_context, models.calibrate_model_5, models.model_5_data),
    ],
    ids=["model_1", "model_2", "model_3", "model_4", "model_5"],
)
def test_structural_pattern_matches_the_numeric_jacobian_at_any_point(build, calibrate, data):
    system, variables, parameters = xtensor_cge_primitives_to_pytensor(build())
    wrt = [*variables, *parameters]
    jacobian = pt.concatenate(
        [
            pt.jacobian(system, tensor, vectorize=True).reshape((system.shape[0], -1))
            for tensor in wrt
        ],
        axis=1,
    )
    pattern_and_jacobian = pytensor.function(
        wrt, [jacobian_sparsity_vjp(system, wrt), jacobian], on_unused_input="ignore"
    )
    calibrated = calibrate(**data)
    rng = np.random.default_rng(0)

    pattern, numeric = pattern_and_jacobian(*nonzero_point_near(calibrated, wrt, rng))
    pattern_elsewhere, _ = pattern_and_jacobian(*nonzero_point_near(calibrated, wrt, rng))
    pattern_at_calibration, _ = pattern_and_jacobian(
        *[
            np.broadcast_to(np.asarray(calibrated[tensor.name], dtype=float), tensor.type.shape)
            for tensor in wrt
        ]
    )

    assert np.isfinite(numeric).all()
    np.testing.assert_array_equal(pattern, np.abs(numeric) > 0)
    np.testing.assert_array_equal(pattern, pattern_elsewhere)
    np.testing.assert_array_equal(pattern, pattern_at_calibration)
