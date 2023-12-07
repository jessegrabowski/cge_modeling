from typing import Callable

import numpy as np
import pytest

from cge_modeling.base.cge import CGEModel
from cge_modeling.base.primitives import Parameter, Variable
from tests.utilities.models import (
    calibrate_model_1,
    calibrate_model_2,
    load_model_1,
    load_model_2,
    model_1_data,
    model_2_data,
)


@pytest.fixture()
def model():
    return CGEModel(coords={"i": ["A", "B", "C"]}, compile=False)


@pytest.mark.parametrize(
    "cls, cls_type",
    [(Variable, "variable"), (Parameter, "parameter")],
    ids=["variable", "parameter"],
)
def test_add_object(cls, cls_type, model):
    x = cls(name="x")

    getattr(model, f"add_{cls_type}")(x)

    names = getattr(model, f"{cls_type}_names")
    objects = getattr(model, f"{cls_type}s")

    assert x.name in names
    assert x in objects


@pytest.mark.parametrize(
    "cls, cls_type",
    [(Variable, "variable"), (Parameter, "parameter")],
    ids=["variable", "parameter"],
)
def test_add_objects(cls, cls_type, model):
    x = cls(name="x")
    y = cls(name="y")

    getattr(model, f"add_{cls_type}s")([x, y])

    names = getattr(model, f"{cls_type}_names")
    objects = getattr(model, f"{cls_type}s")
    assert isinstance(getattr(model, f"_{cls_type}s"), dict)

    for item in [x, y]:
        assert item.name in names
        assert item in objects


@pytest.mark.parametrize(
    "cls, cls_type",
    [(Variable, "variable"), (Parameter, "parameter")],
    ids=["variable", "parameter"],
)
def test_add_wrong_type_raises(cls, cls_type, model):
    other_type = Variable if cls is Parameter else Parameter
    x = other_type(name="x")

    with pytest.raises(ValueError, match=f"Expected instance of type {cls_type.capitalize()}"):
        getattr(model, f"add_{cls_type}")(x)


@pytest.mark.parametrize(
    "cls, cls_type",
    [(Variable, "variable"), (Parameter, "parameter")],
    ids=["variable", "parameter"],
)
def test_get_object(cls, cls_type, model):
    x = cls(name="x")
    getattr(model, f"add_{cls_type}")(x)

    x_out = getattr(model, f"get_{cls_type}")("x")
    assert x_out is x


@pytest.mark.parametrize(
    "cls, cls_type",
    [(Variable, "variable"), (Parameter, "parameter")],
    ids=["variable", "parameter"],
)
@pytest.mark.parametrize("get_args", [["x", "y"], None], ids=["explicit", "default"])
def test_get_objects(cls, cls_type, get_args, model):
    x = cls(name="x")
    y = cls(name="y")
    getattr(model, f"add_{cls_type}s")([x, y])

    x_out, y_out = getattr(model, f"get_{cls_type}s")(get_args)

    assert x_out is x
    assert y_out is y


@pytest.mark.parametrize("args", [["x"], ["y"], ["x", "y"], None], ids=["x", "y", "x,y", "default"])
def test_get(args, model):
    x = Variable(name="x")
    y = Parameter(name="y")
    model.add_variable(x)
    model.add_parameter(y)

    out = model.get(args)
    out_names = [d["name"] for d in out] if isinstance(out, list) else out["name"]

    # None case gets back everything
    if args is None:
        args = model.variable_names + model.parameter_names

    assert all([x in out_names for x in args])


@pytest.mark.parametrize(
    "model_function, calibrate_model, data",
    [
        (load_model_1, calibrate_model_1, model_1_data),
        (load_model_2, calibrate_model_2, model_2_data),
    ],
    ids=["simple_model", "3-goods simple"],
)
def test_model_gradients(model_function, calibrate_model, data):
    pass


@pytest.mark.parametrize(
    "model_function, calibrate_model, data",
    [
        (load_model_1, calibrate_model_1, model_1_data),
        (load_model_2, calibrate_model_2, model_2_data),
    ],
    ids=["simple_model", "3-goods simple"],
)
def test_backends_agree(model_function: Callable, calibrate_model: Callable, data: dict):
    model_numba = model_function(backend="numba")
    model_pytensor = model_function(
        backend="pytensor", mode="FAST_COMPILE", parse_equations_to_sympy=False
    )

    def solver_agreement_checks(results: list, names: list):
        for res, name in zip(results, names):
            assert res.success, f"{name} solver failed to converge"
            assert not np.all(np.isnan(res.x)), f"{name} solver returned NaNs"
            assert np.all(np.isfinite(res.x)), f"{name} solver returned Infs"

        np.testing.assert_allclose(
            np.around(results[0].x, 4), np.around(results[1].x, 4), atol=1e-5, rtol=1e-5
        ), "Solvers disagree"

    calibated_data = calibrate_model(**data)
    x0 = np.concatenate(
        [np.atleast_1d(calibated_data[x]).ravel() for x in model_numba.variable_names], axis=0
    )
    theta0 = np.concatenate(
        [np.atleast_1d(calibated_data[x]).ravel() for x in model_numba.parameter_names], axis=0
    )

    resid_numba = model_numba.f_system(x0, theta0)
    resid_pytensor = model_pytensor.f_system(**calibated_data)

    np.testing.assert_allclose(resid_numba, 0, atol=1e-8)
    np.testing.assert_allclose(resid_pytensor, 0, atol=1e-8)

    labor_increase = calibated_data.copy()
    labor_increase["L_s"] = 10_000
    theta_labor_increase = np.concatenate(
        [np.atleast_1d(labor_increase[x]).ravel() for x in model_numba.parameter_names], axis=0
    )

    # Test the root finder
    res_numba = model_numba._solve_with_root(calibated_data, theta_labor_increase)
    res_pytensor = model_pytensor._solve_with_root(calibated_data, theta_labor_increase)
    solver_agreement_checks([res_numba, res_pytensor], ["Numba", "PyTensor"])

    # Test minimization of residuals
    res_numba = model_numba._solve_with_minimize(
        calibated_data, theta_labor_increase, method="trust-ncg", tol=1e-8
    )
    res_pytensor = model_pytensor._solve_with_minimize(
        calibated_data, theta_labor_increase, method="trust-ncg", tol=1e-8
    )

    solver_agreement_checks([res_numba, res_pytensor], ["Numba", "PyTensor"])

    # Test Euler approximation
    res_numba = model_numba._solve_with_euler_approximation(
        calibated_data, theta_final=theta_labor_increase, n_steps=10_000
    )
    res_pytensor = model_pytensor._solve_with_euler_approximation(
        calibated_data, theta_final=theta_labor_increase, n_steps=10_000
    )

    res_numba = res_numba.parameters.isel(step=-1).to_array().values
    res_pytensor = res_pytensor.parameters.isel(step=-1).to_array().values

    np.testing.assert_allclose(res_numba, res_pytensor)
