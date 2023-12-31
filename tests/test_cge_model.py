from typing import Callable

import numpy as np
import pytest

from cge_modeling.base.cge import CGEModel
from cge_modeling.base.primitives import Parameter, Variable
from cge_modeling.base.utilities import variable_dict_to_flat_array
from tests.utilities.fake_data import generate_data
from tests.utilities.models import (
    calibrate_model_1,
    calibrate_model_2,
    expected_model_1_jacobian,
    expected_model_2_jacobian,
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
    "model_function, jac_function",
    [
        (load_model_1, expected_model_1_jacobian),
        (load_model_2, expected_model_2_jacobian),
    ],
    ids=["simple_model", "3-goods simple"],
)
@pytest.mark.parametrize("backend", ["numba", "pytensor"], ids=["numba", "pytensor"])
def test_model_gradients(model_function, jac_function, backend):
    mode = "FAST_COMPILE" if backend == "pytensor" else None
    mod = model_function(backend=backend, parse_equations_to_sympy=backend == "numba", mode=mode)
    data = generate_data(mod.variables + mod.parameters, mod.coords)

    J_expected = jac_function(**data)

    if backend == "numba":
        x, theta = variable_dict_to_flat_array(data, mod.variables, mod.parameters)
        J_actual = mod.f_jac(x, theta)
    else:
        J_actual = mod.f_jac(**data)

    np.testing.assert_allclose(J_expected, J_actual, atol=1e-8)


@pytest.mark.parametrize(
    "model_function, calibrate_model, data",
    [
        (load_model_1, calibrate_model_1, model_1_data),
        (load_model_2, calibrate_model_2, model_2_data),
    ],
    ids=["simple_model", "3-goods simple"],
)
@pytest.mark.parametrize(
    "method, solver_kwargs",
    [
        ("_solve_with_minimize", {"method": "trust-exact"}),
        ("_solve_with_root", {"method": "hybr"}),
        ("_solve_with_euler_approximation", {"n_steps": 500}),
    ],
    ids=["minimize", "root", "euler"],
)
def test_backends_agree(
    model_function: Callable,
    calibrate_model: Callable,
    data: dict,
    method: str,
    solver_kwargs: dict,
):
    model_numba = model_function(backend="numba")
    model_pytensor = model_function(
        backend="pytensor", mode="FAST_COMPILE", parse_equations_to_sympy=False
    )

    def solver_agreement_checks(results: list, names: list):
        for res, name in zip(results, names):
            if hasattr(res, "success"):
                assert res.success, f"{name} solver failed to converge"
            if hasattr(res, "x"):
                assert not np.all(np.isnan(res.x)), f"{name} solver returned NaNs"
                assert np.all(np.isfinite(res.x)), f"{name} solver returned Infs"

                np.testing.assert_allclose(
                    np.around(results[0].x, 4), np.around(results[1].x, 4), atol=1e-5, rtol=1e-5
                ), "Solvers disagree"

            else:
                assert not np.all(np.isnan(res)), f"{name} solver returned NaNs"
                assert np.all(np.isfinite(res)), f"{name} solver returned Infs"

                np.testing.assert_allclose(
                    np.around(results[0], 4), np.around(results[1], 4), atol=1e-5, rtol=1e-5
                ), "Solvers disagree"

    calibated_data = calibrate_model(**data)
    x0, theta0 = variable_dict_to_flat_array(
        calibated_data, model_numba.variables, model_numba.parameters
    )
    resid_numba = model_numba.f_system(x0, theta0)
    resid_pytensor = model_pytensor.f_system(**calibated_data)

    np.testing.assert_allclose(resid_numba, 0, atol=1e-8)
    np.testing.assert_allclose(resid_pytensor, 0, atol=1e-8)

    labor_increase = calibated_data.copy()
    labor_increase["L_s"] = 10_000

    _, theta_labor_increase = variable_dict_to_flat_array(
        labor_increase, model_numba.variables, model_numba.parameters
    )

    res_numba = getattr(model_numba, method)(calibated_data, theta_labor_increase, **solver_kwargs)
    res_pytensor = getattr(model_pytensor, method)(
        calibated_data, theta_labor_increase, **solver_kwargs
    )

    if method == "_solve_with_euler_approximation":
        res_numba = res_numba.parameters.isel(step=-1).to_array().values
        res_pytensor = res_pytensor.parameters.isel(step=-1).to_array().values

    solver_agreement_checks([res_numba, res_pytensor], ["Numba", "PyTensor"])
