from collections.abc import Callable

import numpy as np
import pytest
import sympy as sp

from cge_modeling.base.cge import CGEModel
from cge_modeling.base.primitives import Equation, Parameter, Variable
from cge_modeling.base.utilities import variable_dict_to_flat_array
from cge_modeling.tools.sympy_tools import get_inputs
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
    return CGEModel(coords={"i": ["A", "B", "C"]}, compile=None)


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


def test_add_equation(model):
    z = Variable(name="z")
    x = Variable(name="x")
    y = Parameter(name="y")
    eq = Equation(name="simple sum", equation="z = x + y")

    model.add_variables([x, z])
    model.add_parameter(y)
    model.add_equations([eq])

    assert "simple sum" in model.equation_names


def test_unpack_equation(model):
    z = Variable(name="z", dims="i")
    x = Variable(name="x", dims="i")
    y = Parameter(name="y")

    eq = Equation(name="sum for group <dim:i>", equation="z = x + y")
    model.parse_equations_to_sympy = True

    model._initialize_group([x, z], "variables")
    model._initialize_group([y], "parameters")
    model._initialize_group([eq], "equations")
    model._simplify_unpacked_sympy_representation()

    assert len(model.unpacked_equation_symbols) == 3
    assert all([f"sum for group {i}" in model.unpacked_equation_names for i in ["A", "B", "C"]])
    for i, eq in zip(["A", "B", "C"], model.unpacked_equation_symbols):
        inputs = get_inputs(eq)
        f = sp.lambdify(inputs, eq)
        kwargs = {arg.name: np.random.normal() for arg in inputs}
        np.testing.assert_allclose(f(**kwargs), (kwargs[f"z_{i}"] - kwargs[f"x_{i}"] - kwargs["y"]))


def test_unpack_equation_with_sum():
    model = CGEModel(
        coords={"i": ["A", "B", "C"], "j": ["A", "B"]},
        compile=None,
        parse_equations_to_sympy=True,
    )

    z = Variable(name="z", dims="i")
    x = Variable(name="x", dims=("i", "j"))
    y = Parameter(name="y")

    eq = Equation(name="sum for group <dim:i>", equation="z = y + Sum(x, (j, 0, 1))")
    model._initialize_group([x, z], "variables")
    model._initialize_group([y], "parameters")
    model._initialize_group([eq], "equations")
    model._simplify_unpacked_sympy_representation()

    assert len(model.unpacked_equation_symbols) == 3

    for i, eq in zip(["A", "B", "C"], model.unpacked_equation_symbols):
        inputs = list(set(get_inputs(eq)))
        f = sp.lambdify(inputs, eq)
        kwargs = {arg.name: np.random.normal() for arg in inputs}
        np.testing.assert_allclose(
            f(**kwargs),
            kwargs[f"z_{i}"] - kwargs["y"] - kwargs[f"x_{i}_A"] - kwargs[f"x_{i}_B"],
        )


def test_unpack_equation_with_many_sums():
    coords = {"i": ["A", "B", "C"], "j": ["A", "B"]}
    model = CGEModel(coords=coords, compile=None, parse_equations_to_sympy=True)

    z = Variable(name="z")
    x = Variable(name="x", dims=("j",))
    b = Variable(name="b", dims=("i",))

    eq = Equation(
        name="sum for group <dim:i>",
        equation="z = Sum(x, (j, 0, 1)) + Sum(b, (i, 0, 2))",
    )
    model._initialize_group([x, z, b], "variables")
    model._initialize_group([eq], "equations")
    model._simplify_unpacked_sympy_representation()

    assert len(model.unpacked_equation_symbols) == 1

    [eq] = model.unpacked_equation_symbols
    inputs = get_inputs(eq)
    f = sp.lambdify(inputs, eq)
    kwargs = {arg.name: np.random.normal() for arg in inputs}
    x_sum = sum(kwargs[f"x_{j}"] for j in coords["j"])
    b_sum = sum(kwargs[f"b_{i}"] for i in coords["i"])
    np.testing.assert_allclose(f(**kwargs), kwargs["z"] - x_sum - b_sum)


def test_unpack_double_index():
    coords = {"i": ["A", "B", "C"], "j": ["E", "F"]}
    model = CGEModel(coords=coords, compile=None, parse_equations_to_sympy=True)
    X = Variable(name="X", dims=("i", "j"))
    VC = Variable(name="VC", dims=("j",))
    psi_X = Parameter("psi_X", dims=("i", "j"))

    eq = Equation("<dim:j> firm demand for <dim:i> inputs", "X = VC * psi_X")
    model._initialize_group([X, VC], "variables")
    model._initialize_group([psi_X], "parameters")
    model._initialize_group([eq], "equations")
    model._simplify_unpacked_sympy_representation()

    expected_output = [
        "X_A_E = VC_E * psi_X_A_E",
        "X_A_F = VC_F * psi_X_A_F",
        "X_B_E = VC_E * psi_X_B_E",
        "X_B_F = VC_F * psi_X_B_F",
        "X_C_E = VC_E * psi_X_C_E",
        "X_C_F = VC_F * psi_X_C_F",
    ]
    local_dict = {
        x.name: x for x in model.unpacked_variable_symbols + model.unpacked_parameter_symbols
    }
    expected_output_sp = [
        sp.parse_expr(expr, local_dict=local_dict, transformations="all")
        for expr in expected_output
    ]
    expected_output_sp = [x.lhs - x.rhs for x in expected_output_sp]
    assert len(model.unpacked_equation_symbols) == 6
    assert all([eq in expected_output_sp for eq in model.unpacked_equation_symbols])


def test_tax_unpack():
    coords = {"i": ["A", "B", "C"], "k": ["E", "F", "G", "H", "I", "J"]}
    model = CGEModel(coords=coords, compile=None, parse_equations_to_sympy=True)
    X = Variable(name="X", dims=("i", "k"))
    P_X = Variable(name="P_X", dims=("i", "k"))
    income = Variable(name="income")
    tau = Parameter("tau", dims=("i", "k"))

    eq = Equation("tax income", "income = Sum(Sum((tau * P_X * X), (i, 0, 2)), (k, 0, 5))")
    model._initialize_group([X, P_X, income], "variables")
    model._initialize_group([tau], "parameters")
    model._initialize_group([eq], "equations")
    model._simplify_unpacked_sympy_representation()

    assert len(model.unpacked_equation_symbols) == 1
    [eq] = model.unpacked_equation_symbols
    str_eq = str(eq)
    assert "[" not in str_eq


def test_long_unpack():
    coords = {
        "i": ["A", "B", "C"],
        "j": ["A", "B", "C"],
        "k": ["E", "F", "G", "H", "I", "J", "K", "L", "M"],
    }

    model = CGEModel(coords=coords, compile=None, parse_equations_to_sympy=True)

    P_Y = Variable("P_Y", dims=("i",))
    P_M = Variable("P_M", dims=("i",))
    P_E = Variable("P_E", dims=("k",))
    w = Variable("w")
    r = Variable("r")
    X_D = Variable("X_D", dims=("i", "j"))
    X_M = Variable("X_M", dims=("i", "j"))
    X_E_D = Variable("X_E_D", dims=("i", "k"))
    X_E_M = Variable("X_E_M", dims=("i", "k"))
    L_d = Variable("L_d", dims=("i",))
    K_d = Variable("K_d", dims=("i",))
    E_d = Variable("E_d", dims=("i",))
    Y = Variable("Y", dims=("i",))

    tau_X_D = Parameter("tau_X_D", dims=("i", "j"))
    tau_X_M = Parameter("tau_X_M", dims=("i", "j"))
    tau_w = Parameter("tau_w", dims=("i",))
    tau_r = Parameter("tau_r", dims=("i",))
    tau_E = Parameter("tau_E", dims=("i",))
    tau_Y = Parameter("tau_Y", dims=("i",))
    tau_X_E_D = Parameter("tau_X_E_D", dims=("i", "k"))
    tau_X_E_M = Parameter("tau_X_E_M", dims=("i", "k"))

    eq = Equation(
        "budget constraint",
        "G = "
        + "+".join(
            [
                "Sum(Sum((tau_X_D * P_Y * X_D), (i, 0, 2)), (j, 0, 2))",
                "Sum(Sum((tau_X_M * P_M * X_M), (i, 0, 2)), (j, 0, 2))",
                "Sum((tau_w * L_d * w), (i, 0, 2))",
                "Sum((tau_r * K_d * r), (i, 0, 2))",
                "Sum((tau_E * E_d * P_E), (i, 0, 2))",
                "Sum((tau_Y * P_Y * Y), (i, 0, 2))",
                "Sum(Sum((tau_X_E_D * P_Y * X_E_D), (i, 0, 2)), (k, 0, 8))",
                "Sum(Sum((tau_X_E_M * P_M * X_E_M), (i, 0, 2)), (k, 0, 8))",
            ]
        ),
    )

    model._initialize_group(
        [P_Y, P_M, P_E, w, r, X_D, X_M, X_E_D, X_E_M, L_d, K_d, E_d, Y], "variables"
    )
    model._initialize_group(
        [tau_X_D, tau_X_M, tau_w, tau_r, tau_E, tau_Y, tau_X_E_D, tau_X_E_M],
        "parameters",
    )
    model._initialize_group([eq], "equations")
    model._simplify_unpacked_sympy_representation()

    [eq] = model.unpacked_equation_symbols
    str_eq = str(eq)
    assert "[" not in str_eq


@pytest.mark.parametrize(
    "model_function, jac_function",
    [
        (load_model_1, expected_model_1_jacobian),
        (load_model_2, expected_model_2_jacobian),
    ],
    ids=["simple_model", "3-goods simple"],
)
@pytest.mark.parametrize(
    "backend, parse_to_sympy",
    [("numba", True), ("pytensor", True), ("pytensor", False)],
    ids=["numba", "pytensor-from-sympy", "pytensor"],
)
def test_model_gradients(model_function, jac_function, backend, parse_to_sympy):
    mode = "FAST_COMPILE" if backend == "pytensor" else None
    mod = model_function(
        backend=backend,
        parse_equations_to_sympy=parse_to_sympy,
        equation_mode=backend if not parse_to_sympy else "numba",
        mode=mode,
    )

    data = generate_data(mod.variables + mod.parameters, mod.coords)

    J_expected = jac_function(**data)

    if backend == "numba":
        x, theta = variable_dict_to_flat_array(data, mod.variables, mod.parameters)
        J_actual = mod.f_jac(x, theta)
    else:
        J_actual = mod.f_jac(**data)

    np.testing.assert_allclose(J_expected, J_actual, atol=1e-8)


@pytest.mark.parametrize(
    "model_function, calibrate_model, f_expected_jac, data",
    [
        (load_model_1, calibrate_model_1, expected_model_1_jacobian, model_1_data),
        (load_model_2, calibrate_model_2, expected_model_2_jacobian, model_2_data),
    ],
    ids=["simple_model", "3-goods simple"],
)
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_pytensor_from_sympy(model_function, calibrate_model, f_expected_jac, data, sparse):
    mod = model_function(
        equation_mode="numba",
        backend="pytensor",
        parse_equations_to_sympy=True,
        mode="FAST_COMPILE",
        use_sparse_matrices=sparse,
    )

    calibated_data = calibrate_model(**data)
    resid = mod.f_system(**calibated_data)
    np.testing.assert_allclose(resid, 0, atol=1e-8)

    jac = mod.f_jac(**calibated_data)
    if sparse:
        jac = jac.todense()
    expected_jac = f_expected_jac(**calibated_data)
    np.testing.assert_allclose(jac, expected_jac, atol=1e-8)


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
        (
            "_solve_with_minimize",
            {
                "method": "trust-exact",
                "use_hess": True,
                "use_hessp": False,
                "niter": 10_000,
                "tol": 1e-16,
            },
        ),
        (
            "_solve_with_root",
            {"method": "hybr", "use_jac": True, "niter": 10_000, "tol": 1e-16},
        ),
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
                assert res.success or res.fun < 1e-8, f"{name} solver failed to converge"
            if hasattr(res, "x"):
                assert not np.all(np.isnan(res.x)), f"{name} solver returned NaNs"
                assert np.all(np.isfinite(res.x)), f"{name} solver returned Infs"

                (
                    np.testing.assert_allclose(
                        np.around(results[0].x, 4),
                        np.around(results[1].x, 4),
                        atol=1e-5,
                        rtol=1e-5,
                    ),
                    "Solvers disagree",
                )

            else:
                assert not np.all(np.isnan(res)), f"{name} solver returned NaNs"
                assert np.all(np.isfinite(res)), f"{name} solver returned Infs"

                (
                    np.testing.assert_allclose(
                        np.around(results[0], 4),
                        np.around(results[1], 4),
                        atol=1e-5,
                        rtol=1e-5,
                    ),
                    "Solvers disagree",
                )

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


def test_generate_SAM():
    mod = load_model_1(backend="pytensor")
    param_dict = {
        "alpha": 0.75,
        "A": 2.0,
        "L_s": 1000,
        "K_s": 5000,
        "w": 1.0,
    }
    initial_guess = {"C": 10000, "Y": 10000, "income": 10000, "K_d": 5000, "L_d": 1000}

    fixed_values = {"r": 1.0, "P": 1.0, "resid": 0.0}
    mod.generate_SAM(
        param_dict=param_dict,
        initial_variable_guess=initial_guess,
        fixed_values=fixed_values,
        solve_method="minimize",
        method="nelder-mead",
        use_jac=False,
        use_hess=False,
    )
