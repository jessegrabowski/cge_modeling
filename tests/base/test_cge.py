import logging

from collections.abc import Callable
from importlib.util import find_spec

import numpy as np
import pytest
import sympy as sp
import xarray as xr

from cge_modeling.base.cge import CGEModel
from cge_modeling.base.primitives import Equation, Parameter, Variable
from cge_modeling.base.utilities import flat_array_to_variable_dict, variable_dict_to_flat_array
from cge_modeling.tools.sympy_tools import get_inputs
from tests.utilities.fake_data import generate_data
from tests.utilities.models import (
    calibrate_model_1,
    calibrate_model_2,
    expected_model_1_jacobian,
    expected_model_2_jacobian,
    load_and_cache_model,
    load_model_1,
    load_model_1_components,
    model_1_data,
    model_2_data,
)

JAX_INSTALLED = find_spec("jax") is not None


@pytest.fixture()
def model():
    return CGEModel(coords={"i": ["A", "B", "C"]})


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
    model = CGEModel(coords=coords, parse_equations_to_sympy=True)

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
    model = CGEModel(coords=coords, parse_equations_to_sympy=True)
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
    model = CGEModel(coords=coords, parse_equations_to_sympy=True)
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

    model = CGEModel(coords=coords, parse_equations_to_sympy=True)

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
    "model_id, jac_function",
    [
        (
            1,
            expected_model_1_jacobian,
        ),  # expected_model_1_grad, expected_model_1_hessian),
        (
            2,
            expected_model_2_jacobian,
        ),  # expected_model_2_grad, expected_model_2_hessian),
    ],
    ids=["simple_model", "3-goods simple"],
)
@pytest.mark.parametrize("backend", ["sympytensor", "pytensor"], ids=["sympytensor", "pytensor"])
def test_model_gradients(model_id, jac_function, backend):
    mod = load_and_cache_model(
        model_id, mode="FAST_RUN", backend=backend, functions_to_compile=("root", "minimize")
    )

    data = generate_data(mod.variables + mod.parameters, mod.coords)
    x, theta = variable_dict_to_flat_array(data, mod.variables, mod.parameters)
    p = np.eye(mod.n_equations)[:, 0]

    J_expected = jac_function(**data)

    J_actual = mod.f_jac(x, theta)
    H_actual = mod.f_hess(x, theta)
    Hp_actual = mod.f_hessp(x, p, theta)

    np.testing.assert_allclose(J_expected, J_actual, atol=1e-8)
    np.testing.assert_allclose(H_actual[:, 0], Hp_actual.ravel())


@pytest.mark.parametrize(
    "model_id, calibrate_model, f_expected_jac, data",
    [
        (1, calibrate_model_1, expected_model_1_jacobian, model_1_data),
        (2, calibrate_model_2, expected_model_2_jacobian, model_2_data),
    ],
    ids=["simple_model", "3-goods simple"],
)
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_pytensor_from_sympy(model_id, calibrate_model, f_expected_jac, data, sparse):
    mod = load_and_cache_model(
        model_id,
        backend="sympytensor",
        mode="FAST_RUN" if not JAX_INSTALLED or sparse else "JAX",
        use_sparse_matrices=sparse,
        functions_to_compile="root",
    )

    calibated_data = calibrate_model(**data)
    x, theta = variable_dict_to_flat_array(calibated_data, mod.variables, mod.parameters)
    resid = mod.f_system(x, theta)
    np.testing.assert_allclose(resid, 0, atol=1e-8)

    jac = mod.f_jac(x, theta)
    if sparse:
        jac = jac.todense()
    expected_jac = f_expected_jac(**calibated_data)
    np.testing.assert_allclose(jac, expected_jac, atol=1e-8)


@pytest.mark.parametrize(
    "model_id, calibrate_model, data",
    [
        (1, calibrate_model_1, model_1_data),
        # (2, calibrate_model_2, model_2_data),
    ],
    ids=["simple_model"],  # , "3-goods simple"],
)
@pytest.mark.parametrize(
    "method, solver_kwargs, functions_to_compile",
    [
        (
            "_solve_with_minimize",
            {
                "method": "trust-exact",
                "use_hess": True,
                "use_hessp": False,
                "maxiter": 10_000,
                "tol": 1e-16,
            },
            "minimize",
        ),
        (
            "_solve_with_minimize",
            {
                "method": "trust-ncg",
                "use_hess": False,
                "use_hessp": True,
                "maxiter": 10_000,
                "tol": 1e-16,
            },
            "minimize",
        ),
        (
            "_solve_with_root",
            {"method": "hybr", "use_jac": True, "maxiter": 10_000, "tol": 1e-16},
            "root",
        ),
        ("_solve_with_euler_approximation", {"n_steps": 500}, "euler"),
    ],
    ids=["minimize_hess", "minimize_hessp", "root", "euler"],
)
def test_jax_mode_agrees_with_default_mode(
    model_id: int,
    calibrate_model: Callable,
    data: dict,
    method: str,
    solver_kwargs: dict,
    functions_to_compile: str,
):
    pytest.importorskip("jax")
    model_default = load_and_cache_model(
        model_id=model_id, backend="pytensor", functions_to_compile=functions_to_compile
    )
    model_jax = load_and_cache_model(
        model_id=model_id, backend="pytensor", mode="JAX", functions_to_compile=functions_to_compile
    )

    def solver_agreement_checks(results: list, names: list):
        solutions = []
        for res, name in zip(results, names):
            if hasattr(res, "success"):
                assert res.success or np.allclose(res.fun, 0.0, rtol=1e-8, atol=1e-8), (
                    f"{name} solver failed to converge"
                )

            solution = res.x if hasattr(res, "x") else res
            assert not np.all(np.isnan(solution)), f"{name} solver returned NaNs"
            assert np.all(np.isfinite(solution)), f"{name} solver returned Infs"
            solutions.append(solution)

        np.testing.assert_allclose(
            np.around(solutions[0], 4),
            np.around(solutions[1], 4),
            atol=1e-5,
            rtol=1e-5,
            err_msg="Solvers disagree",
        )

    calibrated_data = calibrate_model(**data)
    x, theta = variable_dict_to_flat_array(
        calibrated_data, model_default.variables, model_default.parameters
    )
    np.testing.assert_allclose(model_default.f_system(x, theta), 0, atol=1e-8)
    np.testing.assert_allclose(model_jax.f_system(x, theta), 0, atol=1e-8)

    labor_increase = calibrated_data.copy()
    labor_increase["L_s"] = 10_000

    _, theta_labor_increase = variable_dict_to_flat_array(
        labor_increase, model_default.variables, model_default.parameters
    )

    res_default = getattr(model_default, method)(
        calibrated_data, theta_labor_increase, **solver_kwargs
    )
    res_jax = getattr(model_jax, method)(calibrated_data, theta_labor_increase, **solver_kwargs)

    if method == "_solve_with_euler_approximation":
        res_default = res_default["parameters"].dataset.isel(step=-1).to_array().values
        res_jax = res_jax["parameters"].dataset.isel(step=-1).to_array().values

    solver_agreement_checks([res_default, res_jax], ["Default mode", "JAX mode"])


@pytest.mark.parametrize(
    "gtol, expected_success",
    [(1e10, False), (1e-10, True)],
    ids=["stops_before_first_step", "reaches_equilibrium"],
)
def test_minimize_success_reflects_equilibrium_residuals(gtol, expected_success):
    model = load_and_cache_model(model_id=2, backend="pytensor", functions_to_compile="minimize")
    calibrated_data = calibrate_model_2(**model_2_data)
    shocked_data = {**calibrated_data, "L_s": calibrated_data["L_s"] * 1.5}
    _, theta_final = variable_dict_to_flat_array(shocked_data, model.variables, model.parameters)

    result = model._solve_with_minimize(
        calibrated_data,
        theta_final,
        method="trust-ncg",
        progressbar=False,
        verbose=False,
        gtol=gtol,
    )

    assert bool(result.success) is expected_success
    assert "equilibrium_tol" in result.message


@pytest.mark.parametrize("backend", ["pytensor", "sympytensor"])
def test_scan_euler_matches_python_loop_euler(backend):
    scan_model = load_model_1(backend=backend, functions_to_compile=("euler",), use_scan_euler=True)
    loop_model = load_model_1(backend=backend, functions_to_compile=("euler",))
    calibrated_data = calibrate_model_1(**model_1_data)
    _, theta_final = variable_dict_to_flat_array(
        {**calibrated_data, "L_s": 10_000}, scan_model.variables, scan_model.parameters
    )

    scan_result = scan_model._solve_with_euler_approximation(
        calibrated_data, theta_final, n_steps=500
    )
    loop_result = loop_model._solve_with_euler_approximation(
        calibrated_data, theta_final, n_steps=500
    )

    np.testing.assert_allclose(
        scan_result["variables"].dataset.isel(step=-1).to_array().values,
        loop_result["variables"].dataset.isel(step=-1).to_array().values,
        rtol=1e-10,
        atol=1e-8,
    )


def test_simulate_with_scan_euler_reports_no_nans(caplog):
    mod = load_model_1(
        backend="pytensor", functions_to_compile=("euler", "root"), use_scan_euler=True
    )
    initial_state = calibrate_model_1(**model_1_data)

    with caplog.at_level(logging.WARNING, logger="cge_modeling"):
        res = mod.simulate(initial_state, final_delta_pct={"L_s": 0.5}, n_iter_euler=100)

    assert res["optim_res"].success
    assert "NaNs were encountered" not in caplog.text


def test_generate_SAM():
    mod = load_and_cache_model(model_id=1, backend="pytensor", functions_to_compile="minimize")
    param_dict = {
        "alpha": 0.75,
        "A": 2.0,
        "L_s": 1000,
        "K_s": 5000,
        "w": 1.0,
    }
    initial_guess = {
        "C": 10000,
        "Y": 10000,
        "income": 10000,
        "K_d": 5000,
        "L_d": 1000,
        "r": 1.0,
        "P": 1.0,
        "resid": 0.0,
    }
    # fixed_values = {"r": 1.0, "P": 1.0, "resid": 0.0}

    mod.generate_SAM(
        param_dict=param_dict,
        initial_variable_guess=initial_guess,
        # fixed_values=fixed_values,
        solve_method="minimize",
        method="nelder-mead",
        use_jac=False,
        use_hess=False,
    )


def test_generate_SAM_with_fixed_value_matches_free_solution():
    mod = load_and_cache_model(model_id=1, backend="pytensor", functions_to_compile="minimize")
    param_dict = {"alpha": 0.75, "A": 2.0, "L_s": 1000, "K_s": 5000, "w": 1.0}
    initial_guess = {
        "C": 10000,
        "Y": 10000,
        "income": 10000,
        "K_d": 5000,
        "L_d": 1000,
        "r": 1.0,
        "P": 1.0,
        "resid": 0.0,
    }
    solver_kwargs = {
        "solve_method": "minimize",
        "method": "trust-ncg",
        "progressbar": False,
        "verbose": False,
    }

    free_solution = mod.generate_SAM(
        param_dict=param_dict, initial_variable_guess=initial_guess, **solver_kwargs
    )

    # Y is the first variable, so mapping the solution onto the free variables by position would
    # shift every other variable by one.
    fixed_solution = mod.generate_SAM(
        param_dict=param_dict,
        initial_variable_guess=initial_guess,
        fixed_values={"Y": free_solution["Y"]},
        **solver_kwargs,
    )

    for name in mod.variable_names:
        np.testing.assert_allclose(fixed_solution[name], free_solution[name], rtol=1e-6, atol=1e-4)


def test_generate_SAM_rejects_fixed_values_outside_minimize():
    mod = load_and_cache_model(model_id=1, backend="pytensor", functions_to_compile="minimize")

    with pytest.raises(ValueError, match="only supported with solve_method='minimize'"):
        mod.generate_SAM(
            param_dict={}, initial_variable_guess={}, fixed_values={"Y": 1.0}, solve_method="root"
        )


def test_generate_SAM_rejects_unknown_fixed_value_names():
    mod = load_and_cache_model(model_id=1, backend="pytensor", functions_to_compile="minimize")

    with pytest.raises(ValueError, match=r"not model variables: \{'alpha'\}"):
        mod.generate_SAM(
            param_dict={},
            initial_variable_guess={},
            fixed_values={"alpha": 0.5},
            solve_method="minimize",
        )


def test_invalid_simulation_raises():
    mod = load_and_cache_model(
        model_id=1, backend="sympytensor", functions_to_compile=("root", "euler")
    )
    initial_state = calibrate_model_1(**model_1_data)

    with pytest.raises(
        ValueError, match="initial_state must be a Result or a dict of initial values"
    ):
        mod.simulate([1, 2, 3, 4, 5], final_values={"L_s": 1500})

    with pytest.raises(
        ValueError,
        match=r"initial_state must contain values for all variables and parameters in the "
        r"model\. Did not find values for alpha",
    ):
        bad_init = initial_state.copy()
        del bad_init["alpha"]

        mod.simulate(bad_init, final_values={"L_s": 1500})

    with pytest.raises(
        ValueError,
        match="initial_state contains values for variables or parameters not in the model: lol",
    ):
        bad_init = initial_state.copy()
        bad_init["lol"] = 3
        mod.simulate(bad_init, final_values={"L_s": 1500})

    with pytest.raises(ValueError, match="No parameters to update!"):
        mod.simulate(initial_state)

    with pytest.raises(ValueError, match="Arguments K_s are repeated among"):
        mod.simulate(
            initial_state, final_values={"L_s": 1500, "K_s": 5000}, final_delta_pct={"K_s": 0.5}
        )


@pytest.mark.parametrize(
    "scenario_kwargs, expected_result",
    [
        ({"final_values": {"L_s": 1500}}, {"L_s": lambda x: 1500}),
        ({"final_delta_pct": {"L_s": 0.5}}, {"L_s": lambda x: x * 0.5}),
        ({"final_delta": {"L_s": -500}}, {"L_s": lambda x: x - 500}),
        (
            {"final_values": {"L_s": 1500}, "final_delta_pct": {"K_s": 0.5}},
            {"L_s": lambda x: 1500, "K_s": lambda x: x * 0.5},
        ),
    ],
    ids=["final_values", "final_delta_pct", "final_delta", "combined"],
)
def test_simulate(scenario_kwargs, expected_result):
    mod = load_and_cache_model(model_id=1, backend="sympytensor")
    initial_state = calibrate_model_1(**model_1_data)

    res = mod.simulate(initial_state, **scenario_kwargs)

    for key in expected_result:
        final_param = res["optimizer"]["parameters"][key].isel(step=-1)
        assert final_param == expected_result[key](initial_state[key])


def test_simulate_compiles_an_uncompiled_model():
    variables, parameters, equations, coords = load_model_1_components()
    mod = CGEModel(coords=coords, variables=variables, parameters=parameters, equations=equations)
    initial_state = calibrate_model_1(**model_1_data)

    res = mod.simulate(initial_state, final_values={"L_s": 1500}, n_iter_euler=100)

    assert mod._compile_backend == "sympytensor"
    assert list(mod._compiled) == ["euler", "root"]
    assert res["optimizer"]["parameters"]["L_s"].isel(step=-1) == 1500


def test_calibrate_reproduces_sequential_calibration():
    variables, parameters, equations, coords = load_model_1_components()
    mod = CGEModel(coords=coords, variables=variables, parameters=parameters, equations=equations)
    expected = calibrate_model_1(**model_1_data)

    result = mod.calibrate(data={**model_1_data, "w": 1.0})

    assert result.success
    fitted = result.to_dict()["fitted"]
    for name, value in expected.items():
        np.testing.assert_allclose(fitted[name], value)


def test_calibrate_flags_inconsistent_data():
    variables, parameters, equations, coords = load_model_1_components()
    mod = CGEModel(coords=coords, variables=variables, parameters=parameters, equations=equations)

    # The budget constraint already pins consumption to income, so supplying C as well
    # over-determines the system.
    result = mod.calibrate(data={**model_1_data, "w": 1.0, "C": 5.0})

    assert not result.success


def test_check_for_equilibrium_reads_simulation_datatree(caplog):
    mod = load_and_cache_model(model_id=1, backend="sympytensor")
    initial_state = calibrate_model_1(**model_1_data)
    res = mod.simulate(initial_state, final_delta_pct={"L_s": 0.5})

    with caplog.at_level(logging.INFO, logger="cge_modeling"):
        mod.check_for_equilibrium(res["optimizer"])

    assert "Equilibrium found!" in caplog.text


def test_check_for_equilibrium_flags_mismatched_state(caplog):
    mod = load_and_cache_model(model_id=1, backend="sympytensor")
    initial_state = calibrate_model_1(**model_1_data)
    optimizer = mod.simulate(initial_state, final_delta_pct={"L_s": 0.5})["optimizer"]

    pre_shock_variables_post_shock_parameters = xr.DataTree.from_dict(
        {
            "variables": optimizer["variables"].dataset.isel(step=0),
            "parameters": optimizer["parameters"].dataset.isel(step=-1),
        }
    )

    with caplog.at_level(logging.INFO, logger="cge_modeling"):
        mod.check_for_equilibrium(pre_shock_variables_post_shock_parameters)

    assert "Equilibrium not found" in caplog.text


def test_check_for_equilibrium_requires_variables_and_parameters():
    mod = load_and_cache_model(model_id=1, backend="sympytensor")
    variables_only = xr.DataTree.from_dict({"variables": xr.Dataset()})

    with pytest.raises(ValueError, match="DataTree must contain variables and parameters children"):
        mod.check_for_equilibrium(variables_only)
