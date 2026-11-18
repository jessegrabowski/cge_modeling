import logging

import pytest

from cge_modeling import CGEModel
from cge_modeling import model as cm
from cge_modeling.base import build
from cge_modeling.base.primitives import Equation
from tests.utilities.models import load_and_cache_model


@pytest.mark.parametrize(
    "eq_strings, expected_backend",
    [
        (["Y = Sum(X, (i, 0, 1))"], "sympytensor"),
        (["Y = Prod(X, (i, 0, 1))"], "sympytensor"),
        (["Y = X[:, None]"], "pytensor"),
        (["Y = X[None, :]"], "pytensor"),
        (["Y = X + Z"], "sympytensor"),  # Default case
    ],
    ids=["sum", "prod", "broadcast_col", "broadcast_row", "default"],
)
def test_determine_default_backend(eq_strings, expected_backend):
    equations = [Equation(name=f"eq{i}", equation=s) for i, s in enumerate(eq_strings)]
    assert build.determine_default_backend(equations) == expected_backend


@pytest.mark.parametrize(
    "compile_arg, expected_output",
    [
        (None, None),
        ("all", ["root", "minimize", "euler"]),
        (True, ["root", "minimize", "euler"]),
        ("root", ["root"]),
        (["root", "minimize"], ["root", "minimize"]),
    ],
    ids=["none", "all_str", "true_bool", "single_str", "list_str"],
)
def test_parse_compile_kwarg(compile_arg, expected_output):
    assert build._parse_compile_kwarg(compile_arg) == expected_output


@pytest.mark.parametrize("backend", ["pytensor", "sympytensor"])
@pytest.mark.parametrize(
    "functions_to_compile, expected_compiled_list",
    [
        ("all", ["root", "minimize", "euler"]),
        (("root",), ["root"]),
        (("minimize",), ["minimize"]),
        (("euler",), ["euler"]),
        (None, []),
    ],
    ids=["all", "root_only", "minimize_only", "euler_only", "none"],
)
def test_cge_model_factory(backend, functions_to_compile, expected_compiled_list):
    model = load_and_cache_model(
        model_id=1, backend=backend, mode="FAST_RUN", functions_to_compile=functions_to_compile
    )

    assert model._compile_backend == backend
    if expected_compiled_list:
        assert model._compiled
    else:
        assert not model._compiled

    # f_system is always compiled
    assert model.f_system is not None

    compiled_root = "root" in expected_compiled_list
    compiled_minimize = "minimize" in expected_compiled_list
    compiled_euler = "euler" in expected_compiled_list

    assert (model.f_jac is not None) == compiled_root
    for func in [model.f_resid, model.f_grad, model.f_hess, model.f_hessp]:
        assert (func is not None) == compiled_minimize
    assert (model.f_euler is not None) == compiled_euler

    assert model.parse_equations_to_sympy is (backend == "sympytensor")


def check_function_signature(func, expected_input_names):
    input_names = [x.name for x in func.maker.fgraph.inputs]
    assert input_names == expected_input_names


@pytest.mark.parametrize("backend", ["pytensor", "sympytensor"])
def test_compiled_functions_take_packed_inputs(backend):
    model = load_and_cache_model(
        model_id=1, backend=backend, mode="FAST_RUN", functions_to_compile="all"
    )

    for func in [model.f_system, model.f_jac, model.f_resid, model.f_grad, model.f_hess]:
        check_function_signature(func, ["x", "theta"])
    check_function_signature(model.f_hessp, ["x", "p", "theta"])
    check_function_signature(
        model.f_euler.keywords["f_step"], ["x", "theta", "theta_initial", "theta_final", "n_steps"]
    )


def test_compile_model_rejects_a_non_square_context_built_model():
    with CGEModel(coords={"i": ["a", "b"]}) as model:
        P = cm.Variable("P", dims="i")
        cm.Equation("first price", P.sel(i="a"), 1.0)

    with pytest.raises(ValueError, match="Found 2 variables but 1 equations"):
        build.compile_model(model, backend="pytensor")


def test_compile_model_rejects_the_sympytensor_backend_for_a_context_built_model():
    with CGEModel(coords={"i": ["a", "b"]}) as model:
        P = cm.Variable("P", dims="i")
        cm.Equation("prices", P, 1.0)

    with pytest.raises(ValueError, match="compile with backend='pytensor', found 'sympytensor'"):
        build.compile_model(model, backend="sympytensor")


def test_compile_model_logs_the_model_size_once(caplog):
    model = load_and_cache_model(model_id=1, backend="pytensor", functions_to_compile=None)

    with caplog.at_level(logging.INFO, logger="cge_modeling"):
        build.compile_model(model, backend="pytensor", functions_to_compile=None)

    size_messages = [r.message for r in caplog.records if "Compiling a model with" in r.message]
    assert len(size_messages) == 1


def test_compile_model_requires_backend():
    model = load_and_cache_model(model_id=1, backend="pytensor", functions_to_compile=None)

    with pytest.raises(ValueError, match="backend must be one of"):
        build.compile_model(model, backend=None)
