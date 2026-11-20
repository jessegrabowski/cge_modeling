import logging

import pytest

from cge_modeling import CGEModel
from cge_modeling import model as cm
from cge_modeling.base import build
from tests.utilities.models import load_and_cache_model


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
def test_cge_model_factory(functions_to_compile, expected_compiled_list):
    model = load_and_cache_model(
        model_id=1, mode="FAST_RUN", functions_to_compile=functions_to_compile
    )

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


def check_function_signature(func, expected_input_names):
    input_names = [x.name for x in func.maker.fgraph.inputs]
    assert input_names == expected_input_names


def test_compiled_functions_take_packed_inputs():
    model = load_and_cache_model(model_id=1, mode="FAST_RUN", functions_to_compile="all")

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
        build.compile_model(model)


def test_compile_model_logs_the_model_size_once(caplog):
    model = load_and_cache_model(model_id=1, functions_to_compile=None)

    with caplog.at_level(logging.INFO, logger="cge_modeling"):
        build.compile_model(model, functions_to_compile=None)

    size_messages = [r.message for r in caplog.records if "Compiling a model with" in r.message]
    assert len(size_messages) == 1
