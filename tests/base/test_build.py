import importlib.util

import pytest

from cge_modeling.base import build
from cge_modeling.base.primitives import Equation
from tests.utilities.models import load_and_cache_model

JAX_INSTALLED = importlib.util.find_spec("jax") is not None


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


@pytest.mark.parametrize("backend", ["numba", "pytensor", "sympytensor"])
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

    is_numba = backend == "numba"

    should_compile_root = "root" in expected_compiled_list if expected_compiled_list else False
    should_compile_minimize = (
        "minimize" in expected_compiled_list if expected_compiled_list else False
    )
    should_compile_euler = "euler" in expected_compiled_list if expected_compiled_list else False

    # f_system is always compiled
    assert model.f_system is not None

    if should_compile_root:
        assert model.f_jac is not None
    else:
        assert model.f_jac is None

    if should_compile_minimize:
        assert model.f_resid is not None
        assert model.f_grad is not None
        assert model.f_hess is not None
        assert model.f_hessp is not None
    else:
        assert model.f_resid is None
        assert model.f_grad is None
        assert model.f_hess is None
        assert model.f_hessp is None

    if should_compile_euler:
        assert model.f_euler is not None
    else:
        assert model.f_euler is None

    if backend in ["numba", "sympytensor", None]:
        assert model.parse_equations_to_sympy is True
    else:  # backend == "pytensor"
        assert model.parse_equations_to_sympy is False
