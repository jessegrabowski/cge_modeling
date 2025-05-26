import importlib.util

from inspect import signature

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


def check_function_signature(func, backend, expected_input_names):
    if backend == "numba":
        assert func.__code__.co_argcount == len(expected_input_names)
        func_sig = signature(func)
        assert all(x in func_sig.parameters for x in expected_input_names)
    else:
        input_names = [x.name for x in func.maker.fgraph.inputs]
        assert len(input_names) == len(expected_input_names)
        assert all(x in input_names for x in expected_input_names)


@pytest.mark.parametrize("backend", ["pytensor", "sympytensor"])
def test_compiled_functions_have_correct_signature(backend):
    """
    Smoke test to ensure that all compiled functions have the expected signature. The code has
    gone though several iterations on this point, vacillating between f(variables, parameters) (form expected by scipy)
    and f(**variables, **parameters) (a more natural form for users).

    The canonical form is now f(**variables, **parameters), and all compiled functions should have this signature in all
    backends.
    """
    # TODO: Add numba test
    # TODO: Add sympytensor test

    # Skip compiling Euler because it has a much more complex signature
    model = load_and_cache_model(
        model_id=1, backend=backend, mode="FAST_RUN", functions_to_compile=("root", "minimize")
    )

    variable_names = [x.name for x in model.variables]
    parameter_names = [x.name for x in model.parameters]

    for func in [
        model.f_system,
        model.f_jac,
        model.f_resid,
        model.f_grad,
        model.f_hess,
    ]:
        check_function_signature(func, backend, variable_names + parameter_names)

    variable_points = [f"{x}_point" for x in variable_names]
    check_function_signature(
        model.f_hessp, backend, variable_names + parameter_names + variable_points
    )
