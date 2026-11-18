# Tests for cge_modeling.base.function_wrappers
import numpy as np
import pytest

from cge_modeling.base.function_wrappers import (
    select_free_variable_entries,
    wrap_compiled_function_for_scipy,
)


def test_wrap_compiled_function_for_scipy_inserts_fixed_values():
    def f_mse(x, theta):
        return (theta @ x) ** 2

    free_mask = np.array([True, False, True])
    x_full = np.array([np.nan, 2.0, np.nan])
    f_mse_fixed_y = wrap_compiled_function_for_scipy(f_mse, free_mask=free_mask, x_full=x_full)

    # x=1, y=2 (fixed), z=3 with a=1, b=2, c=3 gives (1 + 4 + 9)^2. Swapping x and z would give 100.
    assert f_mse_fixed_y(np.array([1.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 196.0


def test_wrap_compiled_function_for_scipy_keeps_fractional_steps_with_integer_fixed_values():
    def return_x(x, theta):
        return x

    wrapped = wrap_compiled_function_for_scipy(
        return_x, free_mask=np.array([True, False, True]), x_full=np.array([0, 2, 0])
    )

    np.testing.assert_array_equal(wrapped(np.array([1.5, 3.7]), np.ones(3)), [1.5, 2.0, 3.7])


def test_wrap_compiled_function_for_scipy_zeroes_the_point_of_fixed_variables():
    def return_point(x, p, theta):
        return p

    free_mask = np.array([True, False, True])
    x_full = np.array([np.nan, 2.0, np.nan])
    wrapped = wrap_compiled_function_for_scipy(
        return_point, include_p=True, free_mask=free_mask, x_full=x_full
    )

    np.testing.assert_array_equal(
        wrapped(np.ones(2), np.array([5.0, 7.0]), np.ones(3)), [5.0, 0.0, 7.0]
    )


@pytest.mark.parametrize("include_p", [False, True], ids=["value", "hessp"])
@pytest.mark.parametrize(
    "free_mask", [None, np.array([True, False])], ids=["all_free", "one_fixed"]
)
def test_wrap_compiled_function_for_scipy_returns_numpy_for_jax_outputs(include_p, free_mask):
    jnp = pytest.importorskip("jax.numpy")

    def f_jax(*args):
        x, theta = args[0], args[-1]
        return jnp.asarray(x * theta)

    wrapped = wrap_compiled_function_for_scipy(
        f_jax, include_p=include_p, free_mask=free_mask, x_full=np.array([0.0, 4.0])
    )
    x = np.array([2.0]) if free_mask is not None else np.array([2.0, 4.0])
    args = (x, np.zeros_like(x), np.array([3.0, 3.0])) if include_p else (x, np.array([3.0, 3.0]))
    out = wrapped(*args)

    assert type(out) is np.ndarray
    np.testing.assert_allclose(out, [6.0, 12.0])


def test_select_free_variable_entries_drops_fixed_rows_and_columns():
    def hessian(*args):
        return np.arange(9.0).reshape(3, 3)

    free_mask = np.array([True, False, True])
    selected = select_free_variable_entries(hessian, free_mask, variable_axes=(0, 1))

    np.testing.assert_array_equal(selected(), [[0.0, 2.0], [6.0, 8.0]])


def test_select_free_variable_entries_returns_function_when_nothing_is_fixed():
    def hessian(*args):
        return np.arange(9.0).reshape(3, 3)

    no_fixed_mask = np.ones(3, dtype=bool)

    assert select_free_variable_entries(hessian, no_fixed_mask, variable_axes=(0, 1)) is hessian
