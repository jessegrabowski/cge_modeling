# Tests for cge_modeling.base.function_wrappers
import numpy as np
import pytensor
import pytensor.tensor as pt

from cge_modeling import Parameter, Variable
from cge_modeling.base.function_wrappers import (
    select_free_variable_entries,
    wrap_pytensor_func_for_scipy,
)


def test_wrap_pytensor_func_for_scipy():
    variables = [Variable(name, dims=None) for name in ["x", "y", "z"]]
    parameters = [Parameter(name, dims=None) for name in ["a", "b", "c"]]
    coords = {}

    x, y, z = (pt.dscalar(var.name) for var in variables)
    a, b, c = (pt.dscalar(var.name) for var in parameters)

    f = a * x + b * y + c * z
    mse = (f**2).sum()

    f_mse = pytensor.function([x, y, z, a, b, c], mse, mode="FAST_RUN")
    test_value_dict = {"a": 1.0, "b": 2.0, "c": 3.0, "x": 1.0, "y": 2.0, "z": 3.0}
    assert f_mse(**test_value_dict) == 196.0

    f_mse_wrapped = wrap_pytensor_func_for_scipy(f_mse, variables, parameters, coords)
    assert f_mse_wrapped(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 196.0


def test_wrap_pytensor_func_for_scipy_passes_fixed_values():
    variables = [Variable(name, dims=None) for name in ["x", "y", "z"]]
    parameters = [Parameter(name, dims=None) for name in ["a", "b", "c"]]

    x, y, z = (pt.dscalar(var.name) for var in variables)
    a, b, c = (pt.dscalar(var.name) for var in parameters)
    f_mse = pytensor.function([x, y, z, a, b, c], ((a * x + b * y + c * z) ** 2).sum())

    free_variables = [variables[0], variables[2]]
    f_mse_fixed_y = wrap_pytensor_func_for_scipy(
        f_mse, free_variables, parameters, coords={}, fixed_values={"y": 2.0}
    )

    # x=1, y=2 (fixed), z=3 with a=1, b=2, c=3 gives (1 + 4 + 9)^2. Swapping x and z would give 100.
    assert f_mse_fixed_y(np.array([1.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 196.0


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
