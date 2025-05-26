# Tests for cge_modeling.base.function_wrappers
import numpy as np

from cge_modeling import Parameter, Variable
from cge_modeling.base.function_wrappers import wrap_pytensor_func_for_scipy


def test_wrap_pytensor_func_for_scipy():
    import pytensor
    import pytensor.tensor as pt

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
