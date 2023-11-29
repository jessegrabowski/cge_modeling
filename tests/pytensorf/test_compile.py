import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from cge_modeling.base.primitives import Parameter, Variable
from cge_modeling.pytensorf.compile import (
    compile_cge_model_to_pytensor,
    compile_euler_approximation_function,
    make_printer_cache,
    object_to_pytensor,
)
from tests.utilities.models import load_model_1, load_model_2

test_cases = [
    (Parameter, "K_d", {}),
    (Variable, "Y", {"t": [0, 1, 2]}),
    (Variable, "Y", {"t": [0, 1, 2], "s": [0, 1, 2]}),
]


@pytest.mark.parametrize(
    "cls, name, coords", test_cases, ids=["Parameter", "Variable", "Variable2d"]
)
def test_object_to_pytensor(cls, name, coords):
    # noinspection PyArgumentList
    obj = cls(name=name, dims=list(coords.keys()), description="A lovely item from group <dim:i>")

    pt_obj = object_to_pytensor(obj, coords)
    assert pt_obj.name == name
    assert pt_obj.ndim == len(coords)
    assert pt_obj.type.shape == tuple(len(coords[dim]) for dim in coords.keys())


def test_make_cache():
    cge_model = load_model_1()
    variables = [object_to_pytensor(var, cge_model.coords) for var in cge_model.variables]
    parameters = [object_to_pytensor(param, cge_model.coords) for param in cge_model.parameters]

    cache = make_printer_cache(variables, parameters)

    assert len(cache) == len(variables) + len(parameters)
    assert all(
        [key[0] in cge_model.variable_names + cge_model.parameter_names for key in cache.keys()]
    )
    assert all([var in variables + parameters for var in cache.values()])


@pytest.mark.parametrize("model_func", [load_model_1, load_model_2], ids=["1_Sector", "3_Sector"])
def test_compile_to_pytensor(model_func):
    cge_model = model_func(parse_equations_to_sympy=False)
    n_eq = len(cge_model.unpacked_variable_names)
    (f_model, f_jac, f_jac_inv) = compile_cge_model_to_pytensor(cge_model)

    assert f_model.fn.outputs[0].variable.type.shape == (n_eq,)
    assert f_jac.fn.outputs[0].variable.type.shape == (n_eq, n_eq)
    assert f_jac_inv.fn.outputs[0].variable.type.shape == (n_eq, n_eq)


def test_compile_euler_approximation_function():
    # Example problem from Notes and Problems from Applied General Equilibrium Economics, Chapter 3

    variables = v1, v2 = [pt.dscalar(name) for name in ["v1", "v2"]]
    parameters = v3 = pt.dscalar("v3")
    inputs = variables + parameters

    equations = [v1**2 * v3 - 1, v1 + v2 - 2]

    def f_analytic(v3):
        v1 = 1 / np.sqrt(v3)
        v2 = 2 - v1
        return np.array([v1, v2])

    mode = "FAST_COMPILE"
    f_1 = compile_euler_approximation_function(
        equations, variables, [parameters], n_steps=1, mode=mode
    )
    f_10 = compile_euler_approximation_function(
        equations, variables, [parameters], n_steps=10, mode=mode
    )
    f_100 = compile_euler_approximation_function(
        equations, variables, [parameters], n_steps=100, mode=mode
    )
    f_10k = compile_euler_approximation_function(
        equations, variables, [parameters], n_steps=10_000, mode=mode
    )

    initial_point = [1, 1]
    v3_initial = 1
    v3_final = 2

    analytic_solution = f_analytic(v3_final)
    approximate_solutions = [
        f(*initial_point, v3_initial, np.array([v3_final]))[0] for f in [f_1, f_10, f_100, f_10k]
    ]
    errors = np.c_[[solution[-1] - analytic_solution for solution in approximate_solutions]]

    # Test the errors are monotonically decreasing in the number of steps
    assert np.all(np.diff(np.abs(errors), axis=0) < 0)

    # Test that the solution is close to the analytic solution at 10,000 steps
    assert np.allclose(approximate_solutions[-1][-1], analytic_solution, atol=1e-5)
