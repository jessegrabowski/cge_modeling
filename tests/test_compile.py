import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from cge_modeling.base.primitives import Parameter, Variable
from cge_modeling.compile.euler import symbolic_euler_approximation
from cge_modeling.compile.pytensor import (
    cge_primitives_to_pytensor,
)
from cge_modeling.compile.pytensor_tools import make_jacobian, object_to_pytensor
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
    obj = cls(
        name=name,
        dims=list(coords.keys()),
        description="A lovely item from group <dim:i>",
    )

    pt_obj = object_to_pytensor(obj, coords)
    assert pt_obj.name == name
    assert pt_obj.ndim == len(coords)
    assert pt_obj.type.shape == tuple(len(coords[dim]) for dim in coords.keys())


@pytest.mark.parametrize(
    "model_func, kwargs",
    [
        (
            load_model_1,
            {"backend": "sympytensor"},
        ),
        (
            load_model_2,
            {
                "backend": "pytensor",
            },
        ),
    ],
    ids=["1_Sector", "3_Sector"],
)
def test_compile_to_pytensor(model_func, kwargs):
    cge_model = model_func(**kwargs)
    n_variables = n_eq = len(cge_model.unpacked_variable_names)
    n_params = len(cge_model.unpacked_parameter_names)

    system, variables, parameters, _ = cge_primitives_to_pytensor(cge_model)
    jac = make_jacobian(system, variables)
    B = make_jacobian(system, parameters)

    assert system.type.shape == (n_eq,)
    assert jac.type.shape == (n_eq, n_variables)
    assert B.type.shape == (n_eq, n_params)


def test_compile_euler_approximation_function():
    # Example problem from Notes and Problems from Applied General Equilibrium Economics, Chapter 3

    variables = v1, v2 = [pt.dscalar(name) for name in ["v1", "v2"]]
    parameters = v3 = pt.dscalar("v3")
    equations = pt.stack([v1**2 * v3 - 1, v1 + v2 - 2])

    def f_analytic(v3):
        v1 = 1 / np.sqrt(v3)
        v2 = 2 - v1
        return np.array([v1, v2])

    mode = "FAST_COMPILE"

    theta_final, n_steps, result = symbolic_euler_approximation(
        equations, variables=variables, parameters=[parameters]
    )
    f = pytensor.function([*variables, parameters, theta_final, n_steps], result, mode=mode)

    steps = [1, 10, 100, 10_000]
    initial_point = [1, 1]
    v3_initial = 1
    v3_final = 2

    analytic_solution = f_analytic(v3_final)
    approximate_solutions = [np.c_[f(*initial_point, v3_initial, v3_final, n)[:2]].T for n in steps]
    errors = np.c_[[solution[-1] - analytic_solution for solution in approximate_solutions]]

    # Test the errors are monotonically decreasing in the number of steps
    assert np.all(np.diff(np.abs(errors), axis=0) < 0)

    # Test that the solution is close to the analytic solution at 10,000 steps
    assert np.allclose(approximate_solutions[-1][-1, :], analytic_solution, atol=1e-5)
