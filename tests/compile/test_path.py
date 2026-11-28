import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from cge_modeling.base.utilities import variable_dict_to_flat_array
from cge_modeling.compile.path import follow_path_with_python_loop, path_tangent, symbolic_path
from cge_modeling.tools.progress import ROOT_STATISTICS, simulation_progress
from tests.utilities.models import calibrate_model_1, load_model_1, model_1_data

STEP_ORDERS = {"euler": 1, "rk4": 4}


def two_equation_problem():
    # Example problem from Notes and Problems from Applied General Equilibrium Economics, Chapter 3
    variables = v1, v2 = [pt.dscalar(name) for name in ["v1", "v2"]]
    v3 = pt.dscalar("v3")
    system = pt.stack([v1**2 * v3 - 1, v1 + v2 - 2])
    exact_at_v3_equal_2 = np.array([1 / np.sqrt(2), 2 - 1 / np.sqrt(2)])
    return system, variables, [v3], exact_at_v3_equal_2


def scan_path_endpoint(step_method, n_steps, vectorize_jacobian=True):
    system, variables, parameters, _ = two_equation_problem()
    inputs, trajectory = symbolic_path(
        system,
        variables,
        parameters,
        step_method=step_method,
        vectorize_jacobian=vectorize_jacobian,
    )
    f = pytensor.function(inputs, trajectory)
    v1, v2, _ = f(np.ones(2), np.ones(1), np.array([2.0]), n_steps)
    return np.array([v1[-1], v2[-1]])


def loop_path_endpoint(step_method, n_steps):
    system, variables, parameters, _ = two_equation_problem()
    *inputs, x_step = path_tangent(system, variables, parameters, vectorize_jacobian=True)
    v1, v2, _ = follow_path_with_python_loop(
        np.ones(2),
        np.ones(1),
        np.array([2.0]),
        n_steps,
        f_tangent=pytensor.function(inputs, x_step),
        step_method=step_method,
        shapes=[(), (), ()],
        progress=simulation_progress(ROOT_STATISTICS, enabled=False),
        step_statistics=lambda x, theta: {},
    )
    return np.array([v1[-1], v2[-1]])


@pytest.mark.parametrize(
    "path_endpoint", [scan_path_endpoint, loop_path_endpoint], ids=["scan", "loop"]
)
@pytest.mark.parametrize("step_method", ["euler", "rk4"])
def test_path_error_falls_at_the_order_of_the_step_method(path_endpoint, step_method):
    *_, exact = two_equation_problem()
    errors = [np.abs(path_endpoint(step_method, n_steps) - exact).max() for n_steps in (8, 16)]

    observed_order = np.log2(errors[0] / errors[1])

    np.testing.assert_allclose(observed_order, STEP_ORDERS[step_method], atol=0.15)


def test_scan_and_loop_rk4_paths_agree():
    np.testing.assert_allclose(
        scan_path_endpoint("rk4", 10), loop_path_endpoint("rk4", 10), rtol=1e-12
    )


def test_scan_path_with_a_scanned_jacobian_matches_the_vectorized_one():
    np.testing.assert_allclose(
        scan_path_endpoint("rk4", 10, vectorize_jacobian=False),
        scan_path_endpoint("rk4", 10, vectorize_jacobian=True),
        rtol=1e-12,
    )


def test_rk4_path_reaches_the_analytic_solution_in_few_steps():
    *_, exact = two_equation_problem()
    np.testing.assert_allclose(loop_path_endpoint("rk4", 20), exact, atol=1e-8)


def test_python_loop_path_skips_step_statistics_when_the_progress_table_is_off():
    model = load_model_1(functions_to_compile=["path"])
    x, theta = variable_dict_to_flat_array(
        calibrate_model_1(**model_1_data), model.variables, model.parameters
    )
    calls = []

    model.f_path(
        x,
        theta,
        theta * 1.1,
        5,
        progress=simulation_progress(ROOT_STATISTICS, enabled=False),
        step_statistics=lambda x, theta: calls.append(x) or {},
    )

    assert calls == []
