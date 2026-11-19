import numpy as np
import pytest

from cge_modeling import CGEModel
from cge_modeling import model as cm
from cge_modeling.tools.calibration import (
    calibrate_entries,
    compile_calibration_system,
    solve_order,
)


def test_solve_order_puts_joint_blocks_after_their_dependencies():
    incidence = np.array(
        [
            [True, False, False, False],
            [True, True, True, False],
            [False, True, True, False],
            [False, True, False, True],
        ]
    )

    blocks, unmatched_equations, unmatched_unknowns = solve_order(incidence)

    assert [sorted(block.unknowns.tolist()) for block in blocks] == [[0], [1, 2], [3]]
    assert [sorted(block.equations.tolist()) for block in blocks] == [[0], [1, 2], [3]]
    assert unmatched_equations.size == unmatched_unknowns.size == 0


def test_solve_order_reports_equations_and_unknowns_left_unmatched():
    incidence = np.array([[True, False], [True, False], [False, False]])

    _, unmatched_equations, unmatched_unknowns = solve_order(incidence)

    assert unmatched_equations.size == 2
    assert unmatched_unknowns.tolist() == [1]


def test_calibrate_solves_a_block_whose_equations_must_be_solved_jointly():
    with CGEModel() as model:
        x, y = cm.Variable("x"), cm.Variable("y")
        a, b = cm.Parameter("a"), cm.Parameter("b")
        cm.Equation("Sum", x + 2 * y, a)
        cm.Equation("Product", x * y, b)
    system = compile_calibration_system(model)

    entries, residual = calibrate_entries(
        system, data={"a": 7.0, "b": 6.0}, initial_values={}, max_iter=100
    )
    x_value, y_value = entries[:2]

    blocks, _, _ = solve_order(system.pattern[:, :2])
    assert [block.unknowns.size for block in blocks] == [2]
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)
    np.testing.assert_allclose([x_value + 2 * y_value, x_value * y_value], [7.0, 6.0])


def test_calibrate_names_the_entries_no_equation_determines():
    with CGEModel() as model:
        x, y = cm.Variable("x"), cm.Variable("y")
        a = cm.Parameter("a")
        cm.Equation("Scale", y, a * x)
    system = compile_calibration_system(model)

    with pytest.raises(ValueError, match=r"No equation determines \['(x|y)'\]"):
        calibrate_entries(system, data={"a": 2.0}, initial_values={}, max_iter=100)


def test_calibrate_names_the_block_it_cannot_solve():
    with CGEModel() as model:
        x = cm.Variable("x")
        a = cm.Parameter("a")
        cm.Equation("Square", x**2, a)
    system = compile_calibration_system(model)

    with pytest.raises(ValueError, match=r"Could not solve \['Square'\] for \['x'\]"):
        calibrate_entries(system, data={"a": -1.0}, initial_values={}, max_iter=100)


def test_calibrate_rejects_names_that_are_not_model_objects_and_overlapping_inputs():
    with CGEModel() as model:
        x = cm.Variable("x")
        a = cm.Parameter("a")
        cm.Equation("Scale", x, 2 * a)
    system = compile_calibration_system(model)

    with pytest.raises(ValueError, match=r"\['b'\] are not variables or parameters"):
        calibrate_entries(system, data={"a": 1.0, "b": 1.0}, initial_values={}, max_iter=100)
    with pytest.raises(ValueError, match=r"\['a'\] are in both data and initial_values"):
        calibrate_entries(system, data={"a": 1.0}, initial_values={"a": 1.0}, max_iter=100)
