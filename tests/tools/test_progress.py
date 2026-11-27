import io
import re

import pytest

from rich.console import Console

from cge_modeling.tools.progress import ROOT_STATISTICS, simulation_progress, sweep_progress


def make_sweep_progress(statistics, enabled=True):
    return sweep_progress(
        statistics, description="Sweep itr", point_labels=["itr=0.1", "itr=0.2"], enabled=enabled
    )


def render_table(progress) -> str:
    console = Console(file=io.StringIO(), width=160, record=True)
    console.print(progress.make_tasks_table(progress.tasks))
    return console.export_text()


def test_simulation_progress_puts_every_stage_in_one_table_with_step_last():
    progress = simulation_progress(ROOT_STATISTICS)
    progress.add_task("Euler", total=100, f_value=0.5, grad_norm=0.25)
    progress.add_task("Finding Roots", total=5000, f_value=0.001)

    header, *rows = [line for line in render_table(progress).splitlines() if line.strip()]

    assert header.split()[-1] == "Step"
    assert "Euler" in rows[1] and rows[1].rstrip().endswith("   0/ 100")
    assert "Finding Roots" in rows[2] and rows[2].rstrip().endswith("   0/5000")


def test_simulation_progress_leaves_a_statistic_blank_for_a_stage_that_does_not_report_it():
    progress = simulation_progress(ROOT_STATISTICS)
    progress.add_task("Finding Roots", total=10, f_value=0.125)

    header, _, row = [line for line in render_table(progress).splitlines() if line.strip()]

    assert "||Jᵀf||" in header
    assert re.findall(r"\d+\.\d{8}", row) == ["0.12500000"]


@pytest.mark.parametrize("make_progress", [simulation_progress, make_sweep_progress])
def test_disabled_progress_adds_no_tasks(make_progress):
    progress = make_progress(ROOT_STATISTICS, enabled=False)

    with progress:
        task = progress.add_task("Euler", total=10, f_value=1.0)

    assert task is None


def test_sweep_progress_reuses_one_row_per_stage_and_labels_it_with_the_current_point():
    progress = make_sweep_progress(ROOT_STATISTICS)

    for label in ["itr=0.1", "itr=0.2"]:
        progress.start_point(label)
        euler = progress.add_task("Euler", total=100, f_value=0.5)
        progress.add_task("Finding Roots", total=5000, f_value=0.001)
        progress.update(euler, completed=100)
        progress.finish_point(residual=1e-12, converged=True)

    _, _, sweep_row, euler_row, roots_row = [
        line for line in render_table(progress).splitlines() if line.strip()
    ]

    assert len(progress.tasks) == 3
    assert sweep_row.split()[:2] == ["Sweep", "itr"] and sweep_row.rstrip().endswith("2/   2")
    assert euler_row.split()[:2] == ["itr=0.2", "Euler"] and euler_row.rstrip().endswith("100/ 100")
    assert roots_row.split()[:3] == ["itr=0.2", "Finding", "Roots"]
    assert roots_row.rstrip().endswith("0/5000")


def test_sweep_progress_reports_the_largest_residual_and_counts_failures():
    progress = make_sweep_progress(ROOT_STATISTICS)

    for residual, converged in [(0.25, True), (0.5, False), (0.125, False)]:
        progress.finish_point(residual=residual, converged=converged)

    sweep_row = [line for line in render_table(progress).splitlines() if line.strip()][2]

    assert "Sweep itr (2 failed)" in sweep_row
    assert re.findall(r"\d+\.\d{8}", sweep_row) == ["0.50000000"]


def test_sweep_progress_keeps_a_nan_residual_once_one_appears():
    progress = make_sweep_progress(ROOT_STATISTICS)

    for residual in [0.25, float("nan"), 0.5]:
        progress.finish_point(residual=residual, converged=False)

    sweep_row = [line for line in render_table(progress).splitlines() if line.strip()][2]

    assert "nan" in sweep_row
