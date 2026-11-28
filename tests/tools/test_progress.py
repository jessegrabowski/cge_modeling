import io
import re

from rich.console import Console

from cge_modeling.tools.progress import ROOT_STATISTICS, simulation_progress


def render_table(progress) -> str:
    console = Console(file=io.StringIO(), width=160, record=True)
    console.print(progress.make_tasks_table(progress.tasks))
    return console.export_text()


def test_simulation_progress_puts_every_stage_in_one_table_with_step_last():
    progress = simulation_progress(ROOT_STATISTICS)
    progress.add_task("Path (rk4)", total=100, f_value=0.5, grad_norm=0.25)
    progress.add_task("Finding Roots", total=5000, f_value=0.001)

    header, *rows = [line for line in render_table(progress).splitlines() if line.strip()]

    assert header.split()[-1] == "Step"
    assert "Path (rk4)" in rows[1] and rows[1].rstrip().endswith("   0/ 100")
    assert "Finding Roots" in rows[2] and rows[2].rstrip().endswith("   0/5000")


def test_simulation_progress_leaves_a_statistic_blank_for_a_stage_that_does_not_report_it():
    progress = simulation_progress(ROOT_STATISTICS)
    progress.add_task("Finding Roots", total=10, f_value=0.125)

    header, _, row = [line for line in render_table(progress).splitlines() if line.strip()]

    assert "||Jᵀf||" in header
    assert re.findall(r"\d+\.\d{8}", row) == ["0.12500000"]


def test_simulation_progress_resets_the_row_of_a_stage_that_runs_again():
    progress = simulation_progress(ROOT_STATISTICS)
    first = progress.add_task("Finding Roots", total=10, f_value=0.5)
    progress.update(first, completed=10)
    second = progress.add_task("Finding Roots", total=4, f_value=0.25)

    _, _, row = [line for line in render_table(progress).splitlines() if line.strip()]

    assert second == first
    assert len(progress.tasks) == 1
    assert row.rstrip().endswith(" 0/4")
    assert re.findall(r"\d+\.\d{8}", row) == ["0.25000000"]


def test_disabled_simulation_progress_adds_no_tasks():
    progress = simulation_progress(ROOT_STATISTICS, enabled=False)

    with progress:
        task = progress.add_task("Path (rk4)", total=10, f_value=1.0)

    assert task is None
