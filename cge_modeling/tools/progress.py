import math
import sys
import threading

from collections.abc import Iterable

from better_optimize.constants import CONSOLE_WIDTH
from better_optimize.utilities import ToggleableProgress
from IPython.display import display
from rich.console import Console
from rich.jupyter import JupyterRenderable, _render_segments
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Column, Table
from rich.text import Text

NOTEBOOK_REFRESH_SECONDS = 0.1
BAR_WIDTH = 20

ROOT_STATISTICS = {"f_value": "Residual", "grad_norm": "||Jᵀf||"}
MINIMIZE_STATISTICS = {"f_value": "Objective", "grad_norm": "||grad||", "rayleigh": "gᵀHg/|g|²"}
STATISTIC_FORMATS = {"f_value": "{:0.8f}", "grad_norm": "{:0.8f}", "rayleigh": "{:+0.4e}"}

# The rows a simulation's stages add: the Euler approximation, and better_optimize's root and minimize solvers.
STAGE_NAMES = ("Euler", "Finding Roots", "Minimizing")


class FieldColumn(ProgressColumn):
    """Render one statistic from a task's fields, blank for tasks that do not report it."""

    def __init__(self, field: str, format_string: str, table_column: Column | None = None) -> None:
        super().__init__(table_column=table_column)
        self.field = field
        self.format_string = format_string

    def render(self, task: Task) -> Text:
        value = task.fields.get(self.field)
        return Text("" if value is None else self.format_string.format(value))


class StepColumn(MofNCompleteColumn):
    """A ``completed/total`` column padded to the widest total seen, so its width does not change between frames."""

    def __init__(self, table_column: Column | None = None) -> None:
        super().__init__(table_column=table_column)
        self.digits = 1

    def render(self, task: Task) -> Text:
        total = "?" if task.total is None else str(int(task.total))
        return Text(
            f"{int(task.completed):>{self.digits}d}/{total:>{self.digits}}",
            style="progress.download",
        )


def _jupyter_frame(progress: ToggleableProgress) -> JupyterRenderable:
    segments = list(progress.console.render(progress.get_renderable(), progress.console.options))
    return JupyterRenderable(
        _render_segments(segments), "".join(segment.text for segment in segments)
    )


class SimulationProgress(ToggleableProgress):
    """
    One progress table shared by every stage of a simulation, one row per stage.

    In a terminal, rich's live display draws the table. In a notebook, the table is drawn to one display handle that
    updates in place, and its final frame stays in the cell output as static content.
    """

    def __init__(self, *columns, **kwargs) -> None:
        super().__init__(*columns, **kwargs)
        self._render_thread: threading.Thread | None = None
        self._stop_rendering = threading.Event()
        self._display_handle = None
        self._in_notebook = self.is_enabled and self.console.is_jupyter
        if self._in_notebook:
            self.disable = True

    def make_tasks_table(self, tasks: Iterable[Task]) -> Table:
        tasks = list(tasks)
        widest = max(
            (len(str(int(task.total))) for task in tasks if task.total is not None), default=1
        )
        for column in self.columns:
            if isinstance(column, StepColumn):
                column.digits = max(column.digits, widest)
        return super().make_tasks_table(tasks)

    def __enter__(self):
        if self._in_notebook and self._entry_depth == 0:
            self._start_notebook_rendering()
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        if self._in_notebook and self._entry_depth == 0:
            self._stop_notebook_rendering()

    def _start_notebook_rendering(self) -> None:
        handle = display(_jupyter_frame(self), display_id=True)
        assert handle is not None  # display_id=True always returns a handle
        self._display_handle = handle
        self._stop_rendering.clear()

        def render_loop() -> None:
            while not self._stop_rendering.wait(NOTEBOOK_REFRESH_SECONDS):
                handle.update(_jupyter_frame(self))

        self._render_thread = threading.Thread(target=render_loop, daemon=True)
        self._render_thread.start()

    def _stop_notebook_rendering(self) -> None:
        self._stop_rendering.set()
        if self._render_thread is not None:
            self._render_thread.join()
            self._render_thread = None
        if self._display_handle is not None:
            self._display_handle.update(_jupyter_frame(self))


class SweepProgress(SimulationProgress):
    """
    A simulation progress table for a sweep: one row counting the sweep's points, and one row per solver stage that
    every point reuses.

    Stages add their rows with ``add_task`` exactly as in a single simulation. The first point creates each stage's
    row, and later points reset it, so the table keeps the same rows however many points the sweep has.
    """

    def __init__(self, *columns, description: str, total: int, **kwargs) -> None:
        super().__init__(*columns, **kwargs)
        self._description = description
        self._point_label = ""
        self._stage_rows: dict[str, TaskID] = {}
        self._worst_residual = 0.0
        self._n_failed = 0
        self._sweep_row = (
            Progress.add_task(self, description, total=total, show_remaining=True)
            if self.is_enabled
            else None
        )

    def add_task(
        self, description: str, *args, total: float | None = None, **fields
    ) -> TaskID | None:
        if not self.is_enabled:
            return None

        label = f"{self._point_label} {description}"
        if description in self._stage_rows:
            task_id = self._stage_rows[description]
            self.reset(task_id, total=total, description=label, **fields)
            return task_id

        task_id = Progress.add_task(self, label, *args, total=total, **fields)
        self._stage_rows[description] = task_id
        return task_id

    def start_point(self, label: str) -> None:
        """Label the stage rows with the point the sweep is solving next."""
        self._point_label = label

    def finish_point(self, residual: float, converged: bool) -> None:
        """Count a solved point, and show the largest final residual and the number of failures so far."""
        if self._sweep_row is None:
            return
        if math.isnan(residual) or residual > self._worst_residual:
            self._worst_residual = residual
        if not converged:
            self._n_failed += 1
        description = self._description
        if self._n_failed:
            description = f"{description} ({self._n_failed} failed)"
        self.update(
            self._sweep_row, advance=1, description=description, f_value=self._worst_residual
        )


class RemainingColumn(TimeRemainingColumn):
    """The estimated time remaining, shown only for tasks that set the ``show_remaining`` field."""

    def render(self, task: Task) -> Text:
        if not task.fields.get("show_remaining", False):
            return Text("")
        return super().render(task)


def _progress_columns(
    statistics: dict[str, str], description_width: int | None, show_remaining: bool
) -> list[ProgressColumn]:
    remaining = (
        [RemainingColumn(table_column=Column("Remaining", no_wrap=True))] if show_remaining else []
    )
    return [
        TextColumn(
            "{task.description}", table_column=Column("", no_wrap=True, width=description_width)
        ),
        BarColumn(bar_width=BAR_WIDTH, table_column=Column("Progress")),
        TimeElapsedColumn(table_column=Column("Elapsed", no_wrap=True)),
        *remaining,
        *(
            FieldColumn(field, STATISTIC_FORMATS[field], table_column=Column(heading, no_wrap=True))
            for field, heading in statistics.items()
        ),
        StepColumn(table_column=Column("Step", no_wrap=True)),
    ]


def simulation_progress(statistics: dict[str, str], enabled: bool = True) -> SimulationProgress:
    """
    Return a progress table with a label, a bar, the elapsed time, one column per statistic, and the step count.

    Parameters
    ----------
    statistics : dict of str to str
        Task field name to column heading, for the statistics each stage reports. A stage that does not report a
        statistic leaves its cell blank.
    enabled : bool, optional
        Whether to draw the table. Default is True.

    Returns
    -------
    progress : SimulationProgress
        The progress table. Each stage adds its own row with ``add_task``.
    """
    return SimulationProgress(
        *_progress_columns(statistics, description_width=None, show_remaining=False),
        expand=False,
        disable=not enabled,
        console=Console(file=sys.stderr, width=CONSOLE_WIDTH),
    )


def sweep_progress(
    statistics: dict[str, str],
    description: str,
    point_labels: list[str],
    enabled: bool = True,
) -> SweepProgress:
    """
    Return a progress table for a sweep, with the columns of :func:`simulation_progress` and the time remaining.

    Parameters
    ----------
    statistics : dict of str to str
        Task field name to column heading, as in :func:`simulation_progress`. The sweep's own row shows the largest
        final ``f_value`` among the points solved so far.
    description : str
        Label of the row counting the sweep's points.
    point_labels : list of str
        Label of every point the sweep will solve. The label column is sized once from these, so it does not change
        width as the sweep moves between points.
    enabled : bool, optional
        Whether to draw the table. Default is True.

    Returns
    -------
    progress : SweepProgress
        The progress table. Call ``start_point`` before solving each point and ``finish_point`` after.
    """
    longest_stage_label = max(map(len, point_labels), default=0) + 1 + max(map(len, STAGE_NAMES))
    failure_suffix = len(f" ({len(point_labels)} failed)")
    description_width = max(longest_stage_label, len(description) + failure_suffix)

    # A simulation table fits CONSOLE_WIDTH with labels no longer than the stage names. Widen the console by the
    # longer labels and the remaining-time column, so the bar keeps its full width.
    extra_width = description_width - max(map(len, STAGE_NAMES)) + len("Remaining") + 2
    return SweepProgress(
        *_progress_columns(statistics, description_width=description_width, show_remaining=True),
        description=description,
        total=len(point_labels),
        expand=False,
        disable=not enabled,
        console=Console(file=sys.stderr, width=CONSOLE_WIDTH + extra_width),
    )
