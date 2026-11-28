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
)
from rich.table import Column, Table
from rich.text import Text

NOTEBOOK_REFRESH_SECONDS = 0.1
BAR_WIDTH = 20

ROOT_STATISTICS = {"f_value": "Residual", "grad_norm": "||Jᵀf||"}
MINIMIZE_STATISTICS = {"f_value": "Objective", "grad_norm": "||grad||", "rayleigh": "gᵀHg/|g|²"}
STATISTIC_FORMATS = {"f_value": "{:0.8f}", "grad_norm": "{:0.8f}", "rayleigh": "{:+0.4e}"}


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

    A row is identified by its description. A stage that runs more than once, such as the solve at each checkpoint of
    a path, resets its row instead of adding another, so the table keeps one row per stage however many times
    each one runs.

    In a terminal, rich's live display draws the table. In a notebook, the table is drawn to one display handle that
    updates in place, and its final frame stays in the cell output as static content.
    """

    def __init__(self, *columns, **kwargs) -> None:
        super().__init__(*columns, **kwargs)
        self._stage_rows: dict[str, TaskID] = {}
        self._render_thread: threading.Thread | None = None
        self._stop_rendering = threading.Event()
        self._display_handle = None
        self._in_notebook = self.is_enabled and self.console.is_jupyter
        if self._in_notebook:
            self.disable = True

    def add_task(
        self, description: str, *args, total: float | None = None, **fields
    ) -> TaskID | None:
        if not self.is_enabled:
            return None

        if description in self._stage_rows:
            task_id = self._stage_rows[description]
            self.reset(task_id, total=total, **fields)
            return task_id

        task_id = Progress.add_task(self, description, *args, total=total, **fields)
        self._stage_rows[description] = task_id
        return task_id

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
    columns = [
        TextColumn("{task.description}", table_column=Column("", no_wrap=True)),
        BarColumn(bar_width=BAR_WIDTH, table_column=Column("Progress")),
        TimeElapsedColumn(table_column=Column("Elapsed", no_wrap=True)),
        *(
            FieldColumn(field, STATISTIC_FORMATS[field], table_column=Column(heading, no_wrap=True))
            for field, heading in statistics.items()
        ),
        StepColumn(table_column=Column("Step", no_wrap=True)),
    ]
    return SimulationProgress(
        *columns,
        expand=False,
        disable=not enabled,
        console=Console(file=sys.stderr, width=CONSOLE_WIDTH),
    )
