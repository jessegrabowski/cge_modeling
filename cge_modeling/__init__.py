import logging

from cge_modeling.base.primitives import Equation, Parameter, Variable
from cge_modeling.base.cge import CGEModel
from cge_modeling.base.build import cge_model, compile_model
from cge_modeling.compile.pytensor_tools import prod_to_no_zero_prod
from cge_modeling.plotting import plot_bar, plot_areas, plot_lines
from cge_modeling.tools.output_tools import display_info_as_table, latex_print_equations

from importlib.metadata import PackageNotFoundError, version

_log = logging.getLogger("cge_modeling")
if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)

try:
    __version__ = version("cge_modeling")
except PackageNotFoundError:
    try:
        # Fallback for when the package is not installed, e.g., during development
        from cge_modeling._version import __version__
    except ImportError:
        _log.warning(
            "Could not determine package version, you're probably doing development work without having "
            "installed the package. Call `hatch build --hooks-only` to generate the version file, or "
            "install the package with `pip install .`."
        )
        __version__ = "0.0.0"

__all__ = [
    "cge_model",
    "compile_model",
    "CGEModel",
    "Variable",
    "Parameter",
    "Equation",
    "latex_print_equations",
    "display_info_as_table",
    "plot_lines",
    "plot_areas",
    "plot_bar",
]
