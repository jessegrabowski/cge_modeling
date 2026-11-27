import logging

from cge_modeling.base.cge import CGEModel
from cge_modeling.model import Equation, Parameter, Variable
from cge_modeling.compile.pytensor_tools import prod_to_no_zero_prod
from cge_modeling.plotting import plot_bar, plot_areas, plot_lines
from cge_modeling.tools.output_tools import display_info_as_table, index_by_parameter
from cge_modeling.tools.sam_estimation import (
    KL,
    LeastSquares,
    Pearson,
    ReverseKL,
    SymbolicSAM,
    cells_forced_to_zero,
    linear_constraints,
)

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
    "KL",
    "CGEModel",
    "Equation",
    "LeastSquares",
    "Parameter",
    "Pearson",
    "ReverseKL",
    "SymbolicSAM",
    "Variable",
    "cells_forced_to_zero",
    "display_info_as_table",
    "index_by_parameter",
    "linear_constraints",
    "plot_areas",
    "plot_bar",
    "plot_lines",
]
