import logging

from cge_modeling.base.primitives import Equation, Parameter, Variable
from cge_modeling.base.cge import CGEModel
from cge_modeling.base.build import cge_model, compile_model
from cge_modeling.compile.pytensor_tools import prod_to_no_zero_prod
from cge_modeling.plotting import plot_bar, plot_kateplot, plot_lines
from cge_modeling.tools.output_tools import display_info_as_table, latex_print_equations

from cge_modeling._version import get_versions


_log = logging.getLogger("cge_modeling")
if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)


__version__ = get_versions()["version"]

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
    "plot_kateplot",
    "plot_bar",
]
