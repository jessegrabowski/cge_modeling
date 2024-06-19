from cge_modeling.base.cge import CGEModel
from cge_modeling.base.primitives import Equation, Parameter, Variable
from cge_modeling.plotting import plot_kateplot, plot_lines
from cge_modeling.tools.output_tools import display_info_as_table, latex_print_equations

import logging

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)


__all__ = [
    "CGEModel",
    "Variable",
    "Parameter",
    "Equation",
    "latex_print_equations",
    "display_info_as_table",
    "plot_lines",
    "plot_kateplot",
]
