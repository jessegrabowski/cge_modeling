import logging

from collections.abc import Sequence
from typing import Literal

from pytensor.compile import get_mode

from cge_modeling.base.cge import CGEModel
from cge_modeling.compile.constants import CompiledFunctions
from cge_modeling.compile.pytensor import compile_pytensor_cge_functions

_log = logging.getLogger(__name__)


def _parse_compile_kwarg(
    compile: CompiledFunctions | Sequence[CompiledFunctions] | Literal[True] | None = None,
) -> Sequence[CompiledFunctions] | None:
    if compile is None:
        return None
    if compile is True or compile == "all":
        return ["root", "minimize", "euler"]
    if isinstance(compile, str):
        return [compile]
    return compile


def compile_model(
    model: CGEModel,
    mode: str | None = None,
    functions_to_compile: CompiledFunctions | Sequence[CompiledFunctions] | None = "all",
    use_scan_euler: bool = False,
    use_rk4: bool = False,
) -> CGEModel:
    model.check_initialization()
    _log.info(
        f"Compiling a model with {model.n_equations} equations, {model.n_variables} variables, "
        f"{model.n_parameters} parameters"
    )

    compile_list = list(_parse_compile_kwarg(functions_to_compile) or [])

    # TODO: 'inplace_elemwise_optimizer' was triggering an infinite loop during graph rewrites, but that's probably
    #  a bug in PyTensor. Remove this once the bug is fixed.
    mode = get_mode(mode).excluding("inplace_elemwise_optimizer")

    f_system, f_jac, f_resid, f_grad, f_hess, f_hessp, f_euler = compile_pytensor_cge_functions(
        cge_model=model,
        functions_to_compile=compile_list,
        mode=mode,
        use_scan_euler=use_scan_euler,
        use_rk4=use_rk4,
    )
    model.f_system = f_system
    model.f_jac = f_jac
    model.f_resid = f_resid
    model.f_grad = f_grad
    model.f_hess = f_hess
    model.f_hessp = f_hessp
    model.f_euler = f_euler

    model._compiled = compile_list
    model.mode = mode

    return model
