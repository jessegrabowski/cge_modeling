import importlib.util

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Literal

from pytensor.compile import get_mode

from cge_modeling.base.cge import CGEModel
from cge_modeling.base.primitives import Equation, Parameter, Variable
from cge_modeling.compile.constants import CompiledFunctions
from cge_modeling.compile.pytensor import compile_pytensor_cge_functions
from cge_modeling.compile.sympytensor import compile_sympytensor_cge_functions

COMPILE_FUNC_FACTORY: dict[str, Callable[..., tuple[Any, ...]]] = {
    "pytensor": compile_pytensor_cge_functions,
    "sympytensor": compile_sympytensor_cge_functions,
}


def determine_default_backend(
    equations: Sequence[Equation],
) -> Literal["pytensor", "sympytensor"]:
    # Check for Sum/Prod in the equations, if found, use sympytensor
    if any("Sum(" in eq.equation or "Prod(" in eq.equation for eq in equations):
        return "sympytensor"

    # Alternatively, if we find broadcasting logic in the equations, use pytensor
    if any("[:, None]" in eq.equation or "[None, :]" in eq.equation for eq in equations):
        return "pytensor"

    # Default to sympytensor if there is no info
    return "sympytensor"


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
    backend: Literal["pytensor", "sympytensor"] | None = None,
    mode: str | None = None,
    use_sparse_matrices: bool = False,
    functions_to_compile: CompiledFunctions | Sequence[CompiledFunctions] | None = "all",
    use_scan_euler: bool = False,
    use_rk4: bool = False,
) -> CGEModel:
    if (
        backend == "pytensor"
        and mode is not None
        and mode.upper() == "JAX"
        and importlib.util.find_spec("jax") is not None
    ):
        from cge_modeling.compile.jax import compile_jax_cge_functions

        func_maker = partial(compile_jax_cge_functions, use_rk4=use_rk4)
    elif backend is None:
        raise ValueError(f"backend must be one of {list(COMPILE_FUNC_FACTORY)}, found None")
    else:
        func_maker = partial(COMPILE_FUNC_FACTORY[backend], use_rk4=use_rk4)

    compile_list = list(_parse_compile_kwarg(functions_to_compile) or [])

    # TODO: 'inplace_elemwise_optimizer' was triggering an infinite loop during graph rewrites, but that's probably
    #  a bug in PyTensor. Remove this once the bug is fixed.
    mode = get_mode(mode).excluding("inplace_elemwise_optimizer")

    f_system, f_jac, f_resid, f_grad, f_hess, f_hessp, f_euler = func_maker(
        cge_model=model,
        functions_to_compile=compile_list,
        mode=mode,
        use_scan_euler=use_scan_euler,
        use_sparse_matrices=use_sparse_matrices,
    )
    model.f_system = f_system
    model.f_jac = f_jac
    model.f_resid = f_resid
    model.f_grad = f_grad
    model.f_hess = f_hess
    model.f_hessp = f_hessp
    model.f_euler = f_euler

    model._compiled = compile_list
    model._compile_backend = backend
    model.mode = mode.upper() if isinstance(mode, str) else mode

    return model


def cge_model(
    coords: dict[str, list[str]] | None = None,
    variables: list[Variable] | None = None,
    parameters: list[Parameter] | None = None,
    equations: list[Equation] | None = None,
    numeraire: str | None = None,
    apply_sympy_simplify: bool = False,
    backend: Literal["pytensor", "sympytensor"] | None = None,
    mode: str | None = None,
    use_sparse_matrices: bool = False,
    functions_to_compile: CompiledFunctions | Sequence[CompiledFunctions] | None = "all",
    use_scan_euler: bool = False,
    use_rk4_euler: bool = False,
) -> CGEModel:
    if backend is None:
        backend = determine_default_backend(equations or [])

    model = CGEModel(
        coords=coords,
        variables=variables,
        parameters=parameters,
        equations=equations,
        numeraire=numeraire,
        parse_equations_to_sympy=backend == "sympytensor",
    )

    return compile_model(
        model,
        backend=backend,
        mode=mode,
        use_sparse_matrices=use_sparse_matrices,
        functions_to_compile=functions_to_compile,
        use_scan_euler=use_scan_euler,
        use_rk4=use_rk4_euler,
    )
