import importlib.util

from typing import Literal, cast

from cge_modeling.base.cge import CGEModel
from cge_modeling.base.primitives import Equation, Parameter, Variable
from cge_modeling.compile.constants import CompiledFunctions
from cge_modeling.compile.numba import compile_numba_cge_functions
from cge_modeling.compile.pytensor import compile_pytensor_cge_functions
from cge_modeling.compile.sympytensor import compile_sympytensor_cge_functions

COMPILE_FUNC_FACTORY = {
    "pytensor": compile_pytensor_cge_functions,
    "numba": compile_numba_cge_functions,
    "sympytensor": compile_sympytensor_cge_functions,
}


def determine_default_backend(
    equations: list[Equation],
) -> Literal["pytensor", "numba", "sympytensor"]:
    # Check for Sum/Prod in the equations, if found, use sympytensor
    if any("Sum(" in eq.equation or "Prod(" in eq.equation for eq in equations):
        return "sympytensor"

    # Alternatively, if we find broadcasting logic in the equations, use pytensor
    if any("[:, None]" in eq.equation or "[None, :]" in eq.equation for eq in equations):
        return "pytensor"

    # Default to sympytensor if there is no info
    return "sympytensor"


def _parse_compile_kwarg(compile: list[str] | str | None = None) -> list[CompiledFunctions]:
    if compile is not None:
        if compile in ["all", True]:
            compile = ["root", "minimize", "euler"]
        else:
            compile = [compile] if not isinstance(compile, list) else compile

        compile = cast(list[CompiledFunctions], compile)

    return compile


def compile_model(
    model: CGEModel,
    backend: Literal["pytensor", "numba", "sympytensor"] | None = None,
    mode: str | None = None,
    use_sparse_matrices: bool = False,
    functions_to_compile: list[CompiledFunctions] | None = "all",
    use_scan_euler: bool = False,
) -> CGEModel:
    if (
        backend == "pytensor"
        and mode is not None
        and mode.upper() == "JAX"
        and importlib.util.find_spec("jax") is not None
    ):
        from cge_modeling.compile.jax import compile_jax_cge_functions

        func_maker = compile_jax_cge_functions
    else:
        func_maker = COMPILE_FUNC_FACTORY[backend]

    functions_to_compile = _parse_compile_kwarg(functions_to_compile)
    f_system, f_jac, f_resid, f_grad, f_hess, f_hessp, f_euler = func_maker(
        cge_model=model,
        functions_to_compile=functions_to_compile,
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

    model._compiled = functions_to_compile
    model._compile_backend = backend
    model.mode = mode.upper() if isinstance(mode, str) else mode

    return model


def cge_model(
    coords: dict[str, list[str, ...]] | None = None,
    variables: list[Variable] | dict[str, Variable] | None = None,
    parameters: list[Parameter] | dict[str, Parameter] | None = None,
    equations: list[Equation] | dict[str, Equation] | None = None,
    numeraire: Variable | None = None,
    apply_sympy_simplify: bool = False,
    backend: Literal["pytensor", "numba", "sympytensor"] | None = None,
    mode: str | None = None,
    use_sparse_matrices: bool = False,
    functions_to_compile: list[CompiledFunctions] | None = "all",
    use_scan_euler: bool = False,
) -> CGEModel:
    if backend is None:
        backend = determine_default_backend(equations)

    model = CGEModel(
        coords=coords,
        variables=variables,
        parameters=parameters,
        equations=equations,
        numeraire=numeraire,
        parse_equations_to_sympy=backend in ["numba", "sympytensor"],
    )

    return compile_model(
        model,
        backend=backend,
        mode=mode,
        use_sparse_matrices=use_sparse_matrices,
        functions_to_compile=functions_to_compile,
        use_scan_euler=use_scan_euler,
    )
