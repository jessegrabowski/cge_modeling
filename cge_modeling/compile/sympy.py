from typing import cast

import sympy as sp


def make_sympy_jacobian(equations: list[sp.Expr] | sp.Matrix, wrt: list[sp.Symbol]) -> sp.Matrix:
    if not isinstance(equations, sp.Matrix):
        equations = sp.Matrix(equations)

    jac = equations.jacobian(wrt)
    return jac


def make_sympy_gradient(loss: sp.Expr, wrt: list[sp.Symbol]) -> sp.Matrix:
    grad = sp.Matrix([loss.diff(var) for var in wrt])
    return grad


def make_sympy_hessp(
    grad: sp.Matrix, variables: list[sp.Symbol]
) -> tuple[sp.Matrix, sp.IndexedBase]:
    n = len(variables)
    p = sp.IndexedBase("hess_point", shape=n)

    hessp_loss = cast(sp.Expr, sum([grad[i] * p[i] for i in range(n)]))
    hessp = make_sympy_gradient(hessp_loss, variables)

    return hessp, p
