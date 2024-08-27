import logging

from functools import partial, wraps
from typing import cast

import numba as nb
import numpy as np
import sympy as sp

from cge_modeling import CGEModel
from cge_modeling.base.function_wrappers import wrap_numba_euler_function
from cge_modeling.compile.constants import CompiledFunctions
from cge_modeling.compile.numba_tools import numba_lambdify
from cge_modeling.compile.sympy import make_sympy_gradient, make_sympy_hessp, make_sympy_jacobian

_log = logging.getLogger(__name__)


def compile_numba_jacobian_function(
    equations: list[sp.Expr] | sp.Matrix, variables: list[sp.Symbol], parameters: list[sp.Symbol]
):
    jac = make_sympy_jacobian(equations, variables)
    f_jac = numba_lambdify(variables, jac, parameters)
    return jac, f_jac


def compile_numba_error_function(
    equations: list[sp.Expr], variables: list[sp.Symbol], parameters: list[sp.Symbol]
):
    squared_loss = cast(sp.Expr, 0.5 * sum(eq**2 for eq in equations))
    f_error = numba_lambdify(variables, squared_loss, parameters)
    return squared_loss, f_error


def compile_numba_gradient_function(
    loss: sp.Expr, variables: list[sp.Symbol], parameters: list[sp.Symbol]
):
    grad = make_sympy_gradient(loss, variables)
    f_grad = numba_lambdify(variables, grad, parameters, ravel_outputs=True)
    return grad, f_grad


def compile_numba_hessp_function(
    grad: sp.Matrix, variables: list[sp.Symbol], parameters: list[sp.Symbol]
):
    hessp, p = make_sympy_hessp(grad, variables)
    _f_hessp = numba_lambdify([*variables, p], hessp, parameters)

    @wraps(_f_hessp)
    def f_hessp(x, p, parameters):
        return np.r_[_f_hessp([*x.tolist(), p], parameters)].ravel()

    return hessp, f_hessp


@nb.njit(cache=True)
def float_to_array(arr):
    return np.asarray(arr, np.float64)


@nb.njit(cache=True, nogil=True)
def euler_approx(f_step, *, x0, theta0, theta_final, n_steps, progress_bar):
    """
    Compute the solution to a non-linear function g(x, theta + dtheta) by iteratively computing a linear approximation
    f(x[t], theta + epsilon[t]) at the point (f(x[t-1], theta + epsilon[t-1]), theta + epsilon[t-1]), where epsilon[-1] = dtheta

    Parameters
    ----------
    f_step: njit function
        Linearized function to be approximated. Must have signature f(endog, exog) -> array[:]

    x0: np.ndarray
        Array of values of model variables representing the point at which g is linearized.

    theta0: np.ndarray
        Array of model parameter values representing the point at which g is linearized.

    theta_final: np.ndarray
        Values at which g is to be solved. These should correspond to something like "shocks" from the initial parameter
        values theta0.

    n_steps: int
        Number of gradient updates to perform; this is the length of the epsilon vector in the notation above. More steps
        leads to a more precise approximation.

    Returns
    -------
    x: np.ndarray
        Approximate solution to g(x + dx)

    Notes
    -----
    A non-linear function g(x, theta) = 0, can be linearized around a point (x0, theta0) as:

        A(x0, theta0) @ dx + B(x0, theta0) @ dtheta = 0

    Where A is the jacobian of dg/dx, and B is the jacobian dg/dtheta. This system can be solved for x:
        f(x0, theta0, dtheta) := dx = -inv(A(x0, theta0)) @ B(x0, theta0) @ dtheta

    It is well-known that this linear approximation is poor when dtheta is large relative to theta0. A
    solution to this problem is to decompse dtheta into a sequence of smaller -- presumably more accurate -- steps,
    and iteratively update [x0, theta0] in the following fashion:
        1. Initialize x_t = x0, theta_t = theta0
        2. Compute step_size = (theta - theta0) / n_steps
        3. For n_steps:
            1. Compute dx = f(x=x_t, theta=theta_t, dtheta=step_size)
            2. Update x_t = x_t + dx, theta_t = theta_t + step_size

    Using this algorithm, and given an infinite compute budget, g(x0, theta) can be computed to arbitrary precision.
    """
    x0 = np.atleast_1d(float_to_array(x0))
    theta0 = np.atleast_1d(float_to_array(theta0))
    theta_final = np.atleast_1d(float_to_array(theta_final))

    output = np.zeros((n_steps + 1, x0.size + theta0.size))
    output[0, : x0.size] = x0
    output[0, x0.size :] = theta0

    dtheta = theta_final - theta0
    x = np.concatenate((x0, theta0))
    step_size = dtheta / n_steps

    for t in range(1, n_steps + 1):
        dx = f_step(step_size, x).ravel()
        x = x + np.concatenate((dx, step_size))
        output[t, :] = x
        progress_bar.update(1)

    return output


def compile_numba_euler_func(variables, parameters, equations, jacobian=None):
    if jacobian is None:
        A_mat = equations.jacobian(variables)
    else:
        A_mat = jacobian
    B_mat = sp.Matrix([[eq.diff(x) for x in parameters] for eq in equations])

    inital_state_subs = {
        x: sp.Symbol(f"{x.name}_0", **x.assumptions0) for x in variables + parameters
    }

    A_sub = A_mat.subs(inital_state_subs)
    Bv = B_mat.subs(inital_state_subs) @ sp.Matrix([[x] for x in parameters])

    nb_A_sub = numba_lambdify(
        exog_vars=parameters, expr=A_sub, endog_vars=list(inital_state_subs.values())
    )
    nb_B_sub = numba_lambdify(
        exog_vars=parameters, expr=Bv, endog_vars=list(inital_state_subs.values())
    )

    @nb.njit(cache=True)
    def f_dX(endog, exog):
        A = nb_A_sub(endog, exog)
        B = nb_B_sub(endog, exog)

        return -np.linalg.solve(A, B)

    f_euler = partial(euler_approx, f_dX)

    return f_euler


def compile_numba_cge_functions(
    cge_model: CGEModel, functions_to_compile: list[CompiledFunctions], *args, **kwargs
):
    _log.info("Compiling model to numba")

    unpacked_equation_symbols = cge_model.unpacked_equation_symbols
    unpacked_variable_symbols = cge_model.unpacked_variable_symbols
    unpacked_parameter_symbols = cge_model.unpacked_parameter_symbols

    # Always compile the system -- used to check residuals
    f_system = numba_lambdify(
        unpacked_variable_symbols,
        sp.Matrix(unpacked_equation_symbols),
        unpacked_parameter_symbols,
        ravel_outputs=True,
    )

    # Optional functions
    f_jac = None
    f_resid = None
    f_grad = None
    f_hess = None
    f_hessp = None
    f_euler = None

    jac = None

    if "root" in functions_to_compile:
        jac, f_jac = compile_numba_jacobian_function(
            unpacked_equation_symbols, unpacked_variable_symbols, unpacked_parameter_symbols
        )

    if "minimize" in functions_to_compile:
        # Symbolically compute loss function and derivatives
        squared_loss, f_resid = compile_numba_error_function(
            unpacked_equation_symbols, unpacked_variable_symbols, unpacked_parameter_symbols
        )

        grad, f_grad = compile_numba_gradient_function(
            squared_loss, unpacked_variable_symbols, unpacked_parameter_symbols
        )

        hess, f_hess = compile_numba_jacobian_function(
            grad, unpacked_variable_symbols, unpacked_parameter_symbols
        )
        hessp, f_hessp = compile_numba_hessp_function(
            grad, unpacked_variable_symbols, unpacked_parameter_symbols
        )

    if "euler" in functions_to_compile:
        f_euler_inner = compile_numba_euler_func(
            unpacked_variable_symbols, unpacked_parameter_symbols, unpacked_equation_symbols, jac
        )

        f_euler = wrap_numba_euler_function(
            f_euler_inner,
            variables=cge_model.variables,
            parameters=cge_model.parameters,
            coords=cge_model.coords,
        )

    return f_system, f_jac, f_resid, f_grad, f_hess, f_hessp, f_euler
