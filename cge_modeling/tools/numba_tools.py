import re

from collections.abc import Callable

import numba as nb
import numpy as np
import sympy as sp

from sympy.printing.numpy import NumPyPrinter, _known_functions_numpy

_known_functions_numpy.update({"DiracDelta": lambda x: 0.0, "log": "log"})


class NumbaFriendlyNumPyPrinter(NumPyPrinter):
    _kf = _known_functions_numpy

    def _print_Max(self, expr):
        # Use maximum instead of amax, because 1) we only expect scalars, and 2) numba doesn't accept amax
        return "{}({})".format(
            self._module_format(self._module + ".maximum"),
            ",".join(self._print(i) for i in expr.args),
        )

    def _print_Piecewise(self, expr):
        # Use the default python Piecewise instead of the numpy one -- looping with if conditions is faster in numba
        # anyway.
        result = []
        i = 0
        for arg in expr.args:
            e = arg.expr
            c = arg.cond
            if i == 0:
                result.append("(")
            result.append("(")
            result.append(self._print(sp.Float(e)))
            result.append(")")
            result.append(" if ")
            result.append(self._print(c))
            result.append(" else ")
            i += 1
        result = result[:-1]
        if result[-1] == "True":
            result = result[:-2]
            result.append(")")
        else:
            result.append(" else None)")
        return "".join(result)

    def _print_DiracDelta(self, expr):
        # The proper function should return infinity at one point, but the measure of that point is zero so this should
        # be fine. Pytensor defines grad(grad(max(0, x), x), x) to be zero everywhere.
        return "0.0"

    def _print_log(self, expr):
        return "{}({})".format(
            self._module_format(self._module + ".log"),
            ",".join(self._print(i) for i in expr.args),
        )


def numba_lambdify(
    exog_vars: list[sp.Symbol],
    expr: list[sp.Expr] | sp.Matrix | list[sp.Matrix],
    endog_vars: list[sp.Symbol] | None = None,
    func_signature: str | None = None,
    ravel_outputs=False,
) -> Callable:
    """
    Convert a sympy expression into a Numba-compiled function.  Unlike sp.lambdify, the resulting function can be
    pickled. In addition, common sub-expressions are gathered using sp.cse and assigned to local variables,
    giving a (very) modest performance boost. A signature can optionally be provided for numba.njit.

    Finally, the resulting function always returns a numpy array, rather than a list.

    Parameters
    ----------
    exog_vars: list of sympy.Symbol
        A list of "exogenous" variables. The distinction between "exogenous" and "enodgenous" is
        useful when passing the resulting function to a scipy.optimize optimizer. In this context, exogenous
        variables should be the choice varibles used to minimize the function.
    expr : list of sympy.Expr or sp.Matrix
        The sympy expression(s) to be converted. Expects a list of expressions (in the case that we're compiling a
        system to be stacked into a single output vector), a single matrix (which is returned as a single nd-array)
        or a list of matrices (which are returned as a list of nd-arrays)
    endog_vars : Optional, list of sympy.Symbol
        A list of "exogenous" variables, passed as a second argument to the function.
    func_signature: str
        A numba function signature, passed to the numba.njit decorator on the generated function.
    ravel_outputs: bool, default False
        If true, all outputs of the jitted function will be raveled before they are returned. This is useful for
        removing size-1 dimensions from sympy vectors.

    Returns
    -------
    numba.types.function
        A Numba-compiled function equivalent to the input expression.

    Notes
    -----
    The function returned by this function is pickleable.
    """
    ZERO_PATTERN = re.compile(r"(?<![\.\w])0([ ,\]])")
    FLOAT_SUBS = {
        sp.core.numbers.One(): sp.Float(1),
        sp.core.numbers.NegativeOne(): sp.Float(-1),
    }
    printer = NumbaFriendlyNumPyPrinter()

    if func_signature is None:
        decorator = "@nb.njit"
    else:
        decorator = f"@nb.njit({func_signature})"

    len_checks = []
    len_checks.append(
        f'    assert len(exog_inputs) == {len(exog_vars)}, "Expected {len(exog_vars)} exog_inputs"'
    )
    if endog_vars is not None:
        len_checks.append(
            f'    assert len(endog_inputs) == {len(endog_vars)}, "Expected {len(endog_vars)} exog_inputs"'
        )
    len_checks = "\n".join(len_checks)

    # Special case: expr is [[]]. This can occur if no user-defined steady-state values were provided.
    # It shouldn't happen otherwise.
    if expr == [[]]:
        sub_dict = ()
        code = ""
        retvals = ["[None]"]

    else:
        # Need to make the float substitutions so that numba can correctly interpret everything, but we have to handle
        # several cases:
        # Case 1: expr is just a single Sympy thing
        if isinstance(expr, sp.Matrix | sp.Expr):
            expr = expr.subs(FLOAT_SUBS)

        # Case 2: expr is a list. Items in the list are either lists of expressions (systems of equations),
        # single equations, or matrices.
        elif isinstance(expr, list):
            new_expr = []
            for item in expr:
                # Case 2a: It's a simple list of sympy things
                if isinstance(item, sp.Matrix | sp.Expr):
                    new_expr.append(item.subs(FLOAT_SUBS))
                # Case 2b: It's a system of equations, List[List[sp.Expr]]
                elif isinstance(item, list):
                    if all([isinstance(x, sp.Matrix | sp.Expr) for x in item]):
                        new_expr.append([x.subs(FLOAT_SUBS) for x in item])
                    else:
                        raise ValueError("Unexpected input type for expr")
            expr = new_expr
        else:
            raise ValueError("Unexpected input type for expr")
        sub_dict, expr = sp.cse(expr)

        # Converting matrices to a list of lists is convenient because NumPyPrinter() won't wrap them in np.array
        exprs = []
        for ex in expr:
            if hasattr(ex, "tolist"):
                exprs.append(ex.tolist())
            else:
                exprs.append(ex)

        codes = []
        retvals = []
        for i, expr in enumerate(exprs):
            code = printer.doprint(expr)

            delimiter = "]," if "]," in code else ","
            delimiter = ","
            code = code.split(delimiter)
            code = [" " * 8 + eq.strip() for eq in code]
            code = f"{delimiter}\n".join(code)
            code = code.replace("numpy.", "np.")

            # Handle conversion of 0 to 0.0
            code = re.sub(ZERO_PATTERN, r"0.0\g<1>", code)
            code_name = f"retval_{i}"
            retvals.append(code_name)
            code = f"    {code_name} = np.array(\n{code}\n    )"
            if ravel_outputs:
                code += ".ravel()"

            codes.append(code)
        code = "\n".join(codes)

    input_signature = "exog_inputs"
    unpacked_inputs = "\n".join(
        [
            f"    {getattr(x, 'safe_name', x.name)} = exog_inputs[{i}]"
            for i, x in enumerate(exog_vars)
        ]
    )
    if endog_vars is not None:
        input_signature += ", endog_inputs"
        exog_unpacked = "\n".join(
            [
                f"    {getattr(x, 'safe_name', x.name)} = endog_inputs[{i}]"
                for i, x in enumerate(endog_vars)
            ]
        )
        unpacked_inputs += "\n" + exog_unpacked

    assignments = "\n".join(
        [f"    {x} = {printer.doprint(y).replace('numpy.', 'np.')}" for x, y in sub_dict]
    )
    returns = f'[{",".join(retvals)}]' if len(retvals) > 1 else retvals[0]
    full_code = f"{decorator}\ndef f({input_signature}):\n{unpacked_inputs}\n\n{assignments}\n\n{code}\n\n    return {returns}"

    docstring = f"'''Automatically generated code:\n{full_code}'''"
    code = f"{decorator}\ndef f({input_signature}):\n    {docstring}\n{len_checks}\n{unpacked_inputs}\n\n{assignments}\n\n{code}\n\n    return {returns}"

    exec(code)
    return locals()["f"]


@nb.njit(cache=True)
def float_to_array(arr):
    return np.asarray(arr, np.float64)


@nb.njit(cache=True, nogil=True)
def euler_approx(f, x0, theta0, theta, n_steps, progress_bar):
    """
    Compute the solution to a non-linear function g(x, theta + dtheta) by iteratively computing a linear approximation
    f(x[t], theta + epsilon[t]) at the point (f(x[t-1], theta + epsilon[t-1]), theta + epsilon[t-1]), where epsilon[-1] = dtheta

    Parameters
    ----------
    f: njit function
        Linearized function to be approximated. Must have signature f(endog, exog) -> array[:]

    x0: np.ndarray
        Array of values of model variables representing the point at which g is linearized.

    theta0: np.ndarray
        Array of model parameter values representing the point at which g is linearized.

    theta: np.ndarray
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
    theta = np.atleast_1d(float_to_array(theta))

    output = np.zeros((n_steps + 1, x0.size + theta0.size))
    output[0, : x0.size] = x0
    output[0, x0.size :] = theta0

    dtheta = theta - theta0
    x = np.concatenate((x0, theta0))
    step_size = dtheta / n_steps

    for t in range(1, n_steps + 1):
        dx = f(step_size, x).ravel()
        x = x + np.concatenate((dx, step_size))
        output[t, :] = x
        progress_bar.update(1)

    return output
