import functools as ft

import numpy as np
import pytensor
import pytensor.tensor as pt

from cge_modeling.compile.pytensor_tools import flat_tensor_to_ragged_list, flatten_equations


def eval_func_maybe_exog(X, exog, f, has_exog):
    if has_exog:
        out = f(*X, *exog)
    else:
        out = f(*X)

    return out


def _newton_step(flat_X, exog, F, J, step_size, has_exog, shapes):
    X = flat_tensor_to_ragged_list(flat_X, shapes)
    F_X = eval_func_maybe_exog(X, exog, F, has_exog)
    J_X = eval_func_maybe_exog(X, exog, J, has_exog)

    flat_F_X = flatten_equations(F_X)
    flat_new_X = flat_X - step_size * pt.linalg.solve(J_X, flat_F_X)

    new_X = flat_tensor_to_ragged_list(flat_new_X, [x.type.shape for x in X])
    flat_F_new_X = eval_func_maybe_exog(new_X, exog, F, has_exog)

    return flat_X, flat_new_X, flat_F_X, flat_F_new_X


def no_op(X):
    return X, X, X, X


def compute_norms(X, new_X, F_X, F_new_X):
    norm_X = pt.linalg.norm(X, ord=1)
    norm_new_X = pt.linalg.norm(new_X, ord=1)
    norm_root = pt.linalg.norm(F_X, ord=1)
    norm_root_new = pt.linalg.norm(F_new_X, ord=1)
    norm_step = pt.linalg.norm(new_X - X, ord=1)

    return norm_X, norm_new_X, norm_root, norm_root_new, norm_step


def _check_convergence(norm_step, norm_root, tol):
    #     new_converged = pt.or_(pt.lt(norm_step, tol), pt.lt(norm_root, tol))
    new_converged = pt.lt(norm_root, tol)
    return new_converged


def check_convergence(norm_step, norm_root, converged, tol):
    return pytensor.ifelse(converged, np.array(True), _check_convergence(norm_step, norm_root, tol))


def check_stepsize(norm_root, norm_root_new, step_size, initial_step_size):
    is_decreasing = pt.lt(norm_root_new, norm_root)

    return pytensor.ifelse(
        is_decreasing,
        (is_decreasing, initial_step_size),
        (is_decreasing, step_size * 0.5),
    )


def backtrack_if_not_decreasing(is_decreasing, X, new_X):
    return pytensor.ifelse(is_decreasing, new_X, X)


def scan_body(*args, F, J, initial_step_size, tol, has_exog, n_endog, n_exog):
    X = args[:n_endog]
    converged, step_size, n_steps = args[n_endog : n_endog + 3]
    exog = args[-n_exog:]

    shapes = [x.type.shape for x in X]
    flat_X = flatten_equations(X)

    out = pytensor.ifelse(
        converged,
        no_op(flat_X),
        _newton_step(flat_X, exog, F, J, step_size, has_exog, shapes),
    )

    flat_X, flat_new_X, flat_F_X, flat_F_new_X = (out[i] for i in range(4))
    norm_X, norm_new_X, norm_root, norm_root_new, norm_step = compute_norms(
        flat_X, flat_new_X, flat_F_X, flat_F_new_X
    )
    is_converged = check_convergence(norm_step, norm_root, converged, tol)

    is_decreasing, new_step_size = check_stepsize(
        norm_root, norm_root_new, step_size, initial_step_size
    )

    flat_return_X = backtrack_if_not_decreasing(is_decreasing, flat_X, flat_new_X)
    return_X = flat_tensor_to_ragged_list(flat_return_X, shapes)
    new_n_steps = n_steps + (1 - is_converged)

    return [*return_X, is_converged, new_step_size, new_n_steps]


def _process_root_data(data: dict[str, np.ndarray] | None) -> list[pt.TensorLike]:
    if data is None:
        return [pt.as_tensor_variable(np.nan, name="dummy_exog", shape=())]
    else:
        out = []
        for name in data.keys():
            x = np.array(data[name], dtype=pytensor.config.floatX)
            out.append(pt.as_tensor_variable(x, name=name, shape=x.shape))
        return out


def root(
    f: pt.Op,
    f_jac: pt.Op,
    initial_data: dict[str, np.ndarray | float],
    parameters: dict[str, np.ndarray | float] | None = None,
    step_size: int = 1,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[list[pt.TensorVariable], pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]:
    """
    Find the root of a system of equations using Newton's method with backtracking.

    Parameters
    ----------
    f: pytensor Op
        A pytensor Op, typically created by CGEModel.compile_equations_to_pytensor, that takes a ragged list of
        variables and parameters as input and returns a vector of residuls as output.
    f_jac: pytensor Op
        A pytensor Op, typically created by CGEModel.compile_equations_to_pytensor, that takes a ragged list of
        variables and parameters as input and returns the inverse of the Jacobian of the system of equations as output.
    initial_data: dict[str, np.ndarray]
        A dictionary of initial values for the variables in the system of equations. The keys are the names of the
        variables and the values are the initial values.
    parameters: Optional, dict[str, np.ndarray]
        A dictionary of exogenous parameters for the system of equations. The keys are the names of the parameters and
        the values are the parameter values.
    step_size: int
        The initial step size to use in Newton's method
    max_iter: int
        The maximum number of iterations to run Newton's method
    tol: float
        The tolerance for convergence. The algorithm will stop if the norm of the residuals is less than the tolerance
        or if the norm of the step size is less than the tolerance.

    Returns
    -------
    root_histories: list[pt.TensorVariable]
        A list of the values of the variables at each iteration of the algorithm.

    converged: pt.TensorVariable
        A boolean tensor indicating whether the algorithm converged at each iteration.

    step_size: pt.TensorVariable
        The step size used at each iteration. Useful to diagnose whether the algorithm is converging.

    n_steps: pt.TensorVariable
        The number of steps taken at each iteration. Useful to diagnose whether the algorithm is converging.
    """

    init_step_size = np.float64(step_size)
    converged = np.array(False)
    n_steps = 0
    has_exog = parameters is not None

    x0 = _process_root_data(initial_data)
    exog = _process_root_data(parameters)

    n_endog = len(x0)
    n_exog = len(exog)

    root_func = ft.partial(
        scan_body,
        F=f,
        J=f_jac,
        initial_step_size=init_step_size,
        tol=tol,
        has_exog=has_exog,
        n_endog=n_endog,
        n_exog=n_exog,
    )

    outputs, updates = pytensor.scan(
        root_func,
        outputs_info=[*x0, converged, init_step_size, n_steps],
        non_sequences=exog,
        n_steps=max_iter,
        strict=True,
    )

    *root_histories, converged, step_size, n_steps = outputs

    return root_histories, converged, step_size, n_steps
