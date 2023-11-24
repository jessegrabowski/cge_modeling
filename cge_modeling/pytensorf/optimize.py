import functools as ft

import numpy as np
import pytensor
import pytensor.tensor as pt


def eval_func_maybe_exog(X, exog, f):
    if hasattr(exog, "data") and np.all(np.isnan(exog.data)):
        out = f(*X)
    else:
        out = f(*X, *exog)

    return out


def _newton_step(X, exog, F, J_inv, step_size, tol):
    F_X = eval_func_maybe_exog(X, exog, F)
    J_inv_X = eval_func_maybe_exog(X, exog, J_inv)

    new_X = X - step_size * J_inv_X @ F_X
    F_new_X = eval_func_maybe_exog(new_X, exog, F)

    return (X, new_X, F_X, F_new_X)


def no_op(X, F, J_inv, step_size, tol):
    return (X, X, X, X)


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
    return pytensor.ifelse.ifelse(
        converged, np.array(True), _check_convergence(norm_step, norm_root, tol)
    )


def check_stepsize(norm_root, norm_root_new, step_size, initial_step_size):
    is_decreasing = pt.lt(norm_root_new, norm_root)

    return pytensor.ifelse.ifelse(
        is_decreasing, (is_decreasing, initial_step_size), (is_decreasing, step_size * 0.5)
    )


def backtrack_if_not_decreasing(is_decreasing, X, new_X):
    return pytensor.ifelse.ifelse(is_decreasing, new_X, X)


def scan_body(X, converged, step_size, n_steps, tol, exog, F, J_inv, initial_step_size):
    out = pytensor.ifelse.ifelse(
        converged,
        no_op(X, F, J_inv, step_size, tol),
        _newton_step(X, exog, F, J_inv, step_size, tol),
    )

    X, new_X, F_X, F_new_X = (out[i] for i in range(4))
    norm_X, norm_new_X, norm_root, norm_root_new, norm_step = compute_norms(X, new_X, F_X, F_new_X)
    is_converged = check_convergence(norm_step, norm_root, converged, tol)

    is_decreasing, new_step_size = check_stepsize(
        norm_root, norm_root_new, step_size, initial_step_size
    )

    return_X = backtrack_if_not_decreasing(is_decreasing, X, new_X)
    new_n_steps = n_steps + (1 - is_converged)

    return return_X, is_converged, new_step_size, new_n_steps


def root(f, f_jac_inv, x0, exog=None, step_size=1, max_iter=100, tol=1e-8):
    init_step_size = np.float64(step_size)
    root_func = ft.partial(scan_body, F=f, J_inv=f_jac_inv, initial_step_size=init_step_size)
    converged = np.array(False)
    n_steps = 0

    if exog is None:
        exog = pt.as_tensor_variable(np.nan)
    else:
        exog = pt.as_tensor_variable(exog)

    outputs, updates = pytensor.scan(
        root_func,
        outputs_info=[x0, converged, init_step_size, n_steps],
        non_sequences=[tol, exog],
        n_steps=max_iter,
        strict=True,
    )

    root, converged, step_size, n_steps = outputs

    return root, converged, step_size, n_steps
