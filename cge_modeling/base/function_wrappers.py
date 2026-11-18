import functools as ft

from collections.abc import Callable
from typing import Any

import numpy as np


def wrap_compiled_function_for_scipy(
    f: Callable[..., Any],
    include_p: bool = False,
    free_mask: np.ndarray | None = None,
    x_full: np.ndarray | None = None,
) -> Callable[..., np.ndarray]:
    """
    Wrap a compiled model function for use with scipy.optimize.root or scipy.optimize.minimize.

    Parameters
    ----------
    f: Callable
        A compiled model function with signature ``f(x, theta)``, or ``f(x, p, theta)`` when ``include_p`` is True.
    include_p: bool
        If true, ``f`` takes the hessp point as a middle argument.
    free_mask: np.ndarray, optional
        Boolean mask over the flat variable vector, True for variables the solver moves. If None, every variable is
        free.
    x_full: np.ndarray, optional
        Flat variable vector holding the values of the fixed variables. Required when ``free_mask`` has a False entry.

    Returns
    -------
    inner_f: Callable
        A function of the free variables, ``inner_f(x_free, theta)`` or ``inner_f(x_free, p_free, theta)``, that
        returns a numpy array. Fixed variables enter ``f`` at their values in ``x_full``, with a hessp point of zero.
    """
    # JAX-mode functions return JAX arrays, and scipy's solvers are much slower doing their arithmetic on those.
    if free_mask is None or free_mask.all():

        @ft.wraps(f)
        def inner_f(*args):
            return np.asarray(f(*args))

        return inner_f

    if x_full is None:
        raise ValueError("x_full is required when some variables are fixed.")

    def insert_free_entries(free_values, fill_values):
        full_values = np.array(fill_values, dtype=float)
        full_values[free_mask] = free_values
        return full_values

    if not include_p:

        @ft.wraps(f)
        def inner_f(x, theta):
            return np.asarray(f(insert_free_entries(x, x_full), theta))

    else:
        zero_point = np.zeros_like(x_full)

        @ft.wraps(f)
        def inner_f(x, p, theta):
            return np.asarray(
                f(insert_free_entries(x, x_full), insert_free_entries(p, zero_point), theta)
            )

    return inner_f


def select_free_variable_entries(
    f: Callable[..., np.ndarray], free_mask: np.ndarray, variable_axes: tuple[int, ...]
) -> Callable[..., np.ndarray]:
    """
    Drop the entries of a function's output that belong to fixed variables.

    Parameters
    ----------
    f: Callable
        Function whose output is indexed by the flattened model variables along ``variable_axes``.
    free_mask: np.ndarray
        Boolean mask over the flattened model variables, True for variables that are not fixed.
    variable_axes: tuple of int
        Output axes indexed by the flattened model variables.

    Returns
    -------
    inner_f: Callable
        ``f`` with the fixed-variable entries removed along each axis in ``variable_axes``. ``f`` itself when no
        variable is fixed.
    """
    if free_mask.all():
        return f

    @ft.wraps(f)
    def inner_f(*args):
        out = f(*args)
        for axis in variable_axes:
            out = np.compress(free_mask, out, axis=axis)
        return out

    return inner_f
