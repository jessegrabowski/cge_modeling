import functools as ft

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pytensor.compile

from cge_modeling.base.primitives import Parameter, Variable
from cge_modeling.base.utilities import (
    flat_array_to_variable_dict,
    variable_dict_to_flat_array,
)


def wrap_pytensor_func_for_scipy(
    f: Callable[..., Any],
    variable_list: list[Variable],
    parameter_list: list[Parameter],
    coords: Mapping[str, Sequence[str | int]],
    include_p: bool = False,
    fixed_values: Mapping[str, np.ndarray | float] | None = None,
) -> Callable:
    """
    Wrap a PyTensor function for use with scipy.optimize.root or scipy.optimize.minimize.

    Parameters
    ----------
    f: Callable
        A compiled PyTensor function with an input signature f(**data), where data is a dictionary mapping variable
        and parameter names to numpy arrays containing the values of those variables and parameters.

    variable_list: list[Variable]
        List of model variables
    parameter_list: list[Parameter]
        List of model parameters
    coords: dict[str, list[str]]
        Dictionary of coordinates mapping dimension names to lists of labels associated with that dimension.
    include_p: bool
        If true, a 3rd argument is included in the inner function (useful
    fixed_values: dict of str to np.ndarray, optional
        Values for model variables that are held fixed. These variables are excluded from ``variable_list`` and the
        solver vector, and are passed to ``f`` at the given values. Their hessp point is zero.

    Returns
    -------
    inner_f: Callable
        A wrapped version of the input function that accepts a single long vector of variables as input, and a single
        long vector of parameters as second input. The wrapped function will unpack these vectors into a dictionary of
        variables and parameters, and then unpack that dictionary into keyword arguments to the original function.
    """
    fixed_values = {} if fixed_values is None else fixed_values
    fixed_points = {
        f"{name}_point": np.zeros_like(np.asarray(value, dtype=float))
        for name, value in fixed_values.items()
    }

    if not include_p:

        @ft.wraps(f)
        def inner_f(x0, theta):
            # Scipy will pass x0 as a single long vector, and theta separately (but also as a single long vector).
            inputs = np.r_[x0, theta]
            data = flat_array_to_variable_dict(inputs, variable_list + parameter_list, coords)
            return f(**data, **fixed_values)

    else:

        @ft.wraps(f)
        def inner_f(x0, p, theta):
            # Scipy will pass x0 as a single long vector, and theta separately (but also as a single long vector).
            inputs = np.r_[x0, theta]

            data = flat_array_to_variable_dict(inputs, variable_list + parameter_list, coords)
            point_dict = flat_array_to_variable_dict(p, variable_list, coords)

            point_dict = {f"{name}_point": point for name, point in point_dict.items()}
            data.update(point_dict)

            return f(**data, **fixed_values, **fixed_points)

    return inner_f


def wrap_scipy_func_for_pytensor(
    f: pytensor.compile.Function,
    variable_list: list[Variable],
    parameter_list: list[Parameter],
    wrapping_hessp: bool = False,
):
    if wrapping_hessp:

        @ft.wraps(f)
        def inner(**kwargs):
            """
            Wraps a PyTensor function to accept a single long vector of variables and parameters, and returns the
            result of the function.
            """
            points = [f"{x.name}_point" for x in variable_list]
            p = np.concatenate([np.atleast_1d(kwargs.pop(name)).ravel() for name in points])

            x, theta = variable_dict_to_flat_array(kwargs, variable_list, parameter_list)
            return f(x, p, theta)

    else:

        @ft.wraps(f)
        def inner(**kwargs):
            x, theta = variable_dict_to_flat_array(kwargs, variable_list, parameter_list)
            return f(x, theta)

    return inner


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


def return_array_from_jax_wrapper(f):
    @ft.wraps(f)
    def f_array_return(*args, **kwargs):
        return np.array(f(*args, **kwargs))

    return f_array_return
