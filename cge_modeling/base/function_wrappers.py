import functools as ft

from collections.abc import Callable

import numpy as np
import pytensor.compile

from numba_progress.progress import ProgressBar as NumbaProgressBar

from cge_modeling.base.primitives import Parameter, Variable
from cge_modeling.base.utilities import (
    flat_array_to_variable_dict,
    infer_object_shape_from_coords,
    variable_dict_to_flat_array,
)


def wrap_pytensor_func_for_scipy(
    f: pytensor.compile.function.types.Function,
    variable_list: list[Variable],
    parameter_list: list[Parameter],
    coords: dict[str, list[str | int, ...]],
    include_p: bool = False,
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
    coords: dict[str, list[str, ...]]
        Dictionary of coordinates mapping dimension names to lists of labels associated with that dimension.
    include_p: bool
        If true, a 3rd argument is included in the inner function (useful

    Returns
    -------
    inner_f: Callable
        A wrapped version of the input function that accepts a single long vector of variables as input, and a single
        long vector of parameters as second input. The wrapped function will unpack these vectors into a dictionary of
        variables and parameters, and then unpack that dictionary into keyword arguments to the original function.
    """
    if not include_p:

        @ft.wraps(f)
        def inner_f(x0, theta):
            # Scipy will pass x0 as a single long vector, and theta separately (but also as a single long vector).
            inputs = np.r_[x0, theta]
            data = flat_array_to_variable_dict(inputs, variable_list + parameter_list, coords)
            return f(**data)

    else:

        @ft.wraps(f)
        def inner_f(x0, p, theta):
            # Scipy will pass x0 as a single long vector, and theta separately (but also as a single long vector).
            inputs = np.r_[x0, theta]

            data = flat_array_to_variable_dict(inputs, variable_list + parameter_list, coords)
            point_dict = flat_array_to_variable_dict(p, variable_list, coords)

            point_dict = {f"{name}_point": point for name, point in point_dict.items()}
            data.update(point_dict)

            return f(**data)

    return inner_f


def wrap_scipy_func_for_pytensor(
    f: pytensor.compile.function.types.Function,
    variable_list: list[Variable],
    parameter_list: list[Parameter],
):
    @ft.wraps(f)
    def inner(**kwargs):
        x, theta = variable_dict_to_flat_array(kwargs, variable_list, parameter_list)
        return f(x, theta)

    return inner


def wrap_fixed_values(
    f: Callable,
    fixed_values: dict[str, float | int | np.ndarray],
    model,
) -> Callable:
    """
    Wrap a CGE function to require only a subset of its inputs. Useful when optimizing under a constraint that
    certain variables are fixed at initial values.

    Parameters
    ----------
    f: Callable
        A compiled CGE function with signature f(**variables, **parameters)

    fixed_values: dict[str, Union[float, int, np.ndarray]]
        Dictionary mapping variable names to numpy arrays containing the fixed values of those variables. Keys should
        be a subset of the variable names accepted by the function f.

    model: CGEModel
        CGE model associated with the function being wrapped

    Returns
    -------
    inner: Callable
        A wrapped version of the input function that accepts only the variables not in fixed_values as keyword arguments.
    """

    variables = model.variables
    parameters = model.parameters
    coords = model.coords

    var_names = model.unpacked_variable_names

    if any(var not in var_names for var in fixed_values.keys()):
        raise ValueError(
            "User asked to fix the following variables, but these variables are not in the model: "
            f"{set(fixed_values.keys()) - set(var_names)}"
        )
    fixed_vars = list(fixed_values.keys())
    return_mask = np.array([x in fixed_values for x in model.unpacked_variable_names])

    @ft.wraps(f)
    def inner(x, theta):
        data = flat_array_to_variable_dict(np.r_[x, theta], variables, coords)
        if any(var in fixed_vars for var in data.keys()):
            raise ValueError(
                f"Values for the following variables were passed to a function that expected them to be "
                f"fixed: {set(fixed_vars) & set(data.keys())}"
            )
        data.update(fixed_values)
        new_x, new_theta = variable_dict_to_flat_array(data, variables, parameters)

        res = f(new_x, new_theta)
        if isinstance(res, float | int) or res.ndim == 0:
            return res

        if res.ndim == 1:
            return res[return_mask]
        return res[return_mask, :][:, return_mask]

    return inner


def wrap_numba_euler_function(euler_approx, variables, parameters, coords):
    @ft.wraps(euler_approx)
    def f_euler(*, theta_final, n_steps, mode=None, **data):
        # Extract the variable and parameter names from the data dictionary
        x0 = np.concatenate([np.atleast_1d(data[x.name]).ravel() for x in variables], axis=0)
        theta0 = np.concatenate([np.atleast_1d(data[x.name]).ravel() for x in parameters], axis=0)

        with NumbaProgressBar(total=n_steps) as progress_bar:
            result = euler_approx(
                x0=x0,
                theta0=theta0,
                theta_final=theta_final,
                n_steps=n_steps,
                progress_bar=progress_bar,
            )

        # Decompose the result back to a list of numpy arrays
        shapes = [infer_object_shape_from_coords(x, coords) for x in variables + parameters]
        out = []
        cursor = 0
        for shape in shapes:
            s = int(np.prod(shape))
            out.append(result[:, cursor : cursor + s].reshape(-1, *shape))
            cursor += s

        return out

    return f_euler


def return_array_from_jax_wrapper(f):
    @ft.wraps(f)
    def f_array_return(*args, **kwargs):
        return np.array(f(*args, **kwargs))

    return f_array_return
