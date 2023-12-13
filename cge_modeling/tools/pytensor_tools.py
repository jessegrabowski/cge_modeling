from typing import Any, Sequence, Union, cast

import numpy as np
import pytensor
import pytensor.tensor as pt
import sympy as sp

from cge_modeling.base.primitives import Parameter, Variable


def object_to_pytensor(obj: Union[Parameter, Variable], coords: dict[str, list[str, ...]]):
    """
    Convert a CGE model object to a PyTensor object.

    Parameters
    ----------
    obj: Union[Parameter, Variable]
        The CGE model object to convert
    coords: dict[str, Sequence]
        A dictionary of coordinate names and values known by the CGE model. Used to determine the shapes and dimensions
        of the returned PyTensor object

    Returns
    -------
    pytensor_obj: pytensor.tensor.TensorVariable
        The PyTensor object representing the CGE model object

    Notes
    -----
    Distinction between Parameters and Variables is lost in the conversion to symbolic tensors. It is up to the user to
    maintain this distinction.

    Examples
    --------
    .. code-block:: python

        from cge_modeling.base.primitives import Parameter
        from cge_modeling.pytensorf.compile import object_to_pytensor

        K_d = Parameter(name='K_d', dims='i', description='Capital demand')
        coords = {'i': [0, 1, 2]}
        pt_K_d = object_to_pytensor(K_d, coords)
        pt_K_d.eval({pt_K_d: [1, 2, 3]})
        # Out: array([1., 2., 3.])
    """

    name = obj.name
    dims = obj.dims

    shape = tuple(list(map(lambda dim: len(coords[dim]), dims)))

    return pt.tensor(name, shape=shape)


def make_printer_cache(variables: list[pt.TensorLike], parameters: list[pt.TensorLike]) -> dict:
    """
    Create a cache of PyTensor functions for printing sympy equations to pytensor.

    Parameters
    ----------
    cge_model: CGEModel
        CGEModel object
    variables: list[pytensor.tensor.TensorVariable]
        List of cge_model variables, converted to symbolic pytensor tensors
    parameters: list[pytensor.tensor.TensorVariable]
        List of cge_model parameters, converted to symbolic pytensor tensors

    Returns
    -------
    cache: dict[tuple[str, sp.Symbol, tuple, str, tuple], pt.TensorLike]
        A cache of variables used to print equations to pytensor
    """

    def make_key(name) -> tuple:
        return name, sp.Symbol, (), "floatX", ()

    cache = {make_key(var.name): var for var in variables + parameters}
    return cast(dict, cache)


def flatten_equations(eqs: list[pt.TensorLike]) -> pt.TensorLike:
    """
    Flatten a list of pytensor vectors, each representing equations in a system of equations,  into a single vector.

    Parameters
    ----------
    eqs: list[pytensor.tensor.TensorVariable]
        A list of pytensor vectors, each representing equations in a system of equations

    Returns
    -------
    flat_equations: pytensor.tensor.TensorVariable
        A single pytensor vector representing the equations in the system of equations

    Notes
    -----
    In the context of CGE modeling, each equation in the model can be thought of as a vector (or matrix) of equations,
    with each dimension corresponding to a single variable index. This perspective is useful for efficient vectorized
    computation of each "block" of the economic model. However, when solving for the equilibrium, we need to flatten
    these multidimensional expressions into a single vector to compute the Jacobian of the system. This function
    performs that flattening operation.
    """
    expr_len = 0
    for eq in eqs:
        if isinstance(eq, pt.TensorVariable):
            expr_len += int(np.prod(eq.type.shape))
        else:
            expr_len += int(np.prod(np.atleast_1d(eq).size))

    # expr_len = int(np.sum([np.prod(eq.type.shape) for eq in eqs]))
    flat_expr = pt.concatenate(
        [pt.atleast_1d(pt.cast(eq, pytensor.config.floatX)).ravel() for eq in eqs], axis=-1
    )
    flat_expr = pt.specify_shape(flat_expr, shape=(expr_len,))
    return flat_expr


def make_jacobian(system: pt.TensorLike, x: list[pt.TensorLike]) -> pt.TensorLike:
    """
    Make a Jacobian matrix from a system of equations and a list of variables.

    Parameters
    ----------
    system: pytensor.tensor.TensorVariable
        A vector of equations
    x: list[pytensor.tensor.TensorVariable]
        A list of variables

    Returns
    -------
    jac: pytensor.tensor.TensorVariable
        The Jacobian matrix of the system of equations with respect to the variables

    Notes
    -----
    The Jacobian of the system of equations is the matrix of partial derivatives of each equation with respect to each
    variable. The rows of the matrix correspond to the equations in the system, while the columns correspond to the
    variables. This function computes the Jacobian matrix from a vector of equations and a list of variables.
    """
    n_eq = system.type.shape[0]
    n_vars = int(np.sum([np.prod(var.type.shape) for var in x]))

    column_list = pytensor.gradient.jacobian(system, x)
    jac = pt.concatenate([pt.atleast_2d(x).reshape((n_eq, -1)) for x in column_list], axis=-1)
    jac = pt.specify_shape(jac, shape=(n_eq, n_vars))

    return jac


def at_least_list(x: Any) -> list[Any]:
    """
    Wrap non-list objects in a list

    Parameters
    ----------
    x: Any
        An object to wrap in a list if it is not already a list

    Returns
    -------
    x: list
        The input object wrapped in a list if it was not already a list
    """
    if isinstance(x, list):
        return x
    else:
        return [x]


def clone_and_rename(x: pt.TensorVariable, suffix: str = "_next") -> pt.TensorVariable:
    """
    Clone a pytensor tensor and rename it by appending a suffix to its name.

    Cloning variables is useful for creating intermediate "dummy" variables that can be replaced by graph rewrites,
    mostly commonly in vectorization.

    Parameters
    ----------
    x: pytensor.tensor.TensorVariable
        The pytensor variable to clone
    suffix: str
        The suffix to append to the cloned variable's name

    Returns
    -------
    x_new: pytensor.tensor.TensorVariable
        The cloned variable with the new name
    """
    x_new = x.clone()
    x_new.name = f"{x.name}{suffix}"
    return x_new
