import logging

import numpy as np
import pytensor
import sympy as sp

from pytensor import tensor as pt
from pytensor.graph import node_rewriter, rewrite_graph
from pytensor.tensor.math import Prod
from pytensor.tensor.rewriting.basic import register_stabilize

from cge_modeling import CGEModel, Parameter, Variable

_log = logging.getLogger(__name__)


@register_stabilize
@node_rewriter([Prod])
def prod_to_no_zero_prod(fgraph, node):
    """
    JAX doesn't support gradient computation in the case where there are zeros in the product. We're allowed to promise
    there will never be zeros, which should always be the case for CGE models. This rewrite makes this promse for any
    product Ops that are in the graph.

    Note that this only affects product reduction Ops, it's not the same as a multiplication.
    """
    if isinstance(node.op, Prod) and not node.op.no_zeros_in_input:
        (x,) = node.inputs
        new_op = Prod(dtype=node.op.dtype, acc_dtype=node.op.dtype, no_zeros_in_input=True)
        return [new_op(x)]


def rewrite_pregrad(graph):
    """Apply simplifying or stabilizing rewrites to graph that are safe to use
    pre-grad.
    """
    return rewrite_graph(graph, include=("canonicalize", "stabilize"))


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


def flatten_equations(eqs: list[pt.TensorLike]) -> pt.TensorLike:
    """
    Flatten a list of pytensor vectors, each representing equations in a system of equations, into a single vector.

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
        [pt.atleast_1d(pt.cast(eq, pytensor.config.floatX)).ravel() for eq in eqs],
        axis=-1,
    )
    flat_expr = pt.specify_shape(flat_expr, shape=(expr_len,))
    return flat_expr


def make_printer_cache(variables: list[pt.TensorLike], parameters: list[pt.TensorLike]) -> dict:
    """
    Create a cache of PyTensor functions for printing sympy equations to pytensor.

    Parameters
    ----------
    cge_model: CGEModel
        CGEModel object
    variables: list of TensorVariable
        List of cge_model variables, converted to symbolic pytensor tensors
    parameters: list of TensorVariable
        List of cge_model parameters, converted to symbolic pytensor tensors

    Returns
    -------
    cache: dict
        A cache of variables used to print equations to pytensor
    """

    def make_key(var) -> tuple:
        return var.name, sp.Symbol, (), "floatX", ()

    cache = {make_key(var): var for var in variables + parameters}
    return cache


def flat_tensor_to_ragged_list(tensor, shapes):
    out = []
    cursor = 0

    for shape in shapes:
        s = int(np.prod(shape))
        x = tensor[cursor : cursor + s].reshape(shape)
        out.append(x)
        cursor += s

    return out


def object_to_pytensor(obj: Parameter | Variable, coords: dict[str, list[str, ...]]):
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


def make_unpacked_to_packed_map(
    cge_model: CGEModel,
    cache: dict,
    unpacked_cache: dict,
) -> dict:
    pt_vars = list(cache.values())
    cache_var_names = list(x[0] for x in cache.keys())
    unpacked_to_indexed_dict = {}

    for info, var in unpacked_cache.items():
        name, *_ = info  # info is a tuple of (name, sympy_class, broadcastable, dtype, shape)
        parent_obj = cge_model.get_unpacked_parent_object(name)
        parent_pt = pt_vars[cache_var_names.index(parent_obj.base_name)]

        # Map named indices to integer indices
        dims = parent_obj.dims
        dim_idx = [cge_model.coords[dim].index(parent_obj.dim_vals[dim]) for dim in dims]

        if len(dim_idx) == 0:
            unpacked_to_indexed_dict[var] = parent_pt
            continue

        # Index the parent object with the integer indices
        indexed_parent = parent_pt[tuple(dim_idx)]
        name = f"{parent_pt.name}"
        name += f'[{", ".join([str(x) for x in dim_idx])}]' if len(dim_idx) > 0 else ""

        indexed_parent.name = name
        unpacked_to_indexed_dict[var] = indexed_parent

    return unpacked_to_indexed_dict


def unpacked_graph_to_packed_graph(
    unpacked_graph: pt.TensorVariable | list[pt.TensorVariable], unpacked_to_indexed_dict: dict
) -> pt.TensorVariable | list[pt.TensorVariable]:
    packed_graph = pytensor.clone_replace(unpacked_graph, unpacked_to_indexed_dict)
    return packed_graph


def make_jacobian(
    system: pt.TensorLike,
    x: list[pt.TensorLike],
) -> pt.TensorVariable:
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

    rewrite_pregrad(system)
    column_list = pytensor.gradient.jacobian(system, x)

    jac = pt.concatenate([pt.atleast_2d(x).reshape((n_eq, -1)) for x in column_list], axis=-1)
    jac = pt.specify_shape(jac, shape=(n_eq, n_vars))

    return jac
