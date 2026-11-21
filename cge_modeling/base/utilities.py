import logging
import re

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import product
from typing import Any, Literal, cast

import numpy as np

from better_optimize.configuration import MINIMIZE_CONFIGS

from cge_modeling.base.primitives import (
    ModelObject,
    Parameter,
    Variable,
    _pretty_print_dim_flags,
)

_log = logging.getLogger(__name__)


def unpack_equation_name(
    name: str, dims: Sequence[str], coords: Mapping[str, Sequence[str | int]]
) -> list[str]:
    """
    Name every element of an equation whose residual has ``dims``, in row-major order.

    ``<dim:d>`` markers in ``name`` are replaced by the element's label along ``d``. Labels along dims that ``name``
    does not mark are appended as ``[d=label, ...]``, so every element name is distinct.
    """
    marked_dims = [dim for dim in dims if f"<dim:{dim}>" in name]
    unmarked_dims = [dim for dim in dims if dim not in marked_dims]

    element_names = []
    for labels in product(*(coords[dim] for dim in dims)):
        dim_labels = dict(zip(dims, labels))
        element_name = _pretty_print_dim_flags(name, marked_dims, dim_labels)
        if unmarked_dims:
            suffix = ", ".join(f"{dim}={dim_labels[dim]}" for dim in unmarked_dims)
            element_name = f"{element_name} [{suffix}]"
        element_names.append(element_name)

    return element_names


def ensure_input_is_sequence(x: Any) -> Sequence[Any]:
    """
    Convert non-sequence inputs into a singleton list.

    Parameters
    ----------
    x: Any
        An anonymous input

    Returns
    -------
    x: list[Any]
        A list containing the input, or the input itself if it is already a sequence
    """
    if not isinstance(x, list | tuple):
        x = [x]
    return x


def _replace_dim_marker_with_dim_name(s: str) -> str:
    """
    Strip the dimension tag <dim:> from a description string, returning only the dimension label itself.

    Parameters
    ----------
    s: str
        A string containing a dimension marker, e.g. "<dim:i>"

    Returns
    -------
    new_s: str
        The input string with the dimension marker removed, e.g. "i"

    Notes
    -----
    The dimension markers used in description strings are ugly, but they are an easy way to facilitate swapping in
    arbitrary dimension labels when rendering equations to latex. Nevertheless, users will not want to see them when
    inspecting the "compact" form of the model. This function removes them.

    Examples
    --------
    .. code-block:: python

        from cge_modeling.base.utilities import _replace_dim_marker_with_dim_name

        s = 'Sector <dim:i> demand for good <dim:j>'
        _replace_dim_marker_with_dim_name(s)
        # Out: 'Sector i demand for good j'
    """
    s = re.sub("(<dim:(.+?)>)", r"\g<2>", s)
    return s


def infer_object_shape_from_coords(
    obj: Variable | Parameter, coords: Mapping[str, Sequence[str | int]]
) -> tuple[int, ...]:
    """
    Infer the shape of a CGE Model object from a provided coordinate dictionary.

    Parameters
    ----------
    obj: Union[Variable, Parameter]
        A CGE Model object
    coords: dict[str, list[str]]
        A dictionary of coordinates, mapping dimension names to lists of labels associated with that dimension.

    Returns
    -------
    shape: tuple[int]
        A tuple of integers representing the shape of the object.

    Notes
    -----
    CGE Model objects do not themselves carry shape information, only dimension information. This is because the shape
    information is not strictly necessary at model declaration time, and indeed might not even be known by the economist
    when designing the model.

    What will determine the shape information is the coordinate dictionary, which is known by the CGEModel object but
    not the individual Variable and Parameter instances it contains. This function allows us to infer the shape of an
    object from the coordinate dictionary.

    Examples
    --------
    .. code-block:: python

        from cge_modeling.base.primitives import Variable
        from cge_modeling.base.utilities import infer_object_shape_from_coords

        x = Variable(name='x', dims='i', description='Quantity associated with sector <dim:i> ')
        coords = {'i': ['Agriculture', 'Industrial', 'Service']}
        infer_object_shape_from_coords(x, coords)
        # Out: (3,)

        z = Variable(name='phi_X', dims='i, j', description='Quantity of <dim:j> goods in sector <dim:i> value chain')
        coords = {'i': ['Agriculture', 'Industrial', 'Service'], 'j': ['Agriculture', 'Industrial', 'Service']}
        infer_object_shape_from_coords(z, coords)
        # Out: (3, 3)
    """
    dims = cast(tuple[str, ...], obj.dims)
    shape = tuple(len(coords[dim]) for dim in dims)
    return shape


def make_flat_array_return_mask(
    x: np.ndarray,
    all_objects: Sequence[Variable | Parameter],
    omit_object_names: list[str],
    coords: Mapping[str, Sequence[str | int]],
) -> np.ndarray:
    mask = np.full(x.shape[0], True)
    cursor = 0
    for obj in all_objects:
        shape = infer_object_shape_from_coords(obj, coords)
        s = int(np.prod(shape))

        if obj.name in omit_object_names:
            mask[cursor : cursor + s] = False
        cursor += s

    return mask.astype("bool")


def flat_array_to_variable_dict(
    x: np.ndarray,
    objects: Sequence[Variable | Parameter],
    coords: Mapping[str, Sequence[str | int]],
) -> dict[str, np.ndarray]:
    """
    Convert a flat array to a dictionary of variables and parameters.

    Parameters
    ----------
    x: np.ndarray
        Flat array containing all variable and parameters in the model. The ordering is assumed to match the ordering
        given by list passed to the objects argument, with multidimensional objects flattened in row-major order.
    objects: list[Union[Variable, Parameter]]
        List of variables and parameters
    coords: dict[str, list[str]]
        Dictionary of coordinates mapping dimension names to lists of labels associated with that dimension.

    Returns
    -------
    d: dict[str, np.ndarray]
        Dictionary mapping variable and parameter names to numpy arrays containing the values of those variables and
        parameters.

    Notes
    -----
    This function is the inverse of variable_dict_to_flat_array.

    The goal of this pair of functions is to hide  flattening and concatenation operations from the user. The user
    should only have to reason about input and outputs in terms of each individual object, without ever having to
    worry about unifying them into a single long vector or large matrix.

    The ordering of the variables and parameters in the output vector is determined by the order of the variables and
    parameters in the variable_list and parameter_list arguments. Variables are ordered first, followed by parameters.
    Multidimensional objects are flattened in row-major order.

    # TODO: This function cannot currently handle batch dimensions, but it should be able to.

    Examples
    --------
    .. code-block:: python

        import numpy as np

        from cge_modeling.base.primitives import Variable
        from cge_modeling.base.utilities import flat_array_to_variable_dict

        x = Variable(name='x', dims='i', description='Output of the <dim:i> sector')
        coords = {'i': ['Agriculture', 'Industrial', 'Service']}
        flat_array_to_variable_dict(np.array([1, 2, 3]), [x], coords)
        # Out: {'x': array([1, 2, 3])}

        z = Variable(name='phi_X', dims='i, j', description='Quantity of <dim:j> goods in sector <dim:i> value chain')
        coords = {'i': ['Agriculture', 'Industrial', 'Service'], 'j': ['Agriculture', 'Industrial', 'Service']}
        flat_array_to_variable_dict(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), [z], coords)
        # Out: {'phi_X': array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])}
    """
    d = {}
    cursor = 0
    for obj in objects:
        shape = infer_object_shape_from_coords(obj, coords)
        s = int(np.prod(shape))
        value = x[cursor : cursor + s].reshape(shape)
        d[obj.name] = value
        cursor += s

    return d


def variable_dict_to_flat_array(
    d: Mapping[str, Any],
    variable_list: list[Variable],
    parameter_list: list[Parameter],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a dictionary of variables and parameters to a single long vector.

    Parameters
    ----------
    d: dict[str, np.ndarray]
        Data dictionary mapping variable and parameter names to numpy arrays. Numpy arrays can be any dimension, and are
        assumed to have dims corresponding to their object's dims.

    variable_list: list[Variable]
        List of variables in the model
    parameter_list: list[Parameter]
        List of parameters in the model

    Returns
    -------
    variables: np.ndarray
        A single long vector containing all variables in the model

    parameters: np.ndarray
        A single long vector containing all parameters in the model

    Notes
    -----
    This function is the inverse of flat_array_to_variable_dict.

    The goal of this pair of functions is to hide  flattening and concatenation operations from the user. The user
    should only have to reason about input and outputs in terms of each individual object, without ever having to
    worry about unifying them into a single long vector or large matrix.

    The ordering of the variables and parameters in the output vector is determined by the order of the variables and
    parameters in the variable_list and parameter_list arguments. Variables are ordered first, followed by parameters.
    Multidimensional objects are flattened in row-major order.

    # TODO: This function cannot currently handle batch dimensions, but it should be able to.
    """

    variables = np.concatenate([np.atleast_1d(d[var.name]).ravel() for var in variable_list])
    parameters = np.concatenate([np.atleast_1d(d[var.name]).ravel() for var in parameter_list])

    return variables, parameters


def get_method_defaults(use_grad, use_hess, use_hessp, method):
    use_grad, use_hess, use_hessp = (
        getattr(MINIMIZE_CONFIGS[method], f"uses_{name}") if arg is None else arg
        for name, arg in zip(["grad", "hess", "hessp"], [use_grad, use_hess, use_hessp])
    )
    if use_hess and use_hessp:
        use_hess = False

    return use_grad, use_hess, use_hessp


def create_final_param_dict(
    initial_params: dict[str, Any],
    final_values: dict[str, Any] | None,
    final_delta: dict[str, Any] | None,
    final_delta_pct: dict[str, Any] | None,
) -> dict[str, Any]:
    scenario_params = deepcopy(initial_params)

    final_values = final_values if final_values is not None else {}
    final_delta = final_delta if final_delta is not None else {}
    final_delta_pct = final_delta_pct if final_delta_pct is not None else {}

    all_params_to_update = [*final_values.keys(), *final_delta.keys(), *final_delta_pct.keys()]
    if len(all_params_to_update) == 0:
        raise ValueError(
            "No parameters to update! Cannot create a scenario without updating any parameters."
        )

    update_count = Counter(all_params_to_update)
    repeated_arguments = [k for k, v in update_count.items() if v > 1]
    if len(repeated_arguments) > 0:
        raise ValueError(
            f"Arguments {', '.join(repeated_arguments)} are repeated among final_values, final_delta,"
            f" and final_delta_pct. Define each scenario in exactly one way (by giving the final value, "
            f"the offset from the initial value, or the percentage change from the initial value)."
        )

    # For final values, directly insert the provided values, overwriting the initial values
    scenario_params.update(final_values)

    # For deltas, add the delta to the initial value
    for k, v in final_delta.items():
        scenario_params[k] += v

    # For percent changes, multiply the initial value by the percentage change
    for k, v in final_delta_pct.items():
        scenario_params[k] *= v

    return scenario_params


def make_indexed_name(obj: ModelObject) -> str:
    """Return the name of ``obj`` with its dim labels appended, joined by underscores."""
    return "_".join([obj.name, *map(str, cast(dict[str, str], obj.dim_vals).values())])


def _validate_dims(obj: ModelObject, dims: list[str], on_unused_dim="raise") -> list[str]:
    """
    Validate that the provided dims are associated with the provided object. If not, either raise an error or ignore
    the unused dims, based on the value of on_unused_dim.

    Parameters
    ----------
    obj: ModelObject
        A Variable or Parameter
    dims: list of str
        A list of dimension names to be validated
    on_unused_dim: str, optional
        Either 'raise' or 'ignore'. If 'raise', raise an error if any of the dims are not associated with the object.
        If 'ignore', ignore the unused dims.

    Returns
    -------
    dims: list of str
        A list of valid dims for the object
    """

    unknown_dims = set(dims) - set(cast(tuple[str, ...], obj.dims))
    if len(unknown_dims) > 0:
        unk_str = ", ".join(list(unknown_dims))
        if on_unused_dim == "raise":
            raise ValueError(
                f"Dimension expansion was requested for {obj.name} on the following dims "
                f"which are not associated with {obj.name}: {unk_str}"
            )
        elif on_unused_dim == "ignore":
            return list(set(dims) - unknown_dims)

    return dims


def expand_obj_by_indices(
    obj: ModelObject,
    coords: dict[str, list[str]],
    dims: list[str] | None = None,
    on_unused_dim: Literal["raise", "ignore"] = "raise",
) -> list[ModelObject]:
    """
    Expand a model object by creating a new object for each label associated to the requested dimensions.

    Parameters
    ----------
    obj: ModelObject
        A Variable or Parameter
    coords: dict of str-list pairs
        A dictionary mapping dimension names to lists of labels
    dims: list of str, optional
        A list of dimension names to be expanded. If None, all dimensions will be expanded.
    on_unused_dim: str, optional
        Either 'raise' or 'ignore'. If 'raise', raise an error if any of the dims are not associated with the object.
        If 'ignore', ignore the unused dims.

    Returns
    -------
    expanded_objs: list of ModelObject
        A list of model objects with the same name and description as the original object, but with the dimension
        values updated to match the labels in coords.
    """
    out = []

    if dims is None:
        dims = list(cast(tuple[str, ...], obj.dims))
    dims = _validate_dims(obj, dims, on_unused_dim)

    # Take the cartesian product of dimensions to be expanded to form all possible combinations of labels
    labels = [coords[dim] for dim in dims]

    for label_combination in product(*labels):
        new_obj = obj.copy()
        for dim, label in zip(dims, label_combination):
            new_obj.update_dim_value(dim, label)
        out.append(new_obj)
    return out
