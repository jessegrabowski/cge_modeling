import logging
import re

from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from itertools import product
from typing import Any, cast

import numpy as np

from better_optimize.constants import MINIMIZE_MODE_KWARGS

from cge_modeling.base.primitives import (
    Equation,
    Parameter,
    Variable,
    _pretty_print_dim_flags,
)

_log = logging.getLogger(__name__)
CGETypes = Variable | Parameter | Equation


def unpack_equation_strings(
    equations: list[Equation, ...], coords: dict[str, list[str | int, ...]]
) -> list[str, ...]:
    """
    Helper function to unpack equations names with <dim:i> tags to a long list of equations.

    Parameters
    ----------
    equations: Equation
        List of equation objects from a CGE model
    coords: dict
        Dictionary mapping dimension names to  lists of labels

    Returns
    -------
    unpacked_names: list[str, ...]
        List of unpacked equation names
    """

    unpacked_names = []
    dims = list(coords.keys())
    dims_pattern = "|".join([re.escape(x) for x in dims])
    pattern = f"<dim:({dims_pattern})>"

    for eq in equations:
        name = eq.name
        named_dims = re.findall(pattern, name)

        if len(named_dims) == 0:
            unpacked_names.append(name)
            continue
        labels = [coords[dim] for dim in named_dims]
        label_prod = product(*labels)
        for labels in label_prod:
            sub_dict = dict(zip(named_dims, labels))
            new_name = _pretty_print_dim_flags(name, named_dims, sub_dict)
            unpacked_names.append(new_name)

    return unpacked_names


def _validate_input(obj: Any, cls: CGETypes):
    """
    Validate that an input is an instance of a class.

    Parameters
    ----------
    obj: Any
        An object to validate
    cls: Union[Variable, Parameter]
        The class to validate against
    """

    if not isinstance(obj, cast(type, cls)):
        raise ValueError(f"Expected instance of type {cls.__name__}, found {type(obj).__name__}")


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


def _expand_var_by_index(
    obj: Variable | Parameter, coords: dict[str, list[str | int, ...]]
) -> list[Variable | Parameter]:
    """
    Create a set of CGE Model objects from a single object using the cartesian product of the object's dimensions

    Parameters
    ----------
    obj: Union[Variable, Parameter]
        A CGE Model object to be expanded
    coords: dict[str, list[str]]
        A dictionary of coordinates known by the model. Used to determine the number and labels of the new objects.

    Returns
    -------
    output_set: list[Union[Variable, Parameter]]
        A list of new CGE Model objects, one for each combination of dimensions in the input object.

    Notes
    -----
    The new objects will have the same name, description, and type as the input object, but the abstract dimension names
    will be replaced with actual labels from the coordinates dictionary.

    Examples
    --------
    .. code-block:: python

        from cge_modeling.base.primitives import Variable
        from cge_modeling.base.utilities import _expand_var_by_index

        x = Variable(name='x', dims='i', description='Quantity associated with sector <dim:i> ')
        coords = {'i': ['Agriculture', 'Industrial', 'Service']}
        expanded_x = _expand_var_by_index(x, coords)
        for var in expanded_x:
            print(var._full_latex_name)
        # Out: x_{i=\text{Agriculture}}
        #      x_{i=\text{Industrial}}
        #      x_{i=\text{Service}}
    """
    if not isinstance(obj, Variable | Parameter):
        raise ValueError(f"Expected a model object for argument obj, got {type(obj)}")

    dims = list(coords.keys())
    missing_dims = set(dims) - set(obj.dims)
    if len(missing_dims) > 0:
        raise ValueError(
            f'Found indices {", ".join(missing_dims)} on the coords that are not among the variable '
            f'indices: {", ".join(obj.dims)}'
        )

    cls = Variable if isinstance(obj, Variable) else Parameter
    output_set = [obj]

    for dim in dims:
        new_out = []
        for obj in output_set:
            labels = coords.get(dim, []) if dim in obj.dims else []
            for label in labels:
                new_dim_vals = obj.dim_vals.copy()
                new_dim_vals.update({dim: label})

                # noinspection PyArgumentList
                new_out.append(
                    cls(
                        name=obj.name,
                        dims=obj.dims,
                        dim_vals=new_dim_vals,
                        description=obj.description,
                    )
                )
        output_set = new_out.copy()

    return output_set


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
    obj: Variable | Parameter, coords: dict[str, list[str | int, ...]]
) -> tuple[int]:
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
    dims = obj.dims
    shape = tuple(len(coords[dim]) for dim in dims)
    return shape


def make_flat_array_return_mask(
    x: np.ndarray,
    all_objects: list[Variable | Parameter],
    omit_object_names: list[str],
    coords: dict[str, list[str | int, ...]],
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
    objects: list[Variable | Parameter],
    coords: dict[str, list[str | int, ...]],
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
    d: dict[str, np.ndarray],
    variable_list: [list[Variable]],
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


def flat_mask_from_param_names(param_dict, names):
    keys = list(param_dict.keys())
    count_dict = {k: np.prod(np.atleast_1d(v).shape) for k, v in param_dict.items()}
    counts = np.concatenate([np.atleast_1d(count_dict[var]).ravel() for var in keys])
    cumcounts = counts.cumsum()

    slice_dict = count_dict.copy()

    for i, k in enumerate(keys):
        slice_dict[k] = slice(cumcounts[i] - counts[i], cumcounts[i])

    flat_data = np.concatenate([np.atleast_1d(param_dict[var]).ravel() for var in keys])
    mask = np.full(flat_data.shape, False, dtype="bool")

    if isinstance(names, str):
        names = [names]

    for name in names:
        mask[slice_dict[name]] = True

    return mask


def get_method_defaults(use_grad, use_hess, use_hessp, method):
    use_grad, use_hess, use_hessp = (
        MINIMIZE_MODE_KWARGS[method][f"uses_{name}"] if arg is None else arg
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
