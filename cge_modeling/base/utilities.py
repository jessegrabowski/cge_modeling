import functools as ft
import re
from typing import Any, Callable, Sequence, Union, cast

import numpy as np
from fastprogress.fastprogress import ProgressBar, progress_bar

from cge_modeling.base.primitives import Equation, Parameter, Variable

CGETypes = Union[Variable, Parameter, Equation]


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
    if not isinstance(x, (list, tuple)):
        x = [x]
    return x


def _expand_var_by_index(
    obj: Union[Variable, Parameter], coords: dict[str, list[str, ...]]
) -> list[Union[Variable, Parameter]]:
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
    if not isinstance(obj, (Variable, Parameter)):
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
    obj: Union[Variable, Parameter], coords: dict[str, list[str, ...]]
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


def flat_array_to_variable_dict(
    x: np.ndarray, objects: list[Union[Variable, Parameter]], coords: dict[str, list[str, ...]]
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
    d: dict[str, np.ndarray], variable_list: [list[Variable]], parameter_list: list[Parameter]
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


def wrap_pytensor_func_for_scipy(
    f: Callable,
    variable_list: list[Variable],
    parameter_list: list[Parameter],
    coords: dict[str, list[str, ...]],
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

    Returns
    -------
    inner_f: Callable
        A wrapped version of the input function that accepts a single long vector of variables as input, and a single
        long vector of parameters as second input. The wrapped function will unpack these vectors into a dictionary of
        variables and parameters, and then unpack that dictionary into keyword arguments to the original function.
    """

    @ft.wraps(f)
    def inner_f(x0, theta):
        # Scipy will pass x0 as a single long vector, and theta separately (but also as a single long vector).
        inputs = np.r_[x0, theta]
        data = flat_array_to_variable_dict(inputs, variable_list + parameter_list, coords)
        return f(**data)

    return inner_f


class CostFuncWrapper:
    def __init__(self, maxeval=5000, progressbar=True, f=None, f_jac=None, f_hess=None):
        self.n_eval = 0
        self.maxeval = maxeval
        self.f = f
        self.use_jac = False
        self.use_hess = False

        if f_jac is None:
            self.desc = "f = {:,.5g}"
        else:
            if f_hess is not None:
                self.f_hess = f_hess
                self.use_hess = True
                self.desc = "f = {:,.5g}, ||grad|| = {:,.5g}, ||hess|| = {:,.5g}"

            self.f_jac = f_jac
            self.use_jac = True
            self.desc = "f = {:,.5g}, ||grad|| = {:,.5g}"

        self.previous_x = None
        self.progressbar = progressbar
        if progressbar:
            self.progress = progress_bar(range(maxeval), total=maxeval, display=progressbar)
            self.progress.update(0)
        else:
            self.progress = range(maxeval)

    def __call__(self, x, params):
        grad = None
        hess = None
        value = self.f(x, params)

        if self.use_jac:
            grad = self.f_jac(x, params)
            if self.use_hess:
                hess = self.f_hess(x, params)
            if np.all(np.isfinite(x)):
                self.previous_x = x
        else:
            self.previous_x = x

        if self.n_eval % 10 == 0:
            self.update_progress_desc(value, grad, hess)

        if self.n_eval > self.maxeval:
            self.update_progress_desc(value, grad, hess)
            raise StopIteration

        self.n_eval += 1
        if self.progressbar:
            assert isinstance(self.progress, ProgressBar)
            self.progress.update_bar(self.n_eval)

        if self.use_jac:
            if self.use_hess:
                return value, grad  # , hess
            else:
                return value, grad
        else:
            return value

    def update_progress_desc(
        self, value: float, grad: np.float64 = None, hess: np.float64 = None
    ) -> None:
        if isinstance(value, np.ndarray):
            value = (value**2).sum()

        if self.progressbar:
            if grad is None:
                self.progress.comment = self.desc.format(value)
            else:
                if hess is None:
                    norm_grad = np.linalg.norm(grad)
                    self.progress.comment = self.desc.format(value, norm_grad)
                else:
                    norm_grad = np.linalg.norm(grad)
                    norm_hess = np.linalg.norm(hess)
                    self.progress.comment = self.desc.format(value, norm_grad, norm_hess)
