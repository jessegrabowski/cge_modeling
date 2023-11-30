import re
from typing import Union

import numpy as np

from cge_modeling.base.primitives import Parameter, Variable


def _validate_input(obj, cls):
    if not isinstance(obj, cls):
        raise ValueError(f"Expected instance of type {cls.__name__}, found {type(obj).__name__}")


def ensure_input_is_sequence(x):
    if not isinstance(x, (list, tuple)):
        x = [x]
    return x


def _expand_var_by_index(obj: Union[Variable, Parameter], coords: dict[str, list[str]]):
    if not isinstance(obj, (Variable, Parameter)):
        raise ValueError("Expected a model object for argument obj, got {type(obj)}")

    dims = list(coords.keys())
    missing_dims = set(dims) - set(obj.dims)
    if len(missing_dims) > 0:
        raise ValueError(
            f'Found indices {", ".join(missing_dims)} on the coords that are not among the variable '
            f'indices: {", ".join(obj.dims)}'
        )

    cls = Variable if isinstance(obj, Variable) else Parameter
    out = [obj]

    for dim in dims:
        new_out = []
        for obj in out:
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
        out = new_out.copy()

    return out


def _replace_dim_marker_with_dim_name(s):
    s = re.sub("(<dim:(.+?)>)", r"\g<2>", s)
    return s


def infer_object_shape_from_coords(obj, coords):
    dims = obj.dims
    shape = tuple(len(coords[dim]) for dim in dims)
    return shape


def flat_array_to_variable_dict(x, cge_model):
    all_objects = cge_model.variables + cge_model.parameters

    d = {}
    cursor = 0
    for obj in all_objects:
        shape = infer_object_shape_from_coords(obj, cge_model.coords)
        s = int(np.prod(shape))
        d[obj.name] = x[cursor : cursor + s].reshape(shape)
        cursor += s

    return d


def variable_dict_to_flat_array(d, cge_model, concat_returns=True):
    variables = np.r_[*[np.atleast_1d(d[var.name]).ravel() for var in cge_model.variables]]
    parameters = np.r_[*[np.atleast_1d(d[var.name]).ravel() for var in cge_model.parameters]]

    return variables, parameters


def wrap_pytensor_func_for_scipy(f, cge_model):
    def inner_f(x0, theta):
        n_x = x0.shape[0]
        # Scipy will pass x0 as a single long vector, and theta as an arg.
        inputs = np.r_[x0, theta]
        data = flat_array_to_variable_dict(inputs, cge_model)
        return f(**data)

    return inner_f
