import re
from typing import Union

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
