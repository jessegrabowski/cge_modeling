import re
import time

from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import sympy as sp

from sympy.abc import greeks
from sympy.printing.latex import latex

greeks = [*list(greeks), "varepsilon"]
Greeks = [x.title() for x in greeks]
GREEK_PATTERN = "|".join(
    [f"(((^{greek})|((?=[_^]){greek})|((?<=[_^]){greek})))" for greek in greeks + Greeks]
)


def _pretty_print_dim_flags(s, dims, dim_vals):
    for dim in dims:
        dim_val = dim_vals.get(dim, None)
        if dim_val is None:
            continue
        pattern = f"<dim:{dim!s}>"
        s = re.sub(pattern, str(dim_val), s)
    return s


def _tokenize_latex_subscript(subscript):
    """
    Given input of the form "_{...}", where ... is a comma separated list of subscript values,
    return a list of the individual subscript values.
    """
    if subscript.startswith("_{"):
        subscript = subscript[2:]
    if subscript.endswith("}"):
        subscript = subscript[:-1]

    tokens = [x.strip() for x in subscript.split(",")]
    return tokens if tokens != [""] else []


@dataclass(slots=True, order=True, frozen=False, repr=False)
class ModelObject(ABC):
    name: str
    dims: tuple[str, ...] | list[str, ...] | str | None = None
    dim_vals: dict[str, str] | None = None
    latex_name: str | None = None
    _full_latex_name: str | None = None
    description: str | None = None
    base_name: str = None
    real: bool = True
    positive: bool = True
    extend_subscript: bool | int = False
    assumptions: dict = field(default_factory=dict)

    def __post_init__(self):
        self.assumptions["real"] = self.real
        self.assumptions["positive"] = self.positive

        object.__setattr__(self, "sort_index", self.name)
        object.__setattr__(self, "dims", self._initialize_index())
        object.__setattr__(self, "dim_vals", self._initialize_dim_vals())
        object.__setattr__(self, "base_name", self.name)
        object.__setattr__(self, "latex_name", self._initialize_latex_name())
        object.__setattr__(self, "_full_latex_name", self._initialize_full_latex_name())
        object.__setattr__(self, "description", self._initialize_description())

    def _make_latex_subscript(self):
        idx_strs = []
        for dim in self.dims:
            dim_val = self.dim_vals.get(dim, None)
            dim_val_text = "" if dim_val is None else "=\\text{" + str(dim_val) + "}"
            idx_strs.append(f"{dim}{dim_val_text}")

        if len(idx_strs) == 0:
            return ""

        return "_{" + ", ".join(idx_strs) + "}"

    def _initialize_index(self):
        dims = self.dims

        # If dims is None that's fine, replace it with an empty tuple
        if dims is None:
            return ()

        # Case 1: Index is empty (valid), or a tuple of strings (valid)
        if isinstance(dims, tuple | list):
            if len(dims) == 0:
                return dims
            # Convert to a tuple
            dims = tuple(dims)
            if all([isinstance(x, str) for x in dims]):
                return dims
            else:
                raise ValueError(f"dims must be a string or a tuple of strings, found {dims}")

        # Case 2: Index is a single comma delimited string. Convert to a tuple
        elif isinstance(dims, str):
            dims = dims.split(",")
            return tuple(dim.strip() for dim in dims)

        else:
            raise ValueError(f"dims must be a string or a tuple of strings, found {dims}")

    def _initialize_dim_vals(self):
        dims = self.dims
        dim_vals = self.dim_vals

        # The default should be an empty dictionary
        if dim_vals is None:
            return {}

        elif not isinstance(dim_vals, dict):
            raise ValueError(
                f"dim_vals of {self.base_name} should be a dictionary mapping dimensions to specific coord"
                f" values; found {type(dim_vals)}"
            )

        keys = list(dim_vals.keys())

        # Check all the keys are in the dims
        extra_dims = set(keys) - set(dims)
        if len(extra_dims) > 0:
            raise ValueError(
                f"All dim_vals of {self.base_name} must be associated with known dimensions: {self.dims}. "
                f"Found unknown dimensions: {extra_dims}."
            )
        return dim_vals

    def _initialize_latex_name(self):
        latex_name = self.latex_name if self.latex_name is not None else self.name
        latex_name = re.sub(GREEK_PATTERN, r"\\" + r"\g<0>", latex_name)
        return latex_name

    def _initialize_full_latex_name(self):
        idx_subscript = self._make_latex_subscript()
        idx_subscript_tokens = _tokenize_latex_subscript(idx_subscript)
        base_latex_name = self.latex_name
        int_extend = int(self.extend_subscript)

        if self.extend_subscript:
            tokens = [re.sub(r"[\{\}]", "", x.strip()) for x in base_latex_name.split("_")]
            base_tokens, subscript_tokens = tokens[:-int_extend], tokens[-int_extend:]

            if len(base_tokens) == 0:
                base_tokens, subscript_tokens = subscript_tokens, base_tokens
            base_latex_name = "_".join(base_tokens)

            subscript_tokens = subscript_tokens + idx_subscript_tokens

            if len(subscript_tokens) == 0:
                idx_subscript = ""
            else:
                idx_subscript = "_{" + ", ".join(subscript_tokens) + "}"

        if "_" in base_latex_name:
            base_latex_name = r"\text{" + base_latex_name + "}"

        latex_name = base_latex_name + idx_subscript

        return latex_name

    def _initialize_description(self):
        if self.description is not None:
            self.description = _pretty_print_dim_flags(self.description, self.dims, self.dim_vals)
            return self.description
        return f"{self._full_latex_name}, Positive = {self.positive}, Real = {self.real}"

    def __getitem__(self, item: str):
        return getattr(self, item)

    def __repr__(self):
        return self._full_latex_name

    def to_dict(self):
        return {
            "name": self.name,
            "dims": self.dims,
            "sympy": self.to_sympy(long_name=False),
            "latex_name": self._full_latex_name,
            "dim_vals": self.dim_vals,
            "description": self.description,
        }

    def to_sympy(self, long_name=False, use_latex_name=False):
        indices = [sp.Idx(dim) for dim in self.dims]
        if use_latex_name:
            base_name = self._full_latex_name if long_name else self.latex_name
        else:
            base_name = self.name if long_name else self.base_name
        if len(indices) == 0:
            return sp.Symbol(base_name, **self.assumptions)
        base = sp.IndexedBase(base_name, **self.assumptions)
        dim_val_subs = {sp.Idx(k): v for k, v in self.dim_vals.items()}

        return base[indices].subs(dim_val_subs)

    def update_dim_value(self, dim, value):
        if dim not in self.dims:
            raise ValueError(f"{dim} is not a valid dimension of {self.name}.")

        self.dim_vals[dim] = value
        self._full_latex_name = self._initialize_full_latex_name()
        self.description = self._initialize_description()

    def copy(self):
        return deepcopy(self)


@dataclass(order=True, frozen=False, repr=False)
class Variable(ModelObject):
    pass


@dataclass(order=True, frozen=False, repr=False)
class Parameter(ModelObject):
    pass


@dataclass(frozen=True)
class Equation:
    name: str
    equation: str
    eq_id: int | None = None

    def _set_eq_id(self, new_id):
        object.__setattr__(self, "eq_id", new_id)

    def __getitem__(self, item: str):
        return getattr(self, item)

    def __repr__(self):
        return self.name


# noinspection PyDataclass
@dataclass(frozen=True, kw_only=True)
class _SympyEquation(Equation):
    symbolic_eq: sp.Eq
    _eq: sp.Expr
    _fancy_eq: sp.Eq
    dims: tuple[str, ...]
    dim_vals: dict[str, str] = field(default_factory=dict)

    def __repr__(self):
        return self.name

    def _repr_latex_(self):
        s = latex(self.symbolic_eq, mode="plain")
        return f"$\\displaystyle {s}$"

    def copy(self):
        return deepcopy(self)

    def update_dim_value(self, dim, value):
        if dim not in self.dims:
            raise ValueError(f"{dim} is not a valid dimension of {self.name}.")
        sp_dim = sp.Idx(dim)

        object.__setattr__(self, "symbolic_eq", self.symbolic_eq.subs(sp_dim, value))
        object.__setattr__(self, "_eq", self._eq.subs(sp_dim, value))
        object.__setattr__(
            self, "name", _pretty_print_dim_flags(self.name, self.dims, {dim: value})
        )
        object.__setattr__(self, "dim_vals", {dim: value})

    def to_dict(self):
        return {
            "name": self.name,
            "string_eq": self.equation,
            "symbolic_eq": self.symbolic_eq,
            "fancy_eq": self._fancy_eq,
            "standard_eq": self._eq,
            "eq_id": self.eq_id,
        }


class Result:
    def __init__(
        self,
        name,
        variables,
        parameters,
        initial_values,
        fitted_values,
        success,
        meta=None,
    ):
        self.name = name
        self.variables = variables
        self.parameters = parameters
        self.initial_values = initial_values
        self.fitted_values = fitted_values
        self.success = success
        self.fit_time = datetime.now()
        self.meta = meta if meta is not None else {}

    def to_dict(self):
        all_names = self.variables + self.parameters
        return {
            "initial": dict(zip(all_names, self.initial_values)),
            "fitted": dict(zip(all_names, self.fitted_values)),
        }

    def to_frame(self):
        return pd.DataFrame(self.to_dict())

    def format_datetime(self):
        local_timezone = time.tzname[0]
        formatted_date = self.fit_time.strftime("%A, %B %d, %Y %I:%M %p")
        return " ".join([formatted_date, local_timezone])
