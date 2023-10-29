from functools import reduce
from itertools import product

from typing import Sequence, cast, Any, Optional, Literal

import sympy as sp

from cge_modeling.base.primitives import ModelObject
from cge_modeling.utils import brackets_to_snake_case


def _prod(args: Sequence) -> Any:
    """
    Multiply a list of numbers.

    Parameters
    ----------
    args: list
        List of objects to be multiplied

    Returns
    -------
    object
        The product of the objects in args
    """

    return reduce(lambda x, y: x * y, args)


# Mapping between sympy map-reduce operations and python map-reduce operations
OP_TO_PY = {sp.Sum: sum,
            sp.Product: _prod}

# List of sympy map-reduce operations
OPS_TO_FIND = list(OP_TO_PY.keys())


def expand_indices(expressions, idx_symbols, idx_values):
    if not idx_symbols:
        return expressions

    index = idx_symbols[0]
    value_list = idx_values[0]
    remaining_indices = idx_symbols[1:]
    remaining_values = idx_values[1:]

    new_expressions = []

    for expr in expressions:
        if expr.has(index):
            for value in value_list:
                substitution_dict = {index: value}
                new_expr = expr.subs(substitution_dict)
                if new_expr not in new_expressions:
                    new_expressions.append(new_expr)
        else:
            new_expressions.append(expr)

    return expand_indices(new_expressions, remaining_indices, remaining_values)


def make_symbol(name, index=False, assumptions=None):
    if assumptions is None:
        assumptions = {}
    if index:
        return sp.IndexedBase(name, **assumptions)
    else:
        return sp.Symbol(name, **assumptions)


def sub_all_eqs(equations, sub_dict):
    return [eq.subs(sub_dict) for eq in equations]


def make_indexed_name(obj: ModelObject, delimiter='_') -> str:
    """
    Construct an object name with indexing information from a ModelObject.

    Parameters
    ----------
    obj: ModelObject
        One of Variable, Parameter, or Equation.

    delimiter: str, optional
        The delimiter to use between the object name and the index values. Default is '_'.

    Returns
    -------
    name: str
        The name of the object with the index values appended, separated by the delimiter.
    """

    return '_'.join([obj.name] + list(obj.dim_vals.values()))


def indexed_variable_to_sympy_symbol(obj: ModelObject) -> sp.Symbol:
    """
    Convert a ModelObject to a sympy Symbol whose name is the name of the ModelObject with the index values appended.

    Assumptions held by the ModelObject are passed to the sympy Symbol.

    Parameters
    ----------
    obj: ModelObject
        A ModelObject, one of Variable, Parameter, or Equation

    Returns
    -------
    sympy.Symbol
        A sympy Symbol corresponding to the ModelObject's sympy representation, without an IndexBase. Indexing
        information is instead appended to the name of the Symbol.
    """

    name = make_indexed_name(obj)
    return sp.Symbol(name, **obj.assumptions)


def indexed_variables_to_sub_dict(obj_list: list[ModelObject]) -> dict[sp.Symbol, sp.Symbol]:
    """
    Construct a mapping between ModelObject sympy representations using IndexedBase and simple Sympy symbols.

    The IndexedBase class is used by sympy to represent indexed variables, but it is not convenient for use in
    lambdify. This function constructs a mapping between the IndexedBase representation of each ModelObject created by
    the .to_sympy() method, and a simple sympy Symbol. Assumptions held by the ModelObject are passed to the sympy
    Symbol.

    Parameters
    ----------
    obj_list: list of ModelObject
        A list of ModelObjects; one of Variable, Parameter, or Equation.

    Returns
    -------
    sub_dict: dict of sympy.Symbol-sympy.Symbol pairs
        A dictionary mapping the sympy representation of each ModelObject to a simple sympy Symbol.
    """

    indexbase_symbols = [x.to_sympy() for x in obj_list]
    no_index_symbols = [indexed_variable_to_sympy_symbol(x) for x in obj_list]
    return dict(zip(indexbase_symbols, no_index_symbols))


def enumerate_indexbase(exprs, indices, index_dicts, expand_using="index"):
    """
    Expand the index base of a list of expressions.

    Parameters
    ----------
    exprs: list
        List of sympy expressions
    indices: list
        List of sympy indices
    index_dicts: list
        List of dictionaries mapping index symbols to index values
    expand_using: str
        Either "index" or "label". If "index", the index values are used to expand the expressions. If "label", the
        index labels are used to expand the expressions.

    Returns
    -------
    list
        List of expanded sympy expressions

    Notes
    -----
    The length of the returned list depends on the length of exprs and the number of labels in each index. If there are
    n expressions, each with a single index, and m labels in the index, the returned list will have length n * m. In
    the case of multiple indices, the length of the returned list will be the product of the number of labels in each
    index.

    Examples
    --------
    >>> from sympy import symbols
    >>> from cge_modeling.sympy_tools import enumerate_indexbase
    >>> x = sp.IndexedBase('x')
    >>> idx = symbols('i')
    >>> coords = {'i':['a', 'b']}
    >>> variable = [x[idx]]
    >>> indices = [idx]
    >>> enumerate_indexbase(exprs, indices, index_dicts)
    >>> Out: ['x_1', 'x_2']
    """
    idx_values = [list(d.keys()) for d in index_dicts]
    idx_labels = [list(d.values()) for d in index_dicts]

    if expand_using == "index":
        exprs_expanded = expand_indices(exprs, indices, idx_values)
    else:
        exprs_expanded = expand_indices(exprs, indices, idx_labels)

    return exprs_expanded


def indexed_var_to_symbol(index_var):
    return sp.Symbol(brackets_to_snake_case(index_var.name), **index_var.assumptions0)


def make_indexbase_sub_dict(exprs_expanded):
    exprs_as_symbols = [indexed_var_to_symbol(x) for x in exprs_expanded]
    sub_dict = dict(zip(exprs_expanded, exprs_as_symbols))

    return sub_dict


def info_to_symbols(var_info, assumptions):
    names, index_symbols = (list(t) for t in zip(*var_info))
    has_index = [len(idx) > 0 for idx in index_symbols]

    base_vars = [make_symbol(name, has_idx, assumptions) for name, has_idx in zip(names, has_index)]

    def inject_index(x, has_idx, idx):
        if not has_idx:
            return x
        return x[idx]

    variables = [
        inject_index(x, has_idx, idx)
        for x, has_idx, idx in zip(base_vars, has_index, index_symbols)
    ]

    return variables


def symbol(name, *sectors, assumptions=None):
    if assumptions is None:
        assumptions = {}

    if sectors == ():
        return sp.Symbol(name, **assumptions)

    suffix = "_" + "_".join(sectors)
    return sp.Symbol(f"{name}{suffix}", **assumptions)


def symbols(name, value, sectors, assumptions=None):
    return {symbol(name, sector, assumptions=assumptions): value for sector in sectors}


def dict_info_to_symbols(dict_info, assumptions):
    tuple_info = [(d["name"], d.get("index", ())) for d in dict_info]
    names, _ = (list(t) for t in zip(*tuple_info))
    symbols = info_to_symbols(tuple_info, assumptions)
    global_updates = dict(zip(names, symbols))
    return symbols, global_updates


def remove_string_keys(d):
    d_copy = d.copy()
    for k in d:
        if isinstance(k, str):
            del d_copy[k]
    return d_copy


def find_equation_dims(eq: sp.Expr, index_symbols: list[sp.Idx]) -> tuple[str]:
    found_ids = [idx for idx in index_symbols if eq.has(idx)]
    sorted_ids = sorted(found_ids, key=lambda x: index_symbols.index(x))
    return cast(tuple[str], tuple([x.name for x in sorted_ids]))


def substitute_reduce_ops(eq: sp.Expr, coords: dict[str, list[str]]) -> sp.Expr:
    """
    Substitute a sum or product operation with a sum or product of expanded expressions.

    Parameters
    ----------
    eq: sympy.Expr
        The expression to be expanded
    coords: dict of str-list pairs
        A dictionary mapping index names to lists of labels

    Returns
    -------
    sympy.Expr
        The expanded expression
    """

    for op_to_find in OPS_TO_FIND:
        found_ops = list(eq.find(op_to_find))
        for op in found_ops:
            expr, index_info = op.args
            idx, start, stop = index_info

            assert idx.name in coords.keys()
            assert len(coords[idx.name]) == stop - start + 1

            expanded_expr = OP_TO_PY[op_to_find]([expr.subs({idx: val}) for val in coords[idx.name]])
            sub_dict = {op: expanded_expr}
            eq = eq.subs(sub_dict)
    return eq


def _validate_dims(obj: ModelObject, dims: list[str], on_unused_dim='raise') -> list[str]:
    """
    Validate that the provided dims are associated with the provided object. If not, either raise an error or ignore
    the unused dims, based on the value of on_unused_dim.

    Parameters
    ----------
    obj: ModelObject
        A model object with a dims attribute, one of Variable, Parameter, or Equation
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

    unknown_dims = set(dims) - set(obj.dims)
    if len(unknown_dims) > 0:
        unk_str = ', '.join(list(unknown_dims))
        if on_unused_dim == 'raise':
            raise ValueError(f'Dimension expansion was requested for {obj.name} on the following dims '
                             f'which are not associted with {obj.name}: {unk_str}')
        elif on_unused_dim == 'ignore':
            return list(set(dims) - unknown_dims)

    return dims


def expand_obj_by_indices(obj: ModelObject,
                          coords: dict[str:list[str]],
                          dims: Optional[list[str]] = None,
                          on_unused_dim: Literal['raise', 'ignore'] = 'raise') -> list[ModelObject]:
    """
    Expand a model object by creating a new object for each label associated to the requested dimensions.

    Parameters
    ----------
    obj: ModelObject
        A model object with a dims attribute, one of Variable, Parameter, or Equation
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
        A list of model objects with the same name, description, and assumptions as the original object, but with the
        dimension values updated to match the labels in coords.
    """
    out = []

    dims = obj.dims if dims is None else dims
    dims = _validate_dims(obj, dims, on_unused_dim)

    # Take the cartesian product of dimensions to be expanded to form all possible combinations of labels
    labels = [coords[dim] for dim in dims]
    label_prod = product(*labels)

    for labels in label_prod:
        new_obj = obj.copy()
        for dim, label in zip(dims, labels):
            new_obj.update_dim_value(dim, label)
        out.append(new_obj)
    return out
