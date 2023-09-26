import sympy as sp

from cge_modeling.utils import brackets_to_snake_case


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


def enumerate_indexbase(exprs, indices, index_dicts, expand_using="index"):
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
