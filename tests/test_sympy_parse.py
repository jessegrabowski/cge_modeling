from string import ascii_lowercase

import numpy as np
import pytest
import sympy as sp

from cge_modeling.tools.sympy_tools import substitute_reduce_ops


def substitute_expression(expr, values):
    indexed_vars = []
    for var in sp.preorder_traversal(expr):
        if isinstance(var, sp.Indexed):
            indexed_vars.append(var)
    sub_dict = dict(zip(indexed_vars, values))
    return expr.subs(sub_dict)


@pytest.mark.parametrize("op", [sp.Sum, sp.Product])
def test_substitute_reduce_ops(op):
    operator = "+" if op == sp.Sum else "*"
    x = sp.IndexedBase("x")
    i = sp.Idx("i")
    expr = op(x[i], (i, 0, 9))
    expr_subbed = substitute_reduce_ops(expr, coords={"i": list(ascii_lowercase[:10])})

    test_values = np.random.normal(size=10)
    expr_subbed = substitute_expression(expr_subbed, test_values)

    index_dict = {i: sp.Idx(i) for i in ascii_lowercase[:10]}
    expected = sp.parse_expr(
        operator.join(f"x[{i}]" for i in ascii_lowercase[:10]),
        transformations="all",
        local_dict={"x": x} | index_dict,
    )
    expected = substitute_expression(expected, test_values)

    assert expr_subbed == expected


def test_double_sum():
    x = sp.IndexedBase("x")
    i = sp.Idx("i")
    j = sp.Idx("j")
    expr = sp.Sum(sp.Sum(x[i, j], (i, 0, 9)), (j, 0, 3))
    expr_subbed = substitute_reduce_ops(
        expr, coords={"i": list(ascii_lowercase[:10]), "j": list(ascii_lowercase[:4])}
    )

    test_values = np.random.normal(size=(10, 4))
    expr_subbed = substitute_expression(expr_subbed, test_values)

    index_dict = {i: sp.Idx(i) for i in ascii_lowercase[:10]}
    expected = sp.parse_expr(
        "+".join(f"x[{i}]" for i in ascii_lowercase[:10]),
        transformations="all",
        local_dict={"x": x} | index_dict,
    )
    expected = substitute_expression(expected, test_values)

    assert expr_subbed == expected
