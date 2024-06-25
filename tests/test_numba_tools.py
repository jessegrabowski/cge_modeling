import numpy as np
import sympy as sp

from cge_modeling.tools.numba_tools import numba_lambdify


def test_numba_lambdify_single_output():
    a, b, c = sp.symbols("a b c")
    f = a * b + c
    f_func = numba_lambdify(exog_vars=[a, b, c], expr=f)

    assert f_func([1, 2, 3]) == 5


def test_numba_lambdify_multi_output():
    a, b, c = sp.symbols("a b c")
    f1 = a * b + c
    f2 = a + b + c

    f_func = numba_lambdify(exog_vars=[a, b, c], expr=[f1, f2], ravel_outputs=True)

    outputs = f_func([1, 2, 3])
    np.testing.assert_allclose(outputs, [np.array([5]), np.array([6])])


def test_numba_lambdify_matrix_output():
    a, b, c = sp.symbols("a b c")
    f1 = sp.Matrix([a * b + c, a + b + c])

    f_func = numba_lambdify(exog_vars=[a, b, c], expr=f1)

    outputs = f_func([1, 2, 3])
    np.testing.assert_allclose(outputs, np.array([5, 6]).reshape((2, 1)))
