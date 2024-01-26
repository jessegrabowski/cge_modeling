import numpy as np
import pytest
import sympy as sp

from cge_modeling.base.primitives import Variable
from cge_modeling.tools.numba_tools import _generate_numba_signature, numba_lambdify


def test_numba_lambdify_single_output():
    a, b, c = sp.symbols("a b c")
    f = a * b + c
    f_func = numba_lambdify(inputs=[a, b, c], outputs=[f])

    assert f_func(1, 2, 3) == 5


@pytest.mark.parametrize("stack", [True, False], ids=["stack", "no_stack"])
def test_numba_lambdify_multi_output(stack):
    a, b, c = sp.symbols("a b c")
    f1 = a * b + c
    f2 = a + b + c

    f_func = numba_lambdify(inputs=[a, b, c], outputs=[f1, f2], stack_outputs=stack)

    outputs = f_func(1, 2, 3)
    if not stack:
        assert outputs[0] == 5
        assert outputs[1] == 6
    else:
        np.testing.assert_allclose(outputs, np.array([5, 6]))


def test_numba_lambdify_matrix_output():
    a, b, c = sp.symbols("a b c")
    f1 = sp.Matrix([a * b + c, a + b + c])

    f_func = numba_lambdify(inputs=[a, b, c], outputs=[f1])

    outputs = f_func(1, 2, 3)
    np.testing.assert_allclose(outputs, np.array([5, 6]).reshape((2, 1)))


signature_parameterizations = [
    [True, True, True, "float64[:, :](float64, float64, float64[:, :])"],
    [True, True, False, "float64[:, :](float64, float64, float64)"],
    [True, False, True, "float64[:](float64, float64, float64[:, :])"],
    [True, False, False, "float64[:](float64, float64, float64)"],
    [False, True, True, "float64[:, :](float64, float64, float64[:, :])"],
    [False, True, False, "float64[:, :](float64, float64, float64)"],
    [False, False, True, "float64(float64, float64, float64[:, :])"],
]


@pytest.mark.parametrize("stack, matrix_out, matrix_in, expected", signature_parameterizations)
def test_generate_numba_signature(stack, matrix_out, matrix_in, expected):
    variables = [Variable(name, dims=None) for name in ["x", "y"]]
    z = Variable("z", dims=("i", "j") if matrix_in else None)

    if stack and not matrix_out:
        output = [sp.Symbol("Y"), sp.Symbol("Z")]
    else:
        output = sp.MatrixSymbol("Y", 3, 3) if matrix_out else sp.Symbol("Y")

    signature = _generate_numba_signature(
        inputs=variables + [z], outputs=[output], stack_outputs=stack
    )
    assert signature == expected


def test_generate_numba_signature_multiple_output():
    variables = [Variable(name, dims=None) for name in ["x", "y", "z"]]
    outputs = [sp.MatrixSymbol(name, 3, 3) for name in ["Y", "Z"]]

    signature = _generate_numba_signature(inputs=variables, outputs=outputs)
    assert signature == "Tuple((float64[:, :], float64[:, :]))(float64, float64, float64)"
