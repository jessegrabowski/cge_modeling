import numpy as np
import pytensor.xtensor.math as pxm
import pytest

from cge_modeling import CGEModel
from cge_modeling import model as cm
from cge_modeling.tools.latex import equation_to_latex
from tests.utilities.models import (
    build_model_1_in_context,
    build_model_2_in_context,
    build_model_3_in_context,
)


def test_model_1_prints_powers_value_form_demands_and_greek_names():
    model = build_model_1_in_context()
    equations = model._xtensor_equations

    assert equation_to_latex(equations["Final good production"], model) == (
        r"Y = A \cdot K_{d}^{\alpha} \cdot L_{d}^{1 - \alpha}"
    )
    assert equation_to_latex(equations["Final good production: demand for L_d"], model) == (
        r"w \cdot L_{d} = \left(1 - \alpha\right) Y \cdot P"
    )
    assert equation_to_latex(equations["Household income"], model) == (
        r"\Omega = w \cdot L_{s} + r \cdot K_{s}"
    )


def test_leontief_bundle_prints_the_sum_over_the_renamed_dim():
    model = build_model_2_in_context()
    equations = model._xtensor_equations

    assert equation_to_latex(equations["Production of intermediate goods bundle"], model) == (
        r"P_{\mathrm{VC}, j} \cdot \mathrm{VC}_{j} = \sum_{i} P_{i} \cdot X_{i, j}"
    )
    assert equation_to_latex(
        equations["Production of intermediate goods bundle: demand for X"], model
    ) == (r"P_{i} \cdot X_{i, j} = P_{i} \cdot \psi_{X, i, j} \cdot \mathrm{VC}_{j}")


def test_selection_by_label_prints_the_label_in_the_subscript():
    model = build_model_3_in_context()

    assert equation_to_latex(model._xtensor_equations["Numeraire"], model) == (
        r"P_{\text{agriculture}} = \bar{P}"
    )


def test_selection_on_a_leading_dim_keeps_the_dims_before_it():
    with CGEModel(coords={"sector": ["farm", "mill"], "j": ["farm", "mill"]}) as model:
        X = cm.Variable("X", dims=("sector", "j"))
        Y = cm.Variable("Y", dims="sector")
        equation = cm.Equation("Selection", X.isel(j=0), Y)

    assert equation_to_latex(equation, model) == (
        r"X_{\mathrm{sector}, \text{farm}} = Y_{\mathrm{sector}}"
    )


@pytest.mark.parametrize(
    "build_lhs, expected",
    [
        pytest.param(lambda a, b, c: (a + b) * c, r"\left(a + b\right) c", id="sum_times"),
        pytest.param(
            lambda a, b, c: a - (b - c), r"a - \left(b - c\right)", id="nested_difference"
        ),
        pytest.param(
            lambda a, b, c: (a / b) ** c, r"\left(\frac{a}{b}\right)^{c}", id="fraction_base"
        ),
        pytest.param(lambda a, b, c: a ** (b + c), r"a^{b + c}", id="sum_exponent"),
        pytest.param(lambda a, b, c: a + (-b), r"a - b", id="add_negation"),
        pytest.param(lambda a, b, c: -(a + b), r"-\left(a + b\right)", id="negated_sum"),
        pytest.param(lambda a, b, c: 0 + a, r"a", id="zero_addend"),
        pytest.param(
            lambda a, b, c: pxm.exp(a) + pxm.log(b) * pxm.sqrt(c),
            r"\exp\left(a\right) + \log\left(b\right) \sqrt{c}",
            id="functions",
        ),
    ],
)
def test_operators_print_with_the_parentheses_their_precedence_requires(build_lhs, expected):
    with CGEModel() as model:
        a, b, c = cm.Variable("a"), cm.Variable("b"), cm.Variable("c")
        equation = cm.Equation("Expression", build_lhs(a, b, c), 1.0)

    assert equation_to_latex(equation, model) == f"{expected} = 1"


@pytest.mark.parametrize(
    "build_lhs, expected",
    [
        pytest.param(lambda a, b, c: a * 2, r"a \cdot 2", id="number_factor"),
        pytest.param(lambda a, b, c: 0.5 * a, r"0.5 \cdot a", id="leading_number"),
        pytest.param(
            lambda a, b, c: (a + b) * 2,
            r"\left(a + b\right) \cdot 2",
            id="number_after_parentheses",
        ),
        pytest.param(lambda a, b, c: 1e-10 * a, r"10^{-10} \cdot a", id="small_power_of_ten"),
        pytest.param(
            lambda a, b, c: 2.5e6 * a, r"2.5 \times 10^{6} \cdot a", id="large_with_mantissa"
        ),
        pytest.param(lambda a, b, c: a + (-3e-5), r"a - 3 \times 10^{-5}", id="negative_small"),
        pytest.param(
            lambda a, b, c: a + 0.001 + 9999, r"a + 0.001 + 9999", id="thresholds_stay_positional"
        ),
        pytest.param(lambda a, b, c: a + np.inf, r"a + \infty", id="infinity"),
        pytest.param(lambda a, b, c: a + (-np.inf), r"a - \infty", id="negative_infinity"),
        pytest.param(lambda a, b, c: a * np.nan, r"a \cdot \mathrm{NaN}", id="nan"),
    ],
)
def test_constants_print_positionally_or_in_scientific_notation(build_lhs, expected):
    with CGEModel() as model:
        a, b, c = cm.Variable("a"), cm.Variable("b"), cm.Variable("c")
        equation = cm.Equation("Constant", build_lhs(a, b, c), 1.0)

    assert equation_to_latex(equation, model) == f"{expected} = 1"


@pytest.mark.parametrize(
    "function, expected",
    [
        pytest.param(pxm.abs, r"\left|x\right|", id="abs"),
        pytest.param(pxm.sign, r"\operatorname{sign}\left(x\right)", id="sign"),
        pytest.param(pxm.ceil, r"\left\lceil x \right\rceil", id="ceil"),
        pytest.param(pxm.floor, r"\left\lfloor x \right\rfloor", id="floor"),
        pytest.param(pxm.trunc, r"\operatorname{trunc}\left(x\right)", id="trunc"),
        pytest.param(pxm.round, r"\operatorname{round}\left(x\right)", id="round"),
        pytest.param(pxm.sqr, r"x^{2}", id="sqr"),
        pytest.param(pxm.reciprocal, r"\frac{1}{x}", id="reciprocal"),
        pytest.param(pxm.exp2, r"2^{x}", id="exp2"),
        pytest.param(pxm.expm1, r"\exp\left(x\right) - 1", id="expm1"),
        pytest.param(pxm.log2, r"\log_{2}\left(x\right)", id="log2"),
        pytest.param(pxm.log10, r"\log_{10}\left(x\right)", id="log10"),
        pytest.param(pxm.log1p, r"\log\left(1 + x\right)", id="log1p"),
        pytest.param(pxm.log1mexp, r"\log\left(1 - \exp\left(x\right)\right)", id="log1mexp"),
        pytest.param(pxm.sin, r"\sin\left(x\right)", id="sin"),
        pytest.param(pxm.cos, r"\cos\left(x\right)", id="cos"),
        pytest.param(pxm.tan, r"\tan\left(x\right)", id="tan"),
        pytest.param(pxm.arcsin, r"\arcsin\left(x\right)", id="arcsin"),
        pytest.param(pxm.arccos, r"\arccos\left(x\right)", id="arccos"),
        pytest.param(pxm.arctan, r"\arctan\left(x\right)", id="arctan"),
        pytest.param(pxm.sinh, r"\sinh\left(x\right)", id="sinh"),
        pytest.param(pxm.cosh, r"\cosh\left(x\right)", id="cosh"),
        pytest.param(pxm.tanh, r"\tanh\left(x\right)", id="tanh"),
        pytest.param(pxm.arcsinh, r"\operatorname{arsinh}\left(x\right)", id="arcsinh"),
        pytest.param(pxm.arccosh, r"\operatorname{arcosh}\left(x\right)", id="arccosh"),
        pytest.param(pxm.arctanh, r"\operatorname{artanh}\left(x\right)", id="arctanh"),
        pytest.param(pxm.erf, r"\operatorname{erf}\left(x\right)", id="erf"),
        pytest.param(pxm.erfc, r"\operatorname{erfc}\left(x\right)", id="erfc"),
        pytest.param(pxm.erfcx, r"\operatorname{erfcx}\left(x\right)", id="erfcx"),
        pytest.param(pxm.erfinv, r"\operatorname{erf}^{-1}\left(x\right)", id="erfinv"),
        pytest.param(pxm.erfcinv, r"\operatorname{erfc}^{-1}\left(x\right)", id="erfcinv"),
        pytest.param(pxm.gamma, r"\Gamma\left(x\right)", id="gamma"),
        pytest.param(pxm.gammaln, r"\log \Gamma\left(x\right)", id="gammaln"),
        pytest.param(pxm.psi, r"\psi\left(x\right)", id="psi"),
        pytest.param(pxm.tri_gamma, r"\psi_{1}\left(x\right)", id="tri_gamma"),
        pytest.param(pxm.i0, r"I_{0}\left(x\right)", id="i0"),
        pytest.param(pxm.i1, r"I_{1}\left(x\right)", id="i1"),
        pytest.param(pxm.j0, r"J_{0}\left(x\right)", id="j0"),
        pytest.param(pxm.j1, r"J_{1}\left(x\right)", id="j1"),
        pytest.param(pxm.sigmoid, r"\frac{1}{1 + \exp\left(-x\right)}", id="sigmoid"),
        pytest.param(pxm.softplus, r"\log\left(1 + \exp\left(x\right)\right)", id="softplus"),
        pytest.param(pxm.ndtri_exp, r"\Phi^{-1}\left(\exp\left(x\right)\right)", id="ndtri_exp"),
        pytest.param(pxm.isnan, r"\operatorname{isnan}\left(x\right)", id="isnan"),
        pytest.param(pxm.isinf, r"\operatorname{isinf}\left(x\right)", id="isinf"),
        pytest.param(pxm.real, r"\operatorname{Re}\left(x\right)", id="real"),
        pytest.param(pxm.imag, r"\operatorname{Im}\left(x\right)", id="imag"),
        pytest.param(pxm.conj, r"\overline{x}", id="conj"),
        pytest.param(pxm.identity, r"x", id="identity"),
        pytest.param(lambda x: pxm.cast(x, "float32"), r"x", id="cast"),
    ],
)
def test_unary_scalar_ops_print_in_standard_notation(function, expected):
    with CGEModel() as model:
        x = cm.Variable("x")
        equation = cm.Equation("Unary", function(x), 0.0)

    assert equation_to_latex(equation, model) == f"{expected} = 0"


@pytest.mark.parametrize(
    "function, expected",
    [
        pytest.param(pxm.maximum, r"\max\left(x, y\right)", id="maximum"),
        pytest.param(pxm.minimum, r"\min\left(x, y\right)", id="minimum"),
        pytest.param(pxm.arctan2, r"\operatorname{atan2}\left(x, y\right)", id="arctan2"),
        pytest.param(pxm.mod, r"x \bmod y", id="mod"),
        pytest.param(
            lambda x, y: pxm.mod(x + y, x * y),
            r"\left(x + y\right) \bmod \left(x \cdot y\right)",
            id="mod_parenthesized",
        ),
        pytest.param(pxm.int_div, r"\left\lfloor \frac{x}{y} \right\rfloor", id="int_div"),
        pytest.param(pxm.eq, r"x = y", id="eq"),
        pytest.param(pxm.neq, r"x \neq y", id="neq"),
        pytest.param(pxm.lt, r"x < y", id="lt"),
        pytest.param(pxm.le, r"x \leq y", id="le"),
        pytest.param(pxm.gt, r"x > y", id="gt"),
        pytest.param(pxm.ge, r"x \geq y", id="ge"),
        pytest.param(lambda x, y: (x > 0) & (y < 1), r"x > 0 \land y < 1", id="and"),
        pytest.param(lambda x, y: (x > 0) | (y < 1), r"x > 0 \lor y < 1", id="or"),
        pytest.param(lambda x, y: (x > 0) ^ (y < 1), r"x > 0 \oplus y < 1", id="xor"),
        pytest.param(
            lambda x, y: ((x > 0) & (y < 1)) | (x > y),
            r"\left(x > 0 \land y < 1\right) \lor x > y",
            id="nested_logic",
        ),
        pytest.param(lambda x, y: pxm.invert(x > y), r"\lnot \left(x > y\right)", id="invert"),
        pytest.param(
            lambda x, y: (x > 0) + pxm.invert(y > 0),
            r"\left(x > 0\right) + \lnot \left(y > 0\right)",
            id="add_invert_is_not_subtraction",
        ),
        pytest.param(
            lambda x, y: (x > 0) + (y > 0),
            r"\left(x > 0\right) + \left(y > 0\right)",
            id="add_comparisons",
        ),
        pytest.param(pxm.gammainc, r"P\left(x, y\right)", id="gammainc"),
        pytest.param(pxm.gammaincc, r"Q\left(x, y\right)", id="gammaincc"),
        pytest.param(pxm.gammaincinv, r"P^{-1}\left(x, y\right)", id="gammaincinv"),
        pytest.param(pxm.gammainccinv, r"Q^{-1}\left(x, y\right)", id="gammainccinv"),
        pytest.param(pxm.owens_t, r"T\left(x, y\right)", id="owens_t"),
        pytest.param(pxm.jv, r"J_{x}\left(y\right)", id="jv"),
        pytest.param(
            pxm.ive, r"I_{x}\left(y\right) \cdot \exp\left(-\left|y\right|\right)", id="ive"
        ),
        pytest.param(pxm.kve, r"K_{x}\left(y\right) \cdot \exp\left(y\right)", id="kve"),
        pytest.param(lambda x, y: pxm.polygamma(2, y), r"\psi^{(2)}\left(y\right)", id="polygamma"),
        pytest.param(pxm.complex, r"x + i \cdot y", id="complex"),
        pytest.param(pxm.second, r"y", id="second"),
    ],
)
def test_binary_scalar_ops_print_in_standard_notation(function, expected):
    with CGEModel() as model:
        x, y = cm.Variable("x"), cm.Variable("y")
        equation = cm.Equation("Binary", function(x, y), 0.0)

    assert equation_to_latex(equation, model) == f"{expected} = 0"


def test_factors_are_separated_by_a_dot_unless_one_is_parenthesized():
    with CGEModel() as model:
        tm, Er, M, x = cm.Parameter("tm"), cm.Variable("Er"), cm.Variable("M"), cm.Variable("x")
        equation = cm.Equation("Tariff", tm * Er * M, Er * (1 + tm) * (M + x) * x)

    assert equation_to_latex(equation, model) == (
        r"\mathrm{tm} \cdot \mathrm{Er} \cdot M"
        r" = \mathrm{Er} \left(1 + \mathrm{tm}\right) \left(M + x\right) x"
    )


def test_a_name_with_a_superscript_is_parenthesized_as_a_power_base():
    with CGEModel() as model:
        x_star = cm.Variable("x_star")
        y = cm.Variable("y", latex_name="y^j")
        equation = cm.Equation("Powers", x_star**2, y**2)

    assert equation_to_latex(equation, model) == (
        r"\left(x^{\star}\right)^{2} = \left(y^j\right)^{2}"
    )


@pytest.mark.parametrize(
    "reduce, expected",
    [
        (lambda X: X.prod("sector"), r"\prod_{\mathrm{sector}} X_{\mathrm{sector}, j}"),
        (lambda X: X.max("sector"), r"\max_{\mathrm{sector}} X_{\mathrm{sector}, j}"),
        (lambda X: X.min("sector"), r"\min_{\mathrm{sector}} X_{\mathrm{sector}, j}"),
    ],
    ids=["prod", "max", "min"],
)
def test_reductions_print_their_symbol_over_an_upright_multiletter_dim(reduce, expected):
    with CGEModel(coords={"sector": ["farm", "mill"], "j": ["farm", "mill"]}) as model:
        X = cm.Variable("X", dims=("sector", "j"))
        Z = cm.Variable("Z", dims="j")
        equation = cm.Equation("Reduction", reduce(X), Z)

    assert equation_to_latex(equation, model) == f"{expected} = Z_{{j}}"


def test_nested_reductions_print_without_parentheses_between_them():
    with CGEModel(coords={"i": ["a", "b"], "j": ["a", "b"], "k": ["a", "b"]}) as model:
        X = cm.Variable("X", dims=("i", "j", "k"))
        Y = cm.Variable("Y", dims="k")
        equation = cm.Equation("Nested", X.sum("i").prod("j"), Y * X.sum(["i", "j"]))

    assert equation_to_latex(equation, model) == (
        r"\prod_{j} \sum_{i} X_{i, j, k} = Y_{k} \left(\sum_{i, j} X_{i, j, k}\right)"
    )


def test_swapping_dim_names_leaves_the_unswapped_operand_alone():
    with CGEModel(coords={"i": ["a", "b"], "j": ["a", "b"]}) as model:
        X = cm.Variable("X", dims=("i", "j"))
        equation = cm.Equation("Swap", X.rename(i="j", j="i"), X)

    assert equation_to_latex(equation, model) == r"X_{j, i} = X_{i, j}"


def test_rename_onto_a_summed_dim_primes_the_bound_index():
    with CGEModel(coords={"i": ["a", "b"], "j": ["a", "b"]}) as model:
        X = cm.Variable("X", dims=("i", "j"))
        Y = cm.Variable("Y", dims="i")
        equation = cm.Equation("Capture", X.sum("i").rename(j="i"), Y)

    assert equation_to_latex(equation, model) == r"\sum_{i'} X_{i', i} = Y_{i}"


def test_unsupported_operations_name_the_equation_and_the_operation():
    with CGEModel(coords={"i": ["a", "b", "c"]}) as model:
        P = cm.Variable("P", dims="i")
        x = cm.Variable("x")
        selection = cm.Equation("Selection", P.isel(i=[0, 1]).sum("i"), 1.0)
        beta = cm.Equation("Beta", pxm.betainc(x, x, x), 0.0)

    with pytest.raises(NotImplementedError, match=r"'Selection'.*indexing 'i'"):
        equation_to_latex(selection, model)
    with pytest.raises(
        NotImplementedError, match=r"'Beta'.*conversion for the given `Op`: betainc"
    ):
        equation_to_latex(beta, model)
