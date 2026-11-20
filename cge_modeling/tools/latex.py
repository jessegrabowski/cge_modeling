import re

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import singledispatch
from string import Template
from typing import TYPE_CHECKING

import numpy as np
import pytensor.scalar.math as psm

from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.traversal import toposort, walk
from pytensor.scalar import basic as ps
from pytensor.tensor.type_other import MakeSlice, SliceType
from pytensor.xtensor.basic import Rename, XTensorFromTensor
from pytensor.xtensor.indexing import Index
from pytensor.xtensor.reduction import XReduce
from pytensor.xtensor.shape import Transpose
from pytensor.xtensor.type import XTensorConstant
from pytensor.xtensor.vectorization import XElemwise

from cge_modeling.base.context import XTensorEquation
from cge_modeling.base.primitives import upright_multiletter_name

if TYPE_CHECKING:
    from cge_modeling.base.cge import CGEModel

LOGICAL = 3
COMPARISON = 5
ADDITIVE = 10
REDUCTION = 12
NEGATION = 15
MULTIPLICATIVE = 20
POWER = 30
FRACTION = 35
ATOM = 40

SCIENTIFIC_ABOVE = 1e4
SCIENTIFIC_BELOW = 1e-3

FUNCTION_NAMES = {
    ps.Exp: r"\exp",
    ps.Log: r"\log",
    ps.Log2: r"\log_{2}",
    ps.Log10: r"\log_{10}",
    ps.Sin: r"\sin",
    ps.Cos: r"\cos",
    ps.Tan: r"\tan",
    ps.ArcSin: r"\arcsin",
    ps.ArcCos: r"\arccos",
    ps.ArcTan: r"\arctan",
    ps.Sinh: r"\sinh",
    ps.Cosh: r"\cosh",
    ps.Tanh: r"\tanh",
    ps.ArcSinh: r"\operatorname{arsinh}",
    ps.ArcCosh: r"\operatorname{arcosh}",
    ps.ArcTanh: r"\operatorname{artanh}",
    ps.Sign: r"\operatorname{sign}",
    ps.Trunc: r"\operatorname{trunc}",
    ps.RoundHalfToEven: r"\operatorname{round}",
    ps.RoundHalfAwayFromZero: r"\operatorname{round}",
    ps.Real: r"\operatorname{Re}",
    ps.Imag: r"\operatorname{Im}",
    ps.IsNan: r"\operatorname{isnan}",
    ps.IsInf: r"\operatorname{isinf}",
    psm.Erf: r"\operatorname{erf}",
    psm.Erfc: r"\operatorname{erfc}",
    psm.Erfcx: r"\operatorname{erfcx}",
    psm.Erfinv: r"\operatorname{erf}^{-1}",
    psm.Erfcinv: r"\operatorname{erfc}^{-1}",
    psm.Gamma: r"\Gamma",
    psm.GammaLn: r"\log \Gamma",
    psm.Psi: r"\psi",
    psm.TriGamma: r"\psi_{1}",
    psm.I0: r"I_{0}",
    psm.I1: r"I_{1}",
    psm.J0: r"J_{0}",
    psm.J1: r"J_{1}",
    ps.Maximum: r"\max",
    ps.Minimum: r"\min",
    ps.ScalarMaximum: r"\max",
    ps.ScalarMinimum: r"\min",
    ps.ArcTan2: r"\operatorname{atan2}",
    psm.GammaInc: r"P",
    psm.GammaIncC: r"Q",
    psm.GammaIncInv: r"P^{-1}",
    psm.GammaIncCInv: r"Q^{-1}",
    psm.Owens_t: r"T",
}

TEMPLATES = {
    psm.Jv: (Template(r"J_{${x}}\left(${y}\right)"), ATOM),
    psm.Ive: (
        Template(r"I_{${x}}\left(${y}\right) \cdot \exp\left(-\left|${y}\right|\right)"),
        MULTIPLICATIVE,
    ),
    psm.Kve: (Template(r"K_{${x}}\left(${y}\right) \cdot \exp\left(${y}\right)"), MULTIPLICATIVE),
    psm.PolyGamma: (Template(r"\psi^{(${x})}\left(${y}\right)"), ATOM),
    ps.IntDiv: (Template(r"\left\lfloor \frac{${x}}{${y}} \right\rfloor"), ATOM),
    ps.Abs: (Template(r"\left|${x}\right|"), ATOM),
    ps.Ceil: (Template(r"\left\lceil ${x} \right\rceil"), ATOM),
    ps.Floor: (Template(r"\left\lfloor ${x} \right\rfloor"), ATOM),
    ps.Reciprocal: (Template(r"\frac{1}{${x}}"), FRACTION),
    ps.Exp2: (Template(r"2^{${x}}"), POWER),
    ps.Expm1: (Template(r"\exp\left(${x}\right) - 1"), ADDITIVE),
    ps.Log1p: (Template(r"\log\left(1 + ${x}\right)"), ATOM),
    psm.Log1mexp: (Template(r"\log\left(1 - \exp\left(${x}\right)\right)"), ATOM),
    psm.Softplus: (Template(r"\log\left(1 + \exp\left(${x}\right)\right)"), ATOM),
    psm.NdtriExp: (Template(r"\Phi^{-1}\left(\exp\left(${x}\right)\right)"), ATOM),
    ps.Conj: (Template(r"\overline{${x}}"), ATOM),
    ps.Sqrt: (Template(r"\sqrt{${x}}"), ATOM),
}

COMPARISON_SYMBOLS = {
    ps.EQ: "=",
    ps.NEQ: r"\neq",
    ps.LT: "<",
    ps.LE: r"\leq",
    ps.GT: ">",
    ps.GE: r"\geq",
}

LOGICAL_SYMBOLS = {
    ps.AND: r"\land",
    ps.OR: r"\lor",
    ps.XOR: r"\oplus",
}

REDUCTION_SYMBOLS = {
    ps.Add: r"\sum",
    ps.Mul: r"\prod",
    ps.Maximum: r"\max",
    ps.Minimum: r"\min",
}


@dataclass(frozen=True)
class LatexTerm:
    """
    LaTeX text of an xtensor expression, with its dims written as placeholders.

    ``free_dims`` maps each dim of the expression to the placeholder that stands for it in ``text``. ``bound_dims``
    maps the placeholder of each dim a reduction inside the expression has summed over to the name it prints as.
    """

    text: str
    precedence: int
    free_dims: dict[str, str]
    bound_dims: dict[str, str]


def equation_to_latex(equation: XTensorEquation, model: "CGEModel") -> str:
    """
    Return the LaTeX of ``equation``, its two sides joined by ``=``, with dims written by name.

    Raises
    ------
    NotImplementedError
        If the equation uses an op, scalar op or index with no LaTeX rule.
    """
    try:
        lhs, rhs = _unify_dims(
            [_graph_to_term(equation.lhs, model=model), _graph_to_term(equation.rhs, model=model)]
        )
    except NotImplementedError as error:
        raise NotImplementedError(
            f"Can't print equation {equation.name!r} as LaTeX: {error}"
        ) from error
    return f"{_resolve_dims(lhs)} = {_resolve_dims(rhs)}"


@singledispatch
def latex_funcify(op, node=None, inputs=None, model=None, **kwargs) -> LatexTerm:
    """
    Return the LaTeX term of ``node``'s output from ``inputs``, the terms of its inputs.

    Scalar ops dispatch here too, with the terms of the elementwise operands as ``inputs``. ``inputs`` holds ``None``
    for inputs that are not printed, such as the tensor under a model object or an index.
    """
    raise NotImplementedError(f"No LaTeX conversion for the given `Op`: {op}")


@latex_funcify.register(XTensorFromTensor)
def latex_funcify_XTensorFromTensor(op, node, model, **kwargs):
    [tensor] = node.inputs
    model_object = model._variables.get(tensor.name) or model._parameters.get(tensor.name)
    if model_object is None:
        raise NotImplementedError(f"{tensor.name!r} is not a variable or parameter of the model")

    placeholders = [_placeholder(node, key=position) for position in range(len(op.dims))]
    subscript = "_{" + ", ".join(placeholders) + "}" if placeholders else ""
    text = model_object.latex_name_with_subscript(subscript)
    return LatexTerm(
        text=text,
        precedence=POWER if _has_superscript(text) else ATOM,
        free_dims=dict(zip(op.dims, placeholders, strict=True)),
        bound_dims={},
    )


@latex_funcify.register(Rename)
def latex_funcify_Rename(op, node, inputs, **kwargs):
    [term] = inputs
    old_dims = node.inputs[0].type.dims
    free_dims = {new: term.free_dims[old] for old, new in zip(old_dims, op.new_dims, strict=True)}

    introduced = {new for old, new in zip(old_dims, op.new_dims, strict=True) if old != new}
    bound_dims = {}
    for placeholder, name in term.bound_dims.items():
        while name in introduced:
            name = f"{name}'"
        bound_dims[placeholder] = name

    return replace(term, free_dims=free_dims, bound_dims=bound_dims)


@latex_funcify.register(Transpose)
def latex_funcify_Transpose(op, inputs, **kwargs):
    [term] = inputs
    return term


@latex_funcify.register(XReduce)
def latex_funcify_XReduce(op, node, inputs, **kwargs):
    [term] = inputs
    symbol = REDUCTION_SYMBOLS.get(type(op.binary_op))
    if symbol is None:
        raise NotImplementedError(f"No LaTeX conversion for a reduction by {op.binary_op}")

    if term.precedence == REDUCTION:
        operand = term.text
    else:
        operand = _parenthesize(term, minimum_precedence=MULTIPLICATIVE)
    bound_placeholders = [_placeholder(node, key=f"bound.{dim}") for dim in op.dims]
    for dim, bound_placeholder in zip(op.dims, bound_placeholders, strict=True):
        operand = operand.replace(term.free_dims[dim], bound_placeholder)

    return LatexTerm(
        text=rf"{symbol}_{{{', '.join(bound_placeholders)}}} {operand}",
        precedence=REDUCTION,
        free_dims={
            dim: placeholder for dim, placeholder in term.free_dims.items() if dim not in op.dims
        },
        bound_dims={**term.bound_dims, **dict(zip(bound_placeholders, op.dims, strict=True))},
    )


@latex_funcify.register(Index)
def latex_funcify_Index(op, node, inputs, model, **kwargs):
    term = inputs[0]
    x, *indices = node.inputs
    text, free_dims = term.text, dict(term.free_dims)

    for dim, index in zip(x.type.dims, indices, strict=False):
        if _is_full_slice(index):
            continue
        if not (isinstance(index, Constant) and np.ndim(index.data) == 0):
            raise NotImplementedError(
                f"No LaTeX conversion for indexing {dim!r} by anything but a single label"
            )
        label = model.coords[dim][int(index.data)]
        text = text.replace(free_dims.pop(dim), rf"\text{{{label}}}")

    return replace(term, text=text, free_dims=free_dims)


@latex_funcify.register(XElemwise)
def latex_funcify_XElemwise(op, node, inputs, model, **kwargs):
    return latex_funcify(op.scalar_op, node=node, inputs=_unify_dims(inputs), model=model, **kwargs)


@latex_funcify.register(ps.Add)
def latex_funcify_Add(op, inputs, **kwargs):
    terms = [term for term in inputs if term.text != "0"]
    if not terms:
        return _elemwise_term("0", precedence=ATOM, inputs=inputs)
    if len(terms) == 1:
        return _elemwise_term(terms[0].text, precedence=terms[0].precedence, inputs=inputs)

    first, *rest = terms
    text = _parenthesize(first, minimum_precedence=ADDITIVE)
    for term in rest:
        if term.precedence == NEGATION and term.text.startswith("-"):
            text += f" - {term.text.removeprefix('-')}"
        else:
            text += f" + {_parenthesize(term, minimum_precedence=ADDITIVE)}"
    return _elemwise_term(text, precedence=ADDITIVE, inputs=inputs)


@latex_funcify.register(ps.Sub)
def latex_funcify_Sub(op, inputs, **kwargs):
    minuend, subtrahend = inputs
    text = (
        f"{_parenthesize(minuend, minimum_precedence=ADDITIVE)} - "
        f"{_parenthesize(subtrahend, minimum_precedence=MULTIPLICATIVE)}"
    )
    return _elemwise_term(text, precedence=ADDITIVE, inputs=inputs)


@latex_funcify.register(ps.Mul)
def latex_funcify_Mul(op, inputs, **kwargs):
    first, *rest = (_parenthesize(term, minimum_precedence=MULTIPLICATIVE) for term in inputs)
    text = first
    for factor in rest:
        touches_parentheses = factor.startswith(r"\left(") or text.endswith(r"\right)")
        if touches_parentheses and not factor[0].isdigit():
            text += f" {factor}"
        else:
            text += rf" \cdot {factor}"
    return _elemwise_term(text, precedence=MULTIPLICATIVE, inputs=inputs)


@latex_funcify.register(ps.TrueDiv)
def latex_funcify_TrueDiv(op, inputs, **kwargs):
    numerator, denominator = inputs
    text = rf"\frac{{{numerator.text}}}{{{denominator.text}}}"
    return _elemwise_term(text, precedence=FRACTION, inputs=inputs)


@latex_funcify.register(ps.Pow)
def latex_funcify_Pow(op, inputs, **kwargs):
    base, exponent = inputs
    text = f"{_parenthesize(base, minimum_precedence=ATOM)}^{{{exponent.text}}}"
    return _elemwise_term(text, precedence=POWER, inputs=inputs)


@latex_funcify.register(ps.Neg)
def latex_funcify_Neg(op, inputs, **kwargs):
    [term] = inputs
    return _elemwise_term(
        f"-{_parenthesize(term, minimum_precedence=MULTIPLICATIVE)}",
        precedence=NEGATION,
        inputs=inputs,
    )


def latex_funcify_function_call(op, inputs, **kwargs):
    arguments = ", ".join(term.text for term in inputs)
    return _elemwise_term(
        rf"{FUNCTION_NAMES[type(op)]}\left({arguments}\right)", precedence=ATOM, inputs=inputs
    )


def latex_funcify_template(op, inputs, **kwargs):
    template, precedence = TEMPLATES[type(op)]
    text = template.substitute(dict(zip("xy", (term.text for term in inputs), strict=False)))
    return _elemwise_term(text, precedence=precedence, inputs=inputs)


def latex_funcify_comparison(op, inputs, **kwargs):
    left, right = (_parenthesize(term, minimum_precedence=ADDITIVE) for term in inputs)
    text = f"{left} {COMPARISON_SYMBOLS[type(op)]} {right}"
    return _elemwise_term(text, precedence=COMPARISON, inputs=inputs)


def latex_funcify_logical(op, inputs, **kwargs):
    left, right = (_parenthesize(term, minimum_precedence=COMPARISON) for term in inputs)
    text = f"{left} {LOGICAL_SYMBOLS[type(op)]} {right}"
    return _elemwise_term(text, precedence=LOGICAL, inputs=inputs)


@latex_funcify.register(ps.Mod)
def latex_funcify_Mod(op, inputs, **kwargs):
    dividend, divisor = inputs
    text = (
        rf"{_parenthesize(dividend, minimum_precedence=MULTIPLICATIVE)} \bmod "
        rf"{_parenthesize(divisor, minimum_precedence=POWER)}"
    )
    return _elemwise_term(text, precedence=MULTIPLICATIVE, inputs=inputs)


@latex_funcify.register(ps.Complex)
def latex_funcify_Complex(op, inputs, **kwargs):
    real, imaginary = inputs
    text = rf"{real.text} + i \cdot {_parenthesize(imaginary, minimum_precedence=MULTIPLICATIVE)}"
    return _elemwise_term(text, precedence=ADDITIVE, inputs=inputs)


@latex_funcify.register(ps.Second)
def latex_funcify_Second(op, inputs, **kwargs):
    _, term = inputs
    return term


@latex_funcify.register(ps.Identity)
@latex_funcify.register(ps.Cast)
def latex_funcify_Identity(op, inputs, **kwargs):
    [term] = inputs
    return term


@latex_funcify.register(ps.Sqr)
def latex_funcify_Sqr(op, inputs, **kwargs):
    [term] = inputs
    text = f"{_parenthesize(term, minimum_precedence=ATOM)}^{{2}}"
    return _elemwise_term(text, precedence=POWER, inputs=inputs)


@latex_funcify.register(psm.Sigmoid)
def latex_funcify_Sigmoid(op, inputs, **kwargs):
    [term] = inputs
    negated = f"-{_parenthesize(term, minimum_precedence=MULTIPLICATIVE)}"
    text = rf"\frac{{1}}{{1 + \exp\left({negated}\right)}}"
    return _elemwise_term(text, precedence=FRACTION, inputs=inputs)


@latex_funcify.register(ps.Invert)
def latex_funcify_Invert(op, inputs, **kwargs):
    [term] = inputs
    text = rf"\lnot {_parenthesize(term, minimum_precedence=MULTIPLICATIVE)}"
    return _elemwise_term(text, precedence=NEGATION, inputs=inputs)


def _graph_to_term(output: Variable, model: "CGEModel") -> LatexTerm:
    terms: dict[Variable, LatexTerm] = {}

    def term_of(variable: Variable) -> LatexTerm | None:
        if variable in terms:
            return terms[variable]
        if isinstance(variable, XTensorConstant):
            return _constant_term(variable)
        return None

    printed_nodes = {variable.owner for variable in walk([output], _printed_inputs)}
    for node in toposort([output]):
        if node in printed_nodes:
            terms[node.outputs[0]] = latex_funcify(
                node.op,
                node=node,
                inputs=[term_of(variable) for variable in node.inputs],
                model=model,
            )

    term = term_of(output)
    if term is None:
        raise NotImplementedError(f"No LaTeX conversion for the graph input {output}")
    return term


def _printed_inputs(variable: Variable) -> list[Variable] | None:
    """Return the inputs whose LaTeX the LaTeX of ``variable`` contains. An index prints only its indexed expression."""
    node = variable.owner
    if node is None:
        return None
    return node.inputs[:1] if isinstance(node.op, Index) else node.inputs


def _elemwise_term(text: str, precedence: int, inputs: list[LatexTerm]) -> LatexTerm:
    return LatexTerm(
        text=text,
        precedence=precedence,
        free_dims={
            dim: placeholder for term in inputs for dim, placeholder in term.free_dims.items()
        },
        bound_dims={name: dim for term in inputs for name, dim in term.bound_dims.items()},
    )


def _constant_term(constant: XTensorConstant) -> LatexTerm:
    value = np.asarray(constant.data)
    if value.ndim != 0:
        raise NotImplementedError("No LaTeX conversion for a constant that is not a scalar")
    number = float(value)
    if np.isnan(number):
        text, precedence = r"\mathrm{NaN}", ATOM
    elif np.isinf(number):
        text, precedence = ("-" if number < 0 else "") + r"\infty", ATOM
    elif number != 0 and not SCIENTIFIC_BELOW <= abs(number) < SCIENTIFIC_ABOVE:
        mantissa, exponent = np.format_float_scientific(number, trim="-").split("e")
        power = f"10^{{{int(exponent)}}}"
        if mantissa.lstrip("-") == "1":
            text, precedence = mantissa.removesuffix("1") + power, POWER
        else:
            text, precedence = rf"{mantissa} \times {power}", MULTIPLICATIVE
    else:
        text, precedence = np.format_float_positional(number, trim="-"), ATOM

    return LatexTerm(
        text=text,
        precedence=NEGATION if text.startswith("-") else precedence,
        free_dims={},
        bound_dims={},
    )


def _is_full_slice(index: Variable) -> bool:
    if not isinstance(index.type, SliceType):
        return False
    if isinstance(index, Constant):
        return index.data == slice(None)
    return (
        index.owner is not None
        and isinstance(index.owner.op, MakeSlice)
        and all(isinstance(part, Constant) and part.data is None for part in index.owner.inputs)
    )


def _has_superscript(text: str) -> bool:
    depth = 0
    for character in text:
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
        elif character == "^" and depth == 0:
            return True
    return False


def _placeholder(node: Apply, key: int | str) -> str:
    return f"\x00{id(node)}.{key}\x00"


def _parenthesize(term: LatexTerm, minimum_precedence: int) -> str:
    if term.precedence < minimum_precedence:
        return rf"\left({term.text}\right)"
    return term.text


def _unify_dims(terms: Sequence[LatexTerm | None]) -> list[LatexTerm]:
    """Rewrite the terms so that every free dim of the same name uses the placeholder of its first occurrence."""
    placeholders: dict[str, str] = {}
    unified = []
    for term in terms:
        if term is None:
            raise NotImplementedError(
                "No LaTeX conversion for an elementwise input that is not an xtensor expression"
            )
        free_dims = {
            dim: placeholders.setdefault(dim, placeholder)
            for dim, placeholder in term.free_dims.items()
        }
        renamed = {term.free_dims[dim]: shared for dim, shared in free_dims.items()}
        unified.append(
            replace(term, text=_replace_simultaneously(term.text, renamed), free_dims=free_dims)
        )
    return unified


def _replace_simultaneously(text: str, replacements: Mapping[str, str]) -> str:
    """Replace every key of ``replacements`` in one pass, so a replacement is never replaced again."""
    if not replacements:
        return text
    pattern = re.compile("|".join(re.escape(old) for old in replacements))
    return pattern.sub(lambda match: replacements[match.group(0)], text)


def _resolve_dims(term: LatexTerm) -> str:
    text = term.text
    for dim, placeholder in term.free_dims.items():
        text = text.replace(placeholder, _index_name(dim))
    for placeholder, name in term.bound_dims.items():
        text = text.replace(placeholder, _index_name(name))
    return text


def _index_name(name: str) -> str:
    unprimed = name.rstrip("'")
    return upright_multiletter_name(unprimed) + name[len(unprimed) :]


_TABLE_RULES: list[tuple[Iterable[type], Callable[..., LatexTerm]]] = [
    (FUNCTION_NAMES, latex_funcify_function_call),
    (TEMPLATES, latex_funcify_template),
    (COMPARISON_SYMBOLS, latex_funcify_comparison),
    (LOGICAL_SYMBOLS, latex_funcify_logical),
]
for _op_types, _rule in _TABLE_RULES:
    for _op_type in _op_types:
        latex_funcify.register(_op_type, _rule)
