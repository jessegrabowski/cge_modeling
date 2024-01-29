from itertools import combinations, product
from string import Template
from typing import Literal, Optional, Sequence, Union, cast

import sympy as sp

from cge_modeling.tools.pytensor_tools import at_least_list


def _check_pairwise_lengths_match(names, args):
    list_args = [at_least_list(arg) for arg in args]
    arg_pairs = combinations(list_args, 2)
    name_pairs = combinations(names, 2)
    for name_pair, arg_pair in zip(name_pairs, arg_pairs):
        if len(arg_pair[0]) != len(arg_pair[1]):
            raise ValueError(f"Lengths of {name_pair[0]} and {name_pair[1]} do not match")


def unwrap_singleton_list(x):
    if isinstance(x, list) and (len(x) == 1):
        return x[0]
    else:
        return x


def unpack_string_inputs(
    *inputs: Union[str, list[str]],
    dims: Optional[Sequence[str]] = None,
    coords: Optional[dict[str, list[str]]] = None,
) -> list[list[str]]:
    """
    Unpack string inputs into lists of variables using the provided dims and coords.


    Parameters
    ----------
    inputs: str or list of str
        Inputs to be unpacked. If an input is a string, it will be unpacked into N inputs using the provided dims.

    dims: Sequence of str or str
        Sequence of named dimensions for variables in the equations to be generated. All dimensions should appear in
        the coords dictionary as keys, otherwise an error will be raised.

    coords: dict of str: list of str
        Dictionary of coordinates for the model, mapping dimension names to lists of labels.

    Returns
    -------
    unpacked_inputs: list of lists of str
        unpacked factors and factor prices

    Notes
    -----
    If inputs are lists, this function will return them unchanged.
    """
    inputs = list(map(unwrap_singleton_list, inputs))

    if all([isinstance(x, list) for x in inputs]):
        return list(map(at_least_list, inputs))

    if coords is None or dims is None:
        return list(map(at_least_list, inputs))

    if not all([key in coords.keys() for key in dims]):
        raise KeyError(f"Dimensions {dims} not found in coords")

    labels = [coords[dim] for dim in dims]
    label_prod = product(*labels)
    outputs: list[list[str]] = [[] for _ in inputs]

    for labels in label_prod:
        label_list = [x for x in labels]

        for i, input in enumerate(inputs):
            outputs[i].append("_".join([input] + label_list))

    return outputs


def _add_second_alpha(alpha: Union[str, list[str]], factors: Union[str, list[str]]) -> list[str]:
    if not isinstance(factors, list) or len(factors) != 2:
        # If there are not exactly two factors, we don't need to add a second alpha. It will either be unpacked
        # later, or it will be an error.
        return cast(list[str], at_least_list(alpha))

    alpha = cast(list[str], at_least_list(alpha))

    if len(factors) == 2 and len(alpha) == 1:
        alpha.append(f"1 - {alpha[0]}")
        return alpha

    return alpha


def CES(
    factors: Union[list[str], str],
    factor_prices: Union[list[str], str],
    output: str,
    output_price: str,
    A: str,
    alphas: str,
    epsilon: str,
    *args,
    **kwargs,
) -> tuple[str, ...]:
    """
    Generate string equations representing a CES production process.

    A CES production process is defined as:

    .. math::
        Y = A \\left( \\sum_{i \\in I} \alpha_i X_i^{\frac{\\epsilon - 1}{\\epsilon}} \right)^{\frac{\\epsilon}{\\epsilon - 1}}

    Which generates the following factor demands:

    .. math::
        X_i = \frac{Y}{A} \\left( \frac{\alpha_i P_Y A}{P_i} \right)^{\\epsilon}

    This definition is the one most frequently used in the CGE literature.

    Parameters
    ----------
    factors: str or list of str
        Production factors. If a string is provided, it will be unpacked into N prices using the
        provided dims.

    factor_prices: str or list of str
        Production factor prices. If a string is provided, it will be unpacked into N prices using the
        provided dims.

    output: str
        Name of the output of the production function

    output_price: str
        Name of the price of the output

    A: str
        Technology parameter

    alphas: str, or list of str
        Factor shares in the production function. If a string is provided and len(factors) == 2, the second factor
        share will be set to (1 - alpha). If a list of strings is provided, it must be the same length as factors.

    epsilon:
        elasticity parameter

    args, kwargs:
        Ignored; included for signature compatibility with other production functions

    Returns
    -------
    (variables, parameters, equations): tuple of tuples of strings
        output and factor equations
    """

    alphas = _add_second_alpha(alphas, factors)
    _check_pairwise_lengths_match(
        ["factors", "factor_prices", "alpha"], [factors, factor_prices, alphas]
    )

    production_inner_template = Template("($alpha) * $factor ** (($epsilon - 1) / $epsilon)")

    production_inner = [
        production_inner_template.safe_substitute(factor=factor, alpha=alpha, epsilon=epsilon)
        for factor, alpha in zip(factors, alphas)
    ]

    eq_production = Template(
        "$output = $A * ($inner) ** ($epsilon / ($epsilon - 1))"
    ).safe_substitute(output=output, inner=" + ".join(production_inner), A=A, epsilon=epsilon)

    factor_demand_template = Template(
        "$factor = $output / $A * (($alpha) * $output_price * $A / $factor_price) ** " "$epsilon"
    )
    eq_fac_demands = [
        factor_demand_template.safe_substitute(
            factor=factor,
            factor_price=factor_price,
            output=output,
            output_price=output_price,
            alpha=alpha,
            A=A,
            epsilon=epsilon,
        )
        for factor, factor_price, alpha in zip(factors, factor_prices, alphas)
    ]

    return (eq_production,) + tuple(eq_fac_demands)


def dixit_stiglitz(
    factors: Union[str, list[str]],
    factor_prices: Union[str, list[str]],
    output: str,
    output_price: str,
    epsilon: str,
    dims: str,
    coords: dict[str, list[str]],
    backend: Literal["numba", "pytensor"] = "numba",
    A: Optional[str] = None,
    alphas: Optional[str] = None,
) -> tuple[str, str]:
    """
    Generate string equations representing a Dixit-Stiglitz production process.

    Here, the Dixit-Stiglitz production process is defined as:

    .. math::
        Y = A \\left( \\sum_{i \\in I} \alpha_i X_i^{\frac{\\epsilon - 1}{\\epsilon}} \right)^{\frac{\\epsilon}{\\epsilon - 1}}

    Which generates the following factor demands:

    .. math::
        X_i = \frac{Y}{A} \\left( \frac{\alpha_i P_Y A}{P_i} \right)^{\\epsilon}

    Many researchers omit either the technology parameter, the factor shares, or both. If the technology parameter is
    not provided (i.e. :math:`A` is None), it is assumed to be 1. If the factor shares are omitted they are also set
    to 1.

    Parameters
    ----------
    factors: str or list of str
        Production factor. If a list is provided, it must be of length 1.

        .. warning::
        Despite the plural name, the Dixit-Stiglitz production function expects only a single facto input. The name
        is plural to be consistent with other production functions.

    factor_prices: str or list of str
        Production factor prices. If a list is provided, it must be of length 1.

    output: str
        Name of the output of the production function

    output_price: str
        Name of the price of the output

    A: str, optional
        Technology parameter. If not provided, it is omitted from the equations, which is equivalent to setting it to 1.

    alphas: str, optional
        Factor shares in the production function. If a list is provided, it must be of length 1. If not provided, it is
        omitted from the equations, which is equivalent to setting it to 1.

    epsilon:
        Elasticity of substitution parameter

    dims: Sequence of str or str
        Sequence of named dimensions for variables in the equations to be generated. All dimensions should appear in
        the coords dictionary as keys, otherwise an error will be raised.

    coords: dict of str: list of str
        Dictionary of coordinates for the model, mapping dimension names to lists of labels.

    backend: str, one of 'numba' or 'pytensor'
        Backend that will parse the equations. Only relevant for aggregation functions like Sum and Prod. Elementwise
        operations are interoperable between backends.

    Returns
    -------
    (variables, parameters, equations): tuple of tuples of strings
        output and factor equations
    """

    for var, name in zip([factors, factor_prices, alphas], ["factors", "factor_prices", "alphas"]):
        if isinstance(var, list) and len(var) != 1:
            raise ValueError(
                f"Dixit-Stiglitz production function expects only a single factor input, but provided "
                f"{name} has length {len(var)}"
            )

    if isinstance(factors, list):
        factors = factors[0]
    if isinstance(factor_prices, list):
        factor_prices = factor_prices[0]
    if isinstance(alphas, list):
        alphas = alphas[0]

    alpha_str = "$alpha * " if alphas is not None else ""
    A_str = f"$A * " if A is not None else ""

    kernel_str = f"{alpha_str}$factor ** (($epsilon - 1) / $epsilon)"

    kernel_template = Template(kernel_str)
    kernel = kernel_template.safe_substitute(alpha=alphas, factor=factors, epsilon=epsilon)

    dim_len = len(coords[dims]) - 1

    if backend == "numba":
        rhs_str = "Sum($kernel, ($dims, 0, $dim_len)) ** ($epsilon / ($epsilon - 1))"
    elif backend == "pytensor":
        rhs_str = "($kernel).sum() ** ($epsilon / ($epsilon - 1))"
    else:
        raise ValueError(f"backend must be one of 'numba' or 'pytensor', found {backend}")

    production_function = Template(f"$output = {A_str}{rhs_str}").safe_substitute(
        output=output, A=A, kernel=kernel, epsilon=epsilon, dims=dims, dim_len=dim_len
    )

    demand_template = (
        f"$factor = $output / {A_str}({A_str}{alpha_str}$output_price / $factor_price) ** $epsilon"
    )

    factor_demands = Template(demand_template).safe_substitute(
        factor=factors,
        output=output,
        A=A,
        alpha=alphas,
        output_price=output_price,
        factor_price=factor_prices,
        epsilon=epsilon,
    )

    return production_function, factor_demands


def cobb_douglass(
    output: sp.Symbol,
    output_price: sp.Symbol,
    tfp: sp.Symbol,
    inputs: list[sp.Symbol],
    input_prices: list[sp.Symbol],
    shares: list[sp.Symbol],
):
    if len(inputs) != len(shares):
        if (len(shares) + 1) != len(inputs):
            raise ValueError("The length of the shares should len(inputs), or len(inputs) - 1")
        shares.append(1 - sum(shares))

    system = [output - tfp * sp.prod([x**a for x, a in zip(inputs, shares)])]
    for factor, price, share in zip(inputs, input_prices, shares):
        system.append(factor - share * output * output_price / price)

    return system


def leontief(
    output: sp.Symbol,
    output_price: sp.Symbol,
    inputs: list[sp.Symbol],
    input_prices: list[sp.Symbol],
    params: list[sp.Symbol],
):
    if len(inputs) != len(params):
        raise ValueError("The length of the params should be len(inputs)")

    system = [output_price * output - sum(P * x for P, x in zip(input_prices, inputs))]
    for factor, param in zip(inputs, params):
        system.append(factor - param * output)

    return system
