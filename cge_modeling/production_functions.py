from itertools import combinations, product
from string import Template
from typing import Literal, Optional, Sequence, Union, cast

import sympy as sp

from cge_modeling.tools.pytensor_tools import at_least_list

BACKEND_TYPE = Literal["numba", "pytensor"]


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
    factors: Union[list[str, ...], str],
    factor_prices: Union[list[str, ...], str],
    output: str,
    output_price: str,
    TFP: str,
    factor_shares: Union[str, list[str, ...]],
    epsilon: str,
    *args,
    **kwargs,
) -> tuple[str, ...]:
    """
    Generate string equations representing a CES production process.

    A CES production process is defined as:

    .. math::
        Y = A \\left( \\sum_{i \\in I} \alpha_i X_i^{\frac{\\epsilon - 1}{\\epsilon}} \right)^{\frac{\\epsilon}{\\epsilon - 1}}

    Where :math:`A` is the total factor productivity parameter, :math:`\\alpha_i` is the share of factor :math:`X_i` in
    the production process, and :math:`\\epsilon` is the elasticity of substitution parameter. The profit maximization
    problem for this production function generates the following factor demands:

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

    TFP: str
        Technology parameter

    factor_shares: str, or list of str
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

    factor_shares = _add_second_alpha(factor_shares, factors)
    _check_pairwise_lengths_match(
        ["factors", "factor_prices", "factor_shares"], [factors, factor_prices, factor_shares]
    )

    production_inner_template = Template(
        "($factor_shares) * $factor ** (($epsilon - 1) / $epsilon)"
    )

    production_inner = [
        production_inner_template.safe_substitute(
            factor=factor, factor_shares=factor_share, epsilon=epsilon
        )
        for factor, factor_share in zip(factors, factor_shares)
    ]

    eq_production = Template(
        "$output = $TFP * ($inner) ** ($epsilon / ($epsilon - 1))"
    ).safe_substitute(output=output, inner=" + ".join(production_inner), TFP=TFP, epsilon=epsilon)

    factor_demand_template = Template(
        "$factor = $output / $TFP * (($factor_share) * $output_price * $TFP / $factor_price) ** "
        "$epsilon"
    )
    eq_fac_demands = [
        factor_demand_template.safe_substitute(
            factor=factor,
            factor_price=factor_price,
            output=output,
            output_price=output_price,
            factor_share=factor_share,
            TFP=TFP,
            epsilon=epsilon,
        )
        for factor, factor_price, factor_share in zip(factors, factor_prices, factor_shares)
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
    backend: BACKEND_TYPE = "numba",
    TFP: Optional[str] = None,
    factor_shares: Optional[str] = None,
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

    TFP: str, optional
        Technology parameter. If not provided, it is omitted from the equations, which is equivalent to setting it to 1.

    factor_shares: str, optional
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

    for var, name in zip(
        [factors, factor_prices, factor_shares], ["factors", "factor_prices", "factor_shares"]
    ):
        if isinstance(var, list) and len(var) != 1:
            raise ValueError(
                f"Dixit-Stiglitz production function expects only a single factor input, but provided "
                f"{name} has length {len(var)}"
            )

    if isinstance(factors, list):
        factors = factors[0]
    if isinstance(factor_prices, list):
        factor_prices = factor_prices[0]
    if isinstance(factor_shares, list):
        factor_shares = factor_shares[0]

    share_str = "$factor_share * " if factor_shares is not None else ""
    TFP_str = f"$TFP * " if TFP is not None else ""

    kernel_str = f"{share_str}$factor ** (($epsilon - 1) / $epsilon)"

    kernel_template = Template(kernel_str)
    kernel = kernel_template.safe_substitute(
        factor_share=factor_shares, factor=factors, epsilon=epsilon
    )

    dim_len = len(coords[dims]) - 1

    if backend == "numba":
        rhs_str = "Sum($kernel, ($dims, 0, $dim_len)) ** ($epsilon / ($epsilon - 1))"
    elif backend == "pytensor":
        rhs_str = "($kernel).sum() ** ($epsilon / ($epsilon - 1))"
    else:
        raise ValueError(f"backend must be one of 'numba' or 'pytensor', found {backend}")

    production_function = Template(f"$output = {TFP_str}{rhs_str}").safe_substitute(
        output=output, TFP=TFP, kernel=kernel, epsilon=epsilon, dims=dims, dim_len=dim_len
    )

    demand_template = f"$factor = $output / {TFP_str}({TFP_str}{share_str}$output_price / $factor_price) ** $epsilon"

    factor_demands = Template(demand_template).safe_substitute(
        factor=factors,
        output=output,
        TFP=TFP,
        factor_share=factor_shares,
        output_price=output_price,
        factor_price=factor_prices,
        epsilon=epsilon,
    )

    return production_function, factor_demands


def leontief(
    factors: Union[str, list[str]],
    factor_prices: Union[str, list[str]],
    output: str,
    output_price: str,
    factor_shares: Union[str, list[str]],
    dims: str,
    coords: dict[str, list[str]],
    backend: Literal["numba", "pytensor"] = "numba",
) -> tuple[str, str]:
    """
    Generate string equations representing a Leontief production process.

    A Leontief production process is defined as:

    .. math::
        Y = \\min_i \\left( \\frac{X_i}{\\alpha_i} \\right)

    Where :math:`\\alpha_i` is the share of factor :math:`i` in the production process. This generates the following
    factor demands:

    .. math::
        X_i = \\alpha_i Y

    Because of the non-linearity of the minimum operator, this function does not return the production function itself.
    Insead, we return the equivalent zero-profit condition:

    .. math::
        P_Y Y = \\sum_i P_i X_i

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

    pass
