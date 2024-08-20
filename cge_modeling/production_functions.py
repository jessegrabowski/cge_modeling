import string

from collections.abc import Sequence
from itertools import combinations, product
from string import Template
from typing import Literal, cast

from cge_modeling.tools.utils import at_least_list

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
    *inputs: str | list[str, ...],
    dims: Sequence[str] | None = None,
    coords: dict[str, list[str, ...]] | None = None,
) -> list[list[str, ...]]:
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
    outputs: list[list[str, ...]] = [[] for _ in inputs]

    for labels in label_prod:
        label_list = [x for x in labels]

        for i, input in enumerate(inputs):
            outputs[i].append("_".join([input, *label_list]))

    return outputs


def _add_second_alpha(alpha: str | list[str, ...], factors: str | list[str, ...]) -> list[str, ...]:
    if not isinstance(factors, list) or len(factors) != 2:
        # If there are not exactly two factors, we don't need to add a second alpha. It will either be unpacked
        # later, or it will be an error.
        return cast(list[str, ...], at_least_list(alpha))

    alpha = cast(list[str, ...], at_least_list(alpha))

    if len(factors) == 2 and len(alpha) == 1:
        alpha.append(f"1 - {alpha[0]}")
        return alpha

    return alpha


def cobb_douglass(
    factors: list[str, ...],
    factor_prices: list[str, str],
    output: str,
    output_price: str,
    factor_shares: str,
    TFP: str = "1",
    use_value_definition: bool = False,
) -> tuple[str, ...]:
    """
    Generate string equations representing a Cobb-Douglas production process.

    A Cobb-Douglas production process is defined as:

    .. math::
        Y = A \\prod_{i \\in I} X_i^{\\alpha_i}

    Where :math:`A` is the total factor productivity parameter, :math:`\\alpha_i` is the share of factor :math:`X_i` in
    the production process. The profit maximization problem for this production function generates the following factor
    demands:

    .. math::
        X_i = \frac{Y}{A} \frac{\\alpha_i P_Y A}{P_i}

    A Cobb-Douglas can be shown to be a special case of the CES production function, when the elasticity of substitution
    between input factors is :math:`\\epsilon = 1`. In this special case, the ratio of factor inputs ..math:`\\frac{X_i}{X_j}`
    is always equal to the ratio of factor prices ..math:`\\frac{P_i}{P_j}`. That is, the proportion of the i-th
    input factor used in the production process is always exactly ..math:`\\alpha_i`, and prices adjust to ensure that
    this is so.

    Parameters
    ----------
    factors: list of str
        Production factors

    factor_prices: list of str
        Production factor prices. Must be the same length as factors.

    output: str
        Name of the output of the production function.

    output_price: str
        Name of the price of the output.

    factor_shares: str or list of str
        Factor shares in the production function. If a list, it should be the same length as factors, or one less. If
        one less, the last factor share is assumed to be 1 - sum(factor_shares).

        A string is valid only if len(factors) == 2, in which case the string is interpreted as [factor_shares].

    TFP: str, default "1"
        Technology parameter

    Returns
    -------
    production_function: str
        Equation representing the Cobb-Douglas production function
    factor_demands: list of str
        A list of equations, one for each factor, representing the factor demands
    """
    if isinstance(factor_shares, str):
        factor_shares = [factor_shares]

    factor_shares = _add_second_alpha(factor_shares, factors)
    _check_pairwise_lengths_match(
        ["factors", "factor_prices", "factor_shares"],
        [factors, factor_prices, factor_shares],
    )

    if not use_value_definition:
        production_inner_template = Template("$factor ** ($factor_share)")
        production_inner = [
            production_inner_template.safe_substitute(factor=factor, factor_share=factor_share)
            for factor, factor_share in zip(factors, factor_shares)
        ]

        eq_production = Template(
            "$output = $TFP * " + " * ".join(production_inner)
        ).safe_substitute(output=output, TFP=TFP)
    else:
        total_factor_value = " + ".join(
            [f"{factor_price} * {factor}" for factor_price, factor in zip(factor_prices, factors)]
        )
        eq_production = Template(
            "$output * $output_price = $total_factor_value",
        ).safe_substitute(
            output=output,
            output_price=output_price,
            total_factor_value=total_factor_value,
        )

    factor_demand_template = Template(
        "$factor = ($factor_share) * $output * ($output_price) / ($factor_price)"
    )

    eq_fac_demands = [
        factor_demand_template.safe_substitute(
            factor=factor,
            factor_price=factor_price,
            output=output,
            output_price=output_price,
            factor_share=factor_share,
            TFP=TFP,
        )
        for factor, factor_price, factor_share in zip(factors, factor_prices, factor_shares)
    ]

    return (eq_production, *tuple(eq_fac_demands))


def CES(
    factors: list[str, ...] | str,
    factor_prices: list[str, ...] | str,
    output: str,
    output_price: str,
    TFP: str,
    factor_shares: str | list[str, ...],
    epsilon: str,
    expand_price_dim: Literal["input", "output", "both", None] = None,
    use_value_definition=False,
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

    expand_price_dim: str, one of 'input', 'output', 'both', or None
        Price dimension to expand. If expanded, a price P will be printed as P[:, None].

    args, kwargs:
        Ignored; included for signature compatibility with other production functions

    Returns
    -------
    (variables, parameters, equations): tuple of tuples of strings
        output and factor equations
    """

    factor_shares = _add_second_alpha(factor_shares, factors)
    _check_pairwise_lengths_match(
        ["factors", "factor_prices", "factor_shares"],
        [factors, factor_prices, factor_shares],
    )

    if not use_value_definition:
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
        ).safe_substitute(
            output=output, inner=" + ".join(production_inner), TFP=TFP, epsilon=epsilon
        )
    else:
        total_factor_value = " + ".join(
            [f"{factor_price} * {factor}" for factor_price, factor in zip(factor_prices, factors)]
        )
        eq_production = Template(
            "$output * $output_price = $total_factor_value",
        ).safe_substitute(
            output=output,
            output_price=output_price,
            total_factor_value=total_factor_value,
        )

    factor_demand_template = Template(
        "$factor = $output / $TFP * (($factor_share) * $output_price * $TFP / ($factor_price)) ** "
        "$epsilon"
    )

    if expand_price_dim in ["output", "both"]:
        output_price = f"{output_price}[:, None]"
    if expand_price_dim in ["input", "both"]:
        factor_prices = [f"{factor_price}[:, None]" for factor_price in factor_prices]

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

    return (eq_production, *tuple(eq_fac_demands))


def dixit_stiglitz(
    factors: str | list[str, ...],
    factor_prices: str | list[str, ...],
    output: str,
    output_price: str,
    epsilon: str,
    dims: str,
    coords: dict[str, list[str, ...]],
    backend: BACKEND_TYPE = "numba",
    TFP: str | None = None,
    factor_shares: str | None = None,
    use_value_definition=True,
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
        [factors, factor_prices, factor_shares],
        ["factors", "factor_prices", "factor_shares"],
    ):
        if isinstance(var, list) and len(var) != 1:
            raise ValueError(
                f"Dixit-Stiglitz production function expects only a single factor input, but provided "
                f"{name} has length {len(var)}"
            )

    factors, factor_prices, factor_shares = map(
        unwrap_singleton_list, (factors, factor_prices, factor_shares)
    )

    share_str = "$factor_share * " if factor_shares is not None else ""
    TFP_str = "$TFP * " if TFP is not None else ""

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

    if not use_value_definition:
        production_function = Template(f"$output = {TFP_str}{rhs_str}").safe_substitute(
            output=output,
            TFP=TFP,
            kernel=kernel,
            epsilon=epsilon,
            dims=dims,
            dim_len=dim_len,
        )
    else:
        production_function = Template(
            "$output = ($factor_prices * $factors).sum() / ($output_price)",
        ).safe_substitute(
            output=output,
            output_price=output_price,
            factor_prices=factor_prices,
            factors=factors,
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


def _1d_leontief(
    factors: str | list[str, ...],
    factor_prices: str | list[str, ...],
    output: str,
    output_price: str,
    factor_shares: str | list[str, ...],
    *args,
    **kwargs,
) -> tuple[str, ...]:
    """
    Helper function to generate Leontief production equaitons for the case where ndim == 1.

    """
    zp_rhs = " + ".join([f"{price} * {factor}" for factor, price in zip(factors, factor_prices)])
    zero_profit = f"{output} = ({zp_rhs}) / ({output_price})"

    factor_demands = (
        f"{factor} = {output} * {share}" for factor, share in zip(factors, factor_shares)
    )

    return (zero_profit, *tuple(factor_demands))


def _2d_leontief(
    factors: str | list[str, ...],
    factor_prices: str | list[str, ...],
    output: str,
    output_price: str,
    factor_shares: str | list[str, ...],
    dims: str,
    coords: dict[str, list[str, ...]],
    sum_dim: str | None = None,
    expand_price_dim: bool = True,
    transpose_output: bool = True,
    backend: Literal["numba", "pytensor"] = "numba",
) -> tuple[str, ...]:
    r"""
    Helper function to generate Leontief production equations for the case where ndim == 2.

    Notes
    -----
    Sympy is quite restrictive in terms of broadcasting, so there are many hoops to jump through here.
    Assume that X is an N x N matrix of factor demands, phi is an N x N matrix of technological coefficients,
    Y is an N x 1 vector of outputs and P_Y is a vectors of prices Y. P_X is either a vector or a matrix of factor
    prices, depending on the situaiton. If it is a vector, set expand_price_dim to True; if it is a matrix, set it to
    False.

    The important thing about X is that it is indexed by the same labels twice. The columns represent demands by
    the j-th label to the i-th label. The rows, on the other hand, represent supply by the  i-th label to the
    j-th label. This is the standard representation of an input-output matrix in CGE modeling.

    The zero profit condition will ask us to sum across the **rows** of X:

    [X_00,   | , X_02, X_03]
    [X_10,   | , X_12, X_13]
    [--------+-------------]-> This sum gives total demand for goods produced by the i-th label
    [X_20,   | , X_22, X_23]
            \_/
             This sum gives total supply of goods produced by the j-th label

    If broadcasting, this operation is quite simple: we multiply X * phi, then sum across the columns with axis=0.
    With simpy indexed symbols, however, this is not possible. First, these don't have a notion of axis, so we
    have to manually manipulate the indices to get what we want.

    - First, make P_Y * Y a "column vector" by swapping the core and batch dimensions
    - Next, "transpose" X by swapping the core and batch dimension labels. This is necessary because the labels on the
        left- and right-hand sides of the equations need to agree -- since we've switched the core and batch on the
        left, we also have to do it on the right.
    - Finally, we can multiply (X * phi) and sum across the batch dimension

    This is all necessary because from the supply perspective, the 2nd dimension of X is the core dimension.
    """
    dims = list(dims)
    if sum_dim is None:
        sum_dim = dims[-1]
    sum_axis = dims.index(sum_dim)

    def _swapaxes(x, i, j):
        sub_template = Template("$x.subs({$i:$j})")
        return sub_template.safe_substitute(x=x, i=i, j=j)

    def _T(x, i, j, coords):
        named_dims = set(list(coords.keys()))

        # Use the first lowercase letter that hasn't been declared as a dim by the user as a temp
        temp_dim = sorted(list(set(string.ascii_lowercase) - named_dims))[0]
        return x + f".subs([({i}, {temp_dim}), ({j}, {i}), ({temp_dim}, {j})])"

    factors, factor_prices, factor_shares = map(
        unwrap_singleton_list, (factors, factor_prices, factor_shares)
    )

    if backend == "numba":
        batch_dim = dims.pop(sum_axis)
        core_dim = dims[0]
        n_core, n_batch = (len(coords[x]) for x in [core_dim, batch_dim])
        profit_output = output
        profit_price = output_price

        if transpose_output:
            profit_output = _swapaxes(output, batch_dim, core_dim)
            profit_price = _swapaxes(output_price, batch_dim, core_dim)

        zero_profit = f"{profit_output} = Sum({factor_prices} * {factors}, ({batch_dim}, 0, {n_batch - 1})) / {profit_price}"
        # zero_profit = f"{_swapaxes(output, batch_dim, core_dim)} = Sum({factor_prices} * {factors}, ({batch_dim}, 0, {n_batch - 1})) / {_swapaxes(output_price, batch_dim, core_dim)}"
        factor_demands = f"{factors} = {factor_shares} * {output}"

    elif backend == "pytensor":
        price_slice = "[:, None]" if expand_price_dim else ""
        zero_profit = f"{output} = ({factor_prices}{price_slice} * {factors}).sum(axis={sum_axis}).ravel() / ({output_price})"
        factor_demands = f"{factors} = {factor_shares} * {output}[None]"

    else:
        raise ValueError(f"backend must be one of 'numba' or 'pytensor', found {backend}")

    return zero_profit, factor_demands


def leontief(
    factors: str | list[str, ...],
    factor_prices: str | list[str, ...],
    output: str,
    output_price: str,
    factor_shares: str | list[str, ...],
    dims: str | list[str, ...],
    coords: dict[str, list[str, ...]],
    sum_dim: str | None = None,
    expand_price_dim: bool = True,
    transpose_output: bool = True,
    backend: Literal["numba", "pytensor"] = "numba",
) -> tuple[str, ...]:
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
    factors: list of str
        Input factors to the production process.

    factor_prices: list of str
        Production factor prices.

    output: str
        Name of the output of the production function

    output_price: str
        Name of the price of the output

    factor_shares: list of str
        Factor shares in the production function.

    dims: Sequence of str or str
        Sequence of dimensions indexing inputs and outputs into the Leontief production process. If a string or a list
        of length one is provided, the inputs and outputs will be assumed to be indexed by the provided dimension.

        If length == 2, the *second* dimension will be assumed to index the outputs, and the inputs will be reduced over
        the first dimension. This is the most common case in CGE modeling, when inputs to an intermediate consumption/
        value chain process are represented by an N x N matrix, with sectoral supply on the first dimension and sectoral
        demand on the second dimension.

    sum_dim: str, optional
        Dimension to reduce via summation. Default is the last dimension -- dims[-1]. Ignored if len(dims) == 1

    coords: dict of str: list of str
        Dictionary of coordinates for the model, mapping dimension names to lists of labels.

    backend: str, one of 'numba' or 'pytensor'
        Backend that will parse the equations. Only relevant for aggregation functions like Sum and Prod. Elementwise
        operations are interoperable between backends.

    Returns
    -------
    (zero_profit_constraint, factor_demands): tuple of strings
        Equations representing the Leontief production function and resulting factor demands
    """

    _check_pairwise_lengths_match(
        ["factors", "factor_prices", "factor_shares"],
        [factors, factor_prices, factor_shares],
    )

    factors = at_least_list(factors)
    if isinstance(dims, str) or len(dims) == 1:
        if len(factors) == 1:
            raise ValueError(
                f"Leontief production function expects at least two factors when len(dims) == 1, found {len(factors)}"
            )
        return _1d_leontief(
            factors,
            factor_prices,
            output,
            output_price,
            factor_shares,
            dims,
            coords,
            backend,
        )
    else:
        if len(factors) != 1:
            raise ValueError(
                f"Leontief production function expects exactly one factor when len(dims) == 2, found {len(factors)}"
            )
        return _2d_leontief(
            factors=factors,
            factor_shares=factor_shares,
            output=output,
            output_price=output_price,
            factor_prices=factor_prices,
            dims=dims,
            coords=coords,
            sum_dim=sum_dim,
            expand_price_dim=expand_price_dim,
            transpose_output=transpose_output,
            backend=backend,
        )
