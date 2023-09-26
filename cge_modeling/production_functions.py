import sympy as sp


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
