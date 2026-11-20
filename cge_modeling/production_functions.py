from collections.abc import Sequence
from functools import reduce
from operator import add, mul
from typing import TYPE_CHECKING

from pytensor.xtensor.type import XTensorVariable

from cge_modeling.base.context import XTensorEquation, modelcontext
from cge_modeling.model import Equation

if TYPE_CHECKING:
    from cge_modeling.base.cge import CGEModel

EquationSpec = tuple[str, XTensorVariable, XTensorVariable]


def _check_one_per_factor(factors: Sequence[XTensorVariable], **per_factor: Sequence) -> None:
    for label, values in per_factor.items():
        if len(values) != len(factors):
            raise ValueError(
                f"Expected one {label} per factor, found {len(factors)} factors and {len(values)} {label}s."
            )


def _check_factors_named(factors: Sequence[XTensorVariable]) -> None:
    unnamed = [position for position, factor in enumerate(factors) if factor.name is None]
    if unnamed:
        raise ValueError(
            f"Factors at positions {unnamed} have no name. Pass model objects, not expressions built from them."
        )


def _complete_factor_shares(
    factors: Sequence[XTensorVariable],
    factor_prices: Sequence[XTensorVariable],
    factor_shares: Sequence[XTensorVariable | float],
) -> list[XTensorVariable | float]:
    """
    Check that there is one price per factor, and return one share per factor.

    With one share fewer than factors, the last share is one minus the sum of the others.
    """
    _check_one_per_factor(factors, price=factor_prices)
    n_factors = len(factors)
    if len(factor_shares) == n_factors:
        return list(factor_shares)
    if len(factor_shares) == n_factors - 1:
        return [*factor_shares, 1 - reduce(add, factor_shares, 0.0)]
    raise ValueError(
        f"Expected {n_factors} factor shares, or {n_factors - 1} with the last implied, found {len(factor_shares)}."
    )


def _register_equations(
    model: "CGEModel", specs: Sequence[EquationSpec]
) -> tuple[XTensorEquation, ...]:
    """Register ``(name, lhs, rhs)`` specs on ``model`` after checking that every name is free."""
    names = [name for name, _, _ in specs]
    taken = [name for name in names if name in model.equation_names]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if taken or repeated:
        raise ValueError(
            f"Cannot register the block: equation names {taken + repeated} are already taken or repeated."
        )
    return tuple(Equation(name, lhs, rhs, model=model) for name, lhs, rhs in specs)


def _quantity_spec(
    name: str,
    relation: str,
    quantity: XTensorVariable,
    price: XTensorVariable,
    value: XTensorVariable,
) -> EquationSpec:
    """
    Write the equation for ``quantity`` in value form, ``price * quantity = value``, named
    ``"{name}: {relation} {quantity}"``.

    In value form each demand or supply enters Walras' law with a constant weight, so an Euler step, which holds
    every linearized residual at zero, also holds the Walrasian residual at zero when prices are variables.
    """
    return (f"{name}: {relation} {quantity.name}", price * quantity, value)


def _ces_quantity_specs(
    name: str,
    relation: str,
    output: XTensorVariable,
    output_price: XTensorVariable,
    quantities: Sequence[XTensorVariable],
    prices: Sequence[XTensorVariable],
    shares: Sequence[XTensorVariable | float],
    tfp: XTensorVariable | float,
    epsilon: XTensorVariable | float,
) -> list[EquationSpec]:
    return [
        _quantity_spec(
            name=name,
            relation=relation,
            quantity=quantity,
            price=price,
            value=price * output / tfp * (share * output_price * tfp / price) ** epsilon,
        )
        for quantity, price, share in zip(quantities, prices, shares, strict=True)
    ]


def _check_without_factor_dim(factor_dim: str, **objects: XTensorVariable) -> None:
    for label, value in objects.items():
        if factor_dim in value.type.dims:
            raise ValueError(
                f"{label} has the factor dim {factor_dim!r}, which the block sums over."
            )


def CobbDouglas(
    name: str,
    *,
    output: XTensorVariable,
    output_price: XTensorVariable,
    factors: Sequence[XTensorVariable],
    factor_prices: Sequence[XTensorVariable],
    factor_shares: Sequence[XTensorVariable | float],
    tfp: XTensorVariable | float,
    model: "CGEModel | None" = None,
) -> tuple[XTensorEquation, ...]:
    r"""
    Register a Cobb-Douglas production block on ``model``, or on the active :class:`CGEModel`.

    The block is the technology and one cost-minimizing demand per factor:

    .. math::

        Y = A \prod_i X_i^{\alpha_i}, \qquad P_i X_i = \alpha_i P_Y Y

    Parameters
    ----------
    name : str
        Name of the production equation. Each demand equation is named ``"{name}: demand for {factor}"``.
    output, output_price : XTensorVariable
        The output :math:`Y` and its price :math:`P_Y`.
    factors, factor_prices : sequence of XTensorVariable
        The factors :math:`X_i`, which must be model objects, and their prices :math:`P_i`, in matching order.
    factor_shares : sequence of XTensorVariable or float
        The shares :math:`\alpha_i`, one per factor. With one share fewer than factors, the last share is one
        minus the sum of the others.
    tfp : XTensorVariable or float
        Total factor productivity :math:`A`.
    model : CGEModel, optional
        Model to register the equations on. Defaults to the model of the enclosing ``with`` block.

    Returns
    -------
    equations : tuple of XTensorEquation
        The production equation, then one demand equation per factor, in factor order.
    """
    model = modelcontext(model)
    _check_factors_named(factors)
    factor_shares = _complete_factor_shares(factors, factor_prices, factor_shares)

    technology = tfp * reduce(
        mul, [factor**share for factor, share in zip(factors, factor_shares, strict=True)]
    )
    demands = [
        _quantity_spec(
            name=name,
            relation="demand for",
            quantity=factor,
            price=price,
            value=share * output * output_price,
        )
        for factor, price, share in zip(factors, factor_prices, factor_shares, strict=True)
    ]
    return _register_equations(model, [(name, output, technology), *demands])


def CES(
    name: str,
    *,
    output: XTensorVariable,
    output_price: XTensorVariable,
    factors: Sequence[XTensorVariable],
    factor_prices: Sequence[XTensorVariable],
    factor_shares: Sequence[XTensorVariable | float],
    tfp: XTensorVariable | float,
    epsilon: XTensorVariable | float,
    model: "CGEModel | None" = None,
) -> tuple[XTensorEquation, ...]:
    r"""
    Register a CES production block on ``model``, or on the active :class:`CGEModel`.

    The block is the zero-profit condition and one cost-minimizing demand per factor:

    .. math::

        P_Y Y = \sum_i P_i X_i, \qquad P_i X_i = P_i \frac{Y}{A} \left( \frac{\alpha_i P_Y A}{P_i} \right)^{\epsilon}

    Together these hold only when :math:`P_Y` is the CES unit cost, so the zero-profit condition pins the output
    price, and the demands imply the technology
    :math:`Y = A \left( \sum_i \alpha_i X_i^{(\epsilon - 1) / \epsilon} \right)^{\epsilon / (\epsilon - 1)}`.

    Parameters
    ----------
    name : str
        Name of the zero-profit equation. Each demand equation is named ``"{name}: demand for {factor}"``.
    output, output_price : XTensorVariable
        The output :math:`Y` and its price :math:`P_Y`.
    factors, factor_prices : sequence of XTensorVariable
        The factors :math:`X_i`, which must be model objects, and their prices :math:`P_i`, in matching order.
    factor_shares : sequence of XTensorVariable or float
        The shares :math:`\alpha_i`, one per factor. With one share fewer than factors, the last share is one
        minus the sum of the others.
    tfp : XTensorVariable or float
        Total factor productivity :math:`A`.
    epsilon : XTensorVariable or float
        Elasticity of substitution :math:`\epsilon`.
    model : CGEModel, optional
        Model to register the equations on. Defaults to the model of the enclosing ``with`` block.

    Returns
    -------
    equations : tuple of XTensorEquation
        The zero-profit equation, then one demand equation per factor, in factor order.
    """
    model = modelcontext(model)
    _check_factors_named(factors)
    factor_shares = _complete_factor_shares(factors, factor_prices, factor_shares)

    factor_cost = reduce(
        add, [price * factor for factor, price in zip(factors, factor_prices, strict=True)]
    )
    demands = _ces_quantity_specs(
        name=name,
        relation="demand for",
        output=output,
        output_price=output_price,
        quantities=factors,
        prices=factor_prices,
        shares=factor_shares,
        tfp=tfp,
        epsilon=epsilon,
    )
    return _register_equations(model, [(name, output_price * output, factor_cost), *demands])


def CET(
    name: str,
    *,
    output: XTensorVariable,
    output_price: XTensorVariable,
    products: Sequence[XTensorVariable],
    product_prices: Sequence[XTensorVariable],
    product_shares: Sequence[XTensorVariable | float],
    tfp: XTensorVariable | float,
    omega: XTensorVariable | float,
    model: "CGEModel | None" = None,
) -> tuple[XTensorEquation, ...]:
    r"""
    Register a CET transformation block on ``model``, or on the active :class:`CGEModel`.

    The aggregate output :math:`X` is transformed into products :math:`X_i`, such as exports and domestic sales. The
    block is the zero-profit condition and one revenue-maximizing supply per product:

    .. math::

        P_X X = \sum_i P_i X_i, \qquad
        P_i X_i = P_i \frac{X}{A} \left( \frac{P_i}{\alpha_i P_X A} \right)^{\Omega}

    Together these hold only when :math:`P_X` is the CET unit revenue, so the zero-profit condition pins the
    aggregate price, and the supplies imply the frontier
    :math:`X = A \left( \sum_i \alpha_i X_i^{(\Omega + 1) / \Omega} \right)^{\Omega / (\Omega + 1)}`.

    Parameters
    ----------
    name : str
        Name of the zero-profit equation. Each supply equation is named ``"{name}: supply of {product}"``.
    output, output_price : XTensorVariable
        The aggregate output :math:`X` and its price :math:`P_X`.
    products, product_prices : sequence of XTensorVariable
        The products :math:`X_i`, which must be model objects, and their prices :math:`P_i`, in matching order.
    product_shares : sequence of XTensorVariable or float
        The shares :math:`\alpha_i`, one per product. With one share fewer than products, the last share is one
        minus the sum of the others.
    tfp : XTensorVariable or float
        Scale of the frontier :math:`A`.
    omega : XTensorVariable or float
        Elasticity of transformation :math:`\Omega`, which is positive.
    model : CGEModel, optional
        Model to register the equations on. Defaults to the model of the enclosing ``with`` block.

    Returns
    -------
    equations : tuple of XTensorEquation
        The zero-profit equation, then one supply equation per product, in product order.
    """
    model = modelcontext(model)
    _check_factors_named(products)
    product_shares = _complete_factor_shares(products, product_prices, product_shares)

    revenue = reduce(
        add, [price * product for product, price in zip(products, product_prices, strict=True)]
    )
    supplies = _ces_quantity_specs(
        name=name,
        relation="supply of",
        output=output,
        output_price=output_price,
        quantities=products,
        prices=product_prices,
        shares=product_shares,
        tfp=tfp,
        epsilon=-omega,
    )
    return _register_equations(model, [(name, output_price * output, revenue), *supplies])


def DixitStiglitz(
    name: str,
    *,
    output: XTensorVariable,
    output_price: XTensorVariable,
    factor: XTensorVariable,
    factor_price: XTensorVariable,
    factor_dim: str,
    epsilon: XTensorVariable | float,
    factor_share: XTensorVariable | float = 1.0,
    tfp: XTensorVariable | float = 1.0,
    model: "CGEModel | None" = None,
) -> tuple[XTensorEquation, ...]:
    r"""
    Register a Dixit-Stiglitz aggregate on ``model``, or on the active :class:`CGEModel`.

    The varieties are the entries of ``factor`` along ``factor_dim``. The block is the zero-profit condition and the
    demand for each variety:

    .. math::

        P_Y Y = \sum_i P_i X_i, \qquad P_i X_i = P_i \frac{Y}{A} \left( \frac{\alpha_i P_Y A}{P_i} \right)^{\epsilon}

    Together these hold only when :math:`P_Y` is the CES price index, so the zero-profit condition pins the
    aggregate price, and the demands imply the aggregate
    :math:`Y = A \left( \sum_i \alpha_i X_i^{(\epsilon - 1) / \epsilon} \right)^{\epsilon / (\epsilon - 1)}`.

    Parameters
    ----------
    name : str
        Name of the zero-profit equation. The demand equation is named ``"{name}: demand for {factor}"``.
    output, output_price : XTensorVariable
        The aggregate :math:`Y` and its price :math:`P_Y`. Neither may have ``factor_dim``.
    factor, factor_price : XTensorVariable
        The varieties :math:`X_i`, a model object, and their prices :math:`P_i`, indexed by ``factor_dim``.
    factor_dim : str
        The dim along which ``factor`` lists the varieties.
    epsilon : XTensorVariable or float
        Elasticity of substitution :math:`\epsilon`.
    factor_share : XTensorVariable or float, optional
        The shares :math:`\alpha_i`. Default is 1, which with the default ``tfp`` gives the elasticity-only form.
    tfp : XTensorVariable or float, optional
        Total factor productivity :math:`A`. Default is 1.
    model : CGEModel, optional
        Model to register the equations on. Defaults to the model of the enclosing ``with`` block.

    Returns
    -------
    equations : tuple of XTensorEquation
        The zero-profit equation, then the demand equation.
    """
    model = modelcontext(model)
    _check_factors_named([factor])
    _check_without_factor_dim(factor_dim, output=output, output_price=output_price)

    demands = _ces_quantity_specs(
        name=name,
        relation="demand for",
        output=output,
        output_price=output_price,
        quantities=[factor],
        prices=[factor_price],
        shares=[factor_share],
        tfp=tfp,
        epsilon=epsilon,
    )
    zero_profit = (name, output_price * output, (factor_price * factor).sum(factor_dim))
    return _register_equations(model, [zero_profit, *demands])


def Leontief(
    name: str,
    *,
    output: XTensorVariable,
    output_price: XTensorVariable,
    factors: Sequence[XTensorVariable],
    factor_prices: Sequence[XTensorVariable],
    factor_shares: Sequence[XTensorVariable | float],
    factor_dim: str | Sequence[str | None] | None = None,
    model: "CGEModel | None" = None,
) -> tuple[XTensorEquation, ...]:
    r"""
    Register a Leontief production block on ``model``, or on the active :class:`CGEModel`.

    The block is the zero-profit condition and one demand per factor:

    .. math::

        P_Y Y = \sum_i P_i X_i, \qquad P_i X_i = P_i \alpha_i Y

    The technology :math:`Y = \min_i X_i / \alpha_i` is not registered, because it holds whenever the demands do.

    Parameters
    ----------
    name : str
        Name of the zero-profit equation. Each demand equation is named ``"{name}: demand for {factor}"``.
    output, output_price : XTensorVariable
        The output :math:`Y` and its price :math:`P_Y`.
    factors, factor_prices : sequence of XTensorVariable
        The factors :math:`X_i`, which must be model objects, and their prices :math:`P_i`, in matching order.
    factor_shares : sequence of XTensorVariable or float
        The input-output coefficients :math:`\alpha_i`, one per factor.
    factor_dim : str, sequence of str or None, or None, optional
        The dim of each factor that indexes its inputs, summed over in the zero-profit condition. For an
        input-output matrix ``X(i, j)`` with output ``Y(j)``, pass ``factors=[X]`` and ``factor_dim="i"``. A
        single str or None applies to every factor; a sequence gives one entry per factor, so goods inputs
        ``X(i, j)`` and energy inputs ``X_E(k, j)`` enter one block as ``factors=[X, X_E]`` and
        ``factor_dim=["i", "k"]``. Every dim of a factor cost that is not summed must be a dim of the output.
        Default is None, for factors that each enter as one input.
    model : CGEModel, optional
        Model to register the equations on. Defaults to the model of the enclosing ``with`` block.

    Returns
    -------
    equations : tuple of XTensorEquation
        The zero-profit equation, then one demand equation per factor, in factor order.
    """
    model = modelcontext(model)
    _check_factors_named(factors)
    _check_one_per_factor(factors, price=factor_prices, share=factor_shares)
    if factor_dim is None or isinstance(factor_dim, str):
        factor_dims = [factor_dim] * len(factors)
    else:
        factor_dims = list(factor_dim)
        _check_one_per_factor(factors, factor_dim=factor_dims)

    output_value = output_price * output
    factor_costs = []
    for factor, price, dim in zip(factors, factor_prices, factor_dims, strict=True):
        cost = price * factor
        if dim is not None:
            if dim not in cost.type.dims:
                raise ValueError(f"Factor {factor.name} has no dim {dim!r} to sum over.")
            cost = cost.sum(dim)
        extra_dims = [d for d in cost.type.dims if d not in output_value.type.dims]
        if extra_dims:
            raise ValueError(
                f"The cost of factor {factor.name} has dims {extra_dims} that the output does not; pass them as "
                "its factor_dim."
            )
        factor_costs.append(cost)

    demands = [
        _quantity_spec(
            name=name,
            relation="demand for",
            quantity=factor,
            price=price,
            value=price * share * output,
        )
        for factor, price, share in zip(factors, factor_prices, factor_shares, strict=True)
    ]
    return _register_equations(model, [(name, output_value, reduce(add, factor_costs)), *demands])
