from typing import TYPE_CHECKING, Any

from cge_modeling.base.context import ModelObjectVariable, XTensorEquation, modelcontext

if TYPE_CHECKING:
    from cge_modeling.base.cge import CGEModel


def Variable(
    name: str,
    dims: str | list[str] | tuple[str, ...] | None = None,
    *,
    description: str | None = None,
    latex_name: str | None = None,
    extend_subscript: bool | int = False,
    model: "CGEModel | None" = None,
) -> ModelObjectVariable:
    """Create an endogenous variable and register it on ``model``, or on the active :class:`CGEModel`."""
    return modelcontext(model)._register_model_object(
        name=name,
        dims=dims,
        kind="variable",
        description=description,
        latex_name=latex_name,
        extend_subscript=extend_subscript,
    )


def Parameter(
    name: str,
    dims: str | list[str] | tuple[str, ...] | None = None,
    *,
    description: str | None = None,
    latex_name: str | None = None,
    extend_subscript: bool | int = False,
    model: "CGEModel | None" = None,
) -> ModelObjectVariable:
    """Create a parameter and register it on ``model``, or on the active :class:`CGEModel`."""
    return modelcontext(model)._register_model_object(
        name=name,
        dims=dims,
        kind="parameter",
        description=description,
        latex_name=latex_name,
        extend_subscript=extend_subscript,
    )


def Equation(name: str, lhs: Any, rhs: Any, *, model: "CGEModel | None" = None) -> XTensorEquation:
    """Register ``lhs = rhs`` on ``model``, or on the active :class:`CGEModel`. Its residual is ``lhs - rhs``."""
    return modelcontext(model)._register_model_equation(name=name, lhs=lhs, rhs=rhs)
