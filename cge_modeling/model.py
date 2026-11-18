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
    model: "CGEModel | None" = None,
) -> ModelObjectVariable:
    """Create an endogenous variable and register it on ``model``, or on the active :class:`CGEModel`."""
    return modelcontext(model)._register_model_object(
        name, dims, "variable", description, latex_name
    )


def Parameter(
    name: str,
    dims: str | list[str] | tuple[str, ...] | None = None,
    *,
    description: str | None = None,
    latex_name: str | None = None,
    model: "CGEModel | None" = None,
) -> ModelObjectVariable:
    """Create a parameter and register it on ``model``, or on the active :class:`CGEModel`."""
    return modelcontext(model)._register_model_object(
        name, dims, "parameter", description, latex_name
    )


def Equation(name: str, lhs: Any, rhs: Any, *, model: "CGEModel | None" = None) -> XTensorEquation:
    """Register ``lhs = rhs`` on ``model``, or on the active :class:`CGEModel`. Its residual is ``lhs - rhs``."""
    return modelcontext(model)._register_model_equation(name, lhs, rhs)
