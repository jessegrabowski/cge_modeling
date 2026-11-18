from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pytensor.xtensor.type import XTensorVariable

if TYPE_CHECKING:
    from cge_modeling.base.cge import CGEModel

model_stack: ContextVar[tuple["CGEModel", ...]] = ContextVar("cge_model_stack", default=())


def modelcontext(model: "CGEModel | None" = None) -> "CGEModel":
    """Return ``model`` if given, otherwise the innermost :class:`CGEModel` whose ``with`` block is active."""
    if model is not None:
        return model
    stack = model_stack.get()
    if not stack:
        raise RuntimeError(
            "No active model. Create model objects inside a `with CGEModel(coords=...):` block."
        )
    return stack[-1]


class ModelObjectVariable(XTensorVariable):
    """An xtensor model variable or parameter whose labels resolve through the active model's coords."""

    def sel(self, indexers: Mapping[str, Any] | None = None, **indexers_kwargs) -> XTensorVariable:
        """
        Select by coordinate label, like :meth:`xarray.DataArray.sel`.

        Only model objects have this method. Results of operations on them are plain xtensor variables, so select
        from those with ``isel`` and positions from :meth:`CGEModel.label_positions`.
        """
        model = modelcontext()
        indexers = {**(indexers or {}), **indexers_kwargs}
        positions = {dim: model.label_positions(dim, labels) for dim, labels in indexers.items()}
        return self.isel(positions)


@dataclass(frozen=True)
class XTensorEquation:
    name: str
    lhs: XTensorVariable
    rhs: XTensorVariable
    residual: XTensorVariable
