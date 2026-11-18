import numpy as np
import pandas as pd
import pytest

from pytensor.graph import rewrite_graph

from cge_modeling import CGEModel
from cge_modeling import model as cm
from cge_modeling.base.context import modelcontext


def evaluate(xtensor, values):
    """Evaluate ``xtensor`` given values for model objects, which lowering replaces by their tensors."""
    lowered = rewrite_graph(xtensor.values, include=("lower_xtensor",))
    return lowered.eval({obj.owner.inputs[0]: value for obj, value in values.items()})


def test_modelcontext_returns_the_innermost_model():
    with CGEModel() as outer:
        assert modelcontext() is outer
        with CGEModel() as inner:
            assert modelcontext() is inner
        assert modelcontext() is outer


def test_modelcontext_without_an_active_model_raises():
    with pytest.raises(RuntimeError, match="No active model"):
        modelcontext()


@pytest.mark.parametrize(
    "labels, expected",
    [
        ("Service", 3.0),
        (["Service", "Agriculture"], [3.0, 1.0]),
        (np.array(["Service", "Agriculture"]), [3.0, 1.0]),
        (pd.Index(["Service", "Agriculture"]), [3.0, 1.0]),
    ],
    ids=["label", "list", "array", "index"],
)
def test_sel_maps_labels_to_positions_through_the_active_model(labels, expected):
    with CGEModel(coords={"i": ["Agriculture", "Industry", "Service"]}):
        P = cm.Variable("P", dims="i")
        selected = P.sel(i=labels)

    np.testing.assert_allclose(evaluate(selected, {P: np.array([1.0, 2.0, 3.0])}), expected)


def test_sel_outside_a_model_context_raises():
    with CGEModel(coords={"i": ["a", "b"]}):
        P = cm.Variable("P", dims="i")

    with pytest.raises(RuntimeError, match="No active model"):
        P.sel(i="a")


def test_sel_with_an_unknown_label_raises():
    with pytest.raises(KeyError, match="'c' is not a label of dimension 'i'"):
        with CGEModel(coords={"i": ["a", "b"]}):
            P = cm.Variable("P", dims="i")
            P.sel(i="c")


def test_sel_on_an_unknown_dimension_raises():
    with pytest.raises(ValueError, match="Unknown dimension 'k'"):
        with CGEModel(coords={"i": ["a", "b"]}):
            P = cm.Variable("P", dims="i")
            P.sel(k="a")
