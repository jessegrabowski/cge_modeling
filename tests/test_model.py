import numpy as np
import pytest

from pytensor.graph import rewrite_graph
from pytensor.graph.traversal import explicit_graph_inputs

from cge_modeling import CGEModel, Equation, Variable
from cge_modeling import model as cm
from tests.utilities.models import (
    build_model_2_in_context,
    calibrate_model_2,
    load_model_2,
    model_2_data,
)


def evaluate(xtensor, values):
    """Evaluate ``xtensor`` given values for model objects, which lowering replaces by their tensors."""
    lowered = rewrite_graph(xtensor.values, include=("lower_xtensor",))
    return lowered.eval({obj.owner.inputs[0]: value for obj, value in values.items()})


def test_model_2_in_context_matches_string_model_objects_and_size():
    model, _ = build_model_2_in_context()
    string_model = load_model_2(backend="pytensor", functions_to_compile=())

    assert model.variable_names == string_model.variable_names
    assert model.parameter_names == string_model.parameter_names
    assert model.n_variables == model.n_equations == string_model.n_equations
    assert model.n_parameters == string_model.n_parameters


def test_model_2_in_context_residuals_vanish_at_the_calibrated_point():
    _, equations = build_model_2_in_context()
    calibrated = calibrate_model_2(**model_2_data)

    for equation in equations:
        lowered = rewrite_graph(equation.residual.values, include=("lower_xtensor",))
        inputs = {
            tensor: np.asarray(calibrated[tensor.name], dtype=float)
            for tensor in explicit_graph_inputs([lowered])
        }
        np.testing.assert_allclose(lowered.eval(inputs), 0.0, atol=1e-8, err_msg=equation.name)


def test_equation_residual_is_lhs_minus_rhs_in_coords_dim_order():
    with CGEModel(coords={"i": ["a", "b"], "j": ["x", "y", "z"]}):
        w = cm.Variable("w", dims="j")
        m = cm.Parameter("m", dims=("i", "j"))
        broadcast = cm.Equation("broadcast", w, m)

    m_values = np.arange(6.0).reshape(2, 3)
    w_values = np.array([10.0, 20.0, 30.0])
    residual = evaluate(broadcast.residual, {w: w_values, m: m_values})

    assert broadcast.residual.type.dims == ("i", "j")
    np.testing.assert_allclose(residual, w_values - m_values)


def test_equation_built_with_double_equals_raises():
    with pytest.raises(TypeError, match="`==` does not build equations"):
        with CGEModel(coords={"i": ["a", "b"]}):
            P = cm.Variable("P", dims="i")
            Q = cm.Variable("Q", dims="i")
            cm.Equation("clearing", P * 2 == Q, 0.0)


def test_object_registration_needs_an_active_model():
    with pytest.raises(RuntimeError, match="No active model"):
        cm.Variable("Y")


@pytest.mark.parametrize(
    "register, message",
    [
        (lambda: cm.Parameter("Y"), "already has a variable or parameter named 'Y'"),
        (lambda: cm.Variable("Z", dims="k"), r"dims \['k'\] that are not in the model coords"),
        (lambda: cm.Variable("i"), "'i' conflicts with an existing dimension name"),
    ],
    ids=["duplicate_name", "unknown_dim", "dim_name"],
)
def test_invalid_object_registration_raises(register, message):
    with pytest.raises(ValueError, match=message):
        with CGEModel(coords={"i": ["a"]}):
            cm.Variable("Y")
            register()


@pytest.mark.parametrize(
    "lhs, rhs, message",
    [
        (
            lambda P: P.rename(i="k"),
            lambda P: 1.0,
            r"dims \['k'\] that are not in the model coords",
        ),
        (lambda P: 1.0, lambda P: 2.0, "uses no variables or parameters of the model"),
    ],
    ids=["unknown_dim", "no_model_objects"],
)
def test_invalid_equation_raises(lhs, rhs, message):
    with pytest.raises(ValueError, match=message):
        with CGEModel(coords={"i": ["a", "b"]}):
            P = cm.Variable("P", dims="i")
            cm.Equation("bad", lhs(P), rhs(P))


def test_equation_using_another_models_object_raises():
    with pytest.raises(ValueError, match="uses inputs that are not objects of this model"):
        with CGEModel():
            outer_variable = cm.Variable("outer_variable")
            with CGEModel():
                inner_variable = cm.Variable("inner_variable")
                cm.Equation("inner", inner_variable, outer_variable)


def test_string_built_model_cannot_be_entered():
    model = CGEModel(variables=[Variable("x")], equations=[Equation("x equation", "x = 1")])

    with pytest.raises(ValueError, match="cannot be added to it in a `with` block"):
        with model:
            pass


def test_context_built_model_rejects_string_objects():
    with CGEModel() as model:
        x = cm.Variable("x")
        cm.Equation("x equation", x, 1.0)

    with pytest.raises(ValueError, match="this model is built in a `with` block"):
        model.add_variable(Variable("y"))


def test_model_can_be_built_across_several_blocks():
    model = CGEModel(coords={"i": ["a", "b"]})

    with model:
        P = cm.Variable("P", dims="i")
    with model:
        cm.Equation("prices", P, 1.0)

    assert model.n_variables == model.n_equations == 2


def test_objects_register_on_an_explicit_model_outside_a_block():
    model = CGEModel(coords={"i": ["a", "b"]})

    P = cm.Variable("P", dims="i", model=model)
    cm.Equation("prices", P, 1.0, model=model)

    assert model.variable_names == ["P"]
    assert model.equation_names == ["prices"]


def test_registered_objects_are_retrievable_by_name():
    with CGEModel(coords={"i": ["a", "b"]}) as model:
        P = cm.Variable("P", dims="i")

    assert model["P"] is P
    with pytest.raises(KeyError, match="'Q' is not a variable or parameter registered"):
        model["Q"]


def test_reentering_a_model_inside_its_own_block_keeps_it_active():
    with CGEModel() as model:
        with model:
            x = cm.Variable("x")
        cm.Equation("x equation", x, 1.0)

    assert model.n_variables == model.n_equations == 1
    assert model.equation_names == ["x equation"]


def test_nested_contexts_register_on_the_innermost_model():
    with CGEModel() as outer:
        with CGEModel() as inner:
            cm.Variable("inner_variable")
        cm.Variable("outer_variable")

    assert inner.variable_names == ["inner_variable"]
    assert outer.variable_names == ["outer_variable"]
