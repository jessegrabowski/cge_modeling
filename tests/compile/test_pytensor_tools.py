import pytest

from cge_modeling.base.primitives import Parameter, Variable
from cge_modeling.compile.pytensor import cge_primitives_to_pytensor
from cge_modeling.compile.pytensor_tools import make_jacobian, object_to_pytensor
from tests.utilities.models import load_model_1, load_model_2

test_cases = [
    (Parameter, "K_d", {}),
    (Variable, "Y", {"t": [0, 1, 2]}),
    (Variable, "Y", {"t": [0, 1, 2], "s": [0, 1, 2]}),
]


@pytest.mark.parametrize(
    "cls, name, coords", test_cases, ids=["Parameter", "Variable", "Variable2d"]
)
def test_object_to_pytensor(cls, name, coords):
    # noinspection PyArgumentList
    obj = cls(
        name=name,
        dims=list(coords.keys()),
        description="A lovely item from group <dim:i>",
    )

    pt_obj = object_to_pytensor(obj, coords)
    assert pt_obj.name == name
    assert pt_obj.ndim == len(coords)
    assert pt_obj.type.shape == tuple(len(coords[dim]) for dim in coords.keys())


@pytest.mark.parametrize(
    "model_func, kwargs",
    [
        (
            load_model_1,
            {"backend": "sympytensor"},
        ),
        (
            load_model_2,
            {
                "backend": "pytensor",
            },
        ),
    ],
    ids=["1_Sector", "3_Sector"],
)
def test_compile_to_pytensor(model_func, kwargs):
    cge_model = model_func(**kwargs)
    n_variables = n_eq = len(cge_model.unpacked_variable_names)
    n_params = len(cge_model.unpacked_parameter_names)

    system, variables, parameters, _ = cge_primitives_to_pytensor(cge_model)
    jac = make_jacobian(system, variables)
    B = make_jacobian(system, parameters)

    assert system.type.shape == (n_eq,)
    assert jac.type.shape == (n_eq, n_variables)
    assert B.type.shape == (n_eq, n_params)


def test_prod_to_no_zero_prod_rewrite():
    import pytensor
    import pytensor.tensor as pt

    x = pt.dvector("x")
    z = pt.prod(x)

    assert not z.owner.op.no_zeros_in_input
    f = pytensor.function([x], z)
    assert any([isinstance(node.op, pt.math.Prod) for node in f.maker.fgraph.toposort()])

    for node in f.maker.fgraph.toposort():
        if isinstance(node.op, pt.math.Prod):
            assert node.op.no_zeros_in_input
            break
