from string import Template

import numpy as np
import pytest

from cge_modeling.base.primitives import Parameter, Variable
from cge_modeling.base.utilities import (
    expand_obj_by_indices,
    flat_array_to_variable_dict,
    get_method_defaults,
    unpack_equation_name,
    variable_dict_to_flat_array,
)


def test_expand_obj_by_indices():
    coords = {"i": ["A", "B", "C"]}
    x = Variable(name="x", dims="i", description="Sector <dim:i> demand for good <dim:j>")
    expanded = expand_obj_by_indices(x, coords)
    assert len(expanded) == len(coords["i"])
    for variable, coord in zip(expanded, coords["i"]):
        assert variable._full_latex_name == "x_{i=\\text{" + coord + "}}"
        assert variable.description == f"Sector {coord} demand for good <dim:j>"


def test_expand_obj_by_indices_two_dims():
    coords = {"i": ["A", "B", "C"], "j": ["A", "B", "C"]}
    x = Variable(name="x", dims="i, j", description="Sector i demand for good j")
    expanded = expand_obj_by_indices(x, coords)

    assert len(expanded) == len(coords["i"]) * len(coords["j"])
    all_latex = [variable._full_latex_name for variable in expanded]
    s = Template("x_{i=\\text{$i}, j=\\text{$j}}")
    for i in coords["i"]:
        for j in coords["j"]:
            assert s.substitute(i=i, j=j) in all_latex


def test_pack_and_unpack_is_bijective():
    variables = [
        Variable("Y", dims="i", description="<dim:i> output"),
    ]
    parameters = [
        Parameter(
            "phi",
            dims=("j", "i"),
            description="Input-output coefficient between <dim:i> and <dim:j>",
        ),
        Parameter("L_s", description="Total labor supply"),
    ]

    data_dict = {
        "Y": np.array([1, 2, 3]),
        "phi": np.array([[1, 2, 3], [4, 5, 6]]),
        "L_s": np.array(3.0),
    }

    coords = {"i": ["A", "B", "C"], "j": ["Q", "R"]}

    variable_array, param_array = variable_dict_to_flat_array(data_dict, variables, parameters)

    assert variable_array.shape[0] == sum(np.prod(data_dict[x.name].shape) for x in variables)
    assert param_array.shape[0] == sum(np.prod(data_dict[x.name].shape) for x in parameters)

    data_dict_2 = flat_array_to_variable_dict(
        np.concatenate([variable_array, param_array]), variables + parameters, coords
    )

    assert all(
        [data_dict[x.name].shape == data_dict_2[x.name].shape for x in variables + parameters]
    )
    assert all(
        [np.allclose(data_dict[x.name], data_dict_2[x.name]) for x in variables + parameters]
    )


@pytest.mark.parametrize(
    "name, dims, expected",
    [
        ("Numeraire", (), ["Numeraire"]),
        ("Demand for <dim:i>", ("i",), ["Demand for A", "Demand for B"]),
        ("Clearing", ("i",), ["Clearing [i=A]", "Clearing [i=B]"]),
        (
            "Sector <dim:i> inputs",
            ("i", "j"),
            [
                "Sector A inputs [j=x]",
                "Sector A inputs [j=y]",
                "Sector B inputs [j=x]",
                "Sector B inputs [j=y]",
            ],
        ),
    ],
    ids=["scalar", "marked", "unmarked", "partly_marked"],
)
def test_unpack_equation_name(name, dims, expected):
    coords = {"i": ["A", "B"], "j": ["x", "y"]}

    assert unpack_equation_name(name, dims, coords) == expected


@pytest.mark.parametrize(
    "method, use_hessp, expected",
    [
        ("trust-ncg", None, (True, False, True)),
        ("trust-ncg", False, (True, True, False)),
        ("BFGS", None, (True, False, False)),
        ("nelder-mead", None, (False, False, False)),
    ],
    ids=[
        "hessp_preferred_over_hess",
        "hess_when_hessp_declined",
        "gradient_only",
        "derivative_free",
    ],
)
def test_get_method_defaults_reads_the_method_and_prefers_hessp(method, use_hessp, expected):
    assert get_method_defaults(None, None, use_hessp, method) == expected
