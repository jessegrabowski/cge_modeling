from string import Template

import numpy as np

from cge_modeling.base.primitives import Parameter, Variable
from cge_modeling.base.utilities import (
    _expand_var_by_index,
    flat_array_to_variable_dict,
    variable_dict_to_flat_array,
)


def test_expand_variable():
    coords = {"i": ["A", "B", "C"]}
    x = Variable(name="x", dims="i", description="Sector <dim:i> demand for good <dim:j>")
    vars = _expand_var_by_index(x, coords)
    assert len(vars) == len(coords["i"])
    for variable, coord in zip(vars, coords["i"]):
        assert variable._full_latex_name == "x_{i=\\text{" + coord + "}}"
        assert variable.description == f"Sector {coord} demand for good <dim:j>"


def test_expand_variable_two_index():
    coords = {"i": ["A", "B", "C"], "j": ["A", "B", "C"]}
    x = Variable(name="x", dims="i, j", description="Sector i demand for good j")
    vars = _expand_var_by_index(x, coords)

    assert len(vars) == len(coords["i"]) * len(coords["j"])
    all_latex = [x._full_latex_name for x in vars]
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
