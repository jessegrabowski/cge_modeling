from string import Template

from cge_modeling.base.primitives import Variable
from cge_modeling.base.utilities import _expand_var_by_index


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
    data_dict = {
        "Y": np.array([1, 2, 3]),
        "phi": np.array([[1, 2, 3], [4, 5, 6]]),
        "L_s": np.array([3.0]),
    }
