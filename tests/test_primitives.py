import pytest

from cge_modeling.base.primitives import Parameter, Variable


@pytest.mark.parametrize("cls", [Variable, Parameter], ids=["variable", "parameter"])
@pytest.mark.parametrize(
    "name, latex_name, dims, expected",
    [
        ("x", "x", None, ("x", "x, Positive = True, Real = True", ())),
        ("x", "x", ("i",), ("x_{i}", "x_{i}, Positive = True, Real = True", ("i",))),
        (
            "x",
            r"\Omega",
            "i",
            (r"\Omega_{i}", r"\Omega_{i}, Positive = True, Real = True", ("i",)),
        ),
        (
            "x",
            "x",
            "i,j",
            ("x_{i, j}", "x_{i, j}, Positive = True, Real = True", ("i", "j")),
        ),
        (
            "x_d",
            "x_d",
            "i,j",
            ("x_{d, i, j}", "x_{d, i, j}, Positive = True, Real = True", ("i", "j")),
        ),
    ],
    ids=[
        "No_dims",
        "tuple_dims",
        "single_string",
        "double_string",
        "underscore_name",
    ],
)
def test_create_variable_defaults(cls, name, latex_name, dims, expected):
    # noinspection PyArgumentList
    x = cls(name=name, latex_name=latex_name, dims=dims, extend_subscript=True)
    expected_latex_name, description, dims = expected

    assert x._full_latex_name == expected_latex_name
    assert x.description == description
    assert x.dims == dims


@pytest.mark.parametrize("cls", [Variable, Parameter], ids=["variable", "parameter"])
def test_create_variable(cls):
    # noinspection PyArgumentList
    x = cls(name="x", dims="i", description="A lovely variable")

    # noinspection PyArgumentList
    y = cls(
        name="y",
        dims="i",
        description="Another lovely variable",
    )

    assert x._full_latex_name == r"x_{i}"
    assert x.description == "A lovely variable"
    assert x.dims == ("i",)
    assert y > x


@pytest.mark.parametrize("cls", [Variable, Parameter], ids=["variable", "parameter"])
def test_to_dict(cls):
    # noinspection PyArgumentList
    x = cls(name="x", dims="i", description="A lovely thing")
    d = x.to_dict()

    assert isinstance(d, dict)
    assert len(d) == 6
    assert all([key in d.keys() for key in ["name", "dims", "dim_vals", "description"]])


@pytest.mark.parametrize(
    "base_name, dims, extend, expected",
    [
        ("x", ["i"], False, "x_{i=\\text{A}}"),
        ("x^j", ["i"], False, "x^j_{i=\\text{A}}"),
        ("x^{j}", ["i"], False, "x^{j}_{i=\\text{A}}"),
        ("x_F", ["i", "j"], True, "x_{F, i=\\text{A}, j}"),
        ("x_{Fish}", ["i", "i"], True, "x_{Fish, i=\\text{A}, i=\\text{A}}"),
        ("x_K_d", ["i", "j"], 2, "x_{K, d, i=\\text{A}, j}"),
        ("var_with_underscore", ["i"], False, "var_with_underscore_{i=\\text{A}}"),
    ],
)
def test_sub_label(base_name, dims, extend, expected):
    x = Variable(
        name=base_name,
        dims=dims,
        description="A lovely item from group <dim:i>",
        extend_subscript=extend,
    )
    x.update_dim_value("i", "A")
    assert x._full_latex_name == expected


@pytest.mark.parametrize(
    "description, expected",
    [
        ("A lovely item from group <dim:i>", "A lovely item from group A"),
        ("Sector <dim:i> demand for good <dim:j>", "Sector A demand for good <dim:j>"),
        ("<dim:i> demand for <dim:j>", "A demand for <dim:j>"),
    ],
)
def test_sub_description(description, expected):
    x = Variable(name="x", dims=["i", "j"], description=description)
    x.update_dim_value("i", "A")
    assert x.description == expected
