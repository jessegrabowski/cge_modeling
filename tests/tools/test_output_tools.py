import numpy as np
import pytest
import xarray as xr

from IPython.display import Latex

from cge_modeling.tools.output_tools import datatree_to_value_dict, index_by_parameter, make_table


def test_datatree_to_value_dict_merges_variables_and_parameters():
    tree = xr.DataTree.from_dict(
        {
            "variables": xr.Dataset({"Y": ("i", [1.0, 2.0]), "C": 3.0}),
            "parameters": xr.Dataset({"alpha": 0.5}),
        }
    )

    values = datatree_to_value_dict(tree)

    assert set(values) == {"Y", "C", "alpha"}
    np.testing.assert_array_equal(values["Y"], [1.0, 2.0])
    assert values["C"] == 3.0
    assert values["alpha"] == 0.5
    assert all(isinstance(value, np.ndarray) for value in values.values())


def path_tree(labor_supply):
    steps = np.arange(len(labor_supply))
    labor_supply = np.asarray(labor_supply)
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset({"converged": ("checkpoint", [True])}, coords={"checkpoint": [1]}),
            "variables": xr.Dataset(
                {"Y": (("step", "i"), np.arange(6.0).reshape(3, 2))},
                coords={"step": steps, "i": ["a", "b"]},
            ),
            "parameters": xr.Dataset(
                {
                    "L_s": (("step", *("i",) * (labor_supply.ndim - 1)), labor_supply),
                    "alpha": ("step", [0.5, 0.5, 0.5]),
                },
                coords={"step": steps, "i": ["a", "b"]},
            ),
        }
    )


def test_index_by_parameter_replaces_step_with_the_parameter_values():
    indexed = index_by_parameter(path_tree([3.0, 2.0, 1.0]), "L_s")

    np.testing.assert_array_equal(indexed["variables"]["L_s"], [3.0, 2.0, 1.0])
    np.testing.assert_array_equal(indexed["variables"]["Y"].sel(L_s=2.0), [2.0, 3.0])
    np.testing.assert_array_equal(indexed["variables"]["step"], [0, 1, 2])
    assert list(indexed["parameters"].data_vars) == ["alpha"]
    assert indexed["parameters"]["alpha"].dims == ("L_s",)
    np.testing.assert_array_equal(indexed["converged"]["checkpoint"], [1])


@pytest.mark.parametrize(
    "labor_supply, message",
    [
        ([1.0, 2.0, 2.0], "strictly monotonically"),
        ([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], "must be a scalar parameter"),
    ],
    ids=["repeated_value", "indexed_parameter"],
)
def test_index_by_parameter_rejects_a_parameter_that_cannot_index_the_path(labor_supply, message):
    with pytest.raises(ValueError, match=message):
        index_by_parameter(path_tree(labor_supply), "L_s")


def test_make_table_typesets_each_cell_as_escaped_math():
    table = make_table({"": [1], "Name": ["Price_of & i"], "Equation": [Latex(r"P_{i} < \bar{P}")]})

    assert "<table class='mathjax_process'" in table
    assert r"\(\textbf{Equation}\)" in table
    assert r"\(1\)" in table
    assert r"\(\text{Price\_of}\) \(\text{\&amp;}\) \(\text{i}\)" in table
    assert r"\(P_{i} &lt; \bar{P}\)" in table


def test_make_table_fixes_the_layout_only_when_given_column_widths():
    columns = {"Name": ["Wage"], "Equation": [Latex("W = 1")]}

    fitted = make_table(columns)
    fixed = make_table(columns, column_widths=["30%", "70%"])

    assert "table-layout: fixed" not in fitted
    assert "table-layout: fixed" in fixed
    assert "width: 30%" in fixed
    assert "width: 70%" in fixed


def test_make_table_puts_subheadings_in_a_second_header_row():
    table = make_table({"Symbol": [Latex("x")], "run": {"Initial": [1.0], "Final": [2.0]}})

    first_row, second_row = table.split("<thead>")[1].split("</thead>")[0].split("</tr><tr>")
    assert r"\textbf{run}" in first_row
    assert r"\textbf{Initial}" in second_row
    assert r"\textbf{Final}" in second_row
    assert r"\(1.000\)" in table


@pytest.mark.parametrize(
    "columns, column_widths, message",
    [
        ({"a": [1.0], "b": [1.0, 2.0]}, None, "same length"),
        (
            {"Symbol": ["x"], "run": {"Initial": [1.0], "Final": [2.0]}},
            ["50%", "50%"],
            "Expected 3 column widths",
        ),
    ],
    ids=["ragged_columns", "one_width_per_heading"],
)
def test_make_table_rejects_columns_it_cannot_lay_out(columns, column_widths, message):
    with pytest.raises(ValueError, match=message):
        make_table(columns, column_widths=column_widths)
