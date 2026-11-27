import numpy as np
import pytest
import xarray as xr

from cge_modeling.tools.output_tools import datatree_to_value_dict, index_by_parameter


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
