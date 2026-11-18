import numpy as np
import xarray as xr

from cge_modeling.tools.output_tools import datatree_to_value_dict


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
