import matplotlib.pyplot as plt
import pandas as pd
import pytest

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from cge_modeling.plotting import (
    _compute_bar_data,
    _plot_one_bar,
    plot_areas,
    plot_bar,
    plot_lines,
    prepare_gridspec_figure,
)
from tests.utilities.models import calibrate_model_2, load_model_2, model_2_data


@pytest.fixture
def auto_close_all_figures():
    """Fixture to automatically close all matplotlib figures after a test."""
    yield
    plt.close("all")


@pytest.fixture(scope="session")
def cge_model():
    model = load_model_2(backend="numba")
    initial_calib_data = calibrate_model_2(**model_2_data)

    sim_results_idata = model.simulate(
        initial_state=initial_calib_data,
        final_delta_pct={"L_s": 0.1},
        n_iter_euler=5,  # Keep low for speed
    )

    return model, initial_calib_data, sim_results_idata


@pytest.mark.parametrize(
    "n_cols, n_plots, expected_nrows, expected_ncols, expected_last_row_locs",
    [
        (
            3,
            5,
            4,
            6,
            [
                (slice(2, 4, None), slice(1, 3, None)),
                (slice(2, 4, None), slice(3, 5, None)),
            ],
        ),
        (2, 4, 4, 4, None),  # No specific last row centering for full grid
        (3, 3, 2, 6, None),  # Single row, no specific last row centering needed for this check
        (
            3,  # n_cols
            8,  # n_plots
            6,  # expected_nrows = 2 * (8//3 + 1) = 6
            6,  # expected_ncols = 2 * 3 = 6
            [  # expected_last_row_locs for the 2 plots in the last row
                (slice(4, 6, None), slice(1, 3, None)),  # plot_locs[6]
                (slice(4, 6, None), slice(3, 5, None)),  # plot_locs[7]
            ],
        ),
    ],
    ids=["3cols_5plots", "2cols_4plots", "3cols_3plots", "3cols_8plots"],
)
def test_prepare_gridspec_figure(
    n_cols, n_plots, expected_nrows, expected_ncols, expected_last_row_locs
):
    gs, plot_locs = prepare_gridspec_figure(n_cols=n_cols, n_plots=n_plots)
    assert isinstance(gs, GridSpec)
    assert len(plot_locs) == n_plots
    assert gs.nrows == expected_nrows
    assert gs.ncols == expected_ncols

    if expected_last_row_locs is not None:
        remainder = n_plots % n_cols
        # This check is for cases where the last row is not full and we expect specific centered locations.
        assert (
            remainder > 0
        ), "expected_last_row_locs should only be provided if there's a remainder row to center."
        assert (
            len(expected_last_row_locs) == remainder
        ), f"Expected {remainder} locations for the last row, but got {len(expected_last_row_locs)}"

        # Plots in the last row start after all full rows.
        # Index of the first plot in the last (partial) row within plot_locs list.
        start_index_of_last_row_plots = n_plots - remainder
        for i in range(remainder):
            actual_plot_loc_index = start_index_of_last_row_plots + i
            assert (
                plot_locs[actual_plot_loc_index] == expected_last_row_locs[i]
            ), f"Mismatch for plot_locs[{actual_plot_loc_index}] (plot {i+1} in last row)"


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_lines_default(cge_model):
    model, _, idata = cge_model
    var_names = ["Y", "C"]
    fig = plot_lines(idata, model, var_names=var_names)
    assert isinstance(fig, Figure)


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_lines_no_euler_with_rename_legend_cmap(cge_model):
    model, _, idata = cge_model
    var_names = ["Y", "C"]
    fig = plot_lines(
        idata,
        model,
        var_names=var_names,
        plot_euler=False,
        rename_dict={"Y": "Output"},
        legends=True,
        cmap="viridis",
    )
    assert isinstance(fig, Figure)


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_lines_no_optimizer_specific_legend(cge_model):
    model, _, idata = cge_model
    var_names = ["Y", "C"]
    fig = plot_lines(idata, model, var_names=var_names, plot_optimizer=False, legends=["Y"])
    assert isinstance(fig, Figure)


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_lines_invalid_cmap_raises_error(cge_model):
    model, _, idata = cge_model
    var_names = ["Y", "C"]
    with pytest.raises(ValueError, match="Colormap FakeMap not found"):
        plot_lines(idata, model, var_names=var_names, cmap="FakeMap")


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_areas_scalar_vars_raise_error(cge_model):
    model, initial_calib_data, idata = cge_model
    var_names = ["w"]

    with pytest.raises(
        ValueError, match="The selected variable is a scalar; cannot create an area plot."
    ):
        plot_areas(idata, initial_calib_data, model, var_names=var_names)


@pytest.mark.parametrize(
    "metric, initial_vals, final_vals, expected_vals, threshold",
    [
        ("pct_change", [10.0, 20.0, 0.0], [12.0, 18.0, 1.0], [0.2, -0.1, 0.0], 1e-6),
        ("change", [10.0, 20.0], [12.0, 18.0], [2.0, -2.0], 1e-6),
        ("abs_change", [10.0, 20.0], [12.0, 18.0], [2.0, 2.0], 1e-6),
        ("final", [10.0, 20.0], [12.0, 18.0], [12.0, 18.0], 1e-6),
        ("initial", [10.0, 20.0], [12.0, 18.0], [10.0, 20.0], 1e-6),
    ],
)
def test_compute_bar_data(metric, initial_vals, final_vals, expected_vals, threshold):
    idx = [chr(65 + i) for i in range(len(initial_vals))]
    initial_s = pd.Series(initial_vals, index=idx, name="data")
    final_s = pd.Series(final_vals, index=idx, name="data")

    # _compute_bar_data expects DataFrames that result from .to_dataframe()
    # For single series, this means a column named 'data'
    initial_df = initial_s.to_frame()
    final_df = final_s.to_frame()

    result_df = _compute_bar_data(final_df, initial_df, metric=metric, threshold=threshold)
    expected_s = pd.Series(expected_vals, index=idx, name="data")
    pd.testing.assert_series_equal(result_df["data"], expected_s, check_dtype=False)


def test_compute_bar_data_both():
    initial_df = pd.DataFrame({"data": [10.0, 20.0]}, index=["A", "B"])
    final_df = pd.DataFrame({"data": [12.0, 18.0]}, index=["A", "B"])
    result_df = _compute_bar_data(final_df, initial_df, metric="both")
    expected_df = pd.DataFrame(
        {"data_final": [12.0, 18.0], "data_initial": [10.0, 20.0]}, index=["A", "B"]
    )
    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)


def test_compute_bar_data_invalid_metric():
    initial_df = pd.DataFrame({"data": [10.0]}, index=["A"])
    final_df = pd.DataFrame({"data": [12.0]}, index=["A"])
    with pytest.raises(ValueError, match="Invalid value type: non_existent_metric"):
        _compute_bar_data(final_df, initial_df, metric="non_existent_metric")


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_one_bar_vertical(cge_model):
    model, initial_calib_data, idata = cge_model
    var_name = "Y"

    data_xr = idata["optimizer"].isel(step=-1)["variables"][var_name]
    initial_data_xr = idata["optimizer"].isel(step=0)["variables"][var_name]

    fig, ax = plt.subplots()
    ax_out = _plot_one_bar(
        data_xr, initial_data_xr, ax, drop_vars=[], metric="pct_change", orientation="v"
    )
    assert isinstance(ax_out, Axes)


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_one_bar_horizontal(cge_model):
    model, initial_calib_data, idata = cge_model
    var_name = "Y"

    data_xr = idata["optimizer"].isel(step=-1)["variables"][var_name]
    initial_data_xr = idata["optimizer"].isel(step=0)["variables"][var_name]

    fig, ax = plt.subplots()
    ax_out_h = _plot_one_bar(
        data_xr,
        initial_data_xr,
        ax,
        drop_vars=[],
        orientation="h",
        metric="final",
    )
    assert isinstance(ax_out_h, Axes)


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_bar_separate_plots(cge_model):
    model, _, idata = cge_model
    var_names = ["Y", "C"]
    fig = plot_bar(idata, model, var_names, plot_together=False)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == len(var_names)


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_bar_together_horizontal_metric_change(cge_model):
    model, _, idata = cge_model
    var_names = ["Y", "C"]
    fig = plot_bar(
        idata,
        model,
        var_names,
        plot_together=True,
        metric="change",
        orientation="h",
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1  # Plot together


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_bar_with_group_dict(cge_model):
    model, _, idata = cge_model
    var_names = ["Y", "C"]
    fig = plot_bar(
        idata,
        model,
        var_names,
        group_dict={"some_coord": {"A": "Group1"}},  # 'some_coord' won't match index name
        metric="final",
    )
    assert isinstance(fig, Figure)


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_bar_no_initial_values_uses_idata(cge_model):
    model, _, idata = cge_model
    var_names = ["Y", "C"]
    fig = plot_bar(idata, model, var_names)
    assert isinstance(fig, Figure)
