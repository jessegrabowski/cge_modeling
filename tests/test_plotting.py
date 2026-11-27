import matplotlib.pyplot as plt
import numpy as np
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
    model = load_model_2()
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
        assert remainder > 0, (
            "expected_last_row_locs should only be provided if there's a remainder row to center."
        )
        assert len(expected_last_row_locs) == remainder, (
            f"Expected {remainder} locations for the last row, but got {len(expected_last_row_locs)}"
        )

        # Plots in the last row start after all full rows.
        # Index of the first plot in the last (partial) row within plot_locs list.
        start_index_of_last_row_plots = n_plots - remainder
        for i in range(remainder):
            actual_plot_loc_index = start_index_of_last_row_plots + i
            assert plot_locs[actual_plot_loc_index] == expected_last_row_locs[i], (
                f"Mismatch for plot_locs[{actual_plot_loc_index}] (plot {i + 1} in last row)"
            )


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
        ValueError, match=r"The selected variable is a scalar; cannot create an area plot\."
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


@pytest.mark.parametrize(
    "columns, expected_columns",
    [
        (["Y"], ["Final", "Initial"]),
        (["Y", "C"], ["Y (final)", "C (final)", "Y (initial)", "C (initial)"]),
    ],
    ids=["one_variable", "several_variables"],
)
def test_compute_bar_data_both_labels_final_and_initial(columns, expected_columns):
    initial_df = pd.DataFrame({key: [10.0, 20.0] for key in columns}, index=["A", "B"])
    final_df = pd.DataFrame({key: [12.0, 18.0] for key in columns}, index=["A", "B"])

    result_df = _compute_bar_data(final_df, initial_df, metric="both")

    assert list(result_df.columns) == expected_columns
    np.testing.assert_allclose(result_df[expected_columns[0]], [12.0, 18.0])
    np.testing.assert_allclose(result_df[expected_columns[-1]], [10.0, 20.0])


def test_compute_bar_data_invalid_metric():
    initial_df = pd.DataFrame({"data": [10.0]}, index=["A"])
    final_df = pd.DataFrame({"data": [12.0]}, index=["A"])
    with pytest.raises(ValueError, match="Invalid value type: non_existent_metric"):
        _compute_bar_data(final_df, initial_df, metric="non_existent_metric")


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_one_bar_vertical(cge_model):
    _model, _initial_calib_data, idata = cge_model
    var_name = "Y"

    data = idata["optimizer"]["variables"][var_name].isel(step=-1).to_dataframe()[[var_name]]
    initial_data = idata["optimizer"]["variables"][var_name].isel(step=0).to_dataframe()[[var_name]]

    _fig, ax = plt.subplots()
    ax_out = _plot_one_bar(data, initial_data, ax, metric="pct_change", orientation="v")
    assert isinstance(ax_out, Axes)
    assert len(ax_out.patches) == 3


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_one_bar_horizontal(cge_model):
    _model, _initial_calib_data, idata = cge_model
    var_name = "Y"

    data = idata["optimizer"]["variables"][var_name].isel(step=-1).to_dataframe()[[var_name]]
    initial_data = idata["optimizer"]["variables"][var_name].isel(step=0).to_dataframe()[[var_name]]

    _fig, ax = plt.subplots()
    ax_out_h = _plot_one_bar(data, initial_data, ax, orientation="h", metric="final")
    assert isinstance(ax_out_h, Axes)
    np.testing.assert_allclose([bar.get_width() for bar in ax_out_h.patches], data[var_name])


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
def test_plot_bar_draws_each_scalar_variable_in_its_own_panel(cge_model):
    model, _, idata = cge_model

    fig = plot_bar(idata, model, var_names=["r", "w", "income"])

    assert [[label.get_text() for label in axis.get_xticklabels()] for axis in fig.axes] == [
        ["r"],
        ["w"],
        ["income"],
    ]


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_bar_draws_scalar_variables_together_as_one_bar_each(cge_model):
    model, _, idata = cge_model

    fig = plot_bar(idata, model, var_names=["r", "w", "income"], plot_together=True)

    assert [label.get_text() for label in fig.axes[0].get_xticklabels()] == ["r", "w", "income"]


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_bar_sums_labels_into_the_groups_of_group_dict(cge_model):
    model, _, idata = cge_model
    groups = {"i": {"0": "Primary", "1": "Primary", "2": "Other"}}

    fig = plot_bar(idata, model, var_names=["Y"], group_dict=groups, metric="final")

    final_output = idata["optimizer"]["variables"]["Y"].isel(step=-1).values
    assert [label.get_text() for label in fig.axes[0].get_xticklabels()] == ["Other", "Primary"]
    np.testing.assert_allclose(
        [bar.get_height() for bar in fig.axes[0].patches],
        [final_output[2], final_output[0] + final_output[1]],
    )


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_bar_compares_against_the_given_initial_values(cge_model):
    model, _, idata = cge_model
    baseline = idata["optimizer"]["variables"].dataset.isel(step=-1) * 2

    fig = plot_bar(idata, model, var_names=["Y"], initial_values=baseline, metric="pct_change")

    np.testing.assert_allclose([bar.get_height() for bar in fig.axes[0].patches], -0.5)


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_bar_defaults_to_every_variable(cge_model):
    model, _, idata = cge_model

    fig = plot_bar(idata, model)

    assert [axis.get_title() for axis in fig.axes] == model.variable_names


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_bar_both_labels_the_legend_final_and_initial(cge_model):
    model, _, idata = cge_model

    fig = plot_bar(idata, model, var_names=["Y"], metric="both")

    assert [text.get_text() for text in fig.axes[0].get_legend().get_texts()] == [
        "Final",
        "Initial",
    ]


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_bar_selects_one_label_of_a_dim(cge_model):
    model, _, idata = cge_model

    fig = plot_bar(idata, model, var_names=["X"], coords={"i": "0"})

    assert [label.get_text() for label in fig.axes[0].get_xticklabels()] == ["0", "1", "2"]


@pytest.mark.usefixtures("auto_close_all_figures")
def test_plot_lines_passes_figure_keywords_to_the_figure(cge_model):
    model, _, idata = cge_model

    fig = plot_lines(idata, model, var_names=["Y"], facecolor="red", figsize=(4, 3))

    assert fig.get_facecolor() == (1.0, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(fig.get_size_inches(), [4, 3])
