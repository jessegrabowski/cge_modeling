from typing import Literal, Optional, Union, cast

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter

from cge_modeling import CGEModel


def prepare_gridspec_figure(
    n_cols: int, n_plots: int
) -> tuple[GridSpec, list[tuple[slice, slice], ...]]:
    """
     Prepare a figure with a grid of subplots. Centers the last row of plots if the number of plots is not square.

     Parameters
     ----------
     n_cols : int
         The number of columns in the grid.
     n_plots : int
         The number of subplots in the grid.

     Returns
     -------
     GridSpec
         A matplotlib GridSpec object representing the layout of the grid.
    list of tuple(slice, slice)
         A list of tuples of slices representing the indices of the grid cells to be used for each subplot.
    """

    remainder = n_plots % n_cols
    has_remainder = remainder > 0
    n_rows = n_plots // n_cols + int(has_remainder)

    gs = plt.GridSpec(2 * n_rows, 2 * n_cols)
    plot_locs = []

    for i in range(n_rows - int(has_remainder)):
        for j in range(n_cols):
            plot_locs.append((slice(i * 2, (i + 1) * 2), slice(j * 2, (j + 1) * 2)))

    if has_remainder:
        last_row = slice((n_rows - 1) * 2, n_rows * 2)
        left_pad = int(n_cols - remainder)
        for j in range(remainder):
            col_slice = slice(left_pad + j * 2, left_pad + (j + 1) * 2)
            plot_locs.append((last_row, col_slice))

    return gs, plot_locs


def plot_lines(
    idata: az.InferenceData,
    mod: CGEModel,
    n_cols: int = 5,
    var_names: Optional[list[str]] = None,
    initial_values: Optional[dict[str, np.ndarray]] = None,
    plot_euler: bool = True,
    plot_optimizer: bool = True,
    rename_dict: dict[str, str] | None = None,
    legends: bool | list | None = None,
    cmap: str | None = None,
    **figure_kwargs,
) -> plt.Figure:
    """
    Trace the evolution of the variables in the model over the course of the optimizer's iterations.

    Parameters
    ----------
    idata: az.InferenceData
        The InferenceData object returned by the model's simulate method.
    mod: CGEModel
        The model object.
    n_cols: int, default 5
        The number of columns in the grid of plots.
    var_names: list of str, optional
        Name of the variables to plot. If None, all variables will be plotted.
    initial_values: dict[str, np.array], optional
        The initial values of the variables in the model; those passed to the simulate method. If None, the initial
        values will be taken from the InferenceData object.
    plot_euler: bool, default True
        Whether to trace the shape of the function in the shock direction using steps from the Euler approximation.
    plot_optimizer: bool, default True
        Whether to plot the final values of each variable found by the optimizer.
    legends: bool or list, optional
        Whether to add a legend to the plot. If a list is passed, legends will be added to the variable names in the list.
        By default, no legends is added.
    rename_dict : dict[str, str], optional
        A dictionary mapping the variable names to more descriptive (or human readable) names for the plot.
    cmap: str, optional
        Matplotlib colormap, used to color the lines in plots with multiple variables. If None, the default colormap
        will be used.
    figure_kwargs: dict
        Additional keyword arguments to pass to plt.figure.

    Returns
    -------
    fig: plt.Figure
        The figure object containing the plot.
    """
    if var_names is None:
        var_names = mod.variable_names

    if rename_dict is None:
        rename_dict = {}

    if legends is None or legends is False:
        legends = dict.fromkeys(var_names, False)
    elif legends is True:
        legends = dict.fromkeys(var_names, True)
    else:
        no_legend = set(var_names) - set(legends)
        legends = dict.fromkeys(legends, True)
        legends.update(dict.fromkeys(no_legend, False))

    n_vars = len(var_names)
    gs, plot_locs = prepare_gridspec_figure(n_cols=n_cols, n_plots=n_vars)

    figsize = figure_kwargs.get("figsize", (15, 9))
    dpi = figure_kwargs.get("dpi", 144)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    cmap = "tab10" if cmap is None else cmap
    if cmap in plt.colormaps:

        def f_cmap(n):
            return plt.colormaps[cmap](np.linspace(0, 1, n))
    elif cmap in plt.color_sequences:

        def f_cmap(*args):
            return plt.color_sequences[cmap]
    else:
        raise ValueError(f"Colormap {cmap} not found in matplotlib.")

    for idx, var in enumerate(var_names):
        data = idata["euler"].variables[var]

        n_lines = int(np.prod(data.values.shape[1:]))
        cycler = plt.cycler(color=f_cmap(n_lines))
        axis = fig.add_subplot(gs[plot_locs[idx]])
        axis.set_prop_cycle(cycler)

        if data.ndim > 2:
            data = data.stack(pair=data.dims[1:])
        data.plot.line(x="step", ax=axis, add_legend=legends[var])
        axis.set(title=rename_dict.get(var, var), xlabel=None)

        scatter_grid = np.full(
            int(np.prod(idata["optimizer"].variables[var].shape)),
            idata["euler"].variables.coords["step"].max(),
        )
        axis.scatter(
            scatter_grid,
            idata["optimizer"].variables[var].data.ravel(),
            marker="*",
            color="tab:red",
            zorder=10,
            s=100,
        )
        [spine.set_visible(False) for spine in axis.spines.values()]
        axis.grid(ls="--", lw=0.5)

    fig.tight_layout()
    return fig


def plot_kateplot(
    idata: az.InferenceData,
    initial_values: dict[str, np.array],
    mod: CGEModel,
    var_names: Union[str, list[str]],
    shock_name: Optional[str] = None,
    rename_dict: Optional[dict[str, str]] = None,
    cmap: Optional[Union[str, Colormap]] = None,
    **plot_kwargs,
) -> plt.Figure:
    """
    Make an area plot of the initial and final values of the variables in the model.

    Parameters
    ----------
    idata: az.InferenceData
        The InferenceData object returned by the model's simulate method.
    initial_values: dict[str, np.array]
        The initial values of the variables in the model; those passed to the simulate method.
    mod: CGEModel
        The model object.
    var_names: str or list of str
        Name of the variables to plot. All variables must have dimensions, and all dimensions must be the same.
    shock_name: str, optional
        The name of the variable that was shocked, used in the title of the plot. If None, the title will just be
        "Pre-Shock" and "Post-Shock".
    rename_dict: dict[str, str], optional
        A dictionary mapping the variable names to more descriptive (or human readable) names for the plot.
    cmap: str or matplotlib.colors.Colormap, optional
        The colormap to use for the plot. If None, the default colormap will be used.

    **plot_kwargs

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    """
    try:
        import squarify  # noqa
    except ImportError:
        raise ImportError(
            'Package "squarify" is required to make kateplots. '
            "Please install the package using pip install squarify"
        )

    if rename_dict is None:
        rename_dict = {}

    if shock_name is None:
        shock_name = "Shock"

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))

    if cmap is None:
        cmap = plt.colormaps["tab10"]
    elif isinstance(cmap, str):
        cmap = plt.colormaps[cmap]

    dims = [mod.get(var).dims for var in var_names]
    if not all([dim == dims[0] for dim in dims]):
        raise ValueError(
            "Not all variables have the same dimensions, cannot plot together"
        )

    dims = list(dims[0])

    labels = [label for dim in dims for label in mod.coords[dim]]
    pretty_labels = [rename_dict.get(label, label) for label in labels]
    if len(pretty_labels) == 0:
        raise ValueError(
            "The selected variable is a scalar; cannot create an area plot."
        )

    pre_data = np.concatenate(
        [np.atleast_1d(initial_values[var]).ravel() for var in var_names]
    )
    if "optimizer" in idata:
        post_data = np.concatenate(
            [idata["optimizer"].variables[var].data.ravel() for var in var_names]
        )
    else:
        post_data = np.concatenate(
            [
                idata["euler"].isel(step=-1).variables[var].data.ravel()
                for var in var_names
            ]
        )

    zero_mask = np.isclose(pre_data, 0) | np.isclose(post_data, 0)
    pre_data = pre_data[~zero_mask]
    post_data = post_data[~zero_mask]

    pretty_labels = [label for label, mask in zip(pretty_labels, zero_mask) if not mask]

    colors = cmap(np.linspace(0, 1, len(pre_data)))
    for axis, data, title in zip(
        fig.axes, [pre_data, post_data], [f"Pre-{shock_name}", f"Post-{shock_name}"]
    ):
        axis.grid(ls="--", lw=0.5)
        axis.set(title=title)
        squarify.plot(
            sizes=data,
            label=pretty_labels,
            alpha=0.8,
            ax=axis,
            color=colors,
            **plot_kwargs,
        )

    return fig


def _compute_bar_data(data, initial_data, metric):
    if metric == "pct_change":
        return (data - initial_data) / initial_data
    elif metric == "change":
        return data - initial_data
    elif metric == "abs_change":
        return np.abs(data - initial_data)
    elif metric == "final":
        return data
    elif metric == "initial":
        return initial_data
    elif metric == "both":
        return data.join(initial_data, lsuffix="_final", rsuffix="_initial")
    else:
        raise ValueError(f"Invalid value type: {metric}")


def _plot_one_bar(
    data,
    initial_data,
    ax,
    drop_vars,
    orientation="v",
    metric: Literal[
        "pct_change", "change", "abs_change", "final", "initial", "both"
    ] = "pct_change",
    legend=True,
):
    try:
        data = data.to_dataframe().droplevel(level=drop_vars, axis=0)
        initial_data = initial_data.to_dataframe().droplevel(level=drop_vars, axis=0)
    except ValueError:
        data = pd.DataFrame([data.to_dict()]).loc[:, ["data", "name"]].set_index("name")
        initial_data = (
            pd.DataFrame([initial_data.to_dict()])
            .loc[:, ["data", "name"]]
            .set_index("name")
        )

    to_plot = _compute_bar_data(data, initial_data, metric)

    if "step" in to_plot.columns:
        to_plot = to_plot.drop(columns=["step"])

    if orientation == "v":
        to_plot.plot.bar(ax=ax, legend=legend)
        ax.axhline(0, color="black", lw=0.5)
    else:
        to_plot.plot.barh(ax=ax, legend=legend)
        ax.axvline(0, color="black", lw=0.5)

    if metric == "pct_change":
        ticker = PercentFormatter(xmax=1.0)
        if orientation == "v":
            ax.yaxis.set_major_formatter(ticker)
        elif orientation == "h":
            ax.xaxis.set_major_formatter(ticker)
    ax.tick_params(which="both", labelsize=6)
    return ax


def plot_bar(
    idata,
    mod,
    var_names,
    initial_values=None,
    plot_together=False,
    orientation="v",
    n_cols=3,
    coords=None,
    metric: Literal[
        "pct_change", "change", "abs_change", "final", "initial", "both"
    ] = "pct_change",
    **figure_kwargs,
):
    data = None
    coords = coords if coords is not None else {}
    coords = {
        key: value if isinstance(value, list) else [value]
        for key, value in coords.items()
    }
    drop_vars = [var for var in coords.keys() if len(coords[var]) == 1]

    if "optimizer" in idata:
        data = (
            idata["optimizer"]["variables"][var_names]
            .sel(**coords)
            .drop_vars(drop_vars)
        )

    if "euler" in idata:
        if data is None:
            data = (
                idata["euler"]
                .isel(step=-1)["variables"][var_names]
                .sel(**coords)
                .drop_vars(drop_vars)
            )
        if initial_values is None:
            initial_values = (
                idata["euler"]
                .isel(step=0)["variables"][var_names]
                .sel(**coords)
                .drop_vars(drop_vars)
            )

    if data is None:
        raise ValueError("No data found in InferenceData object.")

    dpi = figure_kwargs.pop("dpi", 144)
    figsize = figure_kwargs.pop("figsize", (15, 9))

    if var_names is None:
        var_names = mod.variable_names

    n_vars = len(var_names)
    if not plot_together:
        gs, plot_locs = prepare_gridspec_figure(n_cols=n_cols, n_plots=n_vars)
        fig = plt.figure(dpi=dpi, figsize=figsize, **figure_kwargs)
        for idx, var in enumerate(var_names):
            axis = fig.add_subplot(gs[plot_locs[idx]])
            axis = _plot_one_bar(
                data[var],
                initial_values[var],
                cast(plt.axis, axis),
                drop_vars=drop_vars,
                orientation=orientation,
                metric=metric,
            )
            axis.set(title=var)
            if metric == "final_initial":
                legend = axis.legend()
                for text in legend.get_texts():
                    new_text = text.get_text().split("_")[-1].capitalize()
                    text.set_text(new_text)

    else:
        var_dims = [
            [dim for dim in mod.get(var_name).dims if dim not in drop_vars]
            for var_name in var_names
        ]
        if not all([dim == var_dims[0] for dim in var_dims]):
            msg = "All variables must have the same dimensions to plot together. Found the following: \n"
            msg += "\n".join(
                f"{var_name}: {dim}" for var_name, dim in zip(var_names, var_dims)
            )
            msg += (
                "\nUse the `coords` argument to select values for dimensions that are not shared across variables, "
                "or set plot_together=False to plot separately."
            )
            raise ValueError(msg)
        fig, ax = plt.subplots(dpi=dpi, figsize=figsize, **figure_kwargs)

        # labels = [label for label in [make_labels(var_name, mod) for var_name in var_names]]
        _ = _plot_one_bar(
            data,
            initial_values,
            ax,
            drop_vars=drop_vars,
            orientation=orientation,
            metric=metric,
        )

    return fig
