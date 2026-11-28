from functools import partial
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter

from cge_modeling import CGEModel
from cge_modeling.base.primitives import Parameter, Variable


def prepare_gridspec_figure(
    n_cols: int, n_plots: int
) -> tuple[GridSpec, list[tuple[slice, slice]]]:
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
    idata: dict[str, xr.DataTree],
    mod: CGEModel,
    n_cols: int = 5,
    var_names: list[str] | None = None,
    plot_path: bool = True,
    plot_optimizer: bool = True,
    rename_dict: dict[str, str] | None = None,
    legends: bool | list | None = None,
    cmap: str | None = None,
    **figure_kwargs,
) -> plt.Figure:
    """
    Plot each variable along the path, with stars at the optimizer's initial and final values.

    Parameters
    ----------
    idata : dict of str to xr.DataTree
        The result of :meth:`~cge_modeling.base.cge.CGEModel.simulate`, with an ``"path"`` group, an
        ``"optimizer"`` group, or both.
    mod : CGEModel
        The model that produced ``idata``.
    n_cols : int, optional
        Number of columns in the grid of panels, capped at the number of variables. Default is 5.
    var_names : list of str, optional
        Variables to plot, one panel each. Default is None, which plots every variable of ``mod``.
    plot_path : bool, optional
        Draw the path of each variable. Default is True.
    plot_optimizer : bool, optional
        Mark the optimizer's initial value with a green star and its final value with a red star. Default is True.
    rename_dict : dict of str to str, optional
        Panel titles, keyed by variable name. Default is None, which titles each panel with its variable name.
    legends : bool or list of str, optional
        True adds a legend to every panel, and a list adds one to the panels of the variables it names. Default is
        None, which adds no legend.
    cmap : str, optional
        Name of a matplotlib colormap or color sequence, which colors the lines of an indexed variable. Default is
        None, which uses ``"tab10"``.
    **figure_kwargs
        Passed to ``plt.figure``. Default ``figsize`` is ``(15, 9)`` and default ``dpi`` is 144.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure, with one panel per variable.

    Examples
    --------
    Raise labor supply by 10 percent and trace output and both prices along the way to the new equilibrium:

    .. code-block:: python

        import cge_modeling as cm

        with cm.CGEModel() as model:
            Y = cm.Variable("Y", description="Output")
            K_d = cm.Variable("K_d", description="Capital demand")
            L_d = cm.Variable("L_d", description="Labor demand")
            r = cm.Variable("r", description="Rental rate of capital")
            P = cm.Variable("P", description="Price of output")
            w = cm.Parameter("w", description="Wage, the numeraire")
            alpha = cm.Parameter("alpha", description="Capital share")
            A = cm.Parameter("A", description="Total factor productivity")
            K_s = cm.Parameter("K_s", description="Capital supply")
            L_s = cm.Parameter("L_s", description="Labor supply")

            cm.Equation("Production", Y, A * K_d**alpha * L_d ** (1 - alpha))
            cm.Equation("Capital demand", r * K_d, alpha * P * Y)
            cm.Equation("Labor demand", w * L_d, (1 - alpha) * P * Y)
            cm.Equation("Capital market clearing", K_d, K_s)
            cm.Equation("Labor market clearing", L_d, L_s)

        calibration = model.calibrate({"K_s": 3000.0, "L_s": 7000.0, "w": 1.0, "r": 1.0, "P": 1.0})
        result = model.simulate(initial_state=calibration,
                                final_delta_pct={"L_s": 1.1},
                                n_path_steps=100,
                                progressbar=False,
                                verbose=False)

        fig = cm.plot_lines(result, model, var_names=["Y", "r", "P"], n_cols=3)
    """
    if var_names is None:
        var_names = mod.variable_names

    if rename_dict is None:
        rename_dict = {}

    if legends is None or legends is False:
        legend_flags = dict.fromkeys(var_names, False)
    elif legends is True:
        legend_flags = dict.fromkeys(var_names, True)
    else:
        no_legend = set(var_names) - set(legends)
        legend_flags = dict.fromkeys(legends, True)
        legend_flags.update(dict.fromkeys(no_legend, False))

    if "path" not in idata and plot_path:
        raise ValueError('Cannot plot the path, provided results has no "path" group')
    if "optimizer" not in idata and plot_optimizer:
        raise ValueError(
            'Cannot plot results of optimizer, provided results has no "optimizer" group'
        )

    n_vars = len(var_names)
    n_cols = min(n_cols, n_vars)
    gs, plot_locs = prepare_gridspec_figure(n_cols=n_cols, n_plots=n_vars)

    figure_kwargs.setdefault("figsize", (15, 9))
    figure_kwargs.setdefault("dpi", 144)
    fig = plt.figure(**figure_kwargs)
    cmap = "tab10" if cmap is None else cmap

    if cmap in plt.colormaps:

        def f_cmap(n):
            return plt.colormaps[cmap](np.linspace(0, 1, n))
    elif cmap in plt.color_sequences:

        def f_cmap(n):
            return plt.color_sequences[cmap]
    else:
        raise ValueError(f"Colormap {cmap} not found in matplotlib.")

    for idx, var in enumerate(var_names):
        axis = fig.add_subplot(gs[plot_locs[idx]])
        axis.set(title=rename_dict.get(var, var), xlabel=None)

        if plot_path:
            data = idata["path"]["variables"].dataset[var]

            n_lines = int(np.prod(data.values.shape[1:]))
            cycler = plt.cycler(color=f_cmap(n_lines))
            axis.set_prop_cycle(cycler)

            if data.ndim > 2:
                data = data.stack(pair=list(data.dims[1:]))
            data.plot.line(x="step", ax=axis, add_legend=legend_flags[var])

        if plot_optimizer:
            initial_value = idata["optimizer"]["variables"].dataset[var].isel(step=0).data.ravel()
            final_value = idata["optimizer"]["variables"].dataset[var].isel(step=-1).data.ravel()

            t0 = np.zeros_like(initial_value)
            T = np.ones_like(final_value)

            if plot_path:
                T = T * idata["path"]["variables"]["step"].max().item()

            axis.scatter(
                t0,
                initial_value,
                marker="*",
                color="tab:green",
                zorder=10,
                s=100,
            )
            axis.scatter(
                T,
                final_value,
                marker="*",
                color="tab:red",
                zorder=10,
                s=100,
            )

        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.grid(ls="--", lw=0.5)

    fig.tight_layout()
    return fig


def plot_areas(
    idata: dict[str, xr.DataTree],
    initial_values: dict[str, np.ndarray],
    mod: CGEModel,
    var_names: str | list[str],
    shock_name: str | None = None,
    rename_dict: dict[str, str] | None = None,
    cmap: str | Colormap | None = None,
    **plot_kwargs,
) -> plt.Figure:
    """
    Draw treemaps of indexed variables before and after a shock, side by side, with one tile per label.

    Tiles whose initial or final value is zero are dropped. Drawing the tiles requires the ``squarify`` package.

    Parameters
    ----------
    idata : dict of str to xr.DataTree
        The result of :meth:`~cge_modeling.base.cge.CGEModel.simulate`. The final values come from its
        ``"optimizer"`` group if it has one, and from its ``"path"`` group otherwise.
    initial_values : dict of str to ndarray
        Values before the shock, keyed by variable name, such as the ``initial_state`` passed to ``simulate``.
    mod : CGEModel
        The model that produced ``idata``.
    var_names : list of str
        Variables to draw together. Every one must be indexed, and all must have the same dims.
    shock_name : str, optional
        Name of the shock in the panel titles, "Pre-" and "Post-" followed by it. Default is None, which uses
        ``"Shock"``.
    rename_dict : dict of str to str, optional
        Tile labels, keyed by coordinate label. Default is None, which labels each tile with its coordinate label.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap of the tiles. Default is None, which uses ``"tab10"``.
    **plot_kwargs
        Passed to ``squarify.plot``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure, with the pre-shock treemap on the left and the post-shock treemap on the right.

    Examples
    --------
    Raise the capital stock of a two-sector economy by 20 percent and compare the sectors' output before and after:

    .. code-block:: python

        import cge_modeling as cm

        with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
            Y = cm.Variable("Y", dims="sector", description="Output")
            C = cm.Variable("C", dims="sector", description="Household consumption")
            K_d = cm.Variable("K_d", dims="sector", description="Capital demand")
            L_d = cm.Variable("L_d", dims="sector", description="Labor demand")
            P = cm.Variable("P", dims="sector", description="Price of output")
            r = cm.Variable("r", description="Rental rate of capital")
            income = cm.Variable("income", description="Household income")
            resid = cm.Variable("resid", description="Walras residual")
            w = cm.Parameter("w", description="Wage, the numeraire")
            alpha = cm.Parameter("alpha", dims="sector", description="Capital share")
            A = cm.Parameter("A", dims="sector", description="Total factor productivity")
            beta = cm.Parameter("beta", dims="sector", description="Consumption share")
            K_s = cm.Parameter("K_s", description="Capital supply")
            L_s = cm.Parameter("L_s", description="Labor supply")

            cm.Equation("Production", Y, A * K_d**alpha * L_d ** (1 - alpha))
            cm.Equation("Capital demand", r * K_d, alpha * P * Y)
            cm.Equation("Labor demand", w * L_d, (1 - alpha) * P * Y)
            cm.Equation("Household income", income, w * L_s + r * K_s)
            cm.Equation("Consumption", P * C, beta * income)
            cm.Equation("Goods market clearing", Y, C)
            cm.Equation("Capital market clearing", K_s, K_d.sum("sector"))
            cm.Equation("Labor market clearing", L_s, L_d.sum("sector") + resid)

        calibration = model.calibrate({"K_d": [1000.0, 2000.0],
                                       "L_d": [3000.0, 4000.0],
                                       "L_s": 7000.0,
                                       "w": 1.0,
                                       "r": 1.0,
                                       "P": [1.0, 1.0]})
        result = model.simulate(initial_state=calibration,
                                final_delta_pct={"K_s": 1.2},
                                n_path_steps=100,
                                progressbar=False,
                                verbose=False)

        fig = cm.plot_areas(result, calibration["variables"], model, var_names=["Y"], shock_name="Capital")
    """
    try:
        import squarify
    except ImportError:
        raise ImportError(
            'Package "squarify" is required to make kateplots. '
            "Please install the package using pip install squarify"
        )

    if rename_dict is None:
        rename_dict = {}

    if shock_name is None:
        shock_name = "Shock"

    fig, _ax = plt.subplots(1, 2, figsize=(14, 4))

    if cmap is None:
        cmap = plt.colormaps["tab10"]
    elif isinstance(cmap, str):
        cmap = plt.colormaps[cmap]

    variable_dims = [
        cast(tuple[str, ...], cast(Variable | Parameter, mod.get(var)).dims) for var in var_names
    ]
    if not all([dim == variable_dims[0] for dim in variable_dims]):
        raise ValueError("Not all variables have the same dimensions, cannot plot together")

    dims = list(variable_dims[0])

    labels = [label for dim in dims for label in mod.coords[dim]]
    pretty_labels = [rename_dict.get(label, label) for label in labels]
    if len(pretty_labels) == 0:
        raise ValueError("The selected variable is a scalar; cannot create an area plot.")

    pre_data = np.concatenate([np.atleast_1d(initial_values[var]).ravel() for var in var_names])
    if "optimizer" in idata:
        post_data = np.concatenate(
            [idata["optimizer"].isel(step=-1)["variables"][var].data.ravel() for var in var_names]
        )
    else:
        post_data = np.concatenate(
            [idata["path"].isel(step=-1)["variables"][var].data.ravel() for var in var_names]
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


def _bar_frame(selection: xr.Dataset) -> pd.DataFrame:
    """
    Return the values of ``selection`` as a frame of bars.

    Variables with dims become columns, with a row per label. Scalar variables become rows of one ``value`` column, so
    each gets its own bar.
    """
    if not selection.sizes:
        values = {name: value.item() for name, value in selection.data_vars.items()}
        return pd.DataFrame({"value": values}).rename_axis("variable")
    return selection.to_dataframe()[list(selection.data_vars)]


def _compute_bar_data(data, initial_data, metric, threshold=1e-6):
    if metric == "pct_change":
        pct_change = np.divide(
            data - initial_data,
            initial_data,
            where=initial_data > threshold,
            out=np.zeros_like(data),
        )
        return pct_change
    elif metric == "change":
        return data - initial_data
    elif metric == "abs_change":
        return np.abs(data - initial_data)
    elif metric == "final":
        return data
    elif metric == "initial":
        return initial_data
    elif metric == "both":
        if data.shape[1] == 1:
            return pd.DataFrame({"Final": data.iloc[:, 0], "Initial": initial_data.iloc[:, 0]})
        return pd.concat(
            [data.add_suffix(" (final)"), initial_data.add_suffix(" (initial)")], axis=1
        )
    else:
        raise ValueError(f"Invalid value type: {metric}")


def _plot_one_bar(
    data: pd.DataFrame,
    initial_data: pd.DataFrame,
    ax,
    group_dict=None,
    orientation="v",
    metric: Literal[
        "pct_change", "change", "abs_change", "final", "initial", "both"
    ] = "pct_change",
    legend=True,
    threshold=1e-6,
    labelsize=12,
    xlabel_rotation=0,
):
    if group_dict is not None:
        coord_dict = group_dict.get(data.index.name)
        if coord_dict is not None:
            data = data.groupby(data.index.map(coord_dict)).sum()
            initial_data = initial_data.groupby(initial_data.index.map(coord_dict)).sum()

    to_plot = _compute_bar_data(data, initial_data, metric, threshold)

    if orientation == "v":
        to_plot.plot.bar(ax=ax, legend=legend)
        ax.axhline(0, color="black", lw=2.0)
    else:
        to_plot.plot.barh(ax=ax, legend=legend)
        ax.axvline(0, color="black", lw=2.0)

    if metric == "pct_change":
        ticker = PercentFormatter(xmax=1.0)
        if orientation == "v":
            ax.yaxis.set_major_formatter(ticker)
        elif orientation == "h":
            ax.xaxis.set_major_formatter(ticker)
    ax.tick_params(which="both", labelsize=labelsize)
    ax.tick_params(which="major", axis="x", rotation=xlabel_rotation)
    return ax


def plot_bar(
    idata,
    mod,
    var_names=None,
    initial_values=None,
    plot_together=False,
    group_dict=None,
    orientation="v",
    n_cols=3,
    coords=None,
    metric: Literal[
        "pct_change", "change", "abs_change", "final", "initial", "both"
    ] = "pct_change",
    threshold=1e-6,
    labelsize=12,
    xlabel_rotation=0,
    legend=True,
    **figure_kwargs,
):
    """
    Plot bar charts of how variables change between the start and the end of a simulation.

    The final values come from the last step of the ``"optimizer"`` group of ``idata`` if it has one, and from the
    last step of the ``"path"`` group otherwise.

    Parameters
    ----------
    idata : dict of str to xr.DataTree
        The result of :meth:`~cge_modeling.base.cge.CGEModel.simulate`.
    mod : CGEModel
        The model that produced ``idata``.
    var_names : list of str, optional
        Variables to plot. Default is None, which plots every variable of ``mod``.
    initial_values : xr.Dataset, optional
        Values at the start, as a dataset of ``var_names``. Default is None, which takes them from the first step of
        the same group as the final values.
    plot_together : bool, optional
        Draw every variable in one panel, which requires them to have the same dims after ``coords`` selects from
        them. Default is False, which draws one panel per variable.
    group_dict : dict of str to dict, optional
        Groupings to sum bars into, keyed by dim name, each mapping a coordinate label to its group. Default is None,
        which draws one bar per label.
    orientation : {"v", "h"}, optional
        Vertical or horizontal bars. Default is ``"v"``.
    n_cols : int, optional
        Number of columns in the grid of panels, capped at the number of variables. Default is 3.
    coords : dict of str to str or list of str, optional
        Labels to select, keyed by dim name. A dim given a single label is dropped from the bars. Default is None,
        which keeps every label.
    metric : {"pct_change", "change", "abs_change", "final", "initial", "both"}, optional
        What each bar shows: the fractional change, the change, its absolute value, the final value, the initial
        value, or the final and initial values side by side. Default is ``"pct_change"``.
    threshold : float, optional
        Initial values at or below this get a fractional change of zero, for ``metric="pct_change"``. Default is
        1e-6.
    labelsize : int, optional
        Font size of the tick labels. Default is 12.
    xlabel_rotation : float, optional
        Rotation of the x tick labels, in degrees. Default is 0.
    legend : bool, optional
        Add a legend to each panel. Default is True.
    **figure_kwargs
        Passed to ``plt.figure``, or to ``plt.subplots`` when ``plot_together`` is True. ``dpi`` and ``figsize``
        default to 144 and ``(15, 9)``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.

    Examples
    --------
    Raise the capital stock of a two-sector economy by 20 percent and compare the change in each sector's output
    and price:

    .. code-block:: python

        import cge_modeling as cm

        with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
            Y = cm.Variable("Y", dims="sector", description="Output")
            C = cm.Variable("C", dims="sector", description="Household consumption")
            K_d = cm.Variable("K_d", dims="sector", description="Capital demand")
            L_d = cm.Variable("L_d", dims="sector", description="Labor demand")
            P = cm.Variable("P", dims="sector", description="Price of output")
            r = cm.Variable("r", description="Rental rate of capital")
            income = cm.Variable("income", description="Household income")
            resid = cm.Variable("resid", description="Walras residual")
            w = cm.Parameter("w", description="Wage, the numeraire")
            alpha = cm.Parameter("alpha", dims="sector", description="Capital share")
            A = cm.Parameter("A", dims="sector", description="Total factor productivity")
            beta = cm.Parameter("beta", dims="sector", description="Consumption share")
            K_s = cm.Parameter("K_s", description="Capital supply")
            L_s = cm.Parameter("L_s", description="Labor supply")

            cm.Equation("Production", Y, A * K_d**alpha * L_d ** (1 - alpha))
            cm.Equation("Capital demand", r * K_d, alpha * P * Y)
            cm.Equation("Labor demand", w * L_d, (1 - alpha) * P * Y)
            cm.Equation("Household income", income, w * L_s + r * K_s)
            cm.Equation("Consumption", P * C, beta * income)
            cm.Equation("Goods market clearing", Y, C)
            cm.Equation("Capital market clearing", K_s, K_d.sum("sector"))
            cm.Equation("Labor market clearing", L_s, L_d.sum("sector") + resid)

        calibration = model.calibrate({"K_d": [1000.0, 2000.0],
                                       "L_d": [3000.0, 4000.0],
                                       "L_s": 7000.0,
                                       "w": 1.0,
                                       "r": 1.0,
                                       "P": [1.0, 1.0]})
        result = model.simulate(initial_state=calibration,
                                final_delta_pct={"K_s": 1.2},
                                n_path_steps=100,
                                progressbar=False,
                                verbose=False)

        fig = cm.plot_bar(result, model, var_names=["Y", "P"])

    Put scalar variables in one panel with ``plot_together``, one bar per variable:

    .. code-block:: python

        import cge_modeling as cm

        with cm.CGEModel(coords={"sector": ["Agriculture", "Industry"]}) as model:
            Y = cm.Variable("Y", dims="sector", description="Output")
            C = cm.Variable("C", dims="sector", description="Household consumption")
            K_d = cm.Variable("K_d", dims="sector", description="Capital demand")
            L_d = cm.Variable("L_d", dims="sector", description="Labor demand")
            P = cm.Variable("P", dims="sector", description="Price of output")
            r = cm.Variable("r", description="Rental rate of capital")
            income = cm.Variable("income", description="Household income")
            resid = cm.Variable("resid", description="Walras residual")
            w = cm.Parameter("w", description="Wage, the numeraire")
            alpha = cm.Parameter("alpha", dims="sector", description="Capital share")
            A = cm.Parameter("A", dims="sector", description="Total factor productivity")
            beta = cm.Parameter("beta", dims="sector", description="Consumption share")
            K_s = cm.Parameter("K_s", description="Capital supply")
            L_s = cm.Parameter("L_s", description="Labor supply")

            cm.Equation("Production", Y, A * K_d**alpha * L_d ** (1 - alpha))
            cm.Equation("Capital demand", r * K_d, alpha * P * Y)
            cm.Equation("Labor demand", w * L_d, (1 - alpha) * P * Y)
            cm.Equation("Household income", income, w * L_s + r * K_s)
            cm.Equation("Consumption", P * C, beta * income)
            cm.Equation("Goods market clearing", Y, C)
            cm.Equation("Capital market clearing", K_s, K_d.sum("sector"))
            cm.Equation("Labor market clearing", L_s, L_d.sum("sector") + resid)

        calibration = model.calibrate({"K_d": [1000.0, 2000.0],
                                       "L_d": [3000.0, 4000.0],
                                       "L_s": 7000.0,
                                       "w": 1.0,
                                       "r": 1.0,
                                       "P": [1.0, 1.0]})
        result = model.simulate(initial_state=calibration,
                                final_delta_pct={"K_s": 1.2},
                                n_path_steps=100,
                                progressbar=False,
                                verbose=False)

        fig = cm.plot_bar(result, model, var_names=["r", "income"], plot_together=True)
    """
    if var_names is None:
        var_names = mod.variable_names
    coords = coords if coords is not None else {}
    # A single label selects a scalar, so its dim drops out of the plot instead of becoming an axis of one bar.
    indexers = {
        dim: labels[0] if isinstance(labels, list) and len(labels) == 1 else labels
        for dim, labels in coords.items()
    }

    def select(variables: xr.Dataset) -> xr.Dataset:
        selection = variables[var_names].sel(indexers)
        return selection.drop_vars(
            [name for name in selection.coords if name not in selection.dims]
        )

    group = idata.get("optimizer", idata.get("path"))
    if group is None:
        raise ValueError('idata has neither an "optimizer" nor an "path" group.')
    data = select(group["variables"].dataset.isel(step=-1))
    if initial_values is None:
        initial_values = group["variables"].dataset.isel(step=0)
    initial_values = select(initial_values)

    figure_kwargs.setdefault("figsize", (15, 9))
    figure_kwargs.setdefault("dpi", 144)
    plot_one_bar = partial(
        _plot_one_bar,
        group_dict=group_dict,
        orientation=orientation,
        metric=metric,
        legend=legend,
        threshold=threshold,
        labelsize=labelsize,
        xlabel_rotation=xlabel_rotation,
    )

    if not plot_together:
        n_cols = min(n_cols, len(var_names))
        gs, plot_locs = prepare_gridspec_figure(n_cols=n_cols, n_plots=len(var_names))
        fig = plt.figure(**figure_kwargs)
        for idx, var in enumerate(var_names):
            axis = cast(Axes, fig.add_subplot(gs[plot_locs[idx]]))
            plot_one_bar(_bar_frame(data[[var]]), _bar_frame(initial_values[[var]]), axis)
            axis.set(title=var)
        return fig

    var_dims = [data[var_name].dims for var_name in var_names]
    if not all(dims == var_dims[0] for dims in var_dims):
        msg = (
            "All variables must have the same dimensions to plot together. Found the following: \n"
        )
        msg += "\n".join(f"{var_name}: {list(dims)}" for var_name, dims in zip(var_names, var_dims))
        msg += (
            "\nUse the `coords` argument to select values for dimensions that are not shared across variables, "
            "or set plot_together=False to plot separately."
        )
        raise ValueError(msg)

    fig, ax = plt.subplots(**figure_kwargs)
    plot_one_bar(_bar_frame(data), _bar_frame(initial_values), ax)

    return fig
