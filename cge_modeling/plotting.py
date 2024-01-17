import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def prepare_gridspec_figure(n_cols: int, n_plots: int) -> tuple[GridSpec, list[slice, ...]]:
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
    idata,
    mod,
    n_cols=5,
    var_names=None,
    initial_values=None,
    plot_euler=None,
    plot_optimizer=None,
    **figure_kwargs,
):
    if var_names is None:
        var_names = mod.variable_names
    n_vars = len(var_names)
    gs, plot_locs = prepare_gridspec_figure(n_cols=n_cols, n_plots=n_vars)

    figsize = figure_kwargs.get("figsize", (15, 9))
    dpi = figure_kwargs.get("dpi", 144)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    for idx, var in enumerate(var_names):
        axis = fig.add_subplot(gs[plot_locs[idx]])
        data = idata["euler"].variables[var]
        if data.ndim > 2:
            data = data.stack(pair=data.dims[1:])
        data.plot.line(x="step", ax=axis, add_legend=False)
        axis.set(title=var, xlabel=None)

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


def plot_kateplot(idata, initial_values, var_names, shock_name, mod, rename_dict=None, cmap=None):
    try:
        import squarify
    except ImportError as e:
        raise ImportError(
            'Package "squarify" is required to make kateplots. '
            "Please install the package using pip install squarify"
        )

    if rename_dict is None:
        rename_dict = {}

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))

    if cmap is None:
        cmap = plt.cm.tab10

    dims = [idata["optimizer"].variables[var].dims for var in var_names]
    if not all([dim == dims[0] for dim in dims]):
        raise ValueError("Not all variables have the same dimensions, cannot plot together")

    dims = list(dims[0])

    labels = [label for dim in dims for label in mod.coords[dim]]
    pretty_labels = [rename_dict.get(label, label) for label in labels]

    pre_data = np.r_[*[initial_values[var].ravel() for var in var_names]]
    post_data = np.r_[*[idata["optimizer"].variables[var].data.ravel() for var in var_names]]

    colors = cmap(np.linspace(0, 1, len(pre_data)))
    for axis, data, title in zip(
        fig.axes, [pre_data, post_data], [f"Pre-{shock_name}", f"Post-{shock_name}"]
    ):
        axis.grid(ls="--", lw=0.5)
        axis.set(title=title)
        squarify.plot(sizes=data, label=pretty_labels, alpha=0.8, ax=axis, color=colors)
