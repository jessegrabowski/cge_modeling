import html
import numbers
import warnings

from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import xarray as xr

from IPython.display import HTML, Latex, display

from cge_modeling.base.primitives import ModelObject
from cge_modeling.base.utilities import flat_array_to_variable_dict


def get_node_level(node, level=0):
    if node.get_parent() is None:
        return level

    return get_node_level(node.get_parent(), level + 1)


def get_n_descendents(node, total=0):
    if node.n_children == 0:
        return total

    return node.n_children + sum(get_n_descendents(child, 0) for child in node.children)


def get_n_generations(node, total=0):
    if node.n_children == 0:
        return total
    return int(node.n_children > 0) + sum(get_n_generations(child, 0) for child in node.children)


def compute_padding(node):
    pad = max(0, node.n_descendents - 1)
    pad -= sum(max(0, descendent.n_descendents - 1) for descendent in node.get_all_descendents())

    return pad


class Node:
    """
    Class representing a node in a tree data structure. Nodes are connected in parent-child relationships, and
    can also hold data.

    I currently assume that all nodes have only a single parent.

    TODO: Break the coupling between nodes. The child nodes update the parent when created, which is
        not the best design.
    """

    def __init__(self, name, data=None, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.data = data

        self.update_parent()
        self.level = get_node_level(self)

    def add_child(self, child):
        self.children.append(child)

    @property
    def n_children(self):
        return len(self.children)

    @property
    def n_descendents(self):
        return get_n_descendents(self)

    @property
    def n_generations(self):
        return get_n_generations(self)

    def update_parent(self):
        if self.parent is not None:
            self.parent.add_child(self)

    def get_parent(self):
        return self.parent

    def get_all_descendents(self):
        descendents = self.children.copy()

        for child in self.children:
            descendents.extend(child.get_all_descendents())

        return descendents

    def get_all_ancestors(self):
        if self.parent is not None:
            ancestors = [self.parent]
            ancestors.extend(self.parent.get_all_ancestors())
            return ancestors

        else:
            return []

    def __repr__(self):
        return self.name


class Tree:
    """
    Class representing a tree data structure. Trees are built from nodes. The Tree class holds a list of nodes,
    and some helper methods for working with them.
    """

    def __init__(self, nodes=None):
        nodes = [] if nodes is None else nodes
        self.nodes = nodes

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def max_level(self):
        return max(node.level for node in self.nodes) + 1

    def get_nodes(self, level):
        return [node for node in self.nodes if node.level == level]

    def get_roots(self):
        return [node for node in self.nodes if node.parent is None]

    def get_leaves(self):
        return [node for node in self.nodes if node.n_children == 0]

    def get_family_trees(self):
        roots = [node for node in self.nodes if node.parent is None]
        return [Tree([root, *root.get_all_descendents()]) for root in roots]

    def get_subtree(self, node):
        family = [node, *node.get_all_descendents()]
        return Tree(family)

    def split_tree(self, cut_level):
        pruned_nodes = []
        update_map = {}
        for level in range(cut_level, self.max_level):
            level_nodes = self.get_nodes(level)
            for old_node in level_nodes:
                new_node = Node(
                    old_node.name,
                    parent=None if level == cut_level else update_map[old_node.parent],
                )
                update_map[old_node] = new_node
                pruned_nodes.append(new_node)

        pruned_tree = Tree(pruned_nodes)
        return pruned_tree

    def make_headings(self):
        """
        Create a list of lists of table headings, where parent nodes are listed above child nodes. In this way,
        the children can be understood as "subheadings".
        """
        headings = defaultdict(list)
        for root in self.get_roots():
            family = self.get_subtree(root)
            for level in range(self.max_level):
                parent_level = family.get_nodes(level - 1)
                family_level = family.get_nodes(level)
                if len(family_level) == 0:
                    n_leaves = len(family.get_leaves())
                    headings[level].extend([None] * n_leaves)
                elif level == 0:
                    for node in family_level:
                        headings[level].extend([node.name] + [None] * compute_padding(node))
                else:
                    for parent in parent_level:
                        if parent.n_children == 0:
                            headings[level].extend([None])
                        else:
                            for node in parent.children:
                                headings[level].extend([node.name] + [None] * compute_padding(node))
        return list(headings.values())


def build_tree(d, tree=None, parent=None):
    """
    Construct a Tree from a dictionary of dictionaries. The dictionary should have the following structure:
        - All keys are strings, representing columns in a table
        - If a column has subcolumns, its value should be a dictionary that follows these rules
        - If a column does not have subcolumns, its value should be a non-dictionary sequence of values
        - All column values must be the same length.

    Parameters
    ----------
    d: dict
        Dictionary of column-subcolumn-data relationships. See the rules above for details.
    tree: Tree
        Current Tree being constructed. Used in recursive parsing of nested dictionaries; you should never need
        to use this argumnet yourself.
    parent: Node
        Current node being recursively parsed. You don't need to declare this yourself.

    Returns
    -------
    Tree
        A tree data-structure representing the table to be created
    """
    if tree is None:
        tree = Tree()
    for k, v in d.items():
        node = Node(k, parent=parent)
        if not isinstance(v, dict):
            node.data = v
        tree.add_node(node)
        if isinstance(v, dict):
            tree = build_tree(v, tree, parent=node)

    return tree


TEXT_SPECIAL_CHARACTERS = {char: "\\" + char for char in "_%&#$"}
CELL_STYLE = "text-align: left; vertical-align: middle; padding: 0.5em"
STRIPE_STYLE = "background: rgba(128, 128, 128, 0.1)"


def text_to_latex(text: str, command: str = r"\text") -> list[str]:
    """
    Return one LaTeX block per word of ``text``.

    MathJax never breaks a line inside a single text block, so separate blocks let the browser wrap long text
    between words.
    """
    escaped = text.translate(str.maketrans(TEXT_SPECIAL_CHARACTERS))
    return [command + "{" + word + "}" for word in escaped.split()]


def parse_data(x) -> list[str]:
    if isinstance(x, ModelObject):
        return [str(x._full_latex_name)]
    if isinstance(x, Latex):
        return [str(x.data)]
    if isinstance(x, numbers.Integral):
        return [str(x)]
    if isinstance(x, numbers.Real):
        return [f"{x:.3f}"]
    if isinstance(x, str):
        return [x] if x.isnumeric() else text_to_latex(x)
    raise ValueError(f"Unexpected type {type(x)} in table data")


def math_cell(latex_blocks: list[str], tag: str = "td", width: str | None = None) -> str:
    """
    Return an HTML table cell whose content MathJax typesets as separate inline math blocks.

    Each block is typeset on its own, so no table outgrows the size limit MathJax puts on a single block.
    """
    style = CELL_STYLE if width is None else f"{CELL_STYLE}; width: {width}"
    content = " ".join(f"\\({html.escape(latex)}\\)" for latex in latex_blocks)
    return f"<{tag} style='{style}'>{content}</{tag}>"


def make_table(info_dict, column_widths: Sequence[str] | None = None) -> str:
    """
    Return an HTML table of ``info_dict`` with every cell typeset as math.

    Parameters
    ----------
    info_dict : dict
        Column headings mapped to column data, or to dicts of subheadings mapped to column data. Every column has
        the same length.
    column_widths : sequence of str, optional
        One CSS width per data column, such as ``"20%"``. With widths, the table spans its container with fixed
        column widths, which gives MathJax the width it needs to typeset long equations on one line. Without them,
        columns fit their content. Default is None.
    """
    tree = build_tree(info_dict)
    leaves = tree.get_leaves()
    datas = [leaf.data for leaf in leaves]
    n_rows = len(datas[0])
    if any(len(data) != n_rows for data in datas):
        raise ValueError("Every column of the table must have the same length.")
    if column_widths is not None and len(column_widths) != len(leaves):
        raise ValueError(f"Expected {len(leaves)} column widths, found {len(column_widths)}.")

    widths = column_widths if column_widths is not None else [None] * len(leaves)
    heading_rows = [
        "".join(
            math_cell(
                text_to_latex(heading, command=r"\textbf") if heading else [],
                tag="th",
                width=width if level == 0 else None,
            )
            for heading, width in zip(headings, widths, strict=True)
        )
        for level, headings in enumerate(tree.make_headings())
    ]
    data_rows = [
        "".join(math_cell(parse_data(data[row])) for data in datas) for row in range(n_rows)
    ]

    layout = "width: auto" if column_widths is None else "width: 100%; table-layout: fixed"
    table_style = f"{layout}; font-size: 14px"
    head = "".join(f"<tr>{row}</tr>" for row in heading_rows)
    body = "\n".join(
        f"<tr style='{STRIPE_STYLE}'>{row}</tr>" if index % 2 == 0 else f"<tr>{row}</tr>"
        for index, row in enumerate(data_rows)
    )
    return f"<table class='mathjax_process' style='{table_style}'><thead>{head}</thead><tbody>{body}</tbody></table>"


def display_latex_table(info_dict, column_widths: Sequence[str] | None = None) -> None:
    display(HTML(make_table(info_dict, column_widths=column_widths)))


def display_info_as_table(variable_info):
    display_latex_table(
        {
            "Symbol": [Latex(info["latex_name"]) for info in variable_info],
            "Description": [info["description"] for info in variable_info],
        }
    )


def datatree_to_value_dict(tree: xr.DataTree) -> dict[str, np.ndarray]:
    return {
        str(name): data.values
        for group in ("variables", "parameters")
        for name, data in tree[group].data_vars.items()
    }


def index_by_parameter(tree: xr.DataTree, parameter: str) -> xr.DataTree:
    """
    Index a simulation group by the values of one scalar parameter instead of by ``step``.

    Parameters
    ----------
    tree : xr.DataTree
        A group of a ``CGEModel.simulate`` result, such as ``result["euler"]``, whose ``variables`` and
        ``parameters`` children are indexed by ``step``.
    parameter : str
        Name of a scalar parameter whose value changes strictly monotonically along ``step``.

    Returns
    -------
    indexed : xr.DataTree
        The same tree with ``parameter`` as the dimension of both children. ``step`` stays as a coordinate, and the
        parameter itself moves from the ``parameters`` data variables to the index.
    """
    variables = tree.children["variables"].to_dataset(inherit=False)
    parameters = tree.children["parameters"].to_dataset(inherit=False)
    values = parameters[parameter]
    if values.dims != ("step",):
        raise ValueError(f"{parameter} must be a scalar parameter, but it has dims {values.dims}.")
    increments = np.diff(values.to_numpy())
    if not (np.all(increments > 0) or np.all(increments < 0)):
        raise ValueError(
            f"{parameter} must change strictly monotonically along step to index by it."
        )

    def reindex(group: xr.Dataset) -> xr.Dataset:
        return group.assign_coords({parameter: ("step", values.to_numpy())}).swap_dims(
            step=parameter
        )

    return xr.DataTree.from_dict(
        {
            "/": tree.to_dataset(inherit=False),
            "variables": reindex(variables),
            "parameters": reindex(parameters.drop_vars(parameter)),
        }
    )


def list_of_array_to_idata(list_of_arrays: list, cge_model):
    """
    Convert a list of arrays to an xarray dataset

    Parameters
    ----------
    list_of_arrays: list
        A list of numpy arrays, each one corresponding to a variable or parameter of the model, with left-most dimension
        corresponding to the approximation step index. It is assumed that variables are ordered before the parameters,
        and that the variables and parameters are in the same order as in the model.

    cge_model: CGEModel
        CGEModel object

    Returns
    -------
    idata: xr.DataTree
        DataTree with two children -- variables and parameters -- each holding a dataset with the same dimensions as
        the input arrays. Dimensions are labeled using coordinates from the CGEModel object.

    """

    variables = cge_model.variables
    parameters = cge_model.parameters
    n_variables = len(variables)

    coords = cge_model.coords.copy()
    coords.update({"step": range(len(list_of_arrays[0]))})

    xr_var_dict = {
        obj.name: (("step", *obj.dims), list_of_arrays[i]) for i, obj in enumerate(variables)
    }
    xr_param_dict = {
        obj.name: (("step", *obj.dims), list_of_arrays[i + n_variables])
        for i, obj in enumerate(parameters)
    }

    ds_vars = xr.Dataset(xr_var_dict, coords=coords)
    ds_params = xr.Dataset(xr_param_dict, coords=coords)

    return xr.DataTree.from_dict({"variables": ds_vars, "parameters": ds_params})


def optimizer_result_to_idata(res, theta, initial_values, mod):
    coords = deepcopy(mod.coords)
    coords["step"] = [0, 1]
    result = flat_array_to_variable_dict(
        np.r_[res.x, theta], mod.variables + mod.parameters, coords
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optim_var_dict = {
            obj.name: (
                ("step", *obj.dims),
                np.stack([initial_values[obj.name], result[obj.name]]),
            )
            for obj in mod.variables
        }
        # optim_param_dict = {obj.name: (obj.dims, result[obj.name]) for obj in mod.parameters}
        optim_param_dict = {
            obj.name: (
                ("step", *obj.dims),
                np.stack([initial_values[obj.name], result[obj.name]]),
            )
            for obj in mod.parameters
        }
        optim_idata = xr.DataTree.from_dict(
            {
                "variables": xr.Dataset(optim_var_dict, coords=coords),
                "parameters": xr.Dataset(optim_param_dict, coords=coords),
            }
        )

    return optim_idata
