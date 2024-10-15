import re
import warnings

from collections import defaultdict
from copy import deepcopy

import arviz as az
import numpy as np
import sympy as sp
import xarray as xr

from IPython.display import Latex, display
from latextable import draw_latex
from sympy.printing.latex import LatexPrinter
from texttable import Texttable

from cge_modeling.base.primitives import ModelObject
from cge_modeling.base.utilities import flat_array_to_variable_dict
from cge_modeling.tools.sympy_tools import symbol


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


def heading_to_latex(s):
    return r"\textbf{" + s + "}"


def parse_data(x):
    if isinstance(x, ModelObject):
        return x._full_latex_name

    elif isinstance(x, float | int):
        return x
    elif isinstance(x, str):
        if x.isnumeric():
            return x
        else:
            return r"\text{" + x + "}"
    elif isinstance(x, sp.Eq | sp.Symbol | sp.Add | sp.Mul | sp.Expr):
        return LatexPrinter().doprint(x)
    else:
        raise ValueError(f"Unexpected type {type(x)} in table data")


def make_table(info_dict):
    tree = build_tree(info_dict)
    headings = tree.make_headings()
    headings = [
        [heading_to_latex(s) if s is not None else "" for s in heading] for heading in headings
    ]

    leaves = tree.get_leaves()
    n_cols = len(leaves)

    datas = [leaf.data for leaf in leaves]
    data_lens = [len(data) for data in datas]
    assert all([data_len == data_lens[0] for data_len in data_lens])

    n_data = data_lens[0]
    data_rows = [[parse_data(datas[i][row]) for i in range(n_cols)] for row in range(n_data)]

    equation_table = Texttable()
    equation_table.set_cols_align(["c"] * n_cols)
    equation_table.set_cols_valign(["m"] * n_cols)

    equation_table.add_rows(headings + data_rows)
    latex_table = draw_latex(equation_table)

    return latex_table


def make_summary_table(variables, values=None):
    headings = [r"\text{Symbol}", r"\text{Description}"]
    if values is not None:
        headings += [r"\text{Calibrated Value}"]
    heading_row = [headings]

    summary_table = Texttable()
    summary_table.set_cols_align(["c"] * (2 + int(values is not None)))
    summary_table.set_cols_valign(["m"] * (2 + int(values is not None)))

    if values is None:
        data_rows = [[d["latex_name"], "\\text{" + d["description"] + "}"] for d in variables]
    else:
        data_rows = [
            [d["latex_name"], "\\text{" + d["description"] + "}", value]
            for d, value in zip(variables, values)
        ]

    summary_table.add_rows(heading_row + data_rows)
    latex_table = draw_latex(summary_table)

    return latex_table


def make_equation_table(equation_dict):
    headings = [r"", r"\text{Equation}", r"\text{Equation}"]
    heading_row = [headings]

    equation_table = Texttable()
    equation_table.set_cols_align(["c"] * 3)
    equation_table.set_cols_valign(["m"] * 3)

    data_rows = [
        [d["eq_id"], "\\text{" + d["name"] + "}", LatexPrinter().doprint(d["fancy_eq"])]
        for d in equation_dict
    ]

    equation_table.add_rows(heading_row + data_rows)
    latex_table = draw_latex(equation_table)

    return latex_table


def convert_table_to_array(latex_table):
    # Remove the \begin{table} and \end{table} commands
    latex_table = re.sub(r"\\begin{table}", "", latex_table)
    latex_table = re.sub(r"\\end{table}", "", latex_table)

    # Remove \caption and \centering
    latex_table = re.sub(r"\\caption{.*?}", "", latex_table)
    latex_table = re.sub(r"\\centering", "", latex_table)
    latex_table = re.sub(r"\\begin{center}", "", latex_table)
    latex_table = re.sub(r"\\end{center}", "", latex_table)

    # Replace \begin{tabular} and \end{tabular} with \begin{array} and \end{array}
    latex_table = re.sub(r"\\begin{tabular}", r"\\begin{array}", latex_table)
    latex_table = re.sub(r"\\end{tabular}", r"\\end{array}", latex_table)

    return latex_table


def display_latex_table(info_dict):
    table = make_table(info_dict)
    display(Latex(convert_table_to_array(table)))


def display_info_as_table(variable_info):
    variable_table = make_summary_table(variable_info)
    display(Latex(convert_table_to_array(variable_table)))


def display_eqs_as_table(equation_dicts):
    equation_table = make_equation_table(equation_dicts)
    display(Latex(convert_table_to_array(equation_table)))


def sub_info_dicts(info, sub_dict):
    out = []
    for d in info:
        name = d["name"]
        index = d.get("index", ())
        latex_name = d["latex_name"]
        description = d["description"]

        indices_subbed = tuple(x.subs(sub_dict) for x in index)
        latex_name_subbed = latex_name
        description_subbed = description

        for idx, value in sub_dict.items():
            latex_name_subbed = re.sub(
                r"([\^_,{])" + str(idx), r"\g<1>" + str(value), latex_name_subbed
            )
            latex_name_subbed = re.sub(str(idx) + r"}", str(value) + "}", latex_name_subbed)

            description_subbed = re.sub(f" {idx!s}$", f" {value!s}", description_subbed)
            description_subbed = re.sub(f" {idx!s} ", f" {value!s} ", description_subbed)

        out.append(
            {
                "name": name,
                "index": indices_subbed,
                "latex_name": latex_name_subbed,
                "description": description_subbed,
            }
        )
    return out


def get_info(name, value_dict, sectors):
    sectors = [str(x) for x in sectors]
    x = value_dict.get(symbol(name, *sectors))
    if x is None:
        x = value_dict.get(symbol(name))
    return x


def latex_print_equations(equation_info, return_latex=False):
    names, equations, equation_ids = (
        [d.get(attr) for d in equation_info] for attr in ["name", "fancy_eq", "eq_id"]
    )

    tex_output = r"\begin{align}"
    for name, eq, eq_id in zip(names, equations, equation_ids):
        tex_output += r"\text{" + name + r"} &\\"
        tex_output += sp.latex(eq) + r"&\tag{" + str(eq_id) + "}" + r"\\"
    tex_output += r"\end{align}"

    if return_latex:
        return tex_output

    else:
        display(Latex(tex_output))


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
    idata: az.InferenceData
        arviz InferenceData object with two groups -- variables and parameters -- each containing an xarray dataset
        with the same dimensions as the input arrays. Dimensions are labeled using coordinates from the CGEModel object.

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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = az.InferenceData(variables=ds_vars, parameters=ds_params)

    return idata


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
        optim_idata = az.InferenceData(
            variables=xr.Dataset(optim_var_dict, coords=coords),
            parameters=xr.Dataset(optim_param_dict, coords=coords),
        )

    return optim_idata
