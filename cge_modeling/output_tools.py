from latextable import draw_latex
from texttable import Texttable
import re
from IPython.display import display, Latex

from cge_modeling.sympy_tools import symbol, sub_all_eqs
import sympy as sp


def variables_to_latex(variables, values=None):
    headings = [r'\text{Symbol}', r'\text{Description}']
    if values is not None:
        headings += [r'\text{Calibrated Value}']
    heading_row = [headings]

    variable_table = Texttable()
    variable_table.set_cols_align(['c'] * (2 + int(values is not None)))
    variable_table.set_cols_valign(['m'] * (2 + int(values is not None)))

    if values is None:
        data_rows = [[d['latex_name'], '\\text{' + d['description'] + '}'] for d in variables]
    else:
        data_rows = [[d['latex_name'], '\\text{' + d['description'] + '}', value] for d, value in
                     zip(variables, values)]

    variable_table.add_rows(heading_row + data_rows)
    latex_table = draw_latex(variable_table)

    return latex_table


def convert_table_to_array(latex_table):
    # Remove the \begin{table} and \end{table} commands
    latex_table = re.sub(r'\\begin{table}', '', latex_table)
    latex_table = re.sub(r'\\end{table}', '', latex_table)

    # Remove \caption and \centering
    latex_table = re.sub(r'\\caption{.*?}', '', latex_table)
    latex_table = re.sub(r'\\centering', '', latex_table)
    latex_table = re.sub(r'\\begin{center}', '', latex_table)
    latex_table = re.sub(r'\\end{center}', '', latex_table)

    # Replace \begin{tabular} and \end{tabular} with \begin{array} and \end{array}
    latex_table = re.sub(r'\\begin{tabular}', r'\\begin{array}', latex_table)
    latex_table = re.sub(r'\\end{tabular}', r'\\end{array}', latex_table)

    return latex_table


def display_info_as_table(variable_info):
    variable_table = variables_to_latex(variable_info)
    display(Latex(convert_table_to_array(variable_table)))


def sub_info_dicts(info, sub_dict):
    out = []
    for d in info:
        name = d['name']
        index = d.get('index', ())
        latex_name = d['latex_name']
        description = d['description']

        indices_subbed = tuple([x.subs(sub_dict) for x in index])
        latex_name_subbed = latex_name
        description_subbed = description

        for idx, value in sub_dict.items():
            latex_name_subbed = re.sub(r'([\^_,{])' + str(idx), '\g<1>' + str(value), latex_name_subbed)
            latex_name_subbed = re.sub(str(idx) + r'}', str(value) + '}', latex_name_subbed)

            description_subbed = re.sub(f' {str(idx)}$', f' {str(value)}', description_subbed)
            description_subbed = re.sub(f' {str(idx)} ', f' {str(value)} ', description_subbed)

        out.append(
            {
                'name': name,
                'index': indices_subbed,
                'latex_name': latex_name_subbed,
                'description': description_subbed
            }
        )
    return out


def get_info(name, value_dict, sectors):
    sectors = [str(x) for x in sectors]
    x = value_dict.get(symbol(name, *sectors))
    if x is None:
        x = value_dict.get(symbol(name))
    return x


def latex_print_equations(equation_info, variables, variable_info, parameters, param_info, return_latex=False):
    var_subs = {var: sp.Symbol(d.get('latex_name'), **var._assumptions) for d, var in zip(variable_info, variables)}
    param_subs = {param: sp.Symbol(d.get('latex_name'), **param._assumptions) for d, param in
                  zip(param_info, parameters)}

    sub_dict = var_subs | param_subs
    names, equations = [d.get('name') for d in equation_info], [d.get('equation') for d in equation_info]
    subbed_equations = sub_all_eqs(equations, sub_dict)

    tex_output = r'\begin{align}'
    for i, (name, eq) in enumerate(zip(names, subbed_equations)):
        tex_output += '\t' + r'\text{' + name + r'} \\'
        tex_output += '\t' + sp.latex(eq) + r'\tag{' + str(i + 1) + '}' + r'\\ \\'

    tex_output += r'\end{align}'
    if return_latex:
        return tex_output

    else:
        display(Latex(tex_output))

