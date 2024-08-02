from functools import reduce

import pandas as pd

from cge_modeling.gams.excel_constants import COORDS_BY_SHEET, SHEET_GROUPS
from cge_modeling.gams.gams_constants import ENERGY


def reverse_dictionary(d):
    return {v: k for k, v in d.items()}


def repeat(f, x, n):
    for _ in range(n):
        x = f(x)
    return x


def determine_indices(coords):
    n_headers = None if coords["columns"] is None else len(coords["columns"])
    n_index = len(coords["index"])

    return n_headers, n_index


def set_df_index_names(df, sheet, coords):
    n_headers, n_index = determine_indices(coords)
    if n_index > 1:
        df.index.names = coords["index"]
    else:
        df.index.name = coords["index"][0]
    if n_headers > 1:
        df.columns.names = coords["columns"]
    else:
        df.columns.name = coords["columns"][0]

    return df


def make_tokens_from_sheet_name(sheet, coords):
    n_headers, n_index = determine_indices(coords)
    if n_headers is None:
        return
    tokens = [
        x.strip() for token in sheet.split(",") for x in token.split("by") if len(x.strip()) > 0
    ]
    return tokens


def rebuild_df_index_cols(df, sheet, coords):
    tokens = make_tokens_from_sheet_name(sheet, coords)
    if tokens is None:
        return df, None, sheet, tokens

    groups = coords["groups"]
    index_cols = df.index.names
    df = df.reset_index(drop=False)
    if groups is None:
        name = " ".join(tokens)
    else:
        name = tokens.pop(0)
        for group, token in zip(groups, tokens):
            df[group] = token
            index_cols += [group]
    return df, index_cols, name, tokens


def add_duplicate_indices(df, missing, all_indices):
    if len(missing) == 0:
        return df

    df = df.reset_index()
    assert isinstance(df, pd.DataFrame)

    for col in missing:
        if col == "activity":
            df["activity"] = df["commodity"]
        assert "activity" in df

    return df


def get_missing_indices(df, all_indices):
    return all_indices - set(df.index.names)


def concatenate_group_stack(dfs):
    all_indices = reduce(lambda left, right: left.union(right), [set(x.index.names) for x in dfs])
    missing_indices = [get_missing_indices(df, all_indices) for df in dfs]

    if all([len(x) == 0 for x in missing_indices]):
        return pd.concat(dfs, axis=0)

    else:
        dfs = [
            add_duplicate_indices(df.copy(), missing, all_indices)
            for df, missing in zip(dfs, missing_indices)
        ]
        dfs = [df.reset_index() if isinstance(df, pd.Series) else df for df in dfs]

        assert all([all([x in df for x in all_indices]) for df in dfs])

        return pd.concat(dfs, axis=0).set_index(list(all_indices))[0]


def load_sheet(path, sheet, coords):
    n_headers, n_index = determine_indices(coords)
    df = pd.read_excel(path, sheet_name=sheet, index_col=list(range(n_index)))
    df = set_df_index_names(df, sheet, coords)
    df, index_cols, name, tokens = rebuild_df_index_cols(df, sheet, coords)

    if index_cols:
        df.set_index(sorted(index_cols), inplace=True)
        df = repeat(lambda x: x.stack(), df, n_headers if n_headers is not None else 1)

    return df, name, tokens


def get_load_type(x):
    if x.endswith("BL"):
        return "BL"
    elif x.endswith("P"):
        return "P"
    return None


def list_replace(input_list, sub_dict):
    new_list = input_list.copy()
    for old, new in sub_dict.items():
        idx = input_list.index(old)
        new_list[idx] = new
    return new_list


def load_all_sheets_as_xarray(path):
    data_vars = {}
    for sheets in SHEET_GROUPS:
        group_stack = []
        group_dims = []
        for sheet in sheets:
            coords = COORDS_BY_SHEET[sheet]
            df, name, tokens = load_sheet(path, sheet, coords)

            group_stack.append(df)
            group_dims.append(tokens)
        if len(group_stack) > 1:
            data_df = concatenate_group_stack(group_stack)
        else:
            data_df = group_stack[0]
            if isinstance(data_df, pd.DataFrame):
                data_df = data_df.iloc[:, 0]

        data_vars[name.title()] = data_df.to_xarray()

    return data_vars


def make_code_dicts(path):
    code_to_country = (
        pd.read_excel(
            path,
            sheet_name="Country Codes",
            index_col=0,
            header=None,
            names=["code", "country"],
        )
        .iloc[:, 0]
        .to_dict()
    )

    code_to_commodity = (
        pd.read_excel(
            path,
            sheet_name="Commodities",
            index_col=0,
            header=None,
            names=["code", "commodity"],
        )
        .iloc[:, 0]
        .to_dict()
    )
    commodity_to_code = reverse_dictionary(code_to_commodity)

    code_to_activity = code_to_commodity.copy()

    code_to_energy = {commodity_to_code[k]: k for k in ENERGY}

    code_to_labor = (
        pd.read_excel(
            path,
            sheet_name="Labor Types",
            index_col=0,
            header=None,
            names=["code", "labor"],
        )
        .iloc[:, 0]
        .to_dict()
    )

    code_to_factor = (
        pd.read_excel(
            path,
            sheet_name="Factor Types",
            index_col=0,
            header=None,
            names=["code", "factor"],
        )
        .iloc[:, 0]
        .to_dict()
    )

    code_to_agent = {
        "Firms": "Firms",
        "Govt": "Government",
        "HH": "Household",
        "Inv": "Investment",
        "Grid": "Grid",
    }

    code_to_price = {"BP": "base price", "PP": "purchaser price"}

    code_to_load = {"BL": "base load", "P": "peak load"}

    code_to_margin = {
        "OTP": "Transport nec",
        "WTP": "Water transport",
        "ATP": "Air transport",
    }

    names = [
        "country",
        "source",
        "destination",
        "commodity",
        "activity",
        "energy",
        "load",
        "labor",
        "factor",
        "agent",
        "price",
        "margin",
    ]
    codes = [
        code_to_country,
        code_to_country,
        code_to_country,
        code_to_commodity,
        code_to_activity,
        code_to_energy,
        code_to_load,
        code_to_labor,
        code_to_factor,
        code_to_agent,
        code_to_price,
        code_to_margin,
    ]

    CODES_DICT = {
        name: x.iloc[:, 0].to_dict() if isinstance(x, pd.DataFrame) else x
        for name, x in zip(names, codes)
    }

    return CODES_DICT
