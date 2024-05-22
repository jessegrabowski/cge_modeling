from cge_modeling.gams.from_excel import make_code_dicts
from notebooks.oecd_model.final_goods_firm import load_final_firm
from notebooks.oecd_model.energy_producing_firms import load_energy_producing_firms
from notebooks.oecd_model.energy_grid import load_energy_grid
from notebooks.oecd_model.household import load_household
from notebooks.oecd_model.investment import load_investment
from notebooks.oecd_model.government import load_government
from notebooks.oecd_model.international_trade import load_international_trade
from notebooks.oecd_model.market_clearing import load_market_clearing


def create_coords(df):
    code_dicts = make_code_dicts("data/GTAP_raw_data.xlsx")
    energy_dict = code_dicts["energy"]
    energy_names_base = list(energy_dict.values())
    energy_names = sorted(
        list(
            set(
                [
                    (
                        x.replace("baseload", "")
                        .replace("peakload", "")
                        .replace("Other", "Other power")
                        .strip()
                    )
                    for x in energy_names_base
                ]
            )
        )
    )
    energy_codes = [x.split()[0] + "P" for x in energy_names] + ["TnD"]
    sector_codes = sorted(set(df["Activities"].columns.tolist()) - set(energy_codes))
    sources = ["domestic", "import"]

    coords = {"i": sector_codes, "j": sector_codes, "k": energy_codes, "s": sources}
    return coords


def load_model(df, backend="pytensor", check_model=False):
    coords = create_coords(df)
    loading_functions = [
        load_final_firm,
        load_energy_producing_firms,
        load_energy_grid,
        load_household,
        load_investment,
        load_government,
        load_international_trade,
        load_market_clearing,
    ]
    equations, variables, parameters = [], [], []
    for loading_function in loading_functions:
        v, p, e = loading_function(coords, backend, check_model=check_model)
        variables.extend(v)
        parameters.extend(p)
        equations.extend(e)

    return variables, parameters, equations, coords
