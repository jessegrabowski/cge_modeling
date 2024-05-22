from cge_modeling import Variable, Equation
from notebooks.oecd_model.model_building_tools import test_equations, _sum_reduce
from functools import partial


def market_clearing_variables():
    clearing_variables = [Variable("resid", description="Walrasian residual")]
    return clearing_variables


def market_clearing_parameters():
    return []


def market_clearing_equations(coords, backend):
    sum_reduce = partial(_sum_reduce, coords=coords, backend=backend)

    clearing_equations = [
        # Factor markets
        Equation(
            "Labour market clearing",
            "L_s = " + "+".join([sum_reduce("L_d", "i"), sum_reduce("L_E_d", "k")]),
        ),
        Equation(
            "Capital market clearing",
            "K_s = " + "+".join([sum_reduce("K_d", "i"), sum_reduce("K_E_d", "k")]),
        ),
        Equation(
            "Regulated electricity market clearing",
            "E_R_s = "
            + "+".join([sum_reduce("E_R_d", "i"), sum_reduce("E_E_R_d", "k"), "E_H_d"]),
        ),
        Equation(
            "Unregulated electricity market clearing",
            "E_U_s = "
            + "+".join([sum_reduce("E_U_d", "i"), sum_reduce("E_E_U_d", "k")]),
        ),
        # Investment market
        Equation("Domestic investment market clearing", "I_s = S + S_G + S_M + resid"),
        # Goods markets
        Equation(
            "<dim:k> domestic energy market clearing",
            "Y_E = Y_E_R_D_d + Y_E_U_D_d + (1 - tau_Ex_E) * Ex_E",
        ),
        Equation(
            "<dim:k> import energy market clearing", "M_E = Y_E_R_M_d + Y_E_U_M_d"
        ),
        Equation(
            "Sector <dim:i> domestic goods market clearing",
            f"Y = C_D + C_G_D + I_D_d + {sum_reduce('X_D', 'j', axis=1)} + {sum_reduce('X_E_D', 'k', axis=1)} + (1 - tau_Ex) * Ex",
        ),
        Equation(
            "Sector <dim:i> import goods market clearing",
            f"M = C_M + C_G_M + I_M_d + {sum_reduce('X_M', 'j', axis=1)} + {sum_reduce('X_E_M', 'k', axis=1)}",
        ),
        # Misc.
        Equation("Total time constraint", "T = L_s + F"),
    ]

    return clearing_equations


def load_market_clearing(coords, backend, check_model=True):
    variables = market_clearing_variables()
    parameters = market_clearing_parameters()
    equations = market_clearing_equations(coords, backend)
    if check_model:
        test_equations(variables, parameters, equations, coords)
    return variables, parameters, equations
