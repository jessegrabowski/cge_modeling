from cge_modeling import Parameter
from notebooks.oecd_model.model_building_tools import test_equations


def trade_variables():
    # trade_variables = [Variable(name="P_Ex"),
    #                    Variable(name="P_Ex_M")]
    trade_variables = []
    return trade_variables


def trade_parameters():
    trade_parameters = [
        Parameter(
            name="M", dims="i", description="Supply of imports from <dim:i> sector"
        ),
        Parameter(
            name="Ex", dims="i", description="Supply of exports from <dim:i> sector"
        ),
        Parameter(
            name="M_E",
            dims="k",
            extend_subscript=1,
            description="Supply of energy imports from <dim:k> sector",
        ),
        Parameter(
            name="Ex_E",
            dims="k",
            extend_subscript=1,
            description="Supply of energy exports from <dim:k> sector",
        ),
    ]

    return trade_parameters


def trade_equations(coords, backend):
    return []


def load_international_trade(coords, backend, check_model=True):
    variables = trade_variables()
    parameters = trade_parameters()
    equations = trade_equations(coords, backend)
    if check_model:
        test_equations(variables, parameters, equations, coords)
    return variables, parameters, equations
