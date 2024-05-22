from cge_modeling import Variable, Parameter, Equation
from cge_modeling.production_functions import CES, dixit_stiglitz
from notebooks.oecd_model.model_building_tools import test_equations


def household_variables():
    household_variables = [
        # Inputs to shopping
        Variable(
            name="C_D",  # Called C_M and C_D above
            dims=("i"),
            description="Household demand for domestic <dim:i> goods",
        ),
        Variable(
            name="C_M",  # Called C_M and C_D above
            dims=("i"),
            description="Household demand for imported <dim:i> goods",
        ),
        Variable(
            name="C_bundled",
            dims="i",
            description="Sector <dim:i> import-domestic bundle formed by households",
        ),
        # Inputs to final consumption basket
        Variable(
            name="E_H_d",
            extend_subscript=True,
            description="Household demand for electricity",
        ),
        Variable(
            name="C_total",
            latex_name=r"\bar{C}",
            description="Household final consumption bundle",
        ),
        # Spending variables
        Variable(name="F", description="Household leisure time"),
        Variable(name="S", description="Household savings"),
        Variable(name="CE", description="Consumption-Electricity bundle"),
        # Income definitions
        Variable(name="L_s", description="Household supply of labor"),
        Variable(
            name="income",  # Gross Income above
            latex_name="Omega",
            description="Household income, before taxes",
        ),
        Variable(
            name="net_income",
            latex_name=r"\hat{\Omega}",
            description="Household income, after taxes",
        ),
        # Prices
        Variable(
            name="P_bundled_C",
            dims="i",
            extend_subscript=1,
            description="Price of the import-domestic bundle for <dim:i> goods",
        ),
        Variable(name="P", description="Core consumer price index, excluding energy"),
        Variable(
            name="P_CE", description="Consumer Price Index", extend_subscript=True
        ),
        # Utility
        Variable(name="U", description="Household utility"),
    ]

    return household_variables


def household_parameters():
    household_parameters = [
        # Factor endowments
        Parameter(name="T", description="Time endowment"),
        Parameter(name="K_s", description="Capital stock"),
        # Armington Parameters
        Parameter(
            name="alpha_C_M",
            dims=("i",),
            description="Household home bias in for <dim:i> goods",
        ),
        Parameter(
            name="A_C_M",
            dims=("i",),
            description="Household total factor productivity bundling domestic and imported <dim:i> goods",
        ),
        Parameter(
            name="epsilon_C_M",
            dims=("i",),
            description="Household elasticity of substitution between domestic and imported <dim:i> goods",
        ),
        # Shopping CES parameters
        Parameter(
            name="alpha_C",
            dims="i",
            description="Household elasticity of consumption utility for <dim:i> sector goods",
        ),
        Parameter(
            name="A_C", description="Household total factor productivity of shopping"
        ),
        Parameter(
            name="epsilon_C",
            description="Elasticity of substitution between varieties in final consumption basket",
        ),
        # Electricity-Consumption bundling parameters
        Parameter(
            name="alpha_CE",
            description="Share of consumption in final spending",
        ),
        Parameter(
            name="A_CE",
            description="Household total factor productivity of final bundling",
        ),
        Parameter(
            name="epsilon_CE",
            description="Elasticity of substitution between consumption and electricity",
        ),
        # Utility Parameters
        Parameter(
            name="sigma_C",
            description="Arrow-Pratt risk averson",
        ),
        Parameter(
            name="sigma_L",
            description="Inverse Frisch elasticity between work and leisure",
        ),
        Parameter(name="Theta", description="Household labor dispreference parameter"),
        # Miscellaneous
        Parameter(
            name="mpc",
            latex_name="phi",
            description="Household marginal propensity to consume",
        ),
        # Latent price influences
        Parameter(
            name="P_C_D_bar",
            dims=["i"],
            description="Unobserved shifter of domestic <dim:i> good prices facing households",
        ),
        Parameter(
            name="P_C_M_bar",
            dims=["i"],
            description="Unobserved shifter of imported <dim:i> good prices facing households",
        ),
        # Taxes
        Parameter(
            name="tau_C_D",
            dims=["i"],
            description="Sales tax paid by households for domestic <dim:i> goods",
        ),
        Parameter(
            name="tau_C_M",
            dims=["i"],
            description="Sales tax paid by households for imported <dim:i> goods",
        ),
        Parameter(
            name="tau_E_H", description="Sales tax paid by households for electricity"
        ),
        Parameter(
            name="tau_w_income", description="Income tax from wages paid by households"
        ),
        Parameter(
            name="tau_r_income",
            description="Income tax from capital paid by households",
        ),
    ]

    return household_parameters


def household_equations(coords, backend):
    import_domestic_bundle = CES(
        factors=["C_D", "C_M"],
        factor_prices=[
            "((1 + tau_C_D) * (P_Y + P_C_D_bar))",
            "((1 + tau_C_M) * (P_M + P_C_M_bar))",
        ],
        output="C_bundled",
        output_price="P_bundled_C",
        TFP="A_C_M",
        factor_shares="alpha_C_M",
        epsilon="epsilon_C_M",
        dims=("i",),
        backend=backend,
    )

    shopping_function = dixit_stiglitz(
        factors="C_bundled",
        factor_prices="P_bundled_C",
        output="C_total",
        output_price="P",
        TFP="A_C",
        factor_shares="alpha_C",
        epsilon="epsilon_C",
        dims="i",
        coords=coords,
        backend=backend,
    )

    consumption_electricity_bundle = CES(
        factors=["C_total", "E_H_d"],
        factor_prices=["P", "((1 + tau_E_H) * P_E_R)"],
        output="CE",
        output_price="P_CE",
        TFP="A_CE",
        factor_shares="alpha_CE",
        epsilon="epsilon_CE",
        backend=backend,
    )

    household_equations = [
        # Import-Domestic Bundling
        Equation(
            "Household production of <dim:i> import-domestic bundle",
            import_domestic_bundle[0],
        ),
        Equation(
            "Household demand for domestic <dim:i> goods", import_domestic_bundle[1]
        ),
        Equation(
            "Household demand for imported <dim:i> goods", import_domestic_bundle[2]
        ),
        # Production of consumption basket (shopping)
        Equation("Final consumption bundle", shopping_function[0]),
        Equation("Household demand for <dim:i> goods", shopping_function[1]),
        # # Basket-Electricity bundle
        Equation("Goods-Electricity bundle", consumption_electricity_bundle[0]),
        Equation(
            "Household demand for goods basket", consumption_electricity_bundle[1]
        ),
        Equation("Household demand for electricity", consumption_electricity_bundle[2]),
        # Income
        Equation(
            "Household pre-tax income",
            "income = w * L_s + r * K_s",
        ),
        Equation(
            "Household after-tax income",
            "net_income = (1 - tau_w_income) * w * L_s + (1 - tau_r_income) * r * K_s",
        ),
        Equation("Household budget constraint", "CE * P_CE = mpc * net_income"),
        Equation(
            "Household utility",
            "U = CE ** (1 - sigma_C) / (1 - sigma_C) + F ** (1 - sigma_L) / (1 - sigma_L)",
        ),
        Equation(
            "Household supply of labor",
            "F ** -sigma_L / CE ** -sigma_C = w / P_CE / Theta",
        ),
        Equation("Household savings", "S = (1 - mpc) * net_income"),
    ]

    return household_equations


def load_household(coords, backend, check_model=True):
    variables = household_variables()
    parameters = household_parameters()
    equations = household_equations(coords=coords, backend=backend)
    if check_model and backend == "pytensor":
        test_equations(variables, parameters, equations, coords)

    return variables, parameters, equations
