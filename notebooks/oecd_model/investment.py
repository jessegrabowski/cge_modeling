from cge_modeling import Variable, Parameter, Equation
from cge_modeling.production_functions import CES
from notebooks.oecd_model.model_building_tools import test_equations


def investment_variables():
    investment_variables = [
        Variable(name="I_s", description="Total supply of capital"),
        Variable(
            name="I_D_d",
            dims=("i"),
            extend_subscript=True,
            description="Demand for domestic investment capital by the <dim:i> sector",
        ),
        Variable(
            name="I_M_d",
            dims=("i"),
            extend_subscript=True,
            description="Demand for imported investment capital by the <dim:i> sector",
        ),
        Variable(
            name="I_bundle",
            dims="i",
            description="Armington bundle for investment goods by <dim:i> sector",
        ),
        Variable(
            name="P_I",
            dims="i",
            extend_subscript=True,
            description="Price of Armington investment bundle in <dim:i> sector",
        ),
        Variable(
            name="I_E_D_d",
            dims=("k"),
            extend_subscript=True,
            description="Demand for domestic investment capital by the <dim:k> sector",
        ),
        Variable(
            name="I_E_M_d",
            dims=("k"),
            extend_subscript=True,
            description="Demand for imported investment capital by the <dim:k> sector",
        ),
        Variable(
            name="I_bundle_E",
            dims="k",
            extend_subscript=1,
            description="Armington bundle for investment goods by <dim:k> sector",
        ),
        Variable(
            name="P_I_E",
            dims="k",
            extend_subscript=True,
            description="Price of Armington investment bundle in <dim:k> sector",
        ),
    ]

    return investment_variables


def investment_parameters():
    investment_parameters = [
        Parameter("S_G", description="Supply of government investment capital"),
        Parameter("S_M", description="Supply of international investment capital"),
        Parameter(
            name="alpha_I_M",
            dims=("i",),
            description="Sector <dim:i> bias for domestic investment goods",
        ),
        Parameter(
            name="A_I_M",
            dims=("i",),
            extend_subscript=True,
            description="Sector <dim:i> TFP in combining import and domestic investment goods",
        ),
        Parameter(
            name="epsilon_I_M",
            dims=("i",),
            extend_subscript=True,
            description="<dim:i> elasticity of substitution between foreign and domestic investment goods",
        ),
        Parameter(
            name="alpha_I_E_M",
            dims=("k",),
            extend_subscript=True,
            description="Sector <dim:k> bias for domestic investment goods",
        ),
        Parameter(
            name="A_I_E_M",
            dims=("k",),
            extend_subscript=True,
            description="Sector <dim:k> TFP in combining import and domestic investment goods",
        ),
        Parameter(
            name="epsilon_I_E_M",
            dims=("k",),
            extend_subscript=True,
            description="<dim:k> elasticity of substitution between foreign and domestic investment goods",
        ),
        Parameter("alpha_I", dims="i", description="<dim:i> share of total investment"),
        Parameter(
            "alpha_I_E", dims="k", description="<dim:k> share of total investment"
        ),
        Parameter(
            "tau_I_D",
            dims=("i"),
            description="Tax on domestic capital in sector <dim:i>",
        ),
        Parameter(
            "tau_I_M",
            dims=("i"),
            description="Tax on imported capital in sector <dim:i>",
        ),
        Parameter(
            "tau_I_E_D",
            dims=("k"),
            description="Tax on domestic capital investment in sector <dim:k>",
        ),
        Parameter(
            "tau_I_E_M",
            dims=("k"),
            description="Tax on imported capital investment in sector <dim:k>",
        ),
    ]

    return investment_parameters


def investment_equations(coords, backend):
    import_domestic_bundle = CES(
        factors=["I_D_d", "I_M_d"],
        factor_prices=["(1 + tau_I_D) * P_Y", "(1 + tau_I_M) * P_M"],
        output="I_bundle",
        output_price="P_I",
        TFP="A_I_M",
        factor_shares="alpha_I_M",
        epsilon="epsilon_I_M",
        dims="i",
        backend=backend,
    )

    energy_import_domestic_bundle = CES(
        factors=["I_E_D_d", "I_E_M_d"],
        factor_prices=["(1 + tau_I_E_D) * P_Y_E", "(1 + tau_I_E_M) * P_M_E"],
        output="I_bundle_E",
        output_price="P_I_E",
        TFP="A_I_E_M",
        factor_shares="alpha_I_E_M",
        epsilon="epsilon_I_E_M",
        dims="i",
        backend=backend,
    )

    investment_equations = [
        Equation(
            "<dim:i> sector demand for domestic-import capital bundle",
            import_domestic_bundle[0],
        ),
        Equation(
            "<dim:i> sector demand for domestic capital", import_domestic_bundle[1]
        ),
        Equation(
            "<dim:i> sector demand for imported capital", import_domestic_bundle[2]
        ),
        Equation(
            "<dim:k> sector demand for domestic-import capital bundle",
            energy_import_domestic_bundle[0],
        ),
        Equation(
            "<dim:k> sector demand for domestic capital",
            energy_import_domestic_bundle[1],
        ),
        Equation(
            "<dim:k> sector demand for imported capital",
            energy_import_domestic_bundle[2],
        ),
        Equation("<dim:i> sector demand for capital", "P_I * I_bundle = alpha_I * I_s"),
        Equation(
            "<dim:k> sector demand for capital", "P_I_E * I_bundle_E = alpha_I_E * I_s"
        ),
    ]

    return investment_equations


def load_investment(coords, backend, check_model=True):
    variables = investment_variables()
    parameters = investment_parameters()
    equations = investment_equations(coords=coords, backend=backend)
    if check_model and backend == "pytensor":
        test_equations(variables, parameters, equations, coords)

    return variables, parameters, equations
