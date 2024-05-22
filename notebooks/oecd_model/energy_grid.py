from cge_modeling import Variable, Parameter, Equation
from cge_modeling.production_functions import CES, dixit_stiglitz
from notebooks.oecd_model.model_building_tools import test_equations


def grid_variables():
    grid_variables = [
        # Final Output
        Variable(
            name="E_R_s",
            extend_subscript=2,
            description="Total regulated electricity supplied by the grid",
        ),
        Variable(
            name="E_U_s",
            extend_subscript=2,
            description="Total unregulated electricity supplied by the grid",
        ),
        # Intermediate Output
        Variable(
            "Y_bundled_E_R",
            dims="k",
            extend_subscript=2,
            description="Regulated grid Armington bundle for energy produced by <dim:k> sector",
        ),
        Variable(
            "Y_bundled_E_U",
            dims="k",
            extend_subscript=2,
            description="Unregulated grid Armington bundle for energy produced by <dim:k> sector",
        ),
        Variable(
            name="Y_E_R_D_d",
            dims=("k"),
            extend_subscript=3,
            description="Regulated grid demand for domestic <dim:k> energy",
        ),
        Variable(
            name="Y_E_U_D_d",
            dims=("k"),
            extend_subscript=3,
            description="Unegulated grid demand for domestic <dim:k> energy",
        ),
        Variable(
            name="Y_E_R_M_d",
            dims=("k"),
            extend_subscript=3,
            description="Regulated grid demand for imported <dim:k> energy",
        ),
        Variable(
            name="Y_E_U_M_d",
            dims=("k"),
            extend_subscript=3,
            description="Unregulated grid demand for imported <dim:k> energy",
        ),
        # Intermediate Price
        Variable(
            name="P_bundled_Y_E_R",
            dims="k",
            extend_subscript=3,
            description="Armington price <dim:k> energy in regulated grid",
        ),
        Variable(
            name="P_bundled_Y_E_U",
            dims="k",
            extend_subscript=3,
            description="Armington price <dim:k> energy in unregulated grid",
        ),
        # World price of power input
        Variable(
            name="P_M_E",
            dims=("k",),
            extend_subscript=2,
            description="Price of <dim:k> energy imports",
        ),
    ]

    return grid_variables


def grid_parameters():
    grid_parameters = [
        # CES Parameters (3)
        Parameter(
            name="alpha_G_R_M",
            dims=["k"],
            extend_subscript=True,
            description="Regulated grid home bias for <dim:k> energy purchases",
        ),
        Parameter(
            name="alpha_G_U_M",
            dims=["k"],
            extend_subscript=True,
            description="Unregulated grid home bias for <dim:k> energy purchases",
        ),
        Parameter(
            name="A_G_R_M",
            dims=["k"],
            extend_subscript=True,
            description="Regulated grid <dim:k> Armington TFP",
        ),
        Parameter(
            name="A_G_U_M",
            dims=["k"],
            extend_subscript=True,
            description="Unregulated grid <dim:k> Armington TFP",
        ),
        Parameter(
            name="epsilon_G_R_M",
            dims=["k"],
            extend_subscript=True,
            description="Regulated grid Armington elasticity for <dim:k> energy",
        ),
        Parameter(
            name="epsilon_G_U_M",
            dims=["k"],
            extend_subscript=True,
            description="Unregulated grid Armington elasticity for <dim:k> energy",
        ),
        # DX Parameters (3)
        Parameter(
            name="alpha_G_R",
            dims="k",
            extend_subscript=True,
            description="Share of <dim:k> energy in regulated grid",
        ),
        Parameter(
            name="alpha_G_U",
            dims="k",
            extend_subscript=True,
            description="Share of <dim:k> energy in unregulated grid",
        ),
        Parameter(name="A_G_R", description="Regulated grid TFP"),
        Parameter(name="A_G_U", description="Unregulated grid TFP"),
        Parameter(
            name="epsilon_G_R",
            description="Elasticity of subsitution between energy in regulated grid",
        ),
        Parameter(
            name="epsilon_G_U",
            description="Elasticity of subsitution between energy in unregultaed grid",
        ),
        # Taxes
        # Parameter(
        #     name="tau_G",
        #     dims=("s", "k"),
        #     extend_subscript=True,
        #     description="Sales tax on <dim:k> energy purchases by the grid",
        # ),
        Parameter(
            name="tau_G_R_D",
            dims=("k"),
            extend_subscript=2,
            description="Sales tax on domestic <dim:k> purchases by regulated grid",
        ),
        Parameter(
            name="tau_G_U_D",
            dims=("k"),
            extend_subscript=2,
            description="Sales tax on domestic <dim:k> purchases by unregulated grid",
        ),
        Parameter(
            name="tau_G_R_M",
            dims=("k"),
            extend_subscript=2,
            description="Sales tax on imported <dim:k> purchases by regulated grid",
        ),
        Parameter(
            name="tau_G_U_M",
            dims=("k"),
            extend_subscript=2,
            description="Sales tax on imported <dim:k> purchases by unregulated grid",
        ),
        Parameter(
            name="tau_M_E",
            dims=("k",),
            extend_subscript=True,
            description="Duty levided on <dim:k> energy imports",
        ),
    ]

    return grid_parameters


def grid_equations(coords, backend):
    regulated_import_domestic_bundle = CES(
        factors=["Y_E_R_D_d", "Y_E_R_M_d"],
        factor_prices=["(1 + tau_G_R_D) * P_Y_E", "(1 + tau_G_R_M) * P_M_E"],
        output="Y_bundled_E_R",
        output_price="P_bundled_Y_E_R",
        TFP="A_G_R_M",
        factor_shares="alpha_G_R_M",
        epsilon="epsilon_G_R_M",
        dims=("k",),
        backend=backend,
    )

    regulated_electricity_production = dixit_stiglitz(
        factors="Y_bundled_E_R",
        factor_prices="P_bundled_Y_E_R",
        output="E_R_s",
        output_price="P_E_R_tilde",
        TFP="A_G_R",
        factor_shares="alpha_G_R",
        epsilon="epsilon_G_R",
        dims="k",
        coords=coords,
        backend=backend,
    )
    unregulated_import_domestic_bundle = CES(
        factors=["Y_E_U_D_d", "Y_E_U_M_d"],
        factor_prices=["(1 + tau_G_U_D) * P_Y_E", "(1 + tau_G_U_M) * P_M_E"],
        output="Y_bundled_E_U",
        output_price="P_bundled_Y_E_U",
        TFP="A_G_U_M",
        factor_shares="alpha_G_U_M",
        epsilon="epsilon_G_U_M",
        dims=("k",),
        backend=backend,
    )

    unregulated_electricity_production = dixit_stiglitz(
        factors="Y_bundled_E_U",
        factor_prices="P_bundled_Y_E_U",
        output="E_U_s",
        output_price="P_E_U_tilde",
        TFP="A_G_U",
        factor_shares="alpha_G_U",
        epsilon="epsilon_G_U",
        dims="k",
        coords=coords,
        backend=backend,
    )

    grid_equations = [
        ## Regulated Grid
        # Armington Bundle
        Equation(
            "Regulated grid production of <dim:k> import-domestic bundle",
            regulated_import_domestic_bundle[0],
        ),
        Equation(
            "Regulated grid demand for domestic <dim:k> energy",
            regulated_import_domestic_bundle[1],
        ),
        Equation(
            "Regulated grid demand for imported <dim:k> energy",
            regulated_import_domestic_bundle[2],
        ),
        # Production of electricity
        Equation(
            "Regulated grid production of electricity",
            regulated_electricity_production[0],
        ),
        Equation(
            "Regulated grid demand for <dim:k> energy",
            regulated_electricity_production[1],
        ),
        ## Unregultaed Grid
        # Armington Bundle
        Equation(
            "Unregulated grid production of <dim:k> import-domestic bundle",
            unregulated_import_domestic_bundle[0],
        ),
        Equation(
            "Unregulated grid demand for domestic <dim:k> energy",
            unregulated_import_domestic_bundle[1],
        ),
        Equation(
            "Unregulated grid demand for imported <dim:k> energy",
            unregulated_import_domestic_bundle[2],
        ),
        # Production of electricity
        Equation(
            "Unregulated grid production of electricity",
            unregulated_electricity_production[0],
        ),
        Equation(
            "Unregulated grid demand for <dim:k> energy",
            unregulated_electricity_production[1],
        ),
    ]

    return grid_equations


def load_energy_grid(coords, backend, check_model=True):
    variables = grid_variables()
    parameters = grid_parameters()
    equations = grid_equations(coords=coords, backend=backend)
    if check_model and backend == "pytensor":
        test_equations(variables, parameters, equations, coords)

    return variables, parameters, equations
