from cge_modeling import Variable, Parameter, Equation
from cge_modeling.production_functions import CES, leontief, cobb_douglass
from notebooks.oecd_model.model_building_tools import test_equations


def energy_producer_variables():
    energy_firm_variables = [
        # Top-level output (1)
        Variable(
            name="Y_E",
            dims=("k",),
            extend_subscript=True,
            description="Final output in the <dim:k> energy sector",
        ),
        # Intermediate outputs (4)
        Variable(
            name="VC_E",
            dims="k",
            extend_subscript=True,
            description="Value-chain component of <dim:k> energy producer",
        ),
        Variable(
            name="VA_E",
            dims="k",
            extend_subscript=True,
            description="Value-added component of <dim:k> sector production",
        ),
        Variable(
            name="X_bundled_E",
            dims=("i", "k"),
            extend_subscript=1,
            description="Demand for bundle of import and domestic <dim:i> sector goods bundle by the <dim:k> sector as value-chain inputs",
        ),
        Variable(
            name="KE_E",
            dims="k",
            extend_subscript=True,
            description="Capital-Electricity bundle demanded in the <dim:k> sector",
        ),
        # Root inputs (5)
        # Variable(
        #     name="X_E",
        #     dims=("s", "i", "k"),
        #     extend_subscript=True,
        #     description="Demand for <dim:s> <dim:i> sector goods by the <dim:k> sector",
        # ),
        Variable(
            name="X_E_D",
            dims=("i", "k"),
            extend_subscript=2,
            description="Demand for domestic <dim:i> sector goods by the <dim:k> sector",
        ),
        Variable(
            name="X_E_M",
            dims=("i", "k"),
            extend_subscript=2,
            description="Demand for imported <dim:i> sector goods by the <dim:k> sector",
        ),
        Variable(
            name="L_E_d",
            dims="k",
            extend_subscript=2,
            description="Labor demand in the <dim:k> e-sector",
        ),
        Variable(
            name="K_E_d",
            dims="k",
            extend_subscript=2,
            description="Capital demand in the <dim:k> e-sector",
        ),
        Variable(
            name="E_E_R_d",
            dims="k",
            extend_subscript=2,
            description="Regulated electricity demand by <dim:k> sector",
        ),
        Variable(
            name="E_E_U_d",
            dims="k",
            extend_subscript=2,
            description="Unregulated electricity demand by <dim:k> sector",
        ),
        Variable(
            name="E_E_d",
            dims="k",
            extend_subscript=2,
            description="Electricity demand by <dim:k> sector",
        ),
        ## Prices
        # Final output
        Variable(
            name="P_Y_E",
            dims=("k",),
            extend_subscript=2,
            description="Price of energy output by <dim:k> sector",
        ),
        # Intermediate outputs (4)
        Variable(
            name="P_E_E",
            dims="k",
            extend_subscript=2,
            description="Price of electricty bundle in <dim:>",
        ),
        Variable(
            name="P_KE_E",
            dims="k",
            extend_subscript=2,
            description="Price of capital-electricity bundle formed by <dim:k> sector",
        ),
        Variable(
            name="P_VA_E",
            dims="k",
            extend_subscript=2,
            description="Price of value-add bundle in <dim:k> sector",
        ),
        Variable(
            name="P_X_E",
            dims=["i", "k"],
            extend_subscript=2,
            description="Price of <dim:i> Armington bundle formed by <dim:k> sector",
        ),
        Variable(
            name="P_VC_E",
            dims="k",
            extend_subscript=2,
            description="Price of value chain bundle in <dim:k> sector",
        ),
    ]

    return energy_firm_variables


def energy_producer_parameters():
    energy_firm_parameters = [
        Parameter(
            "alpha_EE_E",
            dims="k",
            extend_subscript=2,
            description="Share of regulated electricity in <dim:k> electricity demand",
        ),
        # CES share parameters (3)
        Parameter(
            "alpha_KE_E",
            dims="k",
            extend_subscript=2,
            description="Share of capital in <dim:k> sector capital-electricity bundle",
        ),
        Parameter(
            "alpha_VA_E",
            dims="k",
            extend_subscript=2,
            description="Share of capital-electricity in production of <dim:k> sector value-add",
        ),
        Parameter(
            "alpha_X_E",
            dims=["i", "k"],
            extend_subscript=2,
            description="Home bias for <dim:i> goods in the <dim:k> sector",
        ),
        # CES TFP parameters (3)
        Parameter(
            "A_KE_E",
            dims="k",
            extend_subscript=2,
            description="Captial-electricity factor productivity of <dim:k> producer",
        ),
        Parameter(
            "A_VA_E",
            dims="k",
            description="Total factor productivity of the <dim:k> sector",
        ),
        Parameter(
            "A_X_E",
            dims=["i", "k"],
            extend_subscript=2,
            description="Total factor productivity bundling <dim:i> goods in the <dim:k> sector ",
        ),
        # CES elasticity parameters (3)
        Parameter(
            name="epsilon_KE_E",
            extend_subscript=2,
            dims="k",
            description="Elasticity of subsitution between capital and electricity in <dim:k>",
        ),
        Parameter(
            name="epsilon_VA_E",
            extend_subscript=2,
            dims="k",
            description="Elasticity of subsitution between VA and VC in <dim:k>",
        ),
        Parameter(
            name="epsilon_X_E",
            dims=["i", "k"],
            extend_subscript=2,
            description="Elesticity of substitution between domestic and imported <dim:i> varieties in <dim:k>",
        ),
        # Leontief share parameters (3)
        Parameter(
            "psi_X_E",
            extend_subscript=2,
            dims=("i", "k"),
            description="Share of <dim:i> sector final goods in the <dim:k> value chain bundle",
        ),
        Parameter(
            "psi_VC_E",
            extend_subscript=2,
            dims="k",
            description="Share of value chain bundle in <dim:k> sector final good production",
        ),
        Parameter(
            "psi_VA_E",
            extend_subscript=2,
            dims="k",
            description="Share of value add bundle in <dim:k> sector final good production",
        ),
        # Tax rates (5)
        Parameter(
            "tau_w_E",
            extend_subscript=2,
            dims="k",
            description="Payroll rax tate in <dim:k> sector",
        ),
        Parameter(
            "tau_r_E",
            extend_subscript=2,
            dims="k",
            description="Capital use tax rate in <dim:k> sector",
        ),
        Parameter(
            "tau_E_E_U",
            extend_subscript=2,
            dims="k",
            description="Unregulated electricity tax rate in <dim:k> sector",
        ),
        Parameter(
            "tau_E_E_R",
            extend_subscript=2,
            dims="k",
            description="Regulated electricity tax rate in <dim:k> sector",
        ),
        # Parameter(
        #     "tau_X_E",
        #     extend_subscript=2,
        #     dims=("s", "i", "k"),
        #     description="VAT tax paid by <dim:k> sector on <dim:i> inputs",
        # ),
        Parameter(
            "tau_X_E_D",
            extend_subscript=3,
            dims=("i", "k"),
            description="VAT tax paid by <dim:k> sector on domestic <dim:i> inputs",
        ),
        Parameter(
            "tau_X_E_M",
            extend_subscript=3,
            dims=("i", "k"),
            description="VAT tax paid by <dim:k> sector on imported <dim:i> inputs",
        ),
        Parameter(
            "tau_Y_E",
            extend_subscript=2,
            dims=("k",),
            description="Output tax paid by <dim:k> sector",
        ),
    ]

    return energy_firm_parameters


def energy_producer_equations(coords, backend):
    energy_mix = cobb_douglass(
        factors=["E_E_R_d", "E_E_U_d"],
        factor_prices=["(1 + tau_E_E_R) * P_E_R", "(1 + tau_E_E_U) * P_E_U"],
        output="E_E_d",
        output_price="P_E_E",
        factor_shares="alpha_EE_E",
    )

    capital_electricity_bundle = CES(
        factors=["K_E_d", "E_E_d"],
        factor_prices=["(1 + tau_r_E) * r", "P_E_E"],
        output="KE_E",
        output_price="P_KE_E",
        TFP="A_KE_E",
        factor_shares="alpha_KE_E",
        epsilon="epsilon_KE_E",
        backend=backend,
    )

    value_add_bundle = CES(
        factors=["KE_E", "L_E_d"],
        factor_prices=["P_KE_E", "(1 + tau_w_E) * w"],
        output="VA_E",
        output_price="P_VA_E",
        TFP="A_VA_E",
        factor_shares="alpha_VA_E",
        epsilon="epsilon_VA_E",
        backend=backend,
    )

    import_domestic_bundle = CES(
        # factors=["X_E[0]", "X_E[1]"],
        # factor_prices=[
        #     "(1 + tau_X_E[0]) * P_Y[:, None]",
        #     "(1 + tau_X_E[1]) * P_M[:, None]",
        # ],
        factors=["X_E_D", "X_E_M"],
        factor_prices=[
            "(1 + tau_X_E_D) * P_Y[:, None]",
            "(1 + tau_X_E_M) * P_M[:, None]",
        ],
        output="X_bundled_E",
        output_price="P_X_E",
        TFP="A_X_E",
        factor_shares="alpha_X_E",
        epsilon="epsilon_X_E",
        dims=["i", "k"],
        backend=backend,
        expand_price_dim=None,
    )

    value_chain_bundle = leontief(
        factors="X_bundled_E",
        factor_prices="P_X_E",
        factor_shares="psi_X_E",
        output="VC_E",
        output_price="P_VC_E",
        dims=["i", "k"],
        sum_dim="i",
        expand_price_dim=False,
        coords=coords,
        transpose_output=False,
        backend=backend,
    )

    final_goods = leontief(
        factors=["VC_E", "VA_E"],
        factor_prices=["P_VC_E", "P_VA_E"],
        factor_shares=["psi_VC_E", "psi_VA_E"],
        output="Y_E",
        output_price="((1 - tau_Y_E) * P_Y_E)",
        dims="k",
        coords=coords,
        backend=backend,
    )

    energy_firm_equations = [
        # Value chain bundle
        Equation(
            "Sector <dim:k> production of intermediate goods bundle",
            value_chain_bundle[0],
        ),
        Equation(
            "Sector <dim:k> demand for sector <dim:i> intermediate input",
            value_chain_bundle[1],
        ),
        # Armington Bundle
        Equation(
            "Sector <dim:k> production of <dim:i> import-domestic bundle",
            import_domestic_bundle[0],
        ),
        Equation(
            "Sector <dim:k> demand for domestic <dim:i> inputs",
            import_domestic_bundle[1],
        ),
        Equation(
            "Sector <dim:k> demand for imported <dim:i> inputs",
            import_domestic_bundle[2],
        ),
        # Energy aggregation
        Equation(
            "Producer <dim:k> combination of regulated and unregulated energy",
            energy_mix[0],
        ),
        Equation("Producer <dim:k> demand for regulated energy", energy_mix[1]),
        Equation("Producer <dim:k> demand for unregulated energy", energy_mix[2]),
        # Capital Labour aggregation
        Equation(
            "Producer <dim:k> production of capital-electricity bundle",
            capital_electricity_bundle[0],
        ),
        Equation("Producer <dim:k> demand for captial", capital_electricity_bundle[1]),
        Equation(
            "Producer <dim:k> demand for electricity", capital_electricity_bundle[2]
        ),
        # Value add bundle
        Equation("Sector <dim:k> production of value add", value_add_bundle[0]),
        Equation(
            "Sector <dim:k> demand for capital-electricity bundle", value_add_bundle[1]
        ),
        Equation("Sector <dim:k> demand for labor", value_add_bundle[2]),
        # Sector Final Goods
        Equation("Final good production of sector <dim:k>", final_goods[0]),
        Equation("Sector <dim:k> demand for intermediate goods bundle", final_goods[1]),
        Equation("Sector <dim:k> demand for value added", final_goods[2]),
    ]

    return energy_firm_equations


def load_energy_producing_firms(coords, backend, check_model=True):
    variables = energy_producer_variables()
    parameters = energy_producer_parameters()
    equations = energy_producer_equations(coords=coords, backend=backend)
    if check_model and backend == "pytensor":
        test_equations(variables, parameters, equations, coords)

    return variables, parameters, equations
