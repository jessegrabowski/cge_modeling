from cge_modeling import Variable, Parameter, Equation
from cge_modeling.production_functions import CES, leontief, cobb_douglass
from notebooks.oecd_model.model_building_tools import test_equations


def final_goods_variables():
    return [
        # Final Output
        Variable(
            name="Y", dims=("i"), description="Final output in the <dim:i> sector"
        ),
        # Intermediate Outputs (4)
        Variable(
            name="X_bundled",
            dims=("i", "j"),
            description="Demand for bundle of import and domestic <dim:i> sector goods bundle by the <dim:j> sector as value-chain inputs",
        ),
        Variable(
            name="VA",
            dims="i",
            description="Value-added component of <dim:i> sector production",
        ),
        Variable(
            name="VC",
            dims="i",
            description="Value-chain component of <dim:i> sector production",
        ),
        Variable(
            name="KE",
            dims="i",
            description="Labor Capital demand in the <dim:i> sector",
        ),
        Variable(
            name="X_D",
            dims=("i", "j"),
            description="Demand for domestic <dim:i> sector goods by the <dim:j> sector as value-chain inputs",
        ),
        Variable(
            name="X_M",
            dims=("i", "j"),
            description="Demand for imported <dim:i> sector goods by the <dim:j> sector as value-chain inputs",
        ),
        Variable(
            name="L_d",
            dims="i",
            extend_subscript=True,
            description="Labor demand in the <dim:i> sector",
        ),
        Variable(
            name="K_d",
            dims="i",
            extend_subscript=True,
            description="Capital demand in the <dim:i> sector",
        ),
        Variable(
            name="E_R_d",
            dims="i",
            extend_subscript=True,
            description="Regulated energy demand in the <dim:i> sector",
        ),
        Variable(
            name="E_U_d",
            dims="i",
            extend_subscript=True,
            description="Unregulated energy demand in the <dim:i> sector",
        ),
        Variable(
            name="E_d",
            dims="i",
            extend_subscript=True,
            description="Energy demand in the <dim:i> sector",
        ),
        ## Prices (9)
        # Root factors (3)
        Variable(name="r", description="Rental rate of capital"),
        Variable(
            name="P_E_R_tilde",
            extend_subscript=True,
            description="Price of electricity from regulated grid",
        ),
        Variable(
            name="P_E_U_tilde",
            extend_subscript=True,
            description="Price of electricity from unregulated grid",
        ),
        Variable(
            name="P_E_R",
            extend_subscript=True,
            description="Price of electricity from regulated grid, including shifter",
        ),
        Variable(
            name="P_E_U",
            extend_subscript=True,
            description="Price of electricity from unregulated grid, including shifter",
        ),
        Variable(
            name="P_E",
            extend_subscript=True,
            dims="i",
            description="Price of electricity bundle in <dim:i>",
        ),
        # Intermediate Outputs (4)
        Variable(
            name="P_X",
            dims=["i", "j"],
            extend_subscript=True,
            description="Price of sector <dim:i> Armington mix in sector <dim:j>",
        ),
        Variable(
            name="P_KE",
            dims="i",
            extend_subscript=True,
            description="Price of the capital-labor bundle in the <dim:i> sector",
        ),
        Variable(
            name="P_VA",
            dims="i",
            extend_subscript=True,
            description="Price of the value-add component in the <dim:i> sector",
        ),
        Variable(
            name="P_VC",
            dims="i",
            extend_subscript=True,
            description="Price of the value-chain component in the <dim:i> sector",
        ),
        # Final Prices (2) [domestic + foreign]
        Variable(
            name="P_Y",
            dims=["i"],
            extend_subscript=True,
            description="Final good price in the <dim:i> sector",
        ),
        Variable(
            name="P_M",
            dims=["i"],
            extend_subscript=True,
            description="Price of imported final goods from the <dim:i> sector",
        ),
    ]


def final_goods_parameters():
    return [
        # Wages are the numeraire; make it a parameter
        Parameter(name="w", description="Wage level"),
        # CES share parameters (3)
        Parameter(
            "alpha_KE",
            dims="i",
            extend_subscript=True,
            description="Share of capital in production of the <dim:i> producer capital labour bundle",
        ),
        Parameter(
            "alpha_VA",
            dims="i",
            extend_subscript=True,
            description="Share of capital in production of the <dim:i> sector value-add bundle",
        ),
        Parameter(
            "alpha_X",
            dims=("i", "j"),
            extend_subscript=True,
            description="Bias for domestic <dim:i> goods in <dim:j>",
        ),
        Parameter(
            "alpha_EE",
            dims=("i",),
            extend_subscript=1,
            description="Share of regulated energy in sector <dim:i> energy usage",
        ),
        # CES TFP Parameters (3)
        Parameter(
            "A_VA",
            dims="i",
            description="Total factor productivity of the <dim:i> sector",
        ),
        Parameter(
            "A_KE",
            dims="i",
            extend_subscript=True,
            description="Total factor productivity of Capital-Energy Bundling in the <dim:i> sector",
        ),
        Parameter(
            "A_X",
            dims=["i", "j"],
            extend_subscript=True,
            description="Total factor productivity of bundling <dim:i> goods in <dim:j> sector",
        ),
        # CES Elasticity Parameters (3)
        Parameter(
            name="epsilon_VA",
            extend_subscript=True,
            dims="i",
            description="Elasticity of subsitution between input factors in <dim:i> sector VA bundle",
        ),
        Parameter(
            name="epsilon_KE",
            extend_subscript=True,
            dims="i",
            description="Elasticity of subsitution between input factors in <dim:i> producer KL bundle",
        ),
        Parameter(
            name="epsilon_X",
            extend_subscript=True,
            dims=["i", "j"],
            description="Elasticity between home and imported good varieties of <dim:i> goods in sector <dim:j>",
        ),
        # Leontief share parameters (3)
        Parameter(
            "psi_X",
            dims=("i", "j"),
            extend_subscript=True,
            description="Share of <dim:i> demanded as input to <dim:j> value chain bundle",
        ),
        Parameter(
            "psi_VA",
            extend_subscript=True,
            dims="i",
            description="Share of value-add bundle in <dim:i> sector final good production",
        ),
        Parameter(
            "psi_VC",
            extend_subscript=True,
            dims="i",
            description="Share of value chain bundle in <dim:i> sector final good production",
        ),
        # Tax rates (5)
        Parameter(
            "tau_w",
            extend_subscript=True,
            dims="i",
            description="Payroll rax tate in <dim:i> sector",
        ),
        Parameter(
            "tau_r",
            extend_subscript=True,
            dims="i",
            description="Capital use tax tax rate in <dim:i> sector",
        ),
        Parameter(
            "tau_E_U",
            extend_subscript=True,
            dims="i",
            description="Unregulated electricity tax rate in <dim:i> sector",
        ),
        Parameter(
            "tau_E_R",
            extend_subscript=True,
            dims="i",
            description="Regulated electricity tax rate in <dim:i> sector",
        ),
        # Parameter(
        #     "tau_X",
        #     extend_subscript=True,
        #     dims=("s", "i", "j"),
        #     description="VAT tax paid by <dim:j> on <dim:s> <dim:i> sector inputs",
        # ),
        Parameter(
            "tau_X_D",
            extend_subscript=True,
            dims=("i", "j"),
            description="VAT tax paid by <dim:j> on domestic <dim:i> sector inputs",
        ),
        Parameter(
            "tau_X_M",
            extend_subscript=True,
            dims=("i", "j"),
            description="VAT tax paid by <dim:j> on imported <dim:i> sector inputs",
        ),
        Parameter(
            "tau_Y",
            extend_subscript=True,
            dims=("i",),
            description="Output tax paid my <dim:i>",
        ),
        Parameter(
            name="tau_M",
            dims=("i",),
            extend_subscript=True,
            description="Import duty on consumption goods",
        ),
        Parameter(
            "P_E_R_bar",
            description="Unobserved shifter of unregulated electricity price",
        ),
        Parameter(
            "P_E_U_bar", description="Unobserved shifter of regulated electricity price"
        ),
    ]


def final_goods_equations(coords, backend="pytensor"):
    energy_mix = cobb_douglass(
        factors=["E_R_d", "E_U_d"],
        factor_prices=["(1 + tau_E_R) * P_E_R", "(1 + tau_E_U) * P_E_U"],
        output="E_d",
        output_price="P_E",
        factor_shares="alpha_EE",
    )

    capital_energy_bundle = CES(
        factors=["K_d", "E_d"],
        factor_prices=["(1 + tau_r) * r", "P_E"],
        output="KE",
        output_price="P_KE",
        TFP="A_KE",
        factor_shares="alpha_KE",
        epsilon="epsilon_KE",
        backend=backend,
    )

    value_add_bundle = CES(
        factors=["KE", "L_d"],
        factor_prices=["P_KE", "(1 + tau_w) * w"],
        output="VA",
        output_price="P_VA",
        TFP="A_VA",
        factor_shares="alpha_VA",
        epsilon="epsilon_VA",
        backend=backend,
    )

    import_domestic_bundle = CES(
        # factors=["X[0]", "X[1]"],
        factors=["X_D", "X_M"],
        # factor_prices=[
        #     "(1 + tau_X[0]) * P_Y[:, None]",
        #     "(1 + tau_X[1]) * P_M[:, None]",
        # ],
        factor_prices=[
            "(1 + tau_X_D) * P_Y[:, None]",
            "(1 + tau_X_M) * P_M[:, None]",
        ],
        output="X_bundled",
        output_price="P_X",
        TFP="A_X",
        factor_shares="alpha_X",
        epsilon="epsilon_X",
        dims=["i", "j"],
        backend=backend,
        expand_price_dim=None,
    )

    value_chain_bundle = leontief(
        factors="X_bundled",
        factor_prices="P_X",
        factor_shares="psi_X",
        output="VC",
        output_price="P_VC",
        dims=["i", "j"],
        sum_dim="i",
        coords=coords,
        expand_price_dim=False,
        backend=backend,
    )

    final_goods = leontief(
        factors=["VC", "VA"],
        factor_prices=["P_VC", "P_VA"],
        factor_shares=["psi_VC", "psi_VA"],
        output="Y",
        output_price="(1 - tau_Y) * P_Y",
        dims="i",
        coords=coords,
        backend=backend,
    )

    final_firm_equations = [
        # Value chain bundle
        Equation(
            "Sector <dim:j> production of intermediate goods bundle",
            value_chain_bundle[0],
        ),
        Equation(
            "Sector <dim:j> demand for sector <dim:i> intermediate input",
            value_chain_bundle[1],
        ),
        # Armington Bundle
        Equation(
            "Sector <dim:j> production of <dim:i> import-domestic bundle",
            import_domestic_bundle[0],
        ),
        Equation(
            "Sector <dim:i> demand for domestic <dim:j> inputs",
            import_domestic_bundle[1],
        ),
        Equation(
            "Sector <dim:i> demand for imported <dim:j> inputs",
            import_domestic_bundle[2],
        ),
        # Energy aggregation
        Equation(
            "Producer <dim:i> combination of regulated and unregulated energy",
            energy_mix[0],
        ),
        Equation("Producer <dim:i> demand for regulated energy", energy_mix[1]),
        Equation("Producer <dim:i> demand for unregulated energy", energy_mix[2]),
        # Capital Labour aggregation
        Equation(
            "Producer <dim:i> production of capital-energy bundle",
            capital_energy_bundle[0],
        ),
        Equation("Producer <dim:i> demand for captial", capital_energy_bundle[1]),
        Equation("Producer <dim:i> demand for energy", capital_energy_bundle[2]),
        # Value add bundle
        Equation("Sector <dim:i> production of value add", value_add_bundle[0]),
        Equation(
            "Sector <dim:i> demand for capital energy bundle", value_add_bundle[1]
        ),
        Equation("Sector <dim:i> demand for labour", value_add_bundle[2]),
        # Sector Final Goods
        Equation("Final good production of sector <dim:i>", final_goods[0]),
        Equation("Sector <dim:i> demand for intermediate goods bundle", final_goods[1]),
        Equation("Sector <dim:i> demand for value added", final_goods[2]),
        # Misc.
        Equation("Observed regulated energy price", "P_E_R = P_E_R_tilde + P_E_R_bar"),
        Equation(
            "Observed unregulated energy price", "P_E_U = P_E_U_tilde + P_E_U_bar"
        ),
    ]

    return final_firm_equations


def load_final_firm(coords, backend, check_model=True):
    variables = final_goods_variables()
    parameters = final_goods_parameters()
    equations = final_goods_equations(coords=coords, backend=backend)
    if check_model and backend == "pytensor":
        test_equations(variables, parameters, equations, coords)

    return variables, parameters, equations
