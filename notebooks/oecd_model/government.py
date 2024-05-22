from cge_modeling import Variable, Parameter, Equation
from cge_modeling.production_functions import CES
from notebooks.oecd_model.model_building_tools import test_equations, _sum_reduce
from functools import partial


def government_variables():
    government_variables = [
        Variable(name="G", description="Government budget"),
        Variable(
            name="C_G_D",
            dims=("i"),
            extend_subscript=True,
            description="Government consumption of domestic <dim:i> goods",
        ),
        Variable(
            name="C_G_M",
            dims=("i"),
            extend_subscript=True,
            description="Government consumption of imported <dim:i> goods",
        ),
        Variable(
            name="C_bundled_G",
            dims=("i",),
            extend_subscript=1,
            description="Government <dim:i> domestic-imported bundle",
        ),
        Variable(
            name="P_C_G",
            dims=("i",),
            extend_subscript=True,
            description="Price of government domestic-import bundle for <dim:i> sector goods",
        ),
    ]

    return government_variables


def government_parameters():
    government_parameteres = [
        Parameter(
            "alpha_Gov",
            dims="i",
            extend_subscript=True,
            description="Share of <dim:i> sector final goods in governmnet consumption",
        ),
        Parameter(
            name="alpha_C_G",
            dims=("i",),
            extend_subscript=True,
            description="Government home bias in for <dim:i> goods",
        ),
        Parameter(
            name="A_C_G",
            dims=("i",),
            extend_subscript=True,
            description="Government total factor productivity bundling domestic and imported <dim:i> goods",
        ),
        Parameter(
            name="epsilon_C_G",
            dims=("i",),
            extend_subscript=True,
            description="Government elasticity of substitution between domestic and imported <dim:i> goods",
        ),
        Parameter(
            name="tau_C_G_D",
            dims=("i",),
            extend_subscript=True,
            description="Tax rate paid by government for domestic sector <dim:i> goods",
        ),
        Parameter(
            name="tau_C_G_M",
            dims=("i",),
            extend_subscript=True,
            description="Tax rate paid by government for imported sector <dim:i> goods",
        ),
        Parameter(
            name="tau_Ex",
            dims="i",
            extend_subscript=True,
            description="Export duty paid by <dim:i> firm to ship goods abroad",
        ),
        Parameter(
            name="tau_Ex_E",
            dims="k",
            extend_subscript=2,
            description="Export duty paid by <dim:k> firm to ship goods abroad",
        ),
    ]

    return government_parameteres


def government_equations(coords, backend):
    sum_reduce = partial(_sum_reduce, coords=coords, backend=backend)
    s = "[:, None]" if backend == "pytensor" else ""

    import_domestic_bundle = CES(
        factors=["C_G_D", "C_G_M"],
        factor_prices=["(1 + tau_C_G_D) * P_Y", "(1 + tau_C_G_M) * P_M"],
        output="C_bundled_G",
        output_price="P_C_G",
        TFP="A_C_G",
        factor_shares="alpha_C_G",
        epsilon="epsilon_C_G",
        dims="i",
        backend=backend,
    )

    government_equations = [
        Equation(
            "Government budget constraint",
            "G + S_G = "
            + "+".join(
                [
                    # Firm taxes
                    sum_reduce(
                        f"(tau_X_D * P_Y{s} * X_D)", ["i", "j"]
                    ),  # Firm domestic VAT
                    sum_reduce(
                        f"(tau_X_M * P_M{s} * X_M)", ["i", "j"]
                    ),  # Firm import VAT
                    sum_reduce("(tau_w * L_d * w)", "i"),  # Firm payroll tax
                    sum_reduce("(tau_r * K_d * r)", "i"),  # Firm capital use tax
                    sum_reduce(
                        "(tau_E_R * E_R_d * P_E_R)", "i"
                    ),  # Firm regulated electricity usage
                    sum_reduce(
                        "(tau_E_U * E_U_d * P_E_U)", "i"
                    ),  # Firm unregulated electricity usage
                    sum_reduce("(tau_Y * P_Y * Y)", "i"),  # Firm output tax
                    # # Energy producer taxes
                    sum_reduce(
                        f"(tau_X_E_D * P_Y{s} * X_E_D)", ["i", "k"]
                    ),  # Energy domestic VAT
                    sum_reduce(
                        f"(tau_X_E_M * P_M{s} * X_E_M)", ["i", "k"]
                    ),  # Energy import VAT
                    sum_reduce("(tau_w_E * L_E_d * w)", "k"),  # Energy payroll tax
                    sum_reduce("(tau_r_E * K_E_d * r)", "k"),  # Energy capital use tax
                    sum_reduce(
                        "(tau_E_E_R * E_E_R_d * P_E_R)", "k"
                    ),  # Energy regulated electricity usage
                    sum_reduce(
                        "(tau_E_E_U * E_E_U_d * P_E_U)", "k"
                    ),  # Energy unregulated electricity usage
                    sum_reduce("(tau_Y_E * P_Y_E * Y_E)", "k"),  # Firm output tax
                    # # Grid
                    sum_reduce(
                        "(tau_G_R_D * Y_E_R_D_d * P_Y_E)", "k"
                    ),  # Sales tax paid by regulated grid
                    sum_reduce(
                        "(tau_G_R_M * Y_E_R_M_d * P_M_E)", "k"
                    ),  # Import tax paid by regulated grid
                    sum_reduce(
                        "(tau_G_U_D * Y_E_U_D_d * P_Y_E)", "k"
                    ),  # Sales tax paid by unregulated grid
                    sum_reduce(
                        "(tau_G_U_M * Y_E_R_M_d * P_M_E)", "k"
                    ),  # Import tax paid by unregulated grid
                    # # Investment
                    sum_reduce(
                        "(tau_I_D * I_D_d * P_Y)", "i"
                    ),  # Tax on domestic capital investment
                    sum_reduce(
                        "(tau_I_M * I_M_d * P_M)", "i"
                    ),  # Tax on imported capital investment
                    sum_reduce(
                        "(tau_I_E_D * I_E_D_d * P_Y_E)", "k"
                    ),  # Tax on domestic energy capital investment
                    sum_reduce(
                        "(tau_I_E_M * I_E_M_d * P_M_E)", "k"
                    ),  # Tax on imported energy capital investment
                    # # Household
                    sum_reduce("(tau_C_D * P_Y * C_D)", "i"),  # Consumption
                    sum_reduce("(tau_C_M * P_M * C_M)", "i"),  # Consumption
                    "(tau_E_H * E_H_d * P_E_R)",  # Household electricity
                    "(tau_w_income * w * L_s)",  # Income tax (direct tax)
                    "(tau_r_income * r * K_s)",  # Income tax (direct tax)
                    # Government
                    sum_reduce(
                        "(tau_C_G_D * P_Y * C_G_D)", "i"
                    ),  # Government consumption
                    sum_reduce(
                        "(tau_C_G_M * P_M * C_G_M)", "i"
                    ),  # Government consumption
                    # Export duties
                    sum_reduce(
                        "(tau_Ex * P_Y * Ex)", "i"
                    ),  # Tax on consumption exports
                    sum_reduce(
                        "(tau_Ex_E * P_Y_E * Ex_E)", "k"
                    ),  # Tax on energy exports
                    # Import duties
                    sum_reduce("(tau_M * P_M * M)", "i"),  # Tax on consumption imports
                    sum_reduce("(tau_M_E * P_M_E * M_E)", "k"),  # Tax on energy imports
                ]
            ),
        ),
        Equation(
            "Government demand for <dim:i> sector domestic-import goods",
            import_domestic_bundle[0],
        ),
        Equation(
            "Government demand for domestic <dim:i> sector goods",
            import_domestic_bundle[1],
        ),
        Equation(
            "Government demand for imported <dim:i> sector goods",
            import_domestic_bundle[2],
        ),
        Equation(
            "Government consumption of <dim:i> sector goods",
            "C_bundled_G = alpha_Gov * G",
        ),
    ]

    return government_equations


def load_government(coords, backend, check_model=True):
    variables = government_variables()
    parameters = government_parameters()
    equations = government_equations(coords, backend)
    if check_model:
        test_equations(variables, parameters, equations, coords)
    return variables, parameters, equations
