from cge_modeling import CGEModel, Equation, Parameter, Variable


def load_model_1(parse_equations_to_sympy=True):
    variable_info = [
        # Firm Variables
        Variable(name="Y", description="Total output of goods"),
        Variable(name="L_d", description="Labor demanded by firms"),
        Variable(name="K_d", description="Capital demanded by firms"),
        # Household Variables
        Variable(name="C", description="Household consumption"),
        Variable(name="income", latex_name="Omega", description="Total household income"),
        # Prices
        Variable(name="r", description="Rental rate of capital"),
        Variable(name="P", description="Price level of the good"),
        # Other
        Variable(name="resid", latex_name=r"varepsilon", description="Walrasian residual"),
    ]

    param_info = [
        # Firm Parameters
        Parameter(name="alpha", description="Share of capital in the production process"),
        Parameter(name="A", description="Total factor productivity"),
        # Exogenous variables
        Parameter(name="L_s", description="Exogenous household labor supply"),
        Parameter(name="K_s", description="Exogenous household capital stock"),
        Parameter(name="w", description="Numeraire wage level"),
    ]

    equations = [
        # Firm Equations
        Equation("Final good production ", "Y = A * K_d ** alpha * L_d ** (1 - alpha)"),
        Equation("Firm demand for capital", "r * K_d = alpha * Y * P"),
        Equation("Firm demand for labor", "w * L_d = (1 - alpha) * Y * P"),
        # Household Equations
        Equation("Household income", "income = w * L_s + r * K_s"),
        Equation("Household budget constraint", "C * P = income"),
        # Market clearning conditions
        Equation("Labor market clearing", "L_s = L_d + resid"),
        Equation("Capital market clearing", "K_s = K_d"),
        Equation("Sector <dim:i> goods market clearing", "Y = C"),
    ]

    return CGEModel(
        variables=variable_info,
        parameters=param_info,
        equations=equations,
        parse_equations_to_sympy=parse_equations_to_sympy,
    )


def load_model_2(parse_equations_to_sympy=True):
    variable_info = [
        # Firm Variables
        Variable(name="Y", dims="i", description="Total output of good <dim:i>"),
        Variable(
            name="VA", dims="i", description="Labor-capital bundle produced by sector <dim:i>"
        ),
        Variable(
            name="VC", dims="i", description="Intermediate goods bundle produced by sector <dim:i>"
        ),
        Variable(
            name="X",
            dims=("i", "j"),
            description="Demand for sector <dim:i> goods by sector <dim:j>",
        ),
        Variable(
            name="L_d",
            dims="i",
            extend_subscript=True,
            description="Labor demanded by sector <dim:i>",
        ),
        Variable(
            name="K_d",
            dims="i",
            extend_subscript=True,
            description="Capital demanded by sector <dim:i>",
        ),
        # Household Variables
        Variable(name="U", description="Household utility"),
        Variable(name="C", dims="i", description="Household consumption of good <dim:i>"),
        Variable(name="income", latex_name="Omega", description="Total household income"),
        # Prices
        Variable(name="r", description="Rental rate of capital"),
        Variable(name="w", description="Wage level"),
        Variable(
            name="P_VA",
            dims="i",
            extend_subscript=True,
            description="Price of the labor-capital bundle in sector <dim:i>",
        ),
        Variable(
            name="P_VC",
            dims="i",
            extend_subscript=True,
            description="Price of the intermediate bundle in sector <dim:i>",
        ),
        Variable(name="P", dims="i", description="Price level of final goods in sector <dim:i>"),
        # Other
        Variable(name="resid", latex_name=r"varepsilon", description="Walrasian residual"),
    ]

    param_info = [
        # Firm Parameters
        Parameter(
            name="psi_VA",
            extend_subscript=True,
            dims="i",
            description="Share of labor-capital in sector <dim:i>'s final good",
        ),
        Parameter(
            name="psi_VC",
            extend_subscript=True,
            dims="i",
            description="Share of intermediate goods bundle in sector <dim:i>'s final product",
        ),
        Parameter(
            name="psi_X",
            extend_subscript=True,
            dims=("i", "j"),
            description="Share of sector <dim:i>'s final good in sector <dim:j>'s value chain bundle",
        ),
        Parameter(
            name="alpha",
            dims="i",
            description="Share of capital in sector <dim:i>'s production process",
        ),
        Parameter(
            name="phi_VA",
            extend_subscript=True,
            dims="i",
            description="Elasticity of subsitution between input factors in <dim:i> sector VA bundle",
        ),
        Parameter(name="A", dims="i", description="Total factor productivity in sector <dim:i>"),
        # Household Parameters
        Parameter(
            name="gamma",
            dims="i",
            description="Household utility weight on consumption of sector <dim:i> goods",
        ),
        # Exogenous variables
        Parameter(name="L_s", description="Exogenous household labor supply"),
        Parameter(name="K_s", description="Exogenous household capital stock"),
        Parameter(
            name="P_Ag_bar",
            latex_name=r"\bar{P}_{Ag}",
            description="Exogenous agricultural price level",
        ),
    ]

    equations = [
        # # Firm Equations
        # Final Goods
        Equation("Final good production of sector <dim:i>", "P * Y = P_VC * VC + P_VA * VA"),
        Equation("Sector <dim:i> demand for intermediate goods bundle", "VC = psi_VC * Y"),
        Equation("Sector <dim:i> demand for labor-capital", "VA = psi_VA * Y"),
        # Value chain bundle
        Equation(
            "Sector <dim:i> production of intermediate goods bundle",
            "VC[None] * P_VC[None] = (P[:, None] * X).sum(axis=0).ravel()",
        ),
        Equation(
            "Sector <dim:i> demand for sector <dim:j> intermediate input", "X = psi_X * VC[:, None]"
        ),
        # Value add bundle
        Equation(
            "Sector <dim:i> production of labor-capital",
            "VA = A * (alpha * K_d**((phi_VA - 1) / phi_VA) +"
            "(1 - alpha) * L_d**((phi_VA - 1) / phi_VA)) ** (phi_VA / (phi_VA - 1))",
        ),
        Equation(
            "Sector <dim:i> demand for capital", "K_d = VA / A * (alpha * P_VA * A / r) ** phi_VA"
        ),
        Equation(
            "Sector <dim:i> demand for labor",
            "L_d = VA / A * ((1 - alpha) * A * P_VA / w) ** phi_VA",
        ),
        # # Household Equations
        Equation("Household income", "income = w * L_s + r * K_s"),
        Equation("Household utility", "U = (C**gamma).prod()"),
        Equation("Household demand for good <dim:i>", "C = gamma * income / P"),
        # # Market clearning conditions
        Equation("Labor market clearing", "L_s = L_d.sum() + resid"),
        Equation("Capital market clearing", "K_s = K_d.sum()"),
        Equation("Sector <dim:i> goods market clearing", "Y = C + X.sum(axis=1)"),
        Equation("Numeraire", "P[0] = P_Ag_bar"),
    ]

    sectors = ["0", "1", "2"]
    coords = {"i": sectors, "j": sectors}

    return CGEModel(
        variables=variable_info,
        parameters=param_info,
        equations=equations,
        coords=coords,
        parse_equations_to_sympy=parse_equations_to_sympy,
    )
