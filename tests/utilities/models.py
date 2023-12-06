import numpy as np

from cge_modeling import CGEModel, Equation, Parameter, Variable


def load_model_1(**kwargs):
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
        **kwargs,
    )


def calibrate_model_1(L_s, K_s, P, r):
    calib_dict = {"L_s": L_s, "K_s": K_s, "P": P, "r": r}

    # Numeraire
    w = 1

    INC = w * L_s + r * K_s
    C = INC / P
    Y = C
    K_d = K_s
    L_d = L_s

    alpha = r * K_d / Y / P
    A = Y / (K_d**alpha * L_d ** (1 - alpha))
    resid = L_d - L_s

    calib_dict["w"] = w
    calib_dict["income"] = INC
    calib_dict["C"] = C
    calib_dict["Y"] = Y
    calib_dict["K_d"] = K_d
    calib_dict["L_d"] = L_d
    calib_dict["alpha"] = alpha
    calib_dict["A"] = A
    calib_dict["resid"] = resid

    return calib_dict


model_1_data = {"L_s": 7000.0, "K_s": 4000.0, "P": 1.0, "r": 1.0}


def load_model_2(**kwargs):
    backend = kwargs.get("backend", "numba")
    sectors = ["0", "1", "2"]
    coords = {"i": sectors, "j": sectors}
    n_sectors = len(sectors)

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
            "VC * P_VC = (P[:, None] * X).sum(axis=0).ravel()"
            if backend == "pytensor"
            else "VC * P_VC = Sum(P.subs({i:j}) * X.subs([(i,k), (j,i), (k,j)]), "
            + f"(j, 0, {n_sectors - 1}))",
        ),
        Equation(
            "Sector <dim:i> demand for sector <dim:j> intermediate input",
            "X = psi_X * VC[None]" if backend == "pytensor" else "X = psi_X * VC.subs({i:j})",
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
        Equation(
            "Household utility",
            "U = (C**gamma).prod()"
            if backend == "pytensor"
            else "U = Product(C**gamma, " + f"(i, 0, {n_sectors - 1}))",
        ),
        Equation("Household demand for good <dim:i>", "C = gamma * income / P"),
        # # Market clearning conditions
        Equation(
            "Labor market clearing",
            "L_s = L_d.sum() + resid"
            if backend == "pytensor"
            else "L_s = resid + Sum(L_d, " + f"(i, 0, {n_sectors - 1}))",
        ),
        Equation(
            "Capital market clearing",
            "K_s = K_d.sum()"
            if backend == "pytensor"
            else f"K_s = Sum(K_d, (i, 0, {n_sectors - 1}))",
        ),
        Equation(
            "Sector <dim:i> goods market clearing",
            "Y = C + X.sum(axis=1)"
            if backend == "pytensor"
            else f"Y = C + Sum(X, (j, 0, {n_sectors - 1}))",
        ),
        Equation(
            "Numeraire", "P[0] = P_Ag_bar" if backend == "pytensor" else "P.subs({i:0}) = P_Ag_bar"
        ),
    ]

    return CGEModel(
        variables=variable_info, parameters=param_info, equations=equations, coords=coords, **kwargs
    )


def calibrate_model_2(L_d, K_d, Y, X, P, P_VA, P_VC, r, w, phi_VA, P_Ag_bar):
    calib_dict = {
        "L_d": L_d,
        "K_d": K_d,
        "Y": Y,
        "X": X,
        "P": P,
        "r": r,
        "w": w,
        "P_VA": P_VA,
        "P_VC": P_VC,
        "phi_VA": phi_VA,
        "P_Ag_bar": P_Ag_bar,
    }

    rho_VA = (phi_VA - 1) / phi_VA

    # Numeraire
    resid = 0.0

    # Household calibration
    L_s = L_d.sum()
    K_s = K_d.sum()
    income = w * L_s + r * K_s
    C = Y - X.sum(axis=1)
    gamma = C / income * P
    U = (C**gamma).prod()

    # Firm calibration
    VA = (w * L_d + r * K_d) / P_VA
    VC = (P[:, None] * X).sum(axis=0) / P_VC

    alpha = r * K_d ** (1 / phi_VA) / (r * K_d ** (1 / phi_VA) + w * L_d ** (1 / phi_VA))
    A = VA * (alpha * K_d**rho_VA + (1 - alpha) * L_d**rho_VA) ** (-1 / rho_VA)

    psi_VA = VA / Y
    psi_VC = VC / Y
    psi_X = X / VC[None]

    calib_dict["VA"] = VA
    calib_dict["VC"] = VC
    calib_dict["psi_VC"] = psi_VC
    calib_dict["psi_VA"] = psi_VA
    calib_dict["psi_X"] = psi_X
    calib_dict["alpha"] = alpha
    calib_dict["A"] = A

    calib_dict["income"] = income
    calib_dict["C"] = C
    calib_dict["U"] = U
    calib_dict["gamma"] = gamma

    calib_dict["K_s"] = K_s
    calib_dict["L_s"] = L_s
    calib_dict["resid"] = resid
    calib_dict["w"] = w

    return calib_dict


model_2_data = {
    "L_d": np.array([1000.0, 2000.0, 4000.0]),
    "K_d": np.array([500.0, 2000.0, 500.0]),
    "X": np.array([[1000.0, 1000.0, 1000.0], [2000.0, 3500.0, 3000.0], [500.0, 2500.0, 1000.0]]),
    "Y": np.array([5000.0, 11000.0, 9500.0]),
    "phi_VA": np.array([3.0, 3.0, 3.0]),
    "P": np.array([1.0, 1.0, 1.0]),
    "P_VA": np.array([1.0, 1.0, 1.0]),
    "P_VC": np.array([1.0, 1.0, 1.0]),
    "r": 1.0,
    "w": 1.0,
    "P_Ag_bar": 1.0,
}
