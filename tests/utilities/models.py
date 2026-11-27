from functools import cache
from typing import Literal

import numpy as np

from cge_modeling import CGEModel
from cge_modeling import model as cm
from cge_modeling.production_functions import CES, CET, CobbDouglas, Leontief


def load_model_1(**compile_kwargs):
    return build_model_1().compile(**compile_kwargs)


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


def build_model_1() -> CGEModel:
    with CGEModel() as model:
        Y = cm.Variable("Y", description="Total output of goods")
        L_d = cm.Variable("L_d", description="Labor demanded by firms")
        K_d = cm.Variable("K_d", description="Capital demanded by firms")
        C = cm.Variable("C", description="Household consumption")
        income = cm.Variable("income", latex_name="Omega", description="Total household income")
        r = cm.Variable("r", description="Rental rate of capital")
        P = cm.Variable("P", description="Price level of the good")
        resid = cm.Variable("resid", latex_name="varepsilon", description="Walrasian residual")

        alpha = cm.Parameter("alpha", description="Share of capital in the production process")
        A = cm.Parameter("A", description="Total factor productivity")
        L_s = cm.Parameter("L_s", description="Exogenous household labor supply")
        K_s = cm.Parameter("K_s", description="Exogenous household capital stock")
        w = cm.Parameter("w", description="Numeraire wage level")

        CobbDouglas(
            "Final good production",
            output=Y,
            output_price=P,
            factors=[K_d, L_d],
            factor_prices=[r, w],
            factor_shares=[alpha],
            tfp=A,
        )
        cm.Equation("Household income", income, w * L_s + r * K_s)
        cm.Equation("Household budget constraint", C * P, income)
        cm.Equation("Labor market clearing", L_s, L_d + resid)
        cm.Equation("Capital market clearing", K_s, K_d)
        cm.Equation("Goods market clearing", Y, C)

    return model


model_1_data = {"L_s": 7000.0, "K_s": 4000.0, "P": 1.0, "r": 1.0}


def load_model_2(**compile_kwargs):
    return build_model_2().compile(**compile_kwargs)


def build_model_2() -> CGEModel:
    sectors = ["0", "1", "2"]

    with CGEModel(coords={"i": sectors, "j": sectors}) as model:
        Y = cm.Variable("Y", dims="i", description="Total output of good <dim:i>")
        VA = cm.Variable("VA", dims="i")
        VC = cm.Variable("VC", dims="i")
        X = cm.Variable("X", dims=("i", "j"))
        L_d = cm.Variable("L_d", dims="i", extend_subscript=True)
        K_d = cm.Variable("K_d", dims="i", extend_subscript=True)
        U = cm.Variable("U")
        C = cm.Variable("C", dims="i")
        income = cm.Variable("income", latex_name="Omega")
        r = cm.Variable("r")
        w = cm.Variable("w")
        P_VA = cm.Variable("P_VA", dims="i", extend_subscript=True)
        P_VC = cm.Variable("P_VC", dims="i", extend_subscript=True)
        P = cm.Variable("P", dims="i")
        resid = cm.Variable("resid", latex_name="varepsilon")

        psi_VA = cm.Parameter("psi_VA", dims="i", extend_subscript=True)
        psi_VC = cm.Parameter("psi_VC", dims="i", extend_subscript=True)
        psi_X = cm.Parameter("psi_X", dims=("i", "j"), extend_subscript=True)
        alpha = cm.Parameter("alpha", dims="i")
        phi_VA = cm.Parameter("phi_VA", dims="i", extend_subscript=True)
        A = cm.Parameter("A", dims="i")
        gamma = cm.Parameter("gamma", dims="i")
        L_s = cm.Parameter("L_s")
        K_s = cm.Parameter("K_s")
        P_Ag_bar = cm.Parameter("P_Ag_bar", latex_name=r"\bar{P}_{Ag}")

        Leontief(
            "Final good production",
            output=Y,
            output_price=P,
            factors=[VC, VA],
            factor_prices=[P_VC, P_VA],
            factor_shares=[psi_VC, psi_VA],
        )
        Leontief(
            "Production of intermediate goods bundle",
            output=VC.rename(i="j"),
            output_price=P_VC.rename(i="j"),
            factors=[X],
            factor_prices=[P],
            factor_shares=[psi_X],
            factor_dim="i",
        )
        CES(
            "Production of labor-capital",
            output=VA,
            output_price=P_VA,
            factors=[K_d, L_d],
            factor_prices=[r, w],
            factor_shares=[alpha],
            tfp=A,
            epsilon=phi_VA,
        )
        cm.Equation("Household income", income, w * L_s + r * K_s)
        cm.Equation("Household utility", U, (C**gamma).prod())
        cm.Equation("Household demand", C, gamma * income / P)
        cm.Equation("Labor market clearing", L_s, L_d.sum() + resid)
        cm.Equation("Capital market clearing", K_s, K_d.sum())
        cm.Equation("Goods market clearing", Y, C + X.sum("j"))
        cm.Equation("Numeraire", P.sel(i="0"), P_Ag_bar)

    return model


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


model_3_data = {
    "K_d": np.array([30.0, 60.0]),
    "L_d": np.array([50.0, 40.0]),
    "X": np.array([[10.0, 20.0], [15.0, 30.0]]),
    "X_EN": np.array([[5.0, 15.0]]),
    "K_d_E": np.array([20.0]),
    "L_d_E": np.array([10.0]),
    "X_E": np.array([[3.0], [7.0]]),
    "P": np.array([1.0, 1.0]),
    "P_E": np.array([1.0]),
    "r": 1.0,
    "w": 1.0,
}


def calibrate_model_3(K_d, L_d, X, X_EN, K_d_E, L_d_E, X_E, P, P_E, r, w):
    """
    Calibrate model 3 to a benchmark with value-added and intermediate-bundle prices of one.

    ``X[i, j]`` is goods sector ``i``'s sales to goods sector ``j``, ``X_EN[k, j]`` energy sector ``k``'s sales to
    goods sector ``j``, and ``X_E[i, k]`` goods sector ``i``'s sales to energy sector ``k``.
    """
    P_VA, P_VC = np.ones_like(P), np.ones_like(P)
    P_VA_E, P_VC_E = np.ones_like(P_E), np.ones_like(P_E)

    VA = (r * K_d + w * L_d) / P_VA
    alpha = r * K_d / (P_VA * VA)
    A = VA / (K_d**alpha * L_d ** (1 - alpha))
    VC = ((P[:, None] * X).sum(axis=0) + (P_E[:, None] * X_EN).sum(axis=0)) / P_VC
    Y = (P_VC * VC + P_VA * VA) / P

    VA_E = (r * K_d_E + w * L_d_E) / P_VA_E
    alpha_E = r * K_d_E / (P_VA_E * VA_E)
    A_E = VA_E / (K_d_E**alpha_E * L_d_E ** (1 - alpha_E))
    VC_E = (P[:, None] * X_E).sum(axis=0) / P_VC_E
    Y_E = (P_VC_E * VC_E + P_VA_E * VA_E) / P_E

    L_s = L_d.sum() + L_d_E.sum()
    K_s = K_d.sum() + K_d_E.sum()
    income = w * L_s + r * K_s
    C = Y - X.sum(axis=1) - X_E.sum(axis=1)
    C_E = Y_E - X_EN.sum(axis=1)

    return {
        "Y": Y,
        "VA": VA,
        "VC": VC,
        "X": X,
        "X_EN": X_EN,
        "K_d": K_d,
        "L_d": L_d,
        "P": P,
        "P_VA": P_VA,
        "P_VC": P_VC,
        "Y_E": Y_E,
        "VA_E": VA_E,
        "VC_E": VC_E,
        "X_E": X_E,
        "K_d_E": K_d_E,
        "L_d_E": L_d_E,
        "P_E": P_E,
        "P_VA_E": P_VA_E,
        "P_VC_E": P_VC_E,
        "C": C,
        "C_E": C_E,
        "income": income,
        "r": r,
        "w": w,
        "resid": 0.0,
        "psi_VA": VA / Y,
        "psi_VC": VC / Y,
        "psi_X": X / VC,
        "psi_X_EN": X_EN / VC,
        "alpha": alpha,
        "A": A,
        "psi_VA_E": VA_E / Y_E,
        "psi_VC_E": VC_E / Y_E,
        "psi_X_E": X_E / VC_E,
        "alpha_E": alpha_E,
        "A_E": A_E,
        "gamma": P * C / income,
        "gamma_E": P_E * C_E / income,
        "L_s": L_s,
        "K_s": K_s,
        "P_bar": P[0],
    }


def build_model_3() -> CGEModel:
    """
    Two goods sectors and one energy sector, each a Leontief combination of a Cobb-Douglas value-added bundle and
    a Leontief intermediate bundle. Goods producers buy goods and energy; the energy producer buys goods.
    """
    with CGEModel(
        coords={
            "i": ["agriculture", "manufacturing"],
            "j": ["agriculture", "manufacturing"],
            "k": ["energy"],
        }
    ) as model:
        Y = cm.Variable("Y", dims="i")
        VA = cm.Variable("VA", dims="i")
        VC = cm.Variable("VC", dims="i")
        X = cm.Variable("X", dims=("i", "j"))
        X_EN = cm.Variable("X_EN", dims=("k", "j"))
        K_d = cm.Variable("K_d", dims="i")
        L_d = cm.Variable("L_d", dims="i")
        P = cm.Variable("P", dims="i")
        P_VA = cm.Variable("P_VA", dims="i")
        P_VC = cm.Variable("P_VC", dims="i")
        Y_E = cm.Variable("Y_E", dims="k")
        VA_E = cm.Variable("VA_E", dims="k")
        VC_E = cm.Variable("VC_E", dims="k")
        X_E = cm.Variable("X_E", dims=("i", "k"))
        K_d_E = cm.Variable("K_d_E", dims="k")
        L_d_E = cm.Variable("L_d_E", dims="k")
        P_E = cm.Variable("P_E", dims="k")
        P_VA_E = cm.Variable("P_VA_E", dims="k")
        P_VC_E = cm.Variable("P_VC_E", dims="k")
        C = cm.Variable("C", dims="i")
        C_E = cm.Variable("C_E", dims="k")
        income = cm.Variable("income")
        r = cm.Variable("r")
        w = cm.Variable("w")
        resid = cm.Variable("resid")

        psi_VA = cm.Parameter("psi_VA", dims="i")
        psi_VC = cm.Parameter("psi_VC", dims="i")
        psi_X = cm.Parameter("psi_X", dims=("i", "j"))
        psi_X_EN = cm.Parameter("psi_X_EN", dims=("k", "j"))
        alpha = cm.Parameter("alpha", dims="i")
        A = cm.Parameter("A", dims="i")
        psi_VA_E = cm.Parameter("psi_VA_E", dims="k")
        psi_VC_E = cm.Parameter("psi_VC_E", dims="k")
        psi_X_E = cm.Parameter("psi_X_E", dims=("i", "k"))
        alpha_E = cm.Parameter("alpha_E", dims="k")
        A_E = cm.Parameter("A_E", dims="k")
        gamma = cm.Parameter("gamma", dims="i")
        gamma_E = cm.Parameter("gamma_E", dims="k")
        L_s = cm.Parameter("L_s")
        K_s = cm.Parameter("K_s")
        P_bar = cm.Parameter("P_bar")

        Leontief(
            "Output of <dim:i>",
            output=Y,
            output_price=P,
            factors=[VC, VA],
            factor_prices=[P_VC, P_VA],
            factor_shares=[psi_VC, psi_VA],
        )
        Leontief(
            "Intermediate bundle of <dim:j>",
            output=VC.rename(i="j"),
            output_price=P_VC.rename(i="j"),
            factors=[X, X_EN],
            factor_prices=[P, P_E],
            factor_shares=[psi_X, psi_X_EN],
            factor_dim=["i", "k"],
        )
        CobbDouglas(
            "Value added of <dim:i>",
            output=VA,
            output_price=P_VA,
            factors=[K_d, L_d],
            factor_prices=[r, w],
            factor_shares=[alpha],
            tfp=A,
        )
        Leontief(
            "Output of <dim:k>",
            output=Y_E,
            output_price=P_E,
            factors=[VC_E, VA_E],
            factor_prices=[P_VC_E, P_VA_E],
            factor_shares=[psi_VC_E, psi_VA_E],
        )
        Leontief(
            "Intermediate bundle of <dim:k>",
            output=VC_E,
            output_price=P_VC_E,
            factors=[X_E],
            factor_prices=[P],
            factor_shares=[psi_X_E],
            factor_dim="i",
        )
        CobbDouglas(
            "Value added of <dim:k>",
            output=VA_E,
            output_price=P_VA_E,
            factors=[K_d_E, L_d_E],
            factor_prices=[r, w],
            factor_shares=[alpha_E],
            tfp=A_E,
        )
        cm.Equation("Household income", income, w * L_s + r * K_s)
        cm.Equation("Household demand for <dim:i>", C, gamma * income / P)
        cm.Equation("Household demand for <dim:k>", C_E, gamma_E * income / P_E)
        cm.Equation("Goods market clearing for <dim:i>", Y, C + X.sum("j") + X_E.sum("k"))
        cm.Equation("Energy market clearing for <dim:k>", Y_E, C_E + X_EN.sum("j"))
        cm.Equation("Labor market clearing", L_s, L_d.sum() + L_d_E.sum() + resid)
        cm.Equation("Capital market clearing", K_s, K_d.sum() + K_d_E.sum())
        cm.Equation("Numeraire", P.sel(i="agriculture"), P_bar)

    return model


model_4_data = {
    "output_value_added": 324.69,
    "private_consumption": 291.69,
    "public_consumption": 35.58,
    "investment": 86.38,
    "exports": 106.39,
    "imports": 144.7,
    "sales_and_excise_tax": 32.03,
    "import_tariffs": 18.62,
    "export_duties": 1.14,
    "payroll_tax": 0.00,
    "capital_income_tax": 12.84,
    "nontax_revenue": 8.02,
    "government_interest_payments": 22.07,
    "transfers_and_subsidies": 26.10,
    "net_profits_and_dividends": -0.78,
    "external_interest_payments": -8.82,
    "net_private_transfers": 11.60,
    "net_official_transfers": 7.90,
    "st": 0.6,
    "sq": 0.6,
}


def calibrate_model_4(
    output_value_added,
    private_consumption,
    public_consumption,
    investment,
    exports,
    imports,
    sales_and_excise_tax,
    import_tariffs,
    export_duties,
    payroll_tax,
    capital_income_tax,
    nontax_revenue,
    government_interest_payments,
    transfers_and_subsidies,
    net_profits_and_dividends,
    external_interest_payments,
    net_private_transfers,
    net_official_transfers,
    st,
    sq,
):
    """
    Calibrate model 4 to the 1991 Sri Lanka data of Devarajan, Go, Lewis, Robinson and Sinko (1997), Table 6.6.

    The data are in billions of rupees and are indexed to output. The formulas follow Tables 6.7 and 6.8 of the same
    chapter, including the direct tax rate, which counts payroll and capital income taxes but not personal income tax.
    """
    E = exports / output_value_added
    M = (imports + import_tariffs) / output_value_added
    X = 1.0
    Ds = X - E
    Dd = Ds
    Qs = M + Dd
    Qd = Qs
    Er, Pm, Pe, Pd, Px, Pq = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

    tm = import_tariffs / imports
    te = export_duties / exports
    ts = sales_and_excise_tax / output_value_added / Qs
    Pt = Pq * (1 + ts)
    wm = Pm / Er / (1 + tm)
    we = Pe * (1 + te) / Er

    tr = (
        (government_interest_payments + transfers_and_subsidies - nontax_revenue)
        / output_value_added
        / Pq
    )
    re = (
        (net_profits_and_dividends + external_interest_payments + net_private_transfers)
        / output_value_added
        / Er
    )
    ft = net_official_transfers / output_value_added / Er
    Y = Px * X + tr * Pq + re * Er
    ty = (payroll_tax + capital_income_tax) / output_value_added / Y

    Cn = private_consumption / output_value_added / Pt
    G = public_consumption / output_value_added / (1 + ts) / Pq
    Z = investment / output_value_added / Pt
    sy = (Y - Cn * Pt - ty * Y) / Y
    B = wm * M - we * E - ft - re

    Tax = tm * wm * Er * M + te * Pe * E + ts * Pq * Qd + ty * Y
    Sg = Tax - G * Pt - tr * Pq + ft * Er
    S = sy * Y + Er * B + Sg

    rt = 1 / st + 1
    bt = 1 / (1 + (Pd / Pe) * (E / Ds) ** (rt - 1))
    at = X / (bt * E**rt + (1 - bt) * Ds**rt) ** (1 / rt)
    rq = 1 / sq - 1
    import_ratio = (Pm / Pd) * (M / Dd) ** (1 + rq)
    bq = import_ratio / (1 + import_ratio)
    aq = Qs / (bq * M ** (-rq) + (1 - bq) * Dd ** (-rq)) ** (-1 / rq)

    return {
        "E": E,
        "M": M,
        "Ds": Ds,
        "Dd": Dd,
        "Qs": Qs,
        "Qd": Qd,
        "Pe": Pe,
        "Pm": Pm,
        "Pd": Pd,
        "Pt": Pt,
        "Px": Px,
        "Pq": Pq,
        "Tax": Tax,
        "Y": Y,
        "S": S,
        "Sg": Sg,
        "Cn": Cn,
        "Z": Z,
        "resid": S - Pt * Z,
        "X": X,
        "Er": Er,
        "wm": wm,
        "we": we,
        "tm": tm,
        "te": te,
        "ts": ts,
        "ty": ty,
        "sy": sy,
        "G": G,
        "tr": tr,
        "ft": ft,
        "re": re,
        "B": B,
        "at": at,
        "bt": bt,
        "st": st,
        "aq": aq,
        "bq": bq,
        "sq": sq,
    }


def build_model_4(*, closure: Literal["savings_driven", "fixed_investment"]) -> CGEModel:
    """
    The 1-2-3 model with government and investment of Devarajan, Go, Lewis, Robinson and Sinko (1997), Table 6.3.

    One country produces an export good and a domestic good on a CET frontier and consumes a CES composite of the
    domestic good and an import. The exchange rate is the numeraire. In the savings-driven closure investment adjusts
    to savings. In the fixed-investment closure investment is fixed and the sales tax rate adjusts.
    """
    investment_object, sales_tax_object = {
        "savings_driven": (cm.Variable, cm.Parameter),
        "fixed_investment": (cm.Parameter, cm.Variable),
    }[closure]

    with CGEModel() as model:
        E = cm.Variable("E", description="Export good")
        M = cm.Variable("M", description="Import good")
        Ds = cm.Variable("Ds", description="Supply of domestic good")
        Dd = cm.Variable("Dd", description="Demand for domestic good")
        Qs = cm.Variable("Qs", description="Supply of composite good")
        Qd = cm.Variable("Qd", description="Demand for composite good")
        Pe = cm.Variable("Pe", description="Domestic price of export good")
        Pm = cm.Variable("Pm", description="Domestic price of import good")
        Pd = cm.Variable("Pd", description="Producer price of domestic good")
        Pt = cm.Variable("Pt", description="Sales price of composite good")
        Px = cm.Variable("Px", description="Price of aggregate output")
        Pq = cm.Variable("Pq", description="Price of composite good")
        Tax = cm.Variable("Tax", description="Tax revenue")
        Y = cm.Variable("Y", description="Total income")
        S = cm.Variable("S", description="Aggregate savings")
        Sg = cm.Variable("Sg", description="Government savings")
        Cn = cm.Variable("Cn", description="Aggregate consumption")
        resid = cm.Variable("resid", latex_name="varepsilon", description="Walrasian residual")
        Z = investment_object("Z", description="Aggregate real investment")
        ts = sales_tax_object("ts", description="Sales, excise and value-added tax rate")

        X = cm.Parameter("X", description="Aggregate output")
        Er = cm.Parameter("Er", description="Exchange rate, the numeraire")
        wm = cm.Parameter("wm", description="World price of import good")
        we = cm.Parameter("we", description="World price of export good")
        tm = cm.Parameter("tm", description="Tariff rate")
        te = cm.Parameter("te", description="Export duty rate")
        ty = cm.Parameter("ty", description="Direct tax rate")
        sy = cm.Parameter("sy", description="Average savings rate")
        G = cm.Parameter("G", description="Real government demand")
        tr = cm.Parameter("tr", description="Government transfers")
        ft = cm.Parameter("ft", description="Foreign transfers to government")
        re = cm.Parameter("re", description="Foreign remittances to private sector")
        B = cm.Parameter("B", description="Balance of trade")
        at = cm.Parameter("at", description="CET scale")
        bt = cm.Parameter("bt", description="CET share of the export good")
        st = cm.Parameter("st", latex_name="Omega", description="Export transformation elasticity")
        aq = cm.Parameter("aq", description="Armington scale")
        bq = cm.Parameter("bq", description="Armington share of the import good")
        sq = cm.Parameter("sq", latex_name="sigma", description="Import substitution elasticity")

        CET(
            "Export transformation",
            output=X,
            output_price=Px,
            products=[E, Ds],
            product_prices=[Pe, Pd],
            product_shares=[bt],
            tfp=at,
            omega=st,
        )
        CES(
            "Armington aggregation",
            output=Qs,
            output_price=Pq,
            factors=[M, Dd],
            factor_prices=[Pm, Pd],
            factor_shares=[bq],
            tfp=aq,
            epsilon=sq,
        )
        cm.Equation("Domestic demand", Qd, Cn + Z + G)
        cm.Equation("Tax revenue", Tax, tm * wm * Er * M + te * Pe * E + ts * Pq * Qd + ty * Y)
        cm.Equation("Total income", Y, Px * X + tr * Pq + re * Er)
        cm.Equation("Aggregate savings", S, sy * Y + Er * B + Sg)
        cm.Equation("Consumption", Pt * Cn, (1 - ty - sy) * Y)
        cm.Equation("Import price", Pm, Er * wm * (1 + tm))
        cm.Equation("Export price", Pe, Er * we / (1 + te))
        cm.Equation("Sales price", Pt, Pq * (1 + ts))
        cm.Equation("Domestic good market", Dd, Ds)
        cm.Equation("Composite good market", Qd, Qs)
        cm.Equation("Current account balance", wm * M - we * E - ft - re, B)
        cm.Equation("Government budget", Sg, Tax - G * Pt - tr * Pq + ft * Er)
        cm.Equation("Savings-investment balance", S, Pt * Z + resid)

    return model


model_5_data = {
    "intermediate_inputs": np.array([[1.0, 4.0, 0.0], [1.0, 8.0, 2.0], [1.0, 5.0, 5.0]]),
    "intermediate_imports": np.array([0.0, 3.0, 1.0]),
    "wages_and_salaries": np.array([1.0, 4.0, 7.0]),
    "other_factor_payments": np.array([4.0, 10.0, 19.0]),
    "indirect_taxes_and_tariffs": np.array([0.0, 1.0, 4.0]),
    "private_consumption": np.array([2.0, 11.0, 21.0]),
    "private_consumption_of_imports": 1.0,
    "investment": np.array([0.0, 8.0, 2.0]),
    "investment_imports": 2.0,
    "government_consumption": np.array([0.0, 1.0, 2.0]),
    "government_wages_and_salaries": 1.0,
    "exports": np.array([1.0, 4.0, 2.0]),
    "exports_of_imports": 1.0,
    "direct_taxes": 2.0,
}


def calibrate_model_5(
    intermediate_inputs,
    intermediate_imports,
    wages_and_salaries,
    other_factor_payments,
    indirect_taxes_and_tariffs,
    private_consumption,
    private_consumption_of_imports,
    investment,
    investment_imports,
    government_consumption,
    government_wages_and_salaries,
    exports,
    exports_of_imports,
    direct_taxes,
):
    """
    Calibrate model 5 to the 1989 Mexican input-output matrix of Kehoe and Kehoe (1994), Tables 1 and 2.

    Values are in tens of trillions of 1989 pesos, and every price is one in the base year. ``intermediate_inputs``
    is indexed (good, sector) over primaries, manufactures and services, and the government's other factor
    payments and indirect taxes, both zero in the data, are left out.
    """
    P, P_VA = np.ones(3), np.ones(3)
    P_M, P_I, P_G, w, r = 1.0, 1.0, 1.0, 1.0, 1.0

    X, X_M, L_d, K_d = (
        intermediate_inputs,
        intermediate_imports,
        wages_and_salaries,
        other_factor_payments,
    )
    VA = (w * L_d + r * K_d) / P_VA
    Y = ((P[:, None] * X).sum(axis=0) + P_M * X_M + P_VA * VA + indirect_taxes_and_tariffs) / P
    t = indirect_taxes_and_tariffs / (P * Y)
    alpha = r * K_d / (P_VA * VA)
    A = VA / (K_d**alpha * L_d ** (1 - alpha))

    X_E, X_MM = exports, exports_of_imports
    Y_M = ((P * X_E).sum() + P_M * X_MM) / P_M
    X_I, X_MI = investment, investment_imports
    Y_I = ((P * X_I).sum() + P_M * X_MI) / P_I
    X_G, L_G = government_consumption, government_wages_and_salaries
    Y_G = ((P * X_G).sum() + w * L_G) / P_G

    L_s = L_d.sum() + L_G
    K_s = K_d.sum()
    tau = direct_taxes / (w * L_s + r * K_s)
    T = tau * (w * L_s + r * K_s) + (t * P * Y).sum()
    income = (1 - tau) * (w * L_s + r * K_s) + T
    C, C_M, C_I, C_G = private_consumption, private_consumption_of_imports, Y_I, Y_G

    return {
        "Y": Y,
        "VA": VA,
        "X": X,
        "X_M": X_M,
        "K_d": K_d,
        "L_d": L_d,
        "P": P,
        "P_VA": P_VA,
        "Y_M": Y_M,
        "X_E": X_E,
        "X_MM": X_MM,
        "P_M": P_M,
        "Y_I": Y_I,
        "X_I": X_I,
        "X_MI": X_MI,
        "P_I": P_I,
        "Y_G": Y_G,
        "X_G": X_G,
        "L_G": L_G,
        "P_G": P_G,
        "C": C,
        "C_M": C_M,
        "C_I": C_I,
        "C_G": C_G,
        "income": income,
        "T": T,
        "w": w,
        "r": r,
        "resid": 0.0,
        "a": X / Y,
        "a_M": X_M / Y,
        "psi_VA": VA / Y,
        "t": t,
        "alpha": alpha,
        "A": A,
        "a_E": X_E / Y_M,
        "a_MM": X_MM / Y_M,
        "a_I": X_I / Y_I,
        "a_MI": X_MI / Y_I,
        "a_G": X_G / Y_G,
        "a_LG": L_G / Y_G,
        "theta": P * C / income,
        "theta_M": P_M * C_M / income,
        "theta_I": P_I * C_I / income,
        "theta_G": P_G * C_G / income,
        "tau": tau,
        "L_s": L_s,
        "K_s": K_s,
    }


def build_model_5() -> CGEModel:
    """
    The simple static model of Mexico of Kehoe and Kehoe (1994), equations (1) to (14) and (20).

    Primaries, manufactures and services combine intermediate goods, imports and a Cobb-Douglas value-added
    bundle in fixed proportions and pay an indirect tax on sales. Investment, government consumption and the import
    good are fixed-proportions activities. The import good is produced from exports, which keeps trade balanced.
    One consumer with Cobb-Douglas utility over all six goods pays a direct tax and receives all tax revenue. A
    consumption-weighted price index is the numeraire.
    """
    sectors = ["primaries", "manufactures", "services"]
    with CGEModel(coords={"i": sectors, "j": sectors}) as model:
        Y = cm.Variable("Y", dims="i", description="Output of good <dim:i>")
        VA = cm.Variable("VA", dims="j", description="Value added in sector <dim:j>")
        X = cm.Variable("X", dims=("i", "j"), description="Use of good <dim:i> by sector <dim:j>")
        X_M = cm.Variable("X_M", dims="j", description="Use of the import good by sector <dim:j>")
        K_d = cm.Variable("K_d", dims="j", description="Capital demanded by sector <dim:j>")
        L_d = cm.Variable("L_d", dims="j", description="Labor demanded by sector <dim:j>")
        P = cm.Variable("P", dims="i", description="Price of good <dim:i>")
        P_VA = cm.Variable("P_VA", dims="j", description="Price of value added in sector <dim:j>")
        Y_M = cm.Variable("Y_M", description="Output of the import good")
        X_E = cm.Variable("X_E", dims="i", description="Exports of good <dim:i>")
        X_MM = cm.Variable("X_MM", description="Re-exports of the import good")
        P_M = cm.Variable("P_M", description="Price of the import good")
        Y_I = cm.Variable("Y_I", description="Output of the investment good")
        X_I = cm.Variable("X_I", dims="i", description="Use of good <dim:i> in investment")
        X_MI = cm.Variable("X_MI", description="Use of the import good in investment")
        P_I = cm.Variable("P_I", description="Price of the investment good")
        Y_G = cm.Variable("Y_G", description="Output of the government consumption good")
        X_G = cm.Variable("X_G", dims="i", description="Use of good <dim:i> by government")
        L_G = cm.Variable("L_G", description="Labor demanded by government")
        P_G = cm.Variable("P_G", description="Price of the government consumption good")
        C = cm.Variable("C", dims="i", description="Consumption of good <dim:i>")
        C_M = cm.Variable("C_M", description="Consumption of the import good")
        C_I = cm.Variable("C_I", description="Purchases of the investment good")
        C_G = cm.Variable("C_G", description="Purchases of the government consumption good")
        income = cm.Variable("income", description="Disposable income including the transfer")
        T = cm.Variable("T", description="Tax receipts, transferred to the consumer")
        w = cm.Variable("w", description="Wage rate")
        r = cm.Variable("r", description="Capital rental rate")
        resid = cm.Variable("resid", latex_name="varepsilon", description="Walrasian residual")

        a = cm.Parameter(
            "a", dims=("i", "j"), description="Input of good <dim:i> per unit of <dim:j>"
        )
        a_M = cm.Parameter("a_M", dims="j", description="Import input per unit of <dim:j>")
        psi_VA = cm.Parameter("psi_VA", dims="j", description="Value added per unit of <dim:j>")
        t = cm.Parameter("t", dims="j", description="Indirect tax rate on sales of <dim:j>")
        alpha = cm.Parameter(
            "alpha", dims="j", description="Capital share of value added in <dim:j>"
        )
        A = cm.Parameter("A", dims="j", description="Value-added productivity in <dim:j>")
        a_E = cm.Parameter("a_E", dims="i", description="Exports of <dim:i> per unit of imports")
        a_MM = cm.Parameter("a_MM", description="Re-exports per unit of imports")
        a_I = cm.Parameter("a_I", dims="i", description="Input of <dim:i> per unit of investment")
        a_MI = cm.Parameter("a_MI", description="Import input per unit of investment")
        a_G = cm.Parameter(
            "a_G", dims="i", description="Input of <dim:i> per unit of government consumption"
        )
        a_LG = cm.Parameter("a_LG", description="Labor per unit of government consumption")
        theta = cm.Parameter("theta", dims="i", description="Budget share of good <dim:i>")
        theta_M = cm.Parameter("theta_M", description="Budget share of the import good")
        theta_I = cm.Parameter("theta_I", description="Budget share of the investment good")
        theta_G = cm.Parameter(
            "theta_G", description="Budget share of the government consumption good"
        )
        tau = cm.Parameter("tau", description="Direct tax rate")
        L_s = cm.Parameter("L_s", description="Labor endowment")
        K_s = cm.Parameter("K_s", description="Capital endowment")

        Leontief(
            "Output of <dim:j>",
            output=Y.rename(i="j"),
            output_price=(1 - t) * P.rename(i="j"),
            factors=[X, X_M, VA],
            factor_prices=[P, P_M, P_VA],
            factor_shares=[a, a_M, psi_VA],
            factor_dim=["i", None, None],
        )
        CobbDouglas(
            "Value added in <dim:j>",
            output=VA,
            output_price=P_VA,
            factors=[K_d, L_d],
            factor_prices=[r, w],
            factor_shares=[alpha],
            tfp=A,
        )
        Leontief(
            "Import activity",
            output=Y_M,
            output_price=P_M,
            factors=[X_E, X_MM],
            factor_prices=[P, P_M],
            factor_shares=[a_E, a_MM],
            factor_dim=["i", None],
        )
        Leontief(
            "Investment good",
            output=Y_I,
            output_price=P_I,
            factors=[X_I, X_MI],
            factor_prices=[P, P_M],
            factor_shares=[a_I, a_MI],
            factor_dim=["i", None],
        )
        Leontief(
            "Government consumption good",
            output=Y_G,
            output_price=P_G,
            factors=[X_G, L_G],
            factor_prices=[P, w],
            factor_shares=[a_G, a_LG],
            factor_dim=["i", None],
        )

        factor_income = w * L_s + r * K_s
        cm.Equation(
            "Tax receipts",
            T,
            tau * factor_income + (t * P.rename(i="j") * Y.rename(i="j")).sum("j"),
        )
        cm.Equation("Disposable income", income, (1 - tau) * factor_income + T)
        cm.Equation("Consumer demand for <dim:i>", P * C, theta * income)
        cm.Equation("Consumer demand for the import good", P_M * C_M, theta_M * income)
        cm.Equation("Consumer demand for the investment good", P_I * C_I, theta_I * income)
        cm.Equation("Consumer demand for the government good", P_G * C_G, theta_G * income)

        cm.Equation("Market for <dim:i>", Y, C + X.sum("j") + X_E + X_I + X_G)
        cm.Equation("Market for the import good", Y_M, C_M + X_M.sum("j") + X_MM + X_MI)
        cm.Equation("Market for the investment good", Y_I, C_I)
        cm.Equation("Market for the government good", Y_G, C_G)
        cm.Equation("Labor market", L_s, L_d.sum("j") + L_G + resid)
        cm.Equation("Capital market", K_s, K_d.sum("j"))
        cm.Equation(
            "Consumption price index",
            (theta * P).sum("i") + theta_M * P_M + theta_I * P_I + theta_G * P_G,
            1.0,
        )

    return model


@cache
def load_and_cache_model(model_id: Literal[1, 2], **compile_kwargs) -> CGEModel:
    """
    Return model ``model_id`` compiled with ``compile_kwargs``, compiling it only on the first request in a session.

    Callers share the returned model, so a test that mutates it, by recompiling it or reassigning roles, builds its own.
    """
    hashable_kwargs = tuple(
        sorted(
            (name, tuple(value) if isinstance(value, list) else value)
            for name, value in compile_kwargs.items()
        )
    )
    return _compiled_model(model_id, hashable_kwargs)


@cache
def _compiled_model(model_id: Literal[1, 2], compile_kwargs: tuple) -> CGEModel:
    builders = {1: build_model_1, 2: build_model_2}
    if model_id not in builders:
        raise ValueError("Invalid model_id. Use 1 or 2.")
    return builders[model_id]().compile(**dict(compile_kwargs))
