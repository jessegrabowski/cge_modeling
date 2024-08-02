import logging

from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt

from scipy import optimize

from cge_modeling.base.utilities import (
    CostFuncWrapper,
    _optimzer_early_stopping_wrapper,
)
from cge_modeling.tools.pytensor_tools import rewrite_pregrad

_log = logging.getLogger(__name__)


class SAMTransformer:
    __slots__ = ("_fit", "scale", "negative_mask")

    def __init__(self):
        self._fit = False
        self.scale: float | None = None
        self.negative_mask: np.ndarray | None = None

    def fit(self, df):
        self.scale = df.abs().sum().sum()
        self.negative_mask = df < 0
        self._fit = True

    def transform(self, df):
        if not self._fit:
            raise ValueError("Must fit transformer before transforming data")
        neg_mask = self.negative_mask

        df = df.copy()
        df = df / self.scale
        df.values.T[neg_mask] = -df.values[neg_mask]
        df.values[neg_mask] = 0

        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df):
        if not self._fit:
            raise ValueError("Must fit transformer before transforming data")
        neg_mask = self.negative_mask

        df = df.copy()
        df.values[neg_mask] = -df.values[neg_mask.T]
        df[neg_mask.T] = 0
        df = df * self.scale

        return df


def get_flat_nonzero_values(df):
    nonzero_values = df.values.ravel()
    return nonzero_values[nonzero_values > 0]


def _crossentropy_method_setup(
    df,
    use_grad=True,
    use_hess=False,
    use_hessp=True,
    use_constraint_grad=True,
    mode=None,
) -> tuple[
    Callable,
    Callable,
    Callable,
    Callable,
    Callable | None,
    Callable | None,
    Callable | None,
    Callable | None,
    Callable | None,
]:
    compile_fn = partial(pytensor.function, mode=mode)
    f_grad, f_1_grad, f_2_grad, f_hess, f_jvp_ce = None, None, None, None, None
    grad_ce = None

    X = pt.zeros(df.shape)
    x = pt.dvector("x")

    X = pt.set_subtensor(X[df.values > 0], x)
    f_X = compile_fn([x], X)

    nonzero_values = get_flat_nonzero_values(df)

    cross_entropy = (x * (pt.log(x + 1e-10) - pt.log(nonzero_values))).sum()

    constraint_1 = pt.linalg.norm(X.sum(axis=0) - X.sum(axis=1), ord=1)
    constraint_2 = x.sum() - 1

    f_ce = compile_fn([x], cross_entropy)

    f_1 = compile_fn([x], constraint_1)
    f_2 = compile_fn([x], constraint_2)

    if use_constraint_grad:
        rewrite_pregrad(constraint_1)
        rewrite_pregrad(constraint_2)

        f_1_grad = compile_fn([x], pt.grad(constraint_1, x))
        f_2_grad = compile_fn([x], pt.grad(constraint_2, x))

    if use_grad:
        rewrite_pregrad(cross_entropy)
        grad_ce = pt.grad(cross_entropy, x)
        f_grad = compile_fn([x], grad_ce)

    if use_hess:
        if not use_grad:
            raise ValueError("Must use gradient to compute Hessian")
        rewrite_pregrad(grad_ce)
        hess_ce = pytensor.gradient.jacobian(grad_ce, x)
        f_hess = compile_fn([x], hess_ce)

    if use_hessp:
        if not use_grad:
            raise ValueError("Must use gradient to compute Hessian")
        rewrite_pregrad(grad_ce)
        p = pt.dvector("p")
        jvp_ce = pytensor.gradient.Rop(grad_ce, x, p)
        f_jvp_ce = compile_fn([x, p], jvp_ce)

    return f_X, f_ce, f_1, f_2, f_grad, f_1_grad, f_2_grad, f_hess, f_jvp_ce


def balance_SAM(
    sam: pd.DataFrame,
    how="cross-entropy",
    use_grad=True,
    use_hess=False,
    use_hessp=True,
    use_constraint_grad=True,
    mode=None,
    method="SLSQP",
    progressbar=True,
    **minimize_kwargs,
):
    if how != "cross-entropy":
        raise NotImplementedError("Only cross-entropy method is implemented")
    if method not in ("SLSQP", "COBYLA", "trust-constr"):
        raise ValueError(
            "SAM rebalancing requires constrained optimization which is only supported by "
            f" the SLSQP, COBYLA, and trust-constr methods. Got {method}"
        )

    if method != "trust-constr" and (use_hess or use_hessp):
        _log.info(
            'Only method = "trust-constr" supports Hessian computation. Setting use_hess=False to save unncessary'
            " computation time."
        )
        use_hess = False
        use_hessp = False

    sam_transformer = SAMTransformer()
    normalized_sam = sam_transformer.fit_transform(sam)

    f_list = _crossentropy_method_setup(
        normalized_sam,
        use_grad=use_grad,
        use_hess=use_hess,
        use_hessp=use_hessp,
        use_constraint_grad=use_constraint_grad,
        mode=mode,
    )
    f_X, f_ce, f_1, f_2, f_grad, f_1_grad, f_2_grad, f_hess, f_jvp_ce = f_list

    nonzero_values = get_flat_nonzero_values(normalized_sam)
    x0 = np.random.dirichlet(alpha=np.ones_like(nonzero_values))

    x0 = minimize_kwargs.pop("x0", x0)
    tol = minimize_kwargs.pop("tol", 1e-12)
    options = minimize_kwargs.pop("options", {})
    maxiter = options.pop("maxiter", 25_000)
    options.update(maxiter=maxiter)

    objective = CostFuncWrapper(
        f_ce,
        args=None,
        f_jac=f_grad,
        f_hess=f_hess,
        maxeval=maxiter,
        progressbar=progressbar,
        update_every=10,
    )

    f_optim = partial(
        optimize.minimize,
        objective,
        x0,
        jac=use_grad,
        hess=objective.f_hess if use_hess else None,
        callback=objective.callback,
        hessp=f_jvp_ce,
        constraints=[
            {"type": "eq", "fun": f_1, "jac": f_1_grad},
            {"type": "eq", "fun": f_2, "jac": f_2_grad},
        ],
        bounds=[(0, 1)] * nonzero_values.shape[0],
        tol=tol,
        method=method,
        options=options,
        **minimize_kwargs,
    )
    res = _optimzer_early_stopping_wrapper(f_optim)

    if not res.success:
        _log.info("Optimization failed. Return values may not represent a balanced SAM")

    balanced_SAM = pd.DataFrame(
        f_X(res.x), index=normalized_sam.index, columns=normalized_sam.columns
    )
    balanced_SAM = sam_transformer.inverse_transform(balanced_SAM)
    return balanced_SAM, res
