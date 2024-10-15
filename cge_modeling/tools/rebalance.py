import logging

from collections.abc import Callable
from functools import partial

import cvxpy as cp
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt

from better_optimize import minimize
from scipy.sparse import csr_array

from cge_modeling.compile.pytensor_tools import rewrite_pregrad

_log = logging.getLogger(__name__)


class SAMTransformer:
    __slots__ = ("_fit", "scale", "negative_locs", "n_levels")

    def __init__(self):
        self._fit = False
        self.scale: float | None = None
        self.negative_locs: list[tuple] | None = None
        self.n_levels: int | None = None

    def fit(self, df):
        n_levels = self.n_levels = len(df.index.names)
        assert self.n_levels == len(
            df.columns.names
        ), "Index and columns must have the same number of levels"

        self.scale = df.sum().sum()
        negative_cells = (
            (df < 0)
            .melt(ignore_index=False)
            .loc[lambda x: x.value]
            .reset_index()
            .drop(columns=["value"])
            .values
        )

        def gather_indices(x):
            if n_levels > 1:
                return tuple(x[:n_levels]), tuple(x[n_levels:])
            return x[:n_levels].item(), x[n_levels:].item()

        self.negative_locs = [gather_indices(x) for x in negative_cells]

        self._fit = True

    def transform(self, df):
        if not self._fit:
            raise ValueError("Must fit transformer before transforming data")
        assert self.n_levels == len(df.columns.names) == len(df.index.names)
        df = df.copy()
        df = df / self.scale
        for row, col in self.negative_locs:
            df.loc[col, row] = -df.loc[row, col]
            df.loc[row, col] = 0

        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df):
        if not self._fit:
            raise ValueError("Must fit transformer before transforming data")
        assert self.n_levels == len(df.columns.names) == len(df.index.names)

        df = df.copy()
        for row, col in self.negative_locs:
            df.loc[row, col] = -df.loc[col, row]
            df.loc[col, row] = 0
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


def sparse_to_vector_and_matrix(Z):
    row_indices, col_indices = Z.nonzero()
    x = Z.data
    nnz = Z.nnz

    num_elements = Z.shape[0] * Z.shape[1]
    A_data = np.ones(nnz)
    A_rows = row_indices * Z.shape[1] + col_indices  # Flattened indices
    A_cols = np.arange(nnz)

    A = csr_array((A_data, (A_rows, A_cols)), shape=(num_elements, nnz))

    return x, A


def _balance_with_cvxpy(normalized_sam: pd.DataFrame, scale, method, **minimize_kwargs):
    method = "CLARABEL" if method is None else method
    delta = minimize_kwargs.pop("delta", 1e-10)

    n = normalized_sam.shape[0]
    Z = csr_array(normalized_sam.values)

    x, A = sparse_to_vector_and_matrix(Z)
    k = x.shape[0]
    x_optim = cp.Variable(name="x_optim", shape=k)

    Z_optim = (A @ x_optim).reshape((n, n), order="C")

    constraints = [
        x_optim.sum() == 1,
        x_optim >= 0,
        (Z_optim.sum(axis=0) * scale) == (Z_optim.sum(axis=1) * scale),
    ]

    objective = cp.kl_div(x_optim, x + delta).sum()

    prob = cp.Problem(cp.Minimize(objective), constraints=constraints)
    prob.solve(solver=method, **minimize_kwargs)

    Z_new = (A @ x_optim.value).reshape((n, n))
    balanced_SAM = pd.DataFrame(Z_new, index=normalized_sam.index, columns=normalized_sam.columns)

    return balanced_SAM, prob


def _balance_with_scipy(
    normalized_sam: pd.DataFrame,
    use_grad=True,
    use_hess=False,
    use_hessp=True,
    use_constraint_grad=True,
    mode=None,
    method: str | None = None,
    progressbar=True,
    **minimize_kwargs,
):
    method = "SLSQP" if method is None else method

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

    res = minimize(
        f_ce,
        x0,
        method=method,
        jac=None if not use_grad else f_grad,
        hess=None if not use_hess else f_hess,
        hessp=None if not use_hessp else f_jvp_ce,
        constraints=[
            {"type": "eq", "fun": f_1, "jac": f_1_grad},
            {"type": "eq", "fun": f_2, "jac": f_2_grad},
        ],
        bounds=[(0, 1)] * nonzero_values.shape[0],
        progressbar=progressbar,
        **minimize_kwargs,
    )

    if not res.success:
        _log.info("Optimization failed. Return values may not represent a balanced SAM")

    balanced_SAM = pd.DataFrame(
        f_X(res.x), index=normalized_sam.index, columns=normalized_sam.columns
    )

    return balanced_SAM, res


def balance_SAM(
    sam: pd.DataFrame,
    how="cross-entropy",
    use_cvxpy=True,
    use_grad=True,
    use_hess=False,
    use_hessp=True,
    use_constraint_grad=True,
    mode=None,
    method: str | None = None,
    progressbar=True,
    **minimize_kwargs,
):
    if how != "cross-entropy":
        raise NotImplementedError("Only cross-entropy method is implemented")

    sam_transformer = SAMTransformer()
    normalized_sam = sam_transformer.fit_transform(sam)

    if use_cvxpy:
        balanced_SAM, res = _balance_with_cvxpy(
            normalized_sam,
            scale=sam_transformer.scale,
            method=method,
            **minimize_kwargs,
        )
        success = res.status == cp.OPTIMAL

    else:
        balanced_SAM, res = _balance_with_scipy(
            normalized_sam,
            use_grad=use_grad,
            use_hess=use_hess,
            use_hessp=use_hessp,
            use_constraint_grad=use_constraint_grad,
            mode=mode,
            method=method,
            progressbar=progressbar,
            **minimize_kwargs,
        )
        success = res.success

    balanced_SAM = sam_transformer.inverse_transform(balanced_SAM)
    return balanced_SAM, res, success
