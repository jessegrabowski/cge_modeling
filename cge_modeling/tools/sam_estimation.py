import warnings

from collections.abc import Hashable
from dataclasses import dataclass, field, replace
from numbers import Real

import cvxpy as cp
import narwhals as nw
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytensor.xtensor as px
import xarray as xr

from narwhals.typing import IntoDataFrame
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.xtensor.type import XTensorVariable
from scipy.linalg import null_space
from scipy.optimize import root
from scipy.sparse import coo_array, csr_array
from scipy.sparse.csgraph import connected_components
from scipy.special import kl_div

ROW = "row"
COLUMN = "col"

DUAL_SOLVER = "dual"
ACCOUNTING_IDENTITY = "accounting identity"
DUAL_TOLERANCE = 1e-12

# A cell or inequality slack within this fraction of the total absolute cell value counts as at its bound.
BOUND_TOLERANCE = 1e-6

Cell = tuple[Hashable, Hashable]


class SAMExpression:
    """
    A linear expression in the cells of a :class:`SymbolicSAM`, with an account label for every position of each dim.

    Expressions add, subtract, scale by numbers, sum over dims, and compare with ``==``, ``<=`` or ``>=`` to give a
    :class:`Constraint`. Anything that would make an expression nonlinear in the cells raises ``TypeError``.

    Data combines with an expression by account label, and NaN entries impose nothing. A ``pandas.Series`` matches a
    one-dim expression, a ``pandas.DataFrame`` matches an expression over both dims of the SAM, its index on the row
    dim and its columns on the column dim, and an ``xarray.DataArray`` matches by dim name over the expression's dims or
    a subset of them. A DataArray has to sit to the right of the expression, because xarray's operators do not defer to
    other types.
    """

    # Make numpy and pandas return NotImplemented from their operators, so that ``series <= expr`` and
    # ``array + expr`` reach this class's reflected methods.
    __array_ufunc__ = None
    __pandas_priority__ = 5000

    def __init__(self, sam: "SymbolicSAM", tensor: XTensorVariable, labels: dict[str, pd.Index]):
        self.sam = sam
        self.tensor = tensor
        self.labels = labels

    @property
    def dims(self) -> tuple[str, ...]:
        return self.tensor.type.dims

    def sum(self, dim: str | None = None) -> "SAMExpression":
        """Sum over ``dim``, or over every dim when ``dim`` is None."""
        summed_dims = self.dims if dim is None else (dim,)
        labels = {name: index for name, index in self.labels.items() if name not in summed_dims}
        return SAMExpression(self.sam, self.tensor.sum(dim), labels)

    def rename(self, mapping: dict[str, str]) -> "SAMExpression":
        """Rename dims, for example ``{"col": "row"}`` to compare column sums with row sums account by account."""
        labels = {mapping.get(name, name): index for name, index in self.labels.items()}
        return SAMExpression(self.sam, self.tensor.rename(mapping), labels)

    def __add__(self, other) -> "SAMExpression":
        return SAMExpression(self.sam, self.tensor + self._operand(other), self.labels)

    def __radd__(self, other) -> "SAMExpression":
        return self + other

    def __sub__(self, other) -> "SAMExpression":
        return SAMExpression(self.sam, self.tensor - self._operand(other), self.labels)

    def __rsub__(self, other) -> "SAMExpression":
        return -self + other

    def __neg__(self) -> "SAMExpression":
        return SAMExpression(self.sam, -self.tensor, self.labels)

    def __mul__(self, other) -> "SAMExpression":
        return SAMExpression(self.sam, self.tensor * _scalar(other, "multiply"), self.labels)

    def __rmul__(self, other) -> "SAMExpression":
        return self * other

    def __truediv__(self, other) -> "SAMExpression":
        return SAMExpression(self.sam, self.tensor / _scalar(other, "divide"), self.labels)

    def __eq__(self, other) -> "Constraint":  # type: ignore[override]
        return Constraint(is_equality=True, residual=self - other)

    def __le__(self, other) -> "Constraint":
        return Constraint(is_equality=False, residual=self - other)

    def __ge__(self, other) -> "Constraint":
        return Constraint(is_equality=False, residual=other - self)

    def __lt__(self, other):
        raise TypeError("Strict inequalities cannot be imposed. Use <= or >=")

    def __gt__(self, other):
        raise TypeError("Strict inequalities cannot be imposed. Use <= or >=")

    def _operand(self, other) -> XTensorVariable:
        """Return ``other`` as an xtensor that adds to this expression position by position."""
        if isinstance(other, SAMExpression):
            if other.sam is not self.sam:
                raise ValueError("Expressions from different SymbolicSAMs cannot be combined")
            if other.dims and self.dims and set(other.dims) != set(self.dims):
                row_dim, column_dim = self.sam.dims
                raise ValueError(
                    f"Expressions with dims {self.dims} and {other.dims} would broadcast into an outer product. "
                    f"Rename a dim first, for example sam.sum({row_dim!r}).rename({{{column_dim!r}: {row_dim!r}}})"
                )
            for dim in set(self.dims) & set(other.dims):
                if not self.labels[dim].equals(other.labels[dim]):
                    raise ValueError(
                        f"Expressions combine position by position, but their accounts on dim {dim!r} differ: "
                        f"{list(self.labels[dim])} and {list(other.labels[dim])}"
                    )
            return other.tensor

        if isinstance(other, Real):
            return px.as_xtensor(np.asarray(float(other)), dims=())

        if isinstance(other, pd.Series):
            if len(self.dims) != 1:
                raise ValueError(
                    f"A Series can only be compared with a one-dim expression, not dims {self.dims}"
                )
            return self._labeled_operand(other, dims=self.dims)

        if isinstance(other, pd.DataFrame):
            row_dim, column_dim = self.sam.dims
            if set(self.dims) != {row_dim, column_dim}:
                raise ValueError(
                    f"A DataFrame can only be compared with an expression over {row_dim!r} and {column_dim!r}, not "
                    f"dims {self.dims}. Its index is {row_dim!r} and its columns are {column_dim!r}"
                )
            return self._labeled_operand(other, dims=self.sam.dims)

        if isinstance(other, xr.DataArray):
            if not set(other.dims) <= set(self.dims):
                raise ValueError(
                    f"A DataArray with dims {other.dims} would broadcast an expression with dims {self.dims} into an "
                    f"outer product"
                )
            if other.ndim == 0:
                return px.as_xtensor(np.asarray(float(other)), dims=())
            values = other.to_series() if other.ndim == 1 else other.to_pandas()
            return self._labeled_operand(values, dims=tuple(map(str, other.dims)))

        if isinstance(other, np.ndarray):
            raise TypeError(
                "Cannot combine a SAM expression with an unlabeled array. Label it as a Series, DataFrame or DataArray, "
                "and put a DataArray to the right of the expression"
            )
        raise TypeError(f"Cannot combine a SAM expression with {type(other).__name__}")

    def _labeled_operand(
        self, values: pd.Series | pd.DataFrame, dims: tuple[str, ...]
    ) -> XTensorVariable:
        """Align ``values`` to this expression's labels, with the index on ``dims[0]`` and any columns on ``dims[1]``."""
        axes = [values.index] if isinstance(values, pd.Series) else [values.index, values.columns]
        for dim, labels in zip(dims, axes, strict=True):
            unknown_labels = labels.difference(self.labels[dim])
            if len(unknown_labels) > 0:
                raise ValueError(f"Labels not in dim {dim!r}: {list(unknown_labels)}")

        if isinstance(values, pd.Series):
            aligned = values.reindex(self.labels[dims[0]])
        else:
            aligned = values.reindex(index=self.labels[dims[0]], columns=self.labels[dims[1]])
        return px.as_xtensor(aligned.to_numpy(dtype=float), dims=dims)


@dataclass(frozen=True, eq=False)
class Constraint:
    """
    A linear constraint ``residual == 0`` or ``residual <= 0`` on the cells of a :class:`SymbolicSAM`.

    A residual with dims is one constraint per position. Positions where the residual's constant part is NaN, as from a
    Series with missing values, are dropped. An equality with a ``penalty`` is a measurement with error, which adds the
    penalty of its residual to the objective in place of holding exactly.
    """

    is_equality: bool
    residual: SAMExpression
    penalty: "Penalty | None" = None

    @property
    def sam(self) -> "SymbolicSAM":
        return self.residual.sam

    def normal(self, sigma: "ScaleData") -> "Constraint":
        r"""
        Return this equality as a measurement with normal error, adding
        :math:`\frac{1}{2} \sum_m (r_m / \sigma_m)^2` to the objective for residuals :math:`r_m`.

        Parameters
        ----------
        sigma : float, pandas.Series, pandas.DataFrame or xarray.DataArray
            Positive standard deviation of the error, in the prior's units. Labeled data aligns with the constraint's
            positions, like the right-hand side of a comparison.
        """
        return self._with_penalty(NormalPenalty(sigma))

    def laplace(self, scale: "ScaleData") -> "Constraint":
        r"""
        Return this equality as a measurement with Laplace error, adding :math:`\sum_m |r_m| / b_m` to the objective
        for residuals :math:`r_m`.

        Parameters
        ----------
        scale : float, pandas.Series, pandas.DataFrame or xarray.DataArray
            Positive scale :math:`b` of the error, in the prior's units. Labeled data aligns with the constraint's
            positions, like the right-hand side of a comparison.
        """
        return self._with_penalty(LaplacePenalty(scale))

    def huber(self, sigma: "ScaleData", threshold: float) -> "Constraint":
        r"""
        Return this equality as a measurement with Huber error, which is normal near zero and Laplace in the tails.

        Each standardized residual :math:`u_m = r_m / \sigma_m` adds :math:`\frac{1}{2} u_m^2` to the objective when
        :math:`|u_m| \le k`, and :math:`k |u_m| - \frac{1}{2} k^2` beyond it.

        Parameters
        ----------
        sigma : float, pandas.Series, pandas.DataFrame or xarray.DataArray
            Positive scale of the error, in the prior's units. Labeled data aligns with the constraint's positions,
            like the right-hand side of a comparison.
        threshold : float
            Positive :math:`k`, the standardized residual where the penalty turns from quadratic to linear.
        """
        if not threshold > 0.0:
            raise ValueError(f"The Huber threshold must be positive, not {threshold}")
        return self._with_penalty(HuberPenalty(sigma, float(threshold)))

    def _with_penalty(self, penalty: "Penalty") -> "Constraint":
        if not self.is_equality:
            raise ValueError(
                "Only an equality can be a measurement with error. Inequalities always hold exactly"
            )
        if self.penalty is not None:
            raise ValueError("This constraint already has an error distribution")
        return replace(self, penalty=penalty)

    def __bool__(self):
        raise TypeError(
            "A SAM constraint has no truth value. Pass it to SymbolicSAM.subject_to, and write a range as two constraints"
        )


@dataclass(frozen=True)
class LinearConstraints:
    """Constraints ``equality_operator @ x == equality_values`` and ``inequality_operator @ x <= inequality_values``."""

    equality_operator: csr_array
    equality_values: np.ndarray
    inequality_operator: csr_array
    inequality_values: np.ndarray


@dataclass(frozen=True)
class SAMEstimate:
    """
    A balanced SAM and the solve that produced it.

    Attributes
    ----------
    sam : pandas.DataFrame, xarray.DataArray or native frame
        The estimated SAM, in the prior's type and layout.
    objective : float
        The objective value at the estimate, including the penalties of measurements with error, in the prior's units.
    status : str
        The solver status, in cvxpy's vocabulary.
    """

    sam: IntoDataFrame | xr.DataArray
    objective: float
    status: str
    _symbolic_sam: "SymbolicSAM" = field(repr=False, compare=False)
    _objective: "Objective" = field(repr=False, compare=False)
    _constraints: dict[str, Constraint] = field(repr=False, compare=False)
    _cells: np.ndarray = field(repr=False, compare=False)

    def covariance(self) -> pd.DataFrame:
        """
        Return the covariance of the estimated cells under a :class:`LeastSquares` objective.

        The objective makes the cells independent and normal around the prior with standard deviation ``sigma``.
        Measurements made with :meth:`Constraint.normal` are normal observations of linear expressions, and every other
        equality, including the accounting identity, holds exactly. The covariance is that of the cells given all of
        them, as the constraints stood when :meth:`SymbolicSAM.fit` ran. It is exact when no cell sits at zero and no
        inequality binds, because the estimate is then the mean of that normal distribution.

        Returns
        -------
        covariance : pandas.DataFrame
            Covariance of the prior's nonzero cells, in the prior's units squared, indexed on both axes by the row and
            column account of each cell.

        Raises
        ------
        ValueError
            If the objective is not :class:`LeastSquares`, a measurement error is not normal, a cell sits at zero, or
            an inequality binds.
        """
        return _least_squares_covariance(
            self._symbolic_sam, self._objective, self._constraints, self._cells
        )


CellData = Real | np.ndarray | pd.DataFrame | xr.DataArray
ScaleData = Real | pd.Series | pd.DataFrame | xr.DataArray


class Penalty:
    """The negative log density of a measurement error, applied to residuals divided by their scale."""

    scale: ScaleData

    def expression(self, standardized_residuals: cp.Expression) -> cp.Expression:
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class NormalPenalty(Penalty):
    scale: ScaleData

    def expression(self, standardized_residuals: cp.Expression) -> cp.Expression:
        return 0.5 * cp.sum_squares(standardized_residuals)


@dataclass(frozen=True, eq=False)
class LaplacePenalty(Penalty):
    scale: ScaleData

    def expression(self, standardized_residuals: cp.Expression) -> cp.Expression:
        return cp.norm1(standardized_residuals)


@dataclass(frozen=True, eq=False)
class HuberPenalty(Penalty):
    scale: ScaleData
    threshold: float

    def expression(self, standardized_residuals: cp.Expression) -> cp.Expression:
        # cvxpy's huber is u^2 inside the threshold and 2 k |u| - k^2 outside, twice the Huber loss.
        return 0.5 * cp.sum(cp.huber(standardized_residuals, self.threshold))


class Objective:
    """
    A distance of the cells from the prior, which :meth:`SymbolicSAM.fit` minimizes.

    Every objective is the negative log density of a prior on the cells, up to a constant, in the prior's units. That
    puts it on the same scale as the penalties of measurements with error.
    """

    def expression(
        self, sam: "SymbolicSAM", magnitudes: cp.Expression, scale: float
    ) -> cp.Expression:
        """Return the objective divided by :meth:`unit`, for cell magnitudes divided by ``scale``."""
        raise NotImplementedError

    def unit(self, scale: float) -> float:
        """Return the factor that converts :meth:`expression` to the prior's units."""
        return scale


@dataclass(frozen=True, eq=False)
class KL(Objective):
    r"""
    Weighted generalized KL divergence of the cell magnitudes from the prior magnitudes,

    .. math::

        \sum_k w_k \left( |x_k| \log \frac{|x_k|}{|z_k|} - |x_k| + |z_k| \right).

    Parameters
    ----------
    weights : float, numpy.ndarray, pandas.DataFrame or xarray.DataArray, optional
        Positive weight of each cell, as a number or laid out like the prior. A larger weight holds a cell closer to
        its prior value. Default is None, which weights every cell by one.
    """

    weights: CellData | None = None

    def expression(
        self, sam: "SymbolicSAM", magnitudes: cp.Expression, scale: float
    ) -> cp.Expression:
        divergence = cp.kl_div(magnitudes, np.abs(sam.prior_values) / scale)
        if self.weights is None:
            return cp.sum(divergence)
        return cp.sum(cp.multiply(_cell_values(sam, self.weights, "weights"), divergence))


@dataclass(frozen=True, eq=False)
class LeastSquares(Objective):
    r"""
    Weighted least squares distance of the cells from the prior, the negative log density of independent normal cells,

    .. math::

        \frac{1}{2} \sum_k \left( \frac{x_k - z_k}{\sigma_k} \right)^2.

    Parameters
    ----------
    sigma : float, numpy.ndarray, pandas.DataFrame or xarray.DataArray, optional
        Positive standard error of each cell, in the prior's units, as a number or laid out like the prior. A smaller
        sigma holds a cell closer to its prior value. Default is None, which uses the magnitude of each prior cell.
    """

    sigma: CellData | None = None

    def expression(
        self, sam: "SymbolicSAM", magnitudes: cp.Expression, scale: float
    ) -> cp.Expression:
        prior_magnitudes = np.abs(sam.prior_values)
        sigma = prior_magnitudes if self.sigma is None else _cell_values(sam, self.sigma, "sigma")
        return 0.5 * cp.sum_squares((magnitudes - prior_magnitudes / scale) / (sigma / scale))

    def unit(self, scale: float) -> float:
        return 1.0


@dataclass(frozen=True, eq=False)
class ReverseKL(Objective):
    r"""
    Generalized KL divergence of the prior magnitudes from the cell magnitudes,

    .. math::

        \sum_k |z_k| \log \frac{|z_k|}{|x_k|} - |z_k| + |x_k|.
    """

    def expression(
        self, sam: "SymbolicSAM", magnitudes: cp.Expression, scale: float
    ) -> cp.Expression:
        return cp.sum(cp.kl_div(np.abs(sam.prior_values) / scale, magnitudes))


@dataclass(frozen=True, eq=False)
class Pearson(Objective):
    r"""
    Pearson chi-squared distance of the cells from the prior, which equals :class:`KL` to second order,

    .. math::

        \frac{1}{2} \sum_k \frac{(x_k - z_k)^2}{|z_k|}.
    """

    def expression(
        self, sam: "SymbolicSAM", magnitudes: cp.Expression, scale: float
    ) -> cp.Expression:
        prior_magnitudes = np.abs(sam.prior_values) / scale
        return 0.5 * cp.sum_squares((magnitudes - prior_magnitudes) / np.sqrt(prior_magnitudes))


class SymbolicSAM:
    """
    A SAM whose nonzero prior cells are symbolic, with named linear constraints on them.

    Each nonzero cell of the prior is an entry of the vector ``cells``, in row-major order, and keeps the sign of its
    prior value. Cells that are zero in the prior stay zero.

    Parameters
    ----------
    prior : pandas.DataFrame, xarray.DataArray or eager narwhals-compatible frame
        Square prior SAM. Rows are receipts and columns are payments, and both axes hold the same account labels in the
        same order. The SAM's dims are the names of a pandas DataFrame's index and columns, ``"row"`` and ``"col"`` for
        an axis without a name, or the two dims of a DataArray, whose coordinates are the account labels. A frame
        without an index, such as a polars DataFrame, holds its row labels in ``index_column`` and is named for its
        accounts in its other columns.
    index_column : str, optional
        The column of a frame prior that holds the row labels, and the name of the row dim. The column dim is
        ``"col"``. Required for a frame without an index. Default is None, which uses a pandas DataFrame's index.
    """

    def __init__(self, prior: IntoDataFrame | xr.DataArray, index_column: str | None = None):
        self._native_prior = prior
        self._index_column = index_column

        prior_frame, row_dim, column_dim = _prior_frame_and_dims(prior, index_column)

        if row_dim == column_dim:
            raise ValueError(
                f"The prior's rows and columns are both named {row_dim!r}. Rename one so that the two dims differ"
            )
        if not prior_frame.index.equals(prior_frame.columns):
            raise ValueError(
                "The prior's index and columns must hold the same account labels in the same order"
            )

        values = prior_frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                "The prior must be finite. Fill missing cells with 0.0 before building a SymbolicSAM"
            )

        self.prior = prior_frame
        self.dims = (row_dim, column_dim)
        self.rows, self.columns = np.nonzero(values)
        self.prior_values = values[self.rows, self.columns]
        self.signs = np.sign(self.prior_values)
        self.constraints: dict[str, Constraint] = {}
        # The solvers work in cells divided by this total, so every SAM is solved at the same scale.
        self._total_magnitude = np.abs(self.prior_values).sum()

        self.cells = pt.vector("cells", shape=(self.n_cells,))
        dense = pt.set_subtensor(pt.zeros(prior_frame.shape)[self.rows, self.columns], self.cells)
        self._matrix = SAMExpression(
            self,
            px.as_xtensor(dense, dims=self.dims),
            {row_dim: prior_frame.index, column_dim: prior_frame.columns},
        )

    @property
    def n_accounts(self) -> int:
        return self.prior.shape[0]

    @property
    def n_cells(self) -> int:
        return self.prior_values.size

    def sel(self, **selections) -> SAMExpression:
        """
        Select cells by account label, keyed by dim, for example ``sam.sel(row=..., col=...)``.

        Each selection is a label, a leading part of a MultiIndex label such as ``"Factor"`` or ``("Factor", "Labor")``
        (selecting every account under it), or a list of those. A single account drops its dim, as in xarray. A dim
        without a selection, or with ``None``, keeps every account.
        """
        unknown_dims = set(selections) - set(self.dims)
        if unknown_dims:
            raise ValueError(f"Unknown dims {sorted(unknown_dims)}. The SAM's dims are {self.dims}")

        indexers = {}
        labels = dict(self._matrix.labels)
        for dim, selection in selections.items():
            if selection is None:
                continue
            positions = _label_positions(labels[dim], selection)
            indexers[dim] = positions
            if np.ndim(positions) == 0:
                del labels[dim]
            else:
                labels[dim] = labels[dim][positions]

        return SAMExpression(self, self._matrix.tensor.isel(indexers), labels)

    def sum(self, dim: str | None = None) -> SAMExpression:
        """Sum the whole SAM over ``dim``: the column dim gives row sums and the row dim gives column sums."""
        return self._matrix.sum(dim)

    def subject_to(self, **constraints: Constraint) -> None:
        """Add constraints, each named by its keyword."""
        for name, constraint in constraints.items():
            if not isinstance(constraint, Constraint) or constraint.sam is not self:
                raise TypeError(f"Constraint {name!r} must compare expressions of this SymbolicSAM")
            if name in self.constraints:
                raise ValueError(f"A constraint named {name!r} already exists")
            if name == ACCOUNTING_IDENTITY:
                raise ValueError(
                    f"The name {ACCOUNTING_IDENTITY!r} is reserved for the identity that fit always imposes"
                )
            self.constraints[name] = constraint

    def fit(
        self, objective: Objective | None = None, solver: str | None = None, **solver_kwargs
    ) -> SAMEstimate:
        r"""
        Estimate the balanced SAM closest to the prior, subject to the constraints.

        Every account's row sum equals its column sum, in addition to :attr:`constraints`. Every cell keeps the sign of
        its prior value, so cells that are zero in the prior stay zero. Each objective is zero exactly when the estimate
        equals the prior. A constraint made a measurement with error by :meth:`Constraint.normal`,
        :meth:`Constraint.laplace` or :meth:`Constraint.huber` adds its penalty to the objective, and does not hold
        exactly.

        Under the accounting identity alone, the unweighted :class:`KL` estimate scales each positive cell
        :math:`(i, j)` by :math:`d_j / d_i` and each negative cell by :math:`d_i / d_j`. The ``"dual"`` solver finds the
        account scales :math:`d` with :func:`scipy.optimize.root`, and every other solver solves the full problem in
        cvxpy.

        Parameters
        ----------
        objective : Objective, optional
            The distance from the prior to minimize: :class:`KL`, :class:`LeastSquares`, :class:`ReverseKL` or
            :class:`Pearson`. Default is None, which uses the unweighted :class:`KL`.
        solver : str, optional
            ``"dual"``, for the unweighted :class:`KL` objective on a SAM with no constraints, or the name of a cvxpy
            solver. Default is None, which picks ``"dual"`` where it applies and CLARABEL otherwise.
        **solver_kwargs
            Passed to :func:`scipy.optimize.root` for the ``"dual"`` solver, which uses ``method="lm"`` unless one is
            given, and to ``cvxpy.Problem.solve`` otherwise. CLARABEL's tolerances apply after every cell is divided by
            the total of the absolute cell values, so its error in each cell is roughly absolute. Tighter
            ``tol_gap_abs``, ``tol_gap_rel`` and ``tol_feas`` reduce it.

        Returns
        -------
        estimate : SAMEstimate
            The estimated SAM, the objective value, and the solver status.

        Raises
        ------
        ValueError
            If balancing would set cells to zero, a constraint involves no nonzero cell where it cannot hold, or the
            solver finds the problem infeasible or unbounded.
        """
        forced_to_zero = cells_forced_to_zero(self)
        if forced_to_zero:
            raise ValueError(
                f"Balancing sets these cells to zero, because they lie on no cycle of the account graph: "
                f"{forced_to_zero}. Remove them from the prior or add the missing flows."
            )

        if objective is None:
            objective = KL()

        dual_applies = (
            isinstance(objective, KL) and objective.weights is None and not self.constraints
        )
        if solver is None:
            solver = DUAL_SOLVER if dual_applies else cp.CLARABEL

        if solver == DUAL_SOLVER:
            if not dual_applies:
                raise ValueError(
                    f"The {DUAL_SOLVER!r} solver covers only the unweighted KL objective under the accounting "
                    f"identity. Use a cvxpy solver for a SAM with constraints or another objective"
                )
            return _fit_dual(self, **solver_kwargs)

        constraints = dict(self.constraints)
        linear = linear_constraints(self, _hard_constraints(self, constraints))

        scale = self._total_magnitude
        magnitudes = cp.Variable(self.n_cells, nonneg=True)
        cells = cp.multiply(self.signs, magnitudes)

        problem_constraints = [linear.equality_operator @ cells == linear.equality_values / scale]
        if linear.inequality_operator.shape[0] > 0:
            problem_constraints.append(
                linear.inequality_operator @ cells <= linear.inequality_values / scale
            )

        unit = objective.unit(scale)
        problem = cp.Problem(
            cp.Minimize(self._scaled_objective(objective, magnitudes, scale, constraints)),
            problem_constraints,
        )
        problem.solve(solver=solver, **solver_kwargs)

        if problem.status in cp.settings.INF_OR_UNB or magnitudes.value is None:
            raise ValueError(
                f"The SAM estimate has no solution: solver status is {problem.status!r}. The constraints contradict "
                f"each other or the accounting identity"
            )
        if problem.status != cp.OPTIMAL:
            warnings.warn(
                f"The SAM estimate did not solve to optimality: solver status is {problem.status!r}",
                stacklevel=2,
            )

        cell_values = self.signs * magnitudes.value * scale
        return SAMEstimate(
            sam=self.like_prior(cell_values),
            objective=float(problem.value) * unit,
            status=problem.status,
            _symbolic_sam=self,
            _objective=objective,
            _constraints=constraints,
            _cells=cell_values,
        )

    def objective_value(
        self, candidate: IntoDataFrame | xr.DataArray, objective: Objective | None = None
    ) -> float:
        """
        Return the objective at ``candidate``, including the penalties of measurements with error, in the prior's units.

        This is the quantity :meth:`fit` minimizes, under the measurements in :attr:`constraints` as they stand now. It
        equals :attr:`SAMEstimate.objective` at an estimate fitted under the same constraints. Hard constraints,
        including the accounting identity, are not checked, so ``candidate`` need not satisfy them.

        Parameters
        ----------
        candidate : pandas.DataFrame, xarray.DataArray or eager narwhals-compatible frame
            A SAM in the prior's type and layout, with the prior's account labels in the prior's order. Every cell
            that is zero in the prior is zero, and every other cell has the sign of its prior value or is zero.
        objective : Objective, optional
            The objective to evaluate, as passed to :meth:`fit`. Default is None, which uses the unweighted
            :class:`KL`.

        Returns
        -------
        value : float
            The objective at ``candidate``.

        Raises
        ------
        ValueError
            If ``candidate`` has different account labels from the prior, a cell that is not finite, a nonzero cell
            where the prior is zero, or a cell with the opposite sign of its prior value.
        """
        if objective is None:
            objective = KL()

        frame, _, _ = _prior_frame_and_dims(candidate, self._index_column)
        if not (frame.index.equals(self.prior.index) and frame.columns.equals(self.prior.columns)):
            raise ValueError(
                "The candidate must have the prior's account labels in the prior's order"
            )

        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("The candidate must be finite")

        outside_prior = np.ones(values.shape, dtype=bool)
        outside_prior[self.rows, self.columns] = False
        if (values[outside_prior] != 0.0).any():
            raise ValueError("The candidate has nonzero cells where the prior is zero")

        cell_values = values[self.rows, self.columns]
        if (cell_values * self.signs < 0.0).any():
            raise ValueError("The candidate has cells with the opposite sign of their prior value")

        scale = self._total_magnitude
        magnitudes = cp.Constant(np.abs(cell_values) / scale)
        scaled_value = self._scaled_objective(objective, magnitudes, scale, self.constraints).value
        return float(scaled_value) * objective.unit(scale)

    def _scaled_objective(
        self,
        objective: Objective,
        magnitudes: cp.Expression,
        scale: float,
        constraints: dict[str, Constraint],
    ) -> cp.Expression:
        """Return the objective plus the penalties of the measurements among ``constraints``, in the units of
        :meth:`Objective.expression`."""
        cells = cp.multiply(self.signs, magnitudes)
        unit = objective.unit(scale)
        terms = [objective.expression(self, magnitudes, scale)]
        for constraint in constraints.values():
            if constraint.penalty is None:
                continue
            operator, values, error_scales = _measurement_rows(self, constraint)
            standardized_residuals = (operator @ cells - values / scale) / (error_scales / scale)
            terms.append(constraint.penalty.expression(standardized_residuals) / unit)
        return cp.sum(cp.hstack(terms))

    def like_prior(self, cell_values: np.ndarray) -> IntoDataFrame | xr.DataArray:
        """Return a SAM in the prior's type and layout, with ``cell_values`` in the prior's nonzero cells."""
        values = np.zeros(self.prior.shape)
        values[self.rows, self.columns] = cell_values

        if isinstance(self._native_prior, xr.DataArray):
            return self._native_prior.copy(data=values)
        if self._index_column is None:
            return pd.DataFrame(values, index=self.prior.index, columns=self.prior.columns)

        native = nw.from_native(self._native_prior, eager_only=True)
        columns = {
            name: (
                self.prior.index.to_list()
                if name == self._index_column
                else values[:, self.prior.columns.get_loc(name)]
            )
            for name in native.columns
        }
        return nw.from_dict(columns, backend=native.implementation).to_native()


def linear_constraints(
    sam: SymbolicSAM, constraints: list[Constraint] | dict[str, Constraint]
) -> LinearConstraints:
    """
    Return the coefficients of ``constraints`` over the cells of ``sam``.

    Each position of a constraint's residual is one row. The coefficients are the Jacobian of the residual with respect
    to the cells, and the values are minus its constant part.

    Parameters
    ----------
    sam : SymbolicSAM
        The SAM whose cells the constraints restrict.
    constraints : list of Constraint or dict of str to Constraint
        The constraints, named by their keys when given as a dict, or by their position in a list.

    Returns
    -------
    linear : LinearConstraints
        The equality rows, then the inequality rows, in the order of ``constraints``.

    Raises
    ------
    ValueError
        If a constraint involves no nonzero cell of the prior at a position where its value cannot hold.
    """
    named = (
        constraints
        if isinstance(constraints, dict)
        else {
            f"constraints[{position}]": constraint
            for position, constraint in enumerate(constraints)
        }
    )
    equalities = {name: constraint for name, constraint in named.items() if constraint.is_equality}
    inequalities = {
        name: constraint for name, constraint in named.items() if not constraint.is_equality
    }
    return LinearConstraints(
        *_linear_rows(sam, equalities, is_equality=True),
        *_linear_rows(sam, inequalities, is_equality=False),
    )


def cells_forced_to_zero(sam: SymbolicSAM) -> list[Cell]:
    """
    Return the cells that every balanced SAM with this pattern sets to zero.

    A balanced SAM with this pattern decomposes into flows around directed cycles of the account graph, where a positive
    cell ``(i, j)`` is an edge from ``i`` to ``j`` and a negative cell an edge from ``j`` to ``i``. A cell whose edge
    joins two strongly connected components lies on no cycle.
    """
    positive = sam.signs > 0
    sources = np.where(positive, sam.rows, sam.columns)
    targets = np.where(positive, sam.columns, sam.rows)

    graph = coo_array(
        (np.ones(sam.n_cells), (sources, targets)), shape=(sam.n_accounts, sam.n_accounts)
    )
    _, component = connected_components(graph, directed=True, connection="strong")

    forced = np.flatnonzero(component[sources] != component[targets])
    return [
        (sam.prior.index[sam.rows[cell]], sam.prior.columns[sam.columns[cell]]) for cell in forced
    ]


def _prior_frame_and_dims(
    prior: IntoDataFrame | xr.DataArray, index_column: str | None
) -> tuple[pd.DataFrame, str, str]:
    """Return ``prior`` as a pandas DataFrame with account labels on both axes, and the names of its two dims."""
    if isinstance(prior, xr.DataArray):
        if index_column is not None:
            raise ValueError(
                "index_column applies to a frame prior. A DataArray prior takes its labels from its coordinates"
            )
        if prior.ndim != 2:
            raise ValueError(f"A DataArray prior must have two dims, not {prior.dims}")
        row_dim, column_dim = map(str, prior.dims)
        prior_frame = pd.DataFrame(
            prior.to_numpy(),
            index=prior.get_index(row_dim),
            columns=prior.get_index(column_dim),
        )
    elif index_column is None:
        if not isinstance(prior, pd.DataFrame):
            library = type(prior).__module__.partition(".")[0]
            raise ValueError(
                f"A {library}.{type(prior).__name__} prior has no index. Pass index_column, the column that holds "
                f"the row labels"
            )
        prior_frame = prior
        row_dim = ROW if prior.index.name is None else str(prior.index.name)
        column_dim = COLUMN if prior.columns.name is None else str(prior.columns.name)
    else:
        frame = nw.from_native(prior, eager_only=True)
        if index_column not in frame.columns:
            raise ValueError(
                f"index_column {index_column!r} is not a column of the prior: {frame.columns}"
            )
        accounts = [name for name in frame.columns if name != index_column]
        row_dim, column_dim = index_column, COLUMN
        prior_frame = pd.DataFrame(
            frame.select(accounts).to_numpy(),
            index=pd.Index(frame[index_column].to_list(), name=index_column),
            columns=pd.Index(accounts),
        )

    return prior_frame, row_dim, column_dim


def _fit_dual(sam: SymbolicSAM, **root_kwargs) -> SAMEstimate:
    """
    Balance ``sam`` under the accounting identity alone by solving for log account scales.

    With log scales ``u``, cell ``k`` from account ``c`` to account ``r`` has magnitude
    ``|z_k| exp(s_k (u_c - u_r))``. Each account's spending minus its income is then the gradient of the dual objective,
    the sum of those magnitudes, so balance is the root of that gradient, and its Jacobian is a graph Laplacian. Fixing
    one log scale in each group of connected accounts removes the Laplacian's null space. The solve is optimal when
    every account balances to ``DUAL_TOLERANCE`` of its turnover.
    """
    scale = sam._total_magnitude
    weights = np.abs(sam.prior_values) / scale

    cell_index = np.arange(sam.n_cells)
    incidence = csr_array(
        (
            np.concatenate([sam.signs, -sam.signs]),
            (np.concatenate([cell_index, cell_index]), np.concatenate([sam.columns, sam.rows])),
        ),
        shape=(sam.n_cells, sam.n_accounts),
    )

    graph = coo_array(
        (np.ones(sam.n_cells), (sam.rows, sam.columns)), shape=(sam.n_accounts, sam.n_accounts)
    )
    _, group = connected_components(graph, directed=False)
    _, first_in_group = np.unique(group, return_index=True)
    free = np.asarray(abs(incidence).sum(axis=0)).ravel() > 0
    free[first_in_group] = False

    def log_scales(free_log_scales):
        scales = np.zeros(sam.n_accounts)
        scales[free] = free_log_scales
        return scales

    def magnitudes(free_log_scales):
        return weights * np.exp(incidence @ log_scales(free_log_scales))

    def jacobian(free_log_scales):
        laplacian = incidence.T @ incidence.multiply(magnitudes(free_log_scales)[:, None])
        return laplacian.toarray()[np.ix_(free, free)]

    result = root(
        lambda free_log_scales: (incidence.T @ magnitudes(free_log_scales))[free],
        x0=np.zeros(free.sum()),
        jac=jacobian,
        **{"method": "lm", **root_kwargs},
    )

    balanced = magnitudes(result.x)
    imbalance = np.abs(incidence.T @ balanced)
    turnover = abs(incidence).T @ balanced
    relative_imbalance = np.divide(
        imbalance, turnover, out=np.zeros_like(imbalance), where=turnover > 0
    )
    status = cp.OPTIMAL if relative_imbalance.max() <= DUAL_TOLERANCE else cp.OPTIMAL_INACCURATE

    if status != cp.OPTIMAL:
        warnings.warn(
            f"The SAM estimate did not solve to optimality: solver status is {status!r}",
            stacklevel=3,
        )

    return SAMEstimate(
        sam=sam.like_prior(sam.signs * balanced * scale),
        objective=float(kl_div(balanced, weights).sum()) * scale,
        status=status,
        _symbolic_sam=sam,
        _objective=KL(),
        _constraints={},
        _cells=sam.signs * balanced * scale,
    )


def _least_squares_covariance(
    sam: SymbolicSAM,
    objective: Objective,
    constraints: dict[str, Constraint],
    cells: np.ndarray,
) -> pd.DataFrame:
    if not isinstance(objective, LeastSquares):
        raise ValueError(
            f"A covariance is defined only for the LeastSquares objective, the negative log density of normal cells. "
            f"This estimate used {type(objective).__name__}"
        )
    measurements = [
        constraint for constraint in constraints.values() if constraint.penalty is not None
    ]
    if not all(isinstance(constraint.penalty, NormalPenalty) for constraint in measurements):
        raise ValueError("A covariance is defined only when every measurement error is normal")

    bound = BOUND_TOLERANCE * sam._total_magnitude
    if (np.abs(cells) <= bound).any():
        raise ValueError(
            "A cell sits at zero, where its sign bound binds and the cells are no longer normal around the estimate"
        )

    linear = linear_constraints(sam, _hard_constraints(sam, constraints))
    if (linear.inequality_values - linear.inequality_operator @ cells <= bound).any():
        raise ValueError(
            "An inequality binds, so the cells are no longer normal around the estimate"
        )

    sigma = (
        np.abs(sam.prior_values)
        if objective.sigma is None
        else _cell_values(sam, objective.sigma, "sigma")
    )
    precision = np.diag(1.0 / sigma**2)
    for measurement in measurements:
        operator, _, error_scales = _measurement_rows(sam, measurement)
        loadings = operator.toarray() / error_scales[:, None]
        precision += loadings.T @ loadings

    free_directions = null_space(linear.equality_operator.toarray())
    covariance = free_directions @ np.linalg.solve(
        free_directions.T @ precision @ free_directions, free_directions.T
    )

    labels = pd.MultiIndex.from_arrays(
        [list(sam.prior.index[sam.rows]), list(sam.prior.columns[sam.columns])],
        names=list(sam.dims),
    )
    return pd.DataFrame(covariance, index=labels, columns=labels)


def _cell_values(sam: SymbolicSAM, data: CellData, name: str) -> np.ndarray:
    """Return ``data`` at the prior's nonzero cells. A DataFrame or DataArray aligns by label, and an array must match
    the prior's shape and order."""
    labeled = data.transpose(*sam.dims).to_pandas() if isinstance(data, xr.DataArray) else data

    if isinstance(labeled, Real):
        values = np.full(sam.prior.shape, float(labeled))
    elif isinstance(labeled, pd.DataFrame):
        values = labeled.reindex(index=sam.prior.index, columns=sam.prior.columns).to_numpy(
            dtype=float
        )
    else:
        values = np.asarray(labeled, dtype=float)

    if values.shape != sam.prior.shape:
        raise ValueError(
            f"{name} has shape {values.shape}, but the prior has shape {sam.prior.shape}"
        )

    cell_values = values[sam.rows, sam.columns]
    if not (np.isfinite(cell_values).all() and (cell_values > 0.0).all()):
        raise ValueError(f"{name} must be positive and finite on every nonzero cell of the prior")
    return cell_values


def _scalar(value, operation: str) -> float:
    if not isinstance(value, Real):
        raise TypeError(
            f"SAM expressions can only {operation} by numbers, so that constraints stay linear"
        )
    return float(value)


def _label_positions(index: pd.Index, labels) -> int | np.ndarray:
    if isinstance(labels, list):
        return np.concatenate([np.atleast_1d(_label_positions(index, label)) for label in labels])

    # SAM accounts are listed in a deliberate order, so an unsorted MultiIndex is the normal case.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        location = index.get_loc(labels)

    positions = np.arange(len(index))[location]
    return int(positions) if np.ndim(positions) == 0 else positions


def _hard_constraints(
    sam: SymbolicSAM, constraints: dict[str, Constraint]
) -> dict[str, Constraint]:
    """Return the accounting identity and the constraints that hold exactly."""
    row_dim, column_dim = sam.dims
    balance = sam.sum(column_dim) == sam.sum(row_dim).rename({column_dim: row_dim})
    exact = {
        name: constraint for name, constraint in constraints.items() if constraint.penalty is None
    }
    return {ACCOUNTING_IDENTITY: balance, **exact}


def _linear_rows(
    sam: SymbolicSAM, constraints: dict[str, Constraint], is_equality: bool
) -> tuple[csr_array, np.ndarray]:
    if not constraints:
        return csr_array((0, sam.n_cells)), np.zeros(0)

    residuals = [constraint.residual for constraint in constraints.values()]
    constant, operator, imposed = _affine_parts(sam, residuals)
    values = -constant[imposed]

    # A row with no cells is 0 == value or 0 <= value, which holds or fails regardless of the cells.
    empty = np.diff(operator.indptr) == 0
    unsatisfiable = empty & (values != 0.0 if is_equality else values < 0.0)
    if unsatisfiable.any():
        positions_per_constraint = [_n_positions(residual) for residual in residuals]
        owners = np.repeat(np.arange(len(constraints)), positions_per_constraint)[imposed]
        name = list(constraints)[owners[unsatisfiable][0]]
        raise ValueError(
            f"Constraint {name!r} involves no nonzero cell of the prior at a position where it cannot hold. Cells "
            f"that are zero in the prior stay zero"
        )

    return operator, values


def _measurement_rows(
    sam: SymbolicSAM, constraint: Constraint
) -> tuple[csr_array, np.ndarray, np.ndarray]:
    """Return the rows, values and error scales of a measurement with error, without its NaN positions."""
    residual = constraint.residual
    constant, operator, imposed = _affine_parts(sam, [residual])

    # Adding the scale to a zero multiple of the residual aligns it with the residual's positions by label.
    error_scales = _constant_part(sam, [residual * 0.0 + constraint.penalty.scale])[imposed]
    if not (np.isfinite(error_scales).all() and (error_scales > 0.0).all()):
        raise ValueError(
            "The scale of a measurement error must be positive and finite at every position the measurement imposes"
        )

    return operator, -constant[imposed], error_scales


def _affine_parts(
    sam: SymbolicSAM, residuals: list[SAMExpression]
) -> tuple[np.ndarray, csr_array, np.ndarray]:
    """
    Return the constant part of the stacked ``residuals``, which are affine in the cells, the Jacobian at the positions
    where that constant is not NaN, and the indices of those positions.
    """
    stacked = _stacked(residuals)
    constant = _evaluate(sam, stacked)
    imposed = np.flatnonzero(~np.isnan(constant))

    jacobian = pytensor.gradient.jacobian(stacked[imposed], sam.cells, vectorize=True)
    return constant, csr_array(_evaluate(sam, jacobian)), imposed


def _constant_part(sam: SymbolicSAM, residuals: list[SAMExpression]) -> np.ndarray:
    return _evaluate(sam, _stacked(residuals))


def _stacked(residuals: list[SAMExpression]) -> pt.TensorVariable:
    return pt.concatenate(
        [
            pt.as_tensor_variable(
                rewrite_graph(residual.tensor.values.ravel(), include=("lower_xtensor",))
            )
            for residual in residuals
        ]
    )


def _evaluate(sam: SymbolicSAM, output: pt.TensorVariable) -> np.ndarray:
    """Evaluate ``output`` with every cell at zero."""
    evaluate = pytensor.function([sam.cells], output, on_unused_input="ignore")
    return evaluate(np.zeros(sam.n_cells))


def _n_positions(residual: SAMExpression) -> int:
    return int(np.prod([len(residual.labels[dim]) for dim in residual.dims]))
