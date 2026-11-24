import warnings

from collections.abc import Hashable
from dataclasses import dataclass, field, replace
from functools import cached_property, partial
from numbers import Real

import cvxpy as cp
import narwhals as nw
import numpy as np
import pandas as pd
import xarray as xr

from narwhals.typing import IntoDataFrame
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.lapack import dpstrf
from scipy.sparse import coo_array, csr_array, diags_array, vstack
from scipy.sparse.csgraph import connected_components
from scipy.special import kl_div

from cge_modeling.solvers.sparse_root import newton_armijo, sparse_root

ROW = "row"
COLUMN = "col"

DUAL_SOLVER = "dual"
ACCOUNTING_IDENTITY = "accounting identity"
DUAL_TOLERANCE = 1e-12

# A cell within this fraction of its prior magnitude, or an inequality slack within this fraction of the absolute
# terms of its inequality, counts as at its bound.
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

    Two expressions combine by dim name in the same way. One over a subset of the other's dims repeats over the dims it
    lacks, so ``sam.sel() == sam.sum("col")`` makes every cell equal its row account's total.

    Parameters
    ----------
    sam : SymbolicSAM
        The SAM whose cells the expression is linear in.
    labels : dict of str to pandas.Index
        The account labels of each dim, in dim order.
    operator : scipy.sparse.csr_array
        Coefficients on the cells, one row per position of the grid of ``labels`` in row-major order.
    constant : numpy.ndarray
        The constant part, shaped like the grid of ``labels``.
    """

    # Make numpy and pandas return NotImplemented from their operators, so that ``series <= expr`` and
    # ``array + expr`` reach this class's reflected methods.
    __array_ufunc__ = None
    __pandas_priority__ = 5000

    def __init__(
        self,
        sam: "SymbolicSAM",
        labels: dict[str, pd.Index],
        operator: csr_array,
        constant: np.ndarray,
    ):
        self.sam = sam
        self.labels = labels
        self.operator = operator
        self.constant = constant

        grid_size = int(np.prod(self.shape))
        if constant.shape != self.shape or operator.shape != (grid_size, sam.n_cells):
            raise ValueError(
                f"An expression over shape {self.shape} needs a constant of that shape and an operator of shape "
                f"{(grid_size, sam.n_cells)}, not {constant.shape} and {operator.shape}"
            )

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(self.labels)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(index) for index in self.labels.values())

    def sum(self, dim: str | None = None) -> "SAMExpression":
        """Sum over ``dim``, or over every dim when ``dim`` is None."""
        if dim is not None and dim not in self.dims:
            raise ValueError(f"Cannot sum over {dim!r}. The expression's dims are {self.dims}")
        if not self.dims:
            return self
        summed_dims = self.dims if dim is None else (dim,)
        kept_dims = [name for name in self.dims if name not in summed_dims]
        labels = {name: self.labels[name] for name in kept_dims}

        # Each position of the grid adds into the position of the summed grid that shares its other coordinates.
        coordinates = np.indices(self.shape).reshape(len(self.shape), -1)
        summed_shape = tuple(len(index) for index in labels.values())
        summed_position = _flat_positions(
            [coordinates[self.dims.index(name)] for name in kept_dims],
            summed_shape,
            count=self.constant.size,
        )
        grouping = csr_array(
            (np.ones(summed_position.size), (summed_position, np.arange(summed_position.size))),
            shape=(int(np.prod(summed_shape)), summed_position.size),
        )
        constant = self.constant.sum(axis=tuple(self.dims.index(name) for name in summed_dims))
        return SAMExpression(self.sam, labels, csr_array(grouping @ self.operator), constant)

    def rename(self, mapping: dict[str, str]) -> "SAMExpression":
        """Rename dims, for example ``{"col": "row"}`` to compare column sums with row sums account by account."""
        unknown_dims = set(mapping) - set(self.dims)
        if unknown_dims:
            raise ValueError(
                f"Cannot rename {sorted(unknown_dims)}. The expression's dims are {self.dims}"
            )
        renamed_dims = [mapping.get(name, name) for name in self.dims]
        if len(set(renamed_dims)) != len(renamed_dims):
            raise ValueError(
                f"Renaming dims {self.dims} with {mapping} gives repeated dims {tuple(renamed_dims)}"
            )
        labels = dict(zip(renamed_dims, self.labels.values(), strict=True))
        return SAMExpression(self.sam, labels, self.operator, self.constant)

    def __add__(self, other) -> "SAMExpression":
        return _add(*_aligned(self, self._operand(other)))

    def __radd__(self, other) -> "SAMExpression":
        return self + other

    def __sub__(self, other) -> "SAMExpression":
        return _add(*_aligned(self, -self._operand(other)))

    def __rsub__(self, other) -> "SAMExpression":
        return -self + other

    def __neg__(self) -> "SAMExpression":
        return self * -1.0

    def __mul__(self, other) -> "SAMExpression":
        factor = _scalar(other, "multiply")
        return SAMExpression(self.sam, self.labels, self.operator * factor, self.constant * factor)

    def __rmul__(self, other) -> "SAMExpression":
        return self * other

    def __truediv__(self, other) -> "SAMExpression":
        return self * (1.0 / _scalar(other, "divide"))

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

    def _operand(self, other) -> "SAMExpression":
        """Return ``other`` as an expression, with data aligned to this expression's labels."""
        if isinstance(other, SAMExpression):
            if other.sam is not self.sam:
                raise ValueError("Expressions from different SymbolicSAMs cannot be combined")
            return other

        if isinstance(other, Real):
            return _constant_expression(self.sam, {}, np.asarray(float(other)))

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
                return self._operand(float(other))
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
    ) -> "SAMExpression":
        """Return ``values`` as a constant expression aligned to this expression's labels, with the index on
        ``dims[0]`` and any columns on ``dims[1]``. Labels it does not give are NaN."""
        axes = [values.index] if isinstance(values, pd.Series) else [values.index, values.columns]
        for dim, labels in zip(dims, axes, strict=True):
            unknown_labels = labels.difference(self.labels[dim])
            if len(unknown_labels) > 0:
                raise ValueError(f"Labels not in dim {dim!r}: {list(unknown_labels)}")

        if isinstance(values, pd.Series):
            aligned = values.reindex(self.labels[dims[0]])
        else:
            aligned = values.reindex(index=self.labels[dims[0]], columns=self.labels[dims[1]])
        labels = {dim: self.labels[dim] for dim in dims}
        return _constant_expression(self.sam, labels, aligned.to_numpy(dtype=float))


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
        The objective value at the estimate, including the penalties of measurements with error.
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

    def covariance(self, cells: list[Cell] | None = None) -> pd.DataFrame:
        """
        Return the covariance of estimated cells under a :class:`LeastSquares` objective.

        The objective makes the cells independent and normal around the prior with standard deviation ``sigma``.
        Measurements made with :meth:`Constraint.normal` are normal observations of linear expressions, and every other
        equality, including the accounting identity, holds exactly. The covariance is that of the cells given all of
        them, as the constraints stood when :meth:`SymbolicSAM.fit` ran. It ignores the sign bounds and inequalities,
        which truncate that normal distribution, so it is the covariance of the estimate only when every bound is many
        standard deviations from it.

        The cost grows with the square of the number of requested cells, so a large SAM asks for the cells it needs,
        or for every cell's variance alone through :meth:`variance`.

        Parameters
        ----------
        cells : list of tuple, optional
            The cells to return, each as a pair of row and column account labels. Default is None, which returns every
            nonzero cell of the prior.

        Returns
        -------
        covariance : pandas.DataFrame
            Covariance of the requested cells, in the prior's units squared, indexed on both axes by the row and column
            account of each cell.
        """
        selected = _cell_indices(self._symbolic_sam, cells)
        labels = _cell_labels(self._symbolic_sam, selected)
        return pd.DataFrame(self._conditioned.covariance(selected), index=labels, columns=labels)

    def variance(self) -> pd.Series:
        """
        Return the variance of every estimated cell under a :class:`LeastSquares` objective.

        The variances are the diagonal of :meth:`covariance` without forming the matrix. They cost one dense Cholesky
        factorization and inverse of a square matrix with a row for each independent exact constraint and measurement
        row, about one per account, however many cells the SAM has.

        Returns
        -------
        variance : pandas.Series
            Variance of each nonzero cell of the prior, in the prior's units squared, indexed by the row and column
            account of each cell.
        """
        every_cell = np.arange(self._symbolic_sam.n_cells)
        return pd.Series(
            self._conditioned.variance(),
            index=_cell_labels(self._symbolic_sam, every_cell),
            name="variance",
        )

    @cached_property
    def _conditioned(self) -> "_ConditionedNormal":
        return _least_squares_conditioning(
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

    Every objective is the negative log density of a prior on the cells, up to a constant, so it adds to the penalties
    of measurements with error.
    """

    def expression(
        self, sam: "SymbolicSAM", magnitudes: cp.Expression, scale: float
    ) -> cp.Expression:
        """Return the objective divided by :meth:`unit`, for cell magnitudes divided by ``scale``."""
        raise NotImplementedError

    def unit(self, scale: float) -> float:
        """Return the factor that converts :meth:`expression` to the objective."""
        raise NotImplementedError


class Divergence(Objective):
    r"""
    An objective that grows in proportion to the cells, divided by a dispersion :math:`\phi` in the prior's units.

    To second order the divergence is :math:`\frac{1}{2} \sum_k (x_k - z_k)^2 / |z_k|`, so dividing it by
    :math:`\phi` makes it the negative log density of cells with variance :math:`\phi |z_k|`, which rescales with the
    prior's units. Without measurements with error, :math:`\phi` scales the whole objective and leaves the estimate
    unchanged.
    """

    dispersion: float | None

    def __post_init__(self):
        if self.dispersion is not None and not (
            np.isfinite(self.dispersion) and self.dispersion > 0.0
        ):
            raise ValueError(f"The dispersion must be positive and finite, not {self.dispersion}")

    def unit(self, scale: float) -> float:
        return scale / (1.0 if self.dispersion is None else self.dispersion)


@dataclass(frozen=True, eq=False)
class KL(Divergence):
    r"""
    Weighted generalized KL divergence of the cell magnitudes from the prior magnitudes, divided by the dispersion,

    .. math::

        \frac{1}{\phi} \sum_k w_k \left( |x_k| \log \frac{|x_k|}{|z_k|} - |x_k| + |z_k| \right).

    Parameters
    ----------
    weights : float, numpy.ndarray, pandas.DataFrame or xarray.DataArray, optional
        Positive weight of each cell, as a number or laid out like the prior. A larger weight holds a cell closer to
        its prior value. Default is None, which weights every cell by one.
    dispersion : float, optional
        Positive :math:`\phi`, in the prior's units, which sets how the objective trades off against measurements with
        error. Default is None, which uses 1.0 and warns when a measurement with error is present.
    """

    weights: CellData | None = None
    dispersion: float | None = None

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
class ReverseKL(Divergence):
    r"""
    Generalized KL divergence of the prior magnitudes from the cell magnitudes, divided by the dispersion,

    .. math::

        \frac{1}{\phi} \sum_k |z_k| \log \frac{|z_k|}{|x_k|} - |z_k| + |x_k|.

    Parameters
    ----------
    dispersion : float, optional
        Positive :math:`\phi`, in the prior's units, which sets how the objective trades off against measurements with
        error. Default is None, which uses 1.0 and warns when a measurement with error is present.
    """

    dispersion: float | None = None

    def expression(
        self, sam: "SymbolicSAM", magnitudes: cp.Expression, scale: float
    ) -> cp.Expression:
        return cp.sum(cp.kl_div(np.abs(sam.prior_values) / scale, magnitudes))


@dataclass(frozen=True, eq=False)
class Pearson(Divergence):
    r"""
    Pearson chi-squared distance of the cells from the prior, divided by the dispersion, which equals :class:`KL` to
    second order,

    .. math::

        \frac{1}{2 \phi} \sum_k \frac{(x_k - z_k)^2}{|z_k|}.

    Parameters
    ----------
    dispersion : float, optional
        Positive :math:`\phi`, in the prior's units, which sets how the objective trades off against measurements with
        error. Default is None, which uses 1.0 and warns when a measurement with error is present.
    """

    dispersion: float | None = None

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

        # The grid position of each cell carries a coefficient of one on that cell.
        cell_positions = self.rows * self.n_accounts + self.columns
        self._matrix = SAMExpression(
            self,
            {row_dim: prior_frame.index, column_dim: prior_frame.columns},
            operator=csr_array(
                (np.ones(self.n_cells), (cell_positions, np.arange(self.n_cells))),
                shape=(self.n_accounts**2, self.n_cells),
            ),
            constant=np.zeros(prior_frame.shape),
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

        matrix = self._matrix
        positions = [
            _label_positions(matrix.labels[dim], selections[dim])
            if selections.get(dim) is not None
            else np.arange(len(matrix.labels[dim]))
            for dim in matrix.dims
        ]
        grid = np.ix_(*[np.atleast_1d(dim_positions) for dim_positions in positions])
        rows = np.arange(np.prod(matrix.shape)).reshape(matrix.shape)[grid]

        # A single account drops its dim, as in xarray.
        kept = [np.ndim(dim_positions) > 0 for dim_positions in positions]
        kept_shape = tuple(length for length, keep in zip(rows.shape, kept, strict=True) if keep)
        labels = {
            dim: matrix.labels[dim][dim_positions]
            for dim, dim_positions, keep in zip(matrix.dims, positions, kept, strict=True)
            if keep
        }
        return SAMExpression(
            self,
            labels,
            operator=csr_array(matrix.operator[rows.ravel()]),
            constant=matrix.constant[grid].reshape(kept_shape),
        )

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
        account scales :math:`d` with :func:`~cge_modeling.solvers.sparse_root.sparse_root`, and every other solver
        solves the full problem in cvxpy.

        Parameters
        ----------
        objective : Objective, optional
            The distance from the prior to minimize: :class:`KL`, :class:`LeastSquares`, :class:`ReverseKL` or
            :class:`Pearson`. Default is None, which uses the unweighted :class:`KL`.
        solver : str, optional
            ``"dual"``, for the unweighted :class:`KL` objective on a SAM with no constraints, or the name of a cvxpy
            solver. Default is None, which picks ``"dual"`` where it applies and CLARABEL otherwise.
        **solver_kwargs
            Passed to :func:`~cge_modeling.solvers.sparse_root.sparse_root` for the ``"dual"`` solver, which uses
            :func:`~cge_modeling.solvers.sparse_root.newton_armijo` unless a ``solver`` is given, and to
            ``cvxpy.Problem.solve`` otherwise. CLARABEL's tolerances apply after every cell is divided by
            the total of the absolute cell values, so its error in each cell is roughly absolute. Tighter
            ``tol_gap_abs``, ``tol_gap_rel`` and ``tol_feas`` reduce it.

        Returns
        -------
        estimate : SAMEstimate
            The estimated SAM, the objective value, and the solver status.
        """
        forced_to_zero = cells_forced_to_zero(self)
        if forced_to_zero:
            raise ValueError(
                f"Balancing sets these cells to zero, because they lie on no cycle of the account graph: "
                f"{forced_to_zero}. Remove them from the prior or add the missing flows."
            )

        if objective is None:
            objective = KL()
        _warn_if_unit_dependent(objective, self.constraints)

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
            return _fit_dual(self, objective, **solver_kwargs)

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
        Return the objective at ``candidate``, including the penalties of measurements with error.

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
        """
        if objective is None:
            objective = KL()
        _warn_if_unit_dependent(objective, self.constraints)

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


def _fit_dual(sam: SymbolicSAM, objective: Objective, **sparse_root_kwargs) -> SAMEstimate:
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

    # The fixed account's imbalance is minus the sum of the others', so fixing the account with the largest turnover in
    # each group keeps that sum small relative to its turnover.
    prior_turnover = abs(incidence).T @ weights
    by_group_then_turnover = np.lexsort((-prior_turnover, group))
    _, first_in_order = np.unique(group[by_group_then_turnover], return_index=True)
    free = prior_turnover > 0
    free[by_group_then_turnover[first_in_order]] = False
    free_accounts = np.flatnonzero(free)

    def log_scales(free_log_scales):
        scales = np.zeros(sam.n_accounts)
        scales[free_accounts] = free_log_scales
        return scales

    def magnitudes(free_log_scales):
        return weights * np.exp(incidence @ log_scales(free_log_scales))

    # Dividing each free account's spending minus income by its prior turnover makes the residual tolerance relative.
    row_scales = diags_array(1.0 / prior_turnover[free_accounts])

    def gradient_and_laplacian(free_log_scales):
        current = magnitudes(free_log_scales)
        laplacian = (incidence.T @ incidence.multiply(current[:, None])).tocsr()
        gradient = (incidence.T @ current)[free_accounts]
        jacobian = row_scales @ laplacian[free_accounts][:, free_accounts]
        return row_scales @ gradient, jacobian.tocsc()

    # A caller's tol sets both tolerances in sparse_root, so the defaults give way to it.
    defaults = {"solver": newton_armijo(), "progressbar": False}
    if "tol" not in sparse_root_kwargs:
        defaults |= {"f_tol": DUAL_TOLERANCE / 10.0, "x_tol": np.finfo(float).eps}

    if free_accounts.size == 0:
        free_log_scales = np.zeros(0)
        solver_message = "No account has a log scale to solve for"
    else:
        result = sparse_root(
            gradient_and_laplacian,
            np.zeros(free_accounts.size),
            **{**defaults, **sparse_root_kwargs},
        )
        # sparse_root returns a failed result in place of an exception raised during the solve, naming it in the message.
        if result.message.startswith("Optimizer raised"):
            raise RuntimeError(f"The dual solver failed. {result.message}")
        free_log_scales = result.x
        solver_message = result.message

    balanced = magnitudes(free_log_scales)
    imbalance = np.abs(incidence.T @ balanced)
    turnover = abs(incidence).T @ balanced
    relative_imbalance = np.divide(
        imbalance, turnover, out=np.zeros_like(imbalance), where=turnover > 0
    )
    status = cp.OPTIMAL if relative_imbalance.max() <= DUAL_TOLERANCE else cp.OPTIMAL_INACCURATE

    if status != cp.OPTIMAL:
        warnings.warn(
            f"The SAM estimate did not solve to optimality: solver status is {status!r}. The dual solver reported: "
            f"{solver_message}",
            stacklevel=3,
        )

    return SAMEstimate(
        sam=sam.like_prior(sam.signs * balanced * scale),
        objective=float(kl_div(balanced, weights).sum()) * objective.unit(scale),
        status=status,
        _symbolic_sam=sam,
        _objective=objective,
        _constraints={},
        _cells=sam.signs * balanced * scale,
    )


def _warn_if_unit_dependent(objective: Objective, constraints: dict[str, Constraint]) -> None:
    """Warn when a measurement with error makes the estimate depend on a dispersion left at its default."""
    if not isinstance(objective, Divergence) or objective.dispersion is not None:
        return
    measurements = [
        name for name, constraint in constraints.items() if constraint.penalty is not None
    ]
    if measurements:
        warnings.warn(
            f"{type(objective).__name__} is in the prior's units, while the measurements with error {measurements} "
            f"are not, so the estimate depends on the units of the prior. Pass dispersion, in the prior's units, to "
            f"fix the trade-off. Using dispersion=1.0",
            stacklevel=3,
        )


def _least_squares_conditioning(
    sam: SymbolicSAM,
    objective: Objective,
    constraints: dict[str, Constraint],
    estimate_cells: np.ndarray,
) -> "_ConditionedNormal":
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

    if (np.abs(estimate_cells) <= BOUND_TOLERANCE * np.abs(sam.prior_values)).any():
        raise ValueError(
            "A cell sits at zero, where its sign bound binds and the cells are no longer normal around the estimate"
        )

    exact = linear_constraints(sam, _hard_constraints(sam, constraints))
    slack = exact.inequality_values - exact.inequality_operator @ estimate_cells
    inequality_size = abs(exact.inequality_operator) @ np.abs(estimate_cells) + np.abs(
        exact.inequality_values
    )
    if (slack <= BOUND_TOLERANCE * inequality_size).any():
        raise ValueError(
            "An inequality binds, so the cells are no longer normal around the estimate"
        )

    sigma = (
        np.abs(sam.prior_values)
        if objective.sigma is None
        else _cell_values(sam, objective.sigma, "sigma")
    )
    variance = sigma**2
    loadings = csr_array(
        vstack(
            [csr_array((0, sam.n_cells))]
            + [
                diags_array(1.0 / error_scales) @ operator
                for operator, _, error_scales in map(partial(_measurement_rows, sam), measurements)
            ]
        )
    )

    equalities = exact.equality_operator[np.diff(exact.equality_operator.indptr) > 0]
    independent = _independent_rows(equalities, variance)
    return _ConditionedNormal(
        prior_variance=variance,
        rows=csr_array(vstack([independent, loadings])),
        noise=np.r_[np.zeros(independent.shape[0]), np.ones(loadings.shape[0])],
    )


def _independent_rows(rows: csr_array, prior_variance: np.ndarray) -> csr_array:
    """Return ``rows`` without the rows that the others imply, judged on their covariance under the prior."""
    # The identity's rows sum to zero within each group of connected accounts, and exact constraints may repeat what the
    # identity and each other imply. Implied rows of a consistent system add nothing to the conditioning, so a pivoted
    # Cholesky factorization of their unit-diagonal covariance keeps the rows of its numerical rank.
    covariance = (rows @ diags_array(prior_variance) @ rows.T).toarray()
    scale = 1.0 / np.sqrt(np.diag(covariance))
    covariance *= scale[:, None]
    covariance *= scale[None, :]
    _, pivots, rank, _ = dpstrf(covariance, overwrite_a=True)
    return csr_array(rows[np.sort(pivots[:rank] - 1)])


@dataclass(frozen=True, eq=False)
class _ConditionedNormal:
    """
    Normal cells with ``prior_variance``, observed through independent ``rows`` with normal errors of variance
    ``noise``, where a variance of zero makes the observation exact.
    """

    prior_variance: np.ndarray
    rows: csr_array
    noise: np.ndarray

    @cached_property
    def conditioning_factor(self) -> tuple[np.ndarray, bool]:
        # Conditioning x ~ N(z, D) on A x + e with e ~ N(0, N) gives S = D - D A' K^-1 A D with K = A D A' + N, which
        # is positive definite for independent rows. K has about one row per account, and its Cholesky factor fills to
        # most of a dense matrix on SAMs, so it is factored densely.
        conditioning = (self.rows @ diags_array(self.prior_variance) @ self.rows.T).toarray()
        conditioning[np.diag_indices_from(conditioning)] += self.noise
        return cho_factor(conditioning, overwrite_a=True)

    def covariance(self, selected: np.ndarray) -> np.ndarray:
        """Return the covariance of the cells at positions ``selected``."""
        prior = np.diag(self.prior_variance[selected])
        if self.rows.shape[0] == 0:
            return prior

        loadings = (self.rows[:, selected] @ diags_array(self.prior_variance[selected])).toarray()
        return prior - loadings.T @ cho_solve(self.conditioning_factor, loadings)

    def variance(self) -> np.ndarray:
        """Return the variance of every cell, from a dense inverse of K."""
        if self.rows.shape[0] == 0:
            return self.prior_variance.copy()

        inverse = cho_solve(self.conditioning_factor, np.eye(self.rows.shape[0]), overwrite_b=True)
        # Cell i has variance d_i - d_i^2 a_i' K^-1 a_i, which reads K^-1 only at the pairs of rows the cell enters. A
        # cell the exact rows determine has variance zero, which the subtraction leaves a roundoff error to either side.
        explained = self.prior_variance**2 * _quadratic_forms(csr_array(self.rows.T), inverse)
        return np.maximum(self.prior_variance - explained, 0.0)


def _quadratic_forms(rows: csr_array, matrix: np.ndarray) -> np.ndarray:
    """Return r_i' M r_i for each sparse row r_i of ``rows``, reading ``matrix`` only at pairs of a row's nonzeros."""
    counts = np.diff(rows.indptr)
    pair_counts = counts**2
    row_of_pair = np.repeat(np.arange(rows.shape[0]), pair_counts)
    pair_starts = np.cumsum(pair_counts) - pair_counts
    offset = np.arange(pair_counts.sum()) - np.repeat(pair_starts, pair_counts)
    row_counts = counts[row_of_pair]
    row_starts = rows.indptr[row_of_pair]
    first = row_starts + offset // row_counts
    second = row_starts + offset % row_counts
    products = (
        rows.data[first] * rows.data[second] * matrix[rows.indices[first], rows.indices[second]]
    )
    return np.bincount(row_of_pair, weights=products, minlength=rows.shape[0])


def _cell_labels(sam: SymbolicSAM, positions: np.ndarray) -> pd.MultiIndex:
    """Return the row and column account labels of the cells at ``positions``."""
    return pd.MultiIndex.from_arrays(
        [
            list(sam.prior.index[sam.rows[positions]]),
            list(sam.prior.columns[sam.columns[positions]]),
        ],
        names=list(sam.dims),
    )


def _cell_indices(sam: SymbolicSAM, cells: list[Cell] | None) -> np.ndarray:
    """Return the positions in the cell vector of ``cells``, or of every cell when ``cells`` is None."""
    if cells is None:
        return np.arange(sam.n_cells)
    row_positions = sam.prior.index.get_indexer([row for row, _ in cells])
    column_positions = sam.prior.columns.get_indexer([column for _, column in cells])
    unknown = [
        cell
        for cell, row, column in zip(cells, row_positions, column_positions, strict=True)
        if row < 0 or column < 0
    ]
    if unknown:
        raise ValueError(f"Cells with labels not in the prior: {unknown}")

    cell_of_position = {
        (row, column): index
        for index, (row, column) in enumerate(zip(sam.rows, sam.columns, strict=True))
    }
    zero_cells = [
        cell
        for cell, row, column in zip(cells, row_positions, column_positions, strict=True)
        if (row, column) not in cell_of_position
    ]
    if zero_cells:
        raise ValueError(f"Cells that are zero in the prior have no covariance: {zero_cells}")
    return np.array(
        [
            cell_of_position[row, column]
            for row, column in zip(row_positions, column_positions, strict=True)
        ],
        dtype=np.int64,
    )


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
    operators = [csr_array((0, sam.n_cells))]
    values = [np.zeros(0)]
    for name, constraint in constraints.items():
        residual = constraint.residual
        constant = residual.constant.ravel()
        imposed = ~np.isnan(constant)
        operator = csr_array(residual.operator[imposed])
        value = -constant[imposed]

        # A row with no cells is 0 == value or 0 <= value, which holds or fails regardless of the cells.
        empty = np.diff(operator.indptr) == 0
        if _cannot_hold(value[empty], is_equality).any():
            raise ValueError(
                f"Constraint {name!r} involves no nonzero cell of the prior at a position where it cannot hold. Cells "
                f"that are zero in the prior stay zero"
            )

        operators.append(operator)
        values.append(value)

    return csr_array(vstack(operators, format="csr")), np.concatenate(values)


def _cannot_hold(values: np.ndarray, is_equality: bool) -> np.ndarray:
    """Return where ``0 == value``, or ``0 <= value`` for an inequality, fails."""
    return values != 0.0 if is_equality else values < 0.0


def _measurement_rows(
    sam: SymbolicSAM, constraint: Constraint
) -> tuple[csr_array, np.ndarray, np.ndarray]:
    """Return the rows, values and error scales of a measurement with error, without its NaN positions."""
    residual = constraint.residual
    constant = residual.constant.ravel()
    imposed = ~np.isnan(constant)

    # Adding the scale to a zero multiple of the residual aligns it with the residual's positions by label.
    error_scale = residual * 0.0 + constraint.penalty.scale
    error_scales = error_scale.constant.ravel()[imposed]
    if not (np.isfinite(error_scales).all() and (error_scales > 0.0).all()):
        raise ValueError(
            "The scale of a measurement error must be positive and finite at every position the measurement imposes"
        )

    return csr_array(residual.operator[imposed]), -constant[imposed], error_scales


def _flat_positions(
    coordinates: list[np.ndarray], shape: tuple[int, ...], count: int
) -> np.ndarray:
    """Return the row-major position in a grid of ``shape`` of each of ``count`` coordinate tuples."""
    if not shape:
        return np.zeros(count, dtype=np.int64)
    return np.ravel_multi_index(coordinates, shape)


def _constant_expression(
    sam: SymbolicSAM, labels: dict[str, pd.Index], constant: np.ndarray
) -> SAMExpression:
    return SAMExpression(sam, labels, csr_array((constant.size, sam.n_cells)), constant)


def _aligned(left: SAMExpression, right: SAMExpression) -> tuple[SAMExpression, SAMExpression]:
    """Return ``left`` and ``right`` on the same labels, broadcasting the one over fewer dims."""
    for dim in set(left.dims) & set(right.dims):
        if not left.labels[dim].equals(right.labels[dim]):
            raise ValueError(
                f"Expressions combine position by position, but their accounts on dim {dim!r} differ: "
                f"{list(left.labels[dim])} and {list(right.labels[dim])}"
            )

    if set(right.dims) <= set(left.dims):
        return left, _broadcast(right, left.labels)
    if set(left.dims) <= set(right.dims):
        return _broadcast(left, right.labels), right

    row_dim, column_dim = left.sam.dims
    raise ValueError(
        f"Expressions with dims {left.dims} and {right.dims} would broadcast into an outer product. Rename a dim "
        f"first, for example sam.sum({row_dim!r}).rename({{{column_dim!r}: {row_dim!r}}})"
    )


def _broadcast(expression: SAMExpression, labels: dict[str, pd.Index]) -> SAMExpression:
    """Return ``expression`` on ``labels``, whose dims include its own, repeating it over the dims it lacks."""
    if expression.dims == tuple(labels):
        return expression

    # Order the expression's axes as in ``labels``, then give each missing dim an axis of length one to broadcast.
    target_order = [dim for dim in labels if dim in expression.dims]
    axes = [expression.dims.index(dim) for dim in target_order]
    expanded_shape = tuple(
        len(index) if dim in expression.dims else 1 for dim, index in labels.items()
    )
    shape = tuple(len(index) for index in labels.values())

    def expanded(values: np.ndarray) -> np.ndarray:
        return np.broadcast_to(values.transpose(axes).reshape(expanded_shape), shape)

    rows = expanded(np.arange(expression.constant.size).reshape(expression.shape)).ravel()
    return SAMExpression(
        expression.sam,
        labels,
        csr_array(expression.operator[rows]),
        expanded(expression.constant).copy(),
    )


def _add(left: SAMExpression, right: SAMExpression) -> SAMExpression:
    """Return the sum of two expressions on the same labels."""
    return SAMExpression(
        left.sam,
        left.labels,
        csr_array(left.operator + right.operator),
        left.constant + right.constant,
    )
