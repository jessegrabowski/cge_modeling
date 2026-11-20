# Vendored from gEconpy (jessegrabowski/gEconpy, commit 8f52053, gEconpy/solvers/sparse_root). Sparse root finding for
# systems whose Jacobian is a scipy sparse matrix, used when a model is compiled with ``use_sparse_matrices=True``.

from cge_modeling.solvers.sparse_root.direction import (
    ChordDirection,
    KrylovDirection,
    NewtonDirection,
)
from cge_modeling.solvers.sparse_root.dogleg import SparseDogleg
from cge_modeling.solvers.sparse_root.gauss_newton import GaussNewtonTrustRegion
from cge_modeling.solvers.sparse_root.globalization import (
    ArmijoBacktracking,
    NonmonotoneBacktracking,
)
from cge_modeling.solvers.sparse_root.levenberg_marquardt import LevenbergMarquardt
from cge_modeling.solvers.sparse_root.line_search import (
    LineSearchSolver,
    chord,
    inexact_newton_krylov,
    newton_armijo,
    newton_nonmonotone,
)
from cge_modeling.solvers.sparse_root.sparse_root import sparse_root

__all__ = [
    "ArmijoBacktracking",
    "ChordDirection",
    "GaussNewtonTrustRegion",
    "KrylovDirection",
    "LevenbergMarquardt",
    "LineSearchSolver",
    "NewtonDirection",
    "NonmonotoneBacktracking",
    "SparseDogleg",
    "chord",
    "inexact_newton_krylov",
    "newton_armijo",
    "newton_nonmonotone",
    "sparse_root",
]
