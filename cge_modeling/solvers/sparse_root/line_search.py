from dataclasses import dataclass, field

import numpy as np

from cge_modeling.solvers.sparse_root.base import (
    RootFunction,
    SolverState,
    StepInfo,
    initial_state,
    residual_summary,
)
from cge_modeling.solvers.sparse_root.direction import (
    ChordDirection,
    DirectionStrategy,
    KrylovDirection,
    NewtonDirection,
)
from cge_modeling.solvers.sparse_root.globalization import (
    ArmijoBacktracking,
    GlobalizationStrategy,
    NonmonotoneBacktracking,
)


@dataclass
class LineSearchSolver:
    """Line-search solver composing a direction strategy with a globalization strategy.

    The named solvers :func:`newton_armijo`, :func:`chord`, :func:`inexact_newton_krylov` and
    :func:`newton_nonmonotone` build instances of this class with different default strategies.

    Parameters
    ----------
    direction : DirectionStrategy, optional
        Strategy for computing the search direction, implementing ``compute(x, res, jac)``. Defaults to
        :class:`~cge_modeling.solvers.sparse_root.direction.NewtonDirection`.
    globalization : GlobalizationStrategy, optional
        Line search strategy, implementing ``search(fun, x, phi, proposal, args)``. Defaults to
        :class:`~cge_modeling.solvers.sparse_root.globalization.ArmijoBacktracking`.

    Examples
    --------
    Compose a solver from a direction and a line search strategy, here a Newton direction with a nonmonotone search
    that remembers five merit values:

    .. code-block:: python

        import numpy as np
        import scipy.sparse as sp

        from cge_modeling.solvers.sparse_root import (
            LineSearchSolver,
            NewtonDirection,
            NonmonotoneBacktracking,
            sparse_root,
        )


        def fun(x):
            return x**2 - np.array([1.0, 4.0]), sp.diags(2 * x, format="csc")


        solver = LineSearchSolver(direction=NewtonDirection(),
                                  globalization=NonmonotoneBacktracking(memory=5))
        result = sparse_root(fun, np.array([2.0, 3.0]), solver=solver, progressbar=False)
        print(result.x)
    """

    direction: DirectionStrategy = field(default_factory=NewtonDirection)
    globalization: GlobalizationStrategy = field(default_factory=ArmijoBacktracking)

    def init(self, fun: RootFunction, x0: np.ndarray, args: tuple) -> SolverState:
        """Evaluate ``fun`` at ``x0``, reset the direction and globalization strategies, and build the initial state.

        Parameters
        ----------
        fun : callable
            Fused residual and Jacobian function, called as ``fun(x, *args)``.
        x0 : ndarray
            Initial guess for the root.
        args : tuple
            Extra positional arguments passed to ``fun``.

        Returns
        -------
        state : ~cge_modeling.solvers.sparse_root.base.SolverState
            State holding the initial point, residuals, Jacobian, merit value and evaluation counts.
        """
        self.direction.reset()
        self.globalization.reset()
        return initial_state(fun, x0, args)

    def step(
        self, fun: RootFunction, state: SolverState, args: tuple
    ) -> tuple[SolverState, StepInfo]:
        """Take one line-search iteration from ``state``.

        Parameters
        ----------
        fun : callable
            Fused residual and Jacobian function, called as ``fun(x, *args)``.
        state : ~cge_modeling.solvers.sparse_root.base.SolverState
            Current solver state.
        args : tuple
            Extra positional arguments passed to ``fun``.

        Returns
        -------
        new_state : ~cge_modeling.solvers.sparse_root.base.SolverState
            State after the accepted step, or the unchanged input state when the line search fails.
        info : ~cge_modeling.solvers.sparse_root.base.StepInfo
            Whether the step was accepted, the step taken, and a failure message when it was not.
        """
        proposal = self.direction.compute(state.x, state.res, state.jac)
        if proposal.slope == 0.0:
            message = (
                f"fatal: the merit function gradient vanished at a non-root "
                f"({residual_summary(state)}), so no direction reduces it."
            )
            return state, StepInfo(accepted=False, step=np.zeros_like(state.x), message=message)

        try:
            accepted = self.globalization.search(fun, state.x, state.phi, proposal, args)
        except RuntimeError as error:
            return state, StepInfo(accepted=False, step=np.zeros_like(state.x), message=str(error))

        njev = 1 if self.globalization.merit_fun is not None else accepted.n_evals
        new_state = SolverState(
            x=accepted.x_new,
            res=accepted.res_new,
            jac=accepted.jac_new,
            phi=accepted.phi_new,
            stats=state.stats.update(nit=1, nfev=accepted.n_evals, njev=njev, nsolve=1),
        )
        return new_state, StepInfo(accepted=True, step=accepted.alpha * proposal.direction)


def newton_armijo(
    direction: DirectionStrategy | None = None,
    globalization: GlobalizationStrategy | None = None,
) -> LineSearchSolver:
    """Build a Newton solver with Armijo backtracking line search.

    Parameters
    ----------
    direction : DirectionStrategy, optional
        Search direction strategy. Defaults to
        :class:`~cge_modeling.solvers.sparse_root.direction.NewtonDirection`.
    globalization : GlobalizationStrategy, optional
        Line search strategy. Defaults to
        :class:`~cge_modeling.solvers.sparse_root.globalization.ArmijoBacktracking`.

    Returns
    -------
    solver : LineSearchSolver
        Solver composing the two strategies.

    Examples
    --------
    Pass the Newton solver to ``sparse_root`` with a gentler backtracking factor than the default:

    .. code-block:: python

        import numpy as np
        import scipy.sparse as sp

        from cge_modeling.solvers.sparse_root import ArmijoBacktracking, newton_armijo, sparse_root


        def fun(x):
            return x**2 - np.array([1.0, 4.0]), sp.diags(2 * x, format="csc")


        solver = newton_armijo(globalization=ArmijoBacktracking(beta=0.8))
        result = sparse_root(fun, np.array([2.0, 3.0]), solver=solver, progressbar=False)
        print(result.x)
    """
    return LineSearchSolver(
        direction=direction if direction is not None else NewtonDirection(),
        globalization=globalization if globalization is not None else ArmijoBacktracking(),
    )


def chord(
    direction: ChordDirection | None = None,
    globalization: GlobalizationStrategy | None = None,
) -> LineSearchSolver:
    """Build a chord solver, which reuses a cached Jacobian, with Armijo backtracking line search.

    Parameters
    ----------
    direction : ChordDirection, optional
        Search direction strategy. Defaults to
        :class:`~cge_modeling.solvers.sparse_root.direction.ChordDirection`.
    globalization : GlobalizationStrategy, optional
        Line search strategy. Defaults to
        :class:`~cge_modeling.solvers.sparse_root.globalization.ArmijoBacktracking`.

    Returns
    -------
    solver : LineSearchSolver
        Solver composing the two strategies.

    Examples
    --------
    Reuse one Jacobian factorization for several steps, which pays off when factorizing costs far more than evaluating
    the residuals:

    .. code-block:: python

        import numpy as np
        import scipy.sparse as sp

        from cge_modeling.solvers.sparse_root import chord, sparse_root


        def fun(x):
            return x**2 - np.array([1.0, 4.0]), sp.diags(2 * x, format="csc")


        solver = chord()
        result = sparse_root(fun, np.array([2.0, 3.0]), solver=solver, progressbar=False)
        print(result.x)
    """
    return LineSearchSolver(
        direction=direction if direction is not None else ChordDirection(),
        globalization=globalization if globalization is not None else ArmijoBacktracking(),
    )


def inexact_newton_krylov(
    direction: KrylovDirection | None = None,
    globalization: GlobalizationStrategy | None = None,
) -> LineSearchSolver:
    """Build an inexact Newton solver that computes the direction with a Krylov method.

    Parameters
    ----------
    direction : KrylovDirection, optional
        Search direction strategy. Defaults to
        :class:`~cge_modeling.solvers.sparse_root.direction.KrylovDirection`, which uses GMRES.
    globalization : GlobalizationStrategy, optional
        Line search strategy. Defaults to
        :class:`~cge_modeling.solvers.sparse_root.globalization.ArmijoBacktracking`.

    Returns
    -------
    solver : LineSearchSolver
        Solver composing the two strategies.

    Examples
    --------
    Solve each Newton system with GMRES instead of a direct factorization:

    .. code-block:: python

        import numpy as np
        import scipy.sparse as sp

        from cge_modeling.solvers.sparse_root import inexact_newton_krylov, sparse_root


        def fun(x):
            return x**2 - np.array([1.0, 4.0]), sp.diags(2 * x, format="csc")


        solver = inexact_newton_krylov()
        result = sparse_root(fun, np.array([2.0, 3.0]), solver=solver, progressbar=False)
        print(result.x)
    """
    return LineSearchSolver(
        direction=direction if direction is not None else KrylovDirection(),
        globalization=globalization if globalization is not None else ArmijoBacktracking(),
    )


def newton_nonmonotone(
    direction: DirectionStrategy | None = None,
    globalization: NonmonotoneBacktracking | None = None,
) -> LineSearchSolver:
    """Build a Newton solver with a nonmonotone (Grippo-Lampariello-Lucidi) line search.

    Parameters
    ----------
    direction : DirectionStrategy, optional
        Search direction strategy. Defaults to
        :class:`~cge_modeling.solvers.sparse_root.direction.NewtonDirection`.
    globalization : NonmonotoneBacktracking, optional
        Line search strategy. Defaults to
        :class:`~cge_modeling.solvers.sparse_root.globalization.NonmonotoneBacktracking`.

    Returns
    -------
    solver : LineSearchSolver
        Solver composing the two strategies.

    Examples
    --------
    Let the merit function rise now and then, which helps when a monotone line search stalls in a narrow valley:

    .. code-block:: python

        import numpy as np
        import scipy.sparse as sp

        from cge_modeling.solvers.sparse_root import newton_nonmonotone, sparse_root


        def fun(x):
            return x**2 - np.array([1.0, 4.0]), sp.diags(2 * x, format="csc")


        solver = newton_nonmonotone()
        result = sparse_root(fun, np.array([2.0, 3.0]), solver=solver, progressbar=False)
        print(result.x)
    """
    return LineSearchSolver(
        direction=direction if direction is not None else NewtonDirection(),
        globalization=globalization if globalization is not None else NonmonotoneBacktracking(),
    )
