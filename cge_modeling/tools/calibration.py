from collections.abc import Callable, Mapping
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import TYPE_CHECKING, cast

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp

from pytensor.gradient import pushforward
from pytensor.graph.replace import vectorize_graph
from scipy.optimize import root
from scipy.sparse.csgraph import connected_components, maximum_bipartite_matching

from cge_modeling.compile.pytensor import xtensor_cge_primitives_to_pytensor
from cge_modeling.compile.pytensor_tools import compile_with_packed_inputs
from cge_modeling.sparsity import jacobian_sparsity_vjp

if TYPE_CHECKING:
    from cge_modeling.base.cge import CGEModel

RESIDUAL_TOLERANCE = 1e-6
DEFAULT_START = 0.8


@dataclass(frozen=True)
class CalibrationSystem:
    """
    The flat system of a model built in a ``with`` block, as a function of every variable and parameter entry.

    Entries are ordered as ``variable_dict_to_flat_array`` orders them: every variable, then every parameter, each
    raveled in C order. ``pattern[i, j]`` is True when equation element ``i`` can depend on entry ``j``.
    """

    residual_and_jacobian_columns: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    pattern: np.ndarray
    equation_names: list[str]
    entry_names: list[str]
    entry_shapes: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class SolveBlock:
    """Equation elements and the unknown entries they determine together, as positions in the flat system."""

    equations: np.ndarray
    unknowns: np.ndarray


def residual_is_converged(residual: np.ndarray) -> bool:
    """Return whether ``residual`` is finite with half its squared norm below ``RESIDUAL_TOLERANCE``."""
    return bool(np.isfinite(residual).all() and 0.5 * np.sum(residual**2) < RESIDUAL_TOLERANCE)


def compile_calibration_system(model: "CGEModel") -> CalibrationSystem:
    """
    Compile the residual, Jacobian columns and structural sparsity pattern of a model built in a ``with`` block.

    Jacobian columns are pushforwards along one-hot tangents, so an equation that overflows leaves the derivatives of
    the other equations finite.
    """
    system, variables, parameters = xtensor_cge_primitives_to_pytensor(model)
    entries = [*variables, *parameters]
    shapes = [tensor.type.shape for tensor in entries]
    n_entries = sum(int(np.prod(shape)) for shape in shapes)

    tangent = pt.vector("tangent", shape=(n_entries,))
    tangent_columns = pt.matrix("tangent_columns", shape=(None, n_entries))
    jvp = cast(
        pt.TensorVariable, pushforward(system, entries, pt.unpack(tangent, packed_shapes=shapes))
    )
    jvp_columns = cast(pt.TensorVariable, vectorize_graph(jvp, replace={tangent: tangent_columns}))
    residual_and_jvp_columns = compile_with_packed_inputs(
        inputs=[("z", entries), tangent_columns], outputs=[system, jvp_columns]
    )

    def residual_and_jacobian_columns(values, columns):
        directions = np.zeros((len(columns), n_entries))
        directions[np.arange(len(columns)), columns] = 1.0
        residual, jacobian_rows = residual_and_jvp_columns(values, directions)
        return residual, jacobian_rows.T

    pattern_function = compile_with_packed_inputs(
        inputs=[("z", entries)], outputs=jacobian_sparsity_vjp(system, entries)
    )
    pattern = np.asarray(pattern_function(np.ones(n_entries)), dtype=bool)

    return CalibrationSystem(
        residual_and_jacobian_columns=residual_and_jacobian_columns,
        pattern=pattern,
        equation_names=model.unpacked_equation_names,
        entry_names=[*model.unpacked_variable_names, *model.unpacked_parameter_names],
        entry_shapes={
            model_object.name: shape
            for model_object, shape in zip(
                [*model.variables, *model.parameters], shapes, strict=True
            )
        },
    )


def solve_order(incidence: np.ndarray) -> tuple[list[SolveBlock], np.ndarray, np.ndarray]:
    """
    Decompose an equation-by-unknown incidence matrix into blocks that can be solved one after another.

    Each equation is matched to a distinct unknown it contains. The strongly connected components of the matched
    dependency graph are the blocks, returned so that every block depends only on unknowns solved by earlier ones.

    Parameters
    ----------
    incidence : ndarray of bool
        ``incidence[i, j]`` is True when equation ``i`` depends on unknown ``j``.

    Returns
    -------
    blocks : list of SolveBlock
        The blocks in solve order, with positions into the rows and columns of ``incidence``.
    unmatched_equations : ndarray of int
        Equations matched to no unknown. They hold only through the other equations or the data.
    unmatched_unknowns : ndarray of int
        Unknowns no equation can determine.
    """
    n_equations, n_unknowns = incidence.shape
    matched_unknown = maximum_bipartite_matching(sp.csr_array(incidence), perm_type="column")
    matched_equations = np.flatnonzero(matched_unknown >= 0)
    equation_of_unknown = np.full(n_unknowns, -1)
    equation_of_unknown[matched_unknown[matched_equations]] = matched_equations

    dependencies = [
        (equation, equation_of_unknown[unknown])
        for equation in matched_equations
        for unknown in np.flatnonzero(incidence[equation])
        if equation_of_unknown[unknown] not in (-1, equation)
    ]
    sources = [equation for equation, _ in dependencies]
    targets = [dependency for _, dependency in dependencies]
    graph = sp.csr_array(
        (np.ones(len(dependencies)), (sources, targets)), shape=(n_equations, n_equations)
    )
    _, component = connected_components(graph, directed=True, connection="strong")

    depends_on: dict[int, set[int]] = {
        int(component[equation]): set() for equation in matched_equations
    }
    for source, target in dependencies:
        if component[source] != component[target]:
            depends_on[int(component[source])].add(int(component[target]))

    blocks = []
    for label in TopologicalSorter(depends_on).static_order():
        equations = matched_equations[component[matched_equations] == label]
        blocks.append(SolveBlock(equations=equations, unknowns=matched_unknown[equations]))

    unmatched_equations = np.flatnonzero(matched_unknown < 0)
    unmatched_unknowns = np.flatnonzero(equation_of_unknown < 0)
    return blocks, unmatched_equations, unmatched_unknowns


def flatten_entries(
    system: CalibrationSystem, values: Mapping[str, float | np.ndarray], fill: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the flat entry vector with ``values`` filled in, and a mask of the entries ``values`` gives."""
    unknown_names = set(values) - set(system.entry_shapes)
    if unknown_names:
        raise ValueError(f"{sorted(unknown_names)} are not variables or parameters of the model.")

    flat, given = [], []
    for name, shape in system.entry_shapes.items():
        size = int(np.prod(shape))
        if name in values:
            flat.append(np.broadcast_to(np.asarray(values[name], dtype=float), shape).ravel())
            given.append(np.ones(size, dtype=bool))
        else:
            flat.append(np.full(size, fill))
            given.append(np.zeros(size, dtype=bool))
    return np.concatenate(flat), np.concatenate(given)


def solve_block(
    system: CalibrationSystem,
    entries: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    max_iter: int,
) -> None:
    """Solve the equations ``rows`` for the entries ``columns`` in place, holding every other entry fixed."""

    def block_residual_and_jacobian(values):
        entries[columns] = values
        residual, jacobian_columns = system.residual_and_jacobian_columns(entries, columns)
        return residual[rows], jacobian_columns[rows]

    result = root(
        block_residual_and_jacobian,
        entries[columns],
        jac=True,
        method="lm",
        options={"xtol": 1e-14, "ftol": 1e-14, "maxiter": max_iter},
    )
    entries[columns] = result.x
    residual, _ = block_residual_and_jacobian(result.x)
    if not residual_is_converged(residual):
        equation_names = [system.equation_names[row] for row in rows]
        unknown_names = [system.entry_names[column] for column in columns]
        raise ValueError(
            f"Could not solve {equation_names} for {unknown_names}: the residual is {residual}. "
            "Starting values in initial_values may help."
        )


def calibrate_entries(
    system: CalibrationSystem,
    data: Mapping[str, float | np.ndarray],
    initial_values: Mapping[str, float | np.ndarray],
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve for every entry ``data`` does not give, block by block.

    Parameters
    ----------
    system : CalibrationSystem
        The compiled system.
    data : dict of str to float or ndarray
        Known values, keyed by variable or parameter name.
    initial_values : dict of str to float or ndarray
        Starting values for unknown entries, keyed by name. Unknown entries without one start at 0.8, inside the unit interval where shares and their complements ``1 - share`` are both valid.
    max_iter : int
        Iteration limit of each block's solve.

    Returns
    -------
    entries : ndarray
        Every entry, given or solved, in flat order.
    residual : ndarray
        The residual of every equation element at ``entries``.

    Raises
    ------
    ValueError
        If ``data`` leaves an unknown that no equation determines, or a block cannot be solved.
    """
    overlap = set(data) & set(initial_values)
    if overlap:
        raise ValueError(f"{sorted(overlap)} are in both data and initial_values.")

    starts, _ = flatten_entries(system=system, values=initial_values, fill=DEFAULT_START)
    entries, given = flatten_entries(system=system, values=data, fill=np.nan)
    unknown_columns = np.flatnonzero(~given)
    entries[unknown_columns] = starts[unknown_columns]

    blocks, unmatched_equations, unmatched_unknowns = solve_order(
        system.pattern[:, unknown_columns]
    )
    if unmatched_unknowns.size:
        names = [system.entry_names[unknown_columns[unknown]] for unknown in unmatched_unknowns]
        raise ValueError(f"No equation determines {names} given the data.")

    for block in blocks:
        try:
            solve_block(
                system=system,
                entries=entries,
                rows=block.equations,
                columns=unknown_columns[block.unknowns],
                max_iter=max_iter,
            )
        except ValueError as error:
            if not unmatched_equations.size:
                raise
            names = [system.equation_names[equation] for equation in unmatched_equations]
            raise ValueError(
                f"{error} The data over-determine the model: {names} contain no unknown, so data that "
                "disagree with them can leave this block without a solution."
            ) from error

    residual, _ = system.residual_and_jacobian_columns(entries, np.array([], dtype=int))
    return entries, residual
