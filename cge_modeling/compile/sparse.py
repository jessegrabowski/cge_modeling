from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import networkx as nx
import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp

from pytensor.compile.mode import Mode
from pytensor.graph.replace import vectorize_graph
from scipy.sparse.linalg import spsolve

from cge_modeling.compile.pytensor_tools import compile_with_packed_inputs, pushforward
from cge_modeling.sparsity import jacobian_sparsity_vjp


def color_jacobian_columns(pattern: np.ndarray) -> np.ndarray:
    """
    Color the columns of a Jacobian sparsity pattern so that no two columns of one color share a nonzero row.

    Parameters
    ----------
    pattern : ndarray of bool
        Structural nonzeros of the Jacobian, of shape ``(n_rows, n_columns)``.

    Returns
    -------
    colors : ndarray of int
        Color of each column, numbered from 0.
    """
    incidence = sp.csc_array(pattern.astype(np.int64))
    overlap = (incidence.T @ incidence).tocoo()
    off_diagonal = overlap.row != overlap.col

    graph = nx.Graph()
    graph.add_nodes_from(range(pattern.shape[1]))
    graph.add_edges_from(
        zip(overlap.row[off_diagonal].tolist(), overlap.col[off_diagonal].tolist())
    )
    coloring = nx.greedy_color(graph, strategy="saturation_largest_first")

    return np.array([coloring[column] for column in range(pattern.shape[1])], dtype=np.int64)


@dataclass(frozen=True)
class SparseJacobianLayout:
    """
    Where each entry of a compressed Jacobian goes in the CSC form of the full Jacobian.

    Attributes
    ----------
    shape : tuple of int
        Shape of the full Jacobian.
    indices : ndarray of int
        CSC row indices of the structural nonzeros.
    indptr : ndarray of int
        CSC column pointers.
    gather_colors : ndarray of int
        For each structural nonzero in CSC order, the color whose pushforward holds its value.
    """

    shape: tuple[int, int]
    indices: np.ndarray
    indptr: np.ndarray
    gather_colors: np.ndarray

    @classmethod
    def from_pattern(cls, pattern: np.ndarray, colors: np.ndarray) -> "SparseJacobianLayout":
        csc_pattern = sp.csc_matrix(pattern)
        csc_pattern.sort_indices()
        columns = np.repeat(np.arange(pattern.shape[1]), np.diff(csc_pattern.indptr))
        return cls(
            shape=pattern.shape,
            indices=csc_pattern.indices,
            indptr=csc_pattern.indptr,
            gather_colors=colors[columns],
        )

    def assemble(self, compressed: np.ndarray) -> sp.csc_matrix:
        """Build the CSC Jacobian from ``compressed``, the pushforwards of shape ``(n_colors, n_rows)``."""
        values = compressed[self.gather_colors, self.indices]
        return sp.csc_matrix((values, self.indices, self.indptr), shape=self.shape)


def compressed_jacobian(
    system: pt.TensorVariable,
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
) -> tuple[pt.TensorVariable, SparseJacobianLayout]:
    """
    Build the compressed Jacobian of ``system`` with respect to ``variables``, one pushforward per column color.

    Parameters
    ----------
    system : TensorVariable
        Flat vector of model residuals.
    variables : list of TensorVariable
        Variables to differentiate with respect to.
    parameters : list of TensorVariable
        Parameters of the system, needed to evaluate its sparsity pattern.

    Returns
    -------
    compressed : TensorVariable
        Graph of shape ``(n_colors, n_rows)`` whose row ``c`` is the Jacobian times the indicator of color ``c``.
    layout : SparseJacobianLayout
        Map from ``compressed`` to the CSC Jacobian.
    """
    shapes = [variable.type.shape for variable in variables]
    n_variables = sum(int(np.prod(shape)) for shape in shapes)

    pattern_function = compile_with_packed_inputs(
        inputs=[("x", variables), ("theta", parameters)],
        outputs=jacobian_sparsity_vjp(system, variables),
        on_unused_input="ignore",
    )
    n_parameters = sum(int(np.prod(parameter.type.shape)) for parameter in parameters)
    pattern = np.asarray(pattern_function(np.ones(n_variables), np.ones(n_parameters)), dtype=bool)
    colors = color_jacobian_columns(pattern)

    tangent = pt.vector("tangent", shape=(n_variables,))
    [jvp] = pushforward([system], variables, pt.unpack(tangent, packed_shapes=shapes))
    seeds = pt.constant(
        (colors[None, :] == np.arange(colors.max(initial=-1) + 1)[:, None]).astype(system.dtype)
    )
    compressed = cast(pt.TensorVariable, vectorize_graph(jvp, replace={tangent: seeds}))

    return compressed, SparseJacobianLayout.from_pattern(pattern, colors)


def compile_sparse_system_and_jacobian(
    system: pt.TensorVariable,
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    compressed: pt.TensorVariable,
    layout: SparseJacobianLayout,
    mode: str | Mode | None = None,
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, sp.csc_matrix]]:
    """
    Compile ``f(x, theta) -> (residual, jacobian)``, with the Jacobian with respect to ``x`` as a CSC matrix.

    Parameters
    ----------
    system : TensorVariable
        Flat vector of model residuals.
    variables : list of TensorVariable
        Variables, packed into ``x``.
    parameters : list of TensorVariable
        Parameters, packed into ``theta``.
    compressed : TensorVariable
        Compressed Jacobian from :func:`compressed_jacobian`.
    layout : SparseJacobianLayout
        Layout of ``compressed`` from :func:`compressed_jacobian`.
    mode : str or Mode, optional
        PyTensor compile mode.

    Returns
    -------
    f : callable
        The fused residual and sparse Jacobian function.
    """
    f_compressed = compile_with_packed_inputs(
        inputs=[("x", variables), ("theta", parameters)], outputs=[system, compressed], mode=mode
    )

    def system_and_sparse_jacobian(
        x: np.ndarray, theta: np.ndarray
    ) -> tuple[np.ndarray, sp.csc_matrix]:
        residual, compressed_values = f_compressed(x, theta)
        return np.asarray(residual), layout.assemble(np.asarray(compressed_values))

    return system_and_sparse_jacobian


def compile_sparse_euler_step(
    system: pt.TensorVariable,
    variables: list[pt.TensorVariable],
    parameters: list[pt.TensorVariable],
    compressed: pt.TensorVariable,
    layout: SparseJacobianLayout,
    mode: str | Mode | None = None,
) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    r"""
    Compile one Euler step that solves the linearized system with a sparse Jacobian.

    The step solves :math:`A \, dx = -B \, d\theta`, where :math:`A` is the Jacobian with respect to the variables,
    assembled sparse, and :math:`B \, d\theta` is one pushforward along the parameter step.

    Parameters
    ----------
    system : TensorVariable
        Flat vector of model residuals.
    variables : list of TensorVariable
        Variables, packed into ``x``.
    parameters : list of TensorVariable
        Parameters, packed into ``theta``.
    compressed : TensorVariable
        Compressed Jacobian from :func:`compressed_jacobian`.
    layout : SparseJacobianLayout
        Layout of ``compressed`` from :func:`compressed_jacobian`.
    mode : str or Mode, optional
        PyTensor compile mode.

    Returns
    -------
    f_step : callable
        ``f_step(x, theta, theta_initial, theta_final, n_steps) -> (x_next, theta_next)``.
    """
    parameter_shapes = [parameter.type.shape for parameter in parameters]
    n_parameters = sum(int(np.prod(shape)) for shape in parameter_shapes)
    parameter_step = pt.vector("parameter_step", shape=(n_parameters,))
    [system_pushforward] = pushforward(
        [system], parameters, pt.unpack(parameter_step, packed_shapes=parameter_shapes)
    )

    f_linearization = compile_with_packed_inputs(
        inputs=[("x", variables), ("theta", parameters), parameter_step],
        outputs=[system_pushforward, compressed],
        mode=mode,
    )

    def sparse_euler_step(
        x: np.ndarray,
        theta: np.ndarray,
        theta_initial: np.ndarray,
        theta_final: np.ndarray,
        n_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        theta_step = (theta_final - theta_initial) / n_steps
        b_times_step, compressed_values = f_linearization(x, theta, theta_step)
        x_step = spsolve(layout.assemble(np.asarray(compressed_values)), -np.asarray(b_times_step))
        return x + x_step, theta + theta_step

    return sparse_euler_step
