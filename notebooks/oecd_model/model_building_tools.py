import re

import graphviz as gr
from cge_modeling.pytensorf.compile import (
    object_to_pytensor,
    make_printer_cache,
    normalize_eq,
)
import pytensor
import numpy as np
import logging


_log = logging.getLogger(__name__)


def draw_graph(edge_list, node_props=None, edge_props=None, graph_direction="UD"):
    """Utility to draw a causal (directed) graph"""
    g = gr.Digraph(graph_attr={"rankdir": graph_direction})

    edge_props = {} if edge_props is None else edge_props
    for e in edge_list:
        props = edge_props[e] if e in edge_props else {}
        g.edge(e[0], e[1], **props)

    if node_props is not None:
        for name, props in node_props.items():
            g.node(name=name, **props)
    return g


def make_or_update_cache(pt_vars, pt_params, cache=None, allow_overwrite=False):
    new_cache = make_printer_cache(pt_vars, pt_params)
    new_cache = {k[0]: v for k, v in new_cache.items()}
    if cache is not None:
        for k, v in new_cache.items():
            if k in cache.keys() and not allow_overwrite:
                raise KeyError(
                    f"{k} already exists in cache, check code for duplicate declarations."
                )
            cache[k] = v
    else:
        cache = new_cache
    return cache


def convert_to_pt(variables, parameters, coords, cache=None, allow_overwrite=False):
    pt_vars = [object_to_pytensor(var, coords) for var in variables]
    pt_params = [object_to_pytensor(param, coords) for param in parameters]
    cache = make_or_update_cache(
        pt_vars, pt_params, cache, allow_overwrite=allow_overwrite
    )

    return pt_vars, pt_params, cache


def convert_equations(equations, coords, cache=None):
    import pytensor.tensor as pt

    cache = cache.copy()
    cache.update({"pt": pt})
    pt_eqs = []
    for eq in equations:
        try:
            x = eval(normalize_eq(eq.equation), cache.copy())
            pt_eqs.append(x)
        except Exception as e:
            _log.info(f"Could not compile equation: {eq.name}")
            tokens = [t.strip() for t in re.split("\W", eq.equation) if t.strip()]
            vars = [cache.get(t) for t in tokens if t.strip() in cache.keys()]
            non_vars = list(set([t for t in tokens if t.strip() not in cache.keys()]))
            _log.info("Found the following variables with the following shapes:")
            for v in vars:
                _log.info(f"\t{v.name}: {v.type.shape}")
            if non_vars:
                _log.info(
                    "Found the following non-variables, which might be junk, or might be missing in the model definition:"
                )  # pragma: no cover
                for nv in non_vars:
                    _log.info(f"\t{nv}")
            raise e

    return pt_eqs


def test_equations(
    variables,
    parameters,
    equations,
    coords,
    cache=None,
    verbose=True,
    allow_overwrite=True,
):
    pt_vars, pt_params, cache = convert_to_pt(
        variables, parameters, coords, cache, allow_overwrite=allow_overwrite
    )

    pt_eqs = convert_equations(equations, coords, cache)

    inputs = list(cache.values())

    rng = np.random.default_rng()
    value_dict = {var.name: rng.beta(1, 1, size=var.type.shape) for var in inputs}

    f = pytensor.function(inputs, pt_eqs, on_unused_input="ignore")
    out = f(**value_dict)
    if verbose:
        for x, name in zip(
            [pt_vars, pt_params, pt_eqs], ["variables", "parameters", "equations"]
        ):
            _log.info(
                f"Found {len(x)} {name}; unrolled count {int(sum([np.prod(xx.type.shape) for xx in x]))}"
            )
        _log.info(f"Output shapes: {[x.shape for x in out]}")


def _sum_reduce(eq, dims, coords, backend, axis=None):
    if backend == "pytensor":
        return f"({eq}).sum(axis={axis})"
    elif backend == "numba":
        if isinstance(dims, str):
            dims = [dims]
        for dim in dims:
            n = len(coords[dim]) - 1
            eq = f"Sum({eq}, ({dim}, 0, {n}))"
        return eq
