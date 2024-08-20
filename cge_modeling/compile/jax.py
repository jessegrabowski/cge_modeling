from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from pytensor.compile import Supervisor, mode
from pytensor.graph import FunctionGraph
from pytensor.link.jax.dispatch import jax_funcify

from cge_modeling import CGEModel
from cge_modeling.base.function_wrappers import return_array_from_jax_wrapper
from cge_modeling.compile.constants import CompiledFunctions
from cge_modeling.compile.pytensor import (
    cge_primitives_to_pytensor,
    pytensor_euler_function_with_python_loop,
    validate_pytensor_parsing_result,
)
from cge_modeling.compile.pytensor_tools import flat_tensor_to_ragged_list


def graph_to_jax_function(inputs, outputs):
    if not isinstance(outputs, list):
        outputs = [outputs]

    fgraph = FunctionGraph(inputs=inputs, outputs=outputs, clone=True)
    fgraph.attach_feature(
        Supervisor(
            input
            for input in fgraph.inputs
            if not (hasattr(fgraph, "destroyers") and fgraph.has_destroyers([input]))
        )
    )
    mode.JAX.optimizer.rewrite(fgraph)
    jax_func = jax_funcify(fgraph)

    return jax_func


def jax_euler_step(system, variables, parameters):
    f_system_jax = graph_to_jax_function(inputs=[*variables, *parameters], outputs=[system])

    x_shapes = [x.type.shape for x in variables]
    theta_shapes = [x.type.shape for x in parameters]

    variable_names = [x.name for x in variables]
    parameter_names = [x.name for x in parameters]

    def f_sys_wrapped(x, theta):
        xs = flat_tensor_to_ragged_list(x, x_shapes)
        thetas = flat_tensor_to_ragged_list(theta, theta_shapes)

        return f_system_jax(*xs, *thetas)[0]

    f_sys = jax.jit(f_sys_wrapped)

    def f_step(**kwargs):
        n_steps = kwargs.pop("n_steps")
        x_list = [kwargs.pop(x) for x in variable_names]

        theta_list = [kwargs.pop(x) for x in parameter_names]
        theta0 = [kwargs.pop(f"{x}_initial") for x in parameter_names]
        theta_final = [kwargs.pop(f"{x}_final") for x in parameter_names]

        x_vec = jnp.concatenate([jnp.ravel(x) for x in x_list])
        theta_vec = jnp.concatenate([jnp.ravel(x) for x in theta_list])
        theta0_vec = jnp.concatenate([jnp.ravel(x) for x in theta0])
        theta_final_vec = jnp.concatenate([jnp.ravel(x) for x in theta_final])

        step_size = (theta_final_vec - theta0_vec) / n_steps

        A = jax.jacobian(f_sys, 0)(x_vec, theta_vec)
        _, Bv = jax.jvp(lambda theta: f_sys(x_vec, theta), (theta_vec,), (step_size,))
        step = -jax.scipy.linalg.solve(A, Bv)

        x_next_vec = x_vec + step
        theta_next_vec = theta_vec + step_size

        x_next = flat_tensor_to_ragged_list(x_next_vec, x_shapes)
        theta_next = flat_tensor_to_ragged_list(theta_next_vec, theta_shapes)

        return [*x_next, *theta_next]

    return f_step


def jax_loss_grad_hessp(system, variables, parameters):
    loss = 0.5 * (system**2).sum()
    f_loss_jax = graph_to_jax_function(variables + parameters, loss)

    x_shapes = [x.type.shape for x in variables]
    theta_shapes = [x.type.shape for x in parameters]

    def f_loss_wrapped(x, theta):
        xs = flat_tensor_to_ragged_list(x, x_shapes)
        thetas = flat_tensor_to_ragged_list(theta, theta_shapes)

        return f_loss_jax(*xs, *thetas)[0]

    grad = jax.grad(f_loss_wrapped, 0)

    def f_grad_jax(x, theta):
        return jnp.stack(grad(x, theta))

    _f_hess_jax = jax.jacfwd(f_grad_jax, argnums=0)

    def f_hess_jax(x, theta):
        return jnp.stack(_f_hess_jax(x, theta))

    def f_hessp_jax(x, p, theta):
        _, u = jax.jvp(lambda x: f_grad_jax(x, theta), (x,), (p,))
        return jnp.stack(u)

    return f_loss_wrapped, f_grad_jax, f_hess_jax, f_hessp_jax


def compile_jax_cge_functions(
    cge_model: CGEModel, functions_to_compile: list[CompiledFunctions], *args, **kwargs
) -> tuple[Callable, Callable, Callable, Callable, Callable, Callable, Callable]:
    system, variables, parameters, (cache, unpacked_cache) = cge_primitives_to_pytensor(cge_model)
    validate_pytensor_parsing_result(system, variables, parameters, cache)

    f_system = graph_to_jax_function(inputs=[*variables, *parameters], outputs=[system])
    f_system = return_array_from_jax_wrapper(f_system)

    # Optional functions
    f_jac = None
    f_resid = None
    f_grad = None
    f_hess = None
    f_hessp = None
    f_euler = None

    if "root" in functions_to_compile:

        def f_jac(x, theta):
            return jnp.stack(jax.jacobian(f_system, 0)(*x, *theta))

    if "minimize" in functions_to_compile:
        # Symbolically compute loss function and derivatives
        f_resid, f_grad, f_hess, f_hessp = jax_loss_grad_hessp(system, variables, parameters)

    if "euler" in functions_to_compile:
        # Compile the one-step linear approximation function used by the iterative Euler approximation
        # We don't need to save this because it's only used internally by the euler_approx function
        f_step = jax_euler_step(system, variables, parameters)

        f_euler = partial(
            pytensor_euler_function_with_python_loop,
            cge_model=cge_model,
            f_step=f_step,
            f_system=f_system,
            f_grad=f_grad,
        )

    return tuple(jax.jit(f) for f in [f_system, f_jac, f_resid, f_grad, f_hess, f_hessp, f_euler])
