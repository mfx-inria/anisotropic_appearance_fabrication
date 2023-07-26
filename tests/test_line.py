import jax
import jax.numpy as jnp

import cglib.direction
import cglib.line


def helper_average_with_weights():
    lines_polar = jnp.array([0.5 * jnp.pi, -0.75 * jnp.pi])
    lines = jax.vmap(cglib.direction.polar_to_cartesian)(lines_polar)
    weights = jnp.array([1.5, 1.])
    res = cglib.line.average_with_weights(lines, weights)
    res_exp = jnp.array([0.20759147, 0.97821563])
    return jnp.all(jnp.isclose(res, res_exp))


def test_average_with_weights():
    assert jax.jit(helper_average_with_weights)()


def helper_smoothness_energy():
    lines_polar = jnp.array([0.5 * jnp.pi, -0.75 * jnp.pi])
    lines = jax.vmap(cglib.direction.polar_to_cartesian)(lines_polar)
    weights = jnp.array([1., 1.])
    line_averaged = cglib.line.average_with_weights(lines, weights)
    res = cglib.line.smoothness_energy(line_averaged, lines) / lines.shape[0]
    res_exp = -0.8535533
    return jnp.isclose(res, res_exp)


def test_smoothness_energy():
    assert jax.jit(helper_smoothness_energy)()


def helper_smoothness_energy_normalized():
    lines_polar = jnp.array([0.5 * jnp.pi, -0.75 * jnp.pi])
    lines = jax.vmap(cglib.direction.polar_to_cartesian)(lines_polar)
    weights = jnp.array([1., 1.])
    line_averaged = cglib.line.average_with_weights(lines, weights)
    res = cglib.line.smoothness_energy_normalized(line_averaged, lines)
    exp_res = 0.1464467
    return jnp.isclose(res, exp_res)


def test_smoothness_energy_normalized():
    assert jax.jit(helper_smoothness_energy_normalized)()
