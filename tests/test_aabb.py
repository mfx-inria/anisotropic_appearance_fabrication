import jax
import jax.numpy as jnp

import cglib.aabb
import cglib.tree_util


def helper_aabb_union_point():
    p_min = jnp.array([1., 2.])
    p_max = jnp.array([3., 4.])
    aabb = cglib.aabb.AABB(p_min, p_max)
    p = jnp.array([0., -1.])
    res = cglib.aabb.aabb_union_point(p, aabb)
    exp_res = cglib.aabb.AABB(
        p_min=jnp.array([0., -1.]),
        p_max=jnp.array([3., 4.]))
    return cglib.tree_util.all_isclose(res, exp_res)


def test_aabb_union_point():
    assert jax.jit(helper_aabb_union_point)()


def helper_aabb_corners():
    p_min = jnp.array([1., 2.])
    p_max = jnp.array([4., 5.])
    aabb = cglib.aabb.AABB(p_min, p_max)
    res = cglib.aabb.aabb_corners(aabb)
    exp_res = jnp.array([[1., 2.],
                         [4., 2.],
                         [1., 5.],
                         [4., 5.]])
    return jnp.all(jnp.isclose(res, exp_res))


def test_aabb_corners():
    assert jax.jit(helper_aabb_corners)()
