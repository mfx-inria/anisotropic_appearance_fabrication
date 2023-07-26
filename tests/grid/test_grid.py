import jax
import jax.numpy as jnp

import cglib.aabb
import cglib.grid
import cglib.tree_util


def helper_aabb():
    cell_ndcount = jnp.array([2, 3])
    origin = jnp.array([-1., 2.])
    cell_sides_length = 0.5
    grid = cglib.grid.Grid(cell_ndcount, origin, cell_sides_length)
    res = cglib.grid.aabb(grid)
    exp_res = cglib.aabb.AABB(
        p_min=jnp.array([-1.,  2.]),
        p_max=jnp.array([0., 3.5]))
    return cglib.tree_util.all_equal(res, exp_res)


def test_aabb():
    assert jax.jit(helper_aabb)()
