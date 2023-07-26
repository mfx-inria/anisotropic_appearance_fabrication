import jax
import jax.numpy as jnp

import cglib.grid.multigrid


def helper_multigrid_restrict_ndindex():
    # Cell O in the fine grid
    cell_ndindex = jnp.array([1, 2])
    # fine  coarse
    # XXXX
    # XOXX
    # XXXX    OX
    # XXXX    XX
    cell_ndindex_coarse_space, shift_fine_space \
        = cglib.grid.multigrid.ndindex_restrict(cell_ndindex)
    res11_exp = jnp.array([0, 1])
    res12_exp = jnp.array([1, 0])
    res11 = jnp.all(jnp.equal(cell_ndindex_coarse_space, res11_exp))
    res12 = jnp.all(jnp.equal(shift_fine_space, res12_exp))
    res1 = jnp.logical_and(res11, res12)

    res2_exp = cell_ndindex_coarse_space * 2 + shift_fine_space
    res2 = jnp.all(jnp.equal(cell_ndindex, res2_exp))
    return jnp.logical_and(res1, res2)


def test_multigrid_restrict_ndindex():
    assert jax.jit(helper_multigrid_restrict_ndindex)()


def helper_multigrid_prolong_ndindex():
    # Cell O in the coarse grid
    cell_ndindex = jnp.array([0, 1])
    # coarse    fine
    #           OOXX
    #           OOXX
    #   OX      XXXX
    #   XX      XXXX
    res = cglib.grid.multigrid.ndindex_prolong(cell_ndindex)
    exp_res = jnp.array([[0, 2],
                         [1, 2],
                         [0, 3],
                         [1, 3]])
    return jnp.all(jnp.equal(res, exp_res))


def test_multigrid_prolong_ndindex():
    assert jax.jit(helper_multigrid_prolong_ndindex)()
