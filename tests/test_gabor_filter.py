import jax
import jax.numpy as jnp

import cglib.gabor_filter
import cglib.grid
import cglib.grid.cell
import cglib.point_data
import cglib.tree_util
import cglib.type


def helper_eval_complex():
    # Gabor filter parameters
    p = jnp.array([2., 3.])
    d_angle = jnp.pi * 0.25
    d = jnp.array([jnp.cos(d_angle), jnp.sin(d_angle)])
    phase = jnp.array([jnp.pi])
    f = 0.1
    r = 5.
    data = jnp.concatenate((d, phase))
    point_dir_phase = cglib.point_data.PointData(p, data)
    # Evaluation point
    x = jnp.array([0., 1.])
    res = cglib.gabor_filter.eval_complex(x, point_dir_phase, f, r)
    res_exp = jnp.array(0.0027814491 + 0.013286955j)
    return jnp.isclose(res, res_exp)


def test_eval_complex():
    assert jax.jit(helper_eval_complex)()


def helper_eval_value_and_spatial_weight():
    # Gabor filter parameters
    p = jnp.array([2., 3.])
    d_angle = jnp.pi * 0.25
    d = jnp.array([jnp.cos(d_angle), jnp.sin(d_angle)])
    phase = jnp.array([jnp.pi])
    f = 0.1
    r = 5.
    data = jnp.concatenate((d, phase))
    point_dir_phase = cglib.point_data.PointData(p, data)
    # Evaluation point
    x = jnp.array([0., 1.])
    res = cglib.gabor_filter.eval_value_and_spatial_weight(
        x, point_dir_phase, f, r)
    res_exp = (jnp.array(0.01328695), jnp.array(0.01357496))
    return cglib.tree_util.all_isclose(res, res_exp)


def test_eval_value_and_spatial_weight():
    assert jax.jit(helper_eval_value_and_spatial_weight)()


def test_eval_array():
    # Each gabor filter will be associated with a unique grid cell.
    grid_cell_2dcount = jnp.array([2, 3])
    grid_origin = jnp.array([-1., 2.])
    grid_cell_sides_length = 0.5
    grid = cglib.grid.Grid(
        grid_cell_2dcount, grid_origin, grid_cell_sides_length)
    seed = 1701
    seed_jax = jax.random.PRNGKey(seed)
    seed_jax = jax.random.split(seed_jax, 3)
    # Random Gabor filters' parameters
    gabor_filter_count = 6
    p = cglib.grid.cell.center_points_jittered(grid, seed_jax[0], 1.)
    d_angle = jax.random.uniform(
        seed_jax[1],
        (gabor_filter_count,),
        cglib.type.float_,
        -jnp.pi, jnp.pi)
    d = jnp.array([jnp.cos(d_angle), jnp.sin(d_angle)]).T
    phase = jax.random.uniform(
        seed_jax[2],
        (gabor_filter_count,),
        cglib.type.float_,
        -jnp.pi, jnp.pi)
    f = 1. / (0.2 * grid.cell_sides_length)
    r = grid.cell_sides_length
    data = jnp.concatenate((d, phase.reshape(gabor_filter_count, 1)), axis=1)
    point_dir_phase = cglib.point_data.PointData(p, data)
    # Evaluation point
    x = grid_origin
    res = cglib.gabor_filter.eval_array(x, point_dir_phase, f, r)
    res_exp = 0.7137441
    return res == res_exp


def test_grid_eval():
    # Each gabor filter will be associated with a unique grid cell.
    grid_cell_2dcount = jnp.array([2, 3])
    grid_origin = jnp.array([-1., 2.])
    grid_cell_sides_length = 0.5
    grid = cglib.grid.Grid(
        grid_cell_2dcount, grid_origin, grid_cell_sides_length)
    seed = 1701
    seed_jax = jax.random.PRNGKey(seed)
    seed_jax = jax.random.split(seed_jax, 3)
    # Random Gabor filters' parameters
    gabor_filter_count = 6
    p = cglib.grid.cell.center_points_jittered(grid, seed_jax[0], 1.)
    d_angle = jax.random.uniform(
        seed_jax[1],
        (gabor_filter_count,),
        cglib.type.float_,
        -jnp.pi, jnp.pi)
    d = jnp.array([jnp.cos(d_angle), jnp.sin(d_angle)]).T
    phase = jax.random.uniform(
        seed_jax[2],
        (gabor_filter_count,),
        cglib.type.float_,
        -jnp.pi, jnp.pi)
    f = 1. / (0.2 * grid.cell_sides_length)
    r = grid.cell_sides_length
    data = jnp.concatenate((d, phase.reshape(gabor_filter_count, 1)), axis=1)
    point_dir_phase = cglib.point_data.PointData(p, data)
    # Evaluation point
    x = grid_origin
    res_exp = cglib.gabor_filter.eval_array(x, point_dir_phase, f, r)
    grid_point_data = cglib.point_data.GridPointData(point_dir_phase, grid)
    res = cglib.gabor_filter.grid_eval(x, f, grid_point_data)
    assert res_exp == res
