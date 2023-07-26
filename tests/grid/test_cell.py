import jax
import jax.numpy as jnp

import cglib.grid
import cglib.grid.cell
import cglib.limits
import cglib.tree_util
import cglib.type


def helper_corner_vertex_ndindices():
    cell_ndindex = jnp.array((2, 3))
    res = cglib.grid.cell.corner_vertex_ndindices(cell_ndindex)
    exp_res = jnp.array([[2, 3],
                         [3, 3],
                         [2, 4],
                         [3, 4]])
    return jnp.all(jnp.isclose(res, exp_res))


def test_corner_vertex_ndindices():
    assert jax.jit(helper_corner_vertex_ndindices)()


def helper_index1_from_masked_ndindex():
    cell_ndcount = jnp.array([4, 5])
    cell_ndindex = jnp.array([cglib.limits.INT_MAX, 1])
    res = cglib.grid.cell.index1_from_masked_ndindex(
        cell_ndindex, cell_ndcount)
    exp_res = cglib.limits.INT_MAX
    return res == exp_res


def test_index1_from_masked_ndindex():
    assert jax.jit(helper_index1_from_masked_ndindex)()


def helper_index1_from_ndindex():
    # The grid has four columns and five rows.
    cell_ndcount = jnp.array([4, 5])
    # Third column, second row
    cell_ndindex = jnp.array([2, 1])
    # XXXX
    # XXXX
    # XXXX
    # XXOX <- [2, 1]
    # XXXX
    res = cglib.grid.cell.index1_from_ndindex(cell_ndindex, cell_ndcount)
    return jnp.equal(res, 6)


def test_index1_from_ndindex():
    assert jax.jit(helper_index1_from_ndindex)()


def helper_ndindex_from_1dindex():
    # The grid has four columns and five rows.
    cell_ndcount = jnp.array([4, 5])
    # Third column, second row
    cell_ndindex = jnp.array([2, 1])
    # XXXX
    # XXXX
    # XXXX
    # XXOX <- [2, 1]
    # XXXX
    cell_1dindex = cglib.grid.cell.index1_from_ndindex(
        cell_ndindex, cell_ndcount)
    res1 = jnp.equal(cell_1dindex, 6)
    cell_ndindex_2 = cglib.grid.cell.ndindex_from_1dindex(
        cell_1dindex, cell_ndcount)
    res2 = jnp.all(jnp.equal(cell_ndindex, cell_ndindex_2))
    return jnp.logical_and(res1, res2)


def test_ndindex_from_1dindex():
    assert jax.jit(helper_ndindex_from_1dindex)()


def helper_ndindex_from_1dindex_2():
    with jax.ensure_compile_time_eval():
        cell_ndcount_array = jnp.array([2, 3], cglib.type.uint)
        uniform_grid_cell_1dindices = jnp.arange(
            jnp.prod(cell_ndcount_array))
    uniform_grid_cell_2dindices = jax.vmap(
        cglib.grid.cell.ndindex_from_1dindex, (0, None))(
        uniform_grid_cell_1dindices, cell_ndcount_array)
    uniform_grid_cell_1dindices_2 = jax.vmap(
        cglib.grid.cell.index1_from_ndindex, (0, None))(
        uniform_grid_cell_2dindices, cell_ndcount_array)
    return jnp.all(
        jnp.equal(
            uniform_grid_cell_1dindices,
            uniform_grid_cell_1dindices_2))


def test_ndindex_from_1dindex_2():
    assert jax.jit(helper_ndindex_from_1dindex_2)()


def test_ndindex_is_valid():
    cell_nd_count = jnp.array((2, 3))
    # Cell 2D indexing
    #  ________ ________
    # |        |        |
    # | (0, 2) | (1, 2) |
    # |________|________|
    # |        |        |
    # | (0, 1) | (1, 1) |
    # |________|________|
    # |        |        |
    # | (0, 0) | (1, 0) |
    # |________|________|
    cell_ndindex = jnp.array((2, 3))
    res = cglib.grid.cell.ndindex_is_valid(cell_ndindex, cell_nd_count)
    assert not res


def helper_ndindex_from_point():
    p = jnp.array([1.5, 3.5])
    origin = jnp.array([0., 1.])
    cell_sides_length = 1.
    res = cglib.grid.cell.ndindex_from_point(p, origin, cell_sides_length)
    exp_res = jnp.array([1, 2])
    return jnp.all(jnp.equal(res, exp_res))


def test_ndindex_from_point():
    assert jax.jit(helper_ndindex_from_point)()


def helper_ndindex_from_masked_point():
    p = jnp.array([jnp.nan, 3.5])
    origin = jnp.array([0., 1.])
    cell_sides_length = 1.
    res = cglib.grid.cell.ndindex_from_masked_point(
        p, origin, cell_sides_length)
    exp_res = jnp.full(p.shape, cglib.limits.INT_MAX)
    return jnp.all(jnp.equal(res, exp_res))


def test_ndindex_from_masked_point():
    assert jax.jit(helper_ndindex_from_masked_point)()


def helper_moore_neighborhood():
    res = cglib.grid.cell.moore_neighborhood(jnp.array([2, 3]))
    exp_res = jnp.array([[1, 2],
                         [2, 2],
                         [3, 2],
                         [1, 3],
                         [2, 3],
                         [3, 3],
                         [1, 4],
                         [2, 4],
                         [3, 4]])
    return jnp.all(jnp.equal(res, exp_res))


def test_moore_neighborhood():
    assert jax.jit(helper_moore_neighborhood)()


def helper_moore_neighborhood_from_point():
    cell_ndcount = jnp.array([2, 3])
    origin = jnp.array([-1., 2.])
    cell_sides_length = 0.5
    grid = cglib.grid.Grid(cell_ndcount, origin, cell_sides_length)
    x = jnp.array([-0.25, 2.25])
    res = cglib.grid.cell.moore_neighborhood_from_point(x, grid)
    exp_res1 = jnp.array([[0, -1],
                          [1, -1],
                          [2, -1],
                          [0,  0],
                          [1,  0],
                          [2,  0],
                          [0,  1],
                          [1,  1],
                          [2,  1]])
    exp_res2 = jnp.array([True, True, True, False,
                          False, True, False, False, True])
    exp_res = (exp_res1, exp_res2)
    return cglib.tree_util.all_equal(res, exp_res)


def test_moore_neighborhood_from_point():
    assert jax.jit(helper_moore_neighborhood_from_point)()


def helper_center_point():
    cell_ndcount = jnp.array([2, 3])
    origin = jnp.array([-1., 2.])
    cell_sides_length = 0.5
    grid = cglib.grid.Grid(cell_ndcount, origin, cell_sides_length)
    # Second column, third row
    cell_ndindex = jnp.array([1, 2])
    res = cglib.grid.cell.center_point(grid, cell_ndindex)
    exp_res = jnp.array([-0.25, 3.25])
    return jnp.all(jnp.isclose(res, exp_res))


def test_center_point():
    assert jax.jit(helper_center_point)()


def test_center_points():

    grid_cell_2dcount = jnp.array([2, 3])
    grid_origin = jnp.array([-1., 2.])
    grid_cell_sides_length = 0.5
    grid = cglib.grid.Grid(
        grid_cell_2dcount, grid_origin, grid_cell_sides_length)
    res = cglib.grid.cell.center_points(grid)
    exp_res = jnp.array([[-0.75, 2.25],
                         [-0.25, 2.25],
                         [-0.75, 2.75],
                         [-0.25, 2.75],
                         [-0.75, 3.25],
                         [-0.25, 3.25]])
    assert jnp.all(jnp.isclose(res, exp_res))


def test_center_points_jittered():

    grid_cell_2dcount = jnp.array([2, 3])
    grid_origin = jnp.array([-1., 2.])
    grid_cell_sides_length = 0.5
    grid = cglib.grid.Grid(grid_cell_2dcount,
                           grid_origin,
                           grid_cell_sides_length)
    seed = 0
    seed_jax = jax.random.PRNGKey(seed)
    res = cglib.grid.cell.center_points_jittered(grid, seed_jax, 1.)
    exp_res = jnp.array([[-0.5584955, 2.0678668],
                         [-0.1643188, 2.3626184],
                         [-0.82621616, 2.877863],
                         [-0.16134614, 2.5096464],
                         [-0.7924377, 3.0805643],
                         [-0.37570453, 3.303907]])
    assert jnp.all(jnp.isclose(res, exp_res))


def helper_boundary_1dindex_from_2dindex():
    grid_cell_2dcount = (3, 4)
    grid_boundary_cell_count = cglib.grid.cell.boundary2_1dcount(
        grid_cell_2dcount)
    grid_boundary_cell_flattened_indices = jnp.arange(grid_boundary_cell_count)

    boundary_cells = jax.vmap(
        cglib.grid.cell.boundary_2dindex_from_1dindex, (0, None))(
        grid_boundary_cell_flattened_indices, grid_cell_2dcount)

    grid_boundary_edge_flattened_indices_2 = jax.vmap(
        cglib.grid.cell.boundary_1dindex_from_2dindex, (0, None))(
        boundary_cells, grid_cell_2dcount)
    return jnp.all(
        jnp.equal(
            grid_boundary_edge_flattened_indices_2,
            grid_boundary_cell_flattened_indices))


def test_boundary_1dindex_from_2dindex():
    assert jax.jit(helper_boundary_1dindex_from_2dindex)()
