import jax
import jax.numpy as jnp

import cglib.grid.edge
import cglib.tree_util
import cglib.type


def helper_count1_from_2dcount():
    grid2_cell_2dcount = (3, 4)
    grid2_edge_1dcount = cglib.grid.edge.count1_from_cell_2dcount(
        grid2_cell_2dcount)
    grid2_edge_2dcount = jnp.array(
        cglib.grid.edge.count2_per_axis(grid2_cell_2dcount))
    grid2_edge_1dindices = jnp.arange(grid2_edge_1dcount)
    grid2_edge_2dindices, grid2_edge_axis = jax.vmap(
        cglib.grid.edge.index2_from_1dindex, (0, None))(
        grid2_edge_1dindices, grid2_edge_2dcount)
    grid2_edge_1dindices_2 = jax.vmap(
        cglib.grid.edge.index1_from_2dindex, (0, 0, None))(
        grid2_edge_2dindices, grid2_edge_axis, grid2_edge_2dcount)
    return jnp.all(jnp.equal(grid2_edge_1dindices_2, grid2_edge_1dindices))


def test_count1_from_2dcount():
    assert jax.jit(helper_count1_from_2dcount)()


def helper_count2_per_axis():
    # Grid definition
    cell_2dcount = jnp.array((2, 3))
    # Edge 2D indexing per axis
    #  ________ ________
    # | (0, 3) | (1, 3) |
    # |(0, 2)  |(1, 2)  |(2, 2)
    # |________|________|
    # | (0, 2) | (1, 2) |
    # |(0, 1)  |(1, 1)  |(2, 1)
    # |________|________|
    # | (0, 1) | (1, 1) |
    # |(0, 0)  |(1, 0)  |(2, 0)
    # |________|________|
    #   (0, 0)   (1, 0)
    res = cglib.grid.edge.count2_per_axis(cell_2dcount)
    expected_res = jnp.array([[2, 4], [3, 3]])
    return jnp.all(jnp.equal(res, expected_res))


def test_count2_per_axis():
    assert jax.jit(helper_count2_per_axis)()


def helper_indices1_from_2dgrid():

    cell_nd_count = jnp.array((2, 3))
    #  __ __
    # |__|__|
    # |__|__|
    # |__|__|
    res = cglib.grid.edge.indices1_from_2dgrid(cell_nd_count)
    exp_res = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                        10, 11, 12, 13, 14, 15, 16])
    return jnp.all(jnp.equal(res, exp_res))


def test_indices1_from_2dgrid():
    assert helper_indices1_from_2dgrid()


def helper_index2_from_1dindex():

    # Grid definition
    cell_2dcount = jnp.array((2, 3))
    edge_2dcount_per_axis = cglib.grid.edge.count2_per_axis(
        cell_2dcount)
    # [[2 4]
    #  [3 3]]

    # Edge 2D indexing per axis
    #  ________ ________
    # | (0, 3) | (1, 3) |
    # |(0, 2)  |(1, 2)  |(2, 2)
    # |________|________|
    # | (0, 2) | (1, 2) |
    # |(0, 1)  |(1, 1)  |(2, 1)
    # |________|________|
    # | (0, 1) | (1, 1) |
    # |(0, 0)  |(1, 0)  |(2, 0)
    # |________|________|
    #   (0, 0)   (1, 0)
    # Edge 1D indexing
    #  ________ ________
    # |   6    |   7    |
    # |14      |15      |16
    # |________|________|
    # |   4    |   5    |
    # |11      |12      |13
    # |________|________|
    # |   2    |   3    |
    # |8       |9       |10
    # |________|________|
    #     0        1
    # We select vertical axes
    edge_axis = 1
    # Third columns, second row
    edge_ndindex = jnp.array([2, 1])
    edge_1dindex = cglib.grid.edge.index1_from_2dindex(
        edge_ndindex, edge_axis, edge_2dcount_per_axis)
    # 13
    edge_ndindex_from_f, edge_axis_from_f = cglib.grid.edge.index2_from_1dindex(
        edge_1dindex, edge_2dcount_per_axis)
    res1 = jnp.all(edge_ndindex_from_f == edge_ndindex)
    res2 = jnp.all(edge_axis_from_f == edge_axis)
    return jnp.logical_and(res1, res2)


def test_index2_from_1dindex():
    assert jax.jit(helper_index2_from_1dindex)()


def helper_indices2_from_grid():

    cell_nd_count = jnp.array((2, 3))
    #  __ __
    # |__|__|
    # |__|__|
    # |__|__|
    res = cglib.grid.edge.indices2_from_grid(cell_nd_count)
    exp_res_0 = jnp.array([[0, 0],
                           [1, 0],
                           [0, 1],
                           [1, 1],
                           [0, 2],
                           [1, 2],
                           [0, 3],
                           [1, 3]])
    exp_res_1 = jnp.array([[0, 0],
                           [1, 0],
                           [2, 0],
                           [0, 1],
                           [1, 1],
                           [2, 1],
                           [0, 2],
                           [1, 2],
                           [2, 2]])
    exp_res = (exp_res_0, exp_res_1)
    return cglib.tree_util.all_equal(res, exp_res)


def test_indices2_from_grid():
    assert helper_indices2_from_grid()


def helper_neighboring_2dindices():

    # Grid definition
    cell_2dcount = jnp.array((2, 3))
    # Edge 2D indexing per axis
    #  ________ ________
    # | (0, 3) | (1, 3) |
    # |(0, 2)  |(1, 2)  |(2, 2)
    # |________|________|
    # | (0, 2) | (1, 2) |
    # |(0, 1)  |(1, 1)  |(2, 1)
    # |________|________|
    # | (0, 1) | (1, 1) |
    # |(0, 0)  |(1, 0)  |(2, 0)
    # |________|________|
    #   (0, 0)   (1, 0)
    # We select vertical axes
    edge_axis = 1
    # Third columns, second row
    edge_ndindex = jnp.array([2, 1])
    res = cglib.grid.edge.neighboring_2dindices(
        edge_ndindex, edge_axis, cell_2dcount, 1)
    exp_res = (jnp.array([[[1, 0],
                           [0, 0],
                           [0, 0],
                           [1, 1],
                           [0, 1],
                           [0, 1],
                           [1, 2],
                           [0, 2],
                           [0, 2]],
                          [[1, 0],
                           [2, 0],
                           [0, 0],
                           [1, 1],
                           [2, 1],
                           [0, 1],
                           [1, 2],
                           [2, 2],
                           [0, 2]]]),
               jnp.array(
        [[False, True, True, False, True, True, False, True, True],
         [False, False, True, False, True, True, False, False, True]]))
    return cglib.tree_util.all_equal(res, exp_res)


def test_neighboring_2dindices():
    assert jax.jit(helper_neighboring_2dindices)()


def helper_neighboring2_1dindices():

    cell_2dcount = jnp.array((2, 3))
    # Edge 1D indexing
    #  ________ ________
    # |   6    |   7    |
    # |14      |15      |16
    # |________|________|
    # |   4    |   5    |
    # |11      |12      |13
    # |________|________|
    # |   2    |   3    |
    # |8       |9       |10
    # |________|________|
    #     0        1
    res = cglib.grid.edge.neighboring2_1dindices(
        cglib.type.uint(13), cell_2dcount, 1)
    exp_res = (
        jnp.array(
            [1, 0, 0, 3, 2, 2, 5, 4, 4, 9, 10, 8, 12, 13, 11, 15, 16, 14]),
        jnp.array(
            [False, True, True, False, True, True, False, True, True,
             False, False, True, False, True, True, False, False, True]))
    return cglib.tree_util.all_equal(res, exp_res)


def test_neighboring2_1dindices():
    assert jax.jit(helper_neighboring2_1dindices)()


def helper_neighboring_2dindices_direct():

    # Grid definition
    cell_2dcount = jnp.array((2, 3))
    # Edge 2D indexing per axis
    #  ________ ________
    # | (0, 3) | (1, 3) |
    # |(0, 2)  |(1, 2)  |(2, 2)
    # |________|________|
    # | (0, 2) | (1, 2) |
    # |(0, 1)  |(1, 1)  |(2, 1)
    # |________|________|
    # | (0, 1) | (1, 1) |
    # |(0, 0)  |(1, 0)  |(2, 0)
    # |________|________|
    #   (0, 0)   (1, 0)
    # We select vertical axes
    edge_axis = 1
    # Third columns, second row
    edge_ndindex = jnp.array([2, 1])
    res_visible = cglib.grid.edge.neighboring_2dindices_direct(
        edge_ndindex,
        edge_axis,
        cell_2dcount,
        cglib.grid.edge.Neighboring2Type.VISIBLE)
    exp_res_visible = (
        jnp.array([[[1, 1],
                    [1, 2],
                    [0, 1],
                    [0, 2]],
                   [[1, 1],
                    [0, 1],
                    [0, 0],
                    [0, 0]]]),
        jnp.array([[False, False,  True,  True],
                   [False,  True,  True,  True]])
    )
    res_visible_bool = cglib.tree_util.all_equal(res_visible, exp_res_visible)
    res_wt_dist = cglib.grid.edge.neighboring_2dindices_direct(
        edge_ndindex,
        edge_axis,
        cell_2dcount,
        cglib.grid.edge.Neighboring2Type.WITHIN_CELL_SIDE_LENDTH)
    exp_res_wt_dist = (
        jnp.array([[[1, 1],
                    [1, 2],
                    [0, 1],
                    [0, 2]],
                   [[2, 0],
                    [2, 2],
                    [0, 0],
                    [0, 0]]]),
        jnp.array([[False, False,  True,  True],
                   [False, False,  True,  True]])
    )
    res_wt_dist_bool = cglib.tree_util.all_equal(res_wt_dist, exp_res_wt_dist)
    return res_visible_bool and res_wt_dist_bool


def test_neighboring_2dindices_direct():
    assert helper_neighboring_2dindices_direct()


def helper_endpoints():
    edge_ndindex = jnp.array([1, 2])
    edge_axis = 0
    cell_ndcount = jnp.array((2, 3))
    origin = jnp.array((-1., 2))
    cell_sides_length = 0.5
    grid = cglib.grid.Grid(cell_ndcount, origin, cell_sides_length)
    # Edge 2D indexing per axis
    #  ________ ________
    # | (0, 3) | (1, 3) |
    # |(0, 2)  |(1, 2)  |(2, 2)
    # |________|________|
    # | (0, 2) | (1, 2) |
    # |(0, 1)  |(1, 1)  |(2, 1)
    # |________|________|
    # | (0, 1) | (1, 1) |
    # |(0, 0)  |(1, 0)  |(2, 0)
    # |________|________|
    #   (0, 0)   (1, 0)
    res = cglib.grid.edge.endpoints(edge_ndindex, edge_axis, grid)
    exp_res = jnp.array([[-0.5, 3.],
                         [0., 3.]])
    return jnp.all(jnp.isclose(res, exp_res))


def test_endpoints():
    assert jax.jit(helper_endpoints)()


def test_boundary_1dindex_from_2dindex():
    grid_cell_2dcount = (3, 4)
    grid_boundary_edge_count = cglib.grid.edge.boundary_1dcount(
        grid_cell_2dcount)
    grid_boundary_edge_flattened_indices = jnp.arange(grid_boundary_edge_count)

    boundary_edges = jax.jit(
        jax.vmap(
            cglib.grid.edge.boundary_1d_to_2dindex,
            (0,
             None)),
        static_argnames='grid_cell_2dcount')(
        grid_boundary_edge_flattened_indices,
        grid_cell_2dcount)

    edge_boundary_2d_to_1dindex_v = jax.vmap(
        cglib.grid.edge.boundary_1dindex_from_2dindex,
        (0, 0, None))
    edge_boundary_2d_to_1dindex_v_jit = jax.jit(
        edge_boundary_2d_to_1dindex_v,
        static_argnames='grid_cell_2dcount')

    grid_boundary_edge_flattened_indices_2 = \
        edge_boundary_2d_to_1dindex_v_jit(
            boundary_edges[0],
            boundary_edges[1],
            grid_cell_2dcount)
    assert jnp.all(
        jnp.equal(
            grid_boundary_edge_flattened_indices,
            grid_boundary_edge_flattened_indices_2))
