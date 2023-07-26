import jax
import jax.numpy as jnp

import cglib.grid
import cglib.point_data
import cglib.tree_util
import cglib.scalar


def helper_grid_get():
    grid_cell_2dcount = jnp.array([2, 3])
    grid_origin = jnp.array([-1., 2.])
    grid_cell_sides_length = 0.5
    grid = cglib.grid.Grid(
        grid_cell_2dcount, grid_origin, grid_cell_sides_length)
    # Point associated with each cell of the grid
    point = jnp.array([[0.,  1.],
                       [2.,  3.],
                       [4.,  5.],
                       [6.,  7.],
                       [8.,  9.],
                       [10., 11.]])
    # Three floats per point
    data = jnp.array([[12., 13., 14.],
                      [15., 16., 17.],
                      [18., 19., 20.],
                      [21., 22., 23.],
                      [24., 25., 26.],
                      [27., 28., 29.]])
    point_data = cglib.point_data.PointData(point, data)
    grid_point_data = cglib.point_data.GridPointData(point_data, grid)
    cell_ndindex = jnp.array([[0, 1],
                              [1, 2]])
    res = cglib.point_data.grid_get(cell_ndindex, grid_point_data)
    res_exp = cglib.point_data.PointData(
        point=jnp.array(
            [[4.,  5.],
             [10., 11.]]),
        data=jnp.array(
            [[18., 19., 20.],
             [27., 28., 29.]]))
    return cglib.tree_util.all_isclose(res, res_exp)


def test_grid_get():
    assert jax.jit(helper_grid_get)()


def helper_all_isclose_masked():
    grid_cell_2dcount = jnp.array([2, 3])
    grid_origin = jnp.array([-1., 2.])
    grid_cell_sides_length = 0.5
    grid = cglib.grid.Grid(
        grid_cell_2dcount, grid_origin, grid_cell_sides_length)
    # Point associated with each cell of the grid
    point = jnp.array([[0.,  1.],
                       [2.,  3.],
                       [4.,  5.],
                       [6.,  7.],
                       [8.,  9.],
                       [10., 11.]])
    # Three floats per point
    data = jnp.array([[12., 13., 14.],
                      [15., 16., 17.],
                      [18., 19., 20.],
                      [21., 22., 23.],
                      [24., 25., 26.],
                      [27., 28., 29.]])
    point_data = cglib.point_data.PointData(point, data)
    grid_point_data = cglib.point_data.GridPointData(point_data, grid)
    p = grid_origin
    res = cglib.point_data.grid_neighborhood_from_point(p, grid_point_data)
    res_exp = cglib.point_data.PointData(
        point=jnp.array(
            [[jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [0.,  1.],
             [2.,  3.],
             [jnp.nan, jnp.nan],
             [4.,  5.],
             [6.,  7.]]),
        data=jnp.array(
            [[jnp.nan, jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan, jnp.nan],
             [12., 13., 14.],
             [15., 16., 17.],
             [jnp.nan, jnp.nan, jnp.nan],
             [18., 19., 20.],
             [21., 22., 23.]]))
    return cglib.tree_util.all_isclose_masked(res, res_exp)


def test_all_isclose_masked():
    assert jax.jit(helper_all_isclose_masked)()


def helper_grid2_repulse_point_from_neighbors():
    # Input scalar field's param initialization
    scalar_field_cell_2dcount = (3, 4)
    cell_2dcount = jnp.array(scalar_field_cell_2dcount)
    origin = jnp.array([0.0, 0.0])
    cell_sides_length = 1.
    scalar_field_param = cglib.grid.Grid(
        cell_2dcount, origin, cell_sides_length)
    # Associate scalars to the vertices
    grid_scalars_flattened = jnp.array(
        [-1., -1., -1., -1., 0.01, -1.,
         -1., 0.01, -1., -1., -1., -1.])
    # autopep8 : off
    #       Input scalar field                   Ouput contour
    #   ________ ________ ________         and point/edge 1-D indices
    #  |        |        |        |            ________ ________
    #  |  -1.   |  -1.   |  -1.   |           |  6     |21  7   |
    #  |________|________|________|           |14      |15      |16
    #  |        |        |        |        |  |_______ooo_______|
    #  |  -1.   |  0.01  |  -1.   |           |  4    |||   5   |
    #  |________|________|________|        |  |11     |||12     |13
    #  |        |        |        |     1.5.  |_______o|o_______|
    #  |  -1.   |  0.01  |  -1.   |        |  |  2    \o/  3    |
    #  |________|________|________|           |8       |9       |10
    #  |        |        |        |     0.5.  |________|________|
    #  |  -1.   |  -1.   |  -1.   |                0        1
    #  x________|________|________|        |
    #  x -> origin == (0., 0.)             x  . - - - -. - - - - -
    #                                        0.5      1.5
    # autopep8 : on
    contour_graph = cglib.scalar.grid2_contour(
        grid_scalars_flattened,
        scalar_field_cell_2dcount,
        scalar_field_param)
    contour_2dgrid = cglib.grid.Grid(
        cell_2dcount - 1,
        origin + 0.5 * cell_sides_length,
        cell_sides_length)
    edge_2dindex = jnp.array([0, 1])
    edge_axis = 0
    res = cglib.point_data.grid2_repulse_point_from_neighbors(
        edge_2dindex,
        edge_axis,
        contour_graph,
        (contour_2dgrid.cell_ndcount[0], contour_2dgrid.cell_ndcount[1]),
        contour_2dgrid,
        cell_sides_length)
    res_exp = cglib.point_data.PointData(
        point=jnp.array([0.9999999, 1.5]),
        data=jnp.array([9., 4.]))
    return cglib.tree_util.all_isclose(res, res_exp)


def test_grid2_repulse_point_from_neighbors():
    assert jax.jit(helper_grid2_repulse_point_from_neighbors)()


def helper_grid2_repulse_points():
    with jax.ensure_compile_time_eval():
        # Input scalar field's param initialization
        scalar_field_cell_2dcount = (3, 4)
        cell_2dcount = jnp.array(scalar_field_cell_2dcount)
        origin = jnp.array([0.0, 0.0])
        cell_sides_length = 1.
        scalar_field_param = cglib.grid.Grid(
            cell_2dcount, origin, cell_sides_length)
        contour_2dgrid = cglib.grid.Grid(
            cell_2dcount - 1,
            origin + 0.5 * cell_sides_length,
            cell_sides_length)
        contour_grid_cell_2dcount = (
            contour_2dgrid.cell_ndcount[0], contour_2dgrid.cell_ndcount[1])
    # Associate scalars to the vertices
    grid_scalars_flattened = jnp.array(
        [-1., -1., -1., -1., 0.01, -1., -1., 0.01, -1., -1., -1., -1.])
    # autopep8 : off
    #           Input contour              After repulsion
    #     and point/edge 1D indices
    #         ________ ________           ________ ________
    #        |  6     |21  7   |         |  6     |21  7   |
    #        |14      |15      |16       |14      |15      |16
    #     |  |_______ooo_______|         |____o---o---o____|
    #        |  4    |||   5   |         |  4 |   |   |5   |
    #     |  |11     |||12     |13       |11  |   | 12|    |13
    #  1.5.  |_______o|o_______|         |____o---o---o____|
    #     |  |  2    \o/  3    |         |  2     |  3     |
    #        |8       |9       |10       |8       |9       |10
    #  0.5.  |________|________|         |________|________|
    #             0        1                  0        1
    #     |
    #     x  . - - - -. - - - - -        x  . - - - -. - - - -
    #       0.5      1.5                   0.5      1.5
    # autopep8 : on
    contour_graph = cglib.scalar.grid2_contour(
        grid_scalars_flattened,
        scalar_field_cell_2dcount,
        scalar_field_param)

    objective_distance = cell_sides_length
    constraint = jnp.full(contour_graph.point.shape[0], False)
    res = cglib.point_data.grid2_repulse_points(
        contour_graph,
        contour_grid_cell_2dcount,
        contour_2dgrid,
        objective_distance,
        constraint)
    res_exp = cglib.point_data.PointData(
        point=jnp.array(
            [[jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [0.9999999, 1.5],
             [2., 1.5],
             [0.9999999, 2.5],
             [2.0000002, 2.5],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [1.5, 1.49],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [1.5, 2.51],
             [jnp.nan, jnp.nan]]),
        data=jnp.array(
            [[jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [9., 4.],
             [9., 5.],
             [2., 15.],
             [3., 15.],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [2., 3.],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [4.,  5.],
             [jnp.nan, jnp.nan]]))
    return cglib.tree_util.all_isclose_masked(res, res_exp)


def test_grid2_repulse_points():
    assert jax.jit(helper_grid2_repulse_points)()
