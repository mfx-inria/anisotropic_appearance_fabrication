import jax
import jax.numpy as jnp

import cglib.cycle
import cglib.grid
import cglib.limits
import cglib.point_data
import cglib.polyline
import cglib.scalar
import cglib.tree_util
import cglib.type


def helper_create_from_graph():
    #            Input graph
    #       o p_1           o p_6
    #      / \             / \
    # p_2 o   o p_0   p_7 o---o p_5
    #      \ /
    #   p_3 o
    #
    P = jnp.array([[1., 0.],  # p_0
                   [0.5, 1.],  # p_1
                   [0., 0.],  # p_2
                   [0.5, 1.],  # p_3
                   [jnp.nan, 1.],  # p_4: masked point
                   [3., 0.],  # p_5
                   [2.5, 1.],  # p_6
                   [2., 0.]])  # p_7
    # Adjacency list per vertex
    data = jnp.array([[1, 3],
                      [0, 2],
                      [1, 3],
                      [0, 2],
                      [0, 0],
                      [7, 6],
                      [7, 5],
                      [6, 5]])
    graph = cglib.point_data.PointData(P, data)
    # Estimation of the number of cycles
    CYCLE_COUNT_MAX = graph.point.shape[0] // 4
    res = cglib.cycle.create_from_graph(graph, CYCLE_COUNT_MAX)
    exp_cycle_point_data = cglib.point_data.PointData(
        jnp.array([[1., 0.],
                   [0.5, 1.],
                   [0., 0.],
                   [0.5, 1.],
                   [jnp.nan, 1.],
                   [3., 0.],
                   [2.5, 1.],
                   [2., 0.]]),
        jnp.array([[1,          3,          0],
                   [2,          0,          0],
                   [3,          1,          0],
                   [0,          2,          0],
                   [4294967295, 4294967295, 4294967295],
                   [7,          6,          1],
                   [5,          7,          1],
                   [6,          5,          1]],
                  dtype=cglib.type.uint))
    exp_res = cglib.cycle.Cycle(
        exp_cycle_point_data,
        jnp.array([[0, 4],
                   [5, 3]],
                  dtype=cglib.type.uint),
        jnp.array(2, dtype=cglib.type.uint))
    return cglib.tree_util.all_isclose_masked(res, exp_res)


def test_create_from_graph():
    assert jax.jit(helper_create_from_graph)()


def helper_point_tangent_half_distance_to_edge():
    #            Input graph
    #       o p_1           o p_5
    #      / \             / \
    # p_2 o   o p_0   p_6 o---o p_4
    #      \ /
    #   p_3 o
    #
    P = jnp.array([[1., 0.],  # p_0
                   [0.5, 1.],  # p_1
                   [0., 0.],  # p_2
                   [0.5, 1.],  # p_3
                   [3., 0.],  # p_4
                   [2.5, 1.],  # p_5
                   [2., 0.]])  # p_6
    # Adjacency list per vertex
    data = jnp.array([[1, 3],
                      [0, 2],
                      [1, 3],
                      [0, 2],
                      [6, 5],
                      [6, 4],
                      [5, 4]])
    graph = cglib.point_data.PointData(P, data)
    cycle = cglib.cycle.create_from_graph(graph, 2)
    point_index = 0
    edge_index = 6
    res1 = cglib.cycle.point_tangent_half_distance_to_segment(
        point_index,
        edge_index,
        cycle)
    exp_res1 = 0.55901694
    edge_index = 2
    res2 = cglib.cycle.point_tangent_half_distance_to_segment(
        point_index,
        edge_index,
        cycle) == cglib.limits.FLOAT_MAX
    # True because vertex 0 and 2 share edges with the same endpoints,
    # i.e., vertex 1.
    return jnp.logical_and(jnp.isclose(res1, exp_res1), res2)


def test_point_tangent_half_distance_to_edge():
    assert helper_point_tangent_half_distance_to_edge()


def test_point_tangent_half_distance_to_neighboring_segments_x():
    # Input scalar field's param initialization
    scalar_field_cell_2dcount = (4, 4)
    cell_2dcount = jnp.array(scalar_field_cell_2dcount)
    origin = jnp.array([0.0, 0.0])
    cell_sides_length = 1.
    scalar_field_param = cglib.grid.Grid(
        cell_2dcount, origin, cell_sides_length)
    contour_2dgrid = cglib.grid.Grid(
        cell_2dcount - 1, origin + 0.5 * cell_sides_length, cell_sides_length)
    # Associate scalars to the vertices
    grid_scalars_flattened = jnp.array(
        [-1., -1., -1., -1., -1., 1., 1., -1.,
         -1., 1., 1., -1., -1., -1., -1., -1.])
    # autopep8 : off
    #            Input scalar field                     Ouput contour
    #   ________ ________ ________ ________       and point/edge 1-D indices
    #  |        |        |        |        |      ________ ________ ________
    #  |  -1.   |  -1.   |  -1.   |  -1.   |     |  9     |21 10   |22  11  |
    #  |________|________|________|________|     |20   /--o--------o--\     |23
    #  |        |        |        |        |    ||____o___|________|___o____|
    #  |  -1.   |   1.   |   1.   |  -1.   |     |  6 |   |    7   |   | 8  |
    #  |________|________|________|________|    ||16  |   |17      |18 |    |19
    #  |        |        |        |        | 1.5.|____o___|________|___o____|
    #  |  -1.   |   1.   |   1.   |  -1.   |    ||  3  \  |13  4   |14/  5  |
    #  |________|________|________|________|     |12    --o--------o--      |15
    #  |        |        |        |        | 0.5.|________|________|________|
    #  |  -1.   |  -1.   |  -1.   |  -1.   |          0        1        2
    #  x________|________|________|________|    |
    #  x -> origin == (0., 0.)                  x.0.5 .1.
    # autopep8 : on
    contour_graph = cglib.scalar.grid2_contour(
        grid_scalars_flattened, scalar_field_cell_2dcount, scalar_field_param)
    contour_cycle = cglib.cycle.create_from_graph(contour_graph, 1)
    neighbor_radius = cglib.type.uint(3)
    point_index = cglib.type.uint(3)
    res = cglib.cycle.point_tangent_half_distance_to_neighboring_segments_x(
        contour_cycle, point_index, neighbor_radius, contour_2dgrid)
    exp_res = 1.0059223
    assert jnp.isclose(res, exp_res)


def test_points_tangent_half_distance_to_neighboring_segments_x():
    # Input scalar field's param initialization
    scalar_field_cell_2dcount = (4, 4)
    cell_2dcount = jnp.array(scalar_field_cell_2dcount)
    origin = jnp.array([0.0, 0.0])
    cell_sides_length = 1.
    scalar_field_param = cglib.grid.Grid(
        cell_2dcount, origin, cell_sides_length)
    contour_2dgrid = cglib.grid.Grid(
        cell_2dcount + 1, origin + 0.5 * cell_sides_length, cell_sides_length)
    # Associate scalars to the vertices
    grid_scalars_flattened = jnp.array(
        [-1., -1., -1., -1., -1., 1., 1., -1.,
         -1., 1., 1., -1., -1., -1., -1., -1.])
    # autopep8 : off
    #           Input scalar field                      Ouput contour
    #  ________ ________ ________ ________        and point/edge 1-D indices
    # |        |        |        |        |       ________ ________ ________
    # |  -1.   |  -1.   |  -1.   |  -1.   |      |  9     |21 10   |22  11  |
    # |________|________|________|________|      |20   /--o--------o--\     |23
    # |        |        |        |        |    | |____o___|________|___o____|
    # |  -1.   |   1.   |   1.   |  -1.   |      |  6 |   |    7   |   | 8  |
    # |________|________|________|________|    | |16  |   |17      |18 |    |19
    # |        |        |        |        | 1.5. |____o___|________|___o____|
    # |  -1.   |   1.   |   1.   |  -1.   |    | |  3  \  |13  4   |14/  5  |
    # |________|________|________|________|      |12    --o--------o--      |15
    # |        |        |        |        | 0.5. |________|________|________|
    # |  -1.   |  -1.   |  -1.   |  -1.   |           0        1        2
    # x________|________|________|________|    |
    # x -> origin == (0., 0.)                  x .0.5 .1.
    # autopep8 : on
    contour_graph = cglib.scalar.grid2_contour(
        grid_scalars_flattened, scalar_field_cell_2dcount, scalar_field_param)
    contour_cycle = cglib.cycle.create_from_graph(contour_graph, 1)
    neighbor_radius = cglib.type.uint(3)
    res = cglib.cycle.points_tangent_half_distance_to_neighboring_segments_x(
        contour_cycle, neighbor_radius, contour_2dgrid)
    exp_res = [3.4028235e+38, 3.4028235e+38, 3.4028235e+38, 1.0059223e+00,
               3.4028235e+38, 1.0059223e+00, 1.0059223e+00, 3.4028235e+38,
               1.0059223e+00, 3.4028235e+38, 3.4028235e+38, 3.4028235e+38,
               3.4028235e+38, 1.0059223e+00, 1.0059223e+00, 3.4028235e+38,
               3.4028235e+38, 3.4028235e+38, 3.4028235e+38, 3.4028235e+38,
               3.4028235e+38, 1.0059223e+00, 1.0059223e+00, 3.4028235e+38]
    exp_res = jnp.array(exp_res)
    assert jnp.all(jnp.isclose(res, exp_res))


def test_to_polyline():
    #            Input graph
    #       o p_1           o p_6
    #      / \             / \
    # p_2 o   o p_0   p_7 o---o p_5
    #      \ /
    #   p_3 o
    #
    P = jnp.array([[1., 0.],  # p_0
                   [0.5, 1.],  # p_1
                   [0., 0.],  # p_2
                   [0.5, 1.],  # p_3
                   [jnp.nan, 1.],  # p_4: masked point
                   [3., 0.],  # p_5
                   [2.5, 1.],  # p_6
                   [2., 0.]])  # p_7
    # Adjacency list per vertex
    data = jnp.array([[1, 3],
                      [0, 2],
                      [1, 3],
                      [0, 2],
                      [0, 0],
                      [7, 6],
                      [7, 5],
                      [6, 5]])
    graph = cglib.point_data.PointData(P, data)
    cycle = cglib.cycle.create_from_graph(graph, 2)
    polyline = cglib.cycle.to_polyline_full_nan(cycle)
    polyline = cglib.cycle.to_polyline(cycle, polyline)
    exp_poly = cglib.polyline.Polyline(
        jnp.array([[[3., 0.],
                    [2., 0.],
                    [2.5, 1.],
                    [jnp.nan, jnp.nan],
                    [jnp.nan, jnp.nan],
                    [jnp.nan, jnp.nan],
                    [jnp.nan, jnp.nan],
                    [jnp.nan, jnp.nan]],
                   [[1., 0.],
                    [0.5, 1.],
                    [0., 0.],
                    [0.5, 1.],
                    [jnp.nan, jnp.nan],
                    [jnp.nan, jnp.nan],
                    [jnp.nan, jnp.nan],
                    [jnp.nan, jnp.nan]]]),
        jnp.array([[[7.,  6.,  1.],
                    [6.,  5.,  1.],
                    [5.,  7.,  1.],
                    [jnp.nan, jnp.nan, jnp.nan],
                    [jnp.nan, jnp.nan, jnp.nan],
                    [jnp.nan, jnp.nan, jnp.nan],
                    [jnp.nan, jnp.nan, jnp.nan],
                    [jnp.nan, jnp.nan, jnp.nan]],
                   [[1.,  3.,  0.],
                    [2.,  0.,  0.],
                    [3.,  1.,  0.],
                    [0.,  2.,  0.],
                    [jnp.nan, jnp.nan, jnp.nan],
                    [jnp.nan, jnp.nan, jnp.nan],
                    [jnp.nan, jnp.nan, jnp.nan],
                    [jnp.nan, jnp.nan, jnp.nan]]]),
        jnp.array([[5., 3.],
                   [0., 4.]])
    )
    assert cglib.tree_util.all_isclose_masked(polyline, exp_poly)


def helper_edge_minimize_patching_energy_wrt_segments():
    #            Input graph
    #       o p_1           o p_6
    #      / \             / \
    # p_2 o   o p_0   p_7 o---o p_5
    #      \ /
    #   p_3 o
    #
    P = jnp.array([[1., 0.],  # p_0
                   [0.5, 1.],  # p_1
                   [0., 0.],  # p_2
                   [0.5, 1.],  # p_3
                   [jnp.nan, 1.],  # p_4: masked point
                   [3., 0.],  # p_5
                   [2.5, 1.],  # p_6
                   [2., 0.]])  # p_7
    P_count = P.shape[0]
    # Adjacency list per vertex
    data = jnp.array([[1, 3],
                      [2, 0],
                      [3, 1],
                      [0, 2],
                      [0, 0],
                      [7, 6],
                      [5, 7],
                      [6, 5]])
    graph = cglib.point_data.PointData(P, data)
    cycle = cglib.cycle.create_from_graph(graph, 2)
    edge_i1 = 0
    graph_edge_indices = jnp.arange(P_count)
    graph_edge_indices_mask = jnp.full((P_count,), False)
    patching_energy, argmin_j1 = \
        cglib.cycle.edge_minimize_patching_energy_wrt_segments(
            edge_i1, graph_edge_indices, graph_edge_indices_mask, cycle)
    # For segment (0, 1), the best edge to patch and stitch is (7, 6)
    # This couple has patching energy 0.7639322280883789
    res1 = argmin_j1 == 7
    res2 = jnp.isclose(patching_energy, 0.7639322280883789)
    return jnp.logical_and(res1, res2)


def test_edge_minimize_patching_energy_wrt_segments():
    assert jax.jit(helper_edge_minimize_patching_energy_wrt_segments)()


def helper_neighbor_edge_with_minimum_patching_energy():
    # Input scalar field's param initialization
    scalar_field_cell_2dcount = (8, 4)
    scalar_grid_cell_2dcount = jnp.array(scalar_field_cell_2dcount)
    origin = jnp.array([0.0, 0.0])
    cell_sides_length = 1.
    scalar_field_param = cglib.grid.Grid(
        scalar_grid_cell_2dcount, origin, cell_sides_length)
    # Associate scalars to the vertices
    grid_scalars_flattened = jnp.array(
        [-1., -1., -1., -1., -1., 1., 1., -1.,
         -1., 1., 1., -1., -1., -1., -1., -1.])
    grid_scalars_shaped = jnp.reshape(
        grid_scalars_flattened,
        (scalar_field_cell_2dcount[0] // 2, scalar_field_cell_2dcount[1]))
    tiled_grid_scalars_shaped: jnp.ndarray = jnp.concatenate(
        (grid_scalars_shaped, grid_scalars_shaped),
        axis=1)
    tiled_grid_scalars_flattened = tiled_grid_scalars_shaped.ravel()
    # autopep8 : off
    #                             Ouput contour
    #                       and point/edge 1-D indices
    #        ________ ________ ________ ________ ________ ________ ________
    #       |  21    |45 22   |46  23  |47 24   |48 25   |49  26  |50  27  |51
    #       |44   /--o---<----o--\     |        |     /--o--<-----o--\     |
    #    |  |____o___|________|___o____|________|____o___|________|___o____|
    #       |  14|   |  15    |   |16  |   17   |  18|   |  19    |   |20  |
    #    |  |36  |   |37      |38 ^    |37      |40  |   |41      |42 ^    |43
    # 1.5.  |____o___|________|___o____|________|____o___|________|___o____|
    #    |  |28 7 \  |29 8    |30/ 9   |31 10   |  11 \  |33 12   |34/ 13  |
    #       |      --o--->----o--      |        |32    --o--->----o--      |35
    # 0.5.  |________|________|________|________|________|________|________|
    #           0        1        2        3        4        5        6
    #    |
    #    x  .0.5 .1.
    # (0., 0.)
    # autopep8 : on
    contour_grid_cell_2dcount = scalar_grid_cell_2dcount - 1
    contour_graph = cglib.scalar.grid2_contour(
        tiled_grid_scalars_flattened,
        scalar_field_cell_2dcount,
        scalar_field_param)
    contour_cycle = cglib.cycle.create_from_graph(contour_graph, 2)
    cycle_id = 0
    res = cglib.cycle.neighboring_edge_with_minimum_patching_energy(
        cycle_id, contour_cycle, contour_grid_cell_2dcount)
    exp_res = (
        jnp.array([9, 18], dtype=cglib.type.uint),
        jnp.array(2.))
    return cglib.tree_util.all_isclose_masked(res, exp_res)


def test_neighbor_edge_with_minimum_patching_energy():
    assert jax.jit(helper_neighbor_edge_with_minimum_patching_energy)()


def helper_edge_with_minimum_patching_energy():
    # Input scalar field's param initialization
    scalar_field_cell_2dcount = (8, 4)
    scalar_grid_cell_2dcount = jnp.array(scalar_field_cell_2dcount)
    origin = jnp.array([0.0, 0.0])
    cell_sides_length = 1.
    scalar_field_param = cglib.grid.Grid(
        scalar_grid_cell_2dcount, origin, cell_sides_length)
    # Associate scalars to the vertices
    grid_scalars_flattened = jnp.array(
        [-1., -1., -1., -1., -1., 1., 1., -1.,
         -1., 1., 1., -1., -1., -1., -1., -1.])
    grid_scalars_shaped = jnp.reshape(
        grid_scalars_flattened,
        (scalar_field_cell_2dcount[0] // 2, scalar_field_cell_2dcount[1]))
    tiled_grid_scalars_shaped: jnp.ndarray = jnp.concatenate(
        (grid_scalars_shaped, grid_scalars_shaped),
        axis=1)
    tiled_grid_scalars_flattened = tiled_grid_scalars_shaped.ravel()
    # autopep8 : off
    #                             Ouput contour
    #                       and point/edge 1-D indices
    #        ________ ________ ________ ________ ________ ________ ________
    #       |  21    |45 22   |46  23  |47 24   |48 25   |49  26  |50  27  |51
    #       |44   /--o---<----o--\     |        |     /--o--<-----o--\     |
    #    |  |____o___|________|___o____|________|____o___|________|___o____|
    #       |  14|   |  15    |   |16  |   17   |  18|   |  19    |   |20  |
    #    |  |36  |   |37      |38 ^    |37      |40  |   |41      |42 ^    |43
    # 1.5.  |____o___|________|___o____|________|____o___|________|___o____|
    #    |  |28 7 \  |29 8    |30/ 9   |31 10   |  11 \  |33 12   |34/ 13  |
    #       |      --o--->----o--      |        |32    --o--->----o--      |35
    # 0.5.  |________|________|________|________|________|________|________|
    #           0        1        2        3        4        5        6
    #    |
    #    x  .0.5 .1.
    # (0., 0.)
    # autopep8 : on
    contour_graph = cglib.scalar.grid2_contour(
        tiled_grid_scalars_flattened,
        scalar_field_cell_2dcount,
        scalar_field_param)
    contour_cycles = cglib.cycle.create_from_graph(
        contour_graph, contour_graph.point.shape[0] // 4)
    cycle_id = 0
    res = cglib.cycle.edge_with_minimum_patching_energy(
        cycle_id, contour_cycles)
    exp_res = (jnp.array([9, 18], dtype=cglib.type.uint),
               jnp.array(2.))
    return cglib.tree_util.all_isclose_masked(res, exp_res)


def test_edge_with_minimum_patching_energy():
    assert jax.jit(helper_edge_with_minimum_patching_energy)()


def helper_stitch_two_edges():
    # Input scalar field's param initialization
    scalar_field_cell_2dcount = (8, 4)
    scalar_grid_cell_2dcount = jnp.array(scalar_field_cell_2dcount)
    origin = jnp.array([0.0, 0.0])
    cell_sides_length = 1.
    scalar_field_param = cglib.grid.Grid(
        scalar_grid_cell_2dcount, origin, cell_sides_length)
    # Associate scalars to the vertices
    grid_scalars_flattened = jnp.array(
        [-1., -1., -1., -1., -1., 1., 1., -1.,
         -1., 1., 1., -1., -1., -1., -1., -1.])
    grid_scalars_shaped = jnp.reshape(
        grid_scalars_flattened,
        (scalar_field_cell_2dcount[0] // 2, scalar_field_cell_2dcount[1]))
    tiled_grid_scalars_shaped: jnp.ndarray = jnp.concatenate(
        (grid_scalars_shaped, grid_scalars_shaped),
        axis=1)
    tiled_grid_scalars_flattened = tiled_grid_scalars_shaped.ravel()
    # autopep8 : off
    #                             Ouput contour
    #                       and point/edge 1-D indices
    #        ________ ________ ________ ________ ________ ________ ________
    #       |  21    |45 22   |46  23  |47 24   |48 25   |49  26  |50  27  |51
    #       |44   /--o---<----o--\     |        |     /--o--<-----o--\     |
    #    |  |____o___|________|___o____|________|____o___|________|___o____|
    #       |  14|   |  15    |   |16  |   17   |  18|   |  19    |   |20  |
    #    |  |36  |   |37      |38 ^    |37      |40  |   |41      |42 ^    |43
    # 1.5.  |____o___|________|___o____|________|____o___|________|___o____|
    #    |  |28 7 \  |29 8    |30/ 9   |31 10   |  11 \  |33 12   |34/ 13  |
    #       |      --o--->----o--      |        |32    --o--->----o--      |35
    # 0.5.  |________|________|________|________|________|________|________|
    #           0        1        2        3        4        5        6
    #    |
    #    x  .0.5 .1.
    # (0., 0.)
    # autopep8 : on
    contour_graph = cglib.scalar.grid2_contour(
        tiled_grid_scalars_flattened,
        scalar_field_cell_2dcount,
        scalar_field_param)
    contour_cycles = cglib.cycle.create_from_graph(contour_graph, 2)
    cycle_id = 0
    best_edge_pair, _ = cglib.cycle.edge_with_minimum_patching_energy(
        cycle_id, contour_cycles)
    # (DeviceArray([ 9, 18], dtype=uint32), DeviceArray(2., dtype=float32))
    cycle_after_stitching = cglib.cycle.stitch_two_edges(
        best_edge_pair[0], best_edge_pair[1], contour_cycles)
    # autopep8 : off
    #                             Cycle after stitching
    #        ________ ________ ________ ________ ________ ________ ________
    #       |  21    |45 22   |46  23  |47 24   |48 25   |49  26  |50  27  |51
    #       |44   /--o---<----o--\     |        |     /--o--<-----o--\     |
    #    |  |____o___|________|___o-<--|---<----|-<--o___|________|___o____|
    #       |  14|   |  15    |    16  |   17   |  18    |  19    |   |20  |
    #    |  |36  |   |37      |38      |37      |40      |41      |42 ^    |43
    # 1.5.  |____o___|________|___o-->-|---->---|->--o___|________|___o____|
    #    |  |28 7 \  |29 8    |30/ 9   |   10   |  11 \  |33 12   |34/ 13  |
    #       |      --o--->----o--      |31      |32    --o--->----o--      |35
    # 0.5.  |________|________|________|________|________|________|________|
    #           0        1        2        3        4        5        6
    #    |
    #    x  .0.5 .1.
    # (0., 0.)
    # autopep8 : on
    exp_res = cglib.cycle.Cycle(
        cglib.point_data.PointData(point=jnp.array(
            [[jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [1., 1.5],
             [jnp.nan, jnp.nan],
             [3., 1.5],
             [jnp.nan, jnp.nan],
             [5., 1.5],
             [jnp.nan, jnp.nan],
             [7., 1.5],
             [1., 2.5],
             [jnp.nan, jnp.nan],
             [3., 2.5],
             [jnp.nan, jnp.nan],
             [5., 2.5],
             [jnp.nan, jnp.nan],
             [7., 2.5],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [1.5, 1.],
             [2.5, 1.],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [5.5, 1.],
             [6.5, 1.],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [1.5, 3.],
             [2.5, 3.],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [5.5, 3.],
             [6.5, 3.],
             [jnp.nan, jnp.nan]]),
            data=jnp.array(
            [[4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [29,         14,          1],
             [4294967295, 4294967295, 4294967295],
             [11,         30,          1],
             [4294967295, 4294967295, 4294967295],
             [33,          9,          1],
             [4294967295, 4294967295, 4294967295],
             [20,         34,          1],
             [7,         45,          1],
             [4294967295, 4294967295, 4294967295],
             [46,         18,          1],
             [4294967295, 4294967295, 4294967295],
             [16,         49,          1],
             [4294967295, 4294967295, 4294967295],
             [50,         13,          1],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [30,          7,          1],
             [9,         29,          1],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [34,         11,          1],
             [13,         33,          1],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [14,         46,          1],
             [45,         16,          1],
             [4294967295, 4294967295, 4294967295],
             [4294967295, 4294967295, 4294967295],
             [18,         50,          1],
             [49,         20,          1],
             [4294967295, 4294967295, 4294967295]], dtype=cglib.type.uint)),
        cycle_data=jnp.array(
            [[4294967295, 4294967295],
             [11,         16]], dtype=cglib.type.uint),
        cycle_count=jnp.array(1, dtype=cglib.type.uint))
    return cglib.tree_util.all_isclose_masked(exp_res, cycle_after_stitching)


def test_stitch_two_edges():
    assert jax.jit(helper_stitch_two_edges)()
