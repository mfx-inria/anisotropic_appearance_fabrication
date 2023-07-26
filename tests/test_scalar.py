import jax
import jax.numpy as jnp

import cglib.grid
import cglib.scalar
import cglib.point_data
import cglib.tree_util


def helper_grid_edge_point_scalars():
    with jax.ensure_compile_time_eval():
        # Grid definition
        cell_ndcount = jnp.array((2, 3))
        cell_1dcount = jnp.prod(cell_ndcount)
        grid_scalars_flattened: jnp.ndarray = jnp.arange(cell_1dcount)
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
        # Visualize the shaped grid with the bottom left origin (outer flip).
        # The first index is for x, and the second is for y (inner flip).
        grid_scalars = jnp.flip(grid_scalars_flattened.reshape(
            jnp.flip(cell_ndcount)), axis=0)
    grid_scalars_exp = jnp.array([[4, 5],
                                  [2, 3],
                                  [0, 1]])
    res0 = jnp.all(jnp.equal(grid_scalars, grid_scalars_exp))

    # We select vertical axis
    edge_axis = 1
    # Second column, first row
    edge_ndindex = jnp.array([1, 0])
    res1 = cglib.scalar.grid_edge_point_scalars(
        edge_ndindex, edge_axis, grid_scalars_flattened, cell_ndcount)
    res1_exp = jnp.array([1, 3])
    res1_bool = jnp.all(jnp.equal(res1, res1_exp))
    return jnp.logical_and(res0, res1_bool)


def test_grid_edge_point_scalars():
    assert jax.jit(helper_grid_edge_point_scalars)()


def helper_grid_edge_root_existence():
    with jax.ensure_compile_time_eval():
        # Grid definition
        cell_ndcount = jnp.array((2, 3))
        origin = jnp.array([0.0, 0.0])
        cell_sides_length = 1.
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
        # Associate scalars to the vertices
        grid_scalars_flattened = jnp.array([-0.5, 0.5, -0.25, 0.75, -0.1, 0.9])

    # Visualize the shaped grid of scalars with the bottom left origin (outer
    # flip). The first index is for x, and the second is for y (inner flip).
    # grid_scalars = jnp.flip(grid_scalars_flattened.reshape(
    #     jnp.flip(cell_ndcount)), axis=0)
    # [[-0.1   0.9 ]
    #  [-0.25  0.75]
    #  [-0.5   0.5 ]]

    # We select horizontal axes
    edge_axis = 0
    # First column, first row
    edge_ndindex = jnp.array([0, 0])
    res0 = cglib.scalar.grid_edge_root_existence(
        edge_ndindex, edge_axis, grid_scalars_flattened, grid)
    res0_bool = res0 == True
    # We select vertical axes
    edge_axis = 1
    # Second column, second row
    edge_ndindex = jnp.array([1, 1])
    res1 = cglib.scalar.grid_edge_root_existence(
        edge_ndindex, edge_axis, grid_scalars_flattened, grid)
    res1_bool = res1 == False
    return jnp.logical_and(res0_bool, res1_bool)


def test_grid_edge_root_existence():
    assert jax.jit(helper_grid_edge_root_existence)()


def helper_grid_edge_root_point():
    # Grid definition
    cell_ndcount = jnp.array((2, 3))
    origin = jnp.array([0.0, 0.0])
    cell_sides_length = 1.
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
    # Associate scalars to the vertices
    grid_scalars_flattened: jnp.ndarray = jnp.array(
        [-0.5, 0.5, -0.25, 0.75, -0.1, 0.9])

    # Visualize the shaped grid of scalars with the bottom left origin (outer
    # flip). The first index is for x, and the second is for y (inner flip).
    # grid_scalars = jnp.flip(
    #     grid_scalars_flattened.reshape(
    #     jnp.flip(cell_ndcount)), axis=0)
    # [[-0.1   0.9 ]
    #  [-0.25  0.75]
    #  [-0.5   0.5 ]]

    # We select horizontal axis
    edge_axis = 0
    # First column, first row
    edge_ndindex = jnp.array([0, 0])
    res = cglib.scalar.grid_edge_root_point(
        edge_ndindex, edge_axis, grid_scalars_flattened, grid)
    res_exp = jnp.array([0.5, 0.])
    return jnp.all(jnp.isclose(res, res_exp))


def test_grid_edge_root_point():
    assert jax.jit(helper_grid_edge_root_point)()


def helper_grid2_contour():
    # Input scalar field's param initialization
    scalar_field_cell_2dcount = (2, 3)
    cell_2dcount = jnp.array(scalar_field_cell_2dcount)
    origin = jnp.array([0.0, 0.0])
    cell_sides_length = 1.
    scalar_field_param = cglib.grid.Grid(
        cell_2dcount, origin, cell_sides_length)
    # Associate scalars to the vertices
    grid_scalars_flattened = jnp.array([-0.5, 0.5, -0.25, 0.75, -0.1, 0.9])
    #   Input scalar field          Ouput contour and
    #   ________ ________           1D edge indexing
    #  |        |        |        |   o_______
    #  |  -0.1  |   0.9  |           | \      |
    #  |________|________|        |  |5 \     |6
    #  |        |        |           |___o____|
    #  | -0.25  |  0.75  |        |  |   | 1  |
    #  |________|________|           |3  \    |4
    #  |        |        |     0.5.  |____o___|
    #  |  -0.5  |   0.5  |                 0
    #  x________|________|        |
    #  x -> origin == (0., 0.)    x  .0.5 .1.
    contour = cglib.scalar.grid2_contour(
        grid_scalars_flattened,
        scalar_field_cell_2dcount,
        scalar_field_param)
    contour_exp = cglib.point_data.PointData(
        point=jnp.array(
            [[1., 0.5],
             [0.75, 1.5],
             [0.6, 2.5],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan]]),
        data=jnp.array(
            [[jnp.nan,  1.],
             [0.,  2.],
             [1., jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan],
             [jnp.nan, jnp.nan]]))
    return cglib.tree_util.all_isclose_masked(contour, contour_exp)


def test_grid2_contour():
    assert jax.jit(helper_grid2_contour)()
