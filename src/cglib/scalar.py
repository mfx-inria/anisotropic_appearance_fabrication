import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct, jit, lax, vmap
from jax._src.lib import xla_client

from .array import NAI
from .grid.cell import (corner_vertex_ndindices, index1_from_ndindex,
                        ndindex_is_valid)
from .grid.edge import (Neighboring2Type, endpoints, index1_from_2dindex,
                        indices2_from_grid, neighboring_2dindices_direct)
from .grid.grid import Grid, shape_dtype_from_dim
from .math import clamp, float_same_sign, solve_linear_interpolation_equation
from .point_data import PointData
from .tree_util import concatenate
from .type import float_, uint


def grid_edge_point_scalars(
        edge_ndindex: jnp.ndarray,
        edge_axis: int,
        grid_scalars_flattened: jnp.ndarray,
        grid_cell_ndcount: jnp.ndarray) -> jnp.ndarray:
    """Returns two scalars defined at the endpoints of a ND grid's edge.

    This function returns two scalars defined at the endpoints of a ND grid's
    edge. An edge is defined as a cell ND index and an axis specifying which
    edge of the ND cell is used. A cell has 2^(N-1) edges parallel to a domain
    axis, and the edge considered is the one with the endpoint with the
    smallest components.

    Parameters
    ----------
    edge_ndindex : ndarray
        The cell ND index of the edge. A valid ND index has `all(zeros(A.shape)
        <= edge_ndindex <= A) == True`, where `A` is the tuple giving the
        number of cells along each axis.
    edge_axis: int
        The axis specifies which domain axis is parallel to the edge.
    flattened_scalar_grid : ndarray
        A 1D array containing the scalars associated at each vertex of a grid.
        The vertices are ordered as follows: their indices first increase along
        the domain's first axis, then the second, etc. A ND grid has a number
        of cells along each axis defined as a 1D array `A` with size N.

    Returns
    -------
    ndarray
        This function returns two scalars defined at the endpoints of a ND
        grid's edge.
    """
    shift = jnp.zeros_like(edge_ndindex).at[edge_axis].set(1)
    edge_vertex_ndindices = jnp.array([edge_ndindex, edge_ndindex + shift])
    edge_vertex_flattened_indices = vmap(
        index1_from_ndindex, (0, None))(
        edge_vertex_ndindices, grid_cell_ndcount)
    edge_scalars = grid_scalars_flattened[edge_vertex_flattened_indices]
    return edge_scalars


def grid_edge_root_existence(
        edge_ndindex: jnp.ndarray,
        edge_axis: int,
        flattened_scalar_field: jnp.ndarray,
        grid: Grid) -> bool:
    """Return true if the two endpoints scalars have the same sign.

    This function returns a boolean saying if the two endpoint scalars of the
    prescribed edge have the same sign. An edge is defined as a cell ND index
    and an axis specifying which domain axis is parallel to the edge. See test.

    Parameters
    ----------
    edge_ndindex : jnp.ndarray
        The ND index of the edge. See test.
    edge_axis : int
        The axis specifies which domain axis is parallel to the edge.
    flattened_scalar_grid : jnp.ndarray
        A 1D array containing the scalars associated at each vertex of a grid.
        The vertices are ordered as follows: their indices first increase along
        the domain's first axis, then the second, etc.
    grid : Grid
        The grid.

    Returns
    -------
    bool
        This function returns a boolean saying if the two endpoint scalars of
        the prescribed edge have the same sign.
    """

    edge_point_scalars_val = grid_edge_point_scalars(
        edge_ndindex,
        edge_axis,
        flattened_scalar_field,
        grid.cell_ndcount)

    solution_exists = jnp.logical_not(
        float_same_sign(edge_point_scalars_val[0], edge_point_scalars_val[1]))
    return solution_exists


def grid_edge_root_point(
        edge_ndindex: jnp.ndarray,
        edge_axis: int,
        flattened_scalar_field: jnp.ndarray,
        grid: Grid) -> jnp.ndarray:
    """Return the root point on the edge from endpoint scalar interpolation.

    This function returns the point on the specified edge giving zero when the
    endpoints scalar associated with the specified edge are interpolated. An
    edge is defined as a cell ND index and an axis specifying which domain axis
    is parallel to the edge. See test.

    Parameters
    ----------
    edge_ndindex : ndarray
        The ND index of the edge. See test.
    edge_axis : int
        The axis specifies which domain axis is parallel to the edge.
    flattened_scalar_grid : ndarray
        A 1D array containing the scalars associated at each vertex of a grid.
        The vertices are ordered as follows: their indices first increase along
        the domain's first axis, then the second, etc.
    grid : Grid
        The grid.

    Returns
    -------
    ndarray
        This function returns the point on the specified edge giving zero when
        the two endpoint scalars associated with the specified edge are
        interpolated. The function returns [nan nan] if there is no such point,
        i.e., if the two endpoint scalars have the same sign. If they are
        equal, return the midpoint between the two edge endpoints.
    """
    edge_endpoints_val = endpoints(edge_ndindex, edge_axis, grid)
    edge_point_scalars_val = grid_edge_point_scalars(
        edge_ndindex,
        edge_axis,
        flattened_scalar_field,
        grid.cell_ndcount)
    mask = float_same_sign(
        edge_point_scalars_val[0], edge_point_scalars_val[1])
    u = solve_linear_interpolation_equation(
        edge_point_scalars_val[0], edge_point_scalars_val[1])

    # To avoid singular case where points are extracted at edge endpoints
    epsilon = 0.01
    u = clamp(u, epsilon, 1. - epsilon)

    # Interpolate between
    root_point = (1. - u) * edge_endpoints_val[0] + u * edge_endpoints_val[1]
    return jnp.where(mask, jnp.full_like(root_point, jnp.nan), root_point)


def grid2_contour_get_output_grid(scalar_field_grid: Grid) -> Grid:
    """Return the grid of `grid2_contour`'s output.

    Parameters
    ----------
    scalar_field_grid : Grid
        The intput grid of `grid2_contour`.

    Returns
    -------
    Grid
        The `grid2_contour`'s output grid. The contour has each vertex on a
        unique edge of the output grid. The output grid is translated by half a
        cell width and has one cell removed on the top and right sides. See
        `grid2_contour` for an illustration.
    """
    return Grid(
        scalar_field_grid.cell_ndcount - 1,
        scalar_field_grid.origin + scalar_field_grid.cell_sides_length * 0.5,
        scalar_field_grid.cell_sides_length)


def grid2_contour_check_all_cycles(contour_graph: PointData) -> bool:
    """Check if the contour graph represents only cycles.

    Parameters
    ----------
    contour_graph : PointData
        A valid contour graph. The contour usually checked is the one from
        `grid2_contour`.

    Returns
    -------
    bool
        The graph represents only cycles if
        1. the two adjacent points are not masked and
        2. the two adjacent points differ.
        In this case, the function returns `True`; otherwise, False.
    """
    adj0 = contour_graph.data[:, 0]
    adj1 = contour_graph.data[:, 1]

    adj0_isnan = jnp.isnan(adj0)
    adj1_isnan = jnp.isnan(adj1)
    both_nan = jnp.logical_and(adj0_isnan, adj1_isnan)
    both_not_nan = jnp.logical_and(
        jnp.logical_not(adj0_isnan),
        jnp.logical_not(adj1_isnan))

    both_not_nan_and_different = jnp.logical_and(
        both_not_nan,
        jnp.not_equal(adj0, adj1))
    res = jnp.logical_or(both_not_nan_and_different, both_nan)
    return jnp.all(res)


def grid2_contour(
        scalar_field_flattened: jnp.ndarray,
        scalar_field_cell_2dcount: tuple[int, int],
        scalar_field_grid: Grid) -> PointData:
    """Compute the contour of a given scalar field.

    The contour is defined as the isolines with iso values zero. The marching
    square algorithm is used.

    Parameters
    ----------
    scalar_field_flattened : ndarray
        The 2D scalar field gives the scalar values for each cell of the input
        grid, called the scalar field. The scalar field is flattened, i.e., it
        is a 1D array. See test.
    scalar_field_cell_2dcount : tuple[int, int]
        This is `scalar_field_param.cell_ndcount`, see next parameter. It is a
        separate parameter for compilation.
    scalar_field_grid : Grid
        Scalar field parameters. The scalar field is defined as a grid with
        scalars defined at the center of its cells. The grid has its point with
        the lowest components called the origin.
    Return
    ------
    PointData
        The function returns points associated with data for each edge of the
        contouring grid. The contouring grid has the same number of cells along
        each domain's axis as the input scalar field grid minus one. The
        vertices' positions of the contouring grid are the centers of the
        scalar field grid's cells. See test. The point ordering follows the
        grid edge 1D ordering. See `grid.edge.index1_from_2dindex`. The
        point array contains positions on the contouring grid's edges
        associated with scalar zero when contouring grid  vertices' scalars are
        linearly interpolated along their edge. A point with `nan` components
        is associated with the edges with endpoints' scalars having the same
        sign. The data associated with each point is an adjacency list with
        size two. Each data item indicates the corresponding adjacent vertex's
        edge 1D index. The data type is float. The component `nan` means the
        vertex is not linked to another vertex. The first component is for the
        bottom or left side. The second component is for the top or right side.
    """

    # BEGIN Private function to grid2_contour
    # ------------------------------------------------

    def uniform_grid_edge_root_point_and_adjacency(
            edge_ndindex: jnp.ndarray,
            edge_axis: int,
            scalar_1darray: jnp.ndarray,
            scalar_grid_param: Grid) -> PointData:

        # BEGIN Private class and functions to
        # uniform_grid_edge_root_point_and_adjacency
        # ------------------------------------------
        class GetEdgeAdjacencyParams(NamedTuple):
            """
            This class groups the parameters of the get_edge_adjacency
            functions.
            """
            edge_ndindex: jnp.ndarray
            edge_axis: int
            edge_side: int
            edge_cell_ndcount: jnp.ndarray
            same_side_corner_and_center: bool

        def _get_edge_adjacency_no_extraction_case(
                params: GetEdgeAdjacencyParams) -> uint:
            return NAI

        def _convert_edge_shift_to_adjacency(
                shift: jnp.ndarray,
                params: GetEdgeAdjacencyParams,
                adjacent_edge_axis: int) -> uint:
            shift = lax.cond(params.edge_axis == 1,
                             lambda x: x[::-1], lambda x: x, shift)
            adjacent_edge_ndindex = params.edge_ndindex + shift
            return index1_from_2dindex(
                adjacent_edge_ndindex,
                adjacent_edge_axis,
                params.edge_cell_ndcount)

        def _get_edge_adjacency_case_001(
                params: GetEdgeAdjacencyParams) -> uint:
            shift = jnp.array([1, -1 + params.edge_side])
            adjacent_edge_axis = (params.edge_axis + 1) % 2
            return _convert_edge_shift_to_adjacency(
                shift,
                params,
                adjacent_edge_axis)

        def _get_edge_adjacency_case_010(
                params: GetEdgeAdjacencyParams) -> uint:
            shift = jnp.array([0, -1 + params.edge_side])
            adjacent_edge_axis = (params.edge_axis + 1) % 2
            return _convert_edge_shift_to_adjacency(
                shift,
                params,
                adjacent_edge_axis)

        def _get_edge_adjacency_case_100(
                params: GetEdgeAdjacencyParams) -> uint:
            shift = jnp.array([0, -1 + 2 * params.edge_side])
            return _convert_edge_shift_to_adjacency(
                shift,
                params,
                params.edge_axis)

        def _get_edge_adjacency_case_111(
                params: GetEdgeAdjacencyParams) -> uint:
            return lax.cond(
                params.same_side_corner_and_center,
                lambda x: _get_edge_adjacency_case_001(x),
                lambda x: _get_edge_adjacency_case_010(x),
                params)

        # END Private class and functions to
        # uniform_grid_edge_root_point_and_adjacency
        # ------------------------------------------

        # Grid edge root vertex computation is trivial
        grid_edge_root_point_val = grid_edge_root_point(
            edge_ndindex,
            edge_axis,
            scalar_1darray,
            scalar_grid_param) + scalar_grid_param.cell_sides_length * 0.5

        # Edge adjacency computation is less trivial.
        hedge_ndcell_count = (
            scalar_grid_param.cell_ndcount[0] - 1,
            scalar_grid_param.cell_ndcount[1])
        vedge_ndcell_count = (
            scalar_grid_param.cell_ndcount[0],
            scalar_grid_param.cell_ndcount[1] - 1)
        edge_cell_ndcount = jnp.array((hedge_ndcell_count, vedge_ndcell_count))
        contour_grid_cell_count = (
            scalar_grid_param.cell_ndcount[0] - 1,
            scalar_grid_param.cell_ndcount[1] - 1)

        visible_neighbors_ndindices = neighboring_2dindices_direct(
            edge_ndindex,
            edge_axis,
            contour_grid_cell_count,
            Neighboring2Type.VISIBLE)

        edge_root_existence = []
        for i_axis in range(2):
            edge_root_existence.append(
                vmap(
                    grid_edge_root_existence,
                    (0,
                     None,
                     None,
                     None))(
                    visible_neighbors_ndindices.array[i_axis],
                    i_axis,
                    scalar_1darray,
                    scalar_grid_param))
        for i_axis in range(2):
            edge_root_existence[i_axis] = jnp.logical_and(
                edge_root_existence[i_axis], jnp.logical_not(
                    visible_neighbors_ndindices.mask[i_axis]))
        edge_root_existence = jnp.array(edge_root_existence)

        # First index is for the side (bottom or top for x-axis, left or right
        # for a y-axis). Second index is for the edge, with the following
        # indexing convention (the edge without index is the one whose
        # adjacency is currently computed, indicated by parameter edge_ndindex)

        # Axis 0, Side 0
        #     - - - - - -
        #     |         |
        #     |1        |2
        #     |         |
        #     -----------
        #          0

        # Axis 0, Side 1
        #          0
        #     -----------
        #     |         |
        #     |1        |2
        #     |         |
        #     - - - - - -

        # Axis 1, Side 0
        #          2
        #     -----------
        #     |
        #     |0        |
        #     |
        #     -----------
        #          1

        # Axis 1, Side 1
        #          2
        #     -----------
        #               |
        #     |         |0
        #               |
        #     -----------
        #          1

        next_edge_axis = (edge_axis + 1) % 2
        # Shape: (edge cell side, visible edge)
        root_exist_config = \
            jnp.array([[edge_root_existence[edge_axis][0],
                        edge_root_existence[next_edge_axis][0],
                        edge_root_existence[next_edge_axis][1]],
                       [edge_root_existence[edge_axis][1],
                        edge_root_existence[next_edge_axis][2],
                        edge_root_existence[next_edge_axis][3]]])

        cell_shift = lax.cond(edge_axis == 0, lambda x: jnp.array(
            [0, -1]), lambda x: jnp.array([-1, 0]), None)
        edge_adjacent_cells_ndindices = jnp.array(
            [edge_ndindex + cell_shift, edge_ndindex])
        edge_adjacent_cells_ndindices_mask = jnp.logical_not(
            vmap(ndindex_is_valid, (0, None))(
                edge_adjacent_cells_ndindices,
                jnp.array(contour_grid_cell_count)))
        edge_adjacent_cells_ndindices = jnp.where(
            edge_adjacent_cells_ndindices_mask.reshape(-1, 1),
            0,
            edge_adjacent_cells_ndindices)
        grid_corner_vertex_ndindices = vmap(
            corner_vertex_ndindices, (0,))(edge_adjacent_cells_ndindices)
        grid_corner_1dindices = vmap(
            index1_from_ndindex, (0, None))(
            grid_corner_vertex_ndindices.reshape(
                (-1, 2)), scalar_grid_param.cell_ndcount)
        # Shape: (edge cell side, corner vertex value)
        corner_scalars = scalar_1darray[grid_corner_1dindices].reshape(2, 4)
        average_scalar = jnp.average(corner_scalars, axis=1)
        same_side_corner_and_center = []
        ref_corner_verter_value = scalar_1darray[index1_from_ndindex(
            edge_ndindex, scalar_grid_param.cell_ndcount)]
        for i_side in range(2):
            same_side_corner_and_center.append(
                average_scalar[i_side] * ref_corner_verter_value)
        same_side_corner_and_center = jnp.array(same_side_corner_and_center)
        same_side_corner_and_center = same_side_corner_and_center >= 0.

        # autopep8: off
        # axis == 0 ->
        # We are currently computing the adjacency of a horizontal edge.
        #
        #     side == 0 -> find adjacency with visible bottom edges
        #
        #      001          010          100          111          111
        #   _ _ _ _o_    _o_ __ _     _ _ _o_ _    _o_ _ _ _    _ _ _ _o_
        #  |        \|  |/        |  |     |   |  |/        |  |        \|
        #  |         o  o         |  |     |   |  o         o  o         o
        #  |         |  |         |  |     |   |  |        /|  |\        |
        #  |_________|  |_________|  |_____o___|  |______ o_|  |_o_______|
        #
        #                                 ||
        #                        Flip Y   ||
        #                                 \/
        #     side == 1 -> find adjacency with top visible edges
        #
        #      001          010          100          111          111
        #   _________    _________    _____o___    ______ o_    _o_______
        #  |         |  |         |  |     |   |  |        \|  |/        |
        #  |         o  o         |  |     |   |  o         o  o         o
        #  |        /|  |\        |  |     |   |  |\        |  |        /|
        #  |_ _ _ _o_|  |_o_ __ _ |  |_ _ _o_ _|  |_o_ _ _ _|  |_ _ _ _o_|
        #
        #  axis == 1 -> We are currently computing the adjacency of a vertical
        #  edge.
        #  Here the cases are the transpose of the previous cases, i.e., the
        #  axes are flipped.
        # ------------------
        # shape: (edge cell side,)
        case_index = jnp.array(
            [4 * root_exist_config[0][0] +
             2 * root_exist_config[0][1] +
             1 * root_exist_config[0][2],
             4 * root_exist_config[1][0] +
             2 * root_exist_config[1][1] +
             1 * root_exist_config[1][2]])

        branches = (
            _get_edge_adjacency_no_extraction_case,  # 000: 0
            _get_edge_adjacency_case_001,            # 001: 1
            _get_edge_adjacency_case_010,            # 010: 2
            _get_edge_adjacency_no_extraction_case,  # 011: 3
            _get_edge_adjacency_case_100,            # 100: 4
            _get_edge_adjacency_no_extraction_case,  # 101: 5
            _get_edge_adjacency_no_extraction_case,  # 110: 6
            _get_edge_adjacency_case_111,            # 111: 7
            )
        # autopep8: on

        adjacency_list = []
        for i_side in range(2):
            get_edge_adjacency_params_i = GetEdgeAdjacencyParams(
                edge_ndindex,
                edge_axis,
                i_side,
                edge_cell_ndcount,
                same_side_corner_and_center[i_side])
            adjacency_list.append(
                lax.switch(
                    case_index[i_side],
                    branches,
                    get_edge_adjacency_params_i))
        adjacency_array = jnp.array(adjacency_list, float_)

        adjacency_array = jnp.where(
            jnp.equal(adjacency_array, NAI),
            jnp.nan,
            adjacency_array)

        return PointData(grid_edge_root_point_val, adjacency_array)
    # END Private functions to grid2_contour
    # -----------------------------------------------

    with jax.ensure_compile_time_eval():
        contour_grid_edge_2dindices = indices2_from_grid(
            jnp.array(scalar_field_cell_2dcount) - 1)
    # Horizontal edges
    v_h: PointData = vmap(
        uniform_grid_edge_root_point_and_adjacency, (0, None, None, None))(
        contour_grid_edge_2dindices[0],
        0,
        scalar_field_flattened,
        scalar_field_grid)
    # Vertical edges
    v_v: PointData = vmap(
        uniform_grid_edge_root_point_and_adjacency, (0, None, None, None))(
        contour_grid_edge_2dindices[1],
        1,
        scalar_field_flattened,
        scalar_field_grid)
    # Concatenate horizontal and vertical edges' point data
    res = concatenate((v_h, v_v))
    return res


def grid2_compile_contour(
        scalar_field_cell_2dcount: tuple[int, int],
        device: xla_client.Device) -> tuple[jax.stages.Compiled, float]:
    """Compiles `grid2_contour` for the scalar field's cell 2D count.

    Parameters
    ----------
    scalar_field_cell_2dcount : tuple[int, int]
        The number of cells along each axis of the scalar field.
        The scalars are associated with the center of the cells.
    device : Device
        The device the compiled function will run on.
        Available devices can be retrieved via `jax.devices()`.

    Returns
    -------
    tuple[Compiled, float]
        out1
            The function returns `grid2_contour` compiled for the
            specified device and scalar field's cell 2D count.
        out2
            The duration of the compilation (seconds).
    """

    scalar_field_cell_1dcount = scalar_field_cell_2dcount[0] * \
        scalar_field_cell_2dcount[1]

    scalar_grid_shape_dtype = ShapeDtypeStruct(
        (scalar_field_cell_1dcount,), float_)
    uniform_grid_shape_dtype = shape_dtype_from_dim(2)

    start = time.perf_counter()
    grid2_contour_jit = jit(
        grid2_contour, static_argnums=1, device=device)
    grid2_contour_lowered = grid2_contour_jit.lower(
        scalar_grid_shape_dtype,
        scalar_field_cell_2dcount,
        uniform_grid_shape_dtype)
    grid2_contour_compiled = grid2_contour_lowered.compile()
    stop = time.perf_counter()

    return grid2_contour_compiled, stop - start
