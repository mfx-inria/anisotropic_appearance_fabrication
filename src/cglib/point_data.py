"""
This module contains functions to represent and manipulate points with
associated data.
"""

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct, jit, lax, vmap
from jax._src.lib import xla_client

from . import array as cga
from . import grid, tree_util
from .array import NAI
from .grid import cell, edge
from .segment import repulse_point_then_project_on_segment
from .type import float_, uint


class PointData(NamedTuple):
    """Represent an array of points with data.

    Attributes
    ----------
    point : ndarray
        The point or array of points.
    data : ndarray
        The data or array of data associated with each point.
    """
    point: jnp.ndarray
    data: jnp.ndarray


class GridPointData(NamedTuple):
    """Represent a grid with square-shaped cells.

    Each cell has a fixed side length. Each cell also has a point with data.

    Attributes
    ----------
    point_data : PointData
        The point with data of all the cells of the grid.
    grid : Grid
        The grid.
    """
    point_data: PointData
    grid: grid.Grid


def grid_save(outfile: str,
              grid_point_data: GridPointData):
    """Save points with data contained in a grid into a single file.

    Save in uncompressed `.npz` format.

    Parameters
    ----------
    outfile : str
        The output file name.
    grid_point_data : GridPointData
        The point(s) with data.

    Returns
    -------
    None
    """
    cell_count = np.array(grid_point_data.grid.cell_ndcount)
    origin = np.array(grid_point_data.grid.origin)
    cell_sides_length = np.array(
        grid_point_data.grid.cell_sides_length)
    point = np.array(grid_point_data.point_data.point)
    point_data = np.array(grid_point_data.point_data.data)
    np.savez(
        outfile,
        cell_count=cell_count,
        origin=origin,
        cell_sides_length=cell_sides_length,
        point=point,
        point_data=point_data)


def grid_load(file) -> GridPointData:
    """Load points with data contained in a grid from a single file.

    Load points with data contained in a grid from a single file in
    uncompressed `.npz` format. The file must have been saved with
    `point_data_save`.

    Parameters
    ----------
    file : str
        The input file name.
    Returns
    -------
    GridPointData
        NumPy arrays are used to store the points with data contained in a
        grid.
    """
    data = np.load(file)
    cell_count = data['cell_count']
    origin = data['origin']
    cell_sides_length = data['cell_sides_length']
    points = data['point']
    points_data = data['point_data']

    grid_param = grid.Grid(cell_count, origin, cell_sides_length)
    points_data = PointData(points, points_data)
    return GridPointData(points_data, grid_param)


def grid_get(cell_ndindex: jnp.ndarray,
             grid_point_data: GridPointData) -> PointData:
    """Get the point with its data indexed by the given cell ND indices.

    Parameters
    ----------
    cell_ndindex : ndarray
        A matrix with rows containing the ND indices.
    grid_point_data : GridPointData
        The grid with points and their data.

    Returns
    -------
    PointData
        The points with their data indexed by the given cell ND indices.
    """
    cell_1dindex = cell.index1_from_ndindex(
        cell_ndindex,
        grid_point_data.grid.cell_ndcount)
    points_data: PointData = tree_util.leaves_at_indices(
        cell_1dindex,
        grid_point_data.point_data)
    return points_data


def grid_neighborhood_from_point(
        p: jnp.ndarray,
        grid_point_data: GridPointData,
        exclude_p: bool = False) -> PointData:
    """Return the neighboring points from a given position in a grid.

    Parameters
    ----------
    p : ndarray
        The position used for the neighborhood request.
    grid_point_data : GridPointData
        The grid of points with data.
    exclude_p : bool (default: False)
        If true, the point sharing the same cell as the given position is
        excluded from the returned set.
    Returns
    -------
    PointData
        The points with data in the Moore neighborhood of the given point's
        cell are returned. In addition, the point in the given point's cell is
        also returned, except if `exclude_p` is set to True. The returned point
        count is always 9. The set represents a 3x3 subgrid. The value `nan` is
        put for points and data in invalid cells (out of bound and/or excluded
        cells).
    """
    p_moore_neighborhood = cell.moore_neighborhood_from_point(
        p, grid_point_data.grid, 1)
    p_moore_neighborhood_count = p_moore_neighborhood.mask.shape[0]
    if exclude_p:
        center_cell_index = p_moore_neighborhood_count // 2
        new_mask = p_moore_neighborhood.mask.at[center_cell_index].set(True)
        p_moore_neighborhood = cga.MaskedArray(
            p_moore_neighborhood.array, new_mask)

    neighboring_cell_1dindices = vmap(
        cell.index1_from_ndindex, (0, None))(
        p_moore_neighborhood.array, grid_point_data.grid.cell_ndcount)
    neighboring_cell_1dindices = jnp.where(
        p_moore_neighborhood.mask, 0, neighboring_cell_1dindices)

    neighboring_points = \
        grid_point_data.point_data.point[neighboring_cell_1dindices]
    neighboring_points = jnp.where(
        p_moore_neighborhood.mask.reshape(-1, 1),
        jnp.nan,
        neighboring_points)
    neighboring_points_data = \
        grid_point_data.point_data.data[neighboring_cell_1dindices]
    neighboring_points_data = jnp.where(
        p_moore_neighborhood.mask.reshape(-1, 1),
        jnp.nan,
        neighboring_points_data)

    return PointData(neighboring_points, neighboring_points_data)


def save(outfile: str,
         point_data: PointData) -> None:
    """Save points with data into a single file in uncompressed `.npz` format.

    Parameters
    ----------
    outfile : str
        The output file name.
    point_data : PointData
        The point(s) with data.

    Returns
    -------
    None
    """
    point = np.array(point_data.point)
    point_data = np.array(point_data.data)
    np.savez(
        outfile,
        point=point,
        point_data=point_data)


def load(file: str) -> PointData:
    """Load point with data from a single file in uncompressed `.npz` format.

    The file must have been saved with `point_data_save`.

    Parameters
    ----------
    file : str
        The input file name.

    Returns
    -------
    PointData
        NumPy arrays are used to store the point(s) with data.
    """
    data = np.load(file)
    point = data['point']
    point_data_data = data['point_data']

    point_data = PointData(point, point_data_data)
    return point_data


def grid2_repulse_point_from_neighbors(
        edge_2dindex: jnp.ndarray,
        edge_axis: int,
        grid_edge_points: PointData,
        grid_cell_2dcount: jnp.ndarray,
        grid_param: grid.Grid,
        objective_distance: float) -> PointData:
    """Repulse one point belonging to one edge of a 2D grid from its neighbors.

    Parameters
    ----------
    edge_2dindex : ndarray
        The edge's 2-D index of the point that is repulsed.
    edge_axis : int
        The edge's axis of the point that is repulsed.
    grid_edge_points : PointData
        The grid's points and their data. The data must be an adjacency array
        of size two, representing each adjacent vertex by its 1D index. The
        ordering of the points follows the 1D indexing described in
        `grid.edge.index1_from_2dindex`.
    grid_cell_2dcount : ndarray
        The number of cells along each axis.
    grid_param : Grid
        The parameters of the grid.
    objective_distance : float
        The objective distance between the point and its neighbors.

    Returns
    -------
    PointData
        The repulsed point with its data. The point is always reprojected on
        its edge.
    """

    # BEGIN Private Class
    class RepulsePointParams(NamedTuple):
        edge_ndindex: jnp.ndarray
        edge_flatten_index: int
        edge_axis: int
        edge_point_data: PointData
        edge_ndcount: jnp.ndarray
        grid_edge_points: PointData
        grid_cell_ndcount: tuple[int, int]
        grid_param: grid.Grid
        objective_distance: float
    # END Private Class

    # BEGIN Private Function
    def _repulse_point_on_2dgrid_edge_from_neighbors_not_masked_case(
            params: RepulsePointParams) -> PointData:

        def _repulse_point_on_2dgrid_edge_from_neighbors_not_all_far_case(
                u_v: float,
                u_neighbors: jnp.ndarray,
                s: jnp.ndarray,
                edge_vertex_adjacency: jnp.ndarray) -> PointData:

            u_neighbors_valid_count = jnp.sum(
                jnp.logical_not(
                    jnp.isnan(u_neighbors))).astype(float_)
            u_neighbors = jnp.where(jnp.isnan(u_neighbors), 0., u_neighbors)
            u_neighbors_average = jnp.sum(
                u_neighbors) / u_neighbors_valid_count
            # Always move by half the distance because neighbor vertices will
            # also move in the opposite direction.
            u_half = 0.5 * (u_v + u_neighbors_average)
            new_position = s[0] + u_half * (s[1] - s[0])
            return PointData(new_position, edge_vertex_adjacency)

        # Get neighboring points potentially within distance
        # grid_params.cell_sides_length
        neighbors_ndindices = grid.edge.neighboring_2dindices_direct(
            params.edge_ndindex, params.edge_axis,
            params.grid_cell_ndcount,
            grid.edge.Neighboring2Type.WITHIN_CELL_SIDE_LENDTH)
        edge_2d_to_flattened_index_v_indices = vmap(
            grid.edge.index1_from_2dindex, (0, None, None))
        edge_2d_to_flattened_index_v_axis_indices = vmap(
            edge_2d_to_flattened_index_v_indices, (0, 0, None))
        neighbors_flatten_indices = edge_2d_to_flattened_index_v_axis_indices(
            neighbors_ndindices.array, jnp.array([0, 1]), params.edge_ndcount)
        neighbors: PointData = tree_util.leaves_at_indices(
            neighbors_flatten_indices,
            params.grid_edge_points)
        neighbors_mask = jnp.logical_or(
            neighbors_ndindices.mask.reshape(2, 4),
            jnp.isnan(neighbors.point[:, :, 0]))
        neighbors_point = jnp.where(neighbors_mask.reshape(
            (2, 4, 1)), jnp.zeros_like(neighbors.point), neighbors.point)
        neighbors_data = jnp.where(neighbors_mask.reshape(
            (2, 4, 1)), jnp.zeros_like(neighbors.data), neighbors.data)
        neighbors = PointData(neighbors_point, neighbors_data)

        # Mask neighboring edge points adjacent to current point
        # shape: (edge axis, edge, adjacent vertex indices)
        adjacent_edges = neighbors.data == params.edge_flatten_index
        # shape: (edge axis, edge)
        adjacent_edges = jnp.any(adjacent_edges, axis=2)
        neighbors_mask = jnp.logical_or(neighbors_mask, adjacent_edges).ravel()

        s = grid.edge.endpoints(
            params.edge_ndindex,
            params.edge_axis,
            params.grid_param)
        s0_v = params.edge_point_data.point - s[0]
        u_v = jnp.linalg.norm(s0_v) / params.grid_param.cell_sides_length
        neighboring_points = neighbors.point.reshape(-1, 2)
        u_neighbors = vmap(
            repulse_point_then_project_on_segment, (None, None, 0, None))(
            u_v, s, neighboring_points, params.objective_distance)
        u_neighbors = jnp.where(neighbors_mask, jnp.nan, u_neighbors)
        # if jnp.all(jnp.isnan(u_neighbors)):
        #     return params.edge_mvwa
        # else:
        # return
        # _repulse_point_on_2dgrid_edge_from_neighbors_not_all_far_case(u_v,
        # u_neighbors, s, params.edge_mvwa.adjacency_list)
        return lax.cond(
            jnp.all(jnp.isnan(u_neighbors)),
            lambda x: x[0],
            lambda x:
                _repulse_point_on_2dgrid_edge_from_neighbors_not_all_far_case(
                    x[1], x[2], x[3], x[4]),
            (params.edge_point_data,
             u_v,
             u_neighbors,
             s,
             params.edge_point_data.data))
    # END Private Function

    edge_2dcount = grid.edge.count2_per_axis(grid_cell_2dcount)

    # Get current point
    edge_flatten_index = grid.edge.index1_from_2dindex(
        edge_2dindex, edge_axis, edge_2dcount)
    edge_point: PointData = tree_util.leaves_at_indices(
        edge_flatten_index, grid_edge_points)
    edge_mpwa_mask = jnp.isnan(edge_point.point[0])
    repulse_point_params = RepulsePointParams(
        edge_2dindex,
        edge_flatten_index,
        edge_axis,
        edge_point,
        edge_2dcount,
        grid_edge_points,
        grid_cell_2dcount,
        grid_param,
        objective_distance)

    point_data_masked = PointData(
        jnp.full_like(edge_point.point, jnp.nan),
        jnp.full_like(edge_point.data, NAI))
    edge_point_repeled = \
        _repulse_point_on_2dgrid_edge_from_neighbors_not_masked_case(
            repulse_point_params)

    return lax.cond(
        edge_mpwa_mask,
        lambda x: x[0],
        lambda x: x[1],
        (point_data_masked, edge_point_repeled))


def grid2_repulse_points(
        grid_edge_points: PointData,
        grid_cell_2dcount: tuple[int, int],
        grid_param: grid.Grid,
        objective_distance: float,
        constraint: jnp.ndarray) -> PointData:
    """Repulse all the points of a graph by a given distance.

    Each vertex of the graph is associated to an unique edge of a 2D grid.

    Parameters
    ----------
    grid_edge_points : PointData
        The points of the grid's edges. One edge = one point/vertex. The data
        for each point is an array of two 1D indices indicating its adjacency.
    grid_cell_2dcount : tuple[int, int]
        The number of cells along each axis.
    grid_param : Grid
        The parameters of the grid.
    objective_distance : float
        The objective distance between a point and its neighbors.
    constraint : ndarray
        An array of booleans indicates which point is constrained.
    Return
    ------
    PointData
        The repulsed points and their data.
    """

    # Compute the edge ndindices from grid cell ndcount
    with jax.ensure_compile_time_eval():
        edge_2dindices = grid.edge.indices2_from_grid(grid_cell_2dcount)

    # Vectorized the function over the points
    repulse_point_from_neighbors_v = vmap(
        grid2_repulse_point_from_neighbors,
        (0, None, None, None, None, None))

    horizontal_edge_points = repulse_point_from_neighbors_v(
        edge_2dindices[0],
        0,
        grid_edge_points,
        grid_param.cell_ndcount,
        grid_param, objective_distance)
    vertical_edge_points = repulse_point_from_neighbors_v(
        edge_2dindices[1],
        1,
        grid_edge_points,
        grid_param.cell_ndcount,
        grid_param, objective_distance)
    # Concatenate horizontal and vertical edge points
    grid_edge_points_updated: PointData = tree_util.concatenate(
        (horizontal_edge_points, vertical_edge_points))
    # Project constrained points to their initial positions
    edges_point_data_point = jnp.where(
        constraint.reshape(-1, 1),
        grid_edge_points.point,
        grid_edge_points_updated.point)

    return PointData(edges_point_data_point, grid_edge_points.data)


def grid2_repulse_points_n_times(
        grid_edge_points: PointData,
        grid_cell_2dcount: tuple[int, int],
        grid_param: grid.Grid,
        objective_distance: float,
        constraint: jnp.ndarray,
        n: int):
    """Repulse n times the points of a graph by a given distance.

    See `grid2_repulse_points`.
    """
    # Unroll the loop
    # Better performance but more compilation time, and more memory usage
    for _ in range(n):
        grid_edge_points = grid2_repulse_points(
            grid_edge_points,
            grid_cell_2dcount,
            grid_param,
            objective_distance,
            constraint)
    return grid_edge_points


def grid2_compile_repulse_points_n_times(
        grid2_cell_2dcount: tuple[int, int],
        device: xla_client.Device,
        repulse_iter: int):
    """Compile `grid2_repulse_points_n_times` for a 2D grid.

    Parameters
    ----------
    grid2_cell_2dcount : tuple[int, int]
        The number of vertices/points along each axis of the 2D grid.
    device : Device
        The device the compiled function will run on.
        Available devices can be retrieved via `jax.devices()`.
    repulse_iter : int
        The number of repulsion iterations.

    Returns
    -------
    tuple[jax.stages.Compiled, float]
        out1
            The compiled function.
        out2
            The duration of the compilation (seconds).
    """
    grid_edge_1dcount = edge.count1_from_cell_2dcount(grid2_cell_2dcount)
    point_shape_dtype = ShapeDtypeStruct((grid_edge_1dcount, 2), float_)
    point_data_data_shape_dtype = ShapeDtypeStruct(
        (grid_edge_1dcount, 2), uint)
    point_data_shape_dtype = PointData(
        point_shape_dtype,
        point_data_data_shape_dtype)
    grid_shape_dtype = grid.shape_dtype_from_dim(2)
    constraints_shape_dtype = ShapeDtypeStruct((grid_edge_1dcount,), bool)

    start = time.perf_counter()
    func_jit = jit(
        grid2_repulse_points_n_times, static_argnums=(1, 5), device=device)
    func_lowered = func_jit.lower(
        point_data_shape_dtype,
        grid2_cell_2dcount,
        grid_shape_dtype,
        ShapeDtypeStruct((), float_),
        constraints_shape_dtype,
        repulse_iter)
    func_compiled = func_lowered.compile()
    stop = time.perf_counter()

    return func_compiled, stop - start
