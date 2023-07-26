import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct, jit, lax, vmap
from jax._src.lib import xla_client

from .array import NAI
from .grid import edge
from .grid.grid import Grid, shape_dtype_from_dim
from .limits import FLOAT_MAX
from .math import circle_smallest_radius_tangent_to_p_passing_through_q
from .point_data import PointData
from .polyline import Polyline
from .segment import (intersection_bool, patching_energy,
                      tangent)
from .type import float_, int_, uint


class Cycle(NamedTuple):
    """
    This class represents one or several cycles with data.

    Attributes
    ----------
    point_data : PointData
        The points and data associated with each vertex of the cycles.
    cycle_data : ndarray
        The data associated with each cycle.
    cycle_count : uint
        The number of cycles.
    """
    point_data: PointData
    cycle_data: jnp.ndarray
    cycle_count: uint


def save(file: str,
         cycle: Cycle) -> None:
    """
    This function saves the cycle to the disk in uncompressed format `.npz`.

    Parameters
    ----------
    file : str or file
        Either the filename (string) or an open file (file-like object)
        where the data will be saved. If file is a string or a Path, the
        ``.npz`` extension will be appended to the filename if it is not
        already there.
    cycle : Cycle
        The cycle instance to save.

    Return
    ------
    None
    """
    point = np.array(cycle.point_data.point)
    point_data = np.array(cycle.point_data.data)
    cycle_data = np.array(cycle.cycle_data)
    cycle_count = np.array(cycle.cycle_count)
    np.savez(
        file,
        point=point,
        point_data=point_data,
        cycle_data=cycle_data,
        cycle_count=cycle_count)


def load(file) -> Cycle:
    """This function loads a cycle from the disk.

    Parameters
    ----------
    file : file-like object, string, or pathlib.Path
        The file to read. File-like objects must support the ``seek()`` and
        ``read()`` methods.

    Returns
    -------
    Cycle
        The loaded cycle instance.
    """
    data = np.load(file)
    point = data['point']
    point_data = data['point_data']
    cycle_data = data['cycle_data']
    cycle_count = data['cycle_count']

    point_d = PointData(point, point_data)
    cycle = Cycle(point_d, cycle_data, cycle_count)
    return cycle


def create_from_graph(graph: PointData, cycle_count_max: int) -> Cycle:
    """Create a cycle from a graph.

    Parameters
    ----------
    graph : PointData
        The graph must be a graph which represents one or multiple cycles,
        where each vertex must have two different adjacent vertices. The
        adjacent vertices' indices must be located in the first and second
        components of the `data` attribute of `PointData`. The result of the
        marching square algorithm with only positive or negative values at the
        boundary gives one or multiple cycles. The input can be masked points,
        i.e., points with `nan` as the first component.
    cycle_count_max : int
        The maximum number of cycles the graph can contain.

    Notes
    -----
        The function can never return if the input is not a valid graph.

    Returns
    -------
    Cycle
        The `out.point_data.point` 2D array is the same as
        `graph.point_data.point`. `out.point_data.data[i, 0]` and
        `out.point_data.data[i, 1]` indicate the cycle's next and previous
        vertices of vertex `i`,  respectively. The returned adjacency can be
        different than the input adjacency `graph.point_data.data` as it is not
        required that it is ordered. `out.point_data.data[i, 2]` is the cycle
        index associated with vertex i. `out.cycle_count` gives the actual
        number of cycles |C|, i.e., the number of connected graph components.
        `out.cycle_data` has shape (`cycle_count_max`, 2). `out.cycle_data[i,
        0]` gives the starting vertex's index of the cycle i.
        `out.cycle_data[i, 1]` gives the number of vertices/edges of the cycle
        i.
    """

    class GraphFloodFromPointParams(NamedTuple):
        point_processed: jnp.ndarray
        graph: PointData
        cycles_start_indices: jnp.ndarray
        cycle_count: int
        point_cycle_id: jnp.ndarray
        point_cycle_next_point: jnp.ndarray
        point_cycle_previous_point: jnp.ndarray
        cycles_edge_count: jnp.ndarray

    def graph_flood_from_point(i: int, params: GraphFloodFromPointParams):

        def do_nothing(*operands):
            # Unpack operands
            cycles_start_indices = operands[1]
            cycle_count = operands[2]
            point_processed: jnp.ndarray = operands[4]
            point_cycle_id: jnp.ndarray = operands[5]
            point_cycle_next_point = operands[6]
            point_cycle_previous_point = operands[7]
            cycles_edge_count = operands[8]

            # Repack a subset of operands
            ret = (
                cycles_start_indices,
                cycle_count,
                point_processed,
                point_cycle_id,
                point_cycle_next_point,
                point_cycle_previous_point,
                cycles_edge_count)
            return ret

        def graph_create_one_cycle(*operands):

            class CyclesCreateFromGraphWhileParams(NamedTuple):
                cycle_point_index: int
                cycle_start_index: int
                cycle_edge_count: int
                point_processed: jnp.ndarray
                point_cycle_id: jnp.ndarray
                cycle_count: uint
                point_cycle_next_point: jnp.ndarray
                cycle_point_index_p1: uint
                point_cycle_previous_point: jnp.ndarray
                cycle_point_index_m1: uint
                graph: PointData

            def cycle_iteration_end(params: CyclesCreateFromGraphWhileParams):
                return jnp.logical_or(
                    jnp.not_equal(
                        params.cycle_point_index,
                        params.cycle_start_index),
                    jnp.equal(params.cycle_edge_count, 0))

            def cycle_compute_data_and_iterate(
                    params: CyclesCreateFromGraphWhileParams):
                point_processed = params.point_processed.at[
                    params.cycle_point_index].set(True)

                point_cycle_id = params.point_cycle_id.at[
                    params.cycle_point_index].set(params.cycle_count)
                point_cycle_next_point = params.point_cycle_next_point.at[
                    params.cycle_point_index].set(params.cycle_point_index_p1)
                point_cycle_previous_point = \
                    params.point_cycle_previous_point.at[
                        params.cycle_point_index].set(
                            params.cycle_point_index_m1)
                cycle_edge_count = params.cycle_edge_count + 1

                cycle_point_index_p2 = jnp.where(
                    jnp.equal(
                        params.graph.data[params.cycle_point_index_p1, 0],
                        params.cycle_point_index),
                    params.graph.data[params.cycle_point_index_p1, 1],
                    params.graph.data[params.cycle_point_index_p1, 0]).astype(
                        uint)

                cycle_point_index_m1 = params.cycle_point_index
                cycle_point_index = params.cycle_point_index_p1
                cycle_point_index_p1 = cycle_point_index_p2

                while_params = CyclesCreateFromGraphWhileParams(
                    cycle_point_index,
                    params.cycle_start_index,
                    cycle_edge_count,
                    point_processed,
                    point_cycle_id,
                    params.cycle_count,
                    point_cycle_next_point,
                    cycle_point_index_p1,
                    point_cycle_previous_point,
                    cycle_point_index_m1,
                    params.graph)
                return while_params

            # Pack operands
            i_point = operands[0]
            cycles_start_indices = operands[1]
            cycle_count = operands[2]
            graph: PointData = operands[3]
            point_processed: jnp.ndarray = operands[4]
            point_cycle_id: jnp.ndarray = operands[5]
            point_cycle_next_point = operands[6]
            point_cycle_previous_point = operands[7]
            cycles_edge_count = operands[8]

            cycle_start_index = i_point
            cycles_start_indices = cycles_start_indices.at[cycle_count].set(
                cycle_start_index)

            cycle_edge_count = 0

            cycle_point_index = cycle_start_index
            cycle_point_index_p1 = graph.data[cycle_point_index, 0].astype(
                uint)
            cycle_point_index_m1 = graph.data[cycle_point_index, 1].astype(
                uint)

            # Pack while parameters
            cycle_iteration_params = CyclesCreateFromGraphWhileParams(
                cycle_point_index,
                cycle_start_index,
                cycle_edge_count,
                point_processed,
                point_cycle_id,
                cycle_count,
                point_cycle_next_point,
                cycle_point_index_p1,
                point_cycle_previous_point,
                cycle_point_index_m1,
                graph)

            # For debugging
            # while cycle_iteration_end(cycle_iteration_params):
            #     cycle_iteration_params = cycle_compute_data_and_iterate(
            #         cycle_iteration_params)
            cycle_iteration_params = jax.lax.while_loop(
                cycle_iteration_end,
                cycle_compute_data_and_iterate,
                cycle_iteration_params)

            # Unpack while parameters
            cycle_point_index = cycle_iteration_params.cycle_point_index
            cycle_start_index = cycle_iteration_params.cycle_start_index
            cycle_edge_count = cycle_iteration_params.cycle_edge_count
            point_processed = cycle_iteration_params.point_processed
            point_cycle_id = cycle_iteration_params.point_cycle_id
            cycle_count = cycle_iteration_params.cycle_count
            point_cycle_next_point = \
                cycle_iteration_params.point_cycle_next_point
            point_cycle_previous_point = \
                cycle_iteration_params.point_cycle_previous_point

            cycles_edge_count = cycles_edge_count.at[cycle_count].set(
                cycle_edge_count)
            cycle_count = cycle_count + 1

            ret = (
                cycles_start_indices,
                cycle_count,
                point_processed,
                point_cycle_id,
                point_cycle_next_point,
                point_cycle_previous_point,
                cycles_edge_count)
            return ret

        is_not_point_processed = jnp.logical_not(params.point_processed[i])
        is_valid_point = jnp.logical_not(jnp.isnan(params.graph.point[i][0]))
        is_valid_starting_point = jnp.logical_and(
            is_not_point_processed, is_valid_point)

        operands = (
            i,
            params.cycles_start_indices,
            params.cycle_count,
            params.graph,
            params.point_processed,
            params.point_cycle_id,
            params.point_cycle_next_point,
            params.point_cycle_previous_point,
            params.cycles_edge_count)

        # For debug
        # if is_valid_starting_point:
        #     res = graph_create_one_cycle(*operands)
        # else:
        #     res = do_nothing(*operands)
        res = lax.cond(
            is_valid_starting_point,
            graph_create_one_cycle,
            do_nothing,
            *operands)
        cycles_start_indices = res[0]
        cycle_count = res[1]
        point_processed = res[2]
        point_cycle_id = res[3]
        point_cycle_next_point = res[4]
        point_cycle_previous_point = res[5]
        cycles_edge_count = res[6]

        return GraphFloodFromPointParams(
            point_processed,
            params.graph,
            cycles_start_indices,
            cycle_count,
            point_cycle_id,
            point_cycle_next_point,
            point_cycle_previous_point,
            cycles_edge_count)

    point_count = graph.point.shape[0]

    # Allocate memory
    point_processed = jnp.full((point_count,), False, bool)
    point_cycle_id = jnp.full((point_count,), NAI, uint)
    point_cycle_next_point = jnp.full((point_count,), NAI, uint)
    point_cycle_previous_point = jnp.full((point_count,), NAI, uint)
    cycles_start_indices = jnp.full((cycle_count_max,), NAI, uint)
    cycles_edge_count = jnp.full((cycle_count_max,), NAI, uint)
    cycle_count = uint(0)

    graph_flood_params = GraphFloodFromPointParams(
        point_processed,
        graph,
        cycles_start_indices,
        cycle_count,
        point_cycle_id,
        point_cycle_next_point,
        point_cycle_previous_point,
        cycles_edge_count)

    # For debugging
    # for i in range(point_count):
    #     print(f"{i}/{point_count}")
    #     graph_flood_params = graph_flood_from_point(i, graph_flood_params)
    graph_flood_params: GraphFloodFromPointParams = lax.fori_loop(
        0,
        point_count,
        graph_flood_from_point,
        graph_flood_params)

    # Unpack for loop parameters
    cycles_start_indices = graph_flood_params.cycles_start_indices
    cycle_count = graph_flood_params.cycle_count
    point_cycle_id = graph_flood_params.point_cycle_id
    point_cycle_next_point = graph_flood_params.point_cycle_next_point
    point_cycle_previous_point = graph_flood_params.point_cycle_previous_point
    cycles_edge_count = graph_flood_params.cycles_edge_count

    # Reshape data for concatenation along the first axis
    cycles_start_indices = cycles_start_indices.reshape((-1, 1))
    cycles_edge_count = cycles_edge_count.reshape((-1, 1))

    # Create cycle data
    cycle_data = jnp.concatenate(
        (cycles_start_indices, cycles_edge_count), axis=1)

    point_cycle_next_point = point_cycle_next_point.reshape((-1, 1))
    point_cycle_previous_point = point_cycle_previous_point.reshape((-1, 1))
    adjacency_list = jnp.concatenate(
        (point_cycle_next_point, point_cycle_previous_point), axis=1)
    point_data = jnp.concatenate(
        (adjacency_list,
         point_cycle_id.reshape(-1, 1)), axis=1)
    return Cycle(
        PointData(graph.point,
                  point_data),
        cycle_data,
        cycle_count)


def compile_create_from_graph(
        point_1dcount: int,
        cycle_count_max: int,
        device: xla_client.Device) -> tuple[jax.stages.Compiled, float]:
    """Compile `create_from_graph`.

    This function compiles `create_from_graph` for the specified device.

    Parameters
    ----------
    point_1dcount: int
        The number of points/vertices of the graph.
    cycle_count_max : int
        The maximum number of cycles the structure can have.
    device : Device
        The device the compiled function will run on.
        Available devices can be retrieved via `jax.devices()`.

    Returns
    -------
    tuple[jax.stages.Compiled, float]
        out1
            The function returns `create_from_graph` compiled for the specified
            device.
        out2
            The duration of the compilation (seconds).
    """
        
    point_shape_dtype = ShapeDtypeStruct((point_1dcount, 2), float_)
    point_data_shape_dtype = ShapeDtypeStruct((point_1dcount, 2), float_)
    graph_shape_dtype = PointData(point_shape_dtype, point_data_shape_dtype)

    start = time.perf_counter()
    func_jit = jit(
        create_from_graph, static_argnums=1, device=device)

    func_lowered = func_jit.lower(
        graph_shape_dtype, cycle_count_max)
    func_compiled = func_lowered.compile()
    stop = time.perf_counter()
    exec_time = stop - start

    return func_compiled, exec_time


def point_tangent_half_distance_to_segment(
        point_i: int,
        segment_j: int,
        cycle: Cycle) -> jnp.ndarray:
    """Compute the tangent half distance to a given edge.

    This function computes the average of the two radii of the two smallest
    circles tangent to a cycle vertex's point i and passing through only one
    point of the vertex j's edges.

    Parameters
    ----------
    i : int
        The index of one vertex on the cycle. The function uses its point and
        two tangents to compute the smallest circles tangent to i and passing
        through j. There are two tangents because a cycle is a closed polygonal
        chain (or several), where each vertex has two edges representing two
        segments. The two smallest circles tangent to each segment and passing
        through j are computed, and the average of their radii is returned.
    j : ndarray
        The index of one vertex on the cycle. The function uses its two edges
        to compute the smallest circle tangent to i and passing through only
        one point of its edges. The function returns the smallest radius of all
        the computed smallest circles. The two edges are sampled with N points.
        N = 3 is hardcoded in the code.
    cycle : Cycle
        The input cycle's data must follow the specification stated in the
        return section of `create_from_graph`'s documentation.

    Returns
    -------
    float
        The average of the two radii of the two smallest circles tangent to the
        cycle vertex's point i and passing through only one point of the vertex
        j's edges. `MAX_FLOAT` if vertex i and j share edges with the same
        endpoints.
    """
    indices_i = jnp.array([cycle.point_data.data[point_i, 1],
                          point_i, cycle.point_data.data[point_i, 0]])
    i_mask = jnp.equal(indices_i[0], NAI)
    indices_i = jnp.where(i_mask, uint(0), indices_i)
    points_i = cycle.point_data.point[indices_i]
    points_i = jnp.where(jnp.isnan(points_i),
                         jnp.zeros_like(points_i), points_i)
    # Compute the two tangents of the two segments associated with vertex i.
    tangents = vmap(tangent, (0, 0))(points_i[:2], points_i[1:])

    indices_j = jnp.array([cycle.point_data.data[segment_j, 1],
                          segment_j, cycle.point_data.data[segment_j, 0]])
    indices_j_mask = jnp.equal(indices_j[0], NAI)
    indices_j = jnp.where(indices_j_mask, uint(0), indices_j)
    edge_points_j = cycle.point_data.point[indices_j]
    edge_points_j = jnp.where(jnp.isnan(edge_points_j),
                              jnp.zeros_like(edge_points_j), edge_points_j)

    EDGE_SAMPLES = 3
    points_j = jnp.zeros((EDGE_SAMPLES*2, points_i.shape[1]))
    j_jp1 = edge_points_j[2] - edge_points_j[1]
    j_jm1 = edge_points_j[0] - edge_points_j[1]
    for i_edge_sample in range(EDGE_SAMPLES):
        t = i_edge_sample / (EDGE_SAMPLES-1.)
        sample_i = edge_points_j[1] + t * j_jp1
        sample_i2 = edge_points_j[1] + t * j_jm1
        points_j = points_j.at[i_edge_sample*2].set(sample_i)
        points_j = points_j.at[i_edge_sample*2+1].set(sample_i2)

    invalid_j = vmap(jnp.equal, (0, None))(indices_i, indices_j)
    invalid_j = jnp.any(invalid_j)

    # Shape (point on one of the two edges, tangent)
    circle_radii = vmap(
        vmap(
            circle_smallest_radius_tangent_to_p_passing_through_q,
            (None, 0, None)),
        (None, None, 0))(
        points_i[1], points_j, tangents)

    circle_radii = jnp.where(invalid_j, FLOAT_MAX, circle_radii)
    circle_radii = jnp.where(i_mask, FLOAT_MAX, circle_radii)
    circle_radii = jnp.where(indices_j_mask, FLOAT_MAX, circle_radii)

    # Take the minimum radius from all the edge's sampled points
    # Shape (tangent)
    circle_radii_min = jnp.min(circle_radii, axis=1)

    # Average the radius computed with the two tangents
    # 0.5 is not factorized otherwize inf is returned instead of FLOAT_MAX
    return circle_radii_min[0] * 0.5 + circle_radii_min[1] * 0.5


def point_tangent_half_distance_to_neighboring_segments_x(
        cycle: Cycle,
        i: uint,
        neighbor_radius: uint,
        grid2: Grid) -> float:
    """Compute the minimum tangent half distance to neighbors.

    This function computes the minimum tangent half distance from a point on a
    cycle to neighboring segments. The computation is accelerated with a
    spatial data structure, here, a uniform grid. The tangent half distance is
    defined as the radius of the smallest circle tangent to the given point and
    passing through another point, here, one on a neighboring segment.

    Parameters
    ----------
    cycle : Cycle
        The input cycle's data must follow the specification stated in the
        return section of `create_from_graph`'s documentation. In addition,
        each point must be associated with a unique edge of a 2D uniform grid.
        This requirement is needed for the acceleration of the neighboring
        request. One possibility to have a cycle that fulfills the requirement
        is to compute the contours of a scalar field with a contouring
        algorithm, with all the boundary cells having a negative or positive
        value. The returned graph must then be passed to `create_from_graph`,
        which in this case, produces a valid cycle for this function.
    i : uint
        The index of the point whose tangent will be used to compute the
        tangent half distance.
    neighbor_radius : uint
        The neighbor radius gives the maximum difference of edge 2D indices'
        components to belong to the neighborhood.
    grid2 : Grid
        The parameters of the acceleration 2D grid for the neighboring cycle's
        edges request.

    Returns
    -------
    float
        The minimum tangent half distance from a point on a cycle to
        neighboring segments.
    """

    grid2_edge_2dcount = edge.count2_per_axis(grid2.cell_ndcount)

    mask = jnp.equal(i, NAI)
    i = jnp.where(mask, 0, i)
    point_i = cycle.point_data.point[i]
    mask = jnp.where(jnp.isnan(point_i[0]), True, mask)
    point_i = jnp.where(mask, jnp.zeros_like(point_i), point_i)

    # Get neighbors within distance specified by user
    point_2dindex, point_axis = edge.index2_from_1dindex(i, grid2_edge_2dcount)
    neighbors_masked_2dindex = edge.neighboring_2dindices(
        point_2dindex, point_axis, grid2.cell_ndcount, neighbor_radius)
    neighbors_mask = jnp.ravel(neighbors_masked_2dindex.mask)
    neighbors_indices = vmap(
        vmap(
            edge.index1_from_2dindex,
            (0, None, None)),
        (0, 0, None))(
        neighbors_masked_2dindex.array,
        jnp.array([0, 1]),
        grid2_edge_2dcount)
    neighbors_indices = jnp.ravel(neighbors_indices)
    neighbors_mask = jnp.logical_or(
        jnp.isnan(cycle.point_data.point[neighbors_indices, 0]),
        neighbors_mask)

    radii = vmap(point_tangent_half_distance_to_segment,
                 (None, 0, None))(i, neighbors_indices, cycle)

    radii = jnp.where(mask, FLOAT_MAX, radii)
    radii = jnp.where(neighbors_mask, FLOAT_MAX, radii)
    radii_min = jnp.min(radii)

    return radii_min


def points_tangent_half_distance_to_neighboring_segments_x(
        cycles: Cycle,
        neighbor_radius: uint,
        grid2: Grid) -> jnp.ndarray:
    """Compute the minimum half distance to neighbors for all the points.

    This function computes for all the points of the given cycle the minimum
    tangent half distance from each point on a cycle to neighboring segments.
    The computation is accelerated with a spatial data structure, here, a
    uniform grid. The tangent half distance is defined as the radius of the
    smallest circle tangent to the given point and passing through another
    point, here, one on a neighboring segment.

    Parameters
    ----------
    cycle : Cycle
        The input cycle's data must follow the specification stated in the
        return section of `create_from_graph`'s documentation. In addition,
        each point must be associated with a unique edge of a 2D uniform grid.
        This requirement is needed for the acceleration of the neighboring
        request. One possibility to have a cycle that fulfills the requirement
        is to compute the contours of a scalar field with a marching square,
        with all the boundary cells having a negative or positive value. The
        returned graph must then be passed to `create_from_graph`, which in
        this case, produces a valid cycle for this function.
    neighbor_radius : uint
        The neighbor radius gives the maximum difference of edge 2D indices'
        components to belong to the neighborhood.
    grid2 : Grid
        The parameters of the acceleration 2D grid for the neighboring cycle's
        edges request.

    Returns
    -------
    ndarray
        The function returns an array containing the minimum tangent half
        distance to neighboring segments for each point of the cycle. The array
        has one dimension and has `cycles.point_data.point.shape[0]` distances.
    """
    point_1dindices = jnp.arange(0, cycles.point_data.point.shape[0], 1, uint)
    min_circle_radius = vmap(
        point_tangent_half_distance_to_neighboring_segments_x,
        (None, 0, None, None))(
            cycles,
            point_1dindices,
            neighbor_radius,
            grid2)
    return min_circle_radius


def shape_dtype(
        point_1dcount: int,
        point_data_size: int,
        point_data_dtype: Any,
        cycle_count: int) -> Cycle:
    """Return the cycle shape and data type.

    Parameters
    ----------
    point_1dcount : int
        The number of points of all the cycles.
    point_data_size : int
        The size of the data of the points of the cycles.
    point_data_dtype : Any
        The type of the data of the points of the cycles.
    cycle_count : int
        The number of cycles.
    Returns
    -------

    Cycle
        Each member of the cycle contains its shape and its data type.
    """

    contour_point_shape_dtype = ShapeDtypeStruct(
        (point_1dcount, 2),
        float_)
    cycles_point_data_data_shape_dtype = ShapeDtypeStruct(
        (point_1dcount, point_data_size),
        point_data_dtype)
    cycles_point_data_shape_dtype = PointData(
        contour_point_shape_dtype,
        cycles_point_data_data_shape_dtype)
    cycles_data_shape_dtype = ShapeDtypeStruct(
        (cycle_count, 2),
        uint)
    cycles_cycle_count = ShapeDtypeStruct((), uint)
    cycles_shape_dtype = Cycle(
        cycles_point_data_shape_dtype,
        cycles_data_shape_dtype,
        cycles_cycle_count)
    return cycles_shape_dtype


def compile_points_tangent_half_distance_to_neighboring_segments_x(
        point_1dcount: int,
        device: xla_client.Device,
        neighbor_radius: uint,
        cycle_count_max: int) -> tuple[jax.stages.Compiled, float]:
    """Compile `points_tangent_half_distance_to_neighboring_segments_x`.

    Parameters
    ----------
    point_1dcount : int
        The number of points/vertices in the cycles.
    device : Device
        The device the compiled function will run on.
        Available devices can be retrieved via `jax.devices()`.
    neighbor_radius : uint
        The neighbor radius gives the maximum difference of edge 2D indices'
        components to belong to the neighborhood.
    cycle_count_max : int
        The maximum number of cycles.

    Returns
    -------
    tuple[Compiled, float]
        out1
            The function returns
            `points_tangent_half_distance_to_neighboring_segments_x` compiled
            for the specified device and contour graph vertex 2D count.
        out2
            The duration of the compilation (seconds).
    """

    cycles_shape_dtype = shape_dtype(point_1dcount, 2, uint, cycle_count_max)
    uniform_grid_shape_dtype = shape_dtype_from_dim(2)
    # cycles_points_min_radius_circle_to_neighboring_curves_x compilation
    start = time.perf_counter()
    func_jit = jit(
        points_tangent_half_distance_to_neighboring_segments_x,
        device=device, static_argnums=1)
    func_lowered = func_jit.lower(
        cycles_shape_dtype,
        neighbor_radius,
        uniform_grid_shape_dtype)
    func_compiled = func_lowered.compile()
    stop = time.perf_counter()

    return func_compiled, stop - start


def to_polyline(cycle: Cycle, polyline: Polyline) -> Polyline:
    """This function converts a cycle to a polyline.

    Unlike a polyline, a cycle has indirect access to its points and data.

    Parameters
    ----------
    cycle : Cycle
        The input cycle's data must follow the specification stated in the
        return section of `create_from_graph`'s documentation.
    polyline : Polyline
        An empty polyline with enough space to store the cycle data. It is
        possible to use `to_polyline_full_nan`.

    Returns
    -------
    Polyline
        The polyline from the cycle.
    """

    def add_points_with_data(cycle_iterator_data):
        entering_loop = cycle_iterator_data[0]
        polyline_point = cycle_iterator_data[1]
        cycle_index = cycle_iterator_data[2]
        polyline_point_index = cycle_iterator_data[3]
        cycle = cycle_iterator_data[4]
        cycle_point_index = cycle_iterator_data[5]
        polyline_point_data = cycle_iterator_data[6]
        cycle_point_index_p1 = cycle_iterator_data[7]
        cycle_point_index_0 = cycle_iterator_data[8]

        entering_loop = False

        polyline_point = \
            polyline_point.at[cycle_index, polyline_point_index].set(
                cycle.point_data.point[cycle_point_index])
        polyline_point_data = \
            polyline_point_data.at[cycle_index, polyline_point_index].set(
                cycle.point_data.data[cycle_point_index])

        # Next edge on cycle
        cycle_point_index_p2 = uint(
            cycle.point_data.data[cycle_point_index_p1, 0])
        cycle_point_index = cycle_point_index_p1
        cycle_point_index_p1 = cycle_point_index_p2
        polyline_point_index += 1

        return (
            entering_loop,
            polyline_point,
            cycle_index,
            polyline_point_index,
            cycle,
            cycle_point_index,
            polyline_point_data,
            cycle_point_index_p1,
            cycle_point_index_0)

    def cycle_is_not_finished(cycle_iterator_data):
        entering_loop = cycle_iterator_data[0]
        cycle_point_index = cycle_iterator_data[5]
        cycle_point_index_0 = cycle_iterator_data[8]

        cycle_is_not_finished = jnp.not_equal(
            cycle_point_index, cycle_point_index_0)
        return jnp.logical_or(cycle_is_not_finished, entering_loop)

    def operate_on_each_cycle(i, cycles_iterator_data):
        polyline_data = cycles_iterator_data[0]
        cycles_data_sorted = cycles_iterator_data[1]
        cycle: Cycle = cycles_iterator_data[2]
        polyline_point = cycles_iterator_data[3]
        polyline_point_data = cycles_iterator_data[4]

        polyline_data = polyline_data.at[i].set(cycles_data_sorted[i])
        # Get starting point index
        cycle_point_index_0 = uint(cycles_data_sorted[i, 0])
        # Initialize current point index
        cycle_point_index = cycle_point_index_0
        # Initialize next point index
        cycle_point_index_p1 = uint(
            cycle.point_data.data[cycle_point_index, 0])

        entering_loop = True
        polyline_point_index = 0

        cycle_iterator_data = (
            entering_loop,
            polyline_point,
            i,
            polyline_point_index,
            cycle,
            cycle_point_index,
            polyline_point_data,
            cycle_point_index_p1,
            cycle_point_index_0)

        cycle_iterator_data = lax.while_loop(
            cycle_is_not_finished, add_points_with_data, cycle_iterator_data)

        polyline_point = cycle_iterator_data[1]
        polyline_point_data = cycle_iterator_data[6]

        return (polyline_data,
                cycles_data_sorted,
                cycle,
                polyline_point,
                polyline_point_data)

    cycles_data = cycle.cycle_data
    # cycles_data[:, 1]: Cycles edge count
    # Sort with increasing cycle edge count
    argsort_cycles = jnp.argsort(cycles_data[:, 1])
    cycles_data_sorted = cycles_data[argsort_cycles]

    polyline_point = polyline.point
    polyline_point_data = polyline.point_data
    polyline_data = polyline.data

    # Pack data for the loop
    cycles_iterator_data = (
        polyline_data,
        cycles_data_sorted,
        cycle,
        polyline_point,
        polyline_point_data)

    cycles_iterator_data = lax.fori_loop(
        uint(0),
        cycle.cycle_count,
        operate_on_each_cycle,
        cycles_iterator_data)

    # Unpack
    polyline_data = cycles_iterator_data[0]
    polyline_point = cycles_iterator_data[3]
    polyline_point_data = cycles_iterator_data[4]

    return Polyline(polyline_point, polyline_point_data, polyline_data)


def to_polyline_full_nan(cycle: Cycle) -> Polyline:
    """Allocates a polyline full of nan from a cycle.

    Parameters
    ----------
    cycle : Cycle
        The input cycle's data must follow the specification stated in the
        return section of `create_from_graph`'s documentation.

    Return
    ------
    Polyline
        The function returns a Polyline full of `nan`. The shape of the
        attribute `point` 3darray is (C, P, N), where C is the number of
        cycles, P is the number of points in the cycle buffer, and N is the
        number of component of a point. The attribute `point_data` shape is (C,
        P, M), where M is the number of items associated with each vertex. The
        attribute `data` shape is (C, Q), where Q is the number of items
        associated with each cycle.
    """
    polyline_point_shape = (
        cycle.cycle_count,
        cycle.point_data.point.shape[0],
        cycle.point_data.point.shape[1])
    polyline_point_data_shape = (
        cycle.cycle_count,
        cycle.point_data.point.shape[0],
        cycle.point_data.data.shape[1])
    polyline_data_shape = (cycle.cycle_count, cycle.cycle_data.shape[1])
    return Polyline(
        jnp.full(polyline_point_shape, jnp.nan),
        jnp.full(polyline_point_data_shape, jnp.nan),
        jnp.full(polyline_data_shape, jnp.nan))


def compile_to_polyline(
        point_1dcount: int,
        polyline_shape_dtype: Polyline,
        device: xla_client.Device,
        cycle_count_max: int) -> tuple[jax.stages.Compiled, float]:
    """Compile `to_polyline`.

    Parameters
    ----------
    point_1dcount : int
        The number of vertices/points along each axis of the contour grid.
    polyline_shape_dtype : Polyline
        The shape of the polyline. Get it with
        `polyline_and_shape_dtype_from_2dgrid`.
    device : Device
        The device the compiled function will run on.
        Available devices can be retrieved via `jax.devices()`.
    cycle_count_max : int
        The maximum number of cycles.

    Returns
    -------
    tuple[jax.stages.Compiled, float]
        out1
            The compiled function.
        out2
            The duration of the compilation (seconds).
    """

    cycle_shape_dtype = shape_dtype(
        point_1dcount,
        polyline_shape_dtype.point_data.shape[2],
        float_,
        cycle_count_max)

    start = time.perf_counter()
    cycles_to_polylines_jit = jit(to_polyline, device=device)
    cycles_to_polylines_lowered = cycles_to_polylines_jit.lower(
        cycle_shape_dtype, polyline_shape_dtype)
    cycles_to_polylines_compiled = cycles_to_polylines_lowered.compile()
    stop = time.perf_counter()

    return cycles_to_polylines_compiled, stop - start


def edge_minimize_patching_energy_wrt_segments(
        edge_i1: int,
        edges_j1: jnp.ndarray,
        edges_j1_mask: jnp.ndarray,
        cycles: Cycle) -> tuple[float, int]:
    """Minimize the patching energy with respect to edges/segments.

    This function finds the vertex in j1 or j0 leading to the minimum patching
    energy between the segment with start point `i1` and the segment with start
    point j in j1 or j0. Here, j0 is defined as the previous vertices to j1's
    vertices, i.e., `cycles.point_data.data[j1, 1]`, see the "Returns" section
    of `create_from_graph`. Here, a segment with start point `i` has end point
    `cycles.point_data.data[i, 0]`.

    Parameters
    ----------
    edge_i1 : int
        The index of the first segment's start point used to calculate energy.
    edges_j1 : ndarray
        The index of the second segment's start point (or end point) used to
        calculate energy belongs to `j1`.
    edges_j1_mask : ndarray
        The mask of array `j1`. Masked vertices will not be considered when
        computing the argmin.
    cycles : Cycle
        The indices are vertices of a graph representing several cycles.
    Returns
    -------
    tuple[float, int]
        out1 :
            The minimum patching energy.
        out2 :
            The index in `j1` leading to the minimum patching energy. The index
            cannot have the same cycle ID as `i1`.
    """

    i1_cycle_id = cycles.point_data.data[edge_i1, 2]
    # i1   i2
    # x----x
    i2 = cycles.point_data.data[edge_i1, 0]
    # Position of the vertex i1 and i2
    p_i1 = cycles.point_data.point[edge_i1]
    p_i2 = cycles.point_data.point[i2]

    # Point can be masked (first component is nan) and user can specify a mask
    # (j_mask parameter).
    # This line merges the two masks
    edges_j1_mask = jnp.logical_or(
        jnp.isnan(cycles.point_data.point[edges_j1, 0]), edges_j1_mask)
    # j0   j1   j2
    # x----x----x
    j2 = cycles.point_data.data[edges_j1, 0]
    j0 = cycles.point_data.data[edges_j1, 1]

    p_j1 = cycles.point_data.point[edges_j1]
    # Duplicate j1 points
    p_j1j1 = jnp.concatenate((p_j1, p_j1), axis=0)

    j1j0 = jnp.concatenate((edges_j1, j0))
    j1j0_mask = jnp.concatenate((edges_j1_mask, edges_j1_mask))

    j1j0_cycle_id = cycles.point_data.data[j1j0, 2]
    j1j0_i1_is_same_cycle_id = jnp.equal(j1j0_cycle_id, i1_cycle_id)
    j1j0_mask = jnp.logical_or(j1j0_i1_is_same_cycle_id, j1j0_mask)

    p_j2 = cycles.point_data.point[j2]
    p_j2j0 = jnp.concatenate((p_j2, cycles.point_data.point[j0]), axis=0)

    cost = vmap(
        patching_energy, (None, None, 0, 0))(
        p_i1, p_i2, p_j1j1, p_j2j0)
    cost = jnp.where(j1j0_mask, FLOAT_MAX, cost)

    best_j = jnp.argmin(cost)
    return cost[best_j], j1j0[best_j]


def neighboring_edge_with_minimum_patching_energy(
        cycle_id: int,
        cycles: Cycle,
        grid_cell_2dcount: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    """Return the pair of edges/segments giving the minimum patching energy.

    This function returns the pair of edges/segments giving the minimum
    patching energy. One edge belongs to the given cycle, and the other
    doesn't, but the other belongs in the neighborhood of the given cycle. The
    vertices/points are located on a 2D grid's edges to accelerate neighborhood
    requests.

    Parameters
    ----------
    cycle_id : int
        The ID of the cycle from which neighborhood requests are performed.
    cycles : Cycle
        The set of cycles.
    grid_cell_2dcount : ndarray
        The number of cells along the x and y-axis of the 2D grid.

    Returns
    -------
    tuple[ndarray, float]
        out1:
            A pair of unsigned integers indicates the two edges with the
            minimum patching energy. A edge is represented with its start
            vertex's index.
        out2:
            The patching energy of the pair of edges.
    Notes
    -----
        See `edge_with_minimum_patching_energy` for a version searching the
        best edge in the set of all the edges.
    """

    def cycle_not_finished(cycle_iterator_params):
        entering_loop = cycle_iterator_params[0]
        cycle_point_i = cycle_iterator_params[1]
        cycle_point_i_0 = cycle_iterator_params[7]

        not_at_the_end = jnp.not_equal(cycle_point_i, cycle_point_i_0)
        return jnp.logical_or(not_at_the_end, entering_loop)

    def compute_minimum_stiching_cost(cycle_iterator_params):
        cycle_point_i = cycle_iterator_params[1]
        cycles: Cycle = cycle_iterator_params[2]
        grid_cell_2dcount = cycle_iterator_params[3]
        cost_min = cycle_iterator_params[4]
        best_edge_pair = cycle_iterator_params[5]
        cycle_point_i_p1 = cycle_iterator_params[6]
        cycle_point_i_0 = cycle_iterator_params[7]

        # Hardcoded value
        NEIGHBORHOOD_RADIUS = 2
        j_indices, j_mask = edge.neighboring2_1dindices(
            cycle_point_i, grid_cell_2dcount, NEIGHBORHOOD_RADIUS)
        j_indices_ip1, j_mask_ip1 = edge.neighboring2_1dindices(
            cycle_point_i_p1, grid_cell_2dcount, NEIGHBORHOOD_RADIUS)
        j_indices = jnp.concatenate((j_indices, j_indices_ip1))
        j_mask = jnp.concatenate((j_mask, j_mask_ip1))
        neighbor_point_cost_min, neighbor_point_min_cost_i = \
            edge_minimize_patching_energy_wrt_segments(
                cycle_point_i, j_indices, j_mask, cycles)

        # Update minimum cost if needed
        cost_min_to_update = jnp.greater(cost_min, neighbor_point_cost_min)
        valid_neighbor_point = jnp.not_equal(
            neighbor_point_cost_min, FLOAT_MAX)
        update_cost_min = jnp.logical_and(
            cost_min_to_update, valid_neighbor_point)

        cost_min = jnp.where(
            update_cost_min, neighbor_point_cost_min, cost_min)
        best_edge_pair = jnp.where(update_cost_min, jnp.array(
            [cycle_point_i, neighbor_point_min_cost_i]), best_edge_pair)

        # Next edge on cycle
        cycle_point_i_p2 = cycles.point_data.data[cycle_point_i_p1, 0]
        cycle_point_i = cycle_point_i_p1
        cycle_point_i_p1 = cycle_point_i_p2

        return (
            False,
            cycle_point_i,
            cycles,
            grid_cell_2dcount,
            cost_min,
            best_edge_pair,
            cycle_point_i_p1,
            cycle_point_i_0)

    cost_min = FLOAT_MAX
    # (best cycle edge, best neighboring edge)
    best_edge_pair = jnp.array([NAI, NAI])

    cycle_point_i_0 = cycles.cycle_data[cycle_id, 0]

    cycle_point_i = cycle_point_i_0
    cycle_point_i_p1 = cycles.point_data.data[cycle_point_i, 0]
    entering_loop = True

    cycle_iterator_params = (
        entering_loop,
        cycle_point_i,
        cycles,
        grid_cell_2dcount,
        cost_min,
        best_edge_pair,
        cycle_point_i_p1,
        cycle_point_i_0)

    # For debug
    # while cycle_not_finished(cycle_iterator_params):
    #     cycle_iterator_params = compute_minimum_stiching_cost(
    #         cycle_iterator_params)
    cycle_iterator_params = lax.while_loop(
        cycle_not_finished,
        compute_minimum_stiching_cost,
        cycle_iterator_params)

    cost_min = cycle_iterator_params[4]
    best_edge_pair = cycle_iterator_params[5]

    return best_edge_pair, cost_min


def compile_neighboring_edge_with_minimum_patching_energy(
        point_1dcount: int,
        device: xla_client.Device,
        cycle_count_max: int) -> tuple[jax.stages.Compiled, float]:
    """Compile `neighboring_edge_with_minimum_patching_energy`.

    Parameters
    ----------
    point_1dcount : int
        The number of points of the cycles (included masked).
    device : Device
        The device the compiled function will run on.
        Available devices can be retrieved via `jax.devices()`.
    cycle_count_max : int
        The maximum number of cycles.

    Returns
    -------
    tuple[Compiled, float]
        out1
            The compiled function.
        out2
            The duration of the compilation (seconds).
    """

    cycles_shape_dtype = shape_dtype(
        point_1dcount, 3, uint, cycle_count_max)

    start = time.perf_counter()
    cycle_id_with_min_edge_count_shape_dtype = ShapeDtypeStruct((), int_)
    func_jit = jit(
        neighboring_edge_with_minimum_patching_energy,
        device=device)
    func_lowered = func_jit.lower(
        cycle_id_with_min_edge_count_shape_dtype,
        cycles_shape_dtype,
        ShapeDtypeStruct((2,), int_))
    func_compiled = func_lowered.compile()
    stop = time.perf_counter()

    return func_compiled, stop - start


def edge_with_minimum_patching_energy(
        cycle_id: int,
        cycles: Cycle) -> jnp.ndarray:
    """Return the pair of edges/segments giving the minimum patching energy.

    One edge belongs to the given cycle, and the other doesn't.

    cycle_id : int
        The ID of the cycle from which neighborhood requests are performed.
    cycles : Cycle
        The set of cycles.

    Returns
    -------
    tuple[jnp.ndarray, float]
        out1:
            A pair of unsigned integers indicates the two edges with the
            minimum patching energy. A edge is represented with its start
            vertex's index.
        out2:
            The patching energy of the pair of edges.

    Notes
    -----
    See `neighboring_edge_with_minimum_patching_energy` for a version searching
    the best edge in the neighborhood of the given cycle.
    """

    def compute_minimum_stiching_cost(cycle_iterator_params):
        # Unpack data
        cycle_point_i = cycle_iterator_params[1]
        points_1dindices = cycle_iterator_params[2]
        points_1dindices_mask = cycle_iterator_params[3]
        cycle: Cycle = cycle_iterator_params[4]
        cost_min = cycle_iterator_params[5]
        best_edge_pair = cycle_iterator_params[6]
        cycle_point_i_p1 = cycle_iterator_params[7]
        cycle_point_i_0 = cycle_iterator_params[8]

        point_min_cost, point_min_cost_i = \
            edge_minimize_patching_energy_wrt_segments(
                cycle_point_i,
                points_1dindices,
                points_1dindices_mask,
                cycle)

        cost_min_to_update = jnp.greater(cost_min, point_min_cost)
        valid_neighbor_point = jnp.not_equal(point_min_cost, FLOAT_MAX)
        update_cost_min = jnp.logical_and(
            cost_min_to_update, valid_neighbor_point)
        cost_min = jnp.where(update_cost_min, point_min_cost, cost_min)
        best_edge_pair = jnp.where(update_cost_min, jnp.array(
            [cycle_point_i, point_min_cost_i], uint), best_edge_pair)

        # Next edge on cycle
        cycle_point_i_p2 = cycle.point_data.data[cycle_point_i_p1, 0]
        cycle_point_i = cycle_point_i_p1
        cycle_point_i_p1 = cycle_point_i_p2

        # Pack data
        return (
            False,
            cycle_point_i,
            points_1dindices,
            points_1dindices_mask,
            cycle,
            cost_min,
            best_edge_pair,
            cycle_point_i_p1,
            cycle_point_i_0)

    def cycle_not_finished(cycle_iterator_params):
        # Unpack data
        entering_loop = cycle_iterator_params[0]
        cycle_point_i = cycle_iterator_params[1]
        cycle_point_i_0 = cycle_iterator_params[8]

        not_at_the_end = jnp.not_equal(cycle_point_i, cycle_point_i_0)
        return jnp.logical_or(not_at_the_end, entering_loop)

    cost_min = FLOAT_MAX
    # (best cycle edge, best neighboring edge)
    best_edge_pair = jnp.array([NAI, NAI])

    cycle_point_i_0 = cycles.cycle_data[cycle_id, 0]

    cycle_point_i = cycle_point_i_0
    cycle_point_i_p1 = cycles.point_data.data[cycle_point_i, 0]
    entering_loop = True

    points_1dindices = jnp.arange(cycles.point_data.point.shape[0])
    points_1dindices_mask = jnp.full((points_1dindices.shape[0],), False)

    # Pack data
    cycle_iterator_params = (
        entering_loop,
        cycle_point_i,
        points_1dindices,
        points_1dindices_mask,
        cycles,
        cost_min,
        best_edge_pair,
        cycle_point_i_p1,
        cycle_point_i_0)

    # For debug
    # while cond_fun(val):
    #     val = body_fun(val)
    cycle_iterator_params = lax.while_loop(
        cycle_not_finished,
        compute_minimum_stiching_cost,
        cycle_iterator_params)

    # Retrieve the important data
    cost_min = cycle_iterator_params[5]
    best_edge_pair = cycle_iterator_params[6]

    return best_edge_pair, cost_min


def compile_edge_with_minimum_patching_energy(
        point_1dcount: int,
        device: xla_client.Device,
        cycle_count_max: int):
    """Compile `edge_with_minimum_patching_energy`.

    Parameters
    ----------
    point_1dcount : int
        The number of points in the cycles.
    device : Device
        The device the compiled function will run on.
        Available devices can be retrieved via `jax.devices()`.
    cycle_count_max : int
        The maximum number of cycles.

    Returns
    -------
    tuple[jax.stages.Compiled, float]
        out1
            The compiled function.
        out2
            The duration of the compilation (seconds).
    """

    cycles_shape_dtype = shape_dtype(
        point_1dcount, 4, uint, cycle_count_max)
    start = time.perf_counter()
    func_jit = jit(
        edge_with_minimum_patching_energy, device=device)
    func_lowered = func_jit.lower(
        ShapeDtypeStruct((), int_), cycles_shape_dtype)
    func_compiled = func_lowered.compile()
    stop = time.perf_counter()

    return func_compiled, stop - start


def stitch_two_edges(i: uint, j: uint, cycles: Cycle) -> Cycle:
    """Stitch two edges belonging to two different cycles.

    Parameters
    ----------
    i : uint
        The index of the start vertex of the first edge.
    j : uint
        The index of the start vertex of the second edge.
    cycles : Cycle
        Represent the set of cycles.

    Returns
    -------
    Cycle
        The modified cycle, where the two edges are stitched. The number of
        cycles is decreased by one (`cycles.cycle_count`), and the cycle of `i`
        takes the ID and the orientation of the cycle of `j`. The data
        associated with the cycle of `i` is filled with the maximum value of
        unsigned int.
    """

    def modify_cycle_point_id_and_adjacency(cycle_i_iterator_params):
        points_cycle_ids = cycle_i_iterator_params[0]
        point_i = cycle_i_iterator_params[1]
        cycle_id_j = cycle_i_iterator_params[2]
        points_adjacencies = cycle_i_iterator_params[3]
        point_i_p1 = cycle_i_iterator_params[4]
        point_i_m1 = cycle_i_iterator_params[5]
        jp1 = cycle_i_iterator_params[6]

        points_cycle_ids = points_cycle_ids.at[point_i].set(cycle_id_j)
        points_adjacencies = points_adjacencies.at[point_i].set(
            jnp.array([point_i_p1, point_i_m1]))

        # Next edge on cycle
        point_i_p2 = jnp.where(
            points_adjacencies[point_i_p1][0] == point_i,
            points_adjacencies[point_i_p1][1],
            points_adjacencies[point_i_p1][0])
        point_i_m1 = point_i
        point_i = point_i_p1
        point_i_p1 = point_i_p2

        return (points_cycle_ids,
                point_i,
                cycle_id_j,
                points_adjacencies,
                point_i_p1,
                point_i_m1,
                jp1)

    def cycle_i_iteration_not_finished(cycle_i_iterator_params):
        point_i = cycle_i_iterator_params[1]
        jp1 = cycle_i_iterator_params[6]

        return jnp.not_equal(point_i, jp1)

    # Start of the function's code

    cycle_id_i = cycles.point_data.data[i, 2]
    cycle_id_j = cycles.point_data.data[j, 2]

    im1 = cycles.point_data.data[i, 1]
    ip1 = cycles.point_data.data[i, 0]
    ip2 = cycles.point_data.data[ip1, 0]

    jm1 = cycles.point_data.data[j, 1]
    jp1 = cycles.point_data.data[j, 0]
    jp2 = cycles.point_data.data[jp1, 0]

    # Choose config based on non intersected segments
    points_to_update_indices = jnp.array([i, ip1, j, jp1])
    config_point = cycles.point_data.point[points_to_update_indices]
    config_1_inter = intersection_bool(
        jnp.array([config_point[0], config_point[2]]),
        jnp.array([config_point[1], config_point[3]]))
    config = jnp.where(config_1_inter, 0, 1)

    new_adjacencies = jnp.where(
        config == 0,
        jnp.array([[jp1, im1],
                   [ip2, j],
                   [ip1, jm1],
                   [jp2, i]]),
        jnp.array([[im1, j],
                   [jp1, ip2],
                   [i, jm1],
                   [jp2, ip1]]))

    points_adjacencies = cycles.point_data.data[:, :2]

    # Update vertex adjacency
    points_adjacencies = points_adjacencies.at[points_to_update_indices].set(
        new_adjacencies)

    points_cycle_ids = cycles.point_data.data[:, 2]
    # Starting from j, update vertex cycle ids and adjacency
    # Stop when come back to j's cycle
    point_i = uint(j)
    point_i_p1 = points_adjacencies[point_i, 0]
    point_i_m1 = points_adjacencies[point_i, 1]

    cycle_i_iterator_params = (
        points_cycle_ids,
        point_i,
        cycle_id_j,
        points_adjacencies,
        point_i_p1,
        point_i_m1,
        jp1)

    # while cycle_i_iteration_not_finished(cycle_i_iterator_params):
    #     cycle_i_iterator_params = modify_cycle_point_id_and_adjacency(
    #         cycle_i_iterator_params)
    cycle_i_iterator_params = lax.while_loop(
        cycle_i_iteration_not_finished,
        modify_cycle_point_id_and_adjacency,
        cycle_i_iterator_params)

    points_cycle_ids = cycle_i_iterator_params[0]
    points_adjacencies = cycle_i_iterator_params[3]

    point_data = jnp.concatenate(
        (points_adjacencies,
         points_cycle_ids.reshape((-1, 1))),
        axis=1)

    cycle_edge_count_j = \
        cycles.cycle_data[cycle_id_j, 1] + \
        cycles.cycle_data[cycle_id_i, 1]
    cycles_data = cycles.cycle_data
    cycles_data = cycles_data.at[cycle_id_i, :].set(NAI)
    cycles_data = cycles_data.at[cycle_id_j, 1].set(cycle_edge_count_j)
    cycle_count = cycles.cycle_count - 1
    return Cycle(
        PointData(cycles.point_data.point,
                  point_data),
        cycles_data,
        cycle_count)


def compile_stitch_two_edges(
        point_1dcount: int,
        device: xla_client.Device,
        cycle_count_max: int) -> tuple[jax.stages.Compiled, float]:
    """Compile `stitch_two_edges`.

    Parameters
    ----------
    point_1dcount : int
        The number of points of the cycles.
    device : Device
        The device the compiled function will run on.
        Available devices can be retrieved via `jax.devices()`.
    cycle_count_max : int
        The maximum number of cycles.

    Returns
    -------
    tuple[jax.stages.Compiled, float]
        out1
            The compiled function.
        out2
            The duration of the compilation (seconds).
    """
    cycles_shape_dtype = shape_dtype(point_1dcount, 3, uint, cycle_count_max)
    start = time.perf_counter()
    func_jit = jit(stitch_two_edges, device=device)
    func_lowered = func_jit.lower(
        ShapeDtypeStruct((), uint),
        ShapeDtypeStruct((), uint),
        cycles_shape_dtype)
    func_compiled = func_lowered.compile()
    stop = time.perf_counter()

    return func_compiled, stop - start
