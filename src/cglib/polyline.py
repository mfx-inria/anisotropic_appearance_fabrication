from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct, device_put, jit, vmap
from jax._src.lib import xla_client

from .grid import edge
from .limits import FLOAT_MAX
from .math import (circle_smallest_radius_tangent_to_p_passing_through_q,
                   vector_normalize)
from .segment import tangent
from .type import float_np, uint


class Polyline(NamedTuple):
    """Represent one or several polylines.

    Each polyline has data and points with data.

    Attributes
    ----------
    point : ndarray
        The polyline points are represented by a |P| x |V| x N 3D array, where
        |P| is the number of polylines and |V| is the number of vertices for
        each polyline, and N is the number of components for each point.
    point_data : ndarray
        The data associated with each vertex of the polyline.
    data : ndarray
        The data associated with the polyline(s).
    """
    point: jnp.ndarray
    point_data: jnp.ndarray
    data: jnp.ndarray


def create_from_2dgrid(
        grid_cell_2dcount: np.ndarray,
        device: xla_client.Device,
        polyline_count: int,
        polyline_point_data_size: int = 4) -> tuple[Polyline, Polyline]:
    """Returns a polyline full of `nan` from a 2D grid.

    Parameters
    ----------
    grid_cell_2dcount : ndarray
        The number of cells along each axis of the 2D grid. Represent the 2D
        grid.
    device : Device
        The device the Polyline will be stored.
        Available devices can be retrieved via `jax.devices()`.
    polyline_count : int
        The number of devices to allocate.
    polyline_point_data_size : int (default: 4)
        The number of floats allocated for each point data.

    Returns
    -------
    tuple[Polyline, Polyline]
        out1 :
            A Polyline full of `nan`. The shape of the attribute `point` is
            (`polyline_count`, P, 2), where P is the number of edges of the 2D
            grid. The shape of the attribute `point_data` is (`polyline_count`,
            P, `polyline_point_data_size`). The shape of the attribute `data`
            is (`polyline_count`, 2).
        out2 :
            A Polyline full of `ShapeDtypeStruct`. The shapes of each attribute
            of Polyline is the same as the shape of each attribute of `out1`.
            The data type of all the attributes is `float_np`.
    """

    contour_grid_edge_1dcount = edge.count1_from_cell_2dcount(
        np.array(grid_cell_2dcount))

    polyline_point_shape = (polyline_count, contour_grid_edge_1dcount, 2)
    polyline_point_data_shape = (
        polyline_count, contour_grid_edge_1dcount, polyline_point_data_size)
    polyline_data = (polyline_count, 2)

    polyline_point = device_put(
        np.full(polyline_point_shape, np.nan, float_np), device=device)
    polyline_point_data = device_put(
        np.full(polyline_point_data_shape, np.nan, float_np), device=device)
    polyline_data = device_put(
        np.full(polyline_data, np.nan, float_np), device=device)

    polyline = Polyline(polyline_point, polyline_point_data, polyline_data)

    polyline_shape_dtype = Polyline(
        ShapeDtypeStruct(polyline_point.shape, polyline_point.dtype),
        ShapeDtypeStruct(polyline_point_data.shape, polyline_point_data.dtype),
        ShapeDtypeStruct(polyline_data.shape, polyline_data.dtype))

    return polyline, polyline_shape_dtype


def save(outfile: str,
         polyline: Polyline) -> None:
    """Save the polyline to the disk in uncompressed format `.npz`.

    Parameters
    ----------
    file : str or file
        Either the filename (string) or an open file (file-like object) where
        the data will be saved. If file is a string or a Path, the `.npz`
        extension will be appended to the filename if it is not already there.
    polyline : Polyline
        The polyline to save.

    Returns
    -------
    None
    """
    point = np.array(polyline.point)
    point_data = np.array(polyline.point_data)
    data = np.array(polyline.data)
    np.savez(
        outfile,
        point=point,
        point_data=point_data,
        data=data)


def load(file: str) -> Polyline:
    """Load a polyline from the disk in uncompressed format `.npz`

    Parameters
    ----------
    file : file-like object, string, or pathlib.Path
        The file to read. File-like objects must support the ``seek()`` and
        ``read()`` methods.

    Returns
    -------
    Polyline
        The loaded cycle.
    """
    data_from_file = np.load(file)
    point = data_from_file['point']
    point_data = data_from_file['point_data']
    data = data_from_file['data']

    point_data = Polyline(point, point_data, data)
    return point_data


def load_from_txtfile(file: str) -> Polyline:
    """This function loads a polyline from a text file.

    Parameters
    ----------
    file : string
        The path of the file to read.

    Returns
    -------
    Polyline
        The loaded polyline.
    """
    with open(file, 'r', encoding="utf-8") as f:
        point = []
        for line in f:
            vec2_str = line.split()
            point.append([float(vec2_str[0]), float(vec2_str[1])])
    point_np = np.array(point)
    point_np_x = point_np[:, 0]
    point_np_y = point_np[:, 1]
    polyline_p_min = np.array([np.min(point_np_x), np.min(point_np_y)])
    polyline_p_max = np.array([np.max(point_np_x), np.max(point_np_y)])
    point_count = point_np.shape[0]
    polyline_data = np.array([point_count,
                              polyline_p_min[0],
                              polyline_p_min[1],
                              polyline_p_max[0],
                              polyline_p_max[1]])

    point_data = Polyline(point, None, polyline_data)
    return point_data


def point_tangent_and_weight(
        i: int,
        polyline: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    """Compute the tangent and its associated weight for the vertex i.

    Parameters
    ----------
    i : int
        The index of one vertex on the polyline.
    closed_polyline : jnp.ndarray
        The input polyline must represent a closed polygonal chain. It
        represents a connected series of line segments specified by a sequence
        of points. The parameter has shape (M, N), where M is the number of
        points of the polyline and N is the number of components per point.
        Each point is connected to the previous and next points in the array.
        The last point is connected to the second last one and the first. The
        first point is connected to the last point and the second.

    Returns
    -------
    tuple[jnp.ndarray, float]
        out0 : jnp.ndarray
            The tangent of the vertex i is represented as a unit vector. It is
            defined as the average of its two segments' tangents. A vertex i
            has edges/segments i-1 and i with endpoints (i-1, i) and (i, i+1),
            respectively. If p(i) is the point associated with vertex i, the
            tangent of segment i is defined as [p(i-1) - p(i)] / norm(p(i-1) -
            p(i)).
        out1 : float
            The weight associated with the vertex i. It is defined as the
            average of the vertex i segments' lengths.
    """
    indices_i = jnp.array(
        [(i-1) % polyline.shape[0], i, (i+1) % polyline.shape[0]])
    points_i = polyline[indices_i]
    tangents = vmap(tangent, (0, 0))(points_i[:2], points_i[1:])
    v1 = points_i[1] - points_i[0]
    v2 = points_i[2] - points_i[1]
    v1_norm = jnp.linalg.norm(v1)
    v2_norm = jnp.linalg.norm(v2)
    # Average tangents
    return vector_normalize(tangents[0] + tangents[1]), (v1_norm + v2_norm)*0.5


def points_tangent_and_weight(
        closed_polyline: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the tangent and weight for all the points of the polyline.

    Parameters
    ----------
    closed_polyline : ndarray
        The input polyline must represent a closed polygonal chain. It
        represents a connected series of line segments specified by a sequence
        of points. The parameter has shape (M, N), where M is the number of
        points of the polyline and N is the number of components per point.
        Each point is connected to the previous and next points in the array.
        The last point is connected to the second last one and the first. The
        first point is connected to the last point and the second.

    Returns
    -------
    tuple[ndarray, ndarray]
        out0 : ndarray
            The tangents of the vertices. They are represented as unit vectors.
            They are defined as the average of their two segments' tangents. A
            vertex i has edges/segments i-1 and i with endpoints (i-1, i) and
            (i, i+1), respectively. If p(i) is the point associated with vertex
            i, the tangent of segment i is defined as [p(i-1) - p(i)] /
            norm(p(i-1) - p(i)).
        out1 : ndarray
            The weights associated with the vertices. They are defined as the
            sum of the vertex i segments' lengths divided by two.
    """
    points_indices = jnp.arange(0, closed_polyline.shape[0], 1, int)
    tangents, weights = vmap(point_tangent_and_weight, (0, None))(
        points_indices, closed_polyline)
    return tangents, weights


def point_tangent_half_distance_to_segment(
        point_i: int,
        edge_point_j: int,
        closed_polyline: jnp.ndarray) -> float:
    """Compute the tangent half distance to an edge.

    This function computes the average of the two radii of the two smallest
    circles tangent to a poyline vertex's point i and passing through only one
    point of the vertex j's edges.

    Parameters
    ----------
    point_i : int
        The index of one vertex on the polyline. The function uses its point
        and two tangents to compute the smallest circles tangent to i and
        passing through j. There are two tangents because a polyline is a
        closed polygonal chain, where each vertex has two edges representing
        two segments. The two smallest circles tangent to each segment and
        passing through j are computed, and the average of their radii is
        returned.
    edge_point_j : ndarray
        The index of one vertex on the polyline. The function uses its two
        edges to compute the smallest circle tangent to i and passing through
        only one point of its edges. The function returns the smallest radius
        of all the computed smallest circles. The two edges are sampled with N
        points. N = 3 is hardcoded in the code.
    closed_polyline : jnp.ndarray
        The input polyline must represent a closed polygonal chain. It
        represents a connected series of line segments specified by a sequence
        of points. The parameter has shape (M, N), where M is the number of
        points of the polyline and N is the number of components per point.
        Each point is connected to the previous and next points in the array.
        The last point is connected to the second last one and the first. The
        first point is connected to the last point and the second.

    Returns
    -------
    float
        The average of the two radii of the two smallest circles tangent to the
        polyline vertex's point i and passing through only one point of the
        vertex j's edges. `MAX_FLOAT` if vertex i and j share an edge with the
        same endpoint.
    """
    indices_i = jnp.array(
        [(point_i-1) % closed_polyline.shape[0],
         point_i, (point_i+1) % closed_polyline.shape[0]])
    points_i = closed_polyline[indices_i]
    tangents = vmap(tangent, (0, 0))(points_i[:2], points_i[1:])

    indices_j = jnp.array([edge_point_j, (edge_point_j+1) %
                          closed_polyline.shape[0]])
    edge_points_j = closed_polyline[indices_j]

    EDGE_SAMPLES = 3
    points_j = jnp.zeros((EDGE_SAMPLES, points_i.shape[1]))
    j_jp1 = edge_points_j[1] - edge_points_j[0]
    for i_edge_sample in range(EDGE_SAMPLES):
        t = i_edge_sample / (EDGE_SAMPLES-1.)
        sample_i = edge_points_j[0] + t * j_jp1
        points_j = points_j.at[i_edge_sample].set(sample_i)
    invalid_j = vmap(jnp.equal, (0, None))(indices_i, indices_j)
    invalid_j = jnp.any(invalid_j)
    circle_radii = vmap(
        vmap(circle_smallest_radius_tangent_to_p_passing_through_q,
             (None, 0, None)), (None, None, 0))(
        points_i[1], points_j, tangents)
    circle_radii = jnp.where(invalid_j, FLOAT_MAX, circle_radii)
    circle_radii_min = jnp.min(circle_radii, axis=1)
    # Average each tangent results
    # 0.5 is not factorized otherwise inf is returned instead of FLOAT_MAX
    return circle_radii_min[0] * 0.5 + circle_radii_min[1] * 0.5


def point_tangent_half_distance_to_all_segments(
        point_i: uint,
        closed_polyline: jnp.ndarray) -> float:
    """Compute the minimum of the tangent half distances to all the segments.

    This function computes the minimum tangent half distance from a polyline's
    point to all its segments. The computation is not accelerated, so time
    complexity is O(N), with N being the number of points on the polyline. The
    tangent half distance is defined as the radius of the smallest circle
    tangent to the given point and passing through another point, here, one on
    the polyline's segment.

    Parameters
    ----------
    i : uint
        The index of one vertex on the polyline. The function uses its point
        and two tangents to compute the smallest circles tangent to i and
        passing through j, where j is another point on the polyline's segments.
        There are two tangents because a polyline is a closed polygonal chain,
        where each vertex has two edges representing two segments. The two
        smallest circles tangent to each segment and passing through j are
        computed, and the average of their radii is returned. As there are
        several possible j, the computation is done for all possible j, i.e.,
        points on the polyline's segments, and the minimum is returned.
    closed_polyline : jnp.ndarray
        The input polyline must represent a closed polygonal chain. It
        represents a connected series of line segments specified by a sequence
        of points. The parameter has shape (M, N), where M is the number of
        points of the polyline and N is the number of components per point.
        Each point is connected to the previous and next points in the array.
        The last point is connected to the second last one and the first. The
        first point is connected to the last point and the second.

    Returns
    -------
    float
        The minimum tangent half distance from a polyline's point to all its
        segments.
    """
    neighbors_indices = jnp.arange(closed_polyline.shape[0])
    radii = vmap(point_tangent_half_distance_to_segment,
                 (None, 0, None))(point_i, neighbors_indices, closed_polyline)
    radii_min = jnp.min(radii)
    return radii_min


def points_tangent_half_distance_to_all_segments(
        closed_polyline: jnp.ndarray) -> jnp.ndarray:
    """Compute for all the points the minimum of the tangent half distances.

    This function computes for all the given polyline's points the minimum
    tangent half distance from the point to all its segments. The computation
    is not accelerated, so time complexity is O(N^2), with N being the number
    of points on the polyline. The tangent half distance is defined as the
    radius of the smallest circle tangent to the given point and passing
    through another point, here, one on the polyline's segment.

    Parameters
    ----------
    closed_polyline : ndarray
        The input polyline must represent a closed polygonal chain. It
        represents a connected series of line segments specified by a sequence
        of points. The parameter has shape (M, N), where M is the number of
        points of the polyline and N is the number of components per point.
        Each point is connected to the previous and next points in the array.
        The last point is connected to the second last one and the first. The
        first point is connected to the last point and the second.

    Returns
    -------
    ndarray
        Each item i in the returned 1-D array corresponds to the minimum
        tangent half distance from the polyline's point i to all the polyline's
        segments.
    """

    # Efficient but memory hungry
    # points_indices = jnp.arange(0, polyline.shape[0], 1, uint)
    # min_circle_radius = vmap(
    #     polyline_point_min_circle_radius_to_neighboring_segments, (0, None))(
    #     points_indices, polyline)

    # Less efficient but more nice with memory
    min_circle_radius = np.zeros((closed_polyline.shape[0],))
    for i in range(closed_polyline.shape[0]):
        min_circle_radius[i] = jit(
            point_tangent_half_distance_to_all_segments)(i, closed_polyline)
    return min_circle_radius
