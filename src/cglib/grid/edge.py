from enum import IntEnum

import jax
import jax.numpy as jnp
from jax import lax, vmap

from ..array import MaskedArray
from ..limits import INT_MAX
from ..type import int_, uint
from . import cell
from .grid import Grid


def count1_from_cell_2dcount(grid_cell_2dcount: jnp.ndarray) -> int:
    """Return the grid edge 1D count.

    This function returns the 2D grid edge 1D count, i.e., the number of edges
    in the grid, from the grid cell 2D count, i.e., the number of cells along
    each axis of the grid.

    Parameters
    ----------
    grid_cell_2dcount : ndarray
        The number of cells along each axis of the grid.

    Returns
    -------
    int
        The total number of edges in the grid.
    """
    return grid_cell_2dcount[0] * (grid_cell_2dcount[1] + 1) + \
        (grid_cell_2dcount[0] + 1) * grid_cell_2dcount[1]


def count2_per_axis(
        grid_cell_2dcount: jnp.ndarray) -> jnp.ndarray:
    """Return the grid edge 2D count per edge axis.

    Parameters
    ----------
    grid_cell_2dcount : ndarray
        The number of cell along each axis, per edge axis.

    Returns
    -------
    ndarray
        Shape: (2, 2).
    """
    edge_2dcount = jnp.array([[grid_cell_2dcount[0],
                               grid_cell_2dcount[1] + 1],
                              [grid_cell_2dcount[0] + 1,
                               grid_cell_2dcount[1]]])
    return edge_2dcount


def index1_from_2dindex(
        edge_2dindex: jnp.ndarray,
        edge_axis: int,
        edge_2dcount: jnp.ndarray) -> uint:
    """Convert an edge specified by its 2D index and axis to its 1D index.

    Parameters
    ----------
    edge_ndindex : jnp.ndarray
        The ND index of the edge. See the test for more details.
    edge_axis : int
        The axis specifies which domain axis is parallel to the edge.
    edge_2dcount : jnp.ndarray
        The grid edge 2D count per axis. See the test for details.

    Returns
    -------
    uint
        The 1D index of the edge's 2D index.
    """
    edge_flattened_index = cell.index1_from_ndindex(
        edge_2dindex, edge_2dcount[edge_axis])
    hedge_flattened_cell_count = edge_2dcount[0][0] * edge_2dcount[0][1]
    edge_flattened_index += edge_axis * hedge_flattened_cell_count
    return uint(edge_flattened_index)


def indices1_from_2dgrid(
        grid_cell_2dcount: jnp.ndarray) -> jnp.ndarray:
    """Give the 1D indices of the edges of the prescribed grid.

    Parameters
    ----------
    grid_cell_2dcount : ndarray
        The number of cells along each axis.

    Returns
    -------
    ndarray
        The function gives the 1D indices of the edges of the prescribed grid.
    """

    with jax.ensure_compile_time_eval():
        edge_ndcount_per_axis = count2_per_axis(grid_cell_2dcount)
        edge_1dcount = \
            edge_ndcount_per_axis[0][0] * edge_ndcount_per_axis[0][1] + \
            edge_ndcount_per_axis[1][0] * edge_ndcount_per_axis[1][1]
        edge_1dindices = jnp.arange(edge_1dcount)
    return edge_1dindices


def index2_from_1dindex(
        grid2_edge_1dindex: uint,
        grid2_edge_2dcount: jnp.ndarray) -> tuple[jnp.ndarray, uint]:
    """Convert an edge specified by its 1D index to its 2D index and axis.

    Parameters
    ----------
    grid2_edge_1dindex : uint
        The 1D index of the edge's 2D index.
    grid2_edge_2dcount : ndarray
        The grid edge 2D count per axis. See the test for details.

    Returns
    -------
    tuple[ndarray, uint]
        out1 : ndarray
            The ND index of the edge. See the test for more details.
        out2 : uint
            The axis specifies which domain axis is parallel to the edge.
    """
    grid2_horizontal_edge_1dcount = uint(grid2_edge_2dcount[0, 0] *
                                         grid2_edge_2dcount[0, 1])
    grid2_edge_axis = uint(lax.cond(
        grid2_edge_1dindex >= grid2_horizontal_edge_1dcount,
        lambda: 1, lambda: 0))
    grid2_edge_1dindex = lax.cond(
        jnp.equal(grid2_edge_axis, 1),
        lambda x: x -
        grid2_horizontal_edge_1dcount,
        lambda x: x,
        grid2_edge_1dindex)
    grid2_edge_2dindex = cell.ndindex_from_1dindex(
        grid2_edge_1dindex, grid2_edge_2dcount[grid2_edge_axis])
    return grid2_edge_2dindex, grid2_edge_axis


def indices2_from_grid(
        grid_cell_2dcount: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Give the 2D indices of the grid edges.

    Parameters
    ----------
    grid_cell_2dcount : ndarray
        The number of cells along each axis.

    Returns
    -------
    tuple[ndarray, ndarray]
        out1
            The 2D indices of the horizontal edges. Shape (m, 2), where m is
            the number of horizontal edges.
        out2
            The 2D indices of the vertical edges. Shape (n, 2), where n is the
            number of vertical edges.
    """

    with jax.ensure_compile_time_eval():
        edge_ndcount_per_axis = count2_per_axis(grid_cell_2dcount)
        edge_flattened_count_per_axis = jnp.array(
            [edge_ndcount_per_axis[0][0] * edge_ndcount_per_axis[0][1],
             edge_ndcount_per_axis[1][0] * edge_ndcount_per_axis[1][1]])
        edge_flattened_indices_per_axis = (
            jnp.arange(edge_flattened_count_per_axis[0]),
            jnp.arange(edge_flattened_count_per_axis[1])
        )
    edge_flattened_to_ndindex_v = vmap(
        cell.ndindex_from_1dindex, (0, None))
    edge_ndindices_per_axis = (
        edge_flattened_to_ndindex_v(
            edge_flattened_indices_per_axis[0],
            edge_ndcount_per_axis[0]),
        edge_flattened_to_ndindex_v(
            edge_flattened_indices_per_axis[1],
            edge_ndcount_per_axis[1])
    )
    # shape: (edge axis, edge 2dindex)
    return edge_ndindices_per_axis


def neighboring2_1dindices(
        edge_1dindex: int,
        grid_cell_ndcount: jnp.ndarray,
        neighborhood_radius: int = 1) -> MaskedArray:
    """Return the neighboring edges' 1D indices of a given edge.

    Parameters
    ----------
    edge_1dindex : int
        The 1-D index of the edge whose neighborhood is computed.
    grid_params : Grid
        The cell ND count of the grid where the edges are defined.
    neighborhood : int (default: 1)
        The "width" of the neighborhood.

    Returns
    -------
    MaskedArray
        The neighboring edges' 1D indices of the given edge.
    """
    grid_edge_2dcount = count2_per_axis(grid_cell_ndcount)
    edge_2dindex, edge_axis = index2_from_1dindex(
        edge_1dindex, grid_edge_2dcount)
    neighbors_masked_2dindex = neighboring_2dindices(
        edge_2dindex, edge_axis, grid_cell_ndcount, neighborhood_radius)
    neighbors_mask = jnp.ravel(neighbors_masked_2dindex.mask)
    neighbors_1dindex = vmap(
        vmap(index1_from_2dindex, (0, None, None)),
        (0, 0, None))(
            neighbors_masked_2dindex.array,
            jnp.array([0, 1]),
            grid_edge_2dcount)
    neighbors_1dindex = jnp.ravel(neighbors_1dindex)
    return MaskedArray(neighbors_1dindex, neighbors_mask)


def neighboring_2dindices(
        edge_2dindex: jnp.ndarray,
        edge_axis: int,
        grid_cell_2dcount: jnp.ndarray,
        neighbor_radius: uint) -> MaskedArray:
    """Return the neighboring edge 2D indices.

    This function returns the 2D indices of the edges neighbors of the
    prescribed edge within a neighbor radius. An edge is defined as a cell ND
    index and an axis specifying which domain axis is parallel to the edge. See
    the test for more details.

    Parameters
    ----------
    edge_ndindex : ndarray
        The ND index of the edge. See the test for more details.
    edge_axis : int
        The axis specifies which domain axis is parallel to the edge.
    grid_cell_2dcount : ndarray
        The number of grid's cells for each axis.
    neighbor_radius : uint
        The neighbor radius gives the maximum difference of edge 2D indices'
        components to belong to the neighborhood.

    Returns
    -------
    MaskedArray
        The function returns a masked array. The ND array has three dimensions,
        with a shape (2, (neighbor_radius * 2 + 1)**2, 2). The first index
        indexes the edge axes, and the adjacent edges' ND indices are indexed
        by the second index. The third index indexes the indices of one ND
        index. The mask indicates invalid ND indices, i.e., those with ND
        indices not in a valid interval or empty ND indices for padding. The
        prescribed edge is masked. The invalid indices' components are set to
        zero.
    """

    width = neighbor_radius * 2 + 1
    neighbor_1dindices_count = width**2
    neighbor_1dindices = jnp.arange(0, neighbor_1dindices_count)
    edge_2dindex_shifts = vmap(
        cell.ndindex_from_1dindex, (0, None))(
        neighbor_1dindices, jnp.array([width, width])) - neighbor_radius
    edge_2dindex_shifts = jnp.array([edge_2dindex_shifts, edge_2dindex_shifts])

    # shape: (edge orientation, edge count per axis)
    edge_2dcount = jnp.array(count2_per_axis(grid_cell_2dcount))

    neighbors_2dindices = edge_2dindex.reshape((1, 1, 2)) + edge_2dindex_shifts
    neighbors_mask = jnp.logical_or(
        neighbors_2dindices >= edge_2dcount.reshape(
            (edge_2dcount.shape[0], 1, edge_2dcount.shape[1])),
        neighbors_2dindices < 0)
    # Put valid indices where indices are masked
    neighbors_2dindices = jnp.where(neighbors_mask, 0, neighbors_2dindices)
    # Flatten the last axis (index components mask)
    neighbors_mask = jnp.any(neighbors_mask, axis=2)
    neighbors_mask = neighbors_mask.at[edge_axis,
                                       neighbor_1dindices_count // 2].set(True)
    return MaskedArray(neighbors_2dindices, neighbors_mask)


class Neighboring2Type(IntEnum):
    """
    This class enumerates the two neighboring types of a 2D grid's edge.

    Attributes
    ----------
    VISIBLE
        A visible neighboring is a neighboring that is visible from an edge.
    WITHIN_CELL_SIDE_LENDTH
        Edges within distance from the specified edge that is less than the
        edge length.
    """
    VISIBLE = 0
    WITHIN_CELL_SIDE_LENDTH = 1


def neighboring_2dindices_direct(
        edge_2dindex: jnp.ndarray,
        edge_axis: int,
        grid_cell_2dcount: jnp.ndarray,
        neighboring_type: Neighboring2Type) -> MaskedArray:
    """Return the edge direct neighbors.

    This function returns the 2D indices of the edges considered direct
    neighbors of the prescribed edge. An edge is defined as a cell ND index and
    an axis specifying which domain axis is parallel to the edge. See the test
    for more details.

    Parameters
    ----------
    edge_ndindex : jnp.ndarray
        The ND index of the edge. See the test for more details.
    edge_axis : int
        The axis specifies which domain axis is parallel to the edge.
    grid_cell_2dcount : jnp.ndarray
        The number of grid's cells for each axis.
    neighboring_type : EdgeNeighboringType
        Specify the type of neighboring.

    Returns
    -------
    MaskedArray
        The function returns a masked array. The ND array has three dimensions,
        with a shape (2, 4, 2). The first index indexes the edge axes, and the
        adjacent edges ND indices are indexed by the second index. The third
        index indexes the indices of one ND index. The mask indicates invalid
        ND indices, i.e., those with ND indices not in a valid interval or
        empty ND indices for  padding. The invalid indices' components are set
        to zero.
    """

    def _get_edge_2dindex_shifts_visible_case():
        return jnp.array([[[0, -1],
                           [0, 1],
                           [INT_MAX, INT_MAX],
                           [INT_MAX, INT_MAX]],
                          [[0, -1],
                           [1, -1],
                           [0, 0],
                           [1, 0]]])

    def _get_edge_2dindex_shifts_p_repulsion_case():
        return jnp.array([[[-1, 0],
                           [1, 0],
                           [INT_MAX, INT_MAX],
                           [INT_MAX, INT_MAX]],
                          [[0, -1],
                           [1, -1],
                           [0, 0],
                           [1, 0]]])

    # shape: (edge orientation, edge count per axis)
    edge_2dcount = jnp.array(count2_per_axis(grid_cell_2dcount))

    # shape: (edge orientation, edge, index components)
    edge_ndindex_shifts = lax.switch(
        neighboring_type,
        [_get_edge_2dindex_shifts_visible_case,
         _get_edge_2dindex_shifts_p_repulsion_case])

    # Flip list if the edge is vertical
    edge_ndindex_shifts = lax.cond(
        edge_axis == 1,
        lambda x: jnp.flip(jnp.flip(x, axis=0), axis=2),
        lambda x: x,
        edge_ndindex_shifts)

    # Reshape for broadcast then shift the indices
    neighbors_ndindices = edge_2dindex.reshape((1, 1, 2)) + edge_ndindex_shifts
    # Mask invalid indices
    neighbors_mask = jnp.logical_or(
        neighbors_ndindices >= edge_2dcount.reshape(
            (edge_2dcount.shape[0], 1, edge_2dcount.shape[1])),
        neighbors_ndindices < 0)
    # Put valid indices where indices are masked
    neighbors_ndindices = jnp.where(neighbors_mask, 0, neighbors_ndindices)
    # Flatten the last axis (index components mask)
    neighbors_mask = jnp.any(neighbors_mask, axis=2)
    return MaskedArray(neighbors_ndindices, neighbors_mask)


def endpoints(
        edge_ndindex: jnp.ndarray,
        edge_axis: int,
        grid: Grid) -> jnp.ndarray:
    """Return the edge endpoints.

    This function returns the endpoints of a given edge in a grid. An edge is
    defined as a cell ND index and an axis specifying which edge of the ND cell
    is used. A cell has 2^(N-1) edges parallel to a domain axis, and the edge
    considered is the one with the endpoint with the smallest components.

    Parameters
    ----------
    edge_ndindex : ndarray
        The cell ND index of the edge. A valid ND index has
        `all(zeros(grid.cell_ndcount.shape) <= edge_ndindex <=
        grid.cell_ndcount) == True`.
    edge_axis: int
        The axis specifies which domain axis is parallel to the edge.
    grid : Grid
        The given grid.

    Returns
    ------
    ndarray
        The function returns a 2 x N matrix. The first row indicates the first
        endpoint, with minimum component values. The second row is the second
        endpoint.
    """
    # Dimension
    n = grid.cell_ndcount.shape[0]
    # Increment of cell sides length along edge_axis
    increment = jnp.zeros((n,), int_).at[edge_axis].set(
        1) * grid.cell_sides_length
    v0 = edge_ndindex * grid.cell_sides_length + grid.origin
    v1 = v0 + increment
    return jnp.array([v0, v1])


def boundary_1dcount(
        grid_cell_2dcount: tuple[int, int] | jnp.ndarray) -> int:
    return 2 * (grid_cell_2dcount[0] + grid_cell_2dcount[1])


def boundary_1d_to_2dindex(
        grid_border_edge_flattened_index: jnp.ndarray,
        grid_cell_2dcount: jnp.ndarray) -> int:

    res = (jnp.array([grid_border_edge_flattened_index, 0]), 0)
    cond1 = jnp.greater_equal(
        grid_border_edge_flattened_index,
        grid_cell_2dcount[0])
    res = lax.cond(cond1,
                   lambda x: (jnp.array([x[1][0],
                                         x[0] - x[1][0]]),
                              1),
                   lambda x: x[2],
                   (grid_border_edge_flattened_index,
                       grid_cell_2dcount,
                       res))
    cond2 = jnp.greater_equal(
        grid_border_edge_flattened_index,
        grid_cell_2dcount[0] + grid_cell_2dcount[1])
    res = lax.cond(cond2,
                   lambda x: (jnp.array([2 *
                                         x[1][0] +
                                         x[1][1] -
                                         x[0] -
                                         1, x[1][1]]), 0),
                   lambda x: x[2],
                   (grid_border_edge_flattened_index,
                       grid_cell_2dcount,
                       res))
    cond3 = jnp.greater_equal(
        grid_border_edge_flattened_index,
        2 * grid_cell_2dcount[0] + grid_cell_2dcount[1])
    res = lax.cond(cond3,
                   lambda x: (jnp.array([0, 2 *
                                         (x[1][0] +
                                          x[1][1]) -
                                         x[0] -
                                         1]), 1),
                   lambda x: x[2],
                   (grid_border_edge_flattened_index,
                       grid_cell_2dcount,
                       res))
    return res


def boundary_1dindex_from_2dindex(
        grid_edge_2dindex: jnp.ndarray,
        grid_edge_axis: int,
        grid_cell_2dcount: jnp.ndarray) -> int:

    def _grid_boundary_edge_2d_to_flattened_index_case00(
            grid_edge_2dindex: jnp.ndarray,
            grid_cell_2dcount: jnp.ndarray) -> int:
        return grid_edge_2dindex[0]

    def _grid_boundary_edge_2d_to_flattened_index_case11(
            grid_edge_2dindex: jnp.ndarray,
            grid_cell_2dcount: jnp.ndarray) -> int:
        return grid_edge_2dindex[1] + grid_cell_2dcount[0]

    def _grid_boundary_edge_2d_to_flattened_index_case10(
            grid_edge_2dindex: jnp.ndarray,
            grid_cell_2dcount: jnp.ndarray) -> int:
        return grid_cell_2dcount[0] - 1 - grid_edge_2dindex[0] + \
            grid_cell_2dcount[0] + grid_cell_2dcount[1]

    def _grid_boundary_edge_2d_to_flattened_index_case01(
            grid_edge_2dindex: jnp.ndarray,
            grid_cell_2dcount: jnp.ndarray) -> int:
        return grid_cell_2dcount[1] - 1 - grid_edge_2dindex[1] + \
            2 * grid_cell_2dcount[0] + grid_cell_2dcount[1]

    branches = [
        _grid_boundary_edge_2d_to_flattened_index_case00,
        _grid_boundary_edge_2d_to_flattened_index_case01,
        _grid_boundary_edge_2d_to_flattened_index_case10,
        _grid_boundary_edge_2d_to_flattened_index_case11
    ]

    grid_edge_2dcount = count2_per_axis(grid_cell_2dcount)
    other_axis = (grid_edge_axis + 1) % 2

    cond = jnp.equal(grid_edge_2dindex[other_axis], jnp.array(
        grid_edge_2dcount)[grid_edge_axis][other_axis] - 1)

    border_side = lax.cond(cond, lambda: 1, lambda: 0)

    case_index = border_side * 2 + grid_edge_axis
    return lax.switch(
        case_index,
        branches,
        grid_edge_2dindex,
        grid_cell_2dcount)
