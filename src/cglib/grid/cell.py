import jax
import jax.numpy as jnp
from jax import lax, vmap

from ..array import MaskedArray
from ..limits import INT_MAX
from ..type import int_, uint
from .grid import Grid


def corner_vertex_ndindices(
        cell_ndindex: jnp.ndarray) -> jnp.ndarray:
    """Give the ND indices of the grid cell corners points.

    This function gives the ND indices of the corner's vertices of the
    prescribed grid cell.

    Parameters
    ----------
    cell_ndindex : ndarray
        The cell ND index.

    Returns
    -------

    ndarray
        The ND indices of the corner's vertices of the prescribed grid cell.
    """
    n = cell_ndindex.shape[0]
    corner_count = 2**n
    corner_cell_ndcount = jnp.full((n,), 2)
    corner_flattened_indices = jnp.arange(corner_count)
    corner_ndindices = vmap(
        ndindex_from_1dindex, (0, None))(
        corner_flattened_indices, corner_cell_ndcount)
    return corner_ndindices + cell_ndindex


def count1_from_ndcount(cell_ndcount: jnp.ndarray) -> int:
    return jnp.prod(cell_ndcount)


def index1_from_ndindex(
        cell_ndindex: jnp.ndarray,
        cell_ndcount: jnp.ndarray) -> int:
    """Map the cell ND index of a grid to its corresponding flattened index.

    Parameters
    ----------
    cell_ndindex : ndarray
        The cell ND index to convert.
    cell_ndcount: ndarray
        The grid's cell ND count, i.e., the number of cells along each axis.

    Returns
    -------
    int
        The function returns the cell 1D index, i.e., the flattened index,
        which corresponds to the ND index of a grid's cell.
    """
    n = cell_ndcount.shape[0]
    flattened_index = cell_ndindex[0]
    for i in range(1, n):
        shift = uint(jnp.prod(jnp.array(cell_ndcount[:i])))
        flattened_index += cell_ndindex[i] * shift
    return flattened_index


def index1_from_masked_ndindex(
        cell_ndindex: jnp.ndarray,
        cell_ndcount: jnp.ndarray) -> int:
    """Map a grid masked cell ND index to a 1D index.

    This function maps the cell ND index of a grid to its corresponding
    flattened index. If the cell ND index is masked, i.e., its first component
    is the maximum value for a variable of type signed integer (INT_MAX), then
    the returned cell 1D index is INT_MAX.

    Parameters
    ----------
    cell_ndindex : ndarray
        The cell ND index to convert.
    cell_ndcount: ndarray
        The grid's cell ND count, i.e., the number of cells along each axis.

    Returns
    -------
    int
        The function returns the cell 1D index, i.e., the flattened index,
        which corresponds to the ND index of a grid's cell. If the cell ND
        index is masked, i.e., its first component is the maximum value for a
        variable of type signed integer (INT_MAX), then the returned cell 1D
        index is INT_MAX.
    """
    cell_ndindex_mask = jnp.equal(cell_ndindex[0], INT_MAX)
    n = cell_ndcount.shape[0]
    flattened_index = cell_ndindex[0]
    for i in range(1, n):
        shift = int_(jnp.prod(jnp.array(cell_ndcount[:i])))
        flattened_index += cell_ndindex[i] * shift
    flattened_index = jnp.where(cell_ndindex_mask, INT_MAX, flattened_index)
    return flattened_index


def ndindex_from_1dindex(
        cell_1dindex: int | uint,
        cell_ndcount: jnp.ndarray) -> jnp.ndarray:
    """Maps the cell 1D index of a grid to its corresponding ND index.

    Parameters
    ----------

    cell_1dindex : int | uint
        The cell 1D index to convert.
    cell_ndcount: ndarray
        The grid's cell ND count, i.e., the number of cells along each axis.

    Returns
    -------
    ndarray
        The function returns the cell ND index corresponding to the 1D index of
        a grid's cell.
    """
    n = cell_ndcount.shape[0]
    cell_ndindex = jnp.zeros((n,), uint)
    cell_ndindex = cell_ndindex.at[0].set(
        uint(
            cell_1dindex %
            cell_ndcount[0]))
    for i in range(1, n):
        shift = uint(jnp.prod(cell_ndcount[:i]))
        cell_ndindex = cell_ndindex.at[i].set(
            uint(
                (cell_1dindex // shift) %
                cell_ndcount[i]))
    return cell_ndindex


def ndindex_from_point(
        p: jnp.ndarray,
        origin: jnp.ndarray,
        cell_sides_length: float) -> jnp.ndarray:
    """Return the cell ND index of a point in a grid.
    Return the grid's cell ndindex containing the point passed as a parameter.
    The grid is specified with an origin and a cell sides length.
    Parameters
    ----------
    p : ndarray
        The point.
    origin : ndarray
        The grid's origin.
    cell_sides_length: float
        The cell sides length.
    Returns
    -------
    ndarray
        The grid's cell ndindex containing the point passed as a parameter.
    """
    cell_ndindex_float = (p - origin) / cell_sides_length
    cell_ndindex = jnp.floor(cell_ndindex_float).astype(int_)
    return cell_ndindex


def ndindex_from_masked_point(
        p: jnp.ndarray,
        origin: jnp.ndarray,
        cell_sides_length: float) -> jnp.ndarray:
    """Return the cell ND index of the given point.

    Return the grid's cell ndindex containing the point passed as a parameter.
    The grid is specified with an origin and a cell sides length. If the point
    is masked, i.e., its first component is not a number (nan), then the
    returned cell ndindex has all its components equaled to the maximum value
    for a variable of type signed integer.

    Parameters
    ----------
    p : ndarray
        The point.
    origin : ndarray
        The grid's origin.
    cell_sides_length: float
        The cell sides length.

    Returns
    -------
    ndarray
        The grid's cell ndindex containing the point passed as a parameter.
        `jnp.full_like(cell_ndindex, INT_MAX)` if `jnp.isnan(p[0])` is `True`.
    """
    p_mask = jnp.isnan(p[0])
    cell_ndindex_float = (p - origin) / cell_sides_length
    cell_ndindex = jnp.floor(cell_ndindex_float).astype(int_)
    cell_ndindex = jnp.where(p_mask, jnp.full_like(
        cell_ndindex, INT_MAX), cell_ndindex)
    return cell_ndindex


def ndindex_is_valid(
        grid_cell_ndindex: jnp.ndarray,
        grid_cell_ndcount: jnp.ndarray) -> bool:
    """Is valid cell ND index.

    This function verifies if the prescribed grid cell ND index is valid with
    respect to the prescribed grid cell ND count.

    Parameters
    ----------
    grid_cell_ndindex : ndarray
        The grid cell ND index.
    grid_cell_ndcount : ndarray
        The grid cell ND count.

    Return
    ------
    bool
        The function returns `jnp.all(jnp.logical_and(grid_cell_ndindex >= 0,
        grid_cell_ndindex < grid_cell_ndcount))`
    """
    valid_cell_ndindex = jnp.logical_and(
        grid_cell_ndindex >= 0, grid_cell_ndindex < grid_cell_ndcount)
    return jnp.all(valid_cell_ndindex)


def moore_neighborhood(
        cell_ndindex: jnp.ndarray,
        d: int = 1) -> jnp.ndarray:
    """Return the Moore neighborhood of a cell.

    This function gives the Moore neighborhood of a cell in a ND grid. The
    Moore neighborhood is usually defined on a two-dimensional square lattice
    and is composed of a central cell and the eight cells surrounding it, i.e.,
    the cells at a Chebyshev distance of one:
    https://en.wikipedia.org/wiki/Moore_neighborhood. Here, an extended Moore
    neighborhood can be returned with neighboring cells with a Chebyshev
    distance greater than one.

    Parameters
    ----------
    cell_ndindex : ndarray
        The cell's ND index indicates the cell from which the Chebyshev
        distance is computed, i.e., the neighborhood center.
    d : int, default: 1
        The neighborhood is defined as cells with Chebyshev distance less equal
        to this parameter.

    Returns
    -------
    ndarray
        The Moore neighborhood is represented by a 2D array with the shape `(m,
        n)`, where m is the number of neighbors and `n` is the dimension count
        of the grid.
    """
    with jax.ensure_compile_time_eval():
        sides_length = 1 + 2 * d
        dim = jnp.full(cell_ndindex.shape, sides_length)
        indices = jnp.flip(jnp.indices(dim), axis=0)
    n = cell_ndindex.shape[0]
    indices = jnp.transpose(indices.reshape(n, -1)) - d
    return cell_ndindex + indices


def moore_neighborhood_from_point(
        x: jnp.ndarray,
        grid: Grid,
        d: int = 1) -> MaskedArray:
    """Return the Moore neighborhood from a point.

    This function gives the Moore neighborhood of a cell in a 2D grid. The
    center cell is computed from a point. The Moore neighborhood is usually
    defined on a two-dimensional square lattice and is composed of a central
    cell and the eight cells surrounding it, i.e., the cells at a Chebyshev
    distance of one: https://en.wikipedia.org/wiki/Moore_neighborhood. Here, an
    extended Moore neighborhood can be returned with neighboring cells with a
    Chebyshev distance greater than one.

    Parameters
    ----------
    x : jnp.ndarray
        The point `x` is used to compute the cell's ND index indicating the
        cell from which the Chebyshev distance is calculated, i.e., the
        neighborhood center. Its shape is (n,), where n is the number of point
        `x` components.
    grid : Grid
        The grid used to compute the Moore neighborhood.
    d : int, default: 1
        The neighborhood is defined as cells with Chebyshev distance less equal
        to this parameter.

    Returns
    -------
    MaskedArray
        The Moore neighborhood is represented by a 2D array with the shape `(m,
        n)`, where m is the number of neighbors and n is the number of
        components of a cell's ND index.
    """
    x_cell = ndindex_from_point(x,
                                grid.origin,
                                grid.cell_sides_length)
    x_moore_neighborhood = moore_neighborhood(x_cell, d)
    # Mask invalid indices
    valid_cells = jnp.full((x_moore_neighborhood.shape[0],), True)
    for i in range(grid.origin.shape[0]):
        valid_cells = valid_cells & (x_moore_neighborhood[:, i] >= 0)
        valid_cells = valid_cells & (
            x_moore_neighborhood[:, i] < grid.cell_ndcount[i])
    return MaskedArray(x_moore_neighborhood, jnp.logical_not(valid_cells))


def center_point(
        grid: Grid,
        cell_ndindex: jnp.ndarray) -> jnp.ndarray:
    """Give the center point of the cell of a grid.

    Parameters
    ----------
    grid : Grid
        The grid.
    cell_ndindex: ndarray
        The grid's cell ND index.

    Returns
    -------
    ndarray
        The function returns the cell's center point. The grid cell is
        indicated by its ND index.
    """
    cell_sides_length = grid.cell_sides_length
    ret = grid.origin + cell_sides_length * (cell_ndindex + 0.5)
    return ret


def center_points(
        grid: Grid) -> jnp.ndarray:
    """Give the cell center points of a grid.

    Parameters
    ----------
    grid : Grid
        The grid

    Returns
    -------
    ndarray
        The function returns the cells center points, each on a row of the
        returned matrix.
    """
    with jax.ensure_compile_time_eval():
        grid_cell_1dcount = jnp.prod(grid.cell_ndcount)
    grid_cell_1dindices = jnp.arange(grid_cell_1dcount)
    grid_cell_2dindices = jax.vmap(
        ndindex_from_1dindex,
        in_axes=(0, None))(grid_cell_1dindices, grid.cell_ndcount)
    grid_cell_centers = jax.vmap(
        center_point,
        in_axes=(None, 0))(grid, grid_cell_2dindices)
    return grid_cell_centers


def center_points_jittered(
        grid: Grid, seed_jax: int, jitter_distance: float) -> jnp.ndarray:
    """Give the center points of all the cells of a grid, with random jitter.

    Parameters
    ----------
    grid : Grid
        The grid.
    seed_jax : KeyArray
        The seed for random, obtained from a JAX function
        (jax.random.[PRNGKey][split])
    jitter_distance : float
        The distance along each axis dimension the point can move, expressed as
        a fraction of the grid cell sides length.

    Returns
    -------
    ndarray
        The function returns the cells' center jittered points, each on a row
        of the returned matrix.
    """
    with jax.ensure_compile_time_eval():
        cell_count = jnp.prod(grid.cell_ndcount)
    uniform_points = jax.random.uniform(seed_jax, (cell_count, 2))
    grid_cell_center_points = center_points(grid)
    return grid_cell_center_points + \
        (uniform_points - 0.5) * grid.cell_sides_length * jitter_distance


def boundary2_1dcount(
        grid_cell_2dcount: tuple[int, int] | jnp.ndarray) -> int:
    return 2 * (grid_cell_2dcount[0] + grid_cell_2dcount[1]) - 4


def boundary_2dindex_from_1dindex(
        grid_border_cell_1dindex: jnp.ndarray,
        grid_cell_2dcount: jnp.ndarray) -> int:
    res = jnp.array([grid_border_cell_1dindex, 0])
    cond1 = jnp.greater_equal(grid_border_cell_1dindex, grid_cell_2dcount[0])
    cond2 = jnp.greater_equal(
        grid_border_cell_1dindex,
        grid_cell_2dcount[0] + grid_cell_2dcount[1] - 1)
    cond3 = jnp.greater_equal(
        grid_border_cell_1dindex,
        2 * grid_cell_2dcount[0] + grid_cell_2dcount[1] - 2)
    res = lax.cond(cond1,
                   lambda x: jnp.array([x[1][0] - 1, x[0] - x[1][0] + 1]),
                   lambda x: x[2],
                   (grid_border_cell_1dindex,
                       grid_cell_2dcount,
                       res))
    res = lax.cond(cond2,
                   lambda x: jnp.array([2 * x[1][0] + x[1][1] - x[0] - 3,
                                        x[1][1] - 1]),
                   lambda x: x[2],
                   (grid_border_cell_1dindex,
                    grid_cell_2dcount,
                    res))
    res = lax.cond(
        cond3,
        lambda x: jnp.array([0, boundary2_1dcount(x[1]) - x[0]]),
        lambda x: x[2], (grid_border_cell_1dindex, grid_cell_2dcount, res))
    return res


def boundary_1dindex_from_2dindex(
        grid_cell_2dindex: jnp.ndarray, grid_cell_2dcount: jnp.ndarray) -> int:

    res = grid_cell_2dindex[0]

    left = jnp.equal(grid_cell_2dindex[0], 0)
    right = jnp.equal(grid_cell_2dindex[0], grid_cell_2dcount[0] - 1)
    bottom = jnp.equal(grid_cell_2dindex[1], 0)
    top = jnp.equal(grid_cell_2dindex[1], grid_cell_2dcount[1] - 1)
    case_1 = jnp.logical_and(right, jnp.logical_not(bottom))
    case_2 = jnp.logical_and(top, jnp.logical_not(right))
    case_3 = jnp.logical_and(
        jnp.logical_and(
            left,
            jnp.logical_not(top)),
        jnp.logical_not(bottom))
    res = lax.cond(
        case_1,
        lambda x: x[1][0] + x[0][1] - 1,
        lambda x: x[2],
        (grid_cell_2dindex, grid_cell_2dcount, res))
    res = lax.cond(
        case_2,
        lambda x: 2 * x[1][0] + x[1][1] - x[0][0] - 3,
        lambda x: x[2],
        (grid_cell_2dindex, grid_cell_2dcount, res))
    res = lax.cond(
        case_3,
        lambda x: boundary2_1dcount(x[1]) - x[0][1],
        lambda x: x[2],
        (grid_cell_2dindex, grid_cell_2dcount, res))
    return res
