from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct

from ..aabb import AABB
from ..math import roundup_power_of_2 as math_roundup_power_of_2
from ..type import float_, int_


class Grid(NamedTuple):
    """Represent a grid with square-shaped cells.

    Attributes
    ----------
    cell_ndcount : tuple[float, ...] | np.ndarray | jnp.ndarray
        The number of cells along each axis.
    origin: tuple[int, ...] | np.ndarray | jnp.ndarray
        The grid's origin is the point in the bottom left corner.
    cell_sides_length : tuple[float, ...] | np.ndarray | jnp.ndarray
        The cell sides length.
    """
    cell_ndcount: tuple[int, ...] | np.ndarray | jnp.ndarray
    origin: tuple[int, ...] | np.ndarray | jnp.ndarray
    cell_sides_length: float | np.ndarray | jnp.ndarray


def save(file, grid: Grid) -> None:
    """
    This function saves the grid to the disk in uncompressed format `.npz`.

    Parameters
    ----------
    file : str or file
        Either the filename (string) or an open file (file-like object)
        where the data will be saved. If file is a string or a Path, the
        ``.npz`` extension will be appended to the filename if it is not
        already there.
    grid : Grid
        The grid instance to save.

    Return
    ------
    None
    """
    cell_ndcount = np.array(grid.cell_ndcount)
    origin = np.array(grid.origin)
    cell_sides_length = np.array(grid.cell_sides_length)
    np.savez(
        file,
        cell_ndcount=cell_ndcount,
        origin=origin,
        cell_sides_length=cell_sides_length)
    

def savetxt(fname, grid: Grid) -> None:
    """
    This function saves the grid to the disk in text format.

    Parameters
    ----------
    fname: fnamefilename or file handle
        The filename.
    grid : Grid
        The grid instance to save.

    Return
    ------
    None
    """
    cell_ndcount = np.array(grid.cell_ndcount)
    origin = np.array(grid.origin)
    cell_sides_length = np.array(grid.cell_sides_length)

    with open(fname, 'w', encoding="utf-8") as f:
        for i in range(cell_ndcount.shape[0]):
            f.write(f"{cell_ndcount[i]}")
            if i == cell_ndcount.shape[0] - 1:
                f.write("\n")
            else:
                f.write(" ")
        for i in range(origin.shape[0]):
            f.write(f"{origin[i]:.4f}")
            if i == origin.shape[0] - 1:
                f.write("\n")
            else:
                f.write(" ")
        f.write(f"{cell_sides_length:.4f}\n")
    

def load(file) -> Grid:
    """This function loads a grid from the disk.

    Parameters
    ----------
    file : file-like object, string, or pathlib.Path
        The file to read. File-like objects must support the ``seek()`` and
        ``read()`` methods.

    Returns
    -------
    Grid
        The loaded grid instance.
    """
    data = np.load(file)
    cell_ndcount = data['cell_ndcount']
    origin = data['origin']
    cell_sides_length = data['cell_sides_length']

    grid = Grid(cell_ndcount, origin, cell_sides_length)
    return grid


def shape_dtype_from_dim(dim: int) -> Grid:
    """Create the pytree's shape and data type of a ND Grid.
    Parameters
    ----------
    dim : int
        The dimension of the grid.
    Returns
    -------
    Grid
        The pytree's shape and data type of a Grid with dimension `dim`.
    """
    return Grid(
        ShapeDtypeStruct((dim,), int_),
        ShapeDtypeStruct((dim,), float_),
        ShapeDtypeStruct((), float_))


def aabb(grid: Grid) -> AABB:
    """Give the axis-aligned bounding box of a grid.
    Parameters
    ----------
    grid : Grid
        The grid.
    Returns
    -------
    AABB
        Gives the axis-aligned bounding box of the grid.
    """
    return AABB(
        grid.origin,
        grid.origin +
        grid.cell_ndcount *
        grid.cell_sides_length)


def roundup_power_of_2(grid2: Grid) -> Grid:
    grid_sqr_side_cell_count = roundup_power_of_2(grid2.cell_ndcount[0])
    grid_sqr_side_cell_count = max(
        grid_sqr_side_cell_count, roundup_power_of_2(grid2.cell_ndcount[1]))

    return Grid(
        jnp.full((2,), grid_sqr_side_cell_count),
        grid2.origin,
        grid2.cell_sides_length)


def roundup_power_of_2(grid2: Grid) -> Grid:
    grid_sqr_side_cell_count = math_roundup_power_of_2(grid2.cell_ndcount[0])
    grid_sqr_side_cell_count = jnp.max(
        jnp.array([grid_sqr_side_cell_count, 
                   math_roundup_power_of_2(grid2.cell_ndcount[1])]))

    return Grid(
        jnp.full((2,), grid_sqr_side_cell_count),
        grid2.origin,
        grid2.cell_sides_length)
