from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import type


class AABB(NamedTuple):
    """A class used to represent an axis-aligned bounding box.

    Attributes
    ----------
    p_min : tuple[float, ...] | ndarray
        Point of the box with minimum components.
    p_max : tuple[float, ...] | ndarray
        Point of the box with maximum components.
    """
    p_min: tuple[float, ...] | np.ndarray | jnp.ndarray
    p_max: tuple[float, ...] | np.ndarray | jnp.ndarray


def aabb_union_point(p: jnp.ndarray, aabb: AABB) -> AABB:
    """ Return the the union of a point and an AABB.

    Parameters
    ----------
    p : ndarray
        The point.
    aabb : AABB
        The axis-aligned bounding box.

    Returns
    -------
    AABB
        The axis-aligned bounding box (AABB) resulting from the union of a
        point and an AABB.
    """
    return AABB(jnp.minimum(aabb.p_min, p), jnp.maximum(aabb.p_max, p))


def aabb_corners(aabb: AABB) -> jnp.ndarray:
    """Return the corners of the axis-aligned bounding box.

    Parameters
    ----------
    aabb : AABB
        The axis-aligned bounding box.

    Returns
    -------
    ndarray
        The array of corners.
    """

    def _cell_1d_to_ndindex(
            cell_1dindex: int | type.uint,
            cell_ndcount: jnp.ndarray) -> jnp.ndarray:
        n = cell_ndcount.shape[0]
        cell_ndindex = jnp.zeros((n,), type.uint)
        cell_ndindex = cell_ndindex.at[0].set(
            type.uint(
                cell_1dindex %
                cell_ndcount[0]))
        for i in range(1, n):
            shift = type.uint(jnp.prod(cell_ndcount[:i]))
            cell_ndindex = cell_ndindex.at[i].set(
                type.uint(
                    (cell_1dindex // shift) %
                    cell_ndcount[i]))
        return cell_ndindex

    n = aabb.p_min.shape[0]
    p_min_max = jnp.concatenate(
        (aabb.p_min.reshape(-1, n), aabb.p_max.reshape(-1, n)))

    cell_count_flattened = jnp.arange(2 ** n)
    nd_cell_count = jax.vmap(
        _cell_1d_to_ndindex, (0, None))(
        cell_count_flattened, jnp.full(n, 2))

    components = []
    for i in range(n):
        components_i = p_min_max[nd_cell_count[:, i], i].reshape(-1, 1)
        components.append(components_i)
    corners = jnp.concatenate(components, axis=1)
    return corners
