import jax.numpy as jnp


def ndindex_restrict(
        cell_ndindex: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Prolong the cell ND index from a fine to a coarse grid.

    Parameters
    ----------
    cell_ndindex : ndarray
        The fine grid's cell ND index.

    Returns
    -------
    tuple[ndarray, ndarray]
        out1 : jnp.ndarray
            The return value is the cell ND index in the coarser grid space,
            defined as the element-wise integer division between `cell_ndindex`
            and 2.
        out2 : ndarray
            The second output is the shift in a fine 2^N hypercube representing
            one coarse cell, i.e., `cell_ndindex = (out1 * 2) + out2`. It is
            the element-wise remainder from floor division between
            `cell_ndindex` and 2.
    """
    return jnp.divmod(cell_ndindex, 2)


def ndindex_prolong(
        cell_ndindex: jnp.ndarray) -> jnp.ndarray:
    """Prolong the cell ND index from a coarse to a fine grid.

    Parameters
    ----------
    cell_ndindex : ndarray
        The coarse grid's cell ND index.

    Returns
    -------
    ndarray
        This function returns the fine grid cells' ND indices corresponding to
        the cell ND index of a coarser grid.
    """
    n = cell_ndindex.shape[0]
    sides_length = 2
    dim = tuple([sides_length] * n)
    indices = jnp.flip(jnp.indices(dim), axis=0)
    indices = jnp.transpose(indices.reshape(n, -1))
    return cell_ndindex * 2 + indices
