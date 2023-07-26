from typing import Any

import jax.numpy as jnp
from jax import ShapeDtypeStruct, tree_util

"""
Extend https://jax.readthedocs.io/en/latest/jax.tree_util.html
"""


def concatenate(pytrees: tuple) -> tuple:
    """Concatenate the pytrees leaves.

    Take a tuple of
    [pytrees](https://jax.readthedocs.io/en/latest/pytrees.html) with identical
    definitions and concatenates every corresponding leaf.
    """
    tree_count = len(pytrees)
    treedef = tree_util.tree_structure(pytrees[0])
    leaf_list = []
    for i_tree in range(tree_count):
        leaves_i_tree = tree_util.tree_leaves(pytrees[i_tree])
        leaf_list.append(leaves_i_tree)
    leaf_count = len(leaf_list[0])

    leaves_concatenated = []
    for i_leaf in range(leaf_count):
        leaf_i_all_trees = []
        for i_tree in range(tree_count):
            leaf_i_all_trees.append(leaf_list[i_tree][i_leaf])
        leaves_concatenated.append(jnp.concatenate(leaf_i_all_trees))
    return treedef.unflatten(leaves_concatenated)


def shape_dtype(pytree: Any) -> Any:
    """Return pytree leaves shape and data type.

    The function takes a
    [pytree](https://jax.readthedocs.io/en/latest/pytrees.html) and returns its
    shape and data type.
    """
    treedef = tree_util.tree_structure(pytree)
    leaf = tree_util.tree_leaves(pytree)
    leaf_count = len(leaf)
    leaves_shape_dtype = []
    for i_leaf in range(leaf_count):
        leaves_shape_dtype.append(
            ShapeDtypeStruct(
                leaf[i_leaf].shape,
                leaf[i_leaf].dtype))
    return treedef.unflatten(leaves_shape_dtype)


def nbytes(pytree: Any) -> int:
    """Total bytes consumed by the elements of the pytree.

    The function takes a
    [pytree](https://jax.readthedocs.io/en/latest/pytrees.html) and returns its
    memory occupancy, i.e., the number of bytes it takes.
    """
    leaf = tree_util.tree_leaves(pytree)
    leaf_count = len(leaf)
    nbytes = 0
    for i_leaf in range(leaf_count):
        nbytes += leaf[i_leaf].nbytes
    return nbytes


def leaves_at_indices(indices: jnp.ndarray, pytree: Any) -> Any:
    """Return the indexed leaves of a pytree.

    The function takes indices and a
    [pytree](https://jax.readthedocs.io/en/latest/pytrees.html) and returns a
    pytree with the same definition full of data indexed by the indices.

    Parameters
    ----------
    indices : ndarray
        An array of integers representing the leaves array's indices.
    pytree : Any
        All the leaves must be a 1-D array of equal size.

    Return
    ------
    Any
        A subset of the input pytree, whose elements are given by the indices.
    """
    treedef = tree_util.tree_structure(pytree)
    leaf = tree_util.tree_leaves(pytree)
    leaf_count = len(leaf)
    leaves_subset = []
    for i_leaf in range(leaf_count):
        leaves_subset.append(leaf[i_leaf][indices])
    return treedef.unflatten(leaves_subset)


def slice(start: int, end: int, pytree: Any) -> Any:
    """Slice a pytree.

    Parameters
    ----------
    start : int
        Start index
    end : int
        End index (not included)
    pytree : Any
        All the leaves must have slicable ndarray with the given start and end
        indices.

    Returns
    -------
    Any
        A slice of the input pytree.
    """
    treedef = tree_util.tree_structure(pytree)
    leaf = tree_util.tree_leaves(pytree)
    leaves_subset = []
    for i_leaf in range(len(leaf)):
        leaves_subset.append(leaf[i_leaf][start:end])
    return treedef.unflatten(leaves_subset)


def ravel(pytree: Any) -> Any:
    """Return a pytree with contiguous flattened arrays as leaves.

    The function transforms each ND array of each given
    [pytree](https://jax.readthedocs.io/en/latest/pytrees.html)'s leaf to a
    contiguous flattened array.

    Parameters
    ----------
    pytree : Any
        All the leaves must be a ndarray.

    Return
    ------
    Any
        The flattened pytree.
    """
    treedef = tree_util.tree_structure(pytree)
    leaf = tree_util.tree_leaves(pytree)
    leaves_flattened = []
    for i_leaf in range(len(leaf)):
        leaves_flattened.append(jnp.ravel(leaf[i_leaf]))
    return treedef.unflatten(leaves_flattened)


def set_leaves_at_indices(
        pytree_to_modify: Any,
        pytree_reference: Any,
        indices: jnp.ndarray) -> Any:
    """Set the indexed leaves to reference leaves.

    The function modifies the leaves of a given
    [pytree](https://jax.readthedocs.io/en/latest/pytrees.html) at specified
    indices with a reference pytree.

    Parameters
    ----------

    pytree_to_modify : Any
        The pytree to modify.
    pytree_reference : Any
        The reference pytree.
    indices: ndarray
        The indices of the elements to modify in the leaves.

    Return
    ------
    Any
        The modified pytree.
    """
    treedef = tree_util.tree_structure(pytree_to_modify)
    leaves_to_modify = tree_util.tree_leaves(pytree_to_modify)
    leaves_reference = tree_util.tree_leaves(pytree_reference)
    leaf_count = len(leaves_to_modify)
    for i_leaf in range(leaf_count):
        leaves_to_modify[i_leaf] = leaves_to_modify[i_leaf].at[indices].set(
            leaves_reference[i_leaf][indices])
    return treedef.unflatten(leaves_to_modify)


def all_equal(
        pytree_1: Any,
        pytree_2: Any) -> bool:
    """Return true if all the leaves are equal.

    Parameters
    ----------
    pytree_1, pytree_2 : Any
        Input pytree. The shape of the leaves must be broadcastable to a common
        shape.

    Returns
    -------
    bool
        True if all the leaves are equal, false otherwise.
    """
    leaves_1 = tree_util.tree_leaves(pytree_1)
    leaves_2 = tree_util.tree_leaves(pytree_2)
    leaf_count = len(leaves_1)
    all_equal_val = True
    for i_leaf in range(leaf_count):
        is_equal_i = jnp.all(jnp.equal(leaves_1[i_leaf], leaves_2[i_leaf]))
        all_equal_val = jnp.logical_and(all_equal_val, is_equal_i)
    return all_equal_val


def all_isclose(
        pytree_1: Any,
        pytree_2: Any) -> bool:
    """Return true if all the leaves are close.

    Parameters
    ----------
    pytree_1, pytree_2 : Any
        Input pytree. The shape of the leaves must be broadcastable to a common
        shape.

    Returns
    -------
    bool
        True if all the leaves are close, false otherwise.
    """
    leaves_1 = tree_util.tree_leaves(pytree_1)
    leaves_2 = tree_util.tree_leaves(pytree_2)
    leaf_count = len(leaves_1)
    all_equal_val = True
    for i_leaf in range(leaf_count):
        is_equal_i = jnp.all(jnp.isclose(leaves_1[i_leaf], leaves_2[i_leaf]))
        all_equal_val = jnp.logical_and(all_equal_val, is_equal_i)
    return all_equal_val


def all_isclose_masked(
        pytree_1: Any,
        pytree_2: Any) -> bool:
    """Return true if all the leaves are close. Handle nan, i.e. masked values.

    Parameters
    ----------
    pytree_1, pytree_2 : Any
        Input pytree. The shape of the leaves must be broadcastable to a common
        shape.

    Returns
    -------
    bool
        True if all the leaves are close, false otherwise. Here `nan`s are
        considered closed.
    """
    leaves_1 = tree_util.tree_leaves(pytree_1)
    leaves_2 = tree_util.tree_leaves(pytree_2)
    leaf_count = len(leaves_1)
    all_equal_val = True
    for i_leaf in range(leaf_count):
        is_equal_i = jnp.isclose(leaves_1[i_leaf], leaves_2[i_leaf])
        is_equal_i_nan = jnp.logical_and(
            jnp.isnan(leaves_1[i_leaf]),
            jnp.isnan(leaves_2[i_leaf]))
        is_equal_i = jnp.all(jnp.logical_or(is_equal_i_nan, is_equal_i))
        all_equal_val = jnp.logical_and(all_equal_val, is_equal_i)
    return all_equal_val
