"""
This module contains a data structure representing a masked array. A masked
array comprises an array of items and an array of masks, i.e., boolean values.
The mask indicates if an item is valid or not.
"""

from typing import NamedTuple

import jax.numpy as jnp

from .type import uint

# Not An Index
NAI = uint(jnp.iinfo(uint).max)


class MaskedArray(NamedTuple):
    """Represent an array with masked elements.

    Attributes
    ----------
    array : jnp.ndarray
        An array of items.
    mask : jnp.ndarray
        A boolean array with the same shape of `array`, where each element
        indicates if the item of `array` is masked or not.
    """
    array: jnp.ndarray
    mask: jnp.ndarray


def isclose_masked(a: jnp.ndarray, b: jnp.ndarray):
    isclose = jnp.isclose(a, b)
    is_both_nan = jnp.logical_and(jnp.isnan(a), jnp.isnan(b))
    return jnp.all(jnp.logical_or(is_both_nan, isclose))
