"""
This module contains functions related to
[directions](https://en.wikipedia.org/wiki/Orientation_(geometry)).
"""

import jax.numpy as jnp

from .math import unit_circle_sample_uniform_polar


def polar_to_cartesian(angle: float) -> jnp.ndarray:
    """Convert a polar direction to cartesian.

    Convert the 2D direction represented with an angle to a direction in S^1
    with Cartesian coordinates.

    Parameters
    ----------
    angle : float
        The angle represents the direction.

    Returns
    -------
    ndarray
        The function returns a jnp.ndarray with the shape (2,) representing the
        direction in Cartesian coordinates.
    """
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])


def cartesian_to_polar(d: jnp.ndarray) -> float:
    """Cartesian to polar direction.

    Convert the 2D direction represented in Cartesian space with a two
    components vector in S^1 to a 2D direction represented with an angle in
    [-pi, pi].

    Parameters
    ----------

    d : jnp.ndarray
        The direction in a Jax array with the shape (2,).

    Returns
    -------
    float
        The function returns a float representing the direction in polar
        coordinates. The angle is in radians, in the range [-pi, pi].
    """
    return jnp.arctan2(d[1], d[0])


def spherical_to_cartesian(d: jnp.ndarray) -> jnp.ndarray:
    """Convert the 3D spherical direction into cartesian space.

    Parameters
    ----------
    d : ndarray
        The spherical direction in a Jax array with the shape (2,). The first
        component gives the polar angle, i.e., the angle between the sphere's
        north pole and direction. The second component gives the azimuthal
        angle., i.e., the angle between the x-axis and the direction.

    Returns
    -------
    ndarray
        The function returns a jnp.ndarray with the shape (3,) representing the
        direction in Cartesian coordinates.
    """
    theta = d[0]
    phi = d[1]
    d = jnp.array([jnp.cos(phi) * jnp.sin(theta),
                   jnp.sin(phi) * jnp.sin(theta),
                   jnp.cos(theta)])
    return d


def cartesian_to_spherical(d: jnp.ndarray) -> jnp.ndarray:
    """Convert the 3D cartesian direction into spherical space.

    Parameters
    ----------
    d : ndarray
        The cartesian direction in a Jax array with the shape (3,).

    Returns
    -------
    ndarray
        The function returns a jnp.ndarray with the shape (2,) representing the
        direction in spherical coordinates. The first component gives the polar
        angle, i.e., the angle between the sphere's north pole and direction.
        The second component gives the azimuthal angle., i.e., the angle
        between the x-axis and the direction.
    """
    d = jnp.array([jnp.arccos(d[2]), jnp.arctan2(d[1], d[0])])
    return d


def average(ds: jnp.ndarray) -> jnp.ndarray:
    """Average the given directions.

    Parameters
    ----------
    ds : jnp.ndarray
        A collection of directions is represented by a (m x n) matrix, where m
        is the number of directions and n is the number of direction
        components.

    Returns
    -------
    jnp.ndarray
        The averaged direction.
    """
    average_d = jnp.sum(ds, axis=0)
    average_d_norm = jnp.linalg.norm(average_d)
    ones = jnp.ones_like(average_d)
    cond = average_d_norm <= 0.
    average_d = jnp.where(
        cond,
        ones / jnp.linalg.norm(ones),
        average_d / average_d_norm)
    return average_d


def sample_uniform_2d(u: float) -> jnp.ndarray:
    """Sample uniformly a 2D direction.

    Transform a uniform scalar in [0, 1[ into a point on the unit circle
    represented in a Cartesian coordinate system.

    Parameters
    ----------
    u : float
        A uniform number in [0, 1[.

    Returns
    -------
    jnp.ndarray
        The point on the unit circle in a Cartesian coordinate system.
    """
    return polar_to_cartesian(unit_circle_sample_uniform_polar(u))
