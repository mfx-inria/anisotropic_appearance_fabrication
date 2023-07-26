"""
This module contains functions related to mathematics: mathematical constants
and procedures related to angles, vectors, circles, and spheres.
"""

import math

import jax.numpy as jnp
from jax import lax

from .type import float_, int_

# Mathematical constants
TWO_SQRT_PI = 3.5449077018110318
TWO_PI = 6.283185307179586
PI_OVER_2 = 1.5707963267948966


def clamp(val: float | jnp.ndarray,
          low: float | jnp.ndarray,
          high: float | jnp.ndarray) -> float | jnp.ndarray:
    """Clip the provided input to the given interval.

    Parameters
    ----------
    val : float | jnp.ndarray
        The value to clamp.
    min : float | jnp.ndarray
        Minimum value of the interval.
    max : float | jnp.ndarray
        Maximum value of the interval.

    Returns
    -------
    float | jnp.ndarray
        Clipped input
    """
    val = jnp.where(jnp.less(val, low), low, val)
    val = jnp.where(jnp.greater(val, high), high, val)
    return val


def float_same_sign(a: float_, b: float_) -> bool:
    """True if both float have the same sign.

    Parameters
    ----------
    a: float_
        First real value.
    b: float_
        Second real value.

    Returns
    -------
    bool
        `True` if both `a` and `b` have the same sign. Otherwise, `False`.
    """
    a_is_neg = jnp.signbit(a)
    b_is_neg = jnp.signbit(b)
    a_is_pos = jnp.logical_not(a_is_neg)
    b_is_pos = jnp.logical_not(b_is_neg)
    both_neg = jnp.logical_and(a_is_neg, b_is_neg)
    both_pos = jnp.logical_and(a_is_pos, b_is_pos)
    return jnp.logical_or(both_neg, both_pos)


def triangle_filter_eval(x: jnp.ndarray, mean: jnp.ndarray, radius: float):
    """The function evaluates the triangle filter.

    Notes
    -----
    https://www.pbr-book.org/3ed-2018/Texture/Image_Texture#IsotropicTriangleFilter
    """
    x_mean = x - mean
    return jnp.maximum(0., -jnp.linalg.norm(x_mean) + radius)


def roundup_power_of_2(x: int) -> int:
    """Round up to the next power of two.

    Parameters
    ----------
    x : int
        The integer to round up to the next power of two.

    Return
    ------
    int
        The next power of two.
    """
    log2_x = jnp.log2(x)
    log2_x_ceil = jnp.ceil(log2_x).astype(int_)
    return 2**log2_x_ceil


def solve_quadratic_equation(p: jnp.ndarray) -> jnp.ndarray:
    """This function solves the quadratic equation.

    Parameters
    ----------
    p : ndarray
        Contain the numbers a, b, and c, which are the coefficients of the
        equation.

    Returns
    -------
    ndarray
        [nan, nan] if there is no solution. Otherwise returns the solution.
    """
    a = p[0]
    b = p[1]
    c = p[2]
    # Discriminant
    discriminant = b * b - 4. * a * c
    root_discriminant = jnp.sqrt(discriminant)
    # if b < 0.:
    #     q = -0.5 * (b - root_discriminant)
    # else:
    #     q = -0.5 * (b + root_discriminant)
    q = lax.cond(
        b < 0.,
        lambda x: -0.5 * (x[0] - x[1]),
        lambda x: -0.5 * (x[0] + x[1]),
        (b, root_discriminant))
    t0 = q / a
    t1 = c / q
    # if t0 > t1:
    #     return t1, t0
    # else:
    #     return t0, t1
    return lax.cond(t0 > t1, lambda x: jnp.array(
        [x[1], x[0]]), lambda x: jnp.array([x[0], x[1]]), (t0, t1))


def solve_linear_interpolation_equation(v1: float, v2: float) -> float:
    """ Solve (1-u)*v1 + u*v2 = 0 for u.

    The left part of the equation gives the linear interpolated value between
    v1 and v2, where u varies between 0 and 1.
    """
    deno = (v2 - v1)
    return jnp.where(jnp.abs(deno) < 0.0001, 0.5, -v1 / deno)


def vector_normalize(v: jnp.ndarray) -> jnp.ndarray:
    """This function normalizes the given vector.

    Parameters
    ----------
    v : ndarray
        The vector to normalize.

    Returns
    -------
    ndarray
        The normalized vector. If the norm of the vector is zero, the function
        returns the unit vector colinear to the first axis of the domain.
    """
    v_length = jnp.linalg.norm(v)
    unit_vector_x_axis = jnp.zeros_like(v).at[0].set(1.)
    v_normalized = jnp.where(v_length <= 0., unit_vector_x_axis, v / v_length)
    return v_normalized


def unit_circle_sample_uniform_polar(u: float) -> float:
    """Sample the unit circle.

    Transform a uniform scalar in [0, 1[ into a point in the unit circle
    represented by an angle.

    Parameters
    ----------
    u : float
        A uniform number in [0, 1[.

    Returns
    -------
    float
        The point in the unit circle represented by an angle in [-pi, pi[.
    """
    return TWO_PI * u - jnp.pi


def circle_smallest_radius_tangent_to_p_passing_through_q(
        p: jnp.ndarray,
        q: jnp.ndarray,
        T: jnp.ndarray) -> float:
    """Compute the tangent distance.

    This function computes the radius of the smallest circle tangent to a point
    p and passing through a point q.

    Parameters
    ----------
    p : ndarray
        The point p.
    q : ndarray
        The point q.
    T : ndarray
        A vector with norm one represents the tangent associated with the point
        p.

    Return
    ------
    float
        The radius of the smallest circle tangent to a point p and passing
        through a point q.
    """
    p_m_q = p - q
    T_cross_p_m_q_z = T[0] * p_m_q[1] - T[1] * p_m_q[0]
    T_cross_p_m_q_z_norm = jnp.abs(T_cross_p_m_q_z)
    p_m_q_norm = jnp.linalg.norm(p_m_q)
    p_m_q_norm_sqr = p_m_q_norm**2
    div = p_m_q_norm_sqr / T_cross_p_m_q_z_norm
    return div * 0.5


def unit_sphere_sample_uniform_sph(u: jnp.ndarray) -> jnp.ndarray:
    """Sample the unit sphere (spherical coodinates).

    Transform a uniform 2D vector in [0, 1[^2 into a point on the unit sphere
    represented in a spherical coordinate system.

    Parameters
    ----------
    u : ndarray
        A uniform 2D vector in [0, 1[^2.

    Returns
    -------
    ndarray
        The point on the unit sphere in a spherical coordinate system.
    """
    phi = TWO_PI * u[1] - jnp.pi
    theta = jnp.arccos(1. - 2. * u[0])
    return jnp.array([theta, phi])


def unit_sphere_sample_uniform(u: jnp.ndarray) -> jnp.ndarray:
    """Sample the unit sphere.

    Transform a uniform 2D vector in [0, 1[^2 into a point on the unit sphere
    represented in a Cartesian coordinate system.

    Parameters
    ----------
    u : ndarray
        A uniform 2D vector in [0, 1[^2.

    Return
    ------
    ndarray
        The point on the unit sphere in a Cartesian coordinate system.
    """
    z = 1. - 2. * u[0]
    r = jnp.sqrt(jnp.maximum(0., 1 - z * z))
    phi = TWO_PI * u[1]
    return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi), z])


def angle_normalized_to_2ddir(normalized_angle: float) -> jnp.ndarray:
    """Convert a normalized angle to a 2D direction.

    The function converts a normalized polar angle in [0, 1[ into a 2D
    direction defined on the right part of the unit circle.

    Parameters
    ----------
    normalized_angle : float
        The normalized polar angle in [0, 1[ representing the line.

    Returns
    -------
    ndarray
        The function returns the representative direction modeling the line.
    """
    angle = normalized_angle * jnp.pi - jnp.pi / 2.
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])


def angle_weighted_average(a: jnp.ndarray, w: jnp.ndarray) -> float:
    """Return the weighted average of the given angles.

    Parameters
    ----------
    a : ndarray
        An array of angles.
    w : ndarray
        The weights used to weigh each angle of `a`.

    Return
    ------
    float
        A scalar represented the weighted average of the given angles.
    """
    average_a = jnp.exp(1j * a)
    average_a *= w
    average_a = jnp.sum(average_a)
    average_a = jnp.angle(average_a)
    return average_a
