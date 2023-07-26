import jax.numpy as jnp
from jax import lax

from cglib.math import solve_quadratic_equation, vector_normalize


def segment_point_from_interpolant(
        interpolant: float, s: jnp.ndarray) -> jnp.ndarray:
    return s[0] + interpolant * (s[1] - s[0])


def intersect_circle(
        s: jnp.ndarray,
        c: jnp.ndarray,
        r: float) -> tuple[float, float]:
    """Intersect a segment with a circle and returns the intersection points.

    Parameters
    ----------
    s : ndarray
        The segment has two endpoints, represented by a matrix where each row
        is a point.
    c : ndarray
        Circle center.
    r : float
        Circle radius.

    Returns
    -------
    ndarray
        The segment is represented with its parametric form S(u) = s[0] + u *
        (s[1] - s[0]), u in [0, 1], giving its set of points. The two
        intersection points are represented by the scalar value u, with the
        first one being the closest of `s[0]`. When there is no intersection
        point, `[nan, nan]` is returned.
    """
    # autopep8: off
    #     .__________________>
    #             v
    #              ____
    # s[0]._____u0/____\u1___.s[1]
    #            |  .c  |
    #             \_|r_/

    # solve || s[0] + u*v - c || = r for u
    # (s[0] + u*v - c) dot (s[0] + u*v - c) - r*r = 0

    # s[0] dot s[0] + s[0] dot u*v - s[0] dot c
    # + u*v dot s[0] + u*u * v dot v - u*v dot c
    # - c dot s[0] - c dot u*v + c dot c - r*r= 0

    # u**2 * v dot v +
    # u    * 2. * (s[0] dot v - v dot c) +
    # s[0] dot s[0] - 2. * s[0] dot c + c dot c
    # autopep8: on
    v = s[1] - s[0]
    a = jnp.vdot(v, v)
    b = 2. * (jnp.vdot(s[0], v) - jnp.vdot(v, c))
    c = jnp.vdot(s[0], s[0]) - 2. * jnp.vdot(s[0], c) + jnp.vdot(c, c) - r * r
    u = solve_quadratic_equation(jnp.array([a, b, c]))
    cond = jnp.logical_or(jnp.logical_or(u < 0., u > 1.), jnp.isnan(u))
    u = jnp.where(cond, jnp.nan, u)
    return jnp.sort(u)


def intersection(s1: jnp.ndarray, s2: jnp.ndarray) -> tuple[bool, jnp.ndarray]:
    """Returns true and the intersection point if there is an intersection.

    Parameters
    ----------
    s1, s2 : ndarray
        The segments s1 and s2 are represented by two endpoints (p, r) and (q,
        s), resp.
    """
    p = s1[0]
    r = s1[1] - p
    q = s2[0]
    s = s2[1] - q

    q_m_p = q - p

    r_cross_s_z = r[0] * s[1] - r[1] * s[0]
    q_m_p_cross_r_z = q_m_p[0] * r[1] - q_m_p[1] * r[0]
    q_m_p_cross_s_z = q_m_p[0] * s[1] - q_m_p[1] * s[0]

    r_cross_s_z_neq_0 = jnp.logical_not(jnp.equal(r_cross_s_z, 0.))

    u = q_m_p_cross_r_z / r_cross_s_z
    t = q_m_p_cross_s_z / r_cross_s_z

    intersection_bool = u >= 0.
    intersection_bool = jnp.logical_and(intersection_bool, u <= 1.)
    intersection_bool = jnp.logical_and(intersection_bool, t >= 0)
    intersection_bool = jnp.logical_and(intersection_bool, t <= 1.)
    intersection_bool = jnp.logical_and(intersection_bool, r_cross_s_z_neq_0)
    intersection_point = q + u * s
    return intersection_bool, intersection_point


def intersection_bool(s1: jnp.ndarray, s2: jnp.ndarray) -> bool:
    """Return true if the segments intersect.

    Parameters
    ----------
    s1, s2 : ndarray
        The segments s1 and s2 are represented by two endpoints (p, r) and (q,
        s), resp.

    Returns
    -------
    bool
        This function returns `True` if the segment s1 = (p, r) intersects the
        segment s2 = (q, s). Otherwise returns false.
    """
    p = s1[0]
    r = s1[1] - p
    q = s2[0]
    s = s2[1] - q

    q_m_p = q - p

    r_cross_s_z = r[0] * s[1] - r[1] * s[0]
    q_m_p_cross_r_z = q_m_p[0] * r[1] - q_m_p[1] * r[0]
    q_m_p_cross_s_z = q_m_p[0] * s[1] - q_m_p[1] * s[0]

    r_cross_s_z_neq_0 = jnp.logical_not(jnp.equal(r_cross_s_z, 0.))

    u = q_m_p_cross_r_z / r_cross_s_z
    t = q_m_p_cross_s_z / r_cross_s_z

    intersection = u >= 0.
    intersection = jnp.logical_and(intersection, u <= 1.)
    intersection = jnp.logical_and(intersection, t >= 0)
    intersection = jnp.logical_and(intersection, t <= 1.)
    intersection = jnp.logical_and(intersection, r_cross_s_z_neq_0)
    return intersection


def patching_energy(
        i1: jnp.ndarray,
        i2: jnp.ndarray,
        j1: jnp.ndarray,
        j2: jnp.ndarray) -> float:
    """Return the patching energy between segments.

    The function returns the patching energy of an edge/segment I connecting
    two vertices, i1 and i2, with another edge/segment J connecting two
    vertices, j1 and j2.

    Parameters
    ----------
    i1 : ndarray
        The first vertex `i1` of the first edge I.
    i2 : ndarray
        The second vertex `i1` of the first edge I.
    j1 : ndarray
        The first vertex `j1` of the second edge J.
    j2 : ndarray
        The second vertex `j2` of the second edge J.

    Returns
    -------
    float
        The patching energy. See Chermain et al. 2023, Eq.14 for the
        mathematical definition.
    """
    norm_i1_i2 = jnp.linalg.norm(i1 - i2)
    norm_j1_j2 = jnp.linalg.norm(j1 - j2)
    norm_i1_j2 = jnp.linalg.norm(i1 - j2)
    norm_i2_j1 = jnp.linalg.norm(i2 - j1)
    norm_i1_j1 = jnp.linalg.norm(i1 - j1)
    norm_i2_j2 = jnp.linalg.norm(i2 - j2)

    energy_removed = norm_i1_i2 + norm_j1_j2
    energy_config_0 = norm_i1_j2 + norm_i2_j1 - energy_removed
    energy_config_1 = norm_i1_j1 + norm_i2_j2 - energy_removed
    patching_energy = jnp.where(
        energy_config_0 <= energy_config_1,
        energy_config_0,
        energy_config_1)
    return patching_energy


def closest_point(
        x: jnp.ndarray,
        s: jnp.ndarray) -> jnp.ndarray:
    """Return the closest segment's point from a given point.

    Parameters
    ----------
    x : ndarray
        The point.
    s : ndarray
        The segment is defined as a 2D array with shape (2, N), with N the
        number of components of the segment's endpoints.

    Return
    ------
    ndarray
        The function returns a N-D array representing the closest point on a
        segment for a query point.

    Notes
    -----
    Mathematical details: https://youtu.be/bZbuKOxH71o?t=679
    """
    p = s[0]
    q = s[1]

    q_m_p = q - p
    q_m_p_norm = jnp.linalg.norm(q_m_p)
    q_m_p_norm_sqr = q_m_p_norm**2
    t = jnp.vdot(x - p, q_m_p) / q_m_p_norm_sqr

    closest_point = (1. - t) * p + t * q
    closest_point = jnp.where(t < 0., p, closest_point)
    closest_point = jnp.where(t > 1., q, closest_point)
    closest_point = jnp.where(jnp.isnan(t), p, closest_point)

    return closest_point


def distance_to_point(x: jnp.ndarray, s: jnp.ndarray) -> float:
    """Return the distance between a segment and a point.

    This function returns the distance between the closest segment's point from
    a given point and the given point.

    Parameters
    ----------
    x : ndarray
        The point.
    s : ndarray
        The segment is defined as a 2D array with shape (2, N), with N the
        number of components of the segment's endpoints.

    Returns
    -------
    float
        This function returns the distance between the closest segment's point
        from a given point and the given point.
    """
    closest_point_val = closest_point(x, s)
    return jnp.linalg.norm(closest_point_val - x)


def tangent(p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
    """Computes the tangent of a segment.

    Parameters
    ----------
    p1, p2 : ndarray
        First and second endpoint of the oriented segment.

    Returns
    ------
    ndarray
        The tangent is defined as (p2 - p1) / norm(p2 - p1).
    """
    p2_p1 = p2 - p1
    return vector_normalize(p2_p1)


def sample_uniform(p1: jnp.ndarray, p2: jnp.ndarray, u: float):
    """Sample uniformly a segment.

    Transform a uniform float in [0, 1[ into a point on the segment represented
    by two endpoints.

    Parameters
    ----------
    p1 : ndarray
        The start point of the segment.
    p2 : ndarray
        The endpoint of the segment.
    u : ndarray
        A uniform float in [0, 1).

    Returns
    -------
    ndarray
        Returns p1 + (p2 - p1) * u.
    """
    v = p2 - p1
    return p1 + v * u


def repulse_point_then_project_on_segment(
        u: float,
        s: jnp.ndarray,
        p: jnp.ndarray,
        d: float) -> jnp.ndarray:
    """Repulse a point from another one, and project it on a segment.

    Parameters
    ----------
    u : float
        The relative distance of the point that will be repulsed on the
        segment.
    s : ndarray
        The segment on which the point to be repulsed is defined. The repulsed
        point is defined as `ps = s[0] + u * (s[1] - s[0])`. The segment has
        two endpoints, represented by a matrix where each row is a point.
    p : ndarray
        The point that repulses, in cartesian coordinate.
    d : float
        The point will be pushed away from point `p` by distance `d`.

    Returns
    -------
    float
        If the point to push away is below distance `d` from `p`, then return
        the closest interpolant from `u`, giving a segment point which is the
        intersection between a circle of radius `d` and center `p` and the
        segment `s`. Otherwise, return `nan`.
    """
    v = s[1] - s[0]
    ps = s[0] + u * v
    ps_p_length = jnp.linalg.norm(ps - p)

    def below_distance_fun(s, p, d, u):
        us_inter = intersect_circle(s, p, d)
        d_us_inter_u = jnp.abs(us_inter - u)
        us_inter_index = jnp.argsort(d_us_inter_u)
        return us_inter[us_inter_index[0]]

    return lax.cond(
        ps_p_length < d,
        lambda x: below_distance_fun(*x),
        lambda x: jnp.nan,
        (s, p, d, u))
