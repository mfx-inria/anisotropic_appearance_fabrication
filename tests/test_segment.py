import jax
import jax.numpy as jnp

import cglib.segment
import cglib.tree_util


def helper_intersect_circle():
    s0 = jnp.array([0., 0.])
    s1 = jnp.array([1., 1.])
    segment = jnp.array([s0, s1])
    circle_center = jnp.array([0.5, 0.5])
    radius = 1.
    res0 = cglib.segment.intersect_circle(segment, circle_center, radius)
    res0_bool = jnp.all(jnp.isnan(res0))
    radius = 0.5
    res1 = cglib.segment.intersect_circle(segment, circle_center, radius)
    res1_exp = jnp.array([0.1464466, 0.8535534])
    res1_bool = jnp.all(jnp.isclose(res1, res1_exp))
    return jnp.logical_and(res0_bool, res1_bool)


def test_intersect_circle():
    assert jax.jit(helper_intersect_circle)()


def helper_intersection():
    p = jnp.array([0., 0.])
    r = jnp.array([1., 1.])
    q = jnp.array([1., 0.])
    s = jnp.array([0., 1.])
    s1 = jnp.array([p, r])
    s2 = jnp.array([q, s])
    res = cglib.segment.intersection(s1, s2)
    res_exp = (True, jnp.array([0.5, 0.5]))
    return cglib.tree_util.all_equal(res, res_exp)


def test_intersection():
    assert jax.jit(helper_intersection)()


def helper_intersection_bool():
    p = jnp.array([0., 0.])
    r = jnp.array([1., 1.])
    q = jnp.array([1., 0.])
    s = jnp.array([0., 1.])
    s1 = jnp.array([p, r])
    s2 = jnp.array([q, s])
    res = cglib.segment.intersection_bool(s1, s2)
    return jnp.where(res, True, False)


def test_intersection_bool():
    assert jax.jit(helper_intersection_bool)()


def helper_patching_energy():
    i1 = jnp.array([0., 0.])
    i2 = jnp.array([1., 0.])
    j1 = jnp.array([0., 1.])
    j2 = jnp.array([1., 1.])
    res = cglib.segment.patching_energy(i1, i2, j1, j2)
    return res == 0.


def test_patching_energy():
    assert jax.jit(helper_patching_energy)()


def helper_closest_point():
    x = jnp.array([0., 0.])
    s = jnp.array([[1., -1.],
                   [1., 1.]])
    res = cglib.segment.closest_point(x, s)
    res_exp = jnp.array([1., 0.])
    return jnp.all(jnp.isclose(res, res_exp))


def test_closest_point():
    assert jax.jit(helper_closest_point)()


def helper_distance_to_point():
    x = jnp.array([0., 0.])
    p = jnp.array([1., 1.])
    q = jnp.array([1., 1.])
    s = jnp.array([p, q])
    res = cglib.segment.distance_to_point(x, s)
    return jnp.isclose(res, jnp.sqrt(2))


def test_distance_to_point():
    assert jax.jit(helper_distance_to_point)()


def helper_tangent():
    #   Input segment
    #   p_1 o----o p_0
    p_0 = jnp.array([2., 0.])
    p_1 = jnp.array([0., 0.])
    tangent = cglib.segment.tangent(p_0, p_1)
    return jnp.all(jnp.equal(tangent, jnp.array([-1., 0.])))


def test_tangent():
    assert jax.jit(helper_tangent)()


def helper_repulse_point_then_project_on_segment():
    s0 = jnp.array([0., 0.])
    s1 = jnp.array([1., 0.])
    segment = jnp.array([s0, s1])
    circle_center = s0
    radius = 0.25
    u = 0.1
    res0 = cglib.segment.repulse_point_then_project_on_segment(
        u, segment, circle_center, radius)
    res0_bool = res0 == 0.25
    radius = 0.05
    res1 = cglib.segment.repulse_point_then_project_on_segment(
        u, segment, circle_center, radius)
    res1_bool = jnp.isnan(res1)
    return jnp.logical_and(res0_bool, res1_bool)


def test_repulse_point_then_project_on_segment():
    assert jax.jit(helper_repulse_point_then_project_on_segment)()
