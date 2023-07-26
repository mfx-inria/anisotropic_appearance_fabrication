import jax
import jax.numpy as jnp

import cglib.polyline
import cglib.tree_util


def helper_point_tangent_and_weight():
    #   Input polyline
    #      p_2
    #       o----o p_1
    #      /     |
    # p_3 o      o p_0
    #      \     |
    #   p_4 o----o p_5
    #
    polyline = jnp.array([[0., 0.],  # p_0
                          [0., 1.],  # p_1
                          [-1., 1.],  # p_2
                          [-1.5, 0.],  # p_3
                          [-1., -1.],  # p_4
                          [0., -1.]])  # p_5
    res0 = cglib.polyline.point_tangent_and_weight(0, polyline)
    res0_exp = (jnp.array([0., 1.]), jnp.array(1.))
    res0_bool = cglib.tree_util.all_isclose(res0, res0_exp)
    res1 = cglib.polyline.point_tangent_and_weight(3, polyline)
    res1_exp = (jnp.array([0., -1.]), jnp.array(1.118034))
    res1_bool = cglib.tree_util.all_isclose(res1, res1_exp)
    return jnp.logical_and(res0_bool, res1_bool)


def test_point_tangent_and_weight():
    assert jax.jit(helper_point_tangent_and_weight)()


def helper_points_tangent_and_weight():
    #   Input polyline
    #      p_2
    #       o----o p_1
    #      /     |
    # p_3 o      o p_0
    #      \     |
    #   p_4 o----o p_5
    #
    polyline = jnp.array([[0., 0.],  # p_0
                          [0., 1.],  # p_1
                          [-1., 1.],  # p_2
                          [-1.5, 0.],  # p_3
                          [-1., -1.],  # p_4
                          [0., -1.]])  # p_5
    res = cglib.polyline.points_tangent_and_weight(polyline)
    res_exp = (jnp.array([[0.,  1.],
                          [-0.70710677,  0.70710677],
                          [-0.8506508, -0.5257311],
                          [0., -1.],
                          [0.8506508, -0.5257311],
                          [0.70710677,  0.70710677]]),
               jnp.array([1., 1., 1.059017, 1.118034, 1.059017, 1.]))
    return cglib.tree_util.all_isclose(res, res_exp)


def test_points_tangent_and_weight():
    assert jax.jit(helper_points_tangent_and_weight)()


def helper_point_tangent_half_distance_to_segment():
    #   Input polyline
    #      p_2
    #       o----o p_1
    #      /     |
    # p_3 o      o p_0
    #      \     |
    #   p_4 o----o p_5
    #
    polyline = jnp.array([[0., 0.],  # p_0
                          [0., 1.],  # p_1
                          [-1., 1.],  # p_2
                          [-1.5, 0.],  # p_3
                          [-1., -1.],  # p_4
                          [0., -1.]])  # p_5
    # Tangent half distance from point p_0 to p_3's edges.
    res = cglib.polyline.point_tangent_half_distance_to_segment(0, 3, polyline)
    res_exp = 0.725
    return res == res_exp


def test_point_tangent_half_distance_to_segment():
    assert jax.jit(helper_point_tangent_half_distance_to_segment)()


def helper_point_tangent_half_distance_to_all_segments():
    #   Input polyline
    #      p_2
    #       o----o p_1
    #      /     |
    # p_3 o      o p_0
    #      \     |
    #   p_4 o----o p_5
    #
    polyline = jnp.array([[0., 0.],  # p_0
                          [0., 1.],  # p_1
                          [-1., 1.],  # p_2
                          [-1.5, 0.],  # p_3
                          [-1., -1.],  # p_4
                          [0., -1.]])  # p_5
    # Tangent half distance from point p_0 to p_3's edges.
    res = cglib.polyline.point_tangent_half_distance_to_all_segments(
        0, polyline)
    res_exp = 0.725
    return res == res_exp


def test_point_tangent_half_distance_to_all_segments():
    assert jax.jit(helper_point_tangent_half_distance_to_all_segments)()


def helper_points_tangent_half_distance_to_all_segments():
    #   Input polyline
    #      p_2
    #       o----o p_1
    #      /     |
    # p_3 o      o p_0
    #      \     |
    #   p_4 o----o p_5
    #
    polyline = jnp.array([[0., 0.],  # p_0
                          [0., 1.],  # p_1
                          [-1., 1.],  # p_2
                          [-1.5, 0.],  # p_3
                          [-1., -1.],  # p_4
                          [0., -1.]])  # p_5
    res = cglib.polyline.points_tangent_half_distance_to_all_segments(polyline)
    res_exp = jnp.array([0.72500002, 1.16666675, 0.87267792,
                        0.8185606, 0.87267792, 1.16666675])
    return jnp.all(jnp.isclose(res, res_exp))


def test_points_tangent_half_distance_to_all_segments():
    assert helper_points_tangent_half_distance_to_all_segments()
