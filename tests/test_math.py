from jax import jit


def helper_test_clamp():
    import jax.numpy as jnp

    import cglib.math

    val = jnp.array([-10., 10.])
    low = jnp.array([-5., 2.])
    high = jnp.array([10., 3.])
    res = cglib.math.clamp(val, low, high)
    return jnp.all(jnp.equal(res, jnp.array([-5., 3.])))


def test_clamp():
    assert jit(helper_test_clamp)()


def helper_test_float_same_sign():
    import jax.numpy as jnp

    import cglib.math

    res1 = cglib.math.float_same_sign(-0., 0.)
    res1 = jnp.equal(res1, False)
    res2 = cglib.math.float_same_sign(0., 0.)
    return jnp.logical_and(res1, res2)


def test_float_same_sign():
    assert jit(helper_test_float_same_sign)()


def helper_test_roundup_power_of_2():
    import jax.numpy as jnp

    import cglib.math

    res = cglib.math.roundup_power_of_2(5)
    return jnp.equal(res, 8)


def test_roundup_power_of_2():
    assert jit(helper_test_roundup_power_of_2)()


def helper_test_solve_quadratic_equation():
    import jax.numpy as jnp

    import cglib.math

    a = 1.
    b = -1.
    c = -2.
    p = jnp.array([a, b, c])
    res = cglib.math.solve_quadratic_equation(p)
    return jnp.all(jnp.equal(res, jnp.array([-1, 2.])))


def test_solve_quadratic_equation():
    assert jit(helper_test_solve_quadratic_equation)()


def helper_test_vector_normalize():
    import jax.numpy as jnp

    import cglib.math

    v = jnp.array([1., 1.])
    res = cglib.math.vector_normalize(v)
    return jnp.all(jnp.isclose(res, jnp.array([0.70710677, 0.70710677])))


def test_vector_normalize():
    assert jit(helper_test_vector_normalize)()


def helper_test_circle_smallest_radius_tangent():
    import jax.numpy as jnp

    import cglib.math

    p = jnp.array([0., 0.])
    q = jnp.array([1., 0.])

    T = jnp.array([0., 1.])
    res1 = cglib.math.circle_smallest_radius_tangent_to_p_passing_through_q(
        p, q, T)
    res1 = jnp.isclose(res1, 0.5)
    # 0.5
    T = jnp.array([1., 0.])
    res2 = cglib.math.circle_smallest_radius_tangent_to_p_passing_through_q(
        p, q, T)
    res2 = jnp.isinf(res2)
    return jnp.logical_and(res1, res2)


def test_circle_smallest_radius_tangent_to_p_passing_through_q():
    assert jit(helper_test_circle_smallest_radius_tangent)()


def helper_test_unit_sphere_sample_uniform_sph():
    import jax.numpy as jnp

    import cglib.math

    res = cglib.math.unit_sphere_sample_uniform_sph(jnp.array([0.5, 0.25]))
    return jnp.all(jnp.isclose(res, jnp.array([1.5707964, -1.5707964])))


def test_unit_sphere_sample_uniform_sph():
    assert jit(helper_test_unit_sphere_sample_uniform_sph)()


def helper_test_unit_sphere_sample_uniform():
    import jax.numpy as jnp

    import cglib.math

    res = cglib.math.unit_sphere_sample_uniform(jnp.array([0.5, 0.25]))
    return jnp.all(jnp.isclose(res, jnp.array(
        [-4.371139e-08, 1.000000e+00, 0.000000e+00])))


def test_unit_sphere_sample_uniform():
    assert jit(helper_test_unit_sphere_sample_uniform)()


def helper_test_angle_normalized_to_2ddir():
    import jax.numpy as jnp

    import cglib.math

    res = cglib.math.angle_normalized_to_2ddir(0.75)
    return jnp.all(jnp.isclose(res, jnp.array([0.70710677, 0.7071068])))


def test_angle_normalized_to_2ddir():
    assert jit(helper_test_angle_normalized_to_2ddir)()


def helper_test_angle_weighted_average():
    import jax.numpy as jnp

    import cglib.math

    angles = jnp.array([0.75 * jnp.pi, -0.75 * jnp.pi])
    weights = jnp.array([1., 1.])
    res = cglib.math.angle_weighted_average(angles, weights)
    return jnp.isclose(res, jnp.pi)


def test_angle_weighted_average():
    assert jit(helper_test_angle_weighted_average)()
