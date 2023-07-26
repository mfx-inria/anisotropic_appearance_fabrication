from jax import jit


def helper_test_polar_to_cartesian():
    import jax.numpy as jnp

    import cglib.direction

    res = cglib.direction.polar_to_cartesian(jnp.pi * 0.5)
    return jnp.all(jnp.isclose(res, jnp.array([-4.371139e-08, 1.000000e+00])))


def test_polar_to_cartesian():
    assert jit(helper_test_polar_to_cartesian)()


def helper_test_cartesian_to_polar():
    import jax.numpy as jnp

    import cglib.direction

    res = cglib.direction.cartesian_to_polar(jnp.array([0., 1.]))
    return jnp.isclose(res, 1.5707964)


def test_cartesian_to_polar():
    assert jit(helper_test_cartesian_to_polar)()


def helper_test_spherical_to_cartesian():
    import jax.numpy as jnp

    import cglib.direction

    res = cglib.direction.spherical_to_cartesian(jnp.array([0., 0.]))
    return jnp.all(jnp.isclose(res, jnp.array([0., 0., 1.])))


def test_spherical_to_cartesian():
    assert jit(helper_test_spherical_to_cartesian)()


def helper_test_cartesian_to_spherical():
    import jax.numpy as jnp

    import cglib.direction

    a = jnp.array([0., 0., 1.])
    sph_a = cglib.direction.cartesian_to_spherical(a)
    inv_sph_sph_a = cglib.direction.spherical_to_cartesian(sph_a)
    return jnp.all(jnp.isclose(inv_sph_sph_a, a))


def test_cartesian_to_spherical():
    assert jit(helper_test_cartesian_to_spherical)()


def helper_test_average():
    import jax.numpy as jnp

    import cglib.direction

    ds = [[1., 0.], [0., 1.]]
    res = cglib.direction.average(jnp.array(ds))
    return jnp.all(jnp.isclose(res, jnp.array([0.70710677, 0.70710677])))


def test_average():
    assert jit(helper_test_average)()
