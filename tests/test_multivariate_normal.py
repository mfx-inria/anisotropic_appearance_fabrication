import jax
import jax.numpy as jnp

import cglib.multivariate_normal


def helper_eval_pdf():
    x = jnp.array([0., 0.])
    mean = jnp.array([0., 0.])
    variance = 1.
    res = cglib.multivariate_normal.eval_pdf(x, mean, variance)
    res_exp = 0.15915495
    return jnp.isclose(res, res_exp)


def test_eval_pdf():
    assert jax.jit(helper_eval_pdf)()
