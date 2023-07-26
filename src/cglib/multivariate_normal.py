"""
This module contains functions related to the [multivariate normal
distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution).
"""


import jax.numpy as jnp
import jax.scipy.stats.multivariate_normal as multivariate_normal


def std_dev_from_radius(radius: float) -> float:
    """
    This function returns the standard deviation of an isotropic Gaussian,
    i.e., a [normal
    distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution),
    whose radius corresponds to the specified given argument. It should be used
    for a Gaussian with less than four dimensions. See the [68-95-99.7
    rule](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule).
    """
    return radius / 3.


def variance_from_radius(radius: float) -> float:
    """
    This function returns the variance of an isotropic Gaussian, i.e., a
    [normal
    distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution),
    whose radius corresponds to the given argument. It should be used for a
    Gaussian with less than four dimensions. See the [68-95-99.7
    rule](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule).
    """
    std_dev = std_dev_from_radius(radius)
    return std_dev * std_dev


def eval_pdf(
        x: jnp.ndarray,
        mean: jnp.ndarray,
        variance: float) -> float:
    """Returns the probability density of the multivariate normal distribution.

    Parameters
    ----------
    x : jnp.ndarray
        The evaluation point with n components. Shape (n).
    mean : jnp.ndarray
        The mean of the distribution. Shape (n).
    variance : float
        The isotropic variance of the distribution is used to build the
        covariance matrix C = variance * identity, where identity is a n x n
        identity matrix.

    Returns
    -------
    float
        The probability density function of the multivariate normal
        distribution.
    """
    cov = jnp.identity(x.shape[0]) * variance
    spatial_weight = multivariate_normal.pdf(x, mean, cov=cov)
    return spatial_weight
