import jax.numpy as jnp


def average_with_weights(
        lines: jnp.ndarray,
        weights: jnp.ndarray) -> jnp.ndarray:
    """Average lines with weights.

    The function returns a weighted average of the lines defined by an array of
    representative directions.

    Parameters
    ----------
    lines : jnp.ndarray
        A collection of lines is represented by a (m x n) matrix, where m is
        the number of lines and n is the number of representative direction
        components. A representative direction represents a line.
    weights : jnp.ndarray
        The weights with shape (m) are used to weight each line.

    Returns
    -------
    jnp.ndarray
        The function returns the representative direction modeling the averaged
        line.
    """
    lines *= weights.reshape((-1, 1))
    Q = lines.T @ lines
    _, v = jnp.linalg.eigh(Q)
    return v[:, -1]


def smoothness_energy(d: jnp.ndarray, D: jnp.ndarray) -> float:
    """Return the smoothness energy between a line and a collection of lines.

    Parameters
    ----------
    d : jnp.ndarray
        A line represented by a direction d with n components.
    D : jnp.ndarray
        A collection of lines is represented by a (m x n) matrix, where m is
        the number of lines and n is the number of representative direction
        components. A representative direction represents a line.

    Returns
    -------
    float
        The function returns the smoothness energy between a line and a
        collection of lines.
    """
    n = d.shape[0]
    d = d.reshape((n, 1))
    D = D.reshape((-1, n))
    cov = D.T @ D
    cov_d = cov @ d
    return (-d.T @ cov_d)[0, 0]


def smoothness_energy_normalized(d: jnp.ndarray, D: jnp.ndarray) -> float:
    """Return the normalized smoothness energy.

    The function returns the smoothness energy between a line and a collection
    of lines. The result is scaled between 0 and 1.

    Parameters
    ----------
    d : jnp.ndarray
        A line represented by a direction d with n components.
    D : jnp.ndarray
        A collection of lines is represented by a (m x n) matrix, where m is
        the number of lines and n is the number of representative direction
        components. A representative direction represents a line.
    Returns
    -------
    float
    """
    other_line_count = D.shape[0]
    res = smoothness_energy(d, D)
    # Normalized
    return (res / other_line_count) + 1
