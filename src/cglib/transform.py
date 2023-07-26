import jax.numpy as jnp
from jax import vmap

from .aabb import AABB, aabb_corners, aabb_union_point


def translate(delta: jnp.ndarray) -> jnp.ndarray:
    T = jnp.identity(delta.shape[0]+1)
    T = T.at[:-1, -1].set(delta)
    return T


def scale(s: jnp.ndarray) -> jnp.ndarray:
    T = jnp.identity(s.shape[0]+1)
    for i in range(s.shape[0]):
        T = T.at[i, i].set(s[i])
    return T


def apply_to_point(T: jnp.ndarray, p: jnp.ndarray):
    p_homogeneous = jnp.append(p, 1.)
    p_transformed = T @ p_homogeneous
    return (p_transformed / p_transformed[-1])[:-1]


def apply_to_vector(T: jnp.ndarray, v: jnp.ndarray):
    return T[:-1, :-1] @ v


def apply_to_AABB(T: jnp.ndarray, aabb: AABB) -> AABB:
    corners = aabb_corners(aabb)
    corners_transformed = vmap(apply_to_point, (None, 0))(T, corners)
    aabb_n = AABB(corners_transformed[0], corners_transformed[0])
    for i in range(corners.shape[0] - 1):
        aabb_n_p_1 = aabb_union_point(corners_transformed[i + 1], aabb_n)
        aabb_n = aabb_n_p_1
    return aabb_n