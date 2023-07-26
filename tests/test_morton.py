import jax.numpy as jnp
from jax import jit, random, vmap

import cglib.morton
import cglib.type


def helper_test_morton():
    TEST_COUNT = 4
    SIZE_2 = (TEST_COUNT, 2)
    SIZE_3 = (TEST_COUNT, 3)
    key = random.PRNGKey(0)

    # Initialization procedural data
    if not cglib.type.x64:
        p_2_random = jnp.floor(
            random.uniform(
                key,
                SIZE_2) *
            0x0000ffff).astype(
            cglib.type.uint)
        key, subkey = random.split(key)
        p_3_random = jnp.floor(
            random.uniform(
                subkey,
                SIZE_3) *
            0x000000ff).astype(
            cglib.type.uint)
    else:
        p_2_random = jnp.floor(
            random.uniform(
                key,
                SIZE_2) *
            0x00000000ffffffff).astype(
            cglib.type.uint)
        key, subkey = random.split(key)
        p_3_random = jnp.floor(
            random.uniform(
                subkey,
                SIZE_3) *
            0x000000000000ffff).astype(
            cglib.type.uint)

    p_2_1 = jnp.array([123, 456], cglib.type.uint)
    p_2_1_encoded = cglib.type.uint(177605)
    p_2_2 = jnp.array([100, 200], cglib.type.uint)
    p_2_2_encoded = cglib.type.uint(46224)
    p_3_1 = jnp.array([100, 200, 50], cglib.type.uint)
    p_3_1_encoded = cglib.type.uint(5162080)

    encode_vmap = vmap(cglib.morton.encode)
    decode_vmap = vmap(cglib.morton.decode, (0, None))
    encoded = encode_vmap(p_2_random)
    decoded = decode_vmap(encoded, 2)

    test_res = jnp.all(decoded == p_2_random)
    encoded = encode_vmap(p_3_random)
    decoded = decode_vmap(encoded, 3)
    test_res = jnp.logical_and(test_res, jnp.all(decoded == p_3_random))

    test_res = jnp.logical_and(
        test_res, cglib.morton.encode(p_2_1) == p_2_1_encoded)
    test_res = jnp.logical_and(
        test_res, jnp.all(
            cglib.morton.decode(
                p_2_1_encoded, 2) == p_2_1))
    test_res = jnp.logical_and(
        test_res, cglib.morton.encode(p_2_2) == p_2_2_encoded)
    test_res = jnp.logical_and(
        test_res, jnp.all(
            cglib.morton.decode(
                p_2_2_encoded, 2) == p_2_2))
    test_res = jnp.logical_and(
        test_res, cglib.morton.encode(p_3_1) == p_3_1_encoded)
    test_res = jnp.logical_and(
        test_res, jnp.all(
            cglib.morton.decode(
                p_3_1_encoded, 3) == p_3_1))
    return test_res


def test_morton():
    assert jit(helper_test_morton)()
