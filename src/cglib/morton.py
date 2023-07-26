# Jax version of morton code

# For 32 bits and 3 components, encoding then decoding works only with values
# less than 0x000000ff (255)

# For 32 bits and 2 components, encoding then decoding works with values less
# than 0x0000ffff (65535)

# For 64 bits and 3 components, encoding then decoding works only with values
# less than 0x000000000000ffff (65535)

# For 64 bits and 2 components, encoding then decoding works with values less
# than 0x00000000ffffffff (4294967295)

# Original code: pymorton (https://github.com/trevorprater/pymorton)
# Author: trevor.prater@gmail.com
# License: MIT

import jax.numpy as jnp

from cglib.type import x64


def __part1by1_32(n_in):
    # base10: 65535,      binary: 1111111111111111,                 len: 16
    n = n_in & 0x0000ffff
    # base10: 16711935,   binary: 111111110000000011111111,         len: 24
    n = (n | (n << 8)) & 0x00FF00FF
    # base10: 252645135,  binary: 1111000011110000111100001111,     len: 28
    n = (n | (n << 4)) & 0x0F0F0F0F
    # base10: 858993459,  binary: 110011001100110011001100110011,   len: 30
    n = (n | (n << 2)) & 0x33333333
    # base10: 1431655765, binary: 1010101010101010101010101010101,  len: 31
    n = (n | (n << 1)) & 0x55555555

    return n


def __part1by2_32(n_in):
    # base10: 1023,       binary: 1111111111,                       len: 10
    n = n_in & jnp.uint32(0x000003ff)
    # base10: 4278190335, binary: 11111111000000000000000011111111, len: 32
    n = (n ^ (n << 16)) & jnp.uint32(0xff0000ff)
    # base10: 50393103,   binary: 11000000001111000000001111,       len: 26
    n = (n ^ (n << 8)) & jnp.uint32(0x0300f00f)
    # base10: 51130563,   binary: 11000011000011000011000011,       len: 26
    n = (n ^ (n << 4)) & jnp.uint32(0x030c30c3)
    # base10: 153391689,  binary: 1001001001001001001001001001,     len: 28
    n = (n ^ (n << 2)) & jnp.uint32(0x09249249)

    return n


def __unpart1by1_32(n_in):
    # base10: 1431655765, binary: 1010101010101010101010101010101,  len: 31
    n = n_in & 0x55555555
    # base10: 858993459,  binary: 110011001100110011001100110011,   len: 30
    n = (n ^ (n >> 1)) & 0x33333333
    # base10: 252645135,  binary: 1111000011110000111100001111,     len: 28
    n = (n ^ (n >> 2)) & 0x0f0f0f0f
    # base10: 16711935,   binary: 111111110000000011111111,         len: 24
    n = (n ^ (n >> 4)) & 0x00ff00ff
    # base10: 65535,      binary: 1111111111111111,                 len: 16
    n = (n ^ (n >> 8)) & 0x0000ffff

    return n


def __unpart1by2_32(n_in):
    # base10: 153391689,  binary: 1001001001001001001001001001,     len: 28
    n = n_in & jnp.uint32(0x09249249)
    # base10: 51130563,   binary: 11000011000011000011000011,       len: 26
    n = (n ^ (n >> jnp.uint32(2))) & jnp.uint32(0x030c30c3)
    # base10: 50393103,   binary: 11000000001111000000001111,       len: 26
    n = (n ^ (n >> jnp.uint32(4))) & jnp.uint32(0x0300f00f)
    # base10: 4278190335, binary: 11111111000000000000000011111111, len: 32
    n = (n ^ (n >> jnp.uint32(8))) & jnp.uint32(0xff0000ff)
    # base10: 1023,       binary: 1111111111,                       len: 10
    n = (n ^ (n >> jnp.uint32(16))) & jnp.uint32(0x000003ff)

    return n


def __part1by1_64(n_in):
    # binary: 11111111111111111111111111111111,
    # len: 32
    n = n_in & jnp.uint64(0x00000000ffffffff)
    # binary: 1111111111111111000000001111111111111111,
    # len: 40
    n = (n | (n << jnp.uint64(16))) & jnp.uint64(0x0000ffff0000ffff)
    # binary: 11111111000000001111111100000000111111110000000011111111,
    # len: 56
    n = (n | (n << jnp.uint64(8))) & jnp.uint64(0x00ff00ff00ff00ff)
    # binary: 111100001111000011110000111100001111000011110000111100001111,
    # len: 60
    n = (n | (n << jnp.uint64(4))) & jnp.uint64(0x0F0F0F0F0F0F0F0F)
    # binary: 11001100110011001100110011001100110011001100110011001100110011,
    # len: 62
    n = (n | (n << jnp.uint64(2))) & jnp.uint64(0x3333333333333333)
    # binary: 101010101010101010101010101010101010101010101010101010101010101,
    # len: 63
    n = (n | (n << jnp.uint64(1))) & jnp.uint64(0x5555555555555555)

    return n


def __part1by2_64(n_in):
    # binary: 111111111111111111111,
    # len: 21
    n = n_in & jnp.uint64(0x1fffff)
    # binary: 11111000000000000000000000000000000001111111111111111,
    # len: 53
    n = (n | (n << jnp.uint64(32))) & jnp.uint64(0x1f00000000ffff)
    # binary: 11111000000000000000011111111000000000000000011111111,
    # len: 53
    n = (n | (n << jnp.uint64(16))) & jnp.uint64(0x1f0000ff0000ff)
    # binary: 1000000001111000000001111000000001111000000001111000000001111,
    # len: 61
    n = (n | (n << jnp.uint64(8))) & jnp.uint64(0x100f00f00f00f00f)
    # binary: 1000011000011000011000011000011000011000011000011000011000011,
    # len: 61
    n = (n | (n << jnp.uint64(4))) & jnp.uint64(0x10c30c30c30c30c3)
    # binary: 1001001001001001001001001001001001001001001001001001001001001,
    # len: 61
    n = (n | (n << jnp.uint64(2))) & jnp.uint64(0x1249249249249249)

    return n


def __unpart1by1_64(n_in):
    # binary: 101010101010101010101010101010101010101010101010101010101010101,
    # len: 63
    n = n_in & jnp.uint64(0x5555555555555555)
    # binary: 11001100110011001100110011001100110011001100110011001100110011,
    # len: 62
    n = (n ^ (n >> jnp.uint64(1))) & jnp.uint64(0x3333333333333333)
    # binary: 111100001111000011110000111100001111000011110000111100001111,
    # len: 60
    n = (n ^ (n >> jnp.uint64(2))) & jnp.uint64(0x0f0f0f0f0f0f0f0f)
    # binary: 11111111000000001111111100000000111111110000000011111111,
    # len: 56
    n = (n ^ (n >> jnp.uint64(4))) & jnp.uint64(0x00ff00ff00ff00ff)
    # binary: 1111111111111111000000001111111111111111,
    # len: 40
    n = (n ^ (n >> jnp.uint64(8))) & jnp.uint64(0x0000ffff0000ffff)
    # binary: 11111111111111111111111111111111,
    # len: 32
    n = (n ^ (n >> jnp.uint64(16))) & jnp.uint64(0x00000000ffffffff)
    return n


def __unpart1by2_64(n_in):
    # binary: 1001001001001001001001001001001001001001001001001001001001001,
    # len: 61
    n = n_in & jnp.uint64(0x1249249249249249)
    # binary: 1000011000011000011000011000011000011000011000011000011000011,
    # len: 61
    n = (n ^ (n >> jnp.uint64(2))) & jnp.uint64(0x10c30c30c30c30c3)
    # binary: 1000000001111000000001111000000001111000000001111000000001111,
    # len: 61
    n = (n ^ (n >> jnp.uint64(4))) & jnp.uint64(0x100f00f00f00f00f)
    # binary: 11111000000000000000011111111000000000000000011111111,
    # len: 53
    n = (n ^ (n >> jnp.uint64(8))) & jnp.uint64(0x1f0000ff0000ff)
    # binary: 11111000000000000000000000000000000001111111111111111,
    # len: 53
    n = (n ^ (n >> jnp.uint64(16))) & jnp.uint64(0x1f00000000ffff)
    # binary: 111111111111111111111,
    # len: 21
    n = (n ^ (n >> jnp.uint64(32))) & jnp.uint64(0x1fffff)
    return n


def encode2_32(x: jnp.ndarray) -> jnp.uint32:
    return __part1by1_32(x[0]) | (__part1by1_32(x[1]) << 1)


def encode2_64(x):
    return __part1by1_64(x[0]) | (__part1by1_64(x[1]) << 1)


def encode2(x):
    if x64:
        return encode2_64(x)
    else:
        return encode2_32(x)


def encode3_32(x):
    return __part1by2_32(
        x[0]) | (
        __part1by2_32(
            x[1]) << 1) | (
                __part1by2_32(
                    x[2]) << 2)


def encode3_64(x):
    return __part1by2_64(
        x[0]) | (
        __part1by2_64(
            x[1]) << jnp.uint64(1)) | (
                __part1by2_64(
                    x[2]) << jnp.uint64(2))


def encode3(x):
    if x64:
        return encode3_64(x)
    else:
        return encode3_32(x)


def encode(x):
    if x.shape[0] == 2:
        return encode2(x)
    else:
        return encode3(x)


def decode2_32(n):
    return jnp.array([__unpart1by1_32(n), __unpart1by1_32(n >> jnp.uint32(1))])


def decode2_64(n):
    return jnp.array([__unpart1by1_64(n), __unpart1by1_64(n >> jnp.uint32(1))])


def decode3_32(n):
    return jnp.array([__unpart1by2_32(n),
                      __unpart1by2_32(n >> 1),
                      __unpart1by2_32(n >> 2)])


def decode3_64(n):
    return jnp.array([__unpart1by2_64(n),
                      __unpart1by2_64(n >> 1),
                      __unpart1by2_64(n >> 2)])


def decode2(n):
    if x64:
        return decode2_64(n)
    else:
        return decode2_32(n)


def decode3(n):
    if x64:
        return decode3_64(n)
    else:
        return decode3_32(n)


def decode(n, dim):
    if dim == 2:
        return decode2(n)
    else:
        return decode3(n)
