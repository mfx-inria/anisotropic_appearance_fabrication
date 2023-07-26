"""
This module contains data types. The number of bits used by each data type
depends on the environment variable JAX_ENABLE_X64. Data types are provided for
NumPy and JAX arrays.
"""

import os

import jax.numpy as jnp
import numpy as np

x64 = os.environ.get('JAX_ENABLE_X64')
if x64 == '1' or x64 == 'True' or x64 == 'true':
    x64 = True
else:
    x64 = False

float_ = jnp.float32
int_ = jnp.int32
uint = jnp.uint32
complex_ = jnp.complex64

float_np = np.float32
int_np = np.int32
uint_np = np.uint32
complex_np = np.complex64

if x64:
    float_ = jnp.float64
    int_ = jnp.int64
    uint = jnp.uint64
    complex_ = jnp.complex128

    float_np = np.float64
    int_np = np.int64
    uint_np = np.uint64
    complex_np = np.complex128
