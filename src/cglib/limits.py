from cglib.type import float_, int_, uint
import jax.numpy as jnp

INT_MAX = int_(jnp.iinfo(int_).max)
UINT_MAX = uint(jnp.iinfo(uint).max)
FLOAT_MAX = float_(jnp.finfo(float_).max)