"""
This module contains functions related to the [Gabor
filter](https://en.wikipedia.org/wiki/Gabor_filter) and how to evaluate them.
"""

import time

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct, jit, vmap
from jax._src.lib import xla_client

from . import type
from .multivariate_normal import eval_pdf, variance_from_radius
from .point_data import (GridPointData, PointData,
                         grid_neighborhood_from_point)
from .sine_wave import eval, eval_complex_angle, grid_shape_dtype, unpack_data


def eval_complex(
        x: jnp.ndarray,
        point_dir_phase: PointData,
        f: float,
        r: float) -> complex:
    """Evaluates the complex Gabor filter at a specified position.

    Parameters
    ----------
    x : ndarray
        The position of evaluation.
    point_dir_phase : PointData
        The point, direction, and phase of the Gabor filter. The direction is
        stored in `sine_wave.data[0:n]` in cartesian coordinates, where `n` is
        the dimension count of the point. The phase is stored in
        `sine_wave.data[n]`.
    f : float
        The frequency of the Gabor filter.
    r : float
        The radius of the Gabor filter.

    Returns
    -------
    complex
        The complex value of the Gabor filter at position `x`.
    """
    point, d, phi = unpack_data(point_dir_phase)

    variance = variance_from_radius(r)
    return eval_pdf(x, point, variance) * \
        eval_complex_angle(x, point, d, phi, f)


def eval_value_and_spatial_weight(
        x: jnp.ndarray,
        point_dir_phase: PointData,
        f: float,
        r: float) -> tuple[float, float]:
    """Evaluate the imaginary part of the Gabor filter at a specified position.

    Parameters
    ----------
    x : ndarray
        The position of evaluation.
    point_dir_phase : PointData
        The point, direction, and phase of the Gabor filter. The direction is
        stored in `sine_wave.data[0:n]` in cartesian coordinates, where `n` is
        the dimension count of the point. The phase is stored in
        `sine_wave.data[n]`.
    f : float
        The frequency of the Gabor filter.
    r : float
        The radius of the Gabor filter.

    Returns
    -------
    tuple[float, float]
        out1 :
            The real value, i.e., imaginary part, of the Gabor filter at
            position `x`.
        out2 :
            The Gaussian value of the Gabor filter at position `x`.
    """

    point, d, phi = unpack_data(point_dir_phase)

    variance = variance_from_radius(r)
    # The spatial weight is the gaussian value
    w_i = eval_pdf(x, point, variance)
    return w_i * eval(x, point, d, phi, f), w_i


def eval_array(
        x: jnp.ndarray,
        point_dir_phase: PointData,
        f: float,
        r: float) -> float:
    """Evaluates the given set of Gabor filters at a given position.

    Use Equation 13 in Chermain et al. 2023.

    Parameters
    ----------
    x : ndarray
        The position of evaluation.
    point_dir_phase : PointData
        The set of points, directions, and phases of the Gabor filters. The ith
        direction is stored in `sine_wave.data[i, 0:n]` in cartesian
        coordinates, where `n` is the dimension count of the point. The ith
        phase is stored in `sine_wave.data[i, n]`.
    f : float
        The frequency of all the Gabor filters.
    r : float
        The radius of all the Gabor filters.
    Returns
    -------
    float
        This functions returns the value of the given set of Gabor filters at a
        given position using Equation 13 in Chermain et al. 2023 where the
        neighborhood is extended to all the Gabor filters in the set.
    """

    gabor_values, spatial_weights = vmap(
        eval_value_and_spatial_weight, (None, 0, None, None))(
        x, point_dir_phase, f, r)

    mask = jnp.isnan(point_dir_phase.data[:, 0])
    gabor_values = jnp.where(mask, 0., gabor_values)
    spatial_weights = jnp.where(mask, 0., spatial_weights)

    gabor_values_sum = jnp.sum(gabor_values)
    spatial_weight_sum = jnp.sum(spatial_weights)

    spatial_weight_lt_zero = spatial_weight_sum <= 0.
    return jnp.where(
        spatial_weight_lt_zero,
        0.,
        gabor_values_sum /
        spatial_weight_sum)


def grid_eval(
        x: jnp.ndarray,
        f: float,
        grid_gabor_filter: GridPointData) -> float:
    """Evaluates a grid of Gabor filters.

    Parameters
    ----------
    x : ndarray
        The evaluation point.
    f : ndarray
        The frequency of all the Gabor filters.
    grid_gabor_filter : GridPointData
        The grid of Gabor filters, with one Gabor filter per cell. The
        direction and phase of the Gabor filter are stored in
        `grid_gabor_filter.point_data.data[0:n]` and
        `grid_gabor_filter.point_data.data[n]`, resp. The direction is in
        cartesian coordinates, and `n` is the dimension count of the point.
    Returns
    -------
    float
        The function returns the value of Equation 13 in Chermain et al. 2013.
    """
    neighboring_kernels = grid_neighborhood_from_point(
        x, grid_gabor_filter)
    gabor_field_val = eval_array(
        x, neighboring_kernels, f, grid_gabor_filter.grid.cell_sides_length)
    return gabor_field_val


def grid_compile_eval(
        evaluation_point_1dcount: int,
        grid_sine_side_cell_count: int,
        dim: int,
        device: xla_client.Device) -> tuple[jax.stages.Compiled, float]:
    """Compile a vectorized version of `grid_eval`.

    Parameters
    ----------
    evaluation_point_1dcount : int
        The number of evaluation points.
    grid_sine_side_cell_count : int
        The number of cells along the side of the square grid.
    dim : int
        The number of dimensions of the grid and the sine waves.
    device : Device
        The device the compiled function will run on.
        Available devices can be retrieved via `jax.devices()`.

    Returns
    -------
    tuple[Compiled, float]
        out1
            The function returns `grid_eval` compiled for the specified
            parameters.
        out2
            The duration of the compilation (seconds).
    """

    grid_sine_shape_dtype = grid_shape_dtype(grid_sine_side_cell_count, dim)
    evaluation_point_shape_dtype = ShapeDtypeStruct(
        (evaluation_point_1dcount, 2), type.float_)

    start = time.perf_counter()
    # Vectorization of grid_eval over evaluation point
    grid_eval_v = vmap(grid_eval, (0, None, None))

    grid_eval_v_jitted = jit(grid_eval_v, device=device)
    grid_eval_v_lowered = grid_eval_v_jitted.lower(
        evaluation_point_shape_dtype,
        ShapeDtypeStruct((), type.float_),
        grid_sine_shape_dtype)
    grid_eval_v_compiled = grid_eval_v_lowered.compile()
    stop = time.perf_counter()

    return grid_eval_v_compiled, stop - start
