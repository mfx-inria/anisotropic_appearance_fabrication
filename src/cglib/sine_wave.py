"""
This module contains functions related to the [sine
wave](https://en.wikipedia.org/wiki/Sine_wave) and how to align them.
"""

import math as stdmath
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct, jit, lax, vmap
from jax._src.lib import xla_client

from . import grid, line, math, multivariate_normal, tree_util, type
from .grid import cell, multigrid
from .point_data import (GridPointData, PointData, grid_get,
                         grid_neighborhood_from_point)


def unpack_data(
        sine_wave: PointData) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Unpack a sine wave represented by a point with data.

    Parameters
    ----------
    sine_wave : PointData
        A sine wave is a point plus data: the direction and the phase. Here the
        frequency is excluded. The direction is stored in `sine_wave.data[0:n]`
        in cartesian coordinates, where `n` is the dimension count of the
        point. The phase is stored in `sine_wave.data[n]`.
    Returns
    -------
    tuple[ndarray, ndarray, float]
        A tuple with the point, the frequency, and the phase of the sine wave.
    """
    point = sine_wave.point
    n = point.shape[0]
    d = sine_wave.data[0:n]
    phi = sine_wave.data[n]
    return point, d, phi


def change_phase_to_align_i_with_j_1d(
        p_i: float,
        phase_j: float,
        f: float,
        inv_d: bool) -> float:
    """Return the sine wave phase giving an alignment with another sine wave.

    This function computes the phase of the 1D sine wave i leading to an
    alignment with the 1D sine wave j. In 1D, to be aligned means to be
    equaled.

    Parameters
    ----------
    p_i : float
        The 1D position of the sine `i` in `j` space.
    phase_j : float
        The phase of the sine wave `j`, which is the guide.
    f : float
        The frequency of the two sine waves.
    inv_d : bool
        True if the directions of the sine waves are in opposite directions.

    Returns
    -------
    float
        The phase of the sine wave `i`, leading to alignment. The value is
        defined in Equation 8 in Chermain et al. 2023, where the scalar
        projection is `p_i`.

    Notes
    -----
    This function modifies the phase of i to minimize
    `phase_alignment_energy_i_with_j_1d`.
    """
    distance = 2. * jnp.pi * f * p_i + phase_j
    return jnp.where(inv_d, -distance + jnp.pi, distance)


def change_phase_to_align_i_with_j_nd(
        sine_j: PointData,
        p_i: jnp.ndarray,
        d_i: jnp.ndarray,
        f: float) -> float:
    """Return the phase of the ND sine wave giving a alignment with another.

    This function computes the phase of the ND sine wave i leading to an
    alignment with the ND sine wave j. Here, to align means to find the phase
    of $i$ giving the minimum value of the measure of the difference between
    the two sine waves. See Equation 8 in Chermain et al. 2023.

    Parameters
    ----------
    sine_j : PointData
        The sine wave j is a point plus data: the direction and the phase. The
        direction is stored in `sine_j.data[0:n]` in cartesian coordinates,
        where `n` is the dimension count of the point. The phase is stored in
        `sine_j.data[n]`.
    p_i : float
        The ND position of the sine `i`.
    d_i : ndarray
        The ND direction of the sine `i`, in cartesian coordinates.
    f : float
        The frequency of the two sine waves.

    Returns
    -------
    float
        The phase of the sine wave `i` leading to an alignment.
    """
    sine_j_p, sine_j_d, sine_j_phi = unpack_data(sine_j)
    inv_d = jnp.where(jnp.vdot(sine_j_d, d_i) < 0., True, False)
    p_i_proj = jnp.vdot(p_i - sine_j_p, sine_j_d)
    phi_i = change_phase_to_align_i_with_j_1d(p_i_proj, sine_j_phi, f, inv_d)
    return phi_i


def change_phase_to_align_with_others(
        sine: PointData,
        other_sines: PointData,
        weights: jnp.ndarray,
        f: float) -> PointData:
    """Align a sine wave with other sines, by only changing its phase.

    Parameters
    ----------
    sine : PointData
        The sine to align. `sine.point.shape` is (n,), where n is the point's
        component count. `sine.data.shape` must be `(>=n+1)`. `sine.data[:n]`
        must be the sine direction. `sine.data[n]` must be the phase.
        `sine.data[n+1:]` can be any data and will be returned.
    other_sines : PointData
        The other sines are used for the alignment. `other_sines.point.shape`
        is (m, n), where `m` is the number of other sines.
        `other_sines.data.shape` must be `(m, >=n+1)`. `other_sines.data[:,
        :n]` must be the sines' directions. `other_sines.data[:, n]` must be
        the phases. `other_sines.data[:, n+1:]` can be any data and will not be
        used.
    weights : jnp.ndarray
        The weight for each other sine. `weights.shape` is (m,). At least one
        weight has to be non-zero to avoid division by zero.
    f : float
        Frequency of all the sines.
    Returns
    -------
    PointData
        The aligned sine with other sines.
    """
    n = sine.point.shape[0]
    sine_d = sine.data[:n]
    other_sines_d = other_sines.data[:, :n]
    aligned_phases = vmap(
        change_phase_to_align_i_with_j_nd, (0, None, None, None))(
        other_sines, sine.point, sine_d, f)
    phase_weights = jnp.abs(vmap(jnp.vdot, (0, None))(other_sines_d, sine_d))

    aligned_phases = math.angle_weighted_average(
        aligned_phases, weights * phase_weights)
    sine_averaged_data = jnp.append(sine_d, aligned_phases)
    if sine.data.shape[0] > n + 1:
        sine_averaged_data = jnp.append(sine_averaged_data, sine.data[n + 1:])
    sine_aligned_phases_averaged = PointData(sine.point, sine_averaged_data)
    return sine_aligned_phases_averaged


def prolong_j_into_i(
        sine_j: PointData,
        sine_i: PointData,
        f: float) -> PointData:
    """Prolong the sine wave j into i.

    This function prolongs/interpolates the sine j computed on a coarser grid
    into the sine i belonging to a finer grid. The function supposes the sine i
    and j to have the same shapes.

    Parameters
    ----------
    sine_j, sine_i : PointData
        The sine prolonged and the sine to update. `sine.point.shape` is
        `(n,)`, where n is the point component's count. `sine.data.shape` must
        be `(>=n+2,)`. `sine.data[:,:n]` must be the sine direction.
        `sine.data[:, n]` must be the phase. `grid_sine.data[:, n+1]` must be
        the constraint: 0. -> no constraints; 1. -> line constraint; 2. ->
        phase constraint; 3. -> line and phase constraints.
    f : float
        The frequency of the two sines.

    Return
    ------
    PointData
        The result of the prolongation/interpolation.
    """
    n = sine_i.point.shape[0]
    sine_default = PointData(
        jnp.zeros_like(
            sine_j.point), jnp.zeros_like(
            sine_j.data))
    sine_masked = PointData(
        jnp.full_like(
            sine_j.point, jnp.nan), jnp.full_like(
            sine_j.data, jnp.nan))
    sine_i_mask = jnp.isnan(sine_i.data[0])
    sine_j_mask = jnp.isnan(sine_j.data[0])
    sine_i = lax.cond(
        sine_i_mask,
        lambda x: x[0],
        lambda x: x[1],
        (sine_default,
         sine_i))
    sine_j = lax.cond(
        sine_j_mask,
        lambda x: x[0],
        lambda x: x[1],
        (sine_default,
         sine_j))

    sine_i_constraint = sine_i.data[n + 1]
    sine_i_line_constraint = jnp.equal(sine_i_constraint, 1.)
    sine_i_line_constraint = jnp.logical_or(
        sine_i_line_constraint, jnp.equal(
            sine_i_constraint, 3.))
    sine_i_phase_constraint = jnp.equal(sine_i_constraint, 2.)
    sine_i_phase_constraint = jnp.logical_or(
        sine_i_phase_constraint, jnp.equal(
            sine_i_constraint, 3.))

    # The sine wave keeps its position
    # but it takes the line and the phase of the other sine
    sine_i_line = jnp.where(sine_i_line_constraint,
                            sine_i.data[:n], sine_j.data[:n])
    sine_i_phase = change_phase_to_align_i_with_j_nd(
        sine_j, sine_i.point, sine_i_line, f)
    sine_i_phase = jnp.where(
        sine_i_phase_constraint,
        sine_i.data[n],
        sine_i_phase)
    sine_i_data = jnp.append(sine_i_line, sine_i_phase)
    sine_i_data = jnp.append(sine_i_data, sine_i.data[n + 1:])
    sine_i_updated = PointData(sine_i.point, sine_i_data)
    sine_i_updated = lax.cond(
        sine_j_mask,
        lambda x: x[0],
        lambda x: x[1],
        (sine_i,
         sine_i_updated))
    sine_i_updated = lax.cond(
        sine_i_mask,
        lambda x: x[0],
        lambda x: x[1],
        (sine_masked,
         sine_i_updated))
    return sine_i_updated


def phase_alignment_energy_i_with_j_1d(
        p_i: float,
        phase_i: float,
        phase_j: float,
        f: float,
        inv_d: bool) -> float:
    """Return the alignment energy between two 1D sine waves.

    This function computes the alignment energy of the 1D sine wave i with the
    wave j. The alignment energy is defined in Chermain et al. 2023, Equation
    7. The integration is not numeric, and the closed-form expression is used
    here. The result is scaled to belong in the interval [0, 1].

    Parameters
    ----------
    p_i : float
        The 1D position of the sine `i` in `j` space.
    phase_i : float
        The phase of the sine wave `i`.
    phase_j : float
        The phase of the sine wave `j`.
    f : float
        The frequency of the two sine waves.
    inv_d : bool
        True if the directions of the 1D sine waves are in opposite
        directions.

    Returns
    -------
    float
        The alignment energy defined in Equation 7 in Chermain et al. 2023.
    """
    cos_arg = phase_j + 2 * jnp.pi * f * p_i
    energy = jnp.where(
        inv_d,
        1. + jnp.cos(cos_arg + phase_i),
        1. - jnp.cos(cos_arg - phase_i))
    # Should be energy / f
    # But I prefer normalized energy, i.e., energy between 0 and 1.
    return energy * 0.5


def phase_alignment_energy_i_with_j_nd(
        sine_i: PointData,
        sine_j: PointData,
        f: float) -> float:
    """Return the alignment energy between two ND sine waves.

    This function computes the alignment energy of the ND sine wave i with the
    ND sine wave j. In other words, the function returns the measure of the
    difference between the two sine waves. See Equation 7 in Chermain et al.
    2023.

    Parameters
    ----------
    sine_i and sine_j : PointData
        The sine wave is a point plus data: the direction and the phase. The
        direction is stored in `sine.data[0:n]` in cartesian coordinates, where
        `n` is the dimension count of the point. The phase is stored in
        `sine.data[n]`.
    f : float
        The frequency of the two sine waves.

    Returns
    -------
    float
        The normalized alignment energy.

    Notes
    -----
    This function modifies the phase of i to minimize
    `phase_alignment_energy_i_with_j_nd`.
    """
    p_i, d_i, phi_i = unpack_data(sine_i)
    p_j, d_j, phi_j = unpack_data(sine_j)
    inv_d = jnp.where(jnp.vdot(d_j, d_i) < 0., True, False)
    p_i_proj = jnp.vdot(p_i - p_j, d_j)
    alignment_energy = phase_alignment_energy_i_with_j_1d(
        p_i_proj, phi_i, phi_j, f, inv_d)
    return alignment_energy


def phase_alignment_energy_with_others(
        sine: PointData,
        other_sines: PointData,
        weights: jnp.ndarray,
        f: float) -> float:
    """Returns the aligment energy of the sine wave i with other sines.

    Parameters
    ----------
    sine : PointData
        The sine whose alignment energy is computed. `sine.point.shape` is
        (n,), where n is the point's component count. `sine.data.shape` must be
        `(>=n+1)`. `sine.data[:n]` must be the sine direction. `sine.data[n]`
        must be the phase. `sine.data[n+1:]` can be other data and will not be
        used.
    other_sines : PointData
        The other sines are used for computing the alignment energy.
        `other_sines.point.shape` is (m, n), where `m` is the number of other
        sines. `other_sines.data.shape` must be `(m, >=n+1)`.
        `other_sines.data[:, :n]` must be the sines' directions.
        `other_sines.data[:, n]` must be the phases. `other_sines.data[:,
        n+1:]` can be other data and will not be used.
    weights : jnp.ndarray
        The weight for each other sine. `weights.shape` is (m,). At least one
        weight has to be non-zero to avoid division by zero.
    f : float
        Frequency of all the sines.

    Returns
    -------
    float
        The normalized alignment energy, i.e., the alignment energy rescaled
        between [0, 1].
    """
    n = sine.point.shape[0]
    sine_d = sine.data[:n]
    other_sines_d = other_sines.data[:, :n]
    alignment_energy = vmap(
        phase_alignment_energy_i_with_j_nd, (None, 0, None))(
        sine, other_sines, f)
    alignment_energy_weight = jnp.abs(
        vmap(
            jnp.vdot, (0, None))(
            other_sines_d, sine_d))
    alignment_energy_weight *= weights
    alignment_energy_weighted = alignment_energy * alignment_energy_weight
    return jnp.sum(alignment_energy_weighted) / \
        jnp.sum(alignment_energy_weight)


def average(
        indices: jnp.ndarray,
        sine_waves: PointData,
        f: float) -> PointData:
    """Average sine waves.

    Parameters
    ----------
    indices :
        The average is performed on the sine waves indicated by the indices.
    sine_waves : PointData
        The sine waves are represented as a list of points with data. The
        indices point to elements in this list. `sine_waves.point.shape` is
        `(m, n)`, where `m` is the number of points and where `n` is the points
        component's count. `sine_waves.data.shape` must be `(m, >=n+2)`.
        `sine_waves.data[:,:n]` must be the sine directions in cartesian
        coordinates. `sine_waves.data[:, n]` must be the phases.
        `sine_waves.data[:, n+1]` must be the constraints.
    f : float
        The frequency of all the sines.
    Return
    ------
    PointData
        The averaged sine wave.
    """
    # Get sines and put non nan values if needed
    sine_waves: PointData = tree_util.leaves_at_indices(
        indices, sine_waves)
    sines_default = PointData(
        jnp.zeros_like(
            sine_waves.point), jnp.zeros_like(
            sine_waves.data))
    sines_mask = jnp.isnan(sine_waves.data[:, 0])
    sines_points: PointData = jnp.where(
        sines_mask.reshape(
            (-1, 1)), sines_default.point, sine_waves.point)
    sines_data: PointData = jnp.where(
        sines_mask.reshape(
            (-1, 1)), sines_default.data, sine_waves.data)
    sine_waves = PointData(sines_points, sines_data)
    # Unpack data
    n = sine_waves.point.shape[1]
    sines_lines = sine_waves.data[:, :n]
    sines_constraints = sine_waves.data[:, n + 1]
    sines_lines_constraints = jnp.equal(sines_constraints, 1.)
    sines_lines_constraints = jnp.logical_or(
        sines_lines_constraints, jnp.equal(
            sines_constraints, 3.))
    sines_phases_constraints = jnp.equal(sines_constraints, 2.)
    sines_phases_constraints = jnp.logical_or(
        sines_phases_constraints, jnp.equal(
            sines_constraints, 3.))

    # Put weigths to zero when the point is masked
    sines_weights = jnp.logical_not(sines_mask)
    sines_weights_gt_zero = sines_weights > 0.
    sines_weights_sum = jnp.sum(sines_weights).astype(type.float_)

    # The averaged sine wave is constrained with respect to the line (or the
    # phase) if any of the sines has a non masked line (or phase) constraint.
    sines_lines_constraints_average = jnp.any(
        jnp.logical_and(
            sines_weights_gt_zero,
            sines_lines_constraints))
    sines_phases_constraints_average = jnp.any(
        jnp.logical_and(
            sines_weights_gt_zero,
            sines_phases_constraints))

    # Compute the weights for the line averaging
    sines_lines_average_mask = jnp.where(
        sines_lines_constraints_average,
        jnp.logical_not(sines_lines_constraints),
        jnp.full_like(sines_lines_constraints, False))
    sines_lines_average_weight = jnp.where(sines_lines_average_mask, 0., 1.)

    # Compute the weights for the phase averaging
    sines_phases_average_mask = jnp.where(
        sines_phases_constraints_average,
        jnp.logical_not(sines_phases_constraints),
        jnp.full_like(sines_phases_constraints, False))
    sines_phases_average_weight = jnp.where(sines_phases_average_mask, 0., 1.)

    # Average the sines points
    sines_points_weighted = sine_waves.point * sines_weights.reshape(-1, 1)
    sines_points_average = jnp.sum(
        sines_points_weighted,
        axis=0) / sines_weights_sum
    sines_points_average = jnp.where(
        sines_weights_sum <= 0.,
        jnp.zeros_like(sines_points_average),
        sines_points_average)

    # Average the sines lines
    sines_lines_average = line.average_with_weights(
        sines_lines, sines_lines_average_weight * sines_weights)

    # Evaluate the phases (equivalent to align) at the sines points average
    phases_aligned = vmap(
        change_phase_to_align_i_with_j_nd, (0, None, None, None))(
        sine_waves, sines_points_average, sines_lines_average, f)
    # Update the phases weights with the lines' average
    sines_phases_average_weight *= jnp.abs(
        vmap(
            jnp.vdot, (0, None))(
            sines_lines, sines_lines_average))
    # Average the phases defined at the sines points average
    phases_aligned_averaged = math.angle_weighted_average(
        phases_aligned, sines_weights * sines_phases_average_weight)

    # "Average" the sines' masks
    sines_mask_average = jnp.where(sines_weights_sum > 0., 0., 1.)

    # "Average" the sines' constraints
    sines_constraints_average = 0.
    sines_constraints_average = jnp.where(
        sines_lines_constraints_average, 1., sines_constraints_average)
    sines_constraints_average = jnp.where(
        sines_phases_constraints_average, 2., sines_constraints_average)
    sines_constraints_average = jnp.where(
        jnp.logical_and(
            sines_lines_constraints_average,
            sines_phases_constraints_average),
        3.,
        sines_constraints_average)

    sines_average_data = jnp.append(
        sines_lines_average,
        phases_aligned_averaged)
    sines_average_data = jnp.append(
        sines_average_data,
        sines_constraints_average)
    sines_average = PointData(sines_points_average, sines_average_data)
    sines_average_masked = PointData(
        jnp.full_like(
            sines_points_average, jnp.nan), jnp.full_like(
            sines_average_data, jnp.nan))
    sines_average = lax.cond(
        sines_mask_average,
        lambda x: x[0],
        lambda x: x[1],
        (sines_average_masked,
         sines_average))
    return sines_average


def grid_align_with_neighbors(
        i: int,
        grid_sine: GridPointData,
        f: float,
        use_spatial_weights: bool) -> PointData:
    """Align the ith sine wave in a grid with its neighbors.

    Parameters
    ----------
    i : int
        Index of the sine wave to align with its neighbors.
    grid_sine : GridPointData
        The grid of sine waves. `grid_sine.point_data.point.shape` is `(m, n)`,
        where `m` is the number of points and where n is the points component's
        count. `grid_sine.point_data.data.shape` must be `(m, n+2)`.
        `grid_sine.point_data.data[:,:n]` must be the sine directions.
        `grid_sine.point_data.data[:, n]` must be the phases.
        `grid_sine.data[:, n+1]` must be the constraints: 0. -> no constraints;
        1. -> line constraint; 2. -> phase constraint; 3. -> line and phase
        constraints.
    f : float
        Frequency of all the sines.
    use_spatial_weights : bool
        If `True`, neighbors on the diagonals are less influential than closer
        neighbors. Otherwise, each neighbor has the same weight.

    Returns
    -------
    PointData
        Return the ith sine wave with the neighboring phase average if it is
        not constrained and if there is at least one nonmasked neighbor. In
        other cases, the function returns the unchanged ith sine wave. A masked
        sine returns `True` when calling `jnp.isnan(sine.data[:, 0])`.

    Notes
    -----
    This function modifies the direction and the phase of i to reduce the
    energy of `grid_alignment_energy_with_neighbors`.
    """
    # Unpack data
    sine: PointData = tree_util.leaves_at_indices(
        i, grid_sine.point_data)
    sine_default = PointData(
        jnp.zeros_like(
            sine.point), jnp.zeros_like(
            sine.data))
    sine_masked = PointData(
        jnp.full_like(
            sine.point, jnp.nan), jnp.full_like(
            sine.data, jnp.nan))
    sine_mask = jnp.isnan(sine.data[0])
    sine = lax.cond(
        sine_mask,
        lambda x: x[0],
        lambda x: x[1],
        (sine_default,
         sine))

    n = sine.point.shape[0]
    sine_is_constrained = sine.data[n + 1]
    line_constraint = jnp.logical_or(
        jnp.equal(
            sine_is_constrained, 1.), jnp.equal(
            sine_is_constrained, 3.))
    phase_constraint = jnp.logical_or(
        jnp.equal(
            sine_is_constrained, 2.), jnp.equal(
            sine_is_constrained, 3.))

    # Get the neighboring sines
    neighbors = grid_neighborhood_from_point(
        sine.point, grid_sine, True)
    neighbors_default = PointData(
        jnp.zeros_like(
            neighbors.point), jnp.zeros_like(
            neighbors.data))
    neighboring_mask = jnp.logical_or(
        jnp.isnan(neighbors.data[:, 0]), sine_mask)
    neighbors_points: PointData = jnp.where(
        neighboring_mask.reshape(
            (-1, 1)), neighbors_default.point, neighbors.point)
    neighbors_data: PointData = jnp.where(
        neighboring_mask.reshape(
            (-1, 1)), neighbors_default.data, neighbors.data)
    neighbors = PointData(neighbors_points, neighbors_data)

    weights = jnp.logical_not(neighboring_mask).astype(type.float_)

    neighbors_all_masked = jnp.all(neighboring_mask)

    # To avoid spatial weights near zero, the variance of the gaussian is
    # calculated with the cell sides length times two.
    neighboring_kernel_variance = multivariate_normal.variance_from_radius(
        grid_sine.grid.cell_sides_length * 2.)
    spatial_weights = vmap(
        multivariate_normal.eval_pdf,
        (0, None, None))(
        neighbors.point,
        sine.point,
        neighboring_kernel_variance)
    constant_weights = jnp.ones_like(weights)
    spatial_weights = jnp.where(
        use_spatial_weights,
        spatial_weights,
        constant_weights)
    weights *= spatial_weights

    line_average = line.average_with_weights(
        neighbors.data[:, :n], weights)
    keep_line = jnp.logical_or(neighbors_all_masked, sine_mask)
    keep_line = jnp.logical_or(keep_line, line_constraint)
    line_average = jnp.where(keep_line, sine.data[:n], line_average)
    sine = PointData(sine.point, jnp.append(line_average, sine.data[n:]))

    sine_aligned = change_phase_to_align_with_others(
        sine, neighbors, weights, f)
    keep_sine = jnp.logical_or(neighbors_all_masked, sine_mask)
    keep_sine = jnp.logical_or(keep_sine, phase_constraint)
    sine_aligned = lax.cond(
        keep_sine,
        lambda x: x[0],
        lambda x: x[1],
        (sine,
         sine_aligned))

    return lax.cond(
        sine_mask,
        lambda x: x[0],
        lambda x: x[1],
        (sine_masked,
         sine_aligned))


def grid_align(
        grid_sine: GridPointData,
        f: float,
        use_spatial_weights: bool) -> GridPointData:
    """Align the sine waves in a grid.

    This function aligns the sine waves in a grid. More precisely, this
    function is a vectorized version of `grid_sine_wave_align_with_neighbors`,
    where the iteration goes through all the sine waves of the grid. This
    function is used several times in one level of a grid hierarchy to align
    sine waves efficiently.

    Parameters
    ----------
    grid_sine : GridPointData
        The grid of sine waves. `grid_sine.point_data.point.shape` is `(m, n)`,
        where `m` is the number of points and where n is the points component's
        count. `grid_sine.point_data.data.shape` must be `(m, n+2)`.
        `grid_sine.point_data.data[:,:n]` must be the sine directions.
        `grid_sine.point_data.data[:, n]` must be the phases.
        `grid_sine.data[:, n+1]` must be the constraints: 0. -> no constraints;
        1. -> line constraint; 2. -> phase constraint; 3. -> line and phase
        constraints.
    f : float
        The frequency of all the sine waves.
    use_spatial_weights : bool
        If `True`, neighbors on the diagonals are less influential than closer
        neighbors. Otherwise, each neighbor has the same weight.

    Returns
    -------
    GridPointData
        The given grid is returned with directions and phases modified to align
        the sine waves better.
    """

    sine_count = grid_sine.point_data.point.shape[0]
    sine_indices = jnp.arange(sine_count)
    sines_aligned = vmap(
        grid_align_with_neighbors, (0, None, None, None))(
        sine_indices, grid_sine, f, use_spatial_weights)
    return GridPointData(
        sines_aligned,
        grid_sine.grid)


def grid_align_n_times(
        grid_sines: GridPointData,
        f: float,
        use_spatial_weights: bool,
        n: int) -> GridPointData:
    """Perform several alignment iteration to a grid of sine waves.

    This function aligns the sine waves in a grid with n iterations. This
    function is `grid_sine_wave_align` within a loop. This function is used
    once in each level of a grid hierarchy to align sine waves efficiently.

    Parameters
    ----------
    grid_sine : GridPointData
        The grid of sine waves. `grid_sine.point_data.point.shape` is `(m, n)`,
        where `m` is the number of points and where n is the points component's
        count. `grid_sine.point_data.data.shape` must be `(m, n+2)`.
        `grid_sine.point_data.data[:,:n]` must be the sine directions.
        `grid_sine.point_data.data[:, n]` must be the phases.
        `grid_sine.data[:, n+1]` must be the constraints: 0. -> no constraints;
        1. -> line constraint; 2. -> phase constraint; 3. -> line and phase
        constraints.
    f : float
        The frequency of all the sine waves.
    use_spatial_weights : bool
        If `True`, neighbors on the diagonals are less influential than closer
        neighbors. Otherwise, each neighbor has the same weight.
    n : int
        The number of iterations.

    Returns
    -------
    GridPointData
        The given grid is returned with directions and phases modified to align
        the sine waves better.
    """

    def _grid_sine_wave_align(i: int, params):
        grid_sines = params[0]
        f = params[1]
        use_spatial_weight = params[2]
        return (
            grid_align(
                grid_sines,
                f,
                use_spatial_weight),
            f,
            use_spatial_weight)

    kernel_grid_aligned = lax.fori_loop(
        0,
        n,
        _grid_sine_wave_align,
        (grid_sines,
         f,
         use_spatial_weights))[0]
    return kernel_grid_aligned


def grid_prolong_cell(
        grid_level_N_cell_ndindex: jnp.ndarray,
        grid_level_N_sine_wave: GridPointData,
        grid_level_Nm1_sine_wave: GridPointData,
        f: float) -> PointData:
    """Prolong a coarse sine wave into finer sine waves.

    This function prolongs/interpolates the sine wave pointed by a cell index
    at level N (coarse grid) into the corresponding sine waves at level N - 1
    (fine grid).

    Parameters
    ----------
    grid_level_N_cell_ndindex : jnp.ndarray
        The cell ND index points the cell to prolong. The ND index is prolonged
        to indices pointing to sine waves to update.
    grid_level_N_sine_wave : GridPointData
        The sine waves in the coarse grid at level N.
    grid_level_Nm1_sine_wave : GridPointData
        The sine waves in the fine grid at level N - 1. The number of sine
        waves at level N - 1 equals the number of sine waves at level N divided
        by four.
    f : float
        The frequency of all the sine waves.

    Returns
    -------
    PointData
        The prolonged sine waves.
    """
    sine_to_propagate = grid_get(
        grid_level_N_cell_ndindex, grid_level_N_sine_wave)
    detailed_grid_cell_ndindices = multigrid.ndindex_prolong(
        grid_level_N_cell_ndindex)
    sines_to_update = vmap(
        grid_get, (0, None))(
        detailed_grid_cell_ndindices, grid_level_Nm1_sine_wave)
    sines_to_update = vmap(
        prolong_j_into_i, (None, 0, None))(
        sine_to_propagate, sines_to_update, f)
    return sines_to_update


def grid_prolong(
        grid_level_N_side_cell_count: int,
        grid_level_N: GridPointData,
        grid_level_Nm1: GridPointData,
        f: float) -> PointData:
    """Prolong a coarse grid of sine waves into a fine grid.

    This function prolongs/interpolates the coarse grid of sine waves at level
    N into the fine grid of sine waves at level N - 1.

    Parameters
    ----------
    grid_level_N_side_cell_count : int
        The number of cells along the axes of the coarse grid at level N.
    grid_level_N : GridPointData
        The sine waves of the coarse grid at level N to prolong.
    grid_level_Nm1 : GridPointData
        The sine waves of the fine grid at level N - 1 that are updated with
        the sine waves of the coarse grid.
    f : float
        The frequency of all the sines.

    Returns
    -------
    PointData
        The prolonged sine waves.
    """

    # For each cell in the coarse grid, propagate the values of its sines to
    # the associated fine cells
    with jax.ensure_compile_time_eval():
        n = grid_level_N.point_data.point.shape[1]
        coarse_grid_cell_1dcount = grid_level_N_side_cell_count**n
        coarse_grid_cell_1dindices = jnp.arange(coarse_grid_cell_1dcount)
        detailed_grid_cell_1dcount = (grid_level_N_side_cell_count * 2)**n
        detailed_grid_cell_1dindices = jnp.arange(detailed_grid_cell_1dcount)
    coarse_grid_cell_ndindices = vmap(
        cell.ndindex_from_1dindex, (0, None))(
        coarse_grid_cell_1dindices, grid_level_N.grid.cell_ndcount)
    detailed_grid_cell_ndindices = vmap(
        cell.ndindex_from_1dindex, (0, None))(
        detailed_grid_cell_1dindices, grid_level_Nm1.grid.cell_ndcount)
    detailed_grid_sines_updated_z_ordering = vmap(
        grid_prolong_cell, (0, None, None, None))(
        coarse_grid_cell_ndindices, grid_level_N, grid_level_Nm1, f)
    coarse_grid_cell_ndindices_z_ordering = jnp.array(
        multigrid.ndindex_restrict(detailed_grid_cell_ndindices))
    coarse_grid_cell_1dindices_z_ordering = vmap(
        vmap(
            cell.index1_from_ndindex, (0, None)), (0, 0))(
        coarse_grid_cell_ndindices_z_ordering, jnp.array(
            [
                grid_level_N.grid.cell_ndcount, jnp.full_like(
                    grid_level_N.grid.cell_ndcount, 2)]))
    detailed_grid_sines_updated_point = \
        detailed_grid_sines_updated_z_ordering.point[
            coarse_grid_cell_1dindices_z_ordering[0],
            coarse_grid_cell_1dindices_z_ordering[1]]
    detailed_grid_sines_updated_data = \
        detailed_grid_sines_updated_z_ordering.data[
            coarse_grid_cell_1dindices_z_ordering[0],
            coarse_grid_cell_1dindices_z_ordering[1]]
    return PointData(detailed_grid_sines_updated_point,
                     detailed_grid_sines_updated_data)


def grid_restrict_from_cell_ndindex(
        grid_level_Np1_cell_ndindex: jnp.ndarray,
        grid_level_N_sine_waves: GridPointData,
        f: float) -> PointData:
    """Restrict four sine waves associated to cells of a fine grid into one.

    This function restricts four cells of a fine grid to one cell of a coarse
    grid. We consider the fine grid at level N and the coarse grid at level N +
    1. Their resolutions are a power of two, and the resolution at level N is
    two times the resolution of the grid at level N + 1. The number of distinct
    cells in each dimension (x and y) is considered equal. To restrict several
    sine waves means to average them following the function `sine_wave_average`
    rules.

    Parameters
    ----------
    grid_level_Np1_cell_ndindex : ndarray
        The cell ND index of the coarse grid (at level N + 1). The given ND
        index at level N + 1 is prolonged to four adjacent ND indices at level
        N. The sine waves indexed by the prolonged ND indices are
        restricted/averaged.
    grid_level_N_sine_waves : ndarray
        The sine waves of the fine grid (at level N).
    f : float
        Frequency of all the sines.

    Returns
    -------
    PointData
        The restricted sine is obtained following the rules described and
        illustrated in Section 5.3, Chermain et al. 2023.
    """

    # Evaluate cell ndindices associated to the detailed grid
    detailed_grid_cell_ndindices = multigrid.ndindex_prolong(
        grid_level_Np1_cell_ndindex)
    # Map to 1d indices
    detailed_grid_cell_1dindices = vmap(
        cell.index1_from_ndindex, (0, None))(
        detailed_grid_cell_ndindices,
        grid_level_N_sine_waves.grid.cell_ndcount)

    sine_average = average(
        detailed_grid_cell_1dindices,
        grid_level_N_sine_waves.point_data,
        f)

    return sine_average


def grid_restrict(
        grid_level_N_side_cell_count: int,
        grid_level_N_sine_waves: GridPointData,
        f: float) -> GridPointData:
    """Restrict a square grid of sine waves to a coarser grid.

    Downsize/restrict a square grid of sine waves. The resolution is divided by
    two for each axis. The downsized grid is considered at level N + 1, and the
    given grid is at level N.

    Parameters
    ----------

    grid_level_N_side_cell_count : int
        The side cell count of the grid at level N. The number of distinct
        cells in each dimension (x and y) is considered equal. Must be a power
        of two.
    grid_level_N_sine_waves : GridPointData
        The sine waves of the fine grid (at level N).
    f : float
        The frequency of the sines.

    Returns
    -------
    GridPointData
        The restricted grid of sine waves.
    """
    with jax.ensure_compile_time_eval():
        n = grid_level_N_sine_waves.point_data.point.shape[1]
        # We assume power of two cell side count and equal cell side count for
        # each axis. The downside grid will have the same cell count as the
        # original grid divided by two.
        downsized_grid_side_cell_count = grid_level_N_side_cell_count // 2
        downsized_grid_cell_1dcount = downsized_grid_side_cell_count**n
        downsized_grid_cell_ndcount = jnp.full(
            (n,), downsized_grid_side_cell_count)
        downsized_grid_cell_1dindices = jnp.arange(downsized_grid_cell_1dcount)

    # The new cell sides length is the cell sides length of the grid to
    # downside multiplied by two
    downsized_grid_cell_sides_length = \
        grid_level_N_sine_waves.grid.cell_sides_length * 2

    downsided_grid_params = grid.Grid(
        downsized_grid_cell_ndcount,
        grid_level_N_sine_waves.grid.origin,
        downsized_grid_cell_sides_length)

    downsized_grid_cell_ndindices = vmap(
        cell.ndindex_from_1dindex, (0, None))(
        downsized_grid_cell_1dindices, downsized_grid_cell_ndcount)
    downsized_grid_sines = vmap(
        grid_restrict_from_cell_ndindex, (0, None, None))(
        downsized_grid_cell_ndindices, grid_level_N_sine_waves, f)

    grid_downsized = GridPointData(
        downsized_grid_sines, downsided_grid_params)

    return grid_downsized


def grid_alignment_energy_with_neighbors(
        i: int,
        grid_sine: GridPointData,
        f: float,
        use_spatial_weights: bool,
        mode: int) -> float:
    """Return the alignment energy of a sine wave in a grid with its neighbors.

    Parameters
    ----------
    i : int
        Index of the sine wave whose alignment energy with its neighbors needs
        to be computed.
    grid_sine : GridPointData
        The grid of sine waves. `grid_sine.point_data.point.shape` is `(m, n)`,
        where `m` is the number of points and where n is the points component's
        count. `grid_sine.point_data.data.shape` must be `(m, >=n+1)`.
        `grid_sine.point_data.data[:,:n]` must be the sine directions.
        `grid_sine.point_data.data[:, n]` must be the phases.
    f : float
        Frequency of all the sines.
    use_spatial_weights : bool
        If `True`, neighbors on the diagonals are less influential than closer
        neighbors. Otherwise, each neighbor has the same weight.
    mode : int
        0: Return the average of the line smoothness and phase alignment
        energy. 1: Return only the smoothness energy. 2: Return the phase
        alignment energy.
    Return
    ------
    float
        Return the alignment energy of the ith sine wave with its neighbors.
        The result is scaled to fit between 0 and 1. See `mode` parameter for
        more details.
    """
    # Unpack data
    sine: PointData = tree_util.leaves_at_indices(
        i, grid_sine.point_data)
    sine_default = PointData(
        jnp.zeros_like(
            sine.point), jnp.zeros_like(
            sine.data))
    sine_mask = jnp.isnan(sine.data[0])
    sine = lax.cond(
        sine_mask,
        lambda x: x[0],
        lambda x: x[1],
        (sine_default,
         sine))

    n = sine.point.shape[0]

    # Get the neighboring sines
    neighbors = grid_neighborhood_from_point(
        sine.point, grid_sine, True)
    neighbors_default = PointData(
        jnp.zeros_like(
            neighbors.point), jnp.zeros_like(
            neighbors.data))
    neighboring_mask = jnp.logical_or(
        jnp.isnan(neighbors.data[:, 0]), sine_mask)
    neighbors_points: PointData = jnp.where(
        neighboring_mask.reshape(
            (-1, 1)), neighbors_default.point, neighbors.point)
    neighbors_data: PointData = jnp.where(
        neighboring_mask.reshape(
            (-1, 1)), neighbors_default.data, neighbors.data)
    neighbors = PointData(neighbors_points, neighbors_data)

    weights = jnp.logical_not(neighboring_mask).astype(type.float_)

    neighbors_all_masked = jnp.all(neighboring_mask)

    # To avoid spatial weights near zero, the variance of the gaussian is
    # calculated with the cell sides length times two.
    neighboring_kernel_variance = multivariate_normal.variance_from_radius(
        grid_sine.grid.cell_sides_length * 2.)
    spatial_weights = vmap(
        multivariate_normal.eval_pdf,
        (0, None, None))(
        neighbors.point, sine.point, neighboring_kernel_variance)
    constant_weights = jnp.ones_like(weights)
    spatial_weights = jnp.where(
        use_spatial_weights,
        spatial_weights,
        constant_weights)
    weights *= spatial_weights

    d = sine.data[:n]
    D = neighbors.data[:, :n]
    D_reshaped = D.reshape(D.shape[0], 1, n)
    line_smoothness_energy = jax.vmap(
        line.smoothness_energy_normalized,
        (None, 0))(
            d,
            D_reshaped)
    line_smoothness_energy_normalized = jnp.sum(
        line_smoothness_energy * weights) / jnp.sum(weights)

    invalid = jnp.logical_or(neighbors_all_masked, sine_mask)
    line_smoothness_energy_normalized = jnp.where(
        invalid, jnp.nan, line_smoothness_energy_normalized)

    sine_alignment_energy = phase_alignment_energy_with_others(
        sine, neighbors, weights, f)
    sine_alignment_energy = jnp.where(invalid, jnp.nan, sine_alignment_energy)

    energy = (line_smoothness_energy_normalized + sine_alignment_energy) * 0.5
    if mode == 1:
        energy = line_smoothness_energy_normalized
    elif mode == 2:
        energy = sine_alignment_energy
    return energy


def grid_alignment_energy(
        grid_sine: GridPointData,
        f: float,
        use_spatial_weights: bool,
        mode: int) -> jnp.ndarray:
    """Return the alignment energy of each sine waves in a grid.

    This function returns the alignment energies of each sine waves in a grid.
    More precisely, this function is a vectorized version of
    `grid_alignment_energy_with_neighbors`, where the iteration goes through
    all the sine waves of the grid.

    Parameters
    ----------
    grid_sine : GridPointData
        The grid of sine waves. `grid_sine.point_data.point.shape` is `(m, n)`,
        where `m` is the number of points and where n is the points component's
        count. `grid_sine.point_data.data.shape` must be `(m, n+2)`.
        `grid_sine.point_data.data[:,:n]` must be the sine directions.
        `grid_sine.point_data.data[:, n]` must be the phases.
    f : float
        The frequency of all the sine waves.
    use_spatial_weights : bool
        If `True`, neighbors on the diagonals are less influential than closer
        neighbors. Otherwise, each neighbor has the same weight.
    mode : int
        0: Return the average of the line smoothness and phase alignment
        energy. 1: Return only the smoothness energy. 2: Return the phase
        alignment energy.

    Returns
    -------
    ndarray
        It is a ndarray with a `(m,)` shape, where `m` is the sine wave count.
        Each element has the alignment energy defined by the `mode` parameter.
    """

    sine_count = grid_sine.point_data.point.shape[0]
    sine_indices = jnp.arange(sine_count)
    sines_energy = vmap(
        grid_alignment_energy_with_neighbors, (0, None, None, None, None))(
        sine_indices, grid_sine, f, use_spatial_weights, mode)
    return sines_energy


def grid_shape_dtype(
        side_cell_count: int,
        dim: int) -> GridPointData:
    """Return the shape and data type of a grid of sine waves.

    This function returns the shape and data type of each attribute of a square
    grid of sine waves with specified side cell count and dimension.

    Parameters
    ----------
    side_cell_count : int
        The number of cells along the side of the square grid.
    dim : int
        The number of dimensions of the grid.

    Returns
    -------
        GridPointData
    """
    grid_sine_wave_1dcount = side_cell_count**2
    uniform_grid_shape_dtype = grid.shape_dtype_from_dim(dim)
    point_shape_dtype = ShapeDtypeStruct(
        (grid_sine_wave_1dcount, dim), type.float_)
    # Direction + phase + constraint
    data_shape_dtype = ShapeDtypeStruct(
        (grid_sine_wave_1dcount, dim + 1 + 1), type.float_)

    sine_wave_shape_dtype = PointData(point_shape_dtype, data_shape_dtype)
    grid_sine_wave_shape_dtype = GridPointData(
        sine_wave_shape_dtype, uniform_grid_shape_dtype)
    return grid_sine_wave_shape_dtype


def multigrid_create(
        grid_level_0_side_cell_count: int,
        grid_level_0_sine_waves: GridPointData,
        f: float) -> list[GridPointData]:
    """Create a multigrid of sine waves.

    This function creates a hierarchy of grids, aka a multiresolution grid,
    abbr. [multigrid](https://en.wikipedia.org/wiki/Multigrid_method). Each
    cell contains a sine wave. The multigrid is created by recursively
    restricting a grid until a grid with one cell is obtained.

    Parameters
    ----------
    grid_level_0_side_cell_count : int
        The side cell count of the grid at level 0, i.e., the level with the
        finest resolution. The number of distinct cells in each dimension (x
        and y) is considered equal. Must be a power of two.
    grid_level_0_sine_waves : GridPointData
        The sine waves of the finest grid (at level 0).
    f : float
        The frequency of the sines.

    Returns
    -------
    list[GridPointData]
        The hierarchy of grids containing the sine waves. Each level is
        contained in a list, indexed by its level.
    """

    # Assume power of two and equal cell count for each axis of the grid
    level_count = 1 + stdmath.floor(stdmath.log2(grid_level_0_side_cell_count))
    hierarchy_grid = [grid_level_0_sine_waves]
    for i_level in range(1, level_count):
        grid_i: GridPointData = grid_restrict(
            grid_level_0_side_cell_count // 2**(i_level - 1),
            hierarchy_grid[i_level - 1],
            f)
        hierarchy_grid.append(grid_i)
    return hierarchy_grid


def multigrid_align(
        side_cell_count: int,
        grid_sine: GridPointData,
        f: float,
        iter_count_per_level: int,
        use_spatial_weights: bool) -> GridPointData:
    """Align the sine waves in a grid using a multigrid method.

    Parameters
    ----------
    side_cell_count : int
        The number of cells along the axes of the grid. The number must be a
        power of two, and each axis has the same side number of cells.
    grid_sine : GridPointData
        The grid of sine waves to align.
    f : float
        The frequency of all the sines.
    iter_count_per_level : int
        The number of alignment iterations for each level of the hierarchy of
        grids.
    use_spatial_weights : bool
        If `True`, neighbors on the diagonals are less influential than closer
        neighbors. Otherwise, each neighbor has the same weight.
    """

    # This hierarchy has the grid with the highest resolution at index zero
    grid_sines_hierarchy = multigrid_create(side_cell_count, grid_sine, f)
    level_count = len(grid_sines_hierarchy)

    # This hierarchy will have the lowest resolution at index zero
    # It will contain the aligned sines at each resolution
    # Aligned sines for one level is an excellent starting guess for  aligning
    # for the next level
    grid_sines_hierarchy_aligned = [grid_sines_hierarchy[-1]]

    for i in range(level_count - 1):
        coarse_grid_i = grid_sines_hierarchy_aligned[i]
        fine_grid_i = grid_sines_hierarchy[-i - 2]
        if i != 0:
            coarse_grid_i: GridPointData = grid_align_n_times(
                coarse_grid_i,
                f,
                use_spatial_weights,
                iter_count_per_level)
            grid_sines_hierarchy_aligned[i] = coarse_grid_i
        detailed_grid_sines_i: PointData = grid_prolong(
            2**i, coarse_grid_i, fine_grid_i, f)
        fine_grid_i = GridPointData(detailed_grid_sines_i, fine_grid_i.grid)
        grid_sines_hierarchy_aligned.append(fine_grid_i)

    # Optimize the highest resolution grid
    most_detailed_grid = grid_sines_hierarchy_aligned[-1]
    most_detailed_grid: GridPointData = grid_align_n_times(
        most_detailed_grid,
        f,
        use_spatial_weights,
        iter_count_per_level)
    return most_detailed_grid


def multigrid_compile_align(
        side_cell_count: int,
        dim: int,
        alignment_iter_per_level: int,
        device: xla_client.Device) -> tuple[jax.stages.Compiled, float]:
    """Compile `multigrid_align`.

    Parameters
    ----------
    side_cell_count : int
        The number of side cells.
    dim : int
        The number of dimensions.
    alignment_iter_per_level : int
        The number of alignment iterations per level.
    device : xla_client.Device
        The device the compiled function will run on.
        Available devices can be retrieved via `jax.devices()`.

    Return
    ------
    tuple[Compiled, float]
        out1
            The function returns `grid_sines_align_with_multigrid` compiled for
            the specified parameters.
        out2
            The duration of the compilation (seconds).
    """
    grid_sines_shape_dtype = grid_shape_dtype(side_cell_count, dim)

    start = time.perf_counter()
    grid_sines_aligned_jit = jit(
        multigrid_align, static_argnums=(
            0, 3, 4), device=device)
    grid_sines_aligned_lowered = grid_sines_aligned_jit.lower(
        side_cell_count,
        grid_sines_shape_dtype,
        ShapeDtypeStruct((),
                         type.float_),
        alignment_iter_per_level,
        True)
    grid_sines_aligned_compiled = grid_sines_aligned_lowered.compile()
    stop = time.perf_counter()

    return grid_sines_aligned_compiled, stop - start


def eval_angle(
        x: jnp.ndarray,
        p: jnp.ndarray,
        d: jnp.ndarray,
        phase: float,
        f: float) -> float:
    """Evaluate the angle of a ND sine wave.

    Parameters
    ----------
    x : ndarray
        The position of evaluation.
    p : ndarray
        The position of the origin of the sine wave.
    d : ndarray
        The direction of the sine wave, in cartesian coordinates.
    phase : float
        The phase of the sine wave.
    f : float
        The frequency of the sine wave.
    Return
    ------
    float
        The angle of the sine wave (Chermain et al. 2023, term inside the sine
        in Eq. 6.).
    """
    return math.TWO_PI * f * jnp.vdot(d, x - p) + phase


def eval_complex_angle(
        x: jnp.ndarray,
        p: jnp.ndarray,
        d: jnp.ndarray,
        phase: jnp.ndarray,
        f: float) -> complex:
    """Return the angle of a ND sine wave represented in the complex plane.

    Parameters
    ----------
    x : ndarray
        The position of evaluation.
    p : ndarray
        The position of the origin of the sine wave.
    d : ndarray
        The direction of the sine wave, in cartesian coordinates.
    phase : float
        The phase of the sine wave.
    f : float
        The frequency of the sine wave.

    Returns
    -------
    complex
        The angle of the sine wave in the complex plane.
    """
    return jnp.exp(1j * (eval_angle(x, p, d, phase, f)))


def eval(
        x: jnp.ndarray,
        p: jnp.ndarray,
        d: jnp.ndarray,
        phase: jnp.ndarray,
        f: float) -> float:
    """Evaluate the value of a ND sine wave.

    Parameters
    ----------
    x : ndarray
        The position of evaluation.
    p : ndarray
        The position of the origin of the sine wave.
    d : ndarray
        The direction of the sine wave, in cartesian coordinates.
    phase : float
        The phase of the sine wave.
    f : float
        The frequency of the sine wave.
    Return
    ------
    float
        The value of the sine wave (Chermain et al. 2023, Eq. 6.).
    """
    return jnp.sin(eval_angle(x, p, d, phase, f))


class SineWaveConstrained(NamedTuple):
    direction: jnp.ndarray
    phase: float
    constraint: float


def pack_constrained(point_data: SineWaveConstrained) -> jnp.ndarray:
    return jnp.concatenate(
        (point_data.direction,
         jnp.array([point_data.phase]),
         jnp.array([point_data.constraint])),
        axis=0)


def unpack_constrained(data: jnp.ndarray) -> SineWaveConstrained:
    return SineWaveConstrained(data[:2], data[2], data[3])
