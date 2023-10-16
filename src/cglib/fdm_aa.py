import json
import os
import time
from enum import IntEnum
from io import TextIOWrapper
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jax import device_get, device_put, lax, vmap
from jax._src.lib import xla_client

from . import cycle, gabor_filter, grid
from . import math as cgm
from . import point_data, scalar, sine_wave, texture, transform, type
from .grid import cell, edge
from .polyline import Polyline


class Parameters:
    def __init__(self):
        self.path = ""
        self.nozzle_width = 0.
        self.svg_path = ""
        self.line_field_path = ""
        self.line_mode_field_path = ""
        self.perimeter_count = 0
        self.alignment_iter_per_level = 0
        self.seed = 0
        self.force_sdf_computation = False
        self.repulse_curves = False
        self.layer_height_rt_nozzle_width = 0.
        self.scalar_field_filename = ""
        self.grid_sines_aligned_filename = ""
        self.contour_graph_filename = ""
        self.cycles_filename = ""
        self.cycle_polyline_filename = ""
        self.cycle_txt_filename = ""
        self.gcode_filename = ""
        self.boundary_polydata_filename = ""
        self.boundary_normal_glyph_filename = ""
        self.log_filename = ""
        self.sdf_filename = ""

    def load(self, path: str) -> None:
        """Load from a json file the parameters needed by the cycle generator.

        Parameters
        ----------
        path : str
            The path to the json file.

        Returns
        -------
        dict
        """
        self.path = path
        with open(path) as file:
            data = file.read()
            params_dict = json.loads(data)

        self.nozzle_width = params_dict['nozzle_width']
        self.svg_path = params_dict['svg_path']
        self.line_field_path = params_dict['line_field_path']
        self.line_mode_field_path = params_dict['line_mode_field_path']
        self.perimeter_count = params_dict['perimeter_count']
        self.alignment_iter_per_level = \
            params_dict['alignment_iter_per_level']
        self.seed = params_dict['seed']
        self.force_sdf_computation = params_dict['force_sdf_computation']
        self.repulse_curves = params_dict['repulse_curves']
        self.layer_height_rt_nozzle_width = \
            params_dict['layer_height_rt_nozzle_width']

        # Derived parameter

        path_head, path_tail = os.path.split(path)
        path_tail_wo_ext = os.path.splitext(path_tail)[0]
        self.scalar_field_filename = path_head + '/np/' + \
            path_tail_wo_ext + '_scalar_field.npy'
        self.grid_sines_aligned_filename = path_head + '/np/' + \
            path_tail_wo_ext + '_grid_sines_aligned.npz'
        self.contour_graph_filename = path_head + '/np/' + \
            path_tail_wo_ext + '_contour_graph.npz'
        self.cycles_filename = path_head + '/np/' + \
            path_tail_wo_ext + '_cycles.npz'
        self.cycle_polyline_filename = path_head + '/np/' + \
            path_tail_wo_ext + '_polyline_3d.npz'
        self.cycle_txt_filename = path_head + '/paths/' + \
            path_tail_wo_ext + '.paths'
        self.gcode_filename = path_head + '/gcode/' + \
            path_tail_wo_ext + '.gcode'
        self.boundary_polydata_filename = path_head + '/vtk/' + \
            path_tail_wo_ext + '_boundary.vtk'
        self.boundary_normal_glyph_filename = path_head + '/vtk/' + \
            path_tail_wo_ext + '_boundary_normal.vtk'
        self.log_filename = path_head + '/log/' + \
            path_tail_wo_ext + '.log'

    def create_SDF_filename(self, shape_grid: grid.Grid) -> None:
        path_head, path_tail = os.path.split(self.path)
        path_tail_wo_ext = os.path.splitext(path_tail)[0]

        self.sdf_filename = path_head + '/np/' + \
            path_tail_wo_ext + '_sdf_' + str(shape_grid.cell_ndcount[0]) \
            + '_' + str(shape_grid.cell_ndcount[1]) + '_' + str(self.seed) + '_' + \
            f"{shape_grid.cell_sides_length:.3f}" + '.npz'


def load_line_field_textures(
        line_field_path: str,
        line_mode_field_path: str,
        device: xla_client.Device):
    """

    Parameters
    ----------
    line_field_path : str
        The filename of the png image containing the line field texture. The
        format of the images is PNG, with one channel per pixel and 8 bits per
        channel. Black (0) and white (255) correspond to a line with angles
        -pi/2 and pi/2, resp. Other values [1, 254] give a line with an angle
        linearly interpolated between -pi/2 and pi/2.
    line_field_path : str
        The filename of the png image containing the line field mode. The
        format of the images is PNG, with one channel per pixel and 8 bits per
        channel. Values in [0, 42) are for the parallel to boundary mode.
        Values in [42, 127) are for the orthogonal to boundary mode. Values in
        [127, 212) are for the smoothest line field mode. Values in [212, 255]
        are for the constrained line mode. See Chermain et al. 2013, Section 4
        for more details on the modes.
    device : Device
        The image textures will reside on `device`. Available devices can be
        retrieved via `jax.devices()`.

    Returns
    -------
    tuple[Image, Image]
        out1, out2
            The line field texture and the line mode field textures. The pixel
            values are in [0, 1]. The texture image wrap is set to clamp.
    """
    line_field_tex = None
    if line_field_path is not None:
        line_field_tex = texture.image_create_from_png_1channel_8bits(
            line_field_path, texture.ImageWrap.CLAMP, device)
    line_mode_field_tex = None
    if line_mode_field_path is not None:
        line_mode_field_tex = texture.image_create_from_png_1channel_8bits(
            line_mode_field_path, texture.ImageWrap.CLAMP, device)
    return line_field_tex, line_mode_field_tex


class NozzleWidthDerivedParam(NamedTuple):
    layer_height: float
    period: float
    frequency: float
    domain_grid_points_perturbation_width: float
    min_width: float
    min_radius: float
    max_width: float
    max_radius: float


def compute_nozzle_width_derived_parameters(
        nozzle_width: float,
        layer_height_rt_nozzle_width: float) -> NozzleWidthDerivedParam:
    """Compute the nozzle width-derived parameters.

    Parameters
    ----------
    nozzle_width: float
        The nozzle width.
    layer_height_rt_nozzle_width: float
        The layer height given as a percentage of the nozzle width.
    """
    domain_grid_cells_width = nozzle_width * 0.5
    layer_height = nozzle_width * layer_height_rt_nozzle_width
    period = nozzle_width * 2.
    frequency = 1. / period
    domain_grid_points_perturbation_width = domain_grid_cells_width * 0.1
    min_width = 0.5 * nozzle_width
    min_radius = min_width * 0.5
    max_width = 2. * nozzle_width
    max_radius = max_width * 0.5

    return NozzleWidthDerivedParam(
        layer_height,
        period,
        frequency,
        domain_grid_points_perturbation_width,
        min_width,
        min_radius,
        max_width,
        max_radius)


def discretize_2dshape_domain_with_cells(
        nozzle_width: float,
        shape_domain_2dsize: jnp.ndarray) -> grid.Grid:
    """Discretized the 2D shape domain with cells and returns a 2D grid.

    The origin of the domain of the 2D shape is at the bottom left. The domain
    is sampled with points towards each domain axis. The sampling start from
    the origin (0., 0.), and the distance between points is half the nozzle
    width along each axis. The last sampling point on each axis is either on
    the boundary or outside the domain. These points are the cell centers of
    the grid which discretizes the 2D shape domain. A margin of two cells is
    added. The grid's origin discretizing the domain is, therefore, -1.5 times
    the nozzle widths.

    Parameters
    ----------
    nozzle_width : float
        The nozzle width, i.e., the target distance between adjacent
        trajectories.        
    domain_2dsize : ndarray
        A ndarray with shape (2,). The first and second elements give the size
        of the 2D shape domain along the x and y axis, respectively.

    Returns
    -------
    UniformGrid
        The grid discretizing the shape domain. The domain cells width of the
        grid is set to nozzle_width / 2. This value assumes that parallel
        trajectories are represented as the iso-curves of a periodic function
        with period nozzle width. See
        https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem
    """
    cells_width = nozzle_width * 0.5
    margin_length = 1.5*nozzle_width
    origin = jnp.full((2,), -margin_length)
    grid_side_size = margin_length * 2. + shape_domain_2dsize

    # Put a margin of two cells.
    cell_2dcount = jnp.ceil(
        grid_side_size / cells_width).astype(type.int_)
    return grid.Grid(cell_2dcount, origin, cells_width)


def compute_boundary_cycles_from_svg_paths(
        domain_size,
        paths,
        nozzle_width,
        device_cpu):

    def compute_normal(i, val):
        path_points = val[0]
        upside_down_to_right_side_up = val[1]
        boundary_cycles_point_index = val[2]
        boundary_cycles_points = val[3]
        boundary_cycles_points_normals = val[4]
        boundary_cycle_i_edge_count = val[5]

        # 90 degree rotation, counter clock wise
        ROTATION_PIOVER2 = jnp.array([[0., -1.],
                                      [1.,  0.]])

        path_point = path_points[i]
        point_i = transform.apply_to_point(
            upside_down_to_right_side_up, path_point)
        boundary_cycles_points = \
            boundary_cycles_points.at[boundary_cycles_point_index].set(point_i)
        boundary_cycles_points_normals = \
            boundary_cycles_points_normals.at[boundary_cycles_point_index].set(
                jnp.zeros_like(point_i))
        # Compute normal for previous point

        # if point_index > 1:
        points = \
            jnp.array([boundary_cycles_points[boundary_cycles_point_index-2],
                       boundary_cycles_points[boundary_cycles_point_index-1],
                       boundary_cycles_points[boundary_cycles_point_index]])
        tangents = vmap(cgm.vector_normalize)(points[1:] - points[:-1])
        tangents_sum = cgm.vector_normalize(jnp.sum(tangents, axis=0))
        normal = jnp.squeeze(jnp.matmul(
            ROTATION_PIOVER2, tangents_sum.reshape(-1, 1)))
        boundary_cycles_points_normals = \
            boundary_cycles_points_normals.at[
                boundary_cycles_point_index-1].set(normal)

        boundary_cycles_point_index += 1
        boundary_cycle_i_edge_count += 1

        return (
            path_points,
            upside_down_to_right_side_up,
            boundary_cycles_point_index,
            boundary_cycles_points,
            boundary_cycles_points_normals,
            boundary_cycle_i_edge_count)

    SAMPLING_RATE = 0.2

    # Prepare data
    boundary_cycles_point_index = 0

    point_count_predicted = int(
        ((domain_size[0] + nozzle_width*2.) * (domain_size[1] + nozzle_width*2.)) / (SAMPLING_RATE)**2)

    boundary_cycles_points_np = np.zeros((point_count_predicted, 2))
    boundary_cycles_points_normals_np = np.zeros((point_count_predicted, 2))

    boundary_cycles_start_index: list[type.uint] = []
    boundary_cycles_edge_count: list[type.uint] = []
    boundary_cycle_count: type.uint = 0
    # Origin is located at top-left corner in the svg file. Here we use an
    # origin at the bottom-left corner, so an y flip is performed.
    trans = transform.translate(device_put(
        np.array([0., -domain_size[1]*0.5]), device=device_cpu))
    scale = transform.scale(device_put(
        np.array([1., -1.]), device=device_cpu))
    upside_down_to_right_side_up = jnp.linalg.inv(trans) @ scale @ trans

    # 90 degree rotation, counter clock wise
    ROTATION_PIOVER2 = np.array([[0., -1.],
                                 [1.,  0.]])
    ROTATION_PIOVER2 = device_put(ROTATION_PIOVER2, device=device_cpu)
    # Go through all the boundary discontinuous paths
    for path_discontinuous in paths:
        for path in path_discontinuous.continuous_subpaths():
            boundary_cycle_i_edge_count = 0
            boundary_cycles_start_index.append(boundary_cycles_point_index)
            path_length = path.length()
            sample_point_count = int(path_length / SAMPLING_RATE)

            boundary_cycle_points: list[tuple[float, float]] = []
            for point_index in range(sample_point_count):
                t = (point_index + 0.5) / sample_point_count
                path_point = path.point(t)
                boundary_cycle_points.append(
                    [path_point.real, path_point.imag])

            init_val = (
                device_put(np.array(boundary_cycle_points), device=device_cpu),
                upside_down_to_right_side_up,
                boundary_cycles_point_index,
                device_put(boundary_cycles_points_np, device=device_cpu),
                device_put(boundary_cycles_points_normals_np,
                           device=device_cpu),
                boundary_cycle_i_edge_count)

            res = lax.fori_loop(
                0, sample_point_count, compute_normal, init_val)
            boundary_cycle_points = res[0]
            upside_down_to_right_side_up = res[1]
            boundary_cycles_point_index = res[2]
            boundary_cycles_points_jnp = res[3]
            boundary_cycles_points_normals_jnp = res[4]
            boundary_cycle_i_edge_count = res[5]

            boundary_cycles_points_np = np.array(
                device_get(boundary_cycles_points_jnp))
            boundary_cycles_points_normals_np = np.array(
                device_get(boundary_cycles_points_normals_jnp))

            # Compute the first and last normals of the current cycle
            last_index = boundary_cycles_point_index - 1
            last_index_m1 = last_index - 1
            first_index = boundary_cycles_point_index - boundary_cycle_i_edge_count
            first_index_p1 = first_index + 1
            points = np.array([boundary_cycles_points_np[last_index_m1],
                               boundary_cycles_points_np[last_index],
                               boundary_cycles_points_np[first_index],
                               boundary_cycles_points_np[first_index_p1]])
            tangents = vmap(cgm.vector_normalize)(device_put(
                points[1:] - points[:-1], device=device_cpu))
            tangents_sum = jnp.array(
                [cgm.vector_normalize(jnp.sum(tangents[:2], axis=0)).reshape(-1, 1),
                 cgm.vector_normalize(jnp.sum(tangents[1:], axis=0)).reshape(-1, 1)])
            normal = vmap(jnp.squeeze)(
                vmap(jnp.matmul, (None, 0))(ROTATION_PIOVER2, tangents_sum))
            normal = device_get(normal)
            boundary_cycles_points_normals_np[last_index] = normal[0]
            boundary_cycles_points_normals_np[first_index] = normal[1]

            # Append the number of edge
            boundary_cycles_edge_count.append(boundary_cycle_i_edge_count)
            # Increase the number of cycles
            boundary_cycle_count += 1

    # Transfer points and normals to the cpu device
    boundary_points = device_put(
        boundary_cycles_points_np[:boundary_cycles_point_index, :], device=device_cpu)
    boundary_cycles_points_normals = device_put(
        boundary_cycles_points_normals_np[:boundary_cycles_point_index, :], device=device_cpu)
    boundary_cycles_start_index = np.array(boundary_cycles_start_index)
    boundary_cycles_start_index = device_put(
        boundary_cycles_start_index, device=device_cpu)
    boundary_cycles_edge_count = np.array(boundary_cycles_edge_count)
    boundary_cycles_edge_count = device_put(
        boundary_cycles_edge_count, device=device_cpu)
    boundary_cycles_data = jnp.concatenate(
        (boundary_cycles_start_index.reshape(-1, 1), boundary_cycles_edge_count.reshape(-1, 1)), axis=1)
    boundary_cycles = cycle.Cycle(point_data.PointData(
        boundary_points, boundary_cycles_points_normals), boundary_cycles_data, boundary_cycle_count)
    return boundary_cycles


def compute_right_side_up_to_normalized_texture_space_transform(
        domain_2dsize: jnp.ndarray) -> jnp.ndarray:
    trans = transform.translate(jnp.array([0., -domain_2dsize[1]*0.5]))
    scale = transform.scale(jnp.array([1., -1.]))
    upside_down_to_right_side_up = jnp.linalg.inv(trans) @ scale @ trans
    right_side_up_to_upside_down = jnp.linalg.inv(upside_down_to_right_side_up)
    upside_down_to_normalized = transform.scale(
        jnp.array([1./domain_2dsize[0], 1./domain_2dsize[1]]))
    right_side_up_to_normalized_texture_space = upside_down_to_normalized \
        @ right_side_up_to_upside_down
    return right_side_up_to_normalized_texture_space


def cycle_to_polydata(
        boundary_cycles: cycle.Cycle,
        height) -> tuple[pv.MultiBlock, pv.MultiBlock]:
    boundary_polydata = pv.PolyData()
    boundary_normal_glyph = pv.PolyData()
    for cycle_index in range(boundary_cycles.cycle_count):
        boundary_polydata_i = pv.PolyData()
        cycle_i_start_index = boundary_cycles.cycle_data[cycle_index, 0]
        cycle_i_edge_count = boundary_cycles.cycle_data[cycle_index, 1]
        cycle_i_end_index = cycle_i_start_index + cycle_i_edge_count
        polyline_points = boundary_cycles.point_data.point[
            cycle_i_start_index:cycle_i_end_index, :]
        polyline_points_3d = jnp.concatenate(
            (polyline_points, jnp.full((polyline_points.shape[0], 1), height)),
            axis=1)
        boundary_polydata_i.points: np.array = device_get(polyline_points_3d)

        cycle_cells = np.full(
            (boundary_polydata_i.points.shape[0], 3), 2, dtype=np.int_)
        cycle_cells[:, 1] = np.arange(0, cycle_cells.shape[0], dtype=np.int_)
        cycle_cells[:, 2] = np.arange(1, cycle_cells.shape[0]+1, dtype=np.int_)
        cycle_cells[-1, 2] = 0

        boundary_polydata_i.lines = cycle_cells

        polyline_normals = boundary_cycles.point_data.data[
            cycle_i_start_index:cycle_i_end_index, :]

        polyline_normals_3d = jnp.concatenate(
            (polyline_normals, jnp.zeros((polyline_normals.shape[0], 1))),
            axis=1)
        boundary_polydata_i.point_data["normals"] = device_get(
            polyline_normals_3d)
        boundary_normal_glyph_i = boundary_polydata_i.glyph(
            orient="normals", factor=1., geom=pv.Arrow())

        # Append the polydata and the glyphs
        boundary_polydata = pv.MultiBlock(
            (boundary_polydata, boundary_polydata_i)).combine()
        boundary_normal_glyph = pv.MultiBlock(
            (boundary_normal_glyph, boundary_normal_glyph_i)).combine()

    return boundary_polydata, boundary_normal_glyph


def compute_signed_distance_from_boundary(
        point2: np.ndarray,
        boundary_polydata: pv.MultiBlock):
    height = boundary_polydata.points[0, 2]
    point3 = np.concatenate(
        (point2, np.full((point2.shape[0], 1), height)), axis=1)

    # Get closest cells and points
    closest_cells, closest_points = boundary_polydata.find_closest_cell(
        point3, return_closest_point=True)
    closest_cell_point_index = boundary_polydata.cells[np.array(
        [closest_cells * 3 + 1, closest_cells * 3 + 2])]
    # Get the nearest normal by interpolating closest cell points' normal
    closest_cell_point = boundary_polydata.points[closest_cell_point_index]
    closest_cell_point_normal = boundary_polydata.point_data["normals"][closest_cell_point_index]
    closest_cell_point_tangent = closest_cell_point[1] - closest_cell_point[0]
    vdotv = np.vectorize(np.vdot, signature='(n),(n)->()')
    closest_cell_point_tangent_norm_sqr = vdotv(
        closest_cell_point_tangent, closest_cell_point_tangent)
    closest_cell_normal_weight = vdotv(
        closest_cell_point_tangent, closest_points - closest_cell_point[0])
    closest_cell_normal_weight = closest_cell_normal_weight / \
        closest_cell_point_tangent_norm_sqr
    # Clamp
    closest_cell_normal_weight = np.where(
        closest_cell_normal_weight < 0., 0., closest_cell_normal_weight)
    closest_cell_normal_weight = np.where(
        closest_cell_normal_weight > 1., 1., closest_cell_normal_weight)
    closest_cell_normal_weight = closest_cell_normal_weight.reshape(-1, 1)
    # Linear interpolation between each cell point normal using the weight
    closest_boundary_normal = (1. - closest_cell_normal_weight) * closest_cell_point_normal[0] + \
        closest_cell_normal_weight * closest_cell_point_normal[1]
    signed_distance_from_boundary = np.linalg.norm(
        point3 - closest_points, axis=1)
    side_scalar = vdotv(closest_boundary_normal,
                        point3 - closest_points)
    inside = np.where(side_scalar < 0., True, False)
    signed_distance_from_boundary = np.where(
        inside, -signed_distance_from_boundary, signed_distance_from_boundary)
    return signed_distance_from_boundary, closest_boundary_normal


class Border(NamedTuple):
    infill_shell: float
    shell_perimeter: float
    perimeter_shifted_outside: float


def create_borders(nozzle_width: float, shell_count: int) -> Border:
    border_infill_shell = -nozzle_width * (1+shell_count)
    border_shell_perimeter = -nozzle_width
    border_perimeter_shifted_outside = -nozzle_width*0.5

    return Border(
        border_infill_shell,
        border_shell_perimeter,
        border_perimeter_shifted_outside)


def get_shifted_outside_area(
        sdf: jnp.ndarray,
        border_perimeter_shifted_outside: float) -> jnp.ndarray:
    return sdf > border_perimeter_shifted_outside


def get_perimeter_area(
        sdf: jnp.ndarray,
        border_shell_perimeter: float,
        border_perimeter_shifted_outside: float
) -> jnp.ndarray:
    return jnp.logical_and(sdf > border_shell_perimeter, sdf <= border_perimeter_shifted_outside)


def get_beyond_shell_area(
        sdf: jnp.ndarray,
        border_shell_perimeter: float
) -> jnp.ndarray:
    return sdf > border_shell_perimeter


def get_shell_area(
        sdf: jnp.ndarray,
        border_infill_shell: float,
        border_shell_perimeter: float
) -> jnp.ndarray:
    return jnp.logical_and(sdf > border_infill_shell, sdf <= border_shell_perimeter)


def get_infill_area(
        sdf: jnp.ndarray,
        border_infill_shell: float) -> jnp.ndarray:
    return sdf <= border_infill_shell


class DirectionMode(IntEnum):
    """
    Enumeration of the direction modes.

    Attributes
    ----------
    PARALLEL_TO_BOUNDARY:
        A direction that is tagged as parallel to the boundary is the smoothest
        direction that is also parallel to the boundary.
    ORTHOGONAL_TO_BOUNDARY:
        A direction that is tagged as orthogonal to the boundary is the smoothest
        direction that is also orthogonal to the boundary.
    SMOOTHEST:
        A direction that is tagged as smoothest is the smoothest direction.
    CONSTRAINED:
        Specify a constrained directions, i.e., an explicit direction.
    """
    PARALLEL_TO_BOUNDARY = 0
    ORTHOGONAL_TO_BOUNDARY = 1
    SMOOTHEST = 2
    CONSTRAINED = 3


def eval_textures(
        points: jnp.ndarray,
        line_mode_field_tex: texture.Image,
        line_field_tex: texture.Image,
        domain_2dsize: jnp.ndarray,
        interpolation: texture.InterpolationType) -> tuple[jnp.ndarray, jnp.ndarray]:

    right_side_up_to_normalized_texture_space = \
        compute_right_side_up_to_normalized_texture_space_transform(
            domain_2dsize)
    rotation_pi = jnp.array([[0., -1.],
                            [1.,  0.]])

    s = vmap(transform.apply_to_point, (None, 0))(
        right_side_up_to_normalized_texture_space, points)
    angles_normalized = vmap(texture.image_eval, (0, None, None))(
        s, line_field_tex, interpolation)
    direction = vmap(cgm.angle_normalized_to_2ddir)(angles_normalized)
    direction = vmap(jnp.matmul, (None, 0))(rotation_pi, direction)

    direction_mode = vmap(texture.image_eval_nearest, (0, None))(
        s, line_mode_field_tex)
    direction_mode = jnp.around(direction_mode*3.).astype(type.int_)

    return direction, direction_mode


class PointConstraint(IntEnum):
    """
    Enumeration of the point constraint.

    Attributes
    ----------
    NONE:
        No constraints.
    LINE:
        Only the line is constrained.
    PHASE:
        Only the phase is constrained.
    BOTH:
        Both the line and the phase is constrained.
    """
    NONE = 0
    LINE = 1
    PHASE = 2
    BOTH = 3


def merge_constraint_v2(line_is_constrained: jnp.ndarray,
                        phase_is_constrained: jnp.ndarray) -> jnp.ndarray:
    both_are_constrained = jnp.logical_and(
        line_is_constrained, phase_is_constrained)

    point_constraint = jnp.full(
        (line_is_constrained.shape[0],), PointConstraint.NONE)
    point_constraint = jnp.where(
        line_is_constrained, PointConstraint.LINE, point_constraint)
    point_constraint = jnp.where(
        phase_is_constrained, PointConstraint.PHASE, point_constraint)
    point_constraint = jnp.where(
        both_are_constrained, PointConstraint.BOTH, point_constraint)

    return point_constraint


def shape_grid_data_sqr(
        shape_grid_ccpj: jnp.ndarray,
        shape_grid_ccpj_data: jnp.ndarray,
        shape_grid: grid.Grid,
        shape_grid_cell_ndcount: tuple[int, int]):
    with jax.ensure_compile_time_eval():
        shape_grid_sqr_cell_ndcount_0 = cgm.roundup_power_of_2(
            shape_grid_cell_ndcount[0])
        shape_grid_sqr_cell_ndcount_1 = cgm.roundup_power_of_2(
            shape_grid_cell_ndcount[1])
        shape_grid_sqr_side_cell_count = jnp.max(
            jnp.array([shape_grid_sqr_cell_ndcount_0,
                       shape_grid_sqr_cell_ndcount_1])
        )

    point_size = shape_grid_ccpj.shape[1]
    data_size = shape_grid_ccpj_data.shape[1]

    shape_grid_sqr = grid.roundup_power_of_2(shape_grid)

    shape_grid_ccpj_reshaped = shape_grid_ccpj.reshape(
        (shape_grid_cell_ndcount[1],
         shape_grid_cell_ndcount[0],
         point_size))
    shape_grid_ccpj_data_reshaped = shape_grid_ccpj_data.reshape(
        (shape_grid_cell_ndcount[1],
         shape_grid_cell_ndcount[0],
         data_size))

    shape_grid_sqr_ccpj = jnp.full(
        (shape_grid_sqr_side_cell_count,
         shape_grid_sqr_side_cell_count,
         point_size),
        jnp.nan
    )
    shape_grid_sqr_ccpj_data = jnp.full(
        (shape_grid_sqr_side_cell_count,
         shape_grid_sqr_side_cell_count,
         data_size),
        jnp.nan
    )
    shape_grid_sqr_ccpj = shape_grid_sqr_ccpj.at[
        :shape_grid_cell_ndcount[1],
        :shape_grid_cell_ndcount[0]
    ].set(shape_grid_ccpj_reshaped)

    shape_grid_sqr_ccpj_data = shape_grid_sqr_ccpj_data.at[
        :shape_grid_cell_ndcount[1],
        :shape_grid_cell_ndcount[0]
    ].set(shape_grid_ccpj_data_reshaped)

    grid_sines = point_data.GridPointData(
        point_data.PointData(
            shape_grid_sqr_ccpj.reshape((-1, point_size)),
            shape_grid_sqr_ccpj_data.reshape(-1, data_size)),
        shape_grid_sqr)
    return grid_sines


def compile_shape_grid_data_sqr(shape_domain_grid_cell_ndcount: int, device):
    DIMENSION_COUNT = 2
    DATA_SIZE = 4
    shape_domain_grid_cell_1dcount = \
        shape_domain_grid_cell_ndcount[0] * shape_domain_grid_cell_ndcount[1]
    shape_grid_ccpj_shape_dtype = jax.ShapeDtypeStruct(
        (shape_domain_grid_cell_1dcount, DIMENSION_COUNT), type.float_)
    shape_grid_ccpj_data_shape_dtype = jax.ShapeDtypeStruct(
        (shape_domain_grid_cell_1dcount, DATA_SIZE), type.float_)
    grid_shape_dtype = grid.shape_dtype_from_dim(DIMENSION_COUNT)

    start = time.perf_counter()
    shape_grid_data_sqr_jit = jax.jit(
        shape_grid_data_sqr, static_argnums=3, device=device)
    shape_grid_data_sqr_lowered = shape_grid_data_sqr_jit.lower(
        shape_grid_ccpj_shape_dtype,
        shape_grid_ccpj_data_shape_dtype,
        grid_shape_dtype,
        shape_domain_grid_cell_ndcount)
    shape_grid_data_sqr_compiled = shape_grid_data_sqr_lowered.compile()
    stop = time.perf_counter()

    return shape_grid_data_sqr_compiled, stop - start


def init_grid_data(
        sdf: jnp.ndarray,
        shape_grid_points: jnp.ndarray,
        direction_mode: jnp.ndarray,
        perimeter_count: int,
        closest_boundary_normal: jnp.ndarray,
        nozzle_width: float,
        line_from_user: jnp.ndarray) -> tuple:

    border = create_borders(nozzle_width, perimeter_count-1)

    # Mask exterior points
    is_shifted_outside = get_shifted_outside_area(
        sdf, border.perimeter_shifted_outside)
    shape_grid_points_masked = jnp.where(
        is_shifted_outside.reshape(-1, 1),
        jnp.nan,
        shape_grid_points)

    # Determine if we need one or two runs
    
    is_parallel_to_boundary = jnp.equal(direction_mode, DirectionMode.PARALLEL_TO_BOUNDARY)
    is_orthogonal_to_boundary = jnp.equal(direction_mode, DirectionMode.ORTHOGONAL_TO_BOUNDARY)
    is_smoothest = jnp.equal(direction_mode, DirectionMode.SMOOTHEST)
    is_line_user_constrained = jnp.equal(direction_mode, DirectionMode.CONSTRAINED)

    any_parallel_to_boundary = jnp.any(is_parallel_to_boundary)
    any_orthogonal_to_boundary = jnp.any(is_orthogonal_to_boundary)
    any_smoothest = jnp.any(is_smoothest)
    any_line_user_constrained = jnp.any(is_line_user_constrained)

    # autopep8: off
    case_index = 8 * any_parallel_to_boundary + \
                 4 * any_orthogonal_to_boundary + \
                 2 * any_smoothest + \
                 1 * any_line_user_constrained
    # autopep8: on

    one_run_array = jnp.array([
        jnp.equal(case_index, 0b0001),
        jnp.equal(case_index, 0b0010),
        jnp.equal(case_index, 0b0011),
        jnp.equal(case_index, 0b1000)
        ])
    one_run = jnp.any(one_run_array)
    one_run = jnp.logical_and(one_run, perimeter_count < 2)

    # Initialize the phase with the signed distance field and the nozzle width
    sdf_shifted = sdf + nozzle_width * 0.5
    phase = sdf_shifted / nozzle_width * jnp.pi
    
    # Set user line constraints only in two cases, and if there is only one
    # run.
    set_user_line_constraints_array = jnp.array([
        jnp.equal(case_index, 0b0001),
        jnp.equal(case_index, 0b0011)
    ])    
    set_user_line_constraints = jnp.logical_and(
        one_run, jnp.any(set_user_line_constraints_array))
    
    # Set user line constraints only in the infill area
    is_infill = sdf <= - perimeter_count * nozzle_width
    is_user_line_constraint = jnp.logical_and(
        is_infill, set_user_line_constraints)
    line = jnp.where(
        is_user_line_constraint.reshape(-1, 1),
        line_from_user,
        closest_boundary_normal)
    line_is_constrained = is_user_line_constraint
    
    # If the perimeter count is not zero, constrain the lines and phase in the
    # perimeter area.
    is_perimeter = get_perimeter_area(
        sdf,
        border.shell_perimeter,
        border.perimeter_shifted_outside)
    is_constrained_perimeter = jnp.logical_and(perimeter_count > 0, is_perimeter)
    line = jnp.where(
        is_constrained_perimeter.reshape((-1, 1)),
        closest_boundary_normal,
        line)
    line_is_constrained = jnp.where(
        is_constrained_perimeter,
        True,
        line_is_constrained)
    
    
    # Constrain lines in the perimeter if there are two runs or if all the
    # lines are parallel to the boundary.
    constrained_in_perimeter = jnp.logical_or(
        jnp.logical_not(one_run), jnp.equal(case_index, 0b1000))
    line = jnp.where(
        constrained_in_perimeter,
        closest_boundary_normal,
        line)
    line_is_constrained = jnp.where(
        constrained_in_perimeter,
        is_perimeter,
        line_is_constrained)
    
    phase_is_constrained = jnp.where(
        jnp.logical_or(perimeter_count > 0, constrained_in_perimeter),
        is_perimeter,
        False)
    
    constraint = merge_constraint_v2(line_is_constrained, phase_is_constrained)

    sine_wave_constrained_unpacked = sine_wave.SineWaveConstrained(
        line, phase, constraint)

    sine_wave_constrained_packed = jax.vmap(sine_wave.pack_constrained)(
        sine_wave_constrained_unpacked)

    points_mask = jnp.isnan(shape_grid_points_masked[:, 0])
    sine_wave_constrained_packed = jnp.where(
        points_mask.reshape(-1, 1),
        jnp.nan,
        sine_wave_constrained_packed)
    res = (
        shape_grid_points_masked,
        sine_wave_constrained_packed,
        one_run
    )
    return res


def compile_init_grid_data(shape_domain_grid_cell_1dcount: int, device):
    sdf_shape_dtype = jax.ShapeDtypeStruct(
        (shape_domain_grid_cell_1dcount,), type.float_)
    shape_grid_points_shape_dtype = jax.ShapeDtypeStruct(
        (shape_domain_grid_cell_1dcount, 2), type.float_)
    line_mode_field_shape_dtype = jax.ShapeDtypeStruct(
        (shape_domain_grid_cell_1dcount, ), type.int_)
    perimeter_count_shape_dtype = jax.ShapeDtypeStruct((), type.int_)
    closest_boundary_normal_shape_dtype = jax.ShapeDtypeStruct(
        (shape_domain_grid_cell_1dcount, 2), type.float_)
    nozzle_width_shape_dtype = jax.ShapeDtypeStruct((), type.float_)
    lines_from_user_shape_dtype = jax.ShapeDtypeStruct(
        (shape_domain_grid_cell_1dcount, 2), type.float_)

    start = time.perf_counter()
    init_grid_data_jit = jax.jit(init_grid_data, device=device)
    init_grid_data_lowered = init_grid_data_jit.lower(
        sdf_shape_dtype,
        shape_grid_points_shape_dtype,
        line_mode_field_shape_dtype,
        perimeter_count_shape_dtype,
        closest_boundary_normal_shape_dtype,
        nozzle_width_shape_dtype,
        lines_from_user_shape_dtype)
    init_grid_data_compiled = init_grid_data_lowered.compile()
    stop = time.perf_counter()

    return init_grid_data_compiled, stop - start


def prepare_grid_data_for_second_run(
        grid_sines_aligned: point_data.GridPointData,
        shape_grid_sqr_side_cell_count: int,
        shape_grid_cell_2dcount: tuple[int, int],
        perimeter_count: int,
        nozzle_width: float,
        sdf: jnp.ndarray,
        direction_mode: jnp.ndarray,
        line_from_user: jnp.ndarray):

    point_size = grid_sines_aligned.point_data.point.shape[1]
    data_size = grid_sines_aligned.point_data.data.shape[1]

    point_reshaped = grid_sines_aligned.point_data.point.reshape(
        (shape_grid_sqr_side_cell_count, shape_grid_sqr_side_cell_count, point_size))
    point_sliced: jnp.ndarray = \
        point_reshaped[:shape_grid_cell_2dcount[1],
                       :shape_grid_cell_2dcount[0]]
    point_sliced = point_sliced.reshape((-1, 2))

    # Get aligned data
    data_reshaped = grid_sines_aligned.point_data.data.reshape(
        (shape_grid_sqr_side_cell_count, shape_grid_sqr_side_cell_count, data_size))

    data_sliced: jnp.ndarray = \
        data_reshaped[:shape_grid_cell_2dcount[1], :shape_grid_cell_2dcount[0]]
    data_sliced = data_sliced.reshape((-1, 4))

    line = data_sliced[:, :2]
    phase = data_sliced[:, 2]

    # Orthogonal to boundary
    is_infill = sdf <= - perimeter_count * nozzle_width
    rotation_pi = jnp.array([[0., -1.],
                             [1.,  0.]])
    line_rotated = vmap(jnp.matmul, (None, 0))(rotation_pi, line)
    is_ortho_to_boundary = jnp.equal(
        direction_mode, DirectionMode.ORTHOGONAL_TO_BOUNDARY)
    line = jnp.where(
        jnp.logical_and(is_infill, is_ortho_to_boundary).reshape(-1, 1),
        line_rotated,
        line)
    
    # Constrain all the lines, except in the smoothest area
    is_line_smoothest = jnp.equal(direction_mode, DirectionMode.SMOOTHEST)
    line_is_constrained = jnp.where(is_line_smoothest, False, True)
    
    # Set line user constraints
    is_line_user_constrained = jnp.equal(direction_mode, DirectionMode.CONSTRAINED)
    line = jnp.where(
        jnp.logical_and(is_infill, is_line_user_constrained).reshape(-1, 1),
        line_from_user,
        line)

    # If the perimeter count is not zero, constrain the phase in the perimeter
    # area.
    border = create_borders(nozzle_width, perimeter_count-1)
    is_perimeter = get_perimeter_area(
        sdf,
        border.shell_perimeter,
        border.perimeter_shifted_outside)
    phase_is_constrained = jnp.where(perimeter_count > 0, is_perimeter, False)

    constraint = merge_constraint_v2(line_is_constrained, phase_is_constrained)

    sine_wave_constrained_unpacked = sine_wave.SineWaveConstrained(
        line, phase, constraint)

    sine_wave_constrained_packed = jax.vmap(sine_wave.pack_constrained)(
        sine_wave_constrained_unpacked)

    points_mask = jnp.isnan(point_sliced[:, 0])

    sine_wave_constrained_packed = jnp.where(
        points_mask.reshape(-1, 1),
        jnp.nan,
        sine_wave_constrained_packed)

    data = sine_wave_constrained_packed.reshape(
        (shape_grid_cell_2dcount[1],
         shape_grid_cell_2dcount[0],
         data_size))

    data_reshaped = data_reshaped.at[
        :shape_grid_cell_2dcount[1],
        :shape_grid_cell_2dcount[0]
    ].set(data)

    sines = point_data.PointData(
        grid_sines_aligned.point_data.point,
        data_reshaped.reshape(-1, data_size))

    res = point_data.GridPointData(sines, grid_sines_aligned.grid)
    return res


def compile_prepare_grid_data_for_second_run(
        grid_cell_2dcount: tuple[int, int],
        grid_sqr_side_cell_count: int,
        dim: int,
        device):
    grid_cell_1dcount = grid_cell_2dcount[0] * grid_cell_2dcount[1]
    grid_sines_shape_dtype = sine_wave.grid_shape_dtype(
        grid_sqr_side_cell_count, dim)
    perimeter_count_shape_dtype = jax.ShapeDtypeStruct((), type.int_)
    nozzle_width_shape_dtype = jax.ShapeDtypeStruct((), type.float_)
    sdf_shape_dtype = jax.ShapeDtypeStruct((grid_cell_1dcount,), type.float_)
    direction_mode_shape_dtype = jax.ShapeDtypeStruct(
        (grid_cell_1dcount,), type.int_)
    line_from_user_shape_dtype = jax.ShapeDtypeStruct(
        (grid_cell_1dcount, 2), type.float_)

    start = time.perf_counter()
    prepare_grid_data_for_second_run_jit = jax.jit(
        prepare_grid_data_for_second_run, static_argnums=(1, 2), device=device)
    prepare_grid_data_for_second_run_lowered = \
        prepare_grid_data_for_second_run_jit.lower(
            grid_sines_shape_dtype,
            grid_sqr_side_cell_count,
            grid_cell_2dcount,
            perimeter_count_shape_dtype,
            nozzle_width_shape_dtype,
            sdf_shape_dtype,
            direction_mode_shape_dtype,
            line_from_user_shape_dtype)
    prepare_grid_data_for_second_run_compiled = \
        prepare_grid_data_for_second_run_lowered.compile()
    stop = time.perf_counter()

    return prepare_grid_data_for_second_run_compiled, stop - start


def write_str_in_file_and_print(str_val: str, file: TextIOWrapper):
    print(str_val)
    file.write(str_val+'\n')


class CompilationTime:
    def __init__(self):
        self.init_grid_data = 0.
        self.shape_grid_data_sqr = 0.
        self.multigrid_align = 0.
        self.prepare_grid_data_for_second_run = 0.
        self.gabor_filter_grid_eval = 0.
        self.scalar_grid2_contour = 0.
        self.cycle_from_graph = 0.
        self.neighboring_edge_with_minimum_patching_energy = 0.
        self.cycle_edge_with_minimum_patching_energy = 0.
        self.cycle_stitch_two_edges = 0.
        self.repulse_points_n_times = 0.
        self.tangent_distance_to_neighbors = 0.
        self.cycle_to_polyline = 0.
        self.total = 0.

    def compute_total_time(self) -> None:
        # Some attributes are missing
        res = 0.
        res += self.init_grid_data
        res += self.shape_grid_data_sqr
        res += self.multigrid_align
        res += self.prepare_grid_data_for_second_run
        res += self.gabor_filter_grid_eval
        res += self.scalar_grid2_contour
        res += self.cycle_from_graph
        res += self.neighboring_edge_with_minimum_patching_energy
        res += self.cycle_edge_with_minimum_patching_energy
        res += self.cycle_stitch_two_edges
        res += self.repulse_points_n_times
        res += self.tangent_distance_to_neighbors
        res += self.cycle_to_polyline
        self.total = res

    def __str__(self):
        res = str()
        res += f"multigrid_align: {self.multigrid_align} s\n"
        res += f"gabor_filter_grid_eval: {self.gabor_filter_grid_eval} s\n"
        res += f"scalar_grid2_contour: {self.scalar_grid2_contour} s\n"
        res += f"cycle_from_graph: {self.cycle_from_graph} s\n"
        res += "neighboring_edge_with_minimum_patching_energy: "
        res += f"{self.neighboring_edge_with_minimum_patching_energy} s\n"
        res += f"cycle_edge_with_minimum_patching_energy: "
        res += f"{self.cycle_edge_with_minimum_patching_energy} s\n"
        res += f"cycle_stitch_two_edges: {self.cycle_stitch_two_edges} s\n"
        res += f"repulse_points_n_times: {self.repulse_points_n_times} s\n"
        res += f"tangent_distance_to_neighbors: "
        res += f"{self.tangent_distance_to_neighbors} s\n"
        res += f"cycle_to_polyline: {self.cycle_to_polyline} s\n"
        res += f"Total: {self.total} s"
        return res


class CompileFunctionParam():
    def __init__(
            self,
            fdm_aa_param: Parameters,
            shape_domain_grid: grid.Grid,
            shape_domain_grid_sqr: grid.Grid,
            trajectory_grid: grid.Grid,
            cycle_polyline_shape_dtype: Polyline,
            device_cpu: xla_client.Device,
            device_gpu: xla_client.Device):
        self.alignment_iter_per_level = fdm_aa_param.alignment_iter_per_level
        self.dimension_count = int(shape_domain_grid_sqr.cell_ndcount.shape[0])
        self.shape_domain_grid_sqr_cell_side_count = int(
            shape_domain_grid_sqr.cell_ndcount[0])
        self.shape_domain_grid_cell_ndcount_tuple = \
            tuple(device_get(shape_domain_grid.cell_ndcount))
        self.shape_domain_grid_cell_1dcount = \
            int(cell.count1_from_ndcount(shape_domain_grid.cell_ndcount))
        # One point per trajectory grid edge
        self.cycle_masked_point_count = int(
            edge.count1_from_cell_2dcount(trajectory_grid.cell_ndcount))
        self.trajectory_grid_cell_ndcount = tuple(
            device_get(trajectory_grid.cell_ndcount))
        self.cycle_count_max = self.cycle_masked_point_count // 4
        self.repulse_curves = fdm_aa_param.repulse_curves
        self.repulse_iter = 8
        self.neighborhood_radius = 3
        self.cycle_polyline_shape_dtype = cycle_polyline_shape_dtype
        self.device_cpu = device_cpu
        self.device_gpu = device_gpu


class CompiledFunctions():
    def __init__(self):
        self.init_grid_data = None
        self.shape_grid_data_sqr = None
        self.sine_wave_multigrid_align = None
        self.prepare_grid_data_for_second_run = None
        self.gabor_filter_grid_eval = None
        self.scalar_grid2_contour = None
        self.cycle_create_from_graph = None
        self.cycle_neighboring_edge_with_minimum_patching_energy = None
        self.cycle_edge_with_minimum_patching_energy = None
        self.cycle_stitch_two_edges = None
        self.repulse_points_n_times = None
        self.tangent_distance_to_neighbors = None
        self.cycle_to_polyline = None


def compile_functions(
        param: CompileFunctionParam,
        log_file: TextIOWrapper) -> tuple[CompiledFunctions, CompilationTime]:

    print("\nCOMPILATION STARTS\n")

    compiled_functions = CompiledFunctions()
    compilation_times = CompilationTime()

    init_grid_data_compiled, compilation_time = \
        compile_init_grid_data(
            param.shape_domain_grid_cell_1dcount, param.device_gpu)

    compiled_functions.init_grid_data = init_grid_data_compiled
    compilation_times.init_grid_data = compilation_time

    str_tmp = "init_grid_data compilation took "
    str_tmp += str(compilation_times.init_grid_data) + " s"
    write_str_in_file_and_print(str_tmp, log_file)

    shape_grid_data_sqr_compiled, compilation_time = \
        compile_shape_grid_data_sqr(
            param.shape_domain_grid_cell_ndcount_tuple, param.device_gpu)

    compiled_functions.shape_grid_data_sqr = shape_grid_data_sqr_compiled
    compilation_times.shape_grid_data_sqr = compilation_time

    str_tmp = "shape_grid_data_sqr compilation took "
    str_tmp += str(compilation_times.shape_grid_data_sqr) + " s"
    write_str_in_file_and_print(str_tmp, log_file)

    sine_wave_multigrid_align_compiled, compilation_time = \
        sine_wave.multigrid_compile_align(
            param.shape_domain_grid_sqr_cell_side_count,
            param.dimension_count,
            param.alignment_iter_per_level,
            param.device_gpu)

    compiled_functions.sine_wave_multigrid_align = \
        sine_wave_multigrid_align_compiled
    compilation_times.multigrid_align = compilation_time

    str_tmp = "sine_wave.multigrid_align compilation took "
    str_tmp += str(compilation_times.multigrid_align) + " s"
    write_str_in_file_and_print(str_tmp, log_file)

    prepare_grid_data_for_second_run_compiled, compilation_time = \
        compile_prepare_grid_data_for_second_run(
            param.shape_domain_grid_cell_ndcount_tuple,
            param.shape_domain_grid_sqr_cell_side_count,
            param.dimension_count,
            param.device_gpu)

    compiled_functions.prepare_grid_data_for_second_run = \
        prepare_grid_data_for_second_run_compiled
    compilation_times.prepare_grid_data_for_second_run = compilation_time

    str_tmp = "sine_wave.prepare_grid_data_for_second_run compilation took "
    str_tmp += str(compilation_times.prepare_grid_data_for_second_run) + " s"
    write_str_in_file_and_print(str_tmp, log_file)

    gabor_filter_grid_eval_compiled, compilation_time = \
        gabor_filter.grid_compile_eval(
            param.shape_domain_grid_cell_1dcount,
            param.shape_domain_grid_sqr_cell_side_count,
            param.dimension_count,
            param.device_gpu)

    compiled_functions.gabor_filter_grid_eval = gabor_filter_grid_eval_compiled
    compilation_times.gabor_filter_grid_eval = compilation_time

    str_tmp = "gabor_filter.grid_eval compilation took "
    str_tmp += str(compilation_times.gabor_filter_grid_eval) + " s"
    write_str_in_file_and_print(str_tmp, log_file)

    scalar_grid2_contour_compiled, compilation_time = \
        scalar.grid2_compile_contour(
            param.shape_domain_grid_cell_ndcount_tuple, param.device_gpu)

    compiled_functions.scalar_grid2_contour = scalar_grid2_contour_compiled
    compilation_times.scalar_grid2_contour = compilation_time

    str_tmp = "scalar.grid2_contour compilation took "
    str_tmp += str(compilation_times.scalar_grid2_contour) + " s"
    write_str_in_file_and_print(str_tmp, log_file)

    cycle_create_from_graph_compiled, compilation_time = \
        cycle.compile_create_from_graph(
            param.cycle_masked_point_count,
            param.cycle_count_max,
            param.device_cpu)

    compiled_functions.cycle_create_from_graph = \
        cycle_create_from_graph_compiled
    compilation_times.cycle_from_graph = compilation_time

    str_tmp = "cycle.create_from_graph compilation took "
    str_tmp += str(compilation_times.cycle_from_graph) + " s"
    write_str_in_file_and_print(str_tmp, log_file)

    cycle_neighboring_edge_with_minimum_patching_energy_compiled, \
        compilation_time = \
        cycle.compile_neighboring_edge_with_minimum_patching_energy(
            param.cycle_masked_point_count,
            param.device_cpu,
            param.cycle_count_max)

    compiled_functions.cycle_neighboring_edge_with_minimum_patching_energy = \
        cycle_neighboring_edge_with_minimum_patching_energy_compiled
    compilation_times.neighboring_edge_with_minimum_patching_energy = \
        compilation_time

    str_tmp = "cycle.compile_neighboring_edge_with_minimum_patching_energy "
    str_tmp += "took "
    str_tmp += \
        str(compilation_times.neighboring_edge_with_minimum_patching_energy)
    str_tmp += " s"
    write_str_in_file_and_print(str_tmp, log_file)

    cycle_edge_with_minimum_patching_energy_compiled, compilation_time = \
        cycle.compile_edge_with_minimum_patching_energy(
            param.cycle_masked_point_count,
            param.device_cpu,
            param.cycle_count_max)

    compiled_functions.cycle_edge_with_minimum_patching_energy = \
        cycle_edge_with_minimum_patching_energy_compiled
    compilation_times.cycle_edge_with_minimum_patching_energy = \
        compilation_time

    str_tmp = "cycle.edge_with_minimum_patching_energy compilation took "
    str_tmp += \
        str(compilation_times.cycle_edge_with_minimum_patching_energy)
    str_tmp += " s"
    write_str_in_file_and_print(str_tmp, log_file)

    cycle_stitch_two_edges_compiled, compilation_time = \
        cycle.compile_stitch_two_edges(
            param.cycle_masked_point_count,
            param.device_cpu,
            param.cycle_count_max)

    compiled_functions.cycle_stitch_two_edges = cycle_stitch_two_edges_compiled
    compilation_times.cycle_stitch_two_edges = compilation_time

    str_tmp = "cycle_stitch_two_edges compilation took "
    str_tmp += str(compilation_times.cycle_stitch_two_edges) + " s"
    write_str_in_file_and_print(str_tmp, log_file)

    if param.repulse_curves:
        repulse_points_n_times_compiled, compilation_time = \
            point_data.grid2_compile_repulse_points_n_times(
                param.trajectory_grid_cell_ndcount,
                param.device_gpu,
                param.repulse_iter)

        compiled_functions.repulse_points_n_times = repulse_points_n_times_compiled
        compilation_times.repulse_points_n_times = compilation_time

        str_tmp = "grid2_compile_repulse_points_n_times compilation took "
        str_tmp += str(compilation_times.repulse_points_n_times) + " s"
        write_str_in_file_and_print(str_tmp, log_file)

    tangent_distance_to_neighbors_compiled, compilation_time = \
        cycle.compile_points_tangent_half_distance_to_neighboring_segments_x(
            param.cycle_masked_point_count,
            param.device_gpu,
            param.neighborhood_radius,
            param.cycle_count_max)

    compiled_functions.tangent_distance_to_neighbors = \
        tangent_distance_to_neighbors_compiled
    compilation_times.tangent_distance_to_neighbors = compilation_time

    str_tmp = "tangent_distance_to_neighbors compilation took "
    str_tmp += str(compilation_times.tangent_distance_to_neighbors) + " s"
    write_str_in_file_and_print(str_tmp, log_file)

    cycle_to_polyline_compiled, compilation_time = \
        cycle.compile_to_polyline(
            param.cycle_masked_point_count,
            param.cycle_polyline_shape_dtype,
            param.device_cpu,
            param.cycle_count_max)

    compiled_functions.cycle_to_polyline = cycle_to_polyline_compiled
    compilation_times.cycle_to_polyline = compilation_time

    str_tmp = "cycle.to_polyline compilation took "
    str_tmp += str(compilation_times.cycle_to_polyline) + " s"
    write_str_in_file_and_print(str_tmp, log_file)

    compilation_times.compute_total_time()

    tmp_str = f"Compilation took {compilation_times.total}"
    print(tmp_str)
    log_file.write(tmp_str)

    return compiled_functions, compilation_times


class ExecutionTime:
    def __init__(self):
        self.cell_point_computation = 0.
        self.boundary_cycles_from_svg_paths = 0.
        self.compute_sdf = 0.
        self.init_grid_data = 0.
        self.sine_wave_multigrid_align = 0.
        self.prepare_grid_data_for_second_run = 0.
        self.sine_wave_multigrid_align_round2 = 0.
        self.gabor_filter_grid_eval = 0.
        self.set_scalar_field_boundary = 0.
        self.scalar_grid2_contour = 0.
        self.cycle_create_from_graph = 0.
        self.match_and_stitch = 0.
        self.repulsion = 0.
        self.tangent_distance_to_neighbors = 0.
        self.radius_concatenation = 0.
        self.cycle_to_polyline = 0.
        self.total = 0.

    def compute_total_time(self) -> None:
        res = 0.
        res += self.init_grid_data
        res += self.sine_wave_multigrid_align
        res += self.prepare_grid_data_for_second_run
        res += self.sine_wave_multigrid_align_round2
        res += self.gabor_filter_grid_eval
        res += self.set_scalar_field_boundary
        res += self.scalar_grid2_contour
        res += self.cycle_create_from_graph
        res += self.match_and_stitch
        res += self.repulsion
        res += self.tangent_distance_to_neighbors
        res += self.radius_concatenation
        res += self.cycle_to_polyline
        self.total = res

    def __str__(self):
        # TODO
        res = str()
        res += f"cell_point_computation: {self.cell_point_computation} s\n"
        res += f"Total: {self.total} s"
        return res


class ShapeDomainGridData:
    def __init__(self):
        self.grid = None
        self.cell_ndcount_tuple = None
        self.cell_center_points = None
        self.cell_center_points_jittered = None
        # ccpj: cell center points jittered
        self.ccpj_signed_distance_from_boundary = None
        self.ccpj_closest_boundary_normal = None
        self.ccpj_data = None

    def save(self, path):
        cell_count = np.array(self.grid.cell_ndcount)
        origin = np.array(self.grid.origin)
        cell_sides_length = np.array(self.grid.cell_sides_length)
        cell_center_points = np.array(self.cell_center_points)
        cell_center_points_jittered = np.array(
            self.cell_center_points_jittered)
        ccpj_signed_distance_from_boundary = np.array(
            self.ccpj_signed_distance_from_boundary)
        ccpj_closest_boundary_normal = np.array(
            self.ccpj_closest_boundary_normal)
        np.savez(
            path,
            cell_count=cell_count,
            origin=origin,
            cell_sides_length=cell_sides_length,
            cell_center_points=cell_center_points,
            cell_center_points_jittered=cell_center_points_jittered,
            ccpj_signed_distance_from_boundary=ccpj_signed_distance_from_boundary,
            ccpj_closest_boundary_normal=ccpj_closest_boundary_normal)

    def load(self, path):
        data = np.load(path)
        cell_count = data['cell_count']
        origin = data['origin']
        cell_sides_length = data['cell_sides_length']
        cell_center_points = data['cell_center_points']
        cell_center_points_jittered = data['cell_center_points_jittered']
        ccpj_signed_distance_from_boundary = data['ccpj_signed_distance_from_boundary']
        ccpj_closest_boundary_normal = data['ccpj_closest_boundary_normal']

        self.grid = grid.Grid(cell_count, origin, cell_sides_length)
        self.cell_center_points = cell_center_points
        self.cell_center_points_jittered = cell_center_points_jittered
        self.ccpj_signed_distance_from_boundary = ccpj_signed_distance_from_boundary
        self.ccpj_closest_boundary_normal = ccpj_closest_boundary_normal

    def device_put(self, device):
        self.grid = jax.device_put(self.grid)
        self.cell_center_points = jax.device_put(
            self.cell_center_points, device)
        self.cell_center_points_jittered = jax.device_put(
            self.cell_center_points_jittered, device)
        self.ccpj_signed_distance_from_boundary = jax.device_put(
            self.ccpj_signed_distance_from_boundary, device)
        self.ccpj_closest_boundary_normal = jax.device_put(
            self.ccpj_closest_boundary_normal, device)
        self.ccpj_data = jax.device_put(self.ccpj_data, device)
