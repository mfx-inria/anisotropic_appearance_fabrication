"""
Fill a 2D shape with an oriented cycle. See Algorithm 1 in Chermain et al.
2023.
"""

import argparse
import gc
import os
import time

import jax.numpy as jnp
import jax.random
import numpy as np
from jax import block_until_ready, device_get, device_put
from svgpathtools import svg2paths

import cglib.backend
import cglib.cycle
import cglib.fdm_aa
import cglib.gabor_filter
import cglib.grid
import cglib.grid.cell
import cglib.grid.edge
import cglib.limits
import cglib.math
import cglib.point_data
import cglib.polyline
import cglib.scalar
import cglib.sine_wave
import cglib.texture
import cglib.tree_util
import cglib.type

# Estimation of the number of polylines in the output of the algorithm. It is
# one, as the article's objective is to have a closed polyline filling a 2D
# shape. If the stitching step is skipped, this value needs to be increased.
POLYLINE_COUNT_ESTIMATION = 1

# If true, the starting point of the cycle will be the cycle's leftest point.
START_AT_LEFT = True


def run():

    device_cpu, device_gpu = cglib.backend.get_cpu_and_gpu_devices()

    parser = argparse.ArgumentParser(
        description='Input: A json file containing all the input parameters. Output: A cycle with varying width.'
    )
    parser.add_argument("input_filename", help="Use the inputs in the file specified here.")
    parser.add_argument("--nodumping", help="To not save intermediate data.")
    args = parser.parse_args()

    input_params_filename = args.input_filename
    dump_intermediate_data = not args.nodumping

    parameters = cglib.fdm_aa.Parameters()
    parameters.load(input_params_filename)

    log_file = open(parameters.log_filename, 'w', encoding="utf-8")

    nozzle_width_derived_param = \
        cglib.fdm_aa.compute_nozzle_width_derived_parameters(
            parameters.nozzle_width,
            parameters.layer_height_rt_nozzle_width)

    # Load svg paths
    # Transformations inside the SVG are ignored, so only SVG files without
    # transformations are valid. Using Inkscape, ensure your contour is not
    # associated with a layer to avoid implicit transformations. Be sure that
    # the contour is a clockwise-oriented closed polyline. Holes are
    # represented with counter-clockwise oriented closed polylines.
    paths, _, svg_attributes = svg2paths(
        parameters.svg_path, return_svg_attributes=True)

    # The shape domain size is determined by the SVG width and heigh
    # -2: remove the unit
    svg_width = float(svg_attributes['width'][:-2])
    svg_height = float(svg_attributes['height'][:-2])
    # [x, y]
    shape_domain_size = np.array([svg_width, svg_height])

    # There are three different grids used to do the computation.
    # 1. The first, i.e., `shape_domain_grid`, discretizes the 2D shape domain
    #    and is used by the sine wave evaluation algorithm and scalar field
    #    contouring algorithm.
    # 2. The second, i.e., `shape_domain_grid_sqr`, is `shape_domain_grid`
    #    round up to the next power of two cell 2D count. It is used by the
    #    sine wave aligning algorithm. The power of two is required for the
    #    multigrid build.
    # 3. The third grid, i.e., `trajectory_grid``, is the grid implicitly
    #    returned by the contouring algorithm. This grid has each edge
    #    associated with nothing or a unique contour point.
    shape_domain_grid = cglib.fdm_aa.discretize_2dshape_domain_with_cells(
        parameters.nozzle_width, shape_domain_size)
    shape_domain_grid: cglib.grid.Grid = device_put(
        shape_domain_grid, device_cpu)
    shape_domain_grid_sqr = cglib.grid.roundup_power_of_2(shape_domain_grid)
    trajectory_grid = cglib.scalar.grid2_contour_get_output_grid(
        shape_domain_grid)

    str_tmp = f"\nInput filename: {input_params_filename}\n\n"
    str_tmp += f"shape_domain_grid: {shape_domain_grid}\n"
    str_tmp += f"shape_domain_grid_sqr: {shape_domain_grid_sqr}"
    cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)

    # Each point will be associated with a line (2 floats), a phase (1 float),
    # and a constraint (1 float).
    POINT_DATA_SIZE = 4

    cycle_polyline, cycle_polyline_shape_dtype = cglib.polyline.create_from_2dgrid(
        trajectory_grid.cell_ndcount,
        device_cpu,
        POLYLINE_COUNT_ESTIMATION,
        POINT_DATA_SIZE)

    # We have to compute the points associated with each cell of the shape
    # domain grid.
    # Then, we must compute the signed distance from the boundary for the
    # points and the closest normal on the boundary.
    shape_domain_grid_data = cglib.fdm_aa.ShapeDomainGridData()

    parameters.create_SDF_filename(shape_domain_grid)
    sdf_exist = os.path.exists(parameters.sdf_filename)
    do_sdf_computation = not sdf_exist or parameters.force_sdf_computation
    execution_times = cglib.fdm_aa.ExecutionTime()
    # If the sdf is not cached or if the user forces the SDF computation
    if do_sdf_computation:

        str_tmp = f"shape_domain_grid_data compute cell points started"
        cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)

        start = time.perf_counter()

        shape_domain_grid_data.grid = shape_domain_grid
        shape_domain_grid_data.cell_center_points = \
            cglib.grid.cell.center_points(shape_domain_grid)
        jitter_distance = \
            nozzle_width_derived_param.domain_grid_points_perturbation_width / \
            (parameters.nozzle_width * 0.5)
        seed_jax = jax.random.PRNGKey(parameters.seed)
        shape_domain_grid_data.cell_center_points_jittered = \
            cglib.grid.cell.center_points_jittered(
                shape_domain_grid,
                seed_jax,
                jitter_distance)
        stop = time.perf_counter()
        execution_times.cell_point_computation += stop - start

        print("compute_boundary_cycles_from_svg_paths started")
        start = time.perf_counter()
        boundary_cycles = cglib.fdm_aa.compute_boundary_cycles_from_svg_paths(
            shape_domain_size, paths, parameters.nozzle_width, device_cpu)
        stop = time.perf_counter()
        execution_times.boundary_cycles_from_svg_paths = stop - start

        str_tmp = f"compute_boundary_cycles_from_svg_paths took {stop - start} s"
        print(str_tmp)
        log_file.write(str_tmp+'\n')

        # Pack each cycle into one polygonal data in order to compute signed
        # distance thereafter
        boundary_polydata, boundary_normal_glyph = cglib.fdm_aa.cycle_to_polydata(
            boundary_cycles, nozzle_width_derived_param.layer_height)
        boundary_polydata.save(parameters.boundary_polydata_filename)
        boundary_normal_glyph.save(parameters.boundary_normal_glyph_filename)

        # Compute signed distance from boundary. More precisely, compute domain grid
        # perturbed points distance from boundary, and sign the distance using
        # nearest boundary normals
        # Return also the normals
        str_tmp = f"compute_signed_distance_from_boundary started"
        cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)

        start = time.perf_counter()

        signed_distance_from_boundary, closest_boundary_normal = \
            cglib.fdm_aa.compute_signed_distance_from_boundary(
                shape_domain_grid_data.cell_center_points_jittered,
                boundary_polydata)
        # Group data in the class
        shape_domain_grid_data.ccpj_signed_distance_from_boundary = \
            signed_distance_from_boundary
        shape_domain_grid_data.ccpj_closest_boundary_normal = \
            closest_boundary_normal[:, :2]

        stop = time.perf_counter()
        execution_times.compute_sdf = stop - start

        str_tmp = "compute_signed_distance_from_boundary took "
        str_tmp += f"{execution_times.compute_sdf} s"
        cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)
        shape_domain_grid_data.save(parameters.sdf_filename)
    else:
        shape_domain_grid_data.load(parameters.sdf_filename)

    # Convert from ndarray to tuple because a tuple is hashable, a property
    # needed by the compilation process.
    shape_domain_grid_data.cell_ndcount_tuple = tuple(
        device_get(shape_domain_grid_data.grid.cell_ndcount))

    borders = cglib.fdm_aa.create_borders(
        parameters.nozzle_width, parameters.perimeter_count)

    # Load the image textures
    line_field_tex, line_mode_field_tex = \
        cglib.fdm_aa.load_line_field_textures(
            parameters.line_field_path,
            parameters.line_mode_field_path,
            device_cpu)
    # Convert them to useable data
    direction, direction_mode = cglib.fdm_aa.eval_textures(
        shape_domain_grid_data.cell_center_points_jittered,
        line_mode_field_tex,
        line_field_tex,
        shape_domain_size,
        cglib.texture.InterpolationType.NEAREST)

    # Put everything on GPU
    direction = device_put(direction, device_gpu)
    direction_mode = device_put(direction_mode, device_gpu)
    shape_domain_grid_data.device_put(device_gpu)

    compile_functions_param = cglib.fdm_aa.CompileFunctionParam(
        parameters,
        shape_domain_grid,
        shape_domain_grid_sqr,
        trajectory_grid,
        cycle_polyline_shape_dtype,
        device_cpu,
        device_gpu)

    compiled_functions, _ = \
        cglib.fdm_aa.compile_functions(compile_functions_param, log_file)
    
    print("\nEXECUTION STARTS\n")

    # Init lines, phases and constraints based on inputs
    start = time.perf_counter()
    # cglib.fdm_aa.init_grid_data(shape_domain_grid_data.ccpj_signed_distance_from_boundary,
    #     shape_domain_grid_data.cell_center_points_jittered,
    #     direction_mode,
    #     parameters.perimeter_count,
    #     shape_domain_grid_data.ccpj_closest_boundary_normal,
    #     parameters.nozzle_width,
    #     direction)
    res = compiled_functions.init_grid_data(
        shape_domain_grid_data.ccpj_signed_distance_from_boundary,
        shape_domain_grid_data.cell_center_points_jittered,
        direction_mode,
        parameters.perimeter_count,
        shape_domain_grid_data.ccpj_closest_boundary_normal,
        parameters.nozzle_width,
        direction)

    shape_domain_grid_data.cell_center_points_jittered = res[0]
    shape_domain_grid_data.ccpj_data = res[1]
    one_run = res[2]

    # Square the grid and its data
    grid_sines = compiled_functions.shape_grid_data_sqr(
        shape_domain_grid_data.cell_center_points_jittered,
        shape_domain_grid_data.ccpj_data,
        shape_domain_grid_data.grid
    )
    grid_sines = block_until_ready(grid_sines)
    stop = time.perf_counter()
    execution_times.init_grid_data = stop - start
    str_tmp = f"Init grid data took {execution_times.init_grid_data} s"
    cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)

    # Align sines
    str_tmp = f"Align sines started"
    cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)
    start = time.perf_counter()
    grid_sines_aligned: cglib.point_data.GridPointData = \
        compiled_functions.sine_wave_multigrid_align(
            grid_sines, nozzle_width_derived_param.frequency)
    grid_sines_aligned = block_until_ready(grid_sines_aligned)
    stop = time.perf_counter()

    execution_times.sine_wave_multigrid_align = stop - start
    str_tmp = f"Align sines took {execution_times.sine_wave_multigrid_align} s"
    cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)

    # If needed, realigned
    if not one_run:

        start = time.perf_counter()
        grid_sines = compiled_functions.prepare_grid_data_for_second_run(
            grid_sines_aligned,
            parameters.perimeter_count,
            parameters.nozzle_width,
            shape_domain_grid_data.ccpj_signed_distance_from_boundary,
            direction_mode,
            direction)
        grid_sines = block_until_ready(grid_sines)
        stop = time.perf_counter()
        execution_times.prepare_grid_data_for_second_run = stop - start
        str_tmp = "Prepare data for 2nd run took "
        str_tmp += f"{execution_times.prepare_grid_data_for_second_run} s"
        cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)

        # Align sines second time
        start = time.perf_counter()
        grid_sines_aligned: cglib.point_data.GridPointData = \
            compiled_functions.sine_wave_multigrid_align(
                grid_sines, nozzle_width_derived_param.frequency)
        grid_sines_aligned = block_until_ready(grid_sines_aligned)
        stop = time.perf_counter()
        time_tmp = stop - start
        execution_times.sine_wave_multigrid_align_round2 += time_tmp
        str_tmp = f"Align sines took (2nd round) {time_tmp} s"
        cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)

    del grid_sines
    gc.collect()

    # Save grid of sines aligned
    if dump_intermediate_data:
        cglib.point_data.grid_save(
            parameters.grid_sines_aligned_filename, grid_sines_aligned)

    # Evaluate the sine values at slice grid perturbed points
    start = time.perf_counter()
    scalar_field = \
        compiled_functions.gabor_filter_grid_eval(
            shape_domain_grid_data.cell_center_points,
            nozzle_width_derived_param.frequency,
            grid_sines_aligned)
    scalar_field = block_until_ready(scalar_field)
    stop = time.perf_counter()
    execution_times.gabor_filter_grid_eval = stop - start
    str_tmp = "gabor_field_value_with_mask_x took "
    str_tmp += f"{execution_times.gabor_filter_grid_eval} s"
    cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)

    start = time.perf_counter()

    del grid_sines_aligned
    gc.collect()

    # Put max sine values at the exterior of the shape
    scalar_field = jnp.where(
        shape_domain_grid_data.ccpj_signed_distance_from_boundary >= borders.perimeter_shifted_outside,
        shape_domain_grid_data.ccpj_signed_distance_from_boundary *
        2. / parameters.nozzle_width + 1.,
        scalar_field)
    scalar_field = jnp.where(
        shape_domain_grid_data.ccpj_signed_distance_from_boundary >= 0., 1., scalar_field)

    stop = time.perf_counter()
    time_tmp = stop - start
    execution_times.set_scalar_field_boundary += time_tmp

    # Dump scalar field
    if dump_intermediate_data:
        shape_domain_grid_cell_ndcount_tuple = tuple(
            device_get(shape_domain_grid.cell_ndcount))
        scalar_field_2dshape = scalar_field.reshape(
            shape_domain_grid_cell_ndcount_tuple)
        jnp.save(parameters.scalar_field_filename, scalar_field_2dshape)

    # Compute the contour of the sine field
    start = time.perf_counter()
    contour: cglib.point_data.PointData = \
        compiled_functions.scalar_grid2_contour(
            scalar_field, shape_domain_grid)
    contour = block_until_ready(contour)
    contour_cpu = device_put(contour, device=device_cpu)
    del scalar_field
    del contour
    gc.collect()
    stop = time.perf_counter()
    execution_times.scalar_grid2_contour = stop - start
    str_tmp = "scalar_grid2_contour took "
    str_tmp += f"{execution_times.scalar_grid2_contour} s"
    cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)

    if dump_intermediate_data:
        cglib.point_data.save(
            parameters.contour_graph_filename, device_get(contour_cpu))

    start = time.perf_counter()
    cycles_cpu: cglib.cycle.Cycle = compiled_functions.cycle_create_from_graph(
        contour_cpu)
    cycles_cpu = block_until_ready(cycles_cpu)
    stop = time.perf_counter()
    execution_times.cycle_create_from_graph = stop - start
    str_tmp = "cycles_create_from_graph took "
    str_tmp += f"{execution_times.cycle_create_from_graph} s"
    cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)

    if dump_intermediate_data:
        cglib.cycle.save(parameters.cycles_filename, cycles_cpu)

    # Cycles: match and stitch
    start = time.perf_counter()
    for _ in range(cycles_cpu.cycle_count-1):
        # cycles_data[:, 1]: Cycle edge count
        # Sort with increasing cycle edge count
        cycle_id_with_min_edge_count = jnp.argmin(cycles_cpu.cycle_data[:, 1])
        best_edge_pair, best_edge_cost = \
            compiled_functions.cycle_neighboring_edge_with_minimum_patching_energy(
                cycle_id_with_min_edge_count,
                cycles_cpu,
                trajectory_grid.cell_ndcount)
        no_neighboring_cycle = best_edge_cost == cglib.limits.FLOAT_MAX
        if no_neighboring_cycle:
            print("no neighboring cycle (different connex component)")
            best_edge_pair, best_edge_cost = \
                compiled_functions.cycle_edge_with_minimum_patching_energy(
                    cycle_id_with_min_edge_count, cycles_cpu)
        cycles_cpu: cglib.cycle.Cycle = \
            compiled_functions.cycle_stitch_two_edges(
                best_edge_pair[0], best_edge_pair[1], cycles_cpu)
        # Debug
    cycles_cpu = block_until_ready(cycles_cpu)
    stop = time.perf_counter()
    execution_times.match_and_stitch = stop - start
    str_tmp = "cycles_match_and_stitch took "
    str_tmp += f"{execution_times.match_and_stitch} s"
    cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)

    if START_AT_LEFT:
        tmp = jnp.where(jnp.isnan(cycles_cpu.point_data.point[:, :]),
                        cglib.limits.FLOAT_MAX,
                        cycles_cpu.point_data.point[:, :])
        leftest_point_id = jnp.argmin(tmp[:, 0])
        cycle_id_with_min_edge_count = jnp.argmin(cycles_cpu.cycle_data[:, 1])
        cycles_cpu = cglib.cycle.Cycle(
            cglib.point_data.PointData(
                cycles_cpu.point_data.point, cycles_cpu.point_data.data),
            cycles_cpu.cycle_data.at[cycle_id_with_min_edge_count, 0].set(
                leftest_point_id),
            cycles_cpu.cycle_count)


    # START REPULSION
    if parameters.repulse_curves:

        start = time.perf_counter()
        cycle_point_data = cglib.point_data.PointData(
            cycles_cpu.point_data.point, cycles_cpu.point_data.data[:, :2])
        arg0 = device_put(cycle_point_data, device=device_gpu)
        arg2 = device_put(trajectory_grid, device=device_gpu)
        arg3 = device_put(parameters.nozzle_width*0.5, device=device_gpu)
        arg4 = device_put(
            np.full((cycles_cpu.point_data.point.shape[0],), False), device=device_gpu)
        cycle_point_data = compiled_functions.repulse_points_n_times(
            arg0,
            arg2,
            arg3,
            arg4)
        cycle_point_data = block_until_ready(cycle_point_data)
        cycle_point_data = device_put(cycle_point_data, device=device_cpu)

        cycles_cpu = device_put(
            cglib.cycle.Cycle(
                cglib.point_data.PointData(
                    cycle_point_data.point,
                    cycles_cpu.point_data.data),
                cycles_cpu.cycle_data,
                cycles_cpu.cycle_count),
            device=device_cpu)
        del arg0
        del arg4
        gc.collect()
        stop = time.perf_counter()
        execution_times.repulsion = stop - start
        str_tmp = "uniform_2dgrid_repulse_edge_points_n_times took "
        str_tmp += f"{execution_times.repulsion} s"
        cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)
    # END REPULSION

    start = time.perf_counter()
    cycles_only_adj = cglib.cycle.Cycle(
        cglib.point_data.PointData(
            cycles_cpu.point_data.point,
            cycles_cpu.point_data.data[:, :2]),
        cycles_cpu.cycle_data,
        cycles_cpu.cycle_count)
    cycles_gpu = device_put(cycles_only_adj, device=device_gpu)
    cycles_gpu = block_until_ready(cycles_gpu)
    min_radius_circle = compiled_functions.tangent_distance_to_neighbors(
        cycles_gpu, trajectory_grid)
    min_radius_circle = block_until_ready(min_radius_circle)
    del cycles_gpu
    gc.collect()
    min_radius_circle = device_put(min_radius_circle, device=device_cpu)
    stop = time.perf_counter()
    execution_times.tangent_distance_to_neighbors = stop - start
    str_tmp = "tangent_distance_to_neighbors took "
    str_tmp += f"{execution_times.tangent_distance_to_neighbors} s"
    cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)

    # Constant radius
    # min_radius_circle = jnp.full((cycles_cpu.points_data.point.shape[0]), parameters.nozzle_width*0.5)

    # Convert cycle data from uint to float
    start = time.perf_counter()

    cycles_cpu_points_data_data_float = cycles_cpu.point_data.data.astype(
        cglib.type.float_)
    # Append the minimum circle radius
    cycles_cpu_points_data_data_float = jnp.concatenate(
        (cycles_cpu_points_data_data_float,
         min_radius_circle.reshape((-1, 1))),
        axis=1)
    # Cycles with minimum circle radius per point
    cycles_cpu = cglib.cycle.Cycle(
        cglib.point_data.PointData(
            cycles_cpu.point_data.point,
            cycles_cpu_points_data_data_float),
        cycles_cpu.cycle_data,
        cycles_cpu.cycle_count)

    cycles_cpu = block_until_ready(cycles_cpu)
    stop = time.perf_counter()
    execution_times.radius_concatenation = stop - start

    str_tmp = f"Radius concatenation: {execution_times.radius_concatenation} s"
    cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)

    # Convert to polylines to remove the indirect accesses to points
    start = time.perf_counter()
    cycle_polyline = compiled_functions.cycle_to_polyline(
        cycles_cpu, cycle_polyline)
    cycle_polyline: cglib.polyline.Polyline = block_until_ready(cycle_polyline)
    stop = time.perf_counter()
    execution_times.cycle_to_polyline = stop - start
    str_tmp = "cycles_to_polylines took: "
    str_tmp += f"{execution_times.cycle_to_polyline} s"
    cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)

    del cycles_cpu
    gc.collect()

    execution_times.compute_total_time()
    str_tmp = "Total exec time (without sdf computation): "
    str_tmp += f"{execution_times.total} s"
    cglib.fdm_aa.write_str_in_file_and_print(str_tmp, log_file)
    
    # Point count per polyline
    point_count_per_polyline = np.array(
        jax.device_get(cycle_polyline.data[:, 1]))
    point_count_per_polyline_max = int(point_count_per_polyline.max())

    polyline_point_2d = np.array(jax.device_get(cycle_polyline.point))
    # Slice to remove most of the nan.
    polyline_point_2d = polyline_point_2d[:,:point_count_per_polyline_max,:]

    # At this point, the data associated with each point has 4 floats. The
    # first three are useless here and are removed. It was the adjacency
    # list (2 floats) and the cycle ID (1 float). The last float is the
    # radius of the trajectory.
    polyline_point_radius = np.array(
        jax.device_get(cycle_polyline.point_data[:, :, 3]))
    # Slice to remove most of the nan.
    polyline_point_radius = polyline_point_radius[
        :,:point_count_per_polyline_max]
    # Clamp radius
    polyline_point_radius = np.minimum(polyline_point_radius, np.full_like(
        polyline_point_radius, nozzle_width_derived_param.max_radius))
    polyline_point_radius = np.maximum(polyline_point_radius, np.full_like(
        polyline_point_radius, nozzle_width_derived_param.min_radius))
    
    cycle_data_final = point_count_per_polyline.reshape((-1, 1))
    cycle_data_final = np.concatenate(
        (cycle_data_final,
         np.full_like(cycle_data_final,
                      nozzle_width_derived_param.layer_height)),
        axis=1)
    cycle_polyline = cglib.polyline.Polyline(
        polyline_point_2d, polyline_point_radius, cycle_data_final)

    cglib.polyline.save(parameters.cycle_polyline_filename, cycle_polyline)
    log_file.close()


if __name__ == "__main__":
    run()
