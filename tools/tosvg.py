import argparse
from enum import Enum

import jax
import numpy as np
from svgpathtools import Line, Path, paths2svg, svg2paths

import jax.numpy as jnp

import cglib.backend
import cglib.cycle
import cglib.fdm_aa
import cglib.polyline

import cglib.transform as transform


class DataToExport(Enum):
    """
    Enumeration of the data that can be exported to svg.

    Attributes
    ----------
    CYCLE:
        The cycle after the stitching.
    CYCLES:
        The cycles after the contouring and before the stitching.
    """
    CYCLE = 'cycle'
    CYCLES = 'cycles'


def cycles_to_svg(input_param_filename, svg_out_filename):

    device_cpu, _ = cglib.backend.get_cpu_and_gpu_devices()

    parameters = cglib.fdm_aa.Parameters()
    parameters.load(input_param_filename)

    # Load svg paths to get svg attributes
    _, _, svg_attributes = svg2paths(
        parameters.svg_path, return_svg_attributes=True)

    cycles = cglib.cycle.load(parameters.cycles_filename)
    cycles = jax.device_put(cycles, device_cpu)
    print(f"Cycle count before stitching: {cycles.cycle_count}")
    polylines = cglib.cycle.to_polyline_full_nan(cycles)
    polylines = jax.jit(cglib.cycle.to_polyline)(cycles, polylines)
    polylines: cglib.polyline.Polyline = jax.device_get(polylines)
    paths = []

    for i in range(cycles.cycle_count):
        cycle_polyline_2dpoint = np.array(polylines.point[i])
        point_count_i = int(polylines.data[i, 1])
        cycle_polyline_2dpoint = cycle_polyline_2dpoint[:point_count_i, :]

        path_i = Path()

        for j in range(1, point_count_i):
            path_i.append(Line(
                complex(cycle_polyline_2dpoint[j-1, 0],
                        cycle_polyline_2dpoint[j-1, 1]),
                complex(cycle_polyline_2dpoint[j, 0], cycle_polyline_2dpoint[j, 1])))
        # Add last point
        path_i.append(Line(
            complex(cycle_polyline_2dpoint[point_count_i-1, 0],
                    cycle_polyline_2dpoint[point_count_i-1, 1]),
            complex(cycle_polyline_2dpoint[0, 0], cycle_polyline_2dpoint[0, 1])))
        paths.append(path_i)
    paths2svg.wsvg(paths, filename=svg_out_filename,
                   svg_attributes=svg_attributes)


def cycle_to_svg(input_param_filename, svg_out_filename):

    device_cpu, _ = cglib.backend.get_cpu_and_gpu_devices()

    parameters = cglib.fdm_aa.Parameters()
    parameters.load(input_param_filename)
    cycle_polyline = cglib.polyline.load(parameters.cycle_polyline_filename)
    cycle_polyline_2dpoint: np.ndarray = cycle_polyline.point[0]
    cycle_polyline_2dpoint = jax.device_put(cycle_polyline_2dpoint, device_cpu)

    # Load svg paths to get svg attributes
    _, _, svg_attributes = svg2paths(
        parameters.svg_path, return_svg_attributes=True)
    
    # The shape domain size is determined by the SVG width and heigh
    # -2: remove the unit
    svg_width = float(svg_attributes['width'][:-2])
    svg_height = float(svg_attributes['height'][:-2])
    # [x, y]
    shape_domain_size = jnp.array([svg_width, svg_height])

    trans = transform.translate(jnp.array([0., -shape_domain_size[1]*0.5]))
    scale = transform.scale(jnp.array([1., -1.]))
    upside_down_to_right_side_up = jnp.linalg.inv(trans) @ scale @ trans
    upside_down_to_right_side_up = jax.device_put(upside_down_to_right_side_up, device_cpu)

    path_i = Path()

    cycle_polyline_2dpoint = jax.jit(jax.vmap(transform.apply_to_point, (None, 0)))(
        upside_down_to_right_side_up, cycle_polyline_2dpoint)
    
    cycle_polyline_2dpoint = jax.device_get(cycle_polyline_2dpoint)
    cycle_polyline_2dpoint = np.array(cycle_polyline_2dpoint)

    for j in range(1, cycle_polyline_2dpoint.shape[0]):
        path_i.append(Line(
            complex(cycle_polyline_2dpoint[j-1, 0],
                    cycle_polyline_2dpoint[j-1, 1]),
            complex(cycle_polyline_2dpoint[j, 0], cycle_polyline_2dpoint[j, 1])))
    # Add last point
    path_i.append(Line(
        complex(cycle_polyline_2dpoint[cycle_polyline_2dpoint.shape[0]-1, 0],
                cycle_polyline_2dpoint[cycle_polyline_2dpoint.shape[0]-1, 1]),
        complex(cycle_polyline_2dpoint[0, 0], cycle_polyline_2dpoint[0, 1])))
    paths2svg.wsvg(path_i, filename=svg_out_filename,
                   svg_attributes=svg_attributes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert the trajectory to SVG. Input: The JSON file used by fill_2d_shape.py to genrate the trajectory and a path for the SVG. Output: The trajectory in SVG format.'
    )
    parser.add_argument(
        "input_filename", help="The JSON file used by fill_2d_shape.py to genrate the trajectory.")
    parser.add_argument(
        "datatoexport", help="The data to export. Either `cycle` (the cycle after stitching) or `cycles` (the cycles before stitching).")
    parser.add_argument("svg_out_filename", help="The path of the output SVG.")
    args = parser.parse_args()

    input_param_filename = args.input_filename
    datatoexport = args.datatoexport
    svg_out_filename = args.svg_out_filename

    if datatoexport == DataToExport.CYCLES.value:
        cycles_to_svg(input_param_filename, svg_out_filename)
    elif datatoexport == DataToExport.CYCLE.value:
        cycle_to_svg(input_param_filename, svg_out_filename)
    else:
        exit("Either `cycle` or `cycles` is accepted for the second positional argument.")
