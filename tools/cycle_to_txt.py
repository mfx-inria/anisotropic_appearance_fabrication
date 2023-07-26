import argparse

import numpy as np

import cglib.fdm_aa
import cglib.polyline

if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description='Convert a polyline with varying radius generated with fill_2d_shape.py to a text file. Input: The JSON file used by fill_2d_shape.py (must have been launched first). Output: The text file, where each line is <x> <y> <layer_height> <radius>.'
    )
    parser.add_argument(
        "input_filename", help="The JSON file used by fill_2d_shape.py.")
    args = parser.parse_args()

    input_param_filename = args.input_filename

    parameters = cglib.fdm_aa.Parameters()
    parameters.load(input_param_filename)

    cycle_polyline = cglib.polyline.load(parameters.cycle_polyline_filename)
    layer_height = cycle_polyline.data[0, 1]

    # To .paths
    with open(parameters.cycle_txt_filename, 'w', encoding="utf-8") as f:
        cycle_polyline_2dpoint: np.ndarray = cycle_polyline.point[0]
        cycle_radius: np.ndarray = cycle_polyline.point_data[0]
        for point_index in range(cycle_polyline_2dpoint.shape[0]):
            point = cycle_polyline_2dpoint[point_index]
            radius = cycle_radius[point_index]
            width = radius * 2.
            f.write(f"{point[0]} {point[1]} {layer_height} {width}\n")
