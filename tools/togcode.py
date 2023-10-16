import argparse
from enum import Enum
import numpy as np

from svgpathtools import svg2paths

import cglib.fdm_aa
import cglib.polyline


class PrinterParameters:
    def __init__(self):
        self.printer_profile = None
        self.nozzle_width = None
        self.speed = None
        self.flow_multiplier = None
        self.bed_temp = None
        self.extruder_temp = None
        self.layer_height = None
        self.filament_diameter = None
        self.filament_priming = None
        self.z_lift = None
        self.total_extrusion_length = None

    def get_feedrate(self) -> int:
        # mm/s to mm/min
        return int(self.speed * 60)


def compute_extrusion_length(width: float,
                             local_length: float,
                             filament_diameter: float,
                             layer_height: float,
                             flow_multiplier: float) -> float:
    """Compute the length of the filament to extrude.

    Notes
    -----
    https://3dprinting.stackexchange.com/questions/6289/how-is-the-e-argument-calculated-for-a-given-g1-command
    """
    crsec = np.pi * filament_diameter**2 / 4.0
    v_mm3 = local_length * layer_height * width
    return flow_multiplier * v_mm3 / crsec


def features_get_bed_size_xy(features_str: str) -> np.ndarray:

    bed_circular = features_str.find("bed_circular  = true") != -1
    bed_circular = bed_circular or features_str.find(
        "bed_circular = true") != -1
    if not bed_circular:
        # The bed is not circular

        features_bed_size_x_i = features_str.find("bed_size_x_mm")
        bed_size_x = features_str[features_bed_size_x_i:]
        bed_size_x_i_end = bed_size_x.find('\n')
        bed_size_x = bed_size_x[:bed_size_x_i_end]
        bed_size_x = float(bed_size_x.split()[2])

        features_bed_size_y_i = features_str.find("bed_size_y_mm")
        bed_size_y = features_str[features_bed_size_y_i:]
        bed_size_y_i_end = bed_size_y.find('\n')
        bed_size_y = bed_size_y[:bed_size_y_i_end]
        bed_size_y = float(bed_size_y.split()[2])

        return np.array([bed_size_x, bed_size_y])
    else:

        bed_radius = features_str.find("bed_radius")
        bed_radius = features_str[bed_radius:]
        bed_radius_i_end = bed_radius.find('\n')
        bed_radius = bed_radius[:bed_radius_i_end]
        bed_radius = float(bed_radius.split()[2])

        return np.array([bed_radius, bed_radius]) * 2.


def features_origin_is_bed_center(features_str: str) -> np.ndarray:
    origin_is_bed_center = False
    index = features_str.find("bed_origin_x = bed_size_x_mm / 2.0")
    origin_is_bed_center = index != -1 or origin_is_bed_center
    index = features_str.find("bed_origin_x = bed_size_x_mm/2")
    origin_is_bed_center = index != -1 or origin_is_bed_center

    return origin_is_bed_center


def travel(point_start: np.ndarray, point_end: np.ndarray, printer_param: PrinterParameters) -> str:

    point_im1_z_lifted = point_start[2] + printer_param.z_lift
    point_im1_z_lifted_str = f"{point_im1_z_lifted:.6f}"

    point_x_im1_str = f"{point_start[0]:.3f}"
    point_y_im1_str = f"{point_start[1]:.3f}"
    point_x_i_str = f"{point_end[0]:.3f}"
    point_y_i_str = f"{point_end[1]:.3f}"
    point_z_i_str = f"{point_end[2]:.6f}"
    # Retract
    travel_str = ";retract\n"
    printer_param.total_extrusion_length -= printer_param.filament_priming
    total_extrusion_length_str = f"{printer_param.total_extrusion_length:.6f}"
    travel_str += f"G1 F2700 E{total_extrusion_length_str}\n"
    # Travel
    travel_str += ";travel\n"
    travel_str += f"G0 F600 X{point_x_im1_str} Y{point_y_im1_str} Z{point_im1_z_lifted_str}\n"
    travel_str += f"G0 F{printer_param.get_feedrate()} X{point_x_i_str} Y{point_y_i_str} Z{point_im1_z_lifted_str}\n"
    travel_str += f"G0 F600 X{point_x_i_str} Y{point_y_i_str} Z{point_z_i_str}\n"
    # Prime
    travel_str += ";prime\n"
    printer_param.total_extrusion_length += printer_param.filament_priming
    total_extrusion_length_str = f"{printer_param.total_extrusion_length:.6f}"
    travel_str += f"G1 F2700 E{total_extrusion_length_str}\n"
    return travel_str


def travel_to(point_end: np.ndarray, 
              printer_param: PrinterParameters, 
              prime: bool = True) -> str:

    point_x_i_str = f"{point_end[0]:.3f}"
    point_y_i_str = f"{point_end[1]:.3f}"
    point_z_i_str = f"{point_end[2]:.6f}"
    # Retract
    travel_str = ";retract\n"
    printer_param.total_extrusion_length -= printer_param.filament_priming
    total_extrusion_length_str = f"{printer_param.total_extrusion_length:.6f}"
    travel_str += f"G1 F2700 E{total_extrusion_length_str}\n"
    # Travel
    travel_str += ";travel\n"
    travel_str += f"G0 F{printer_param.get_feedrate()} X{point_x_i_str} Y{point_y_i_str} Z{point_z_i_str}\n"
    if prime:
        # Prime
        travel_str += ";prime\n"
        printer_param.total_extrusion_length += printer_param.filament_priming
        total_extrusion_length_str = f"{printer_param.total_extrusion_length:.6f}"
        travel_str += f"G1 F2700 E{total_extrusion_length_str}\n"
    return travel_str


def run():
    parser = argparse.ArgumentParser(
        description='Convert the cycle generated by fill_2d_shape.py to machine instructions. Input: The JSON file used by fill_2d_shape.py. Output: The G-code to print the generated cycle.'
    )
    parser.add_argument(
        "input_filename", help="The JSON file used by fill_2d_shape.py.")
    parser.add_argument(
        "printer_profile", help="The printer profile. Only 'CR10S_Pro' was tested. In theory, the name of the folders at `src/ext/iceslprinters/fff` are valid inputs.")
    parser.add_argument(
        "-nw",
        "--nozzle_width",
        help="The nozzle diameter in millimeter. Default: the nozzle width in the input file.",
        type=float)
    parser.add_argument(
        "-s",
        "--speed",
        help="The speed of the moving head, in mm/s. Default: 30.",
        type=float,
        default=30.)
    parser.add_argument(
        "-fm",
        "--flow_multiplier",
        help="The flow multiplier. Default: 1",
        type=float,
        default=1.0)
    parser.add_argument(
        "-lc",
        "--layer_count",
        help="The number of layers. Default 3.",
        type=int,
        default=3)
    parser.add_argument(
        "-bt",
        "--bed_temp",
        help="The bed temperature in degree Celsius. Default: 55.",
        type=int,
        default=55)
    parser.add_argument(
        "-et",
        "--extruder_temp",
        help="The extruder temperature in degree Celsius. Default: 215.",
        type=int,
        default=215)
    parser.add_argument(
        "-lh",
        "--layer_height",
        help="The layer height. Default: the layer height associated with the polyline.",
        type=float)
    parser.add_argument(
        "-fd",
        "--filament_diameter",
        help="The filament diameter of the filament used by the printer. Default: 1.75 mm.",
        type=float,
        default=1.75)
    parser.add_argument(
        "-fp",
        "--filament_priming",
        help="Retraction setting. Between 0.4mm and 0.8mm of retract/prime for direct-drive setup, between 3mm and 6mm for bowden (stock) setup. Default: 0.4 mm.",
        type=float,
        default=0.4)
    parser.add_argument(
        "-zl",
        "--z_lift",
        help="Distance to move the printhead up (or the build plate down), after each retraction, right before a travel move takes place. Default: 0.4",
        type=float,
        default=0.4)
    
    args = parser.parse_args()

    input_param_filename = args.input_filename
    printer_profile = args.printer_profile

    printer_param = PrinterParameters()

    printer_param.nozzle_width = args.nozzle_width
    printer_param.speed = args.speed
    printer_param.flow_multiplier = args.flow_multiplier
    printer_param.bed_temp = args.bed_temp
    printer_param.extruder_temp = args.extruder_temp
    printer_param.layer_height = args.layer_height
    printer_param.filament_diameter = args.filament_diameter
    printer_param.filament_priming = args.filament_priming
    printer_param.z_lift = args.z_lift

    layer_count = args.layer_count

    # Should remove dependency to cglib.fdm_aa
    parameters = cglib.fdm_aa.Parameters()
    parameters.load(input_param_filename)

    # Load svg paths
    # Transformations inside the SVG are ignored, so only SVG files without
    # transformations are valid. Using Inkscape, ensure your contour is not
    # associated with a layer to avoid implicit transformations. Be sure that
    # the contour is a clockwise-oriented closed polyline. Holes are
    # represented with counter-clockwise oriented closed polylines.
    _, _, svg_attributes = svg2paths(
        parameters.svg_path, return_svg_attributes=True)
    # The shape domain size is determined by the SVG width and heigh
    # -2: remove the unit
    object_width_x = float(svg_attributes['width'][:-2])
    object_width_y = float(svg_attributes['height'][:-2])
    object_width_xy = np.array([object_width_x, object_width_y])

    # If the user does not specify the nozzle width,
    # use by default the one in the input file
    if printer_param.nozzle_width is None:
        printer_param.nozzle_width = parameters.nozzle_width

    cycle_polyline = cglib.polyline.load(parameters.cycle_polyline_filename)
    if printer_param.layer_height is None:
        printer_param.layer_height = cycle_polyline.data[0, 1]

    gcode_filename = parameters.gcode_filename

    # Get the printer profile header and footer
    printer_profile_path_head = f"src/ext/iceslprinters/fff/{printer_profile}/"
    printer_profile_header_path = printer_profile_path_head + "header.gcode"
    printer_profile_footer_path = printer_profile_path_head + "footer.gcode"
    printer_profile_features_path = printer_profile_path_head + "features.lua"
    printer_profile_printer_path = printer_profile_path_head + "printer.lua"

    # File to string
    with open(printer_profile_header_path, 'r') as f:
        header_str = f.read()
    with open(printer_profile_footer_path, 'r') as f:
        footer_str = f.read()
    with open(printer_profile_features_path, 'r') as f:
        features_str = f.read()
    with open(printer_profile_printer_path, 'r') as f:
        printer_str = f.read()

    # Extract the bed size from features string
    bed_size_xy = features_get_bed_size_xy(features_str)
    origin_is_bed_center = features_origin_is_bed_center(printer_str)

    # Put user defined parameters in the header
    header_str = header_str.replace("<HBPTEMP>", str(int(printer_param.bed_temp)))
    header_str = header_str.replace("<TOOLTEMP>", str(int(printer_param.extruder_temp)))
    header_str = header_str.replace("<BEDLVL>", "G0 F6200 X0 Y0")
    header_str = header_str.replace("<NOZZLE_DIAMETER>", str(printer_param.nozzle_width))
    header_str = header_str.replace("<ACCELERATIONS>", "")
    header_str = header_str.replace("<FILAMENT>", "0.08")

    # mm/s to mm/min
    feedrate = int(printer_param.speed * 60.)
    # Total extrusion length
    total_extrusion_length = 0.0

    world_to_object = -object_width_xy*0.5
    if origin_is_bed_center:
        object_to_bed = np.array([0., 0.])
    else:
        # Assume left-bottom origin
        object_to_bed = bed_size_xy*0.5

    with open(gcode_filename, 'w', encoding="utf-8") as f:
        # Write header
        f.write(header_str)

        # Write the same cycle for each layer
        cycle_polyline_2dpoint: np.ndarray = cycle_polyline.point[0]
        cycle_radius: np.ndarray = cycle_polyline.point_data[0]
        for layer_index in range(layer_count):
            layer_height_i = (layer_index + 1) * printer_param.layer_height
            layer_height_i_str = f"{layer_height_i:.6f}"
            for point_index in range(cycle_polyline_2dpoint.shape[0]):
                point_i = cycle_polyline_2dpoint[point_index]
                point_i = point_i + world_to_object + object_to_bed
                point_x_str = f"{point_i[0]:.3f}"
                point_y_str = f"{point_i[1]:.3f}"
                width_i = cycle_radius[point_index] * 2.
                if point_index == 0:
                    f.write(f"G0 F{feedrate} X{point_x_str} Y{point_y_str} Z{layer_height_i_str}\n")
                else:
                    point_im1 = cycle_polyline_2dpoint[point_index-1]
                    point_im1 = point_im1 + world_to_object + object_to_bed
                    dist_pi_im1 = np.linalg.norm(point_i - point_im1)
                    width_im1 = cycle_radius[point_index-1] * 2.
                    width_i = (width_im1 + width_i) * 0.5
                    extrusion_length = compute_extrusion_length(
                        width_i,
                        dist_pi_im1,
                        printer_param.filament_diameter,
                        printer_param.layer_height,
                        printer_param.flow_multiplier)
                    total_extrusion_length += extrusion_length
                    total_extrusion_length_str = f"{total_extrusion_length:.6f}"
                    f.write(f"G1 F{feedrate} X{point_x_str} Y{point_y_str} Z{layer_height_i_str} E{total_extrusion_length_str}\n")
        
        # Write footer
        f.write(footer_str)


if __name__ == "__main__":
    run()
