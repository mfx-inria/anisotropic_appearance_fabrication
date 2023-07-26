# Description

This folder contains
- 2D shapes that can be filled with an orientable cycle with the script
  `tools/fill_2d_shape.py`,
- The scripts using the tool `tool/tosvg.py` generate SVG files in the folder
  `cycle`.

# Input 2D Shapes

The 2D shapes are represented with files in SVG format. 

Transformations inside the SVG are ignored, so only SVG files without
transformations are valid. Using Inkscape, ensure your contour is not
associated with a layer to avoid implicit transformations. Be sure that the
contour is a clockwise-oriented closed polyline. Holes are represented with
counter-clockwise oriented closed polylines.

Only SVG object **path** can be imported.

The shape domain size is determined by the SVG width and heigh attributes.
