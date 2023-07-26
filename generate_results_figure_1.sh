#!/bin/bash
python3 tools/fill_2d_shape.py data/teaser_coarse.json
python3 tools/tosvg.py data/teaser_coarse.json cycle data/svg/cycle/teaser_coarse.svg

python3 tools/fill_2d_shape.py data/teaser.json
python3 tools/togcode.py data/teaser.json CR10S_Pro
python3 tools/tosvg.py data/teaser.json cycle data/svg/cycle/teaser.svg
