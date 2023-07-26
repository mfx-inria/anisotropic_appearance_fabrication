$StartMs = Get-Date
# Figure 1, left
python tools/fill_2d_shape.py data/teaser_coarse.json
python tools/tosvg.py data/teaser_coarse.json cycle data/svg/cycle/teaser_coarse.svg

# Figure 1, right
python tools/fill_2d_shape.py data/teaser.json
python tools/togcode.py data/teaser.json CR10S_Pro
python tools/tosvg.py data/teaser.json cycle data/svg/cycle/teaser.svg

# Figure 2, right
python tools/fill_2d_shape.py data/tiles_square.json
python tools/togcode.py data/tiles_square.json CR10S_Pro
python tools/tosvg.py data/tiles_square.json cycle data/svg/cycle/tiles_square.svg

# Figure 4, right
python tools/fill_2d_shape.py data/square_dir_gradient.json
python tools/togcode.py data/square_dir_gradient.json CR10S_Pro
python tools/tosvg.py data/square_dir_gradient.json cycle data/svg/cycle/square_dir_gradient.svg

# Figure 8, bottom left
python tools/fill_2d_shape.py data/brain_for_overview_figure.json
python tools/togcode.py data/brain_for_overview_figure.json CR10S_Pro
python tools/tosvg.py data/brain_for_overview_figure.json cycle data/svg/cycle/brain_for_overview_figure.svg

# Figure 11, bottom
python tools/fill_2d_shape.py data/wave_coarse.json
python tools/tosvg.py data/wave_coarse.json cycle data/svg/cycle/wave_coarse.svg

# Figure 11, top
python tools/fill_2d_shape.py data/wave.json
python tools/togcode.py data/wave.json CR10S_Pro
python tools/tosvg.py data/wave.json cycle data/svg/cycle/wave.svg

# Figure 12, top left
python tools/fill_2d_shape.py data/ani_to_iso.json
python tools/togcode.py data/ani_to_iso.json CR10S_Pro
python tools/tosvg.py data/ani_to_iso.json cycle data/svg/cycle/ani_to_iso.svg

# Figure 12, top middle
python tools/fill_2d_shape.py data/people.json
python tools/togcode.py data/people.json CR10S_Pro
python tools/tosvg.py data/people.json cycle data/svg/cycle/people.svg

# Figure 12, top left
python tools/fill_2d_shape.py data/brain.json
python tools/togcode.py data/brain.json CR10S_Pro
python tools/tosvg.py data/brain.json cycle data/svg/cycle/brain.svg

# Figure 13, from the left to the right
python tools/fill_2d_shape.py data/stress_0p4.json
python tools/togcode.py data/stress_0p4.json CR10S_Pro
python tools/tosvg.py data/stress_0p4.json cycle data/svg/cycle/stress_0p4.svg

python tools/fill_2d_shape.py data/stress_0p8.json
python tools/togcode.py data/stress_0p8.json CR10S_Pro
python tools/tosvg.py data/stress_0p8.json cycle data/svg/cycle/stress_0p8.svg

python tools/fill_2d_shape.py data/stress_1p6.json
python tools/togcode.py data/stress_1p6.json CR10S_Pro
python tools/tosvg.py data/stress_1p6.json cycle data/svg/cycle/stress_1p6.svg

python tools/fill_2d_shape.py data/stress_3p2.json
python tools/togcode.py data/stress_3p2.json CR10S_Pro
python tools/tosvg.py data/stress_3p2.json cycle data/svg/cycle/stress_3p2.svg

# Figure 15, from the left to the right
python tools/fill_2d_shape.py data/bunny_0p25.json
python tools/togcode.py data/bunny_0p25.json CR10S_Pro
python tools/tosvg.py data/bunny_0p25.json cycle data/svg/cycle/bunny_0p25.svg

python tools/fill_2d_shape.py data/monkey_0p25.json
python tools/togcode.py data/monkey_0p25.json CR10S_Pro
python tools/tosvg.py data/monkey_0p25.json cycle data/svg/cycle/monkey_0p25.svg

python tools/fill_2d_shape.py data/siggraph2023qrcode.json
python tools/togcode.py data/siggraph2023qrcode.json CR10S_Pro
python tools/tosvg.py data/siggraph2023qrcode.json cycle data/svg/cycle/siggraph2023qrcode.svg

python tools/fill_2d_shape.py data/teapot_0p25.json
python tools/togcode.py data/teapot_0p25.json CR10S_Pro
python tools/tosvg.py data/teapot_0p25.json cycle data/svg/cycle/teapot_0p25.svg

python tools/fill_2d_shape.py data/dragon_0p25.json
python tools/togcode.py data/dragon_0p25.json CR10S_Pro
python tools/tosvg.py data/dragon_0p25.json cycle data/svg/cycle/dragon_0p25.svg
$EndMs = Get-Date
$Diff = $EndMs - $StartMs
$DiffTotalS = $Diff.TotalSeconds
Write-Host "This script took $DiffTotalS seconds to run"