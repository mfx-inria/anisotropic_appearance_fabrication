$StartMs = Get-Date
python tools/fill_2d_shape.py data/teaser_coarse.json
python tools/tosvg.py data/teaser_coarse.json cycle data/svg/cycle/teaser_coarse.svg

python tools/fill_2d_shape.py data/teaser.json
python tools/togcode.py data/teaser.json CR10S_Pro
python tools/tosvg.py data/teaser.json cycle data/svg/cycle/teaser.svg
$EndMs = Get-Date
$Diff = $EndMs - $StartMs
$DiffTotalS = $Diff.TotalSeconds
Write-Host "This script took $DiffTotalS seconds to run"