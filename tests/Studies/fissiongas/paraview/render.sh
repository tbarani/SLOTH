#!/usr/bin/env bash
# render.sh -- render a fancy ParaView scene + movie for a fission-gas run.
#
#   ./render.sh <Saves_dir> <out_dir> [extra options for fissiongas_viz.py]
#
# Examples:
#   ./render.sh ~/projects/sloth_trials/fissiongas_results/big_df1e4/Saves \
#       ~/projects/sloth_trials/fissiongas_results/big_df1e4/viz
#   ./render.sh ../../../build/tests/Studies/fissiongas/test4/Saves /tmp/hex3d_viz \
#       --periodic 3,3,3 --stride 2
#
# Requires paraview's pvpython on PATH (here: /usr/bin/pvpython) and ffmpeg for movie.mp4.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVES="${1:?usage: render.sh <Saves_dir> <out_dir> [options]}"
OUT="${2:?usage: render.sh <Saves_dir> <out_dir> [options]}"
shift 2

pvpython "$HERE/fissiongas_viz.py" -- --saves "$SAVES" --out "$OUT" "$@"
