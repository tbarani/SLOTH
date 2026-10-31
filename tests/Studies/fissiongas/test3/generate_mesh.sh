#!/bin/bash
# Regenerate poly.geo / poly.msh for test3: 5-grain periodic polycrystal with 4 bubbles
# carved on the grain boundaries, produced by merope.mesh.MeshGenerator2D (python API).
#
# Requires the Merope clone+build at /home/tommy/projects/sloth_trials/Merope
# (python module in Merope/install/lib). The committed poly.msh is the output of this
# script; the input is pinned by the seed and the parameters below.
set -e
cd "$(dirname "$0")"
MEROPE=/home/tommy/projects/sloth_trials/Merope
PYTHONPATH=$MEROPE/install/lib python3 /home/tommy/projects/sloth_trials/merope2d/make_geo.py \
    --lx 600 --ly 600 --ngrains 5 --seed 0 \
    --bubbles 4 --bubble-radius 22 \
    --lc-coarse 25 --lc-fine 5 --dist-max 15 \
    --out poly.geo
# RecombineAll=1 -> quads (SLOTH's NC-mesh path here only converts quad meshes).
gmsh poly.geo -2 -format msh2 -setnumber Mesh.RecombineAll 1 -o poly.msh
