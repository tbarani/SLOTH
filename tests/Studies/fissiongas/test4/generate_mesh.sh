#!/bin/bash
# Regenerate hex.msh for test4: the 3D mesh of the 4-hexagonal-grain periodic cell of Fig. 3 of
# Aagesen et al., Comput. Mater. Sci. 161 (2019) 35-45 (hexagonal grains extruded in z).
#
# Geometry of Section 3.1: hexagons with edge-to-edge distance d = 0.6 um, i.e. circumradius
# a = d/sqrt(3) = 346.41 nm, cell 1.039 x 1.2 um (our x is the paper's y), Lz = 0.28 um (the
# NTJ = 7.14/um reference case of Fig. 4/6c). The 40 nm base mesh is meshed once; production
# runs refine it near the interfaces with -amr (the paper uses ~10 nm there).
#
#   make_hex_geo.py   -> hex3d.geo   (exact honeycomb cell clipped to the box, Periodic Line
#                                      pairs in x/y, Extrude in z with nz layers)
#   gmsh -3           -> hex_raw.msh (conformal tet mesh, all elements saved)
#   finalize_mesh.py  -> hex.msh     (grain/box physical names, $Periodic in x/y/z appended
#                                      after $Elements)
#
# The committed hex.msh is the output of this script, pinned by the parameters below.
set -e
cd "$(dirname "$0")"
python3 make_hex_geo.py --a 346.4101615137755 --lc 40 --layout center --lz 280 --nz 7 --out hex3d.geo
gmsh hex3d.geo -3 -format msh2 -setnumber Mesh.SaveAll 1 -o hex_raw.msh
python3 finalize_mesh.py --in hex_raw.msh --out hex.msh --a 346.4101615137755 --layout center
