#!/bin/bash
# Regenerate poly.geo / poly.msh / bubbles.txt for test5: the 3D periodic Voronoi
# polycrystal (8 grains, 600 x 600 x 280 nm, no carved inclusions) and the Aagesen-style
# initial fission-gas bubble list.
#
# Reference: L.K. Aagesen, D. Schwen, M.R. Tonks, Y. Zhang, "Phase-field modeling of
# fission gas bubble growth on grain boundaries and triple junctions in UO2 nuclear
# fuel", Comput. Mater. Sci. 161 (2019) 35-45, Sec. 3.1 / Table 2 reference case:
# NGB = 20/um^2, NTJ = 7.14/um, r = 44 nm. The paper uses extruded hexagonal grains;
# here the same densities are applied to a Voronoi tessellation of 8 seeds.
#
# Pipeline (all in make_voronoi3d.py):
#   sac_de_billes.fillMaxRSA_3D (Merope python bindings, one phase per seed)
#     -> periodic Voronoi cell per seed, clipped/wrapped into the box
#     -> poly.geo  (gmsh BRep: one Physical Volume per grain, Physical Surface "box",
#                   Distance+Threshold mesh-size field near the grain boundaries,
#                   one `Periodic Surface` pair per boundary polygon)
#     -> gmsh poly.geo -3 -format msh2 -> poly_raw.msh
#     -> poly.msh  (msh2 with $PhysicalNames, box triangles, $Periodic in x/y/z written
#                   after $Elements; MFEM 4.9's v2 reader needs that order, see
#                   test4/finalize_mesh.py)
#     -> bubbles.txt (triple-junction bubbles first, then grain-boundary bubbles)
#
# Requirements
#   - Merope clone+build at /home/tommy/projects/sloth_trials/Merope, python module in
#     Merope/install/lib (built for the system python3, 3.12). sac_de_billes is only used
#     for the seed packing.
#   - gmsh >= 4.12 (/usr/bin/gmsh), no ANN support needed (the "Gmsh must be compiled with
#     ANN support" warnings from the periodic matching are harmless).
#
# Pinned parameters (do not change without regenerating and re-validating the mesh):
#   - 8 grains, Merope packing seed 0.  The tessellation is Voronoi, not Laguerre:
#     fillMaxRSA gives all seeds the same radius, so the Laguerre weights are equal.
#   - mesh size: lc_coarse = 40 nm away from the grain boundaries, lc_fine = 16 nm on
#     them, grading over dist_max = 15 nm.  (30/8/15, the first choice, gives 600k tets;
#     the ~120k cap of this study needs ~16 nm.)
#   - bubble seed 2019, radius 44 nm.
# Output: 19329 nodes, 103710 tets, 10152 boundary triangles, 63 bubbles (42 on triple
# lines, 21 on grain boundaries).
#
# Notes
#   - Why not Merope's own 3D MeshGenerator: it meshes the "periodic envelope", i.e. the
#     Laguerre cells translated so that each one is connected.  The volume and the torus
#     topology are right, but the bounding box is ~750 x 774 x 459 nm, so it is not a mesh
#     of the 600 x 600 x 280 box the rest of test5 assumes.  Same tessellation, clipped to
#     the fundamental box here.
#   - A grain whose cell spans a whole period (common in this thin box, like the extruded
#     hexagons of test4) appears as several pieces welded along an internal "seam"; those
#     faces are not counted as grain boundaries (the report prints their count).
#   - The 7.14/um triple-line density saturates the short Voronoi triple lines (85
#     segments, 5.9 um total).  The placement keeps the 2r = 88 nm spacing wherever it
#     can; the few positions that cannot are reported in bubbles.txt and by the script.
set -e
cd "$(dirname "$0")"

MEROPE=/home/tommy/projects/sloth_trials/Merope
export PYTHONPATH=$MEROPE/install/lib

python3 make_voronoi3d.py \
    --ngrains 8 --seed 0 \
    --lc-coarse 40 --lc-fine 16 --dist-max 15 \
    --bubble-seed 2019 --bubble-radius 44 \
    --geo poly.geo --msh poly.msh --bubbles bubbles.txt

echo "wrote poly.geo, poly.msh, bubbles.txt"
