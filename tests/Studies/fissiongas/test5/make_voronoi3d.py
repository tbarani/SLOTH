#!/usr/bin/env python3
"""Build the 3D periodic Voronoi polycrystal mesh and the initial bubble list for test5.

Geometry: 8 equiaxed grains filling the periodic box 600 x 600 x 280 nm, without carved
inclusions (the fission-gas bubbles are analytic tanh fields in the initial condition, not
mesh holes). The mesh is test4-compatible: gmsh msh2, tetrahedra, one Physical Volume per
grain ("grain_1".."grain_8"), Physical Surface "box" on the outer faces, and a $Periodic
block pairing the opposite box faces in x, y and z. The $Periodic block is written after
$Elements because MFEM 4.9's msh2 reader applies the slave->master map only to the
elements read so far (see test4/finalize_mesh.py).

Pipeline
  1. seeds: sac_de_billes.fillMaxRSA_3D(Tore, ...) from the Merope python bindings, one
     phase per seed. Equal radii => the Laguerre tessellation Merope builds is the
     Voronoi tessellation of the seed centres;
  2. cells: the periodic Voronoi cell of each seed (box clipped by the bisector planes of
     every periodic image of every other seed, plus the seed's own images so the cell is
     bounded), then wrapped into the box: piece_k = (cell n (box + kL)) - kL for the 27
     k in {-1,0,1}^3. A cell is at most LxLxL in extent, so the pieces tile the box;
  3. poly.geo: hand-written gmsh BRep. Grain-boundary faces shared by two grains are a
     single entity (opposite orientation), so the tetrahedra are conformal. A Distance +
     Threshold background field refines the mesh near the grain boundaries, and one
     `Periodic Surface` pair per boundary polygon makes gmsh build identical meshes on
     opposite box faces;
  4. gmsh poly.geo -3 -format msh2 -> poly.msh (finalized: physical names, box surface,
     $Periodic node pairs computed from the coordinates);
  5. bubbles.txt: Aagesen, Schwen, Tonks, Zhang, Comput. Mater. Sci. 161 (2019) 35-45,
     Sec. 3.1 / Table 2 reference case (NGB = 20/um^2 on the grain boundaries,
     NTJ = 7.14/um on the triple lines, r = 44 nm) applied to this Voronoi tessellation
     instead of the paper's hexagonal grains.

Why not Merope's own 3D MeshGenerator: it meshes the "periodic envelope", i.e. the
Laguerre cells translated so that each one stays connected. That body has the right volume
and the right torus topology, but its bounding box is about 750 x 774 x 459 nm (cells stick
out of the fundamental box by up to a cell diameter), so it is not a mesh of the
600 x 600 x 280 nm box the rest of test5 assumes. The tessellation is the same; only the
clipping to the fundamental box differs.

usage:
  PYTHONPATH=/home/tommy/projects/sloth_trials/Merope/install/lib python3 make_voronoi3d.py
The defaults are the production parameters (pinned again in generate_mesh.sh). They give
19329 nodes, 103710 tetrahedra and 63 bubbles. The first choice of the study,
--lc-fine 8 --lc-coarse 30 --dist-max 15, is 6x finer at the grain boundaries and gives
about 600k tetrahedra, well above the ~120k cap of this test.
"""

import argparse
import itertools
import math
import os
import random
import subprocess
import sys

TET, TRI = 4, 2
BOX_PHYS = 9          # physical tag of the "box" surface


# --------------------------------------------------------------------------------------
# small vector helpers
# --------------------------------------------------------------------------------------
def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vmul(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def vnorm(a):
    return math.sqrt(vdot(a, a))


def vunit(a):
    n = vnorm(a)
    return (a[0] / n, a[1] / n, a[2] / n)


# --------------------------------------------------------------------------------------
# convex polyhedra: list of faces, each face a list of 3D points, CCW seen from outside
# --------------------------------------------------------------------------------------
def box_polyhedron(L, shift=(0.0, 0.0, 0.0)):
    lx, ly, lz = L
    sx, sy, sz = shift
    b = [
        [(0, 0, 0), (0, 0, lz), (0, ly, lz), (0, ly, 0)],                 # x = 0, -x
        [(lx, 0, 0), (lx, ly, 0), (lx, ly, lz), (lx, 0, lz)],             # x = lx, +x
        [(0, 0, 0), (lx, 0, 0), (lx, 0, lz), (0, 0, lz)],                 # y = 0, -y
        [(0, ly, 0), (0, ly, lz), (lx, ly, lz), (lx, ly, 0)],             # y = ly, +y
        [(0, 0, 0), (0, ly, 0), (lx, ly, 0), (lx, 0, 0)],                 # z = 0, -z
        [(0, 0, lz), (lx, 0, lz), (lx, ly, lz), (0, ly, lz)],             # z = lz, +z
    ]
    return [[vadd(p, shift) for p in f] for f in b]


def poly_volume(faces):
    """Volume of a closed polyhedron whose faces are outward oriented."""
    c = (0.0, 0.0, 0.0)
    n = 0
    for f in faces:
        for p in f:
            c = vadd(c, p)
            n += 1
    c = vmul(c, 1.0 / n)
    vol = 0.0
    for f in faces:
        for i in range(1, len(f) - 1):
            vol += vdot(vsub(f[0], c), vcross(vsub(f[i], c), vsub(f[i + 1], c)))
    return vol / 6.0


def loop_normal(loop, verts):
    """Newell normal of a polygon given as vertex ids."""
    n = (0.0, 0.0, 0.0)
    m = len(loop)
    for i in range(m):
        a = verts[loop[i] - 1]
        b = verts[loop[(i + 1) % m] - 1]
        n = (n[0] + (a[1] - b[1]) * (a[2] + b[2]),
             n[1] + (a[2] - b[2]) * (a[0] + b[0]),
             n[2] + (a[0] - b[0]) * (a[1] + b[1]))
    return n


def _basis(n):
    """Orthonormal (u, v) with u x v = n."""
    a = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = vunit(vcross(n, a))
    v = vcross(n, u)
    return u, v


def clip_polyhedron(faces, n, d, eps):
    """Clip by {x : n.x <= d}. n need not be a unit vector."""
    nn = vnorm(n)
    if nn == 0.0:
        return faces
    n = vmul(n, 1.0 / nn)
    d /= nn
    out_faces = []
    cut = []
    for f in faces:
        m = len(f)
        poly = []
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            da = vdot(n, a) - d
            db = vdot(n, b) - d
            if da <= eps:
                poly.append(a)
            if (da > eps) != (db > eps):
                t = da / (da - db)
                p = vadd(a, vmul(vsub(b, a), t))
                poly.append(p)
                cut.append(p)
        clean = []
        for p in poly:
            if not clean or vnorm(vsub(p, clean[-1])) > eps:
                clean.append(p)
        if len(clean) > 2 and vnorm(vsub(clean[0], clean[-1])) <= eps:
            clean.pop()
        if len(clean) >= 3:
            out_faces.append(clean)
    if len(cut) >= 3:
        c = (0.0, 0.0, 0.0)
        for p in cut:
            c = vadd(c, p)
        c = vmul(c, 1.0 / len(cut))
        u, v = _basis(n)
        uniq = []
        for p in cut:
            if all(vnorm(vsub(p, q)) > eps for q in uniq):
                uniq.append(p)
        if len(uniq) >= 3:
            uniq.sort(key=lambda p: math.atan2(vdot(vsub(p, c), v),
                                               vdot(vsub(p, c), u)))
            out_faces.append(uniq)
    return out_faces


def clip_by_box(faces, L, shift, eps):
    """Intersect a convex polyhedron with the box translated by `shift`."""
    out = faces
    for axis in range(3):
        n = [0.0, 0.0, 0.0]
        n[axis] = 1.0
        # n.x <= shift[axis] + L[axis]  and  -n.x <= -shift[axis]
        out = clip_polyhedron(out, tuple(n), shift[axis] + L[axis], eps)
        if not out:
            return []
        out = clip_polyhedron(out, tuple(-x for x in n), -shift[axis], eps)
        if not out:
            return []
    return out


# --------------------------------------------------------------------------------------
# seeds and periodic Voronoi cells
# --------------------------------------------------------------------------------------
def merope_seeds(L, ngrains, seed):
    try:
        import sac_de_billes
    except ImportError:
        sys.exit("cannot import sac_de_billes: run with "
                 "PYTHONPATH=/home/tommy/projects/sloth_trials/Merope/install/lib")
    spheres = sac_de_billes.fillMaxRSA_3D(sac_de_billes.Tore, list(L), ngrains, seed, 0.0)
    if len(spheres) != ngrains:
        sys.exit(f"Merope placed {len(spheres)} of {ngrains} seeds")
    return [(float(s.center[0]), float(s.center[1]), float(s.center[2])) for s in spheres]


def periodic_cell(seed, seeds, L, eps):
    """Voronoi cell of `seed` in the periodic metric (a bounded convex polyhedron).

    The cell is contained in the seed-centred box of size L (the bisectors with the seed's
    own periodic images bound it), so that box is a valid starting region.
    """
    shift = tuple(seed[a] - 0.5 * L[a] for a in range(3))
    faces = box_polyhedron(L, shift)
    for k in itertools.product((-1, 0, 1), repeat=3):
        sh = (k[0] * L[0], k[1] * L[1], k[2] * L[2])
        for j, t in enumerate(seeds):
            if t == seed and k == (0, 0, 0):
                continue
            s = (t[0] + sh[0], t[1] + sh[1], t[2] + sh[2])
            n = vsub(s, seed)
            if vnorm(n) < 1e-12:
                continue
            d = 0.5 * (vdot(s, s) - vdot(seed, seed))
            faces = clip_polyhedron(faces, n, d, eps)
            if not faces:
                raise RuntimeError("empty Voronoi cell")
    return faces


def wrapped_pieces(cell, L, eps):
    """cell n box, folded back into the box for every periodic image."""
    pieces = []
    for k in itertools.product((-1, 0, 1), repeat=3):
        sh = (k[0] * L[0], k[1] * L[1], k[2] * L[2])
        piece = clip_by_box(cell, L, sh, eps)
        if not piece:
            continue
        piece = [[vsub(p, sh) for p in f] for f in piece]
        pieces.append(piece)
    return pieces


# --------------------------------------------------------------------------------------
# gmsh .geo writer
# --------------------------------------------------------------------------------------
def fmt(x):
    return f"{x:.17g}"


def build_geo(grains, L, lc_coarse, lc_fine, dist_max, tol):
    """grains = list (one per grain) of lists of convex pieces (lists of faces).

    Returns the .geo text and the geometry bookkeeping.
    """
    lx, ly, lz = L
    # ---- global vertex table (snap nearly equal points) -------------------------
    vkey2id = {}
    verts = []
    for pieces in grains:
        for piece in pieces:
            for f in piece:
                for p in f:
                    key = (round(p[0], 4), round(p[1], 4), round(p[2], 4))
                    if key not in vkey2id:
                        vkey2id[key] = len(verts) + 1
                        verts.append(key)

    def vid(p):
        return vkey2id[(round(p[0], 4), round(p[1], 4), round(p[2], 4))]

    # ---- face table ------------------------------------------------------------
    face_loop = {}            # frozenset(vertex ids) -> reference loop
    face_uses = {}            # frozenset(vertex ids) -> list of grain ids
    pieces_of_grain = []      # per grain, per piece: list of loops
    for g, pieces in enumerate(grains, start=1):
        gl = []
        for piece in pieces:
            pl = []
            for f in piece:
                loop = []
                for p in f:
                    i = vid(p)
                    if not loop or loop[-1] != i:
                        loop.append(i)
                while len(loop) > 1 and loop[0] == loop[-1]:
                    loop.pop()
                if len(loop) < 3:
                    continue
                key = frozenset(loop)
                face_uses.setdefault(key, []).append(g)
                if key not in face_loop:
                    face_loop[key] = loop
                pl.append(loop)
            if pl:
                gl.append(pl)
        pieces_of_grain.append(gl)

    # two pieces of the same grain may share a face: the grain touches its own periodic
    # image there (its cell spans the whole box in that direction, like the extruded
    # hexagons of test4). Such a "seam" is welded like a grain boundary but is not one;
    # analyse_gb() reports how many there are.

    # ---- box-boundary faces ----------------------------------------------------
    def box_plane(loop):
        pts = [verts[i - 1] for i in loop]
        for axis, val in ((0, 0.0), (0, lx), (1, 0.0), (1, ly), (2, 0.0), (2, lz)):
            if all(abs(p[axis] - val) < tol for p in pts):
                return (axis, val)
        return None

    box_faces = {}
    for key, loop in face_loop.items():
        if len(face_uses[key]) != 1:
            continue
        pl = box_plane(loop)
        if pl is not None:
            box_faces[key] = pl
    covered = sum(face_area(loop, verts) for key, loop in face_loop.items()
                  if len(face_uses[key]) == 1)
    A_box = 2 * (lx * ly + lx * lz + ly * lz)
    if abs(covered - A_box) > 1e-6 * A_box:
        raise RuntimeError(f"boundary faces cover {covered}, box surface is {A_box}")

    # ---- lines -----------------------------------------------------------------
    line_ids = {}

    def line_id(a, b):
        k = (min(a, b), max(a, b))
        if k not in line_ids:
            line_ids[k] = len(line_ids) + 1
        return line_ids[k], (a < b)

    surf_ids = {}
    surf_lines = {}
    for key in sorted(face_loop, key=lambda k: min(k)):
        loop = face_loop[key]
        lids = []
        for i in range(len(loop)):
            lid, forward = line_id(loop[i], loop[(i + 1) % len(loop)])
            lids.append(lid if forward else -lid)
        sid = len(surf_ids) + 1
        surf_ids[key] = sid
        surf_lines[sid] = lids

    # ---- periodic surface pairs ------------------------------------------------
    periodic_pairs = []
    for axis, length in ((0, lx), (1, ly), (2, lz)):
        lo = [k for k in box_faces if all(verts[i - 1][axis] < tol for i in k)]
        hi = {frozenset(vid(vadd(verts[i - 1], tuple(
            length if a == axis else 0.0 for a in range(3)))) for i in k): k for k in lo}
        if len(hi) != len(lo):
            raise RuntimeError(f"axis {axis}: {len(lo)} boundary pieces are not periodic")
        used = set()
        shift = [0.0, 0.0, 0.0]
        shift[axis] = length
        for k in lo:
            tgt = frozenset(vid(vadd(verts[i - 1], tuple(shift))) for i in k)
            if tgt not in hi or tgt in used:
                raise RuntimeError(f"axis {axis}: no periodic image for a boundary surface")
            used.add(tgt)
            # gmsh: "Periodic Surface {master} = {slave} Translate {..}", the translation
            # mapping the master onto the slave; the slave is meshed as a copy.
            periodic_pairs.append((surf_ids[tgt], surf_ids[k], shift))

    # ---- lines to refine: everything that is not a box edge ---------------------
    gb_lines = []
    for k, lid in line_ids.items():
        a, b = verts[k[0] - 1], verts[k[1] - 1]
        on_box_edges = 0
        for ax in range(3):
            if abs(a[ax] - b[ax]) < tol and (abs(a[ax]) < tol or abs(a[ax] - L[ax]) < tol):
                on_box_edges += 1
        if on_box_edges < 2:
            gb_lines.append(lid)

    # ---- write the .geo ---------------------------------------------------------
    out = []
    out.append("// File automatically generated by make_voronoi3d.py (test5)\n")
    out.append(f"// 3D periodic Voronoi polycrystal, box = {lx:g} x {ly:g} x {lz:g} nm, "
               f"{len(grains)} grains, no carved bubbles\n")
    out.append("// one Physical Volume per grain, Physical Surface \"box\", "
               "Periodic Surface pairs in x/y/z\n")
    out.append(f"lc_coarse = {lc_coarse:g};\nlc_fine = {lc_fine:g};\n\n")
    out.append("// BEGIN Points\n")
    for i, p in enumerate(verts, start=1):
        out.append(f"Point({i}) = {{{fmt(p[0])}, {fmt(p[1])}, {fmt(p[2])}, lc_coarse}};\n")
    out.append("// END Points\n\n// BEGIN Lines\n")
    for (a, b), lid in sorted(line_ids.items(), key=lambda kv: kv[1]):
        out.append(f"Line({lid}) = {{{a}, {b}}};\n")
    out.append("// END Lines\n\n// BEGIN Line Loops\n")
    for key, sid in surf_ids.items():
        out.append(f"Line Loop({sid}) = {{{', '.join(map(str, surf_lines[sid]))}}};\n")
    out.append("// END Line Loops\n\n// BEGIN Surfaces\n")
    for key, sid in surf_ids.items():
        out.append(f"Plane Surface({sid}) = {{{sid}}};\n")
    out.append("// END Surfaces\n\n// BEGIN Surface Loops\n")
    vol_of_piece = []          # (grain, volume id)
    nvol = 0
    for g, gl in enumerate(pieces_of_grain, start=1):
        for pl in gl:
            nvol += 1
            sids = []
            for loop in pl:
                sid = surf_ids[frozenset(loop)]
                ref = face_loop[frozenset(loop)]
                sids.append(sid if vdot(loop_normal(loop, verts),
                                        loop_normal(ref, verts)) > 0 else -sid)
            out.append(f"Surface Loop({nvol}) = {{{', '.join(map(str, sids))}}};\n")
            vol_of_piece.append((g, nvol))
    out.append("// END Surface Loops\n\n// BEGIN Volumes\n")
    for g, v in vol_of_piece:
        out.append(f"Volume({v}) = {{{v}}};\n")
    out.append("// END Volumes\n\n// BEGIN Periodic surfaces of the box\n")
    for lo, hi, shift in periodic_pairs:
        out.append(f"Periodic Surface {{{lo}}} = {{{hi}}} Translate "
                   f"{{{fmt(shift[0])}, {fmt(shift[1])}, {fmt(shift[2])}}};\n")
    out.append("// END Periodic surfaces of the box\n\n// BEGIN Physical volumes\n")
    for g in range(1, len(grains) + 1):
        vols = [v for gg, v in vol_of_piece if gg == g]
        out.append(f'Physical Volume({g}) = {{{", ".join(map(str, vols))}}};\n')
    out.append("// END Physical volumes\n\n// BEGIN Physical surfaces\n")
    out.append(f'Physical Surface({BOX_PHYS}) = '
               f'{{{", ".join(str(surf_ids[k]) for k in sorted(box_faces, key=lambda k: surf_ids[k]))}}};\n')
    out.append("// END Physical surfaces\n\n// BEGIN Mesh size field\n")
    out.append("Field[1] = Distance;\nField[1].CurvesList = {"
               + ", ".join(map(str, gb_lines)) + "};\n")
    out.append("Field[2] = Threshold;\nField[2].InField = 1;\n"
               f"Field[2].SizeMin = lc_fine;\nField[2].SizeMax = lc_coarse;\n"
               f"Field[2].DistMin = 0;\nField[2].DistMax = {dist_max:g};\n"
               "Background Field = 2;\n")
    out.append("\nMesh.ElementOrder = 1;\nMesh.MshFileVersion = 2.2;\n")
    out.append("Mesh 2;\nMesh 3;\n")
    return ("".join(out), verts, pieces_of_grain, face_loop, face_uses, box_faces,
            surf_ids, line_ids)


# --------------------------------------------------------------------------------------
# msh2 finalize
# --------------------------------------------------------------------------------------
def read_msh2(path):
    nodes = {}
    elems = []
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "$Nodes":
            n = int(lines[i + 1])
            for k in range(n):
                parts = lines[i + 2 + k].split()
                nodes[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
            i += 2 + n
        elif s == "$Elements":
            n = int(lines[i + 1])
            for k in range(n):
                parts = lines[i + 2 + k].split()
                etype, ntags = int(parts[1]), int(parts[2])
                tags = [int(t) for t in parts[3:3 + ntags]]
                ens = [int(t) for t in parts[3 + ntags:]]
                elems.append((etype, tags, ens))
            i += 2 + n
        else:
            i += 1
    return nodes, elems


def finalize_mesh(raw_path, out_path, L, ngrains):
    nodes, elems = read_msh2(raw_path)
    lx, ly, lz = L
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    zs = [p[2] for p in nodes.values()]
    for got, want, name in ((max(xs), lx, "x"), (max(ys), ly, "y"), (max(zs), lz, "z")):
        if abs(got - want) > 1e-6 * want:
            raise RuntimeError(f"mesh extent in {name} = {got}, expected {want}")
    if abs(min(xs)) > 1e-9 or abs(min(ys)) > 1e-9 or abs(min(zs)) > 1e-9:
        raise RuntimeError("mesh does not start at the origin")
    tol = 1e-9 * max(L)

    tets = [e for e in elems if e[0] == TET]
    tris = [e for e in elems if e[0] == TRI]
    if not tets:
        raise RuntimeError("no tetrahedra in the raw mesh")

    tets_out = []
    for etype, tags, ens in tets:
        g = tags[0]
        if not 1 <= g <= ngrains:
            raise RuntimeError(f"tet with physical tag {tags} outside 1..{ngrains}")
        tets_out.append((g, ens))

    def on_box_face(tri):
        for axis, val in ((0, 0.0), (0, lx), (1, 0.0), (1, ly), (2, 0.0), (2, lz)):
            if all(abs(nodes[t][axis] - val) < tol for t in tri):
                return True
        return False

    btris = [ens for etype, tags, ens in tris if on_box_face(ens)]

    # conformity: every tet face is shared by at most 2 tets, and the singly owned ones
    # are exactly the boundary triangles
    face_count = {}
    for _, tet in tets_out:
        for f in ((tet[1], tet[2], tet[3]), (tet[0], tet[2], tet[3]),
                  (tet[0], tet[1], tet[3]), (tet[0], tet[1], tet[2])):
            k = tuple(sorted(f))
            face_count[k] = face_count.get(k, 0) + 1
    if any(c > 2 for c in face_count.values()):
        raise RuntimeError("non-conformal 3D mesh")
    c1 = {f for f, c in face_count.items() if c == 1}
    if {tuple(sorted(t)) for t in btris} != c1:
        raise RuntimeError("boundary triangles do not match the tet boundary faces")

    # $Periodic node pairs (written after $Elements, as gmsh does)
    blocks = []
    for axis, length in ((0, lx), (1, ly), (2, lz)):
        master = [nid for nid, p in nodes.items() if abs(p[axis]) < tol]
        slave = [nid for nid, p in nodes.items() if abs(p[axis] - length) < tol]
        if len(master) != len(slave):
            raise RuntimeError(f"axis {axis}: {len(slave)} slaves vs {len(master)} masters")
        index = {}
        for m in master:
            pm = nodes[m]
            k = tuple(round(pm[a], 6) for a in range(3) if a != axis)
            if k in index:
                raise RuntimeError(f"axis {axis}: duplicate master node {k}")
            index[k] = m
        pairs = []
        for s in slave:
            ps = nodes[s]
            k = tuple(round(ps[a], 6) for a in range(3) if a != axis)
            if k not in index:
                raise RuntimeError(f"axis {axis}: no periodic master for slave node {s}")
            pairs.append((s, index[k]))
        tr = [0.0, 0.0, 0.0]
        tr[axis] = length
        blocks.append((pairs, tr))
        print(f"  axis {axis}: {len(pairs)} periodic node pairs")

    with open(out_path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$PhysicalNames\n{ngrains + 1}\n")
        for g in range(1, ngrains + 1):
            f.write(f'3 {g} "grain_{g}"\n')
        f.write(f'2 {BOX_PHYS} "box"\n')
        f.write("$EndPhysicalNames\n")
        f.write(f"$Nodes\n{len(nodes)}\n")
        for nid in sorted(nodes):
            x, y, z = nodes[nid]
            f.write(f"{nid} {x:.17g} {y:.17g} {z:.17g}\n")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{len(tets_out) + len(btris)}\n")
        eid = 0
        for g, tet in tets_out:
            eid += 1
            f.write(f"{eid} {TET} 2 {g} 1 {' '.join(map(str, tet))}\n")
        for t in btris:
            eid += 1
            f.write(f"{eid} {TRI} 2 {BOX_PHYS} 1 {' '.join(map(str, t))}\n")
        f.write("$EndElements\n")
        f.write("$Periodic\n3\n")
        for bi, (pairs, tr) in enumerate(blocks):
            f.write(f"2 {20 + bi} {bi + 1}\n")
            f.write(f"Affine 1 0 0 {tr[0]:.17g} 0 1 0 {tr[1]:.17g} "
                    f"0 0 1 {tr[2]:.17g} 0 0 0 1\n")
            f.write(f"{len(pairs)}\n")
            for s, m in pairs:
                f.write(f"{s} {m}\n")
        f.write("$EndPeriodic\n")

    per_grain = {}
    for g, _ in tets_out:
        per_grain[g] = per_grain.get(g, 0) + 1
    print(f"{out_path}: {len(nodes)} nodes, {len(tets_out)} tets "
          f"(per grain: {dict(sorted(per_grain.items()))}), {len(btris)} boundary triangles")
    return len(nodes), len(tets_out), len(btris)


# --------------------------------------------------------------------------------------
# grain-boundary analysis and bubbles
# --------------------------------------------------------------------------------------
def face_area(loop, verts):
    p0 = verts[loop[0] - 1]
    a = 0.0
    for i in range(1, len(loop) - 1):
        a += vnorm(vcross(vsub(verts[loop[i] - 1], p0),
                          vsub(verts[loop[i + 1] - 1], p0))) * 0.5
    return a


def analyse_gb(verts, face_uses, face_loop):
    """Grain-boundary faces (area) and triple lines (length)."""
    gb_faces = []
    n_seam = 0
    for key, grains in face_uses.items():
        if len(grains) != 2:
            continue
        if grains[0] == grains[1]:
            n_seam += 1
            continue
        gb_faces.append((face_area(face_loop[key], verts), sorted(grains), face_loop[key]))
    A_GB = sum(f[0] for f in gb_faces)
    if n_seam:
        print(f"  {n_seam} internal faces are periodic self-contacts of a grain "
              f"(welded in the mesh, not grain boundaries)")

    edge_gb = {}
    for key, grains in face_uses.items():
        if len(grains) != 2 or grains[0] == grains[1]:
            continue
        loop = face_loop[key]
        for i in range(len(loop)):
            a, b = loop[i], loop[(i + 1) % len(loop)]
            k = (min(a, b), max(a, b))
            edge_gb[k] = edge_gb.get(k, 0) + 1
    segs = []
    for k, c in edge_gb.items():
        if c == 3:                      # triple line: three grains meet
            segs.append((verts[k[0] - 1], verts[k[1] - 1]))
    tot_tj = sum(vnorm(vsub(a, b)) for a, b in segs)
    return gb_faces, A_GB, segs, tot_tj


def sample_polygon(loop, verts, rng):
    """Uniform point in a convex polygon (fan from vertex 0)."""
    p0 = verts[loop[0] - 1]
    tris = []
    tot = 0.0
    for i in range(1, len(loop) - 1):
        a = verts[loop[i] - 1]
        b = verts[loop[i + 1] - 1]
        tot += vnorm(vcross(vsub(a, p0), vsub(b, p0))) * 0.5
        tris.append((tot, a, b))
    t = rng.random() * tot
    i = 0
    while i < len(tris) - 1 and tris[i][0] < t:
        i += 1
    _, a, b = tris[i]
    u = rng.random()
    v = rng.random()
    if u + v > 1.0:
        u, v = 1.0 - u, 1.0 - v
    return vadd(p0, vadd(vmul(vsub(a, p0), u), vmul(vsub(b, p0), v)))


def place_bubbles(gb_faces, segs, verts, L, r, rng):
    """Aagesen Sec. 3.1: NGB = 20/um^2 on the grain boundaries, NTJ = 7.14/um on the
    triple lines; radius r; centres at least 2r apart under the periodic minimum image.

    Each bubble is drawn uniformly at random (segment weighted by length, uniform point in
    the polygon) and rejected when it comes closer than 2r to an already placed bubble,
    for at most 2000 tries. If the whole network is saturated at that spacing, the
    candidate that maximises the distance to the others is kept and the failure is
    counted (the callers reports it); this can happen because the Voronoi triple lines
    are much shorter than the extruded hexagon edges of the paper.
    """
    A_um2 = sum(f[0] for f in gb_faces) * 1e-6
    L_um = sum(vnorm(vsub(a, b)) for a, b in segs) * 1e-3
    nGB = int(round(20.0 * A_um2))
    nTJ = int(round(7.14 * L_um))
    rmin = 2.0 * r
    placed = []
    fails = []

    cum = []
    tot = 0.0
    for f in gb_faces:
        tot += f[0]
        cum.append(tot)
    lcum = []
    ltot = 0.0
    for s in segs:
        ltot += vnorm(vsub(s[0], s[1]))
        lcum.append(ltot)

    def min_dist(c):
        best = float("inf")
        for p in placed:
            d = [abs(c[k] - p[1][k]) for k in range(3)]
            d = [min(d[k], L[k] - d[k]) for k in range(3)]
            best = min(best, math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2))
        return best

    def pick(cum, total, rng):
        t = rng.random() * total
        i = 0
        while i < len(cum) - 1 and cum[i] < t:
            i += 1
        return i

    def sample_tj(rng):
        i = pick(lcum, ltot, rng)
        a, b = segs[i]
        return vadd(a, vmul(vsub(b, a), rng.random()))

    def sample_gb(rng):
        i = pick(cum, tot, rng)
        return sample_polygon(gb_faces[i][2], verts, rng)

    for kind, n, sampler in (("TJ", nTJ, sample_tj), ("GB", nGB, sample_gb)):
        for _ in range(n):
            best = None
            bestd = -1.0
            ok = False
            for _try in range(2000):
                c = sampler(rng)
                dd = min_dist(c)
                if dd >= rmin:
                    best, bestd, ok = c, dd, True
                    break
                if dd > bestd:
                    best, bestd = c, dd
            if not ok:
                fails.append((kind, bestd))
            placed.append((kind, best))

    return placed, nGB, nTJ, fails


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lx", type=float, default=600.0)
    ap.add_argument("--ly", type=float, default=600.0)
    ap.add_argument("--lz", type=float, default=280.0)
    ap.add_argument("--ngrains", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="seed placement seed (Merope)")
    ap.add_argument("--lc-coarse", type=float, default=40.0,
                    help="mesh size far from the grain boundaries")
    ap.add_argument("--lc-fine", type=float, default=16.0,
                    help="mesh size on the grain boundaries")
    ap.add_argument("--dist-max", type=float, default=15.0,
                    help="distance over which the size grades fine -> coarse")
    ap.add_argument("--bubble-seed", type=int, default=2019)
    ap.add_argument("--bubble-radius", type=float, default=44.0)
    ap.add_argument("--geo", default="poly.geo")
    ap.add_argument("--msh", default="poly.msh")
    ap.add_argument("--bubbles", default="bubbles.txt")
    ap.add_argument("--skip-mesh", action="store_true",
                    help="only rebuild poly.geo and bubbles.txt")
    args = ap.parse_args()

    L = [args.lx, args.ly, args.lz]
    eps = 1e-7 * max(L)

    print(f"seeds: {args.ngrains} spheres in {L[0]:g} x {L[1]:g} x {L[2]:g} nm "
          f"(sac_de_billes, seed {args.seed})")
    seeds = merope_seeds(L, args.ngrains, args.seed)

    grains = []
    vols = []
    for i, s in enumerate(seeds):
        cell = periodic_cell(s, seeds, L, eps)
        pieces = wrapped_pieces(cell, L, eps)
        grains.append(pieces)
        vols.append(sum(poly_volume(p) for p in pieces))
    boxvol = L[0] * L[1] * L[2]
    print(f"tessellation: {len(grains)} grains, {sum(len(g) for g in grains)} pieces, "
          f"volume {sum(vols):.6f} (box {boxvol:.6f}, "
          f"rel. err {abs(sum(vols) - boxvol) / boxvol:.2e})")
    if abs(sum(vols) - boxvol) > 1e-6 * boxvol:
        sys.exit("the wrapped cells do not tile the box")
    for i, v in enumerate(vols, start=1):
        print(f"  grain {i}: V = {v / 1e6:.4f} um^3, equivalent diameter "
              f"{2 * (3 * v / (4 * math.pi)) ** (1 / 3):.1f} nm")

    geo, verts, pieces_of_grain, face_loop, face_uses, box_faces, surf_ids, line_ids = \
        build_geo(grains, L, args.lc_coarse, args.lc_fine, args.dist_max, eps)
    with open(args.geo, "w") as f:
        f.write(geo)
    print(f"wrote {args.geo}: {len(verts)} points, {len(face_loop)} surfaces, "
          f"{len(box_faces)} box pieces")

    raw = os.path.splitext(args.msh)[0] + "_raw.msh"
    if not args.skip_mesh:
        cmd = ["gmsh", args.geo, "-3", "-format", "msh2", "-v", "2", "-o", raw]
        print("running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        finalize_mesh(raw, args.msh, L, args.ngrains)

    gb_faces, A_GB, segs, tot_tj = analyse_gb(verts, face_uses, face_loop)
    print(f"grain boundaries: A_GB = {A_GB:.6e} nm^2 = {A_GB * 1e-6:.4f} um^2 "
          f"({len(gb_faces)} faces); triple lines: {len(segs)} segments, "
          f"{tot_tj:.1f} nm = {tot_tj * 1e-3:.4f} um")

    rng = random.Random(args.bubble_seed)
    placed, nGB, nTJ, fails = place_bubbles(gb_faces, segs, verts, L,
                                            args.bubble_radius, rng)
    n_close = len(fails)
    print(f"bubbles: nTJ = {nTJ}, nGB = {nGB}, r = {args.bubble_radius:g} nm, "
          f"positions that could not keep the 2r spacing: {n_close}")

    rmin = 2 * args.bubble_radius
    dmin = None
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            d = [abs(placed[i][1][k] - placed[j][1][k]) for k in range(3)]
            d = [min(d[k], L[k] - d[k]) for k in range(3)]
            dd = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            dmin = dd if dmin is None else min(dmin, dd)
    if n_close:
        worst = min(d for _, d in fails)
        print(f"  WARNING: {n_close} bubble(s) closer than {rmin:g} nm to a neighbour "
              f"(closest {worst:.1f} nm): the Voronoi triple lines are short, the "
              f"7.14/um density saturates them")

    with open(args.bubbles, "w") as f:
        f.write("# fission-gas bubbles for test5: 3D periodic Voronoi polycrystal\n")
        f.write(f"# box (nm): {L[0]:g} {L[1]:g} {L[2]:g}\n")
        f.write(f"# ngrains: {args.ngrains}\n")
        f.write(f"# seed: {args.bubble_seed}\n")
        f.write(f"# A_GB (nm^2): {A_GB:.10g}  ({A_GB * 1e-6:.6f} um^2, "
                f"{len(gb_faces)} grain-boundary faces)\n")
        f.write(f"# triple-line length (nm): {tot_tj:.10g}  ({tot_tj * 1e-3:.6f} um, "
                f"{len(segs)} segments)\n")
        f.write(f"# nGB: {nGB} (20/um^2), nTJ: {nTJ} (7.14/um)\n")
        f.write(f"# r (nm): {args.bubble_radius:g}\n")
        f.write(f"# min center spacing (nm): {rmin:g} (requested)\n")
        f.write(f"# achieved min center spacing (nm): {dmin:.6f} "
                f"({n_close} position(s) could not keep {rmin:g})\n")
        f.write("# columns: x y z r (nm), triple-junction bubbles first\n")
        for kind, c in placed:
            f.write(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f} {args.bubble_radius:g}\n")
    print(f"wrote {args.bubbles}: {len(placed)} bubbles "
          f"(min periodic spacing {dmin:.3f} nm)")
    if not args.skip_mesh and os.path.exists(raw):
        os.remove(raw)


if __name__ == "__main__":
    main()
