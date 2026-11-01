#!/usr/bin/env python3
"""Finalize hex_raw.msh (gmsh -3 msh2 of the extruded 4-hexagon cell) into hex.msh.

- Volume elements (type 4 tets) are tagged with the grain: Physical Volume "grain_1..4"
  (dim-3 physical names), by nearest-center lookup of the centroid (the honeycomb is the
  Voronoi diagram of its hexagon centers).
- Boundary triangles (type 2 whose 3 nodes lie on one box face) get the dim-2 physical name
  "box". Internal faces (grain boundaries) are left untagged and are not boundary elements.
- A $Periodic section is written AFTER $Elements (the order gmsh uses: MFEM 4.9's v2 reader
  applies the slave->master mapping only to the elements read so far, and marks the mesh
  curved so the slave vertex coordinates are kept in the Nodes field, which is what keeps the
  geometry of the "wrapping" elements correct). Three dim-2 blocks pair the opposite box
  faces in x, y and z by nearest-neighbour node coordinates.

Usage: finalize_mesh.py --in hex_raw.msh --out hex.msh --a 80
"""

import argparse
import math
from collections import defaultdict

TET, TRI = 4, 2


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="hex_raw.msh")
    ap.add_argument("--out", default="hex.msh")
    ap.add_argument("--a", type=float, default=80.0, help="hexagon circumradius in nm")
    args = ap.parse_args()

    a = args.a
    nodes, elems = read_msh2(args.inp)
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    zs = [p[2] for p in nodes.values()]
    lx, ly, lz = max(xs), max(ys), max(zs)
    assert abs(lx - 3 * a) < 1e-6 * lx and abs(ly - 2 * math.sqrt(3) * a) < 1e-6 * ly, \
        f"mesh extent {lx} x {ly} does not match the cell of a = {a}"
    tol = 1e-9 * max(lx, ly, lz)

    tets_raw = [e[2] for e in elems if e[0] == TET]
    tris_raw = [e[2] for e in elems if e[0] == TRI]
    assert tets_raw, "no tetrahedra in the raw mesh"

    # ---- grain lookup: nearest hexagon center (honeycomb = Voronoi diagram of the centers)
    centers = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            cx, cy = 3 * i * a / 2, (2 * j + i % 2) * math.sqrt(3) * a / 2
            centers.append((cx, cy, (i % 2) + 2 * (j % 2) + 1))

    def grain_of(x, y):
        best, bg = None, None
        for cx, cy, g in centers:
            d = (x - cx) ** 2 + (y - cy) ** 2
            if best is None or d < best:
                best, bg = d, g
        assert best < (1.01 * a) ** 2, f"point ({x},{y}) not inside any hexagon"
        return bg

    tets = []  # (grain_tag, [4 node ids])
    for tet in tets_raw:
        cx = sum(nodes[t][0] for t in tet) / 4
        cy = sum(nodes[t][1] for t in tet) / 4
        tets.append((grain_of(cx, cy), tet))

    # ---- boundary triangles: all 3 nodes on one box face
    def on_box_face(tri):
        for axis, val in ((0, 0.0), (0, lx), (1, 0.0), (1, ly), (2, 0.0), (2, lz)):
            if all(abs(nodes[t][axis] - val) < tol for t in tri):
                return True
        return False

    # ---- conformity check: every tet face is shared by at most 2 tets, and the single-owned
    # faces are exactly the boundary triangles. Boundary triangles on the periodic seams can be
    # duplicated by gmsh (one copy per periodic image); keep only those adjacent to a tet.
    face_count = defaultdict(int)
    for _, tet in tets:
        for f in ((tet[1], tet[2], tet[3]), (tet[0], tet[2], tet[3]),
                  (tet[0], tet[1], tet[3]), (tet[0], tet[1], tet[2])):
            face_count[tuple(sorted(f))] += 1
    assert all(c <= 2 for c in face_count.values()), "non-conformal 3D mesh"
    c1_faces = {f for f, c in face_count.items() if c == 1}
    btris = [t for t in tris_raw
             if on_box_face(t) and tuple(sorted(t)) in c1_faces]
    assert {tuple(sorted(t)) for t in btris} == c1_faces, \
        "boundary faces do not match the boundary triangles"

    # ---- $Periodic blocks (written AFTER $Elements, as gmsh does: MFEM 4.9's v2 reader
    # applies the slave->master mapping to the elements read so far and marks the mesh curved,
    # keeping the slave coordinates in the Nodes field so wrapping elements keep the right
    # geometry): pair opposite box faces in x, y, z by nearest-neighbour coordinates.
    ptol = 1e-6 * max(lx, ly, lz)

    blocks = []
    for axis, length in ((0, lx), (1, ly), (2, lz)):
        master = [nid for nid, p in nodes.items() if abs(p[axis]) < tol]
        slave = [nid for nid, p in nodes.items() if abs(p[axis] - length) < tol]
        assert len(master) == len(slave), \
            f"axis {axis}: {len(slave)} slaves vs {len(master)} masters"
        used = set()
        pairs = []
        for s in slave:
            ps = nodes[s]
            best, bm = None, None
            for m in master:
                if m in used:
                    continue
                pm = nodes[m]
                d = sum((ps[k] - pm[k]) ** 2 for k in range(3) if k != axis)
                if best is None or d < best:
                    best, bm = d, m
            assert best < ptol ** 2, f"no periodic master for slave node {s} on axis {axis}"
            used.add(bm)
            pairs.append((s, bm))
        tx = (lx, 0, 0) if axis == 0 else ((0, ly, 0) if axis == 1 else (0, 0, lz))
        blocks.append((pairs, tx))
        print(f"axis {axis}: {len(pairs)} periodic node pairs")

    # ---- write hex.msh (gmsh 2.2 ASCII; $Periodic last, after $Elements)
    box_tag = 10
    with open(args.out, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$PhysicalNames\n5\n")
        for g in (1, 2, 3, 4):
            f.write(f'3 {g} "grain_{g}"\n')
        f.write(f'2 {box_tag} "box"\n')
        f.write("$EndPhysicalNames\n")
        f.write(f"$Nodes\n{len(nodes)}\n")
        for nid in sorted(nodes):
            x, y, z = nodes[nid]
            f.write(f"{nid} {x:.17g} {y:.17g} {z:.17g}\n")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{len(tets) + len(btris)}\n")
        eid = 0
        for g, tet in tets:
            eid += 1
            f.write(f"{eid} {TET} 2 {g} 1 {' '.join(map(str, tet))}\n")
        for t in btris:
            eid += 1
            f.write(f"{eid} {TRI} 2 {box_tag} 1 {' '.join(map(str, t))}\n")
        f.write("$EndElements\n")
        f.write("$Periodic\n3\n")
        for bi, (pairs, (tx, ty, tz)) in enumerate(blocks):
            f.write(f"2 {20 + bi} {bi + 1}\n")
            f.write(f"Affine 1 0 0 {tx:.17g} 0 1 0 {ty:.17g} 0 0 1 {tz:.17g} 0 0 0 1\n")
            f.write(f"{len(pairs)}\n")
            for s, m in pairs:
                f.write(f"{s} {m}\n")
        f.write("$EndPeriodic\n")

    per_grain = defaultdict(int)
    for g, _ in tets:
        per_grain[g] += 1
    print(f"{args.out}: {len(nodes)} nodes, {len(tets)} tets "
          f"(per grain: {dict(sorted(per_grain.items()))}), {len(btris)} boundary triangles, "
          f"box {lx:g} x {ly:g} x {lz:g} nm, $Periodic in x/y/z")


if __name__ == "__main__":
    main()
