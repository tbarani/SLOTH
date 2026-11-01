#!/usr/bin/env python3
"""Generate hex3d.geo: four hexagonal grains tiling the periodic rectangle [0,3a]x[0,2*sqrt(3)*a].

This is the x-y cross-section of the 3D microstructure of Fig. 3 of L.K. Aagesen, D. Schwen,
M.R. Tonks, Y. Zhang, "Phase-field modeling of fission gas bubble growth on grain boundaries and
triple junctions in UO2 nuclear fuel", Comput. Mater. Sci. 161 (2019) 35-45: a periodic array of
four uniform hexagonal grains, extruded in z (see extrude_mesh.py).

Geometry is built in an exact integer lattice: x = m*a/2, y = n*sqrt(3)*a/2, so the cell is the
rectangle m in [0,6], n in [0,4] (Lx = 3a, Ly = 2*sqrt(3)*a). Hexagonal grains are centered at
(3i, 2j + i%2); the four grains of the cell are the centers (0,0), (3,1), (0,2), (3,3), grain
index g = (i%2) + 2*(j%2). Each hexagon is clipped to the cell (Sutherland-Hodgman, exact
rational arithmetic) over its 9 periodic images, so the pieces tile the rectangle exactly and
opposite cell edges carry identical, paired line segments (Periodic Line). The 4 hexagons of the
cell have 8 triple points per layer, i.e. 8 vertical grain edges (triple lines) once extruded,
as in the paper.

Two layouts are available (--layout):
  corner (default in earlier revisions): hexagon centers at (3i, 2j + i%2), so one grain sits on
      the cell corner;
  center: centers at (3 + 3i, 2 + 2j + i%2), so one full grain is centered in the box and the
      other three are split in half by the periodic boundaries, as in Fig. 3 of the paper.

Usage: make_hex_geo.py [--a 80] [--lc 12] [--lz 120] [--nz 10] [--layout center] [--out hex3d.geo]
"""

import argparse
from fractions import Fraction

# CCW hexagon vertex offsets in lattice units (x = m*a/2, y = n*sqrt(3)*a/2)
OFFSETS = [(2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1)]
# Cell size in lattice units
CELL_M, CELL_N = 6, 4


def clip_halfplane(poly, axis, bound, keep_greater):
    """Sutherland-Hodgman clip of poly (list of (Fraction, Fraction)) against one half-plane."""
    out = []
    for k in range(len(poly)):
        p, q = poly[k], poly[(k + 1) % len(poly)]
        pin = (p[axis] >= bound) if keep_greater else (p[axis] <= bound)
        qin = (q[axis] >= bound) if keep_greater else (q[axis] <= bound)
        if pin:
            out.append(p)
        if pin != qin:
            # intersection with the boundary line axis == bound
            t = (bound - p[axis]) / (q[axis] - p[axis])
            other = 1 - axis
            out.append((bound, p[other] + t * (q[other] - p[other])) if axis == 0
                       else (p[other] + t * (q[other] - p[other]), bound))
    return out


def clip_cell(poly):
    poly = clip_halfplane(poly, 0, Fraction(0), True)
    poly = clip_halfplane(poly, 0, Fraction(CELL_M), False)
    poly = clip_halfplane(poly, 1, Fraction(0), True)
    poly = clip_halfplane(poly, 1, Fraction(CELL_N), False)
    return poly


def polygon_area(poly):
    s = Fraction(0)
    for k in range(len(poly)):
        p, q = poly[k], poly[(k + 1) % len(poly)]
        s += p[0] * q[1] - q[0] * p[1]
    return s / 2


def grain_of(center, layout):
    off = (3, 2) if layout == "center" else (0, 0)
    i = (center[0] - off[0]) // 3
    parity = i % 2
    j = (center[1] - off[1] - parity) // 2
    return (i % 2) + 2 * (j % 2) + 1  # grain_1..grain_4


def build_pieces(args):
    """Clip every hexagon of a patch covering the cell plus a margin; return pieces per grain.

    The patch (no extra periodic shifts, the patch itself contains the images) is large enough
    that the clipped pieces tile the cell exactly once.
    """
    pieces = {g: [] for g in (1, 2, 3, 4)}
    off = (3, 2) if args.layout == "center" else (0, 0)
    for i in range(-2, 4):
        for j in range(-2, 4):
            center = (off[0] + 3 * i, off[1] + 2 * j + i % 2)
            g = grain_of(center, args.layout)
            poly = [(center[0] + om, center[1] + on) for om, on in OFFSETS]
            piece = clip_cell(poly)
            if len(piece) >= 3 and polygon_area(piece) > 0:
                pieces[g].append(piece)
    return pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, default=80.0, help="hexagon circumradius in nm")
    ap.add_argument("--lc", type=float, default=12.0, help="mesh size in nm")
    ap.add_argument("--lz", type=float, default=120.0, help="extrusion length in nm")
    ap.add_argument("--nz", type=int, default=10, help="number of element layers in z")
    ap.add_argument("--layout", choices=["corner", "center"], default="center",
                    help="hexagon lattice origin: corner or one grain centered in the box")
    ap.add_argument("--out", default="hex3d.geo")
    args = ap.parse_args()

    a = args.a
    lx, ly = 3 * a, 2 * (3 ** 0.5) * a
    pieces = build_pieces(args)

    # sanity: the pieces tile the cell exactly, 4 grains of equal area
    total = sum(polygon_area(p) for g in pieces for p in pieces[g])
    assert total == CELL_M * CELL_N, f"area {total} != {CELL_M * CELL_N}"
    for g in pieces:
        area = sum(polygon_area(p) for p in pieces[g])
        assert area == CELL_M * CELL_N // 4, f"grain {g} area {area}"

    # points and lines, deduplicated by exact lattice coordinates
    points = {}   # (m, n) -> pid
    lines = {}    # (min pid, max pid) -> (lid, (p, q) canonical)
    pid = lid = 0

    def get_point(mn):
        nonlocal pid
        if mn not in points:
            pid += 1
            points[mn] = pid
        return points[mn]

    def get_line(p, q):
        nonlocal lid
        key = (min(p, q), max(p, q))
        if key not in lines:
            lid += 1
            lines[key] = (lid, (p, q))
        return lines[key]

    surfs = []  # (grain, [signed line ids])
    for g in sorted(pieces):
        for piece in pieces[g]:
            pids = [get_point(mn) for mn in piece]
            loop = []
            for k in range(len(pids)):
                p, q = pids[k], pids[(k + 1) % len(pids)]
                lid_, canon = get_line(p, q)
                loop.append(lid_ if canon == (p, q) else -lid_)
            surfs.append((g, loop))

    # boundary lines and their periodic twins
    pid2mn = {v: k for k, v in points.items()}
    bnd_lines = []  # (lid, axis, val, pmn, qmn) on the cell boundary
    for (p, q), (lid_, _canon) in lines.items():
        pmn, qmn = pid2mn[p], pid2mn[q]
        for axis, val in ((0, 0), (0, CELL_M), (1, 0), (1, CELL_N)):
            if pmn[axis] == val and qmn[axis] == val:
                bnd_lines.append((lid_, axis, val, pmn, qmn))

    periodic = []  # (slave lid, master lid, (tx, ty))
    for lid_, axis, val, pmn, qmn in bnd_lines:
        other = CELL_M if axis == 0 else CELL_N
        if val == 0:
            twin_mn = [(other if axis == 0 else pmn[0], other if axis == 1 else pmn[1]),
                       (other if axis == 0 else qmn[0], other if axis == 1 else qmn[1])]
            assert twin_mn[0] in points and twin_mn[1] in points, \
                f"missing periodic twin point for {pmn}-{qmn}"
            tp, tq = points[twin_mn[0]], points[twin_mn[1]]
            tkey = (min(tp, tq), max(tp, tq))
            assert tkey in lines, f"missing periodic twin line for {pmn}-{qmn}"
            twin = lines[tkey]
            tx, ty = (lx, 0.0) if axis == 0 else (0.0, ly)
            periodic.append((twin[0], lid_, (tx, ty)))

    with open(args.out, "w") as f:
        f.write(f"// Automatically generated by make_hex_geo.py (layout={args.layout}): 4 hexagonal grains, "
                f"cell [0,{lx:g}]x[0,{ly:g}]x[0,{args.lz:g}] nm, a = {a:g} nm, lc = {args.lc:g} nm\n")
        f.write(f"lc = {args.lc:g};\n")
        for mn, pp in sorted(points.items(), key=lambda kv: kv[1]):
            x, y = float(mn[0]) * a / 2.0, float(mn[1]) * (3 ** 0.5) * a / 2.0
            f.write(f"Point({pp}) = {{{x:.17g}, {y:.17g}, 0, lc}};\n")
        for (p, q), (lid_, canon) in sorted(lines.items(), key=lambda kv: kv[1][0]):
            f.write(f"Line({lid_}) = {{{canon[0]}, {canon[1]}}};\n")
        sid = 0
        for g, loop in surfs:
            sid += 1
            f.write(f"Line Loop({sid}) = {{{', '.join(str(l) for l in loop)}}};\n")
            f.write(f"Plane Surface({sid}) = {{{sid}}};\n")
        for slave, master, (tx, ty) in periodic:
            f.write(f"Periodic Line {{{slave}}} = {{{master}}} Translate {{{tx:g}, {ty:g}, 0}};\n")
        f.write(f"Extrude {{0, 0, {args.lz:g}}} {{ Surface{{:}}; Layers{{{args.nz}}}; }}\n")

    n_bnd = len(bnd_lines)
    print(f"{args.out}: {len(points)} points, {len(lines)} lines, {len(surfs)} surfaces, "
          f"{n_bnd} boundary lines, {len(periodic)} periodic pairs, "
          f"cell {lx:g} x {ly:g} x {args.lz:g} nm, nz = {args.nz}")


if __name__ == "__main__":
    main()
