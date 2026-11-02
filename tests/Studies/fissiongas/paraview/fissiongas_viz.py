#!/usr/bin/env python3
"""
fissiongas_viz.py -- fancy ParaView scenes for the SLOTH fission-gas studies
(tests/Studies/fissiongas: 2D Bubbles/Polycrystal, 3D Hex3D of Aagesen et al. 2019).

Headless (this machine has ParaView 5.11 with pvpython):

    pvpython fissiongas_viz.py -- --saves /path/to/Saves --out /path/to/viz [options]
    # or the wrapper, which also encodes the movie with ffmpeg:
    ./render.sh /path/to/Saves /path/to/viz [options]

In the ParaView GUI, simply open the generated scene.pvsm (File > Load State) and press
play: the time series is already wired into the animation scene.

The script finds every *.pvd under --saves and classifies them by their point arrays
(eta_b* -> order parameters, mu_g/mu_v -> chemical potentials, h_* -> switching functions),
so it works unchanged for the 2D runs (Saves/OrderParameters, ...) and for the 3D Hex3D run.

What it builds (one 2x2 layout, dark gradient background, FXAA + depth peeling):

  * hero view   : eta_b = 0.5 iso-surface (smoothed), colored by the gas chemical
                  potential mu_g resampled onto it, optionally replicated NX x NY x NZ
                  times to show the periodic images; the grain boundaries (eta_gi = 0.5)
                  are translucent shells.  In 2D the domain surface is colored by eta_b
                  and the bubbles / grain boundaries are drawn as tubes.
  * field view  : 3D: mid-z slice colored by mu_g with bubble outlines.
                  2D: the domain colored by mu_g.
  * chart view  : bubble volume (area in 2D) and the mu_g / mu_v inventories vs time,
                  computed live with Integrate Variables over the whole time series.
  * volume view : 3D only: volume rendering of eta_b with a custom opacity ramp.

Outputs in --out: scene.pvsm, stats.csv, preview.png, frames/frame_XXXX.png, movie.mp4.

Options (after the `--` separator):
  --saves DIR        root containing the *.pvd series (e.g. a run's Saves directory)
  --out DIR          output directory (default: <saves>/../viz)
  --mode LIST        comma list of state,stats,preview,frames,movie (default: all)
  --stride N         use every N-th time step (default 1)
  --width W --height H   frame size in pixels (default 1920 1080)
  --fps N            movie frame rate (default 10)
  --periodic NX,NY,NZ   replicate the hero geometry (default 3,3,3 in 3D, 3,3,1 in 2D;
                       1,1,1 disables the replication)
  --no-volume        skip the volume-rendering view
  --eta NAME         bubble order-parameter array (default: auto-detect eta_b*)
"""

import argparse
import glob
import math
import os
import shutil
import subprocess
import sys

from paraview.simple import *
import paraview.servermanager as sm
import builtins
_max = builtins.max  # paraview.simple shadows the builtins with numpy.max/min
_min = builtins.min


# --------------------------------------------------------------------------- helpers

def point_arrays(proxy):
    info = proxy.GetDataInformation()
    pd = info.GetPointDataInformation()
    return [pd.GetArrayInformation(i).GetName() for i in range(pd.GetNumberOfArrays())]


def open_readers(saves):
    """Open every *.pvd under `saves` and classify by array names."""
    pvds = sorted(glob.glob(os.path.join(saves, "**", "*.pvd"), recursive=True))
    if not pvds:
        raise SystemExit("no *.pvd found under %s" % saves)
    readers = {}
    for p in pvds:
        r = OpenDataFile(p)
        r.UpdatePipeline()
        readers[p] = r
    op = next((r for p, r in readers.items()
               if "switch" not in os.path.basename(p)
               and any(a.startswith("eta_b") for a in point_arrays(r))), None)
    if op is None:
        raise SystemExit("no order-parameter series (eta_b*) found under %s" % saves)
    op_arrs = set(point_arrays(op))
    # test4 (vtk_unified) writes one .pvd holding every field; the 2D runs write one
    # .pvd per problem. If mu_g/mu_v live in the op series, reuse it as their source.
    mug = op if "mu_g" in op_arrs else next(
        (r for p, r in readers.items() if "mu_g" in point_arrays(r)), None)
    muv = op if "mu_v" in op_arrs else next(
        (r for p, r in readers.items() if "mu_v" in point_arrays(r)), None)
    switch = next((r for p, r in readers.items()
                   if "switch" in os.path.basename(p)), None)
    if switch is None and {"h_b", "h_m"} & op_arrs:
        switch = op
    return {"op": op, "mug": mug, "muv": muv, "switch": switch,
            "all": list(readers.values())}


def merge(objs):
    objs = list(objs)
    if len(objs) == 1:
        return objs[0]
    return GroupDatasets(Input=objs)


_TRANSLATE_SCRIPT = """
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
# in pvpython, inputs[0]/output are paraview.vtk dataset adapters: unwrap .VTKObject
src = inputs[0].VTKObject if hasattr(inputs[0], "VTKObject") else inputs[0]
dst = output.VTKObject if hasattr(output, "VTKObject") else output
dst.DeepCopy(src)
off = np.array([%(dx).17g, %(dy).17g, %(dz).17g])
def shift(ds):
    if ds.IsA("vtkCompositeDataSet"):
        it = ds.NewIterator()
        it.InitTraversal()
        while not it.IsDoneWithTraversal():
            b = it.GetCurrentDataObject()
            if b is not None and b.IsA("vtkDataSet"):
                shift(b)
            it.GoToNextItem()
    elif ds.IsA("vtkDataSet"):
        p = ds.GetPoints()
        a = vtk_to_numpy(p.GetData()).copy() + off
        p.SetData(numpy_to_vtk(a, deep=True))
shift(dst)
"""


def translate(obj, dx, dy, dz):
    """A translated deep copy of obj (ProgrammableFilter; the Transform filter's proxy
    property is not settable from python in this build)."""
    pf = ProgrammableFilter(Input=obj)
    pf.Script = _TRANSLATE_SCRIPT % {"dx": dx, "dy": dy, "dz": dz}
    return pf


def replicate(obj, counts, bounds):
    """Periodic replication: NX x NY x NZ translated copies merged into one object."""
    lx = bounds[1] - bounds[0]
    ly = bounds[3] - bounds[2]
    lz = bounds[5] - bounds[4]
    offs = []
    for n, l in zip(counts, (lx, ly, lz)):
        offs.append([i * l for i in range(-(n // 2), n - n // 2)])
    copies = []
    for dx in offs[0]:
        for dy in offs[1]:
            for dz in offs[2]:
                if dx == 0 and dy == 0 and dz == 0:
                    copies.append(obj)
                else:
                    copies.append(translate(obj, dx, dy, dz))
    m = merge(copies)
    try:
        return MergeBlocks(Input=m)  # Calculator/Contour need a single block, not GroupDatasets
    except Exception:
        return m


def resample_onto(target, source_reader, array):
    """Sample `array` from source_reader onto target's points (ParaView >= 5.7 API:
    DestinationMesh keeps its geometry, SourceDataArrays provides the data)."""
    rs = ResampleWithDataset()
    rs.DestinationMesh = target
    rs.SourceDataArrays = [source_reader]
    rs.UpdatePipeline()
    d = sm.Fetch(rs)
    if d.GetPointData().GetArray(array) is not None:
        return rs
    Delete(rs)
    return target  # give up gracefully: keep target, caller colors by its own array


def make_lut(name, ramp, rng=None, opacity=None):
    """ramp: list of (fraction, (r, g, b)); opacity: list of (value, alpha)."""
    lut = GetColorTransferFunction(name)
    if rng is None:
        rng = (0.0, 1.0)
    pts = []
    for f, (r, g, b) in ramp:
        pts += [rng[0] + f * (rng[1] - rng[0]), r, g, b]  # vtk format: [value, r, g, b, ...]
    lut.RGBPoints = pts
    lut.ColorSpace = "RGB"
    if opacity is not None:
        pwf = lut.ScalarOpacityFunction
        pwf.Points = [v for pair in opacity for v in pair]
        if hasattr(lut, "EnableOpacityMapping"):
            lut.EnableOpacityMapping = True
    return lut


INFERNO = [(0.00, (0.00, 0.00, 0.05)), (0.25, (0.35, 0.05, 0.45)),
           (0.50, (0.70, 0.20, 0.35)), (0.75, (0.98, 0.55, 0.15)),
           (1.00, (1.00, 1.00, 0.85))]
CYAN = [(0.00, (0.05, 0.10, 0.45)), (0.35, (0.05, 0.55, 0.75)),
        (0.70, (0.55, 0.95, 0.85)), (1.00, (1.00, 1.00, 1.00))]


def style_view(v, depth_peeling=True, fxaa=True, multisamples=8):
    v.Background = [0.045, 0.055, 0.10]
    v.Background2 = [0.005, 0.005, 0.02]
    if hasattr(v, "UseColorPaletteForBackground"):
        v.UseColorPaletteForBackground = 0
    if hasattr(v, "BackgroundColorMode"):
        v.BackgroundColorMode = "Gradient"
    if hasattr(v, "UseFXAA"):
        v.UseFXAA = fxaa
    # NB: depth peeling breaks GPU volume rendering in this offscreen build, so the
    # volume view must be styled with depth_peeling=False.
    if depth_peeling:
        for prop in ("UseDepthPeeling", "DepthPeeling"):
            if hasattr(v, prop):
                setattr(v, prop, True)
                break
    if hasattr(v, "MultiSamples"):
        v.MultiSamples = multisamples


def style_rep(rep, color_array=None, lut=None, opacity=1.0, solid=None,
              specular=0.6, power=40.0):
    rep.Representation = "Surface"
    if solid is not None:
        rep.ColorArrayName = ["POINTS", ""]
        rep.DiffuseColor = solid
    elif color_array is not None:
        rep.ColorArrayName = ["POINTS", color_array]
        if lut is not None:
            rep.LookupTable = lut
    rep.Opacity = opacity
    rep.Ambient = 0.25
    rep.Diffuse = 0.9
    rep.Specular = specular
    if hasattr(rep, "SpecularPower"):
        rep.SpecularPower = power


def set_camera(v, bounds, angle=0.0, zoom=1.0, is3d=True):
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])
    d = _max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1e-12)
    v.ResetCamera()
    if is3d:
        if angle == 0.0:
            angle, zoom = 0.55, 0.78  # canonical 3/4 perspective, not dead-on the X axis
        r = 1.7 * d * zoom
        v.CameraPosition = [cx + r * math.cos(angle), cy + r * math.sin(angle),
                            cz + 0.65 * r]
        v.CameraViewUp = [0, 0, 1]
        v.CameraViewAngle = 30.0
    else:
        # 2D: orthographic top-down view of the (replicated) domain, no perspective crop
        if hasattr(v, "CameraParallelProjection"):
            v.CameraParallelProjection = True
            v.CameraParallelScale = 0.58 * d * zoom
        v.CameraPosition = [cx, cy, cz + 2.3 * d * zoom]
        v.CameraViewUp = [0, 1, 0]
    v.CameraFocalPoint = [cx, cy, cz]
    try:
        if hasattr(v, "CameraClippingRange"):
            r = 3.0 * d
            v.CameraClippingRange = [0.05 * r, 5.0 * r]
    except Exception:
        pass


def array_range(reader, name):
    """Range of a point array at the reader's current time (via data information,
    falling back to a fetch, then to (0, 1))."""
    try:
        info = reader.GetDataInformation()
        pd = info.GetPointDataInformation()
        for i in range(pd.GetNumberOfArrays()):
            ai = pd.GetArrayInformation(i)
            if ai.GetName() == name:
                r = ai.GetComponentRange(0)
                return (float(r[0]), float(r[1]))
    except Exception:
        pass
    reader.UpdatePipeline()
    d = sm.Fetch(reader)
    arr = d.GetPointData().GetArray(name)
    if arr is None and hasattr(d, "GetBlock") and d.GetBlock(0) is not None:
        arr = d.GetBlock(0).GetPointData().GetArray(name)
    if arr is None:
        return (0.0, 1.0)
    return tuple(arr.GetRange()[:2])


def integrated(iv, t):
    """Return dict of the scalar IntegrateVariables output arrays at time t."""
    iv.UpdatePipeline(time=t)
    d = sm.Fetch(iv)
    out = {}
    for attr in (d.GetCellData(), d.GetPointData(), d.GetFieldData()):
        for i in range(attr.GetNumberOfArrays()):
            a = attr.GetArray(i)
            nm = a.GetName()
            if a.GetNumberOfComponents() == 1 and a.GetNumberOfTuples() >= 1:
                try:
                    out[nm] = float(a.GetValue(0))
                except (TypeError, ValueError):
                    pass
    return out


def split_and_fill(layout, view, vertical, frac, new_view):
    """Split view's cell: `view` moves to the returned location, the new empty cell is
    one past it; assign `new_view` there. Returns view's new location."""
    r = layout.SplitViewVertical(view, frac) if vertical \
        else layout.SplitViewHorizontal(view, frac)
    layout.AssignView(r + 1, new_view)
    return r


# --------------------------------------------------------------------------- scene

def build(args):
    readers = open_readers(args.saves)
    op = readers["op"]
    op.UpdatePipeline()
    info = op.GetDataInformation()
    bounds = info.GetBounds()
    arrs = point_arrays(op)
    eta_b = args.eta or next(a for a in arrs if a.startswith("eta_b"))
    grains = sorted(a for a in arrs if a.startswith("eta_m") or a.startswith("eta_g"))
    is3d = (bounds[5] - bounds[4]) > 1e-9 * _max(bounds[1] - bounds[0],
                                                   bounds[3] - bounds[2], 1.0)
    counts = tuple(int(x) for x in args.periodic.split(",")) if args.periodic else (
        (2, 2, 1) if is3d else (3, 3, 1))
    times = list(op.TimestepValues)[:: args.stride]
    lx, ly, lz = bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]
    # bounds of the replicated hero geometry, centered on the actual copies
    rbounds = [bounds[0] - (counts[0] // 2) * lx,
               bounds[1] + (counts[0] - 1 - counts[0] // 2) * lx,
               bounds[2] - (counts[1] // 2) * ly,
               bounds[3] + (counts[1] - 1 - counts[1] // 2) * ly,
               bounds[4] - (counts[2] // 2) * lz,
               bounds[5] + (counts[2] - 1 - counts[2] // 2) * lz]

    print("saves=%s  eta_b=%s  grains=%s  3D=%s  periodic=%s  timesteps=%d"
          % (args.saves, eta_b, grains, is3d, counts, len(times)))

    GOLD = (1.00, 0.76, 0.20)
    GB_BLUE = (0.62, 0.72, 0.86)
    sw_reader = readers["switch"] if readers["switch"] is not None else (
        op if "h_b" in arrs else None)
    has_switch = sw_reader is not None

    # bubble outline (single cell), used for the bubble count and field overlays
    contour = Contour(Input=op)
    contour.ContourBy = ["POINTS", eta_b]
    contour.Isosurfaces = [0.5]
    contour.ComputeNormals = True

    # ---- stats over the whole time series (leaves readers at the last timestep) ----
    th = Threshold(Input=op)
    th.Scalars = ["POINTS", eta_b]
    th.ThresholdMethod = "Between"
    th.LowerThreshold = 0.5
    th.UpperThreshold = 1.1
    ivs = [("bubble", IntegrateVariables(Input=th))]
    mu_readers = {"mu_g": readers["mug"], "mu_v": readers["muv"]}
    for key, rd in mu_readers.items():
        if rd is not None:
            ivs.append((key, IntegrateVariables(Input=rd)))
    # Aagesen et al. 2019 coverage metrics (their Figs. 6-8): grain-boundary coverage
    # X_GB from the Phi_GB = 0.14 surface, triple-junction coverage X_TJ (3D) from the
    # Phi_GB = 0.27 line
    pairs = ["%s*%s" % (grains[i], grains[j])
             for i in range(len(grains)) for j in range(i + 1, len(grains))]
    phi_expr = " + ".join(pairs) if pairs else None  # Phi_GB: 0 in bulks and in bubbles,
    # 0.25 on 2-grain faces, 1/3 at 3-grain triple junctions
    calc_phi = Calculator(Input=op) if phi_expr else None
    if calc_phi is not None:
        calc_phi.ResultArrayName = "phi_poly"
        calc_phi.Function = phi_expr
    iv_gb = iv_tj = conn = None
    if calc_phi is not None:
        cgb = Contour(Input=calc_phi)
        cgb.ContourBy = ["POINTS", "phi_poly"]
        cgb.Isosurfaces = [0.14]
        iv_gb = IntegrateVariables(Input=cgb)
        if is3d:
            ctj = Contour(Input=calc_phi)
            ctj.ContourBy = ["POINTS", "phi_poly"]
            ctj.Isosurfaces = [0.27]
            iv_tj = IntegrateVariables(Input=ctj)
    conn = Connectivity(Input=contour)  # RegionId per connected bubble outline
    tk = GetTimeKeeper()
    rows = []
    for t in times:
        tk.Time = t
        for r in readers["all"]:
            r.UpdatePipeline(time=t)
        row = {"time": t}
        for key, iv in ivs:
            vals = integrated(iv, t)
            if key == "bubble":
                meas = vals.get("Volume", vals.get("Area", vals.get("Length")))
                if meas is None:
                    meas = vals.get(eta_b)
            else:
                meas = vals.get(key)
            row[key] = meas
        if iv_gb is not None:
            gbv = integrated(iv_gb, t)
            area = gbv.get("Area", gbv.get("Volume"))
            hb = gbv.get("h_b")
            row["X_GB"] = (hb / area) if (hb is not None and area) else None
            if iv_tj is not None:
                tjv = integrated(iv_tj, t)
                ln = tjv.get("Length", tjv.get("Volume"))
                hb2 = tjv.get("h_b")
                row["X_TJ"] = (hb2 / ln) if (hb2 is not None and ln) else None
        try:
            conn.UpdatePipeline(time=t)
            d = sm.Fetch(conn)
            rid = d.GetPointData().GetArray("RegionId")
            if rid is None:
                rid = d.GetCellData().GetArray("RegionId")
            row["N_b"] = (int(rid.GetRange()[1]) + 1) if rid is not None else None
        except Exception:
            row["N_b"] = None
        rows.append(row)

    def nice_range(key):
        # readers are at the LAST time step here: range of the final state
        if mu_readers.get(key) is None:
            return (0.0, 1.0)
        lo, hi = array_range(mu_readers[key], key)
        if not hi > lo:
            hi = lo + (abs(lo) * 1e-3 if lo else 1.0)
        return (lo, hi)

    # ---- pipeline on the replicated reader (only readers are replicated: replicating
    #      filter outputs segfaults VTK on some meshes) ----
    op_rep = replicate(op, counts, bounds)
    contour_rep = Contour(Input=op_rep)
    contour_rep.ContourBy = ["POINTS", eta_b]
    contour_rep.Isosurfaces = [0.5]
    contour_rep.ComputeNormals = True
    smooth_rep = Smooth(Input=contour_rep)
    smooth_rep.NumberofIterations = 15
    bubbles = smooth_rep

    # grain-boundary network from phi_poly (single surface, gap-free at triple junctions)
    calc_phi_rep = Calculator(Input=op_rep) if phi_expr else None
    gb_faces = tj_lines = None
    if calc_phi_rep is not None:
        calc_phi_rep.ResultArrayName = "phi_poly"
        calc_phi_rep.Function = phi_expr
        gb_faces = Contour(Input=calc_phi_rep)
        gb_faces.ContourBy = ["POINTS", "phi_poly"]
        gb_faces.Isosurfaces = [0.14]
        tj = Contour(Input=calc_phi_rep)
        tj.ContourBy = ["POINTS", "phi_poly"]
        tj.Isosurfaces = [0.27 if is3d else 0.24]
        tj_lines = tj

    # ---- layout: hero (top-left), field (top-right), chart (bottom-left), sec (bottom-right)
    layout = CreateLayout("fissiongas")
    hero = CreateView("RenderView")
    style_view(hero)
    layout.AssignView(0, hero)

    # hero: gold eta_b bubbles (Aagesen style) + blue-grey grain faces + h_b-colored
    # triple-junction tubes
    if is3d:
        style_rep(Show(bubbles, hero), solid=GOLD, opacity=1.0, specular=0.55, power=35)
    else:
        tb = Tube(Input=bubbles)
        tb.Radius = 0.004 * _max(lx, ly)
        tb.NumberofSides = 12
        style_rep(Show(tb, hero), solid=GOLD)
        style_rep(Show(op_rep, hero), color_array=eta_b,
                  lut=make_lut(eta_b, CYAN, rng=(0.0, 1.0)))
    if gb_faces is not None:
        if is3d:
            style_rep(Show(gb_faces, hero), solid=GB_BLUE, opacity=0.18, specular=0.2)
        else:
            tg = Tube(Input=gb_faces)
            tg.Radius = 0.0015 * _max(lx, ly)
            tg.NumberofSides = 8
            rep_gb = Show(tg, hero)
            style_rep(rep_gb, solid=GB_BLUE, opacity=0.55)
            try:
                rep_gb.Position = [0.0, 0.0, 1e-3 * _max(lx, ly)]  # avoid z-fighting with the domain surface
                rep_gb.LineWidth = 2.0
            except Exception:
                pass
    if tj_lines is not None and has_switch:
        if is3d:
            # the Phi_GB = 0.27 isosurface is already a thin tube mesh around each
            # triple line: show it directly, colored by h_b
            style_rep(Show(tj_lines, hero), color_array="h_b",
                      lut=make_lut("h_b_tj", INFERNO, rng=(0.0, 1.0)), opacity=0.9)
        else:
            tt = Tube(Input=tj_lines)
            tt.Radius = 0.004 * _max(lx, ly)
            tt.NumberofSides = 10
            style_rep(Show(tt, hero), color_array="h_b",
                      lut=make_lut("h_b_tj", INFERNO, rng=(0.0, 1.0)), opacity=0.9)

    # ---- field view: mu_g on a single unit cell + gas-flux glyphs ----
    field = CreateView("RenderView")
    style_view(field)
    split_and_fill(layout, hero, vertical=False, frac=0.62, new_view=field)
    if is3d:
        sl = Slice(Input=op)
        sl.SliceType = "Plane"
        sl.SliceType.Origin = [(bounds[0] + bounds[1]) / 2,
                               (bounds[2] + bounds[3]) / 2,
                               (bounds[4] + bounds[5]) / 2]
        sl.SliceType.Normal = [0, 0, 1]
        fld = resample_onto(sl, readers["mug"], "mu_g") if readers["mug"] else sl
    else:
        fld = resample_onto(op, readers["mug"], "mu_g") if readers["mug"] else op
    farr = "mu_g" if readers["mug"] else eta_b
    lut_field = make_lut(farr, INFERNO,
                         rng=nice_range("mu_g") if farr == "mu_g" else (0.0, 1.0))
    style_rep(Show(fld, field), color_array=farr, lut=lut_field)
    if is3d:
        csl = Contour(Input=sl)
        csl.ContourBy = ["POINTS", eta_b]
        csl.Isosurfaces = [0.5]
        tb2 = Tube(Input=csl)
    else:
        tb2 = Tube(Input=contour)
    tb2.Radius = 0.0025 * _max(lx, ly)
    tb2.NumberofSides = 10
    style_rep(Show(tb2, field), solid=(1.0, 1.0, 1.0), opacity=0.55)
    # flux glyphs: J_g ~ -grad(mu_g) on the field
    try:
        grad = ComputeDerivatives(Input=fld)
        grad.Scalars = ["POINTS", farr]
        c2p = CellDatatoPointData(Input=grad)
        calcg = Calculator(Input=c2p)
        calcg.ResultArrayName = "flux"
        calcg.Function = "-1.0*ScalarGradient"
        glyph = Glyph(Input=calcg)
        glyph.OrientationArray = ["POINTS", "flux"]
        glyph.ScaleArray = ["POINTS", "flux"]
        glyph.ScaleFactor = 0.08 * _max(lx, ly, lz)
        if hasattr(glyph, "GlyphMode"):
            try:
                glyph.GlyphMode = "Uniform Spatial Distribution"
                glyph.MaximumNumberOfSamplePoints = 150
            except Exception:
                glyph.GlyphMode = "Every Nth Point"
        style_rep(Show(glyph, field), solid=(1.0, 1.0, 1.0), opacity=0.75)
    except Exception as e:
        print("flux glyphs skipped:", e)

    # ---- chart: relative change + Aagesen coverage metrics ----
    chart = CreateView("XYChartView")
    split_and_fill(layout, hero, vertical=True, frac=0.60, new_view=chart)
    stats_path = os.path.join(args.out, "stats.csv")
    os.makedirs(args.out, exist_ok=True)
    keys = [k for k, _ in ivs]
    for k in keys:
        ref = next((r.get(k) for r in rows if r.get(k) not in (None, 0)), None)
        for row in rows:
            v = row.get(k)
            if v is None or ref in (None, 0):
                row[k + "_rel"] = None
            elif row["time"] == times[0] and v == 0:
                row[k + "_rel"] = 0.0  # avoid the artificial -100% spike at t = 0
            else:
                row[k + "_rel"] = 100.0 * (v / ref - 1.0)
    for row in rows:
        for k in ("X_GB", "X_TJ"):
            if row.get(k) is not None:
                row[k + "_pct"] = 100.0 * row[k]
    extra = [k for k in ("X_GB_pct", "X_TJ_pct", "N_b") if rows and rows[0].get(k) is not None]
    cols = keys + [k + "_rel" for k in keys] + extra
    with open(stats_path, "w") as f:
        f.write("time," + ",".join(cols) + "\n")
        for row in rows:
            f.write("%.6g" % row.get("time", 0.0) + "," +
                    ",".join("%.6g" % row[c] if row.get(c) is not None else ""
                             for c in cols) + "\n")
    rep_c = Show(OpenDataFile(stats_path), chart)
    rep_c.AttributeType = "Row Data"
    if hasattr(rep_c, "XArrayName"):
        rep_c.XArrayName = "time"
    series = [k + "_rel" for k in keys] + extra
    try:
        rep_c.SeriesVisibility = series
        rep_c.SeriesLineThickness = 3
        rep_c.SeriesMarkerStyle = "None"
        try:
            rep_c.SeriesLineStyle = ["mu_g_rel", "2"]  # dashed: visible under mu_v_rel
        except Exception:
            pass
    except Exception:
        pass
    for prop, val in (("ChartTitle", "bubble growth, inventories & coverage"),
                      ("ShowLegend", True),
                      ("LeftAxisTitle", "change / coverage [%]"),
                      ("BottomAxisTitle", "time")):
        if hasattr(chart, prop):
            try:
                setattr(chart, prop, val)
            except Exception:
                pass
    try:
        plot_vals = [row[c] for row in rows for c in series if row.get(c) is not None]
        if plot_vals and len(times) >= 2:
            lo, hi = _min(plot_vals), _max(plot_vals)
            pad = _max(0.1 * (hi - lo), 1.0)
            chart.LeftAxisRangeMinimum = lo - pad
            chart.LeftAxisRangeMaximum = hi + pad
        else:
            chart.LeftAxisRangeMinimum = -1.0
            chart.LeftAxisRangeMaximum = 1.0
        chart.LeftAxisUseCustomRange = True
        chart.BottomAxisUseCustomRange = True
        t0, t1 = times[0], times[-1]
        chart.BottomAxisRangeMinimum = t0
        chart.BottomAxisRangeMaximum = t1 if t1 > t0 else t0 + 1.0
        chart.BottomAxisLabelNotation = "Fixed"
        chart.BottomAxisLabelPrecision = 0
        if len(times) < 2 and hasattr(chart, "ShowBottomAxisLabels"):
            chart.ShowBottomAxisLabels = 0  # single-step runs: hide the degenerate 0,0,0,1,1,1 ticks
        for prop, val in (("LeftAxisTitleFontSize", 10), ("BottomAxisTitleFontSize", 10),
                          ("LeftAxisLabelFontSize", 9), ("BottomAxisLabelFontSize", 9),
                          ("ChartTitleFontSize", 11)):
            if len(times) < 2 and prop == "BottomAxisLabelFontSize":
                continue
            if hasattr(chart, prop):
                try:
                    setattr(chart, prop, val)
                except Exception:
                    pass
    except Exception:
        pass

    # ---- bottom-right panel ----
    sec = None
    if is3d and (has_switch or not args.no_volume):
        sec = CreateView("RenderView")
        if has_switch:
            style_view(sec)  # polygonal translucent shells need depth peeling
        else:
            style_view(sec, depth_peeling=False, fxaa=False, multisamples=0)
        split_and_fill(layout, field, vertical=True, frac=0.5, new_view=sec)
        if has_switch:
            # single unit cell: switching function h_b as nested contours 0.2/0.5/0.8
            # with the eta_b = 0.5 bubble surface on top (Aagesen-style, their Fig. 3)
            sw_objs = []
            for iso in (0.8, 0.5, 0.2):
                cs = Contour(Input=op)
                cs.ContourBy = ["POINTS", "h_b"]
                cs.Isosurfaces = [iso]
                sw_objs.append(cs)
            style_rep(Show(merge(sw_objs), sec), solid=(1.0, 0.80, 0.35),
                      opacity=0.35, specular=0.3)
            style_rep(Show(contour, sec), solid=(1.0, 1.0, 1.0),
                      opacity=0.9, specular=0.5)
            if calc_phi is not None:
                cgb2 = Contour(Input=calc_phi)
                cgb2.ContourBy = ["POINTS", "phi_poly"]
                cgb2.Isosurfaces = [0.14]
                style_rep(Show(cgb2, sec), solid=GB_BLUE, opacity=0.15, specular=0.1)
        else:
            rep_v = Show(op, sec)
            rep_v.Representation = "Volume"
            rep_v.ColorArrayName = ["POINTS", eta_b]
            rep_v.LookupTable = make_lut(
                eta_b + "_vol", CYAN, rng=(0.0, 1.0),
                opacity=[(0.0, 0.0), (0.55, 0.0), (0.72, 0.18), (0.88, 0.55), (1.0, 0.95)])
        set_camera(sec, bounds, angle=0.9, zoom=1.0, is3d=True)
    elif not is3d:
        sec = CreateView("RenderView")
        style_view(sec)
        split_and_fill(layout, field, vertical=True, frac=0.5, new_view=sec)
        # single unit cell: mu_v (or h_b) field + h_b contour lines 0.2/0.5/0.8
        if readers["muv"] is not None and readers["muv"] is not op:
            sec_src = resample_onto(op, readers["muv"], "mu_v")
            sarr, srange = "mu_v", nice_range("mu_v")
        elif has_switch:
            sec_src, sarr, srange = sw_reader, "h_b", (0.0, 1.0)
        else:
            sec_src, sarr, srange = op, eta_b, (0.0, 1.0)
        style_rep(Show(sec_src, sec), color_array=sarr,
                  lut=make_lut(sarr, INFERNO, rng=srange))
        if has_switch:
            for iso in (0.2, 0.5, 0.8):
                cs = Contour(Input=sw_reader)
                cs.ContourBy = ["POINTS", "h_b"]
                cs.Isosurfaces = [iso]
                tbs = Tube(Input=cs)
                tbs.Radius = (0.0032 if iso == 0.5 else 0.0022) * _max(lx, ly)
                tbs.NumberofSides = 10
                style_rep(Show(tbs, sec),
                          solid=(1.0, 0.80, 0.35) if iso == 0.5 else (1.0, 1.0, 1.0),
                          opacity=0.95 if iso == 0.5 else 0.5)
        set_camera(sec, bounds, is3d=False)

    # ---- scalar bar: mu_g on the field view in 3D, eta_b on the hero in 2D ----
    try:
        if is3d and readers["mug"]:
            sb = GetScalarBar(lut_field, field)
            sb.Title = "mu_g"
        else:
            sb = GetScalarBar(make_lut(eta_b, CYAN, rng=(0.0, 1.0)), hero)
            sb.Title = eta_b
        sb.Visibility = True
        sb.ComponentTitle = ""
        sb.Orientation = "Vertical"
        sb.Position = [0.89, 0.12]
        if hasattr(sb, "ScalarBarLength"):
            sb.ScalarBarLength = 0.32
        if hasattr(sb, "ScalarBarThickness"):
            sb.ScalarBarThickness = 12
        if hasattr(sb, "TitleFontSize"):
            sb.TitleFontSize = 11
        if hasattr(sb, "LabelFontSize"):
            sb.LabelFontSize = 9
    except Exception as e:
        print("scalar bar skipped:", e)

    # ---- annotations: hero time + caption, panel titles ----
    def add_text(view, text, corner, size, color):
        t = Text()
        t.Text = text
        r = Show(t, view)
        try:
            r.WindowLocation = corner
            r.Position = [14, 10]
            r.FontSize = size
            r.Color = color
        except Exception:
            pass
        return t

    txt = add_text(hero, "t = %g" % times[-1], "Upper Left Corner", 15,
                   [0.92, 0.93, 0.97])
    add_text(hero, "SLOTH fission gas | eta_b = 0.5 gold | h_b contours 0.2/0.5/0.8"
             if has_switch else "SLOTH fission gas | eta_b = 0.5",
             "Lower Left Corner", 10, [0.60, 0.66, 0.78])
    add_text(field, "mu_g" + (" (mid-z slice)" if is3d else ""), "Upper Left Corner",
             11, [0.80, 0.84, 0.92])
    if sec is not None:
        add_text(sec, "h_b contours 0.2/0.5/0.8" if (is3d and has_switch)
                 else ("volume rendering of eta_b" if is3d else "mu_v / h_b + h_b contours"),
                 "Upper Left Corner", 11, [0.80, 0.84, 0.92])

    # hide orientation gizmos except on the 3D hero
    for v in (field, sec):
        if v is not None and hasattr(v, "OrientationAxesVisibility"):
            v.OrientationAxesVisibility = 0
    if not is3d and hasattr(hero, "OrientationAxesVisibility"):
        hero.OrientationAxesVisibility = 0

    # animation scene so the .pvsm plays in the GUI
    scene = GetAnimationScene()
    scene.PlayMode = "Sequence"

    set_camera(hero, rbounds, is3d=is3d)
    set_camera(field, bounds, angle=0.65, is3d=is3d)

    return dict(readers=readers, op=op, bounds=bounds, rbounds=rbounds, is3d=is3d,
                times=times, rows=rows, layout=layout, hero=hero, field=field, sec=sec,
                txt=txt, tk=tk, eta_b=eta_b, stats_path=stats_path)


# --------------------------------------------------------------------------- main

def main(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--saves", default=None,
                   help="root containing the *.pvd series (default: active source "
                        "or ./Saves)")
    p.add_argument("--out", default=None)
    p.add_argument("--mode", default="all",
                   help="comma list of state,stats,preview,frames,movie")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--periodic", default=None, help="NX,NY,NZ (default 3,3,1)")
    p.add_argument("--no-volume", action="store_true")
    p.add_argument("--eta", default=None)
    args, _ = p.parse_known_args(argv)

    if args.saves is None:
        src = GetActiveSource()
        if src is not None and getattr(src, "FileName", None):
            args.saves = os.path.dirname(src.FileName[0])
        else:
            args.saves = "Saves"
    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(args.saves)), "viz")
    os.makedirs(args.out, exist_ok=True)
    modes = set(args.mode.split(",")) if args.mode != "all" else \
        {"state", "stats", "preview", "frames", "movie"}

    ctx = build(args)
    W, H = args.width, args.height

    if "state" in modes:
        SaveState(os.path.join(args.out, "scene.pvsm"))
        print("wrote", os.path.join(args.out, "scene.pvsm"))

    if "preview" in modes or "frames" in modes or "movie" in modes:
        frames_dir = os.path.join(args.out, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        # canonical preview first (hero camera on the replicated bounds)
        set_camera(ctx["hero"], ctx["rbounds"], is3d=ctx["is3d"])
        if "preview" in modes:
            Render()
            SaveScreenshot(os.path.join(args.out, "preview.png"),
                           ctx["layout"], ImageResolution=[W, H])
            print("wrote", os.path.join(args.out, "preview.png"))
        n = len(ctx["times"])
        for i, t in enumerate(ctx["times"]):
            ctx["tk"].Time = t
            for r in ctx["readers"]["all"]:
                r.UpdatePipeline(time=t)
            ctx["txt"].Text = "t = %g" % t
            ang = 2.0 * math.pi * i / _max(n, 1) if ctx["is3d"] else 0.0
            set_camera(ctx["hero"], ctx["rbounds"], angle=ang,
                       is3d=ctx["is3d"])
            Render()
            if "frames" in modes or "movie" in modes:
                SaveScreenshot(os.path.join(frames_dir, "frame_%04d.png" % i),
                               ctx["layout"], ImageResolution=[W, H])
        if "movie" in modes and shutil.which("ffmpeg"):
            mp4 = os.path.join(args.out, "movie.mp4")
            subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps),
                            "-i", os.path.join(frames_dir, "frame_%04d.png"),
                            "-vf", "format=yuv420p", "-crf", "18", mp4],
                           check=True)
            print("wrote", mp4)


if __name__ == "__main__":
    main(sys.argv)
