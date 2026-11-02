# Post-processing the fission-gas studies

Every test writes a ParaView-ready time series plus CSV tables into its run directory:

- `Saves/<Problem>/<Name>.pvd` (+ `CycleNNNNNN/data.pvtu` pieces) — the fields
  (`eta_b`, `eta_g1..4` or `eta_m1/2`, `h_b`, `h_m`, `mu_g`, `mu_v`), one per save
  (`-freq` option). `test4` writes a single unified `Saves/Hex3D/Hex3D.pvd` with all
  fields. The `.pvd` is the file to open in ParaView (it carries the time steps).
- `Saves/<Problem>/time_specialized.csv` — per-step integrals (bubble measure,
  `mu_g`/`mu_v` inventories, energies), compared against `ref/` by ctest.
- Rerun in a dirty directory only after `rm -rf Saves` — `time_specialized.csv` is
  appended to, never truncated.

## Fancy scenes + movies (paraview/)

`paraview/fissiongas_viz.py` builds a ready-made 2x2 scene (hero bubbles on the grain
network, `mu_g` field with flux glyphs, `h_b` switching-function contours, coverage
chart) and renders frames/movies headless:

```bash
cd tests/Studies/fissiongas/paraview
./render.sh ../../../build/tests/Studies/fissiongas/test4/Saves ../test4_viz   # any Saves dir
# or: pvpython fissiongas_viz.py -- --saves <Saves> --out <out> [--stride 4 --periodic 3,3,1]
```

Outputs: `scene.pvsm` (open in the ParaView GUI and press play), `preview.png`,
`stats.csv`, `frames/`, `movie.mp4`. See `paraview/README.md` for all options.

## Plain ParaView GUI

`File > Open` the `.pvd`, color by `eta_b` (or `h_b`), add a *Contour* at 0.5 for the
bubble surface, and use the animation controls — the time steps are already wired in.
