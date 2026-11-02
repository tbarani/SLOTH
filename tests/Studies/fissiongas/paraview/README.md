# ParaView visualization for the fission-gas studies

`fissiongas_viz.py` builds a presentation-ready ParaView scene from any run's `Saves`
directory (it finds every `*.pvd` under the given root and classifies them by their point
arrays, so it works unchanged for the 2D `Bubbles`/`Polycrystal` runs and for the 3D
`Hex3D` run of `test4`):

- **hero view**: the `eta_b = 0.5` iso-surface (smoothed) in gold, on the periodic
  replication (`--periodic 3,3,1` by default) so bubbles cut by the periodic box show
  their full shape. Grain boundaries come from the single polycrystal indicator
  `phi_poly = 1 - sum_i eta_gi^2` contoured at 0.42 (translucent blue-grey, gap-free at
  triple junctions), and the triple junctions are the `phi_poly = 0.60` contour drawn as
  tubes colored by `h_b` (empty lines dark, bubble-covered lines bright). This follows the
  visual language of Aagesen et al. 2019 (CMS 161), Figs. 3-5, 9: yellow bubbles on
  semitransparent blue grain boundaries. In 2D the domain is colored by `eta_b` and the
  bubbles / grain boundaries are drawn as tubes.
- **field view** (top-right, single unit cell): a mid-z slice (3D) or the domain (2D)
  colored by `mu_g` (color range taken at the final time step), bubble outlines on top,
  and gas-flux glyphs (`-grad mu_g`, via *Compute Derivatives* + *Glyph*) showing where
  gas flows. Carries the `mu_g` scalar bar in 3D.
- **chart view** (bottom-left): live time series from *Integrate Variables* /
  *Connectivity* (`stats.csv`): relative change (%) of the bubble volume and the
  `mu_g`/`mu_v` inventories, plus the Aagesen coverage metrics **X_GB** (grain-boundary
  coverage, from the `phi_poly = 0.42` surface) and **X_TJ** (triple-junction coverage,
  3D, from the `phi_poly = 0.60` line) in %, and the bubble count **N_b** — i.e. the
  quantities of Aagesen et al. Figs. 6-8.
- **switching-function view** (bottom-right, single unit cell): the bubble switching
  function `h_b` drawn as multiple contours (0.2 / 0.5 / 0.8) together with the
  `eta_b = 0.5` surface — nested shells in 3D, gold/white contour lines over the `mu_v`
  (or `h_b`) field in 2D, picturing the switching function with the order parameters, as
  in Aagesen et al. Fig. 3. In 3D, if the series has no `h_b`, this panel falls back to a
  volume rendering of `eta_b` (`--no-volume` disables it).

Everything sits in one 2x2 layout on a dark gradient background with FXAA and depth
peeling, plus a time annotation that follows the animation.

## Headless (pvpython / pvbatch, ParaView 5.11 on this machine)

```bash
cd tests/Studies/fissiongas/paraview
./render.sh /path/to/run/Saves /path/to/out_viz                 # scene + frames + movie
# or directly:
pvpython fissiongas_viz.py -- --saves /path/to/run/Saves --out /path/to/out_viz \
    --periodic 3,3,3 --stride 2 --width 2560 --height 1440 --fps 12
```

Outputs in the out dir: `scene.pvsm` (state file), `stats.csv`, `preview.png`,
`frames/frame_XXXX.png`, `movie.mp4` (needs `ffmpeg`, present here).

Examples with the current runs (already rendered, outputs next to each `Saves`):

```bash
# 2D production runs (time series + movie.mp4):
./render.sh ~/projects/sloth_trials/fissiongas_results/big_df1e4/Saves \
    ~/projects/sloth_trials/fissiongas_results/big_df1e4/viz
./render.sh ~/projects/sloth_trials/fissiongas_results/sq_df1e4/Saves \
    ~/projects/sloth_trials/fissiongas_results/sq_df1e4/viz --stride 4
# ctest smoke-test outputs in the build tree (test1 = 2D Bubbles, test2 = 2D Polycrystal,
# test4 = 3D Hex3D of Aagesen et al. Fig. 3):
./render.sh ../../../build/tests/Studies/fissiongas/test1/Saves \
    ../../../build/tests/Studies/fissiongas/test1/viz
./render.sh ../../../build/tests/Studies/fissiongas/test2/Saves \
    ../../../build/tests/Studies/fissiongas/test2/viz
./render.sh ../../../build/tests/Studies/fissiongas/test4/Saves \
    ../../../build/tests/Studies/fissiongas/test4/viz
```

Useful options: `--mode state,frames` (skip the movie), `--stride N` (subsample time),
`--periodic 1,1,1` (no replication), `--no-volume`, `--eta eta_b0` (override the
auto-detected bubble field). Full list: `pvpython fissiongas_viz.py -- --help`.

## In the ParaView GUI

`File > Load State...` and pick the generated `scene.pvsm`. The readers are already in the
animation scene, so the play button steps through the time series in all views at once
(chart included). Add a camera orbit on the hero view with *Camera > Orbit* if you want a
turntable animation, then `File > Save Animation`.

## Notes

- Requires `pvpython` from the system ParaView (5.11.2 here). This build has no OSPRay, so
  "fancy" comes from GPU volume rendering, transparency + depth peeling, lighting, layout
  and animation rather than ray tracing.
- The VTU series written by SLOTH (`*.pvd` + `*.pvtu`, binary) loads directly in 5.11.
- Everything is generated; nothing here feeds back into the simulation. Delete the out
  directory freely.
