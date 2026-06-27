# Motion Tracker

A Python utility for measuring the motion of objects in a video. It uses OpenCV's
Shi–Tomasi corner detection and Lucas–Kanade optical flow to track feature points
inside a user-selected region of interest (ROI), converts the pixel displacements
to real-world units, and produces both a CSV of per-frame motion and an annotated
output video with live speed plots.

It was originally written to measure motion in snow/avalanche field footage, but
works for any video where you want to track the displacement and speed of features
within a chosen area.

## Features

- **Interactive ROI selection** — draw a rectangle over the area you want to track.
- **Scale calibration** — draw a rectangle of known real-world size to convert
  pixels to meters, so output is reported in m and m/s.
- **Optical-flow tracking** — Lucas–Kanade tracking of up to 20 feature points.
- **Camera-motion compensation** — tracks background features outside the ROI to
  estimate the camera's own motion (pan/rotate/zoom via partial-affine + RANSAC,
  with a median-translation fallback) and subtracts it, so a handheld or panning
  camera doesn't show up as false object motion. By default it uses everything
  outside the ROI as the reference; you can also **mark explicit static area(s)**
  for scenes where the background contains moving distractions (people, trees,
  water).
- **Annotated video** — tracked points, motion trails, median dx/dt and dy/dt
  text, and an optional overlaid speed-vs-frame plot.
- **Data export** — per-frame displacement of every tracked point saved to CSV.
- **Video tools** — trim to a frame range and play back through the frame buffer.
- **Two interfaces** — a Tkinter GUI (default) or a scripted "auto" sequence.

## Requirements

- Python 3 with **Tkinter** available (used for file dialogs and the GUI)
- Python packages:
  - `numpy`
  - `opencv-python`
  - `pandas`
  - `tqdm`
  - `matplotlib`

Install the dependencies with:

```bash
pip install numpy opencv-python pandas tqdm matplotlib
```

## Files

- **`video_tracker.py`** — the reusable tracking engine (the `VideoUtil` class):
  load, ROI (rectangle or polygon), scale calibration, static zones, optical-flow
  tracking with optional stabilization. No GUI.
- **`motion_tracker.py`** — the Tkinter GUI app and command-line entry point built
  on `VideoUtil` (re-exported here, so `from motion_tracker import VideoUtil` still
  works).
- **`pst_analysis.py`** — Propagation Saw Test (PST) analysis built on the tracker
  (see [PST analysis](#pst-analysis-propagation-saw-test)).
- **`pst_plots.py`** — all PST figures in one module: per-video plots (called by
  `pst_analysis.py`) and the cross-video `compare`/`box` commands
  (see [Comparing videos](#comparing-videos)).
- **`run_pipeline.py`** — runs the whole workflow start to finish: per-video
  `pst_analysis.py --reuse --stabilize` for each video in `metadata.json`, then the
  `pst_plots.py` comparison figures (see [Running the pipeline](#running-the-pipeline)).
- **`archive/`** — the superseded original implementation, kept for reference only.

## Usage

### GUI mode (default)

```bash
python motion_tracker.py
```

This opens the **Video Motion Tracker** window. The typical workflow is:

1. **Load Video** — pick a video file via the dialog.
2. **Set Area to Track (ROI)** — scrub to a frame and draw a rectangle around the
   region to track. Press `-` to zoom out if the frame is larger than the screen,
   then `Esc` to save.
3. **Set Static Area (optional)** — draw one or more rectangles over parts of the
   scene that are genuinely still, to use *only* those for stabilization. Recommended
   when the background has moving distractions (people, wind-blown trees, water).
   Drag a box, `a`/`Enter` to add it, `c` to clear, `-` to zoom, `Esc` to finish.
   Skip this to use the default (everything outside the ROI).
4. **Set Pixel Size (Calibration)** — draw a rectangle over an object of known
   size and enter its height and/or width in meters. (Skip this to keep results in
   pixels.)
5. **Track Motion** — optionally restrict to a frame range, then watch the points
   get tracked. You'll be prompted to save the motion data as CSV.
6. **Save Tracked Video** — write the annotated video to an MP4.

Additional buttons: **Trim Video** (permanently crop the loaded buffer to a frame
range) and **Play Original Video**.

### Auto mode

Runs the load → ROI → (optional static area) → calibrate → track → save sequence
in order. The ROI, static-area, and calibration steps are still interactive (it
asks on the console whether to mark a static area).

```bash
python motion_tracker.py --mode auto --path /path/to/video.mp4
```

| Flag | Description | Default |
| --- | --- | --- |
| `-m`, `--mode` | `manual` (GUI) or `auto` (scripted sequence) | `manual` |
| `-v`, `--path` | Path to the video file (required for `auto`) | _(empty)_ |

## Interactive controls

**ROI / calibration windows**

| Key | Action |
| --- | --- |
| Mouse drag | Draw the rectangle |
| `-` | Zoom out (halve the displayed frame) |
| `Esc` | Save the selection and close |

**Trim window**

| Key | Action |
| --- | --- |
| `s` | Mark current frame as start |
| `e` | Mark current frame as end |
| `f` | Step forward one frame |
| `b` | Step back one frame |
| `Esc` | Save and close |

## Output

**Sign convention:** `+x` is rightward and **`+y` is upward** (the raw image
Y axis is flipped so output follows the usual physics/math convention rather than
screen coordinates). This applies to every y column and the `dy/dt` readout.

- **CSV** — one row per frame. Columns `p0_x, p0_y, p1_x, p1_y, …` give each
  tracked point's raw displacement (in meters if calibrated, otherwise pixels).
  When stabilization is on, matching `p0_x_corr, p0_y_corr, …` columns hold the
  camera-corrected displacement, plus `bg_points` (background features used that
  frame — a confidence signal) and `cam_dx, cam_dy` (the camera motion that was
  removed).
- **MP4** — the source video with tracked points, motion trails, median speed
  readouts, a `bg pts` confidence overlay (when stabilizing), and an optional
  speed-vs-frame plot overlaid.

### Camera-motion compensation

Stabilization is **on by default** in `track()`. It tracks features in the
background (everything outside a dilated ROI), fits a partial-affine transform
(translation + rotation + uniform scale, via RANSAC) to that background motion
each frame, and subtracts it from the ROI motion — so the reported speeds reflect
the object's motion *relative to the scene* rather than apparent motion from a
moving camera. When fewer than three background points survive, it falls back to a
median-translation estimate; with none, that frame is left uncorrected.

**Choosing the reference area.** By default "background" means everything outside
the (dilated) ROI. That assumes the rest of the scene is static — which breaks if
it contains moving people, wind-blown trees, water, etc. Use **Set Static Area** to
draw one or more rectangles over parts you know are still; the stabilizer then draws
its reference features *only* from those (still excluding the tracked object). The
tracked object always feeds the ROI motion, never the camera estimate.

Disable stabilization entirely (e.g. for a tripod-mounted camera, or to compare)
with `track(stabilize=False)`. Watch the `bg pts` overlay / `bg_points` column: a
low count means the camera estimate is unreliable — which happens when the moving
object fills most of the frame, or when a too-small static area was selected.

## PST analysis (Propagation Saw Test)

`pst_analysis.py` drives the tracker on a side-view PST video and derives
fracture-propagation metrics from the slab-marker trajectories:

- **Collapse magnitude** — vertical drop of the slab onto the collapsed weak layer,
  reported as the full along-column profile (amplitude per marker, in mm).
- **Crack propagation speed** — from each marker's collapse-onset time vs its
  along-column position (linear fit; the van Herwijnen-style PTV approach).
- **Propagation distance** — how far the dynamic crack travelled (measured from the
  critical cut length), and whether it **arrested** before the far end of the column.
- **Touchdown distance** — the length over which the slab bends from the crack
  front down to re-contact the collapsed layer behind it (the bending zone),
  estimated as `crack_speed × marker rise-time`.

**Saw-cut length.** Propagation distance is measured from the critical cut length,
and the sawn markers are excluded from the speed fit. The cut length is an
experimental value the operator measures — it can't be reliably inferred from the
video (the sawn region settles before filming, or the whole slab moves at once), so
pass it with **`--cut-length-cm`**. Without it the critical cut defaults to 0 (with a
note), and propagation distance is measured from the column origin.

**Bad-track rejection.** Optical-flow points that lose lock jump to a wild position,
producing a huge spurious "collapse" (e.g. 70 mm where the slab settled ~2 mm). Such
markers — collapse far above their *local* neighbours — are flagged (`n_bad_tracks`,
magenta in the overlay) and excluded from the collapse stats, the speed fit, and the
front, so a single bad track can't distort the results.

**Arrest needs full coverage.** Arrest (`arrested: true`) is only reported when the
ROI spans the full column *and* intact markers remain ahead of the collapse front.
With incomplete coverage it reports `arrested: null` (you can't tell a real arrest
from the markers simply running out); reaching the column end reports `false`
(propagated).

```bash
# red-painted slab, high-speed footage (interactive ROI + scale calibration)
python pst_analysis.py --path data/PST_01.mp4 --column-length-cm 200 \
    --fps 240 --seed color --cut-from left

# non-interactive scale and fixed ROI, handheld (camera stabilization):
python pst_analysis.py --path data/PST_01.mp4 --column-length-cm 200 --fps 240 \
    --mm-per-px 0.8 --roi 200,150,1400,400 --stabilize --out pst_results

# re-analyze the same clip without re-drawing the ROI / re-calibrating:
python pst_analysis.py --path data/PST_01.mp4 --column-length-cm 200 --fps 240 \
    --seed color --reuse --cut-length-cm 30
```

**Reusing setup.** The interactive ROI, scale calibration, and static zone are
auto-saved to `<out>/<video>/<video>_config.json` (rectangles + pixel sizes,
human-editable). Re-run with **`--reuse`** to load them and skip those steps — handy
for re-analyzing with different parameters (`--cut-length-cm`, `--seed`, …) without
redrawing anything. Use `--config PATH` to load a specific config file.

> **Frame rate matters.** PST crack propagation is fast (~10–40 m/s), so high-speed
> footage (≥120–240 fps) is needed to resolve the front. Many cameras write the
> *playback* fps (e.g. 30) into the file even for 240 fps capture — pass `--fps` with
> the true capture rate or every time (and the crack speed) is wrong by the ratio.
> The tool warns when the fps looks too low.

For handheld footage, `--stabilize` removes camera motion. By default it uses
everything outside the slab ROI as the static reference; for scenes with moving
background (people, trees), restrict it to a known-still region with `--draw-static`
(draw the zone interactively) or `--static-roi x,y,w,h`.

It seeds points in the slab ROI (no physical markers required) and tracks them as
stable trajectories via `VideoUtil.track_markers()`. Two seeding modes:

- `--seed features` (default) — Shi-Tomasi corners; tune with `--n-markers`,
  `--quality`, `--min-distance`. Good when the slab is textured.
- `--seed grid` — a regular grid across the ROI (`--grid-spacing-px`). Use this for
  **low-texture snow**, where corner detection finds only a handful of points
  (you'll see a "low-texture" note if `features` mode returns few markers).
- `--seed color` — centroids of **red spray-paint dots** within the ROI. Use this
  when the slab is marked with red paint fiducials; the markers land on the dots
  instead of on incidental corners.

Outputs go to a per-video subdirectory, `<out>/<video name>/`, with the video name as
the file prefix (so runs on different clips don't collide). For example
`--out pst_results` on `data/PST_01.mp4` writes `pst_results/PST_01/PST_01_summary.json`,
etc. The files:

- `*_summary.json` — the metrics above.
- `*_markers.csv` — per-marker table (position, collapse, onset, etc.).
- `*_trajectories.csv` — **per-frame, per-marker movement**: each point's distance
  from the cut end plus its along-column and collapse (normal) displacement in mm,
  for every frame in the propagation window (`--full-trajectories` for the whole
  record).
- `*_event.mp4` — **annotated event video**: the markers drawn on the real frames
  through the propagation window, **colored by along-column distance** and with
  motion trails, so you can eyeball the propagation frame-by-frame.
- Plots: `*_markers_overlay.png` (seeded points on the first frame, color-coded
  propagation / not-collapsed / saw-cut / out-of-column — verify coverage),
  `*_collapse_curves.png`, `*_kymograph.png`, `*_crack_speed.png`,
  `*_collapse_profile.png`.

See **[docs/pst_output_description.md](docs/pst_output_description.md)** for how to read each output image
(with examples) and a checklist for a trustworthy result.

**Geometry** (side-view camera): the column's long axis is fitted from the initial
marker positions (PCA), so a tilted-in-frame column is handled — along-column
position is the projection onto that axis and collapse is the component normal to
it (the reported `column_tilt_deg` shows the fitted tilt). The saw cut starts at the
`--cut-from` end (sets the along-column origin) and calibration sets the scale.

`--column-length-cm` is compared against the tracked extent to judge arrest: if the
ROI/markers don't reach close to the column end, **propagation distance and arrest
can't be trusted** — the tool reports `arrested: null` (indeterminate) and prints a
coverage warning, since a crack reaching the edge of the ROI is indistinguishable
from one that kept going. Extend the ROI to span the full column for reliable arrest
detection. Also inspect the crack-speed fit R²: a low value means no coherent
propagation front (or the ROI/scale/cut direction need adjusting).

**Scale / stray-feature guard.** Features tracked outside the column (saw, hand, pit
wall) or a wrong scale show up as a tracked extent longer than `--column-length-cm`.
A few such markers are trimmed (reported as `n_outliers_trimmed`); if many fall
outside, it's treated as a likely scale error — the data is kept but `scale_warning`
is set and a warning printed, because distances and speed can't be trusted until the
calibration is fixed.

**Propagation window.** The fast collapse event is isolated automatically from any
slow pre/post drift: the tool finds the peak of the collective collapse *velocity*
and the window around it (reported as `propagation_window_s`/`_frames`), then measures
onset and crack speed **within that window only**. Two diagnostic plots reveal the
dynamics directly: `*_collapse_curves.png` (collapse vs time, colored by along-column
position, window shaded) and `*_kymograph.png` (collapse over position × time — a
**tilted** front means a resolvable propagation speed; a **vertical** edge means the
collapse is near-simultaneous at the frame rate, so the speed can't be measured). The
tool warns when the speed fit is poor (R² low / near-simultaneous onsets) — that
usually means the front crossed the tracked span faster than the fps resolves, so you
need a higher frame rate or a longer tracked span (not a code problem).

### Running the pipeline

`run_pipeline.py` runs everything end to end from the existing per-video configs
and `metadata.json`: it analyzes each video (`pst_analysis.py --reuse --stabilize`),
then builds the cross-video comparison figures.

```bash
python run_pipeline.py                    # analyze all + build comparisons
python run_pipeline.py --only Beehive_3   # just one (or a few) videos
python run_pipeline.py --skip-analysis    # only rebuild the comparison plots
```

Stabilization is on by default; to analyze a specific clip without it (e.g. one the
stabilizer inverts), add `"stabilize": false` to that video's entry in
`metadata.json` — the per-video value overrides the global `--no-stabilize` default.
Videos without a saved config are skipped (run the interactive setup once first).

### Comparing videos

`pst_plots.py` reads the per-video outputs and compares videos. Every figure —
per-video and cross-video — draws from one collapse-data definition
(`collapse_sample`) aligned to each video's **measured critical cut**, so results
stay consistent. Two cross-video subcommands:

```bash
python pst_plots.py compare --base pst_results    # collapse-profile comparison + stats CSV
python pst_plots.py box --base pst_results \
    --collapse-time 2026-03-17T09:40 --reference Beehive_0   # amplitude box plot
```

`compare` writes `collapse_comparison.png` — two panels: collapse magnitude vs
distance from the cut (absolute profiles, one curve per video), and the same
normalized to the near-cut collapse (so the *relative* change as the crack runs is
comparable: rising = collapse grows with propagation, falling = damping) — plus
`collapse_comparison.csv` with per-video stats (critical cut, near/far collapse,
collapse slope mm/m, far/near ratio). Add `--color-by-time` to color each curve by
its time relative to `--collapse-time`.

`box` writes `collapse_comparison_box.png` — one box per video of its collapse
amplitude (with the raw markers overlaid as jittered scatter, Tukey outliers
dropped), placed on a time axis relative to the slope collapse; a far-back
reference video gets its own panel via a broken x-axis. Both commands default
their output to `<base>/_comparison/`. Bad tracks and out-of-column markers are
excluded throughout.

## Notes

- The entire video is loaded into memory for fast, non-linear frame access, so
  very long or high-resolution clips can be memory-intensive.
- Calibration assumes a linear pixel-to-meter scale; if you enter only one
  dimension, square pixels are assumed for the other.

## Author

Ron Simenhois
