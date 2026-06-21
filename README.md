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

- **`motion_tracker.py`** — the application (the `VideoUtil` tracker and Tkinter GUI).
- **`pst_analysis.py`** — Propagation Saw Test (PST) analysis built on the tracker
  (see [PST analysis](#pst-analysis-propagation-saw-test)).
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
- **Propagation distance** — how far the crack travelled from the cut end, and
  whether it **arrested** before reaching the far end of the column.
- **Touchdown distance** — the length over which the slab bends from the crack
  front down to re-contact the collapsed layer behind it (the bending zone),
  estimated as `crack_speed × marker rise-time`.

```bash
# interactive ROI + scale calibration (OpenCV windows + console prompts)
python pst_analysis.py --path data/PST_01.mp4 --column-length-cm 100 --cut-from left

# non-interactive scale and fixed ROI, with camera stabilization:
python pst_analysis.py --path data/PST_01.mp4 --column-length-cm 100 \
    --mm-per-px 0.8 --roi 200,150,1400,400 --stabilize --out pst_results
```

It detects feature points in the slab ROI (no physical markers required — tune
coverage with `--n-markers`, `--quality`, `--min-distance`), tracks them as stable
trajectories via `VideoUtil.track_markers()`, and writes a summary (`*_summary.json`),
a per-marker table (`*_markers.csv`), and diagnostic plots (collapse-vs-time,
onset-vs-position with the speed fit, and the collapse profile) to `--out`.

**Geometry assumptions** (side-view camera): the column is horizontal, the slab
collapses downward, image x runs along the column, and the saw cut starts at the
`--cut-from` end (which sets the along-column origin). Calibration sets the spatial
scale; `--column-length-cm` is needed to judge arrest. Inspect the crack-speed fit
R² in the summary — a low value means there was no coherent propagation front (or
the ROI/scale/cut direction need adjusting).

## Notes

- The entire video is loaded into memory for fast, non-linear frame access, so
  very long or high-resolution clips can be memory-intensive.
- Calibration assumes a linear pixel-to-meter scale; if you enter only one
  dimension, square pixels are assumed for the other.

## Author

Ron Simenhois
