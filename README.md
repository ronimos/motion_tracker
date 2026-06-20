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
  camera doesn't show up as false object motion.
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
3. **Set Pixel Size (Calibration)** — draw a rectangle over an object of known
   size and enter its height and/or width in meters. (Skip this to keep results in
   pixels.)
4. **Track Motion** — optionally restrict to a frame range, then watch the points
   get tracked. You'll be prompted to save the motion data as CSV.
5. **Save Tracked Video** — write the annotated video to an MP4.

Additional buttons: **Trim Video** (permanently crop the loaded buffer to a frame
range) and **Play Original Video**.

### Auto mode

Runs the load → ROI → calibrate → track → save sequence in order. The ROI and
calibration steps are still interactive.

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

Disable it (e.g. for a tripod-mounted camera, or to compare) with
`track(stabilize=False)`. Watch the `bg pts` overlay / `bg_points` column: a low
count means the camera estimate is unreliable, which happens when the moving
object fills most of the frame.

## Notes

- The entire video is loaded into memory for fast, non-linear frame access, so
  very long or high-resolution clips can be memory-intensive.
- Calibration assumes a linear pixel-to-meter scale; if you enter only one
  dimension, square pixels are assumed for the other.

## Author

Ron Simenhois
