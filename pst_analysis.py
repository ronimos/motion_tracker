# -*- coding: utf-8 -*-
"""
PST (Propagation Saw Test) analysis
===================================

Drives the motion tracker on a side-view PST video and derives fracture-
propagation metrics from the slab-marker trajectories:

1.  Collapse magnitude  - the vertical drop (collapse) of the slab onto the
    collapsed weak layer, reported as the full along-column profile (per marker).
2.  Crack propagation speed - from the collapse-onset time of each marker versus
    its along-column position (linear fit; the van Herwijnen-style PTV approach).
3.  Propagation distance - how far the crack travelled from the cut end, and
    whether it arrested before reaching the far end of the column.
4.  Touchdown distance - the length over which the slab bends from the crack
    front down to re-contact the collapsed layer behind it (the bending zone /
    "slab propagation wave length"), estimated as crack_speed x marker rise-time.

Coordinate / geometry assumptions (side-view camera):
- The column lies horizontally; the slab collapses downward.
- Image x runs along the column; image y increases downward (so collapse is +y).
- The saw cut starts at one end (`--cut-from`); along-column position is measured
  from that end.

Usage:
    # interactive ROI + calibration (draws OpenCV windows, prompts on console)
    python pst_analysis.py --path data/PST_01.mp4 --column-length-cm 100 --cut-from left

    # non-interactive scale, fixed ROI, camera stabilization on:
    python pst_analysis.py --path data/PST_01.mp4 --column-length-cm 100 \
        --mm-per-px 0.8 --roi 200,150,1400,400 --stabilize --out pst_results

@author: Ron Simenhois
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe: figures are written to disk, never shown
import matplotlib.pyplot as plt

from motion_tracker import VideoUtil


class PSTAnalyzer:
    """
    Computes PST propagation metrics from stable marker trajectories.

    Args:
        track (dict): output of ``VideoUtil.track_markers`` (positions, times,
                      pixel scales, fps, frame0).
        column_length_m (float): full isolated-column length, used to judge arrest.
        cut_from (str): 'left' or 'right' - which image end the saw cut starts from
                        (sets the along-column origin).
        onset_frac (float): fraction of a marker's own collapse used as the
                            collapse-onset threshold (default 0.10).
        touchdown_frac (float): fraction used as the touchdown (fully-collapsed)
                                threshold (default 0.90).
        min_collapse_mm (float): markers whose collapse is below this are treated
                                 as "not collapsed" (noise / ahead of an arrest).
    """

    def __init__(self, track, column_length_m, cut_from='left',
                 onset_frac=0.10, touchdown_frac=0.90, min_collapse_mm=1.0):
        self.column_length = float(column_length_m)
        self.cut_from = cut_from
        self.onset_frac = float(onset_frac)
        self.touchdown_frac = float(touchdown_frac)
        self.min_collapse_mm = float(min_collapse_mm)

        self.t = np.asarray(track['times'], dtype=float)        # (F,)
        self.fps = float(track['fps'])
        self.sx = float(track['pix_width'])                     # m/px, horizontal
        self.sy = float(track['pix_height'])                    # m/px, vertical
        self.frame0 = track.get('frame0')
        pos = np.asarray(track['positions'], dtype=float)       # (F, N, 2) px
        self.pos = pos

        self._build_per_marker()
        self._fit_crack_speed()
        self._propagation_distance()
        self._touchdown_distance()

    # ------------------------------------------------------------------ #
    # Per-marker quantities
    # ------------------------------------------------------------------ #
    def _build_per_marker(self):
        pos = self.pos
        F, N, _ = pos.shape

        x0 = pos[0, :, 0]                       # initial pixel x of each marker
        y = pos[:, :, 1]                        # (F, N) pixel y over time

        # Along-column position (m), origin at the cut end.
        if self.cut_from == 'right':
            X = (np.nanmax(x0) - x0) * self.sx
        else:  # 'left'
            X = (x0 - np.nanmin(x0)) * self.sx

        # Vertical displacement, downward-positive, in mm (collapse).
        w = (y - y[0]) * self.sy * 1000.0       # (F, N) mm

        # Collapse amplitude per marker = plateau over the final 10% of frames.
        tail = max(1, int(round(0.10 * F)))
        amp = np.nanmedian(w[-tail:], axis=0)   # (N,) mm
        amp = np.where(np.isfinite(amp), amp, 0.0)

        onset = np.full(N, np.nan)
        touchdown = np.full(N, np.nan)
        collapsed = amp >= self.min_collapse_mm

        for i in range(N):
            if not collapsed[i]:
                continue
            wi = w[:, i]
            onset[i] = self._cross_time(wi, self.onset_frac * amp[i])
            touchdown[i] = self._cross_time(wi, self.touchdown_frac * amp[i])

        self.X = X
        self.w = w
        self.amp = amp
        self.collapsed = collapsed
        self.onset = onset
        self.touchdown = touchdown
        self.rise = touchdown - onset

    def _cross_time(self, signal, level):
        """First time (s, linearly interpolated) `signal` reaches `level`."""
        s = np.asarray(signal, dtype=float)
        above = s >= level
        if not np.any(above):
            return np.nan
        k = int(np.argmax(above))               # first index at/above level
        if k == 0:
            return self.t[0]
        s0, s1 = s[k - 1], s[k]
        if not np.isfinite(s0) or s1 == s0:
            return self.t[k]
        frac = (level - s0) / (s1 - s0)
        return self.t[k - 1] + frac * (self.t[k] - self.t[k - 1])

    # ------------------------------------------------------------------ #
    # Crack propagation speed: onset time vs along-column position
    # ------------------------------------------------------------------ #
    def _fit_crack_speed(self):
        mask = self.collapsed & np.isfinite(self.onset)
        X = self.X[mask]
        t = self.onset[mask]
        self.speed = np.nan
        self.speed_r2 = np.nan
        self.speed_n = int(mask.sum())
        if self.speed_n < 3 or np.ptp(X) == 0:
            return
        # t_onset = X / c + b  ->  slope = 1/c
        slope, intercept = np.polyfit(X, t, 1)
        if slope == 0:
            return
        self.speed = 1.0 / slope                # m/s
        pred = slope * X + intercept
        ss_res = np.sum((t - pred) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        self.speed_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        self._speed_fit = (slope, intercept)

    # ------------------------------------------------------------------ #
    # Propagation distance and arrest
    # ------------------------------------------------------------------ #
    def _propagation_distance(self):
        if not np.any(self.collapsed):
            self.prop_distance = 0.0
            self.arrested = True
            self.arrest_X = 0.0
            return
        Xc = self.X[self.collapsed]
        front = float(np.nanmax(Xc))            # furthest collapsed marker (m)
        self.prop_distance = front
        # Arrested if there are markers beyond the front that did NOT collapse.
        beyond = self.X[~self.collapsed] > front
        reached_end = front >= (self.column_length - max(0.05 * self.column_length,
                                                         2.0 * self._marker_spacing()))
        self.arrested = bool(np.any(beyond)) and not reached_end
        self.arrest_X = front if self.arrested else np.nan

    def _marker_spacing(self):
        xs = np.sort(self.X[np.isfinite(self.X)])
        if xs.size < 2:
            return 0.0
        return float(np.median(np.diff(xs)))

    # ------------------------------------------------------------------ #
    # Touchdown distance (bending-zone length) = speed x rise-time
    # ------------------------------------------------------------------ #
    def _touchdown_distance(self):
        self.touchdown_distance = np.nan
        self.touchdown_distance_std = np.nan
        if not np.isfinite(self.speed):
            return
        rise = self.rise[self.collapsed & np.isfinite(self.rise)]
        rise = rise[rise > 0]
        if rise.size == 0:
            return
        lengths = abs(self.speed) * rise        # m
        self.touchdown_distance = float(np.mean(lengths))
        self.touchdown_distance_std = float(np.std(lengths))

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def summary(self):
        amp_c = self.amp[self.collapsed]
        return {
            "crack_speed_m_s": _round(self.speed, 2),
            "crack_speed_fit_r2": _round(self.speed_r2, 3),
            "crack_speed_n_markers": self.speed_n,
            "propagation_distance_m": _round(self.prop_distance, 3),
            "column_length_m": _round(self.column_length, 3),
            "arrested": bool(self.arrested),
            "arrest_position_m": _round(self.arrest_X, 3),
            "touchdown_distance_m": _round(self.touchdown_distance, 3),
            "touchdown_distance_std_m": _round(self.touchdown_distance_std, 3),
            "collapse_mean_mm": _round(float(np.mean(amp_c)) if amp_c.size else np.nan, 2),
            "collapse_max_mm": _round(float(np.max(amp_c)) if amp_c.size else np.nan, 2),
            "n_markers_total": int(self.amp.size),
            "n_markers_collapsed": int(self.collapsed.sum()),
        }

    def per_marker_table(self):
        return pd.DataFrame({
            "marker": np.arange(self.amp.size),
            "along_column_m": np.round(self.X, 4),
            "collapse_mm": np.round(self.amp, 3),
            "onset_s": np.round(self.onset, 4),
            "touchdown_s": np.round(self.touchdown, 4),
            "rise_s": np.round(self.rise, 4),
            "collapsed": self.collapsed,
        }).sort_values("along_column_m").reset_index(drop=True)

    def report(self, outdir="pst_results", prefix="pst"):
        os.makedirs(outdir, exist_ok=True)
        summary = self.summary()

        # --- console ---
        print("\n=== PST analysis summary ===")
        for k, v in summary.items():
            print(f"  {k:28s}: {v}")
        if self.arrested:
            print(f"  -> crack ARRESTED at {summary['propagation_distance_m']} m "
                  f"of {summary['column_length_m']} m")
        else:
            print(f"  -> crack propagated to the end of the column")

        # --- files ---
        with open(os.path.join(outdir, f"{prefix}_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        table = self.per_marker_table()
        table.to_csv(os.path.join(outdir, f"{prefix}_markers.csv"), index=False)
        self._plots(outdir, prefix)
        print(f"\nWrote summary, per-marker CSV, and plots to: {outdir}/")
        return summary

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    def _plots(self, outdir, prefix):
        # 1) Collapse-vs-time curves for collapsed markers
        fig, ax = plt.subplots(figsize=(8, 5))
        idx = np.where(self.collapsed)[0]
        order = idx[np.argsort(self.X[idx])]
        for i in order:
            ax.plot(self.t, self.w[:, i], lw=1, alpha=0.8)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("collapse (mm, downward +)")
        ax.set_title("Slab marker collapse vs time")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{prefix}_collapse_curves.png"), dpi=130)
        plt.close(fig)

        # 2) Onset time vs along-column position (crack speed)
        fig, ax = plt.subplots(figsize=(8, 5))
        m = self.collapsed & np.isfinite(self.onset)
        ax.scatter(self.X[m], self.onset[m], s=25, c="tab:red", label="markers")
        if np.isfinite(self.speed) and hasattr(self, "_speed_fit"):
            slope, intercept = self._speed_fit
            xs = np.linspace(self.X[m].min(), self.X[m].max(), 50)
            ax.plot(xs, slope * xs + intercept, "k--",
                    label=f"fit: c = {self.speed:.1f} m/s  (R²={self.speed_r2:.2f})")
            ax.legend()
        ax.set_xlabel("along-column position (m)")
        ax.set_ylabel("collapse-onset time (s)")
        ax.set_title("Crack propagation: onset time vs position")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{prefix}_crack_speed.png"), dpi=130)
        plt.close(fig)

        # 3) Collapse amplitude profile vs position (+ propagation distance / column end)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(self.X[self.collapsed], self.amp[self.collapsed], s=25,
                   c="tab:blue", label="collapsed")
        if np.any(~self.collapsed):
            ax.scatter(self.X[~self.collapsed], self.amp[~self.collapsed], s=18,
                       c="lightgray", label="not collapsed")
        ax.axvline(self.prop_distance, color="tab:green", ls="--",
                   label=f"propagation = {self.prop_distance:.2f} m")
        ax.axvline(self.column_length, color="k", ls=":",
                   label=f"column end = {self.column_length:.2f} m")
        ax.set_xlabel("along-column position (m)")
        ax.set_ylabel("collapse amplitude (mm)")
        ax.set_title("Collapse profile along the column")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{prefix}_collapse_profile.png"), dpi=130)
        plt.close(fig)


def _round(v, n):
    """Round, turning non-finite values into None for clean JSON/printing."""
    try:
        if v is None or not np.isfinite(v):
            return None
    except TypeError:
        return v
    return round(float(v), n)


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def _set_roi_from_arg(v, roi):
    x, y, w, h = (int(s) for s in roi.split(","))
    mask = np.zeros_like(v.video_buffer[0])
    mask[y:y + h, x:x + w] = 255
    v.mask = mask


def run(args):
    v = VideoUtil(args.path, interactive=False)

    # 1) Region of interest = the slab (where markers/features live)
    if args.roi:
        _set_roi_from_arg(v, args.roi)
    else:
        print("Draw a box around the slab markers (the part that collapses), then Esc.")
        v.set_roi()

    # 2) Spatial scale
    if args.mm_per_px:
        v.pix_width = v.pix_height = args.mm_per_px / 1000.0
        print(f"Scale set: {args.mm_per_px} mm/px")
    else:
        print("Calibrate scale: draw a rectangle of known size, then enter its size.")
        v.set_pxl_size()

    if args.fps:
        v.fps = args.fps

    # 3) Trajectories
    track = v.track_markers(start=args.start, end=args.end, n_markers=args.n_markers,
                            quality_level=args.quality, min_distance=args.min_distance,
                            stabilize=args.stabilize)

    # 4) Analysis
    analyzer = PSTAnalyzer(track,
                           column_length_m=args.column_length_cm / 100.0,
                           cut_from=args.cut_from,
                           onset_frac=args.onset_frac,
                           min_collapse_mm=args.min_collapse_mm)
    analyzer.report(outdir=args.out, prefix=args.prefix)


def build_parser():
    p = argparse.ArgumentParser(description="PST fracture-propagation analysis from video.")
    p.add_argument("-v", "--path", required=True, help="Path to the PST video file.")
    p.add_argument("--column-length-cm", type=float, required=True,
                   help="Isolated column length in cm (used to judge arrest).")
    p.add_argument("--cut-from", choices=["left", "right"], default="left",
                   help="Which image end the saw cut starts from (along-column origin).")
    p.add_argument("--mm-per-px", type=float, default=None,
                   help="Spatial scale in mm/pixel. If omitted, calibrate interactively.")
    p.add_argument("--roi", default=None,
                   help="Slab ROI as 'x,y,w,h' in pixels. If omitted, draw it interactively.")
    p.add_argument("--fps", type=float, default=None, help="Override video fps.")
    p.add_argument("--start", type=int, default=0, help="First frame to track.")
    p.add_argument("--end", type=int, default=None, help="Last frame to track (exclusive).")
    p.add_argument("--n-markers", type=int, default=100, help="Max features to track in the ROI.")
    p.add_argument("--quality", type=float, default=0.05,
                   help="Shi-Tomasi quality threshold for marker detection (lower = more markers).")
    p.add_argument("--min-distance", type=int, default=8,
                   help="Minimum pixel spacing between detected markers.")
    p.add_argument("--stabilize", action="store_true",
                   help="Subtract camera motion using background features.")
    p.add_argument("--onset-frac", type=float, default=0.10,
                   help="Fraction of a marker's collapse used as the onset threshold.")
    p.add_argument("--min-collapse-mm", type=float, default=1.0,
                   help="Collapse below this (mm) counts as 'not collapsed'.")
    p.add_argument("--out", default="pst_results", help="Output directory.")
    p.add_argument("--prefix", default="pst", help="Output file prefix.")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
