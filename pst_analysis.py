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
        exclude_saw (bool): if True, detect crack initiation and exclude the
                            saw-cut phase from the speed/touchdown/propagation
                            analysis.
        cut_length_m (float | None): manual critical-cut length; markers within
                            this distance of the cut end are treated as saw-cut.
                            Overrides automatic detection when given.
    """

    def __init__(self, track, column_length_m, cut_from='left',
                 onset_frac=0.10, touchdown_frac=0.90, min_collapse_mm=1.0,
                 exclude_saw=True, cut_length_m=None):
        self.column_length = float(column_length_m)
        self.cut_from = cut_from
        self.onset_frac = float(onset_frac)
        self.touchdown_frac = float(touchdown_frac)
        self.min_collapse_mm = float(min_collapse_mm)
        self.exclude_saw = bool(exclude_saw)
        self.cut_length_m = cut_length_m

        self.t = np.asarray(track['times'], dtype=float)        # (F,)
        self.fps = float(track['fps'])
        self.sx = float(track['pix_width'])                     # m/px, horizontal
        self.sy = float(track['pix_height'])                    # m/px, vertical
        self.frame0 = track.get('frame0')
        pos = np.asarray(track['positions'], dtype=float)       # (F, N, 2) px
        self.pos = pos

        self._build_per_marker()
        self._flag_outliers()           # scale / out-of-column features
        self._detect_crack_initiation()  # split saw-cut phase from propagation
        self._fit_crack_speed()
        self._propagation_distance()
        self._touchdown_distance()

    # ------------------------------------------------------------------ #
    # Per-marker quantities
    # ------------------------------------------------------------------ #
    def _build_per_marker(self):
        pos = self.pos
        F, N, _ = pos.shape

        # Work in metric space (meters) so the fitted axis isn't distorted by
        # different horizontal/vertical pixel scales.
        Pm = pos.copy()
        Pm[:, :, 0] *= self.sx
        Pm[:, :, 1] *= self.sy
        P0 = Pm[0]                              # (N, 2) initial marker positions, m

        # --- Column axis from the initial marker spread (PCA), handles tilt ---
        e, n, tilt = self._column_axis(P0)
        self.axis, self.normal, self.tilt_deg = e, n, tilt

        # Along-column position (m): project onto the axis, origin at the cut end.
        proj = (P0 - P0.mean(axis=0)) @ e       # centered projection (m)
        if self.cut_from == 'right':
            X = proj.max() - proj
        else:  # 'left'
            X = proj - proj.min()

        # Collapse (mm): displacement component normal to the axis, downward-positive.
        disp = Pm - P0[None, :, :]              # (F, N, 2) m
        w = (disp @ n) * 1000.0                 # (F, N) mm

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

    def _flag_outliers(self):
        """
        Flag markers that fall outside the physical column (stray features on the
        saw, hand, pit wall, ...). A few are trimmed; many means the scale is
        probably wrong, so we keep the data but raise a scale warning - distances
        can't be trusted in that case.
        """
        spacing = self._marker_spacing()
        margin = max(0.05 * self.column_length, 2.0 * spacing)
        self.raw_extent = float(np.nanmax(self.X)) if np.isfinite(self.X).any() else 0.0

        out = self.X > (self.column_length + margin)
        frac_out = out.mean() if out.size else 0.0
        self.n_outliers_trimmed = 0

        if frac_out > 0.20:
            # Too many to be stray features -> likely a scale error. Keep all
            # markers (trimming would discard good data and still be mis-scaled)
            # and warn that distances/speed can't be trusted.
            self.scale_warning = True
            self.in_column = np.ones_like(self.X, dtype=bool)
        else:
            # A few features outside the column (saw, hand, pit wall): trim them.
            self.scale_warning = False
            self.in_column = ~out
            self.n_outliers_trimmed = int(out.sum())

    def _detect_crack_initiation(self):
        """
        Separate the saw-cut phase from the dynamic propagation phase.

        During sawing, markers collapse slowly as the saw advances; at crack
        initiation the remaining column collapses in a fast burst. On the
        cumulative-onset-vs-time curve that is a slow rise followed by a steep
        ramp - the knee is crack initiation. Markers collapsing before it are
        saw-cut and excluded from speed / touchdown / front analysis.

        A manual `cut_length_m` overrides this; `exclude_saw=False` disables it.
        """
        N = self.amp.size
        self.saw_excluded = np.zeros(N, dtype=bool)
        self.t_init = None
        self.critical_cut_length = 0.0

        valid = self.collapsed & self.in_column & np.isfinite(self.onset)

        # Manual override: everything within cut_length of the cut end is saw-cut.
        if self.cut_length_m is not None:
            self.saw_excluded = valid & (self.X <= float(self.cut_length_m))
            self.critical_cut_length = float(self.cut_length_m)
            return

        if not self.exclude_saw or valid.sum() < 6:
            return  # nothing to do / too few markers to split reliably

        idx = np.where(valid)[0]
        order = idx[np.argsort(self.onset[idx])]
        ts = self.onset[order]                      # sorted onset times

        # Kneedle on the cumulative-onset curve (normalised). For a slow-then-fast
        # curve the points sit below the endpoint chord; the max gap is the knee.
        x = (ts - ts[0]) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else np.zeros_like(ts)
        y = np.linspace(0.0, 1.0, ts.size)
        gap = x - y
        k = int(np.argmax(gap))

        # Only split if the knee is pronounced (clear slow phase before a burst).
        if gap[k] < 0.15 or k == 0:
            return
        self.t_init = float(ts[k])
        self.saw_excluded = valid & (self.onset <= self.t_init)
        if np.any(self.saw_excluded):
            self.critical_cut_length = float(np.nanmax(self.X[self.saw_excluded]))

    def _column_axis(self, P0):
        """
        Fit the column's long axis from the initial marker positions (PCA).

        Returns (e, n, tilt_deg): the along-column unit vector `e` (oriented
        toward +x so 'left'/'right' are consistent), the normal unit vector `n`
        oriented downward in the image (collapse-positive), and the axis tilt
        from horizontal in degrees. Falls back to the horizontal axis if the
        markers don't form a clear line.
        """
        if P0.shape[0] >= 2:
            C = P0 - P0.mean(axis=0)
            _, s, vt = np.linalg.svd(C, full_matrices=False)
            if s[0] > 0:
                e = vt[0]
            else:
                e = np.array([1.0, 0.0])
        else:
            e = np.array([1.0, 0.0])
        if e[0] < 0:                            # orient toward +x (image right)
            e = -e
        n = np.array([-e[1], e[0]])             # perpendicular
        if n[1] < 0:                            # orient downward (+y image) = collapse
            n = -n
        tilt = float(np.degrees(np.arctan2(e[1], e[0])))
        return e, n, tilt

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
    def _propagation_mask(self):
        """Markers used for crack-speed / touchdown: collapsed, inside the
        column, with a valid onset, and not in the excluded saw-cut phase."""
        return (self.collapsed & self.in_column & np.isfinite(self.onset)
                & ~self.saw_excluded)

    def _fit_crack_speed(self):
        mask = self._propagation_mask()
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
        # Restrict to valid in-column markers (drop out-of-column outliers).
        incol = self.in_column
        Xv = self.X[incol]
        self.marker_span = float(np.nanmax(Xv)) if np.isfinite(Xv).any() else 0.0
        spacing = self._marker_spacing()
        margin = max(0.05 * self.column_length, 2.0 * spacing)
        # Does the tracked region cover (close to) the full column?
        self.coverage_ok = self.marker_span >= (self.column_length - margin)

        collapsed_in = self.collapsed & incol
        if not np.any(collapsed_in):
            self.prop_distance = 0.0
            self.arrested = None                # nothing collapsed: can't say
            self.arrest_X = None
            return

        front = float(np.nanmax(self.X[collapsed_in]))     # furthest collapsed marker (m)
        # Propagation distance is measured from the critical cut length (the end
        # of the saw-cut phase), i.e. how far the dynamic crack actually ran.
        self.prop_distance = max(0.0, front - self.critical_cut_length)
        self.front_position = front

        # Evidence types (intact markers must also be inside the column):
        intact_in = (~self.collapsed) & incol
        beyond_intact = bool(np.any(self.X[intact_in] > front + spacing))
        reached_column_end = front >= (self.column_length - margin)
        reached_marker_edge = front >= (self.marker_span - max(spacing, 1e-9))

        if beyond_intact:
            # Directly observed: collapse stops while intact markers remain ahead.
            self.arrested = True
            self.arrest_X = front
        elif reached_column_end:
            # Propagated to (near) the actual column end.
            self.arrested = False
            self.arrest_X = None
        elif reached_marker_edge and not self.coverage_ok:
            # Crack reached the edge of the tracked region, but that edge is short
            # of the column end - can't tell arrest from running out of ROI.
            self.arrested = None
            self.arrest_X = None
        else:
            self.arrested = None
            self.arrest_X = None

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
        rise = self.rise[self._propagation_mask() & np.isfinite(self.rise)]
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
        amp_c = self.amp[self.collapsed & self.in_column]
        return {
            "crack_speed_m_s": _round(self.speed, 2),
            "crack_speed_fit_r2": _round(self.speed_r2, 3),
            "crack_speed_n_markers": self.speed_n,
            "critical_cut_length_m": _round(self.critical_cut_length, 3),
            "crack_initiation_s": _round(self.t_init, 4),
            "n_markers_saw_excluded": int(self.saw_excluded.sum()),
            "propagation_distance_m": _round(self.prop_distance, 3),
            "front_position_m": _round(getattr(self, "front_position", np.nan), 3),
            "column_length_m": _round(self.column_length, 3),
            "tracked_extent_m": _round(self.marker_span, 3),
            "roi_covers_column": bool(self.coverage_ok),
            "scale_warning": bool(self.scale_warning),
            "n_outliers_trimmed": int(self.n_outliers_trimmed),
            "arrested": self.arrested,                    # True / False / None (indeterminate)
            "arrest_position_m": _round(self.arrest_X, 3),
            "column_tilt_deg": _round(self.tilt_deg, 2),
            "touchdown_distance_m": _round(self.touchdown_distance, 3),
            "touchdown_distance_std_m": _round(self.touchdown_distance_std, 3),
            "collapse_mean_mm": _round(float(np.mean(amp_c)) if amp_c.size else np.nan, 2),
            "collapse_max_mm": _round(float(np.max(amp_c)) if amp_c.size else np.nan, 2),
            "n_markers_total": int(self.amp.size),
            "n_markers_collapsed": int((self.collapsed & self.in_column).sum()),
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
            "saw_excluded": self.saw_excluded,
            "in_column": self.in_column,
        }).sort_values("along_column_m").reset_index(drop=True)

    def report(self, outdir="pst_results", prefix="pst"):
        os.makedirs(outdir, exist_ok=True)
        summary = self.summary()

        # --- console ---
        print("\n=== PST analysis summary ===")
        for k, v in summary.items():
            print(f"  {k:28s}: {v}")
        if self.arrested is True:
            print(f"  -> crack ARRESTED at {summary['propagation_distance_m']} m "
                  f"of {summary['column_length_m']} m")
        elif self.arrested is False:
            print(f"  -> crack propagated to the end of the column")
        else:
            print(f"  -> arrest INDETERMINATE: collapse reaches {summary['propagation_distance_m']} m "
                  f"but the tracked region may not cover the full column")
        if self.t_init is not None or self.cut_length_m is not None:
            print(f"  -> saw-cut phase excluded up to {self.critical_cut_length:.2f} m "
                  f"({int(self.saw_excluded.sum())} markers); speed/distance use the "
                  f"propagation phase only")
        if self.scale_warning:
            print(f"  WARNING: tracked extent ({self.raw_extent:.2f} m) exceeds the "
                  f"{self.column_length:.2f} m column. The scale (--mm-per-px / calibration) "
                  f"is likely wrong, or features outside the column (saw, hand, pit wall) are "
                  f"being tracked - distances and speed are unreliable until this is fixed.")
        if not self.coverage_ok:
            print(f"  WARNING: tracked region spans only {self.marker_span:.2f} m of the "
                  f"{self.column_length:.2f} m column - propagation distance and arrest are "
                  f"limited by ROI/marker coverage, not necessarily the crack. Extend the ROI "
                  f"to cover the full column for reliable arrest detection.")

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
        prop = self._propagation_mask()
        saw = self.saw_excluded & np.isfinite(self.onset)
        ax.scatter(self.X[prop], self.onset[prop], s=25, c="tab:red", label="propagation")
        if np.any(saw):
            ax.scatter(self.X[saw], self.onset[saw], s=22, c="tab:gray", marker="x",
                       label="saw-cut (excluded)")
        if np.isfinite(self.speed) and hasattr(self, "_speed_fit"):
            slope, intercept = self._speed_fit
            xs = np.linspace(self.X[prop].min(), self.X[prop].max(), 50)
            ax.plot(xs, slope * xs + intercept, "k--",
                    label=f"fit: c = {self.speed:.1f} m/s  (R²={self.speed_r2:.2f})")
        if self.critical_cut_length > 0:
            ax.axvline(self.critical_cut_length, color="tab:purple", ls="-.",
                       label=f"critical cut = {self.critical_cut_length:.2f} m")
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
        ax.axvline(self.marker_span, color="tab:orange", ls="-.",
                   label=f"tracked extent = {self.marker_span:.2f} m")
        ax.axvline(self.column_length, color="k", ls=":",
                   label=f"column end = {self.column_length:.2f} m")
        ax.set_xlabel("along-column position (m)")
        ax.set_ylabel("collapse amplitude (mm)")
        ax.set_title("Collapse profile along the column")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{prefix}_collapse_profile.png"), dpi=130)
        plt.close(fig)

        # 4) Marker overlay on the first frame (where the points are tracked)
        if self.frame0 is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(self.frame0[:, :, ::-1])      # BGR -> RGB
            x0, y0 = self.pos[0, :, 0], self.pos[0, :, 1]
            cats = [
                (~self.in_column, "red", "out of column (excluded)"),
                (self.saw_excluded, "orange", "saw-cut (excluded)"),
                (self.collapsed & self.in_column & ~self.saw_excluded, "lime", "propagation"),
                (~self.collapsed & self.in_column & ~self.saw_excluded, "deepskyblue", "not collapsed"),
            ]
            for m, color, label in cats:
                if np.any(m):
                    ax.scatter(x0[m], y0[m], s=18, c=color, edgecolors="black",
                               linewidths=0.4, label=f"{label} ({int(m.sum())})")
            ax.set_title(f"Tracked markers (n={self.amp.size}) on first frame")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
            ax.axis("off")
            fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{prefix}_markers_overlay.png"), dpi=130)
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
                            stabilize=args.stabilize, seed=args.seed,
                            grid_spacing=args.grid_spacing_px)

    # 4) Analysis
    analyzer = PSTAnalyzer(track,
                           column_length_m=args.column_length_cm / 100.0,
                           cut_from=args.cut_from,
                           onset_frac=args.onset_frac,
                           min_collapse_mm=args.min_collapse_mm,
                           exclude_saw=not args.no_exclude_saw,
                           cut_length_m=(args.cut_length_cm / 100.0
                                         if args.cut_length_cm is not None else None))
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
    p.add_argument("--n-markers", type=int, default=200, help="Max markers to track in the ROI.")
    p.add_argument("--seed", choices=["features", "grid", "color"], default="features",
                   help="Marker seeding: 'features' (corners), 'grid' (regular grid, for "
                        "low-texture snow), or 'color' (red paint-dot centroids, for a "
                        "spray-marked slab).")
    p.add_argument("--grid-spacing-px", type=int, default=25,
                   help="Grid spacing in pixels when --seed grid.")
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
    p.add_argument("--cut-length-cm", type=float, default=None,
                   help="Manual critical-cut length in cm; markers within it of the cut "
                        "end are excluded as saw-cut (overrides auto detection).")
    p.add_argument("--no-exclude-saw", action="store_true",
                   help="Disable saw-cut exclusion (analyze all collapsed markers).")
    p.add_argument("--out", default="pst_results", help="Output directory.")
    p.add_argument("--prefix", default="pst", help="Output file prefix.")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
