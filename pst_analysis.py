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
import cv2
import matplotlib
matplotlib.use("Agg")  # headless-safe: figures are written to disk, never shown
import matplotlib.pyplot as plt

from video_tracker import VideoUtil


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
                 onset_frac=0.10, touchdown_frac=0.90, min_collapse_mm=0.25,
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
        # raw (un-stabilized) pixel trajectories, for drawing on the actual frames
        self.pos_raw = np.asarray(track.get('positions_raw', pos), dtype=float)

        self._build_per_marker()
        self._flag_outliers()           # scale / out-of-column features
        self._flag_bad_tracks()         # optical-flow tracks that lost lock
        self._detect_crack_initiation()  # saw-cut length from pre-window collapse
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
        self.X = X
        self.w = w

        # --- Dynamic propagation window (from collective collapse velocity) ---
        # Locates the collapse event in time so onset/speed reflect the crack
        # front, not gradual settling. Used for TIMING only: the magnitude of
        # collapse is measured from the total settle (below), so a window that
        # lands late (e.g. on a creep/noise velocity blip) can no longer shrink
        # the reported collapse to nothing.
        self.win0, self.win1 = self._propagation_window(w)

        # Collapse magnitude (mm) = total settle: the final settled position minus
        # the initial (pre-collapse) position, each a robust median over a chunk of
        # the clip ends. This captures the full drop whether the slab collapses in
        # one fast step or keeps settling, and does not depend on where the window
        # landed - the previous plateau-just-after minus baseline-just-before the
        # window read ~1 mm when the window mis-located, hiding a real ~7 mm drop.
        k_end = max(3, int(round(0.03 * F)))
        baseline = np.nanmedian(w[:k_end], axis=0)          # initial position
        plateau = np.nanmedian(w[-k_end:], axis=0)          # settled position
        amp = plateau - baseline
        amp = np.where(np.isfinite(amp), amp, 0.0)

        onset = np.full(N, np.nan)
        touchdown = np.full(N, np.nan)
        collapsed = amp >= self.min_collapse_mm

        # Onset/touchdown TIMING: cross times of each marker's own rise, searched
        # only within the event window (plus a small pad) so slow pre/post drift
        # can't set a false onset. The crossing levels are fractions of the rise
        # *within the window* (evt_amp), not of the total settle, so a marker that
        # keeps creeping after the event still has a well-defined onset/touchdown
        # inside it.
        pad = int(round(0.25 * (self.win1 - self.win0))) + 2
        lo, hi = max(0, self.win0 - pad), min(F, self.win1 + pad + 1)
        evt_base = np.nanmedian(w[lo:self.win0 + 1], axis=0)
        evt_plateau = (np.nanmedian(w[self.win1:hi], axis=0)
                       if self.win1 < hi else w[min(self.win1, F - 1)])
        evt_amp = evt_plateau - evt_base
        for i in range(N):
            if not collapsed[i] or not (evt_amp[i] > 0):
                continue
            wi = w[:, i]
            onset[i] = self._cross_time(wi, evt_base[i] + self.onset_frac * evt_amp[i], lo, hi)
            touchdown[i] = self._cross_time(wi, evt_base[i] + self.touchdown_frac * evt_amp[i], lo, hi)

        self.baseline = baseline
        self.amp = amp
        self.collapsed = collapsed
        self.onset = onset
        self.touchdown = touchdown
        self.rise = touchdown - onset

    def _propagation_window(self, w):
        """
        Frame window [f0, f1] of the dynamic collapse event.

        Found from the collective downward-collapse velocity (median |dw/dt| across
        markers), smoothed. We take the *earliest* peak that is a strong fraction of
        the global maximum, not the global maximum itself: the slab often keeps
        creeping (or the camera jitters) long after it has collapsed, and that late
        motion can produce the single largest velocity blip. Locking onto it would
        place the window after the collapse, so onset/speed would miss the real
        front. Robust to slow pre/post drift that fools a cumulative window.
        """
        F = w.shape[0]
        vy = np.abs(np.gradient(np.nan_to_num(w, nan=0.0), axis=0))   # mm/frame
        coll = np.nanmedian(vy, axis=1)
        k = max(3, int(round(0.012 * self.fps)))     # ~12 ms smoothing
        if k % 2 == 0:
            k += 1
        coll = np.convolve(coll, np.ones(k) / k, mode='same')
        self._coll_vel = coll
        pv = float(np.nanmax(coll)) if coll.size else 0.0
        if not np.isfinite(pv) or pv <= 0:
            return 0, F - 1
        # First frame of the earliest run that reaches 60% of the global peak -
        # i.e. the first genuine collapse, even if a later blip is larger.
        strong = coll >= 0.60 * pv
        starts = np.where(strong & ~np.r_[False, strong[:-1]])[0]
        if starts.size:
            run0 = int(starts[0])
            run1 = run0
            while run1 < F - 1 and strong[run1 + 1]:
                run1 += 1
            peak = run0 + int(np.nanargmax(coll[run0:run1 + 1]))
        else:
            peak = int(np.nanargmax(coll))
        # Expand around that peak down to 25% of ITS height.
        thr = 0.25 * coll[peak]
        f0 = peak
        while f0 > 0 and coll[f0 - 1] > thr:
            f0 -= 1
        f1 = peak
        while f1 < F - 1 and coll[f1 + 1] > thr:
            f1 += 1
        return f0, f1

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

    def _flag_bad_tracks(self):
        """
        Flag collapsed markers whose collapse amplitude is a lost optical-flow
        track (e.g. a 426 mm "collapse" where the slab actually settled ~8 mm).
        Two complementary, scale-free tests - no absolute cutoff, so a video with
        genuinely large but *uniform* collapse isn't filtered:

        1. GLOBAL: amplitude far above the robust spread of all collapsed markers.
           This catches a *cluster* of lost tracks (common at the disturbed cut end
           / under stabilization), which a local-only test misses because the
           outliers sit next to each other and validate one another.
        2. LOCAL: amplitude far above immediate neighbours - catches an isolated
           jump inside a genuine spatial collapse trend, where the global spread is
           wide enough to hide it.

        Amplitude is a robust median of the clip ends, so a transient single-frame
        blip that recovers does not flag a marker. Flagged markers are excluded from
        the collapse stats, the speed fit, the propagation front, and every plot.
        """
        self.bad_track = np.zeros(self.amp.size, dtype=bool)
        idx = np.where(self.collapsed & self.in_column)[0]
        if idx.size < 5:
            return
        a = self.amp
        # (1) Global robust outlier (median absolute deviation of the whole set).
        gmed = np.median(a[idx])
        gmad = np.median(np.abs(a[idx] - gmed)) + 1e-9
        glob = (a > gmed + 12.0 * gmad) & (a > 4.0 * gmed)
        # (2) Local outlier vs nearby markers along the column.
        win = max(0.15, 3.0 * self._marker_spacing())
        for i in idx:
            if glob[i]:
                self.bad_track[i] = True
                continue
            neigh = idx[(np.abs(self.X[idx] - self.X[i]) <= win) & (idx != i)]
            if neigh.size < 3:
                neigh = idx[idx != i]
            med = np.median(a[neigh])
            mad = np.median(np.abs(a[neigh] - med)) + 1e-9
            if a[i] > med + 8.0 * mad and a[i] > 3.0 * med:
                self.bad_track[i] = True

    def _valid(self):
        """Markers usable for analysis: inside the column and not a bad track."""
        return self.in_column & ~self.bad_track

    def _detect_crack_initiation(self):
        """
        Estimate the saw-cut length (critical cut) and the saw-cut markers.

        The sawn part of the column has no weak layer left, so it settles *before*
        the dynamic propagation event. Such markers therefore show collapse already
        in their pre-window baseline. The critical cut is how far that pre-settled
        region reaches from the cut end; those markers are excluded from the speed /
        touchdown / front analysis.

        A manual `cut_length_m` overrides this (recommended - the operator knows the
        cut length); `exclude_saw=False` disables it. Auto-detection returns ~0 when
        the slab settles all at once (no distinct sawn-region pre-collapse on
        camera), in which case pass `--cut-length-cm`.
        """
        N = self.amp.size
        self.saw_excluded = np.zeros(N, dtype=bool)
        self.t_init = None
        self.critical_cut_length = 0.0

        # The saw-cut length is an experimental value the operator measures; it
        # can't be reliably inferred from the collapse video (the sawn region
        # settles before filming, or the whole slab moves at once). So it is taken
        # from --cut-length-cm when given, and left at 0 (with a note) otherwise.
        if self.cut_length_m is not None:
            self.critical_cut_length = float(self.cut_length_m)
            self.saw_excluded = self._valid() & (self.X <= self.critical_cut_length)

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

    def _cross_time(self, signal, level, lo=0, hi=None):
        """First time (s, linearly interpolated) `signal` reaches `level`,
        searching only within frame range [lo, hi)."""
        s = np.asarray(signal, dtype=float)
        hi = s.size if hi is None else hi
        seg = s[lo:hi]
        above = seg >= level
        if not np.any(above):
            return np.nan
        k = int(np.argmax(above)) + lo          # first index at/above level
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
        return (self.collapsed & self._valid() & np.isfinite(self.onset)
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
        # Restrict to valid markers (drop out-of-column outliers and bad tracks).
        valid = self._valid()
        Xv = self.X[valid]
        self.marker_span = float(np.nanmax(Xv)) if np.isfinite(Xv).any() else 0.0
        spacing = self._marker_spacing()
        margin = max(0.05 * self.column_length, 2.0 * spacing)
        # Does the tracked region cover (close to) the full column?
        self.coverage_ok = self.marker_span >= (self.column_length - margin)

        collapsed_in = self.collapsed & valid
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

        # If the front sits past the column end, the scale/ROI is off, not the
        # crack - the geometry can't be trusted, so don't claim arrest either way.
        if self.scale_warning or front > self.column_length + margin:
            self.arrested = None
            self.arrest_X = None
            return

        # Evidence types (intact markers must also be valid and in the column):
        intact_in = (~self.collapsed) & valid
        beyond_intact = bool(np.any(self.X[intact_in] > front + spacing))
        reached_column_end = front >= (self.column_length - margin)

        if reached_column_end:
            # Propagated to (near) the actual column end.
            self.arrested = False
            self.arrest_X = None
        elif self.coverage_ok and beyond_intact:
            # Full coverage and collapse stops while intact markers remain ahead:
            # a real arrest, observed within the column.
            self.arrested = True
            self.arrest_X = front
        else:
            # Incomplete coverage (front at the tracked edge, short of the column
            # end): can't tell a real arrest from simply running out of ROI.
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
        amp_c = self.amp[self.collapsed & self._valid()]
        return {
            "crack_speed_m_s": _round(self.speed, 2),
            "crack_speed_fit_r2": _round(self.speed_r2, 3),
            "crack_speed_n_markers": self.speed_n,
            "propagation_window_s": [_round(self.t[self.win0], 4), _round(self.t[self.win1], 4)],
            "propagation_window_frames": [int(self.win0), int(self.win1)],
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
            "n_bad_tracks": int(self.bad_track.sum()),
            "arrested": self.arrested,                    # True / False / None (indeterminate)
            "arrest_position_m": _round(self.arrest_X, 3),
            "column_tilt_deg": _round(self.tilt_deg, 2),
            "touchdown_distance_m": _round(self.touchdown_distance, 3),
            "touchdown_distance_std_m": _round(self.touchdown_distance_std, 3),
            "collapse_mean_mm": _round(float(np.mean(amp_c)) if amp_c.size else np.nan, 2),
            "collapse_max_mm": _round(float(np.max(amp_c)) if amp_c.size else np.nan, 2),
            "n_markers_total": int(self.amp.size),
            "n_markers_collapsed": int((self.collapsed & self.in_column).sum()),
            "collapse_net_negative": self._collapse_inverted()[0],
        }

    def _collapse_inverted(self):
        """Detect a net-UPWARD "collapse" over the event.

        A settling slab can only move downward, so a population of markers whose
        event amplitude is net-negative is physically impossible - it signals an
        inverted axis normal or (far more often) camera-motion stabilization that
        removed and inverted the real downward collapse. This is what turns a real
        collapse into "0 markers collapsed".

        Returns (is_inverted, n_up, n_down, frac_up) over valid in-column markers,
        where up/down count markers past +/- min_collapse_mm.
        """
        valid = self._valid()
        n = int(valid.sum())
        thr = self.min_collapse_mm
        n_down = int((valid & (self.amp >= thr)).sum())
        n_up = int((valid & (self.amp <= -thr)).sum())
        frac_up = (n_up / n) if n else 0.0
        inverted = bool(n and frac_up >= 0.30 and n_up > n_down)
        return inverted, n_up, n_down, frac_up

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
            "bad_track": self.bad_track,
        }).sort_values("along_column_m").reset_index(drop=True)

    def _window_pad(self):
        """Frame range [f0, f1) of the event window plus a small pad."""
        F = self.w.shape[0]
        pad = int(round(0.25 * (self.win1 - self.win0))) + 3
        return max(0, self.win0 - pad), min(F, self.win1 + pad + 1)

    def trajectory_table(self, window_only=True):
        """
        Per-frame, per-marker movement: along-column displacement and collapse
        (normal-to-axis), both in mm, with each marker's distance from the cut end.
        Restricted to the propagation window (+pad) unless window_only is False.
        """
        F = self.w.shape[0]
        f0, f1 = self._window_pad() if window_only else (0, F)
        Pm = self.pos.copy()
        Pm[:, :, 0] *= self.sx
        Pm[:, :, 1] *= self.sy
        disp = Pm - Pm[0][None, :, :]
        along = (disp @ self.axis) * 1000.0          # (F, N) mm along the column
        rows = []
        for i in range(self.amp.size):
            if not self.in_column[i]:
                continue
            for f in range(f0, f1):
                p = self.pos_raw[f, i]
                if not np.isfinite(p).all():
                    continue
                rows.append((int(i), round(float(self.X[i]), 4), int(f),
                             round(float(self.t[f]), 5),
                             round(float(along[f, i]), 3), round(float(self.w[f, i]), 3)))
        return pd.DataFrame(rows, columns=["marker", "along_column_m", "frame", "time_s",
                                           "along_disp_mm", "collapse_mm"])

    def _marker_bgr(self, i, xmin, xmax):
        """BGR color for marker i from a colormap of its along-column distance."""
        frac = (self.X[i] - xmin) / (xmax - xmin) if xmax > xmin else 0.5
        r, g, b, _ = plt.get_cmap("turbo")(float(np.clip(frac, 0, 1)))
        return int(b * 255), int(g * 255), int(r * 255)

    def save_event_video(self, frames, outpath, fps_out=30, trail=8):
        """
        Write an MP4 of the propagation event: markers drawn on the real frames,
        colored by along-column distance (colormap), with short motion trails.

        `frames` must be the BGR buffer aligned with the tracked positions
        (i.e. video_buffer[start:end]), shape (F, H, W, 3).
        """
        f0, f1 = self._window_pad()
        incol = np.where(self._valid())[0]
        if incol.size == 0:
            return None
        xmin, xmax = self.X[incol].min(), self.X[incol].max()
        h, w = frames.shape[1:3]
        writer = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_out, (w, h), isColor=True)
        for f in range(f0, f1):
            img = frames[f].copy()
            for i in incol:
                col = self._marker_bgr(i, xmin, xmax)
                for tf in range(max(f0, f - trail), f):
                    a, b = self.pos_raw[tf, i], self.pos_raw[tf + 1, i]
                    if np.isfinite(a).all() and np.isfinite(b).all():
                        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 1)
                p = self.pos_raw[f, i]
                if np.isfinite(p).all():
                    cv2.circle(img, (int(p[0]), int(p[1])), 4, col, -1)
            in_win = self.win0 <= f <= self.win1
            label = f"t={self.t[f]:.3f}s" + ("  [EVENT]" if in_win else "")
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 255) if in_win else (255, 255, 255), 2)
            writer.write(img)
        writer.release()
        return outpath

    def report(self, outdir="pst_results", prefix="pst", full_trajectories=False):
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
        print(f"  -> propagation window: {self.t[self.win0]:.3f}-{self.t[self.win1]:.3f} s "
              f"(frames {self.win0}-{self.win1}); onset/speed use this window only")
        if np.isfinite(self.speed_r2) and self.speed_r2 < 0.5:
            win_frames = self.win1 - self.win0
            print(f"  WARNING: crack-speed fit is poor (R²={self.speed_r2:.2f}). Onsets are "
                  f"near-simultaneous (the event spans only {win_frames} frames) - the front "
                  f"likely crossed the tracked region faster than the frame rate resolves. "
                  f"See the kymograph; a higher fps or a longer tracked span is needed for speed.")
        inverted, n_up, n_down, _ = self._collapse_inverted()
        if inverted:
            print(f"  WARNING: collapse is net-NEGATIVE - {n_up} of "
                  f"{int(self._valid().sum())} markers moved UP during the event "
                  f"(only {n_down} settled down). A slab can only settle downward, so "
                  f"this is not real collapse: the camera-motion stabilization is "
                  f"removing/inverting it (drop --stabilize, or draw AND commit a "
                  f"static zone with 'a'/Enter), or the column-axis normal is flipped. "
                  f"Collapse stats below are unreliable until this is fixed.")
        if int(self.bad_track.sum()):
            print(f"  -> excluded {int(self.bad_track.sum())} bad track(s) "
                  f"(collapse far above local neighbours - optical flow lost lock)")
        if self.critical_cut_length > 0:
            print(f"  -> saw-cut length {self.critical_cut_length:.2f} m "
                  f"({int(self.saw_excluded.sum())} markers); speed/distance use the "
                  f"propagation phase only")
            # Guard: a cut length past where collapse actually is removes every
            # propagation marker, leaving nothing for the speed fit.
            cv = self.collapsed & self._valid()
            remaining = int((cv & ~self.saw_excluded).sum())
            if cv.sum() and remaining < 3:
                print(f"  WARNING: the saw-cut length {self.critical_cut_length:.2f} m covers "
                      f"almost all collapsed markers (only {remaining} left past it). Likely the "
                      f"--cut-length-cm is too large, the --cut-from end is wrong, or the scale "
                      f"is off (collapse spans X={self.X[cv].min():.2f}-{self.X[cv].max():.2f} m).")
        elif self.cut_length_m is None:
            print(f"  note: saw-cut length auto-detected as 0 (slab may settle all at "
                  f"once on camera); pass --cut-length-cm with the measured cut length.")
        if self.fps <= 60:
            print(f"  WARNING: fps is {self.fps:.0f}. PST crack propagation is fast "
                  f"(~10-40 m/s); pass --fps for high-speed footage (e.g. 240) or the "
                  f"crack speed will be wrong by the frame-rate ratio.")
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
        traj = self.trajectory_table(window_only=not full_trajectories)
        traj.to_csv(os.path.join(outdir, f"{prefix}_trajectories.csv"), index=False)
        import pst_plots
        pst_plots.per_video_figures(self, outdir, prefix)
        scope = "full record" if full_trajectories else "propagation window"
        print(f"\nWrote summary, per-marker CSV, per-frame trajectories ({scope}), "
              f"and plots to: {outdir}/")
        return summary


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
    v.roi_polygon = None


def _set_roi_polygon_from_arg(v, spec):
    """Build a polygon ROI from 'x1,y1 x2,y2 ...' (or 'x1,y1,x2,y2,...')."""
    nums = [int(s) for s in spec.replace(",", " ").split()]
    if len(nums) < 6 or len(nums) % 2:
        raise SystemExit("error: --roi-poly needs >= 3 'x,y' vertex pairs.")
    v._mask_from_polygon(list(zip(nums[0::2], nums[1::2])))


def _mask_bbox(mask_gray):
    """Bounding box [x, y, w, h] of the nonzero region, or None."""
    ys, xs = np.where(mask_gray > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def _rects_from_mask(mask):
    """Bounding boxes [x, y, w, h] of each connected region in a binary mask."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    return [[int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])]
            for i in range(1, n)]


def save_config(v, path):
    """Save the ROI, pixel scale, and static zone(s) so a re-run can skip the
    interactive setup. Stored as rectangles + pixel sizes (human-editable JSON)."""
    roi_gray = cv2.cvtColor(v.mask, cv2.COLOR_BGR2GRAY)
    cfg = {
        "video": os.path.basename(v.video_file),
        "pix_width_m_per_px": float(v.pix_width),
        "pix_height_m_per_px": float(v.pix_height),
        # roi_xywh is the bounding box (kept for reference); a polygon ROI, when
        # used, is the authoritative shape and is restored from roi_polygon.
        "roi_xywh": _mask_bbox(roi_gray),
        "roi_polygon": ([[int(x), int(y)] for x, y in v.roi_polygon]
                        if getattr(v, "roi_polygon", None) else None),
        "static_zones_xywh": _rects_from_mask(v.stab_mask) if v.stab_mask is not None else [],
    }
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    return cfg


def load_config(v, path):
    """Apply a saved config to a VideoUtil: ROI mask, pixel scale, static zone(s)."""
    with open(path) as fh:
        cfg = json.load(fh)
    v.pix_width = float(cfg["pix_width_m_per_px"])
    v.pix_height = float(cfg["pix_height_m_per_px"])
    poly = cfg.get("roi_polygon")
    roi = cfg.get("roi_xywh")
    if poly:                                    # polygon ROI takes precedence
        v._mask_from_polygon([(int(x), int(y)) for x, y in poly])
    elif roi:
        x, y, w, h = roi
        mask = np.zeros_like(v.video_buffer[0])
        mask[y:y + h, x:x + w] = 255
        v.mask = mask
    zones = cfg.get("static_zones_xywh") or []
    if zones:
        sm = np.zeros(v.video_buffer[0].shape[:2], dtype=np.uint8)
        for x, y, w, h in zones:
            sm[y:y + h, x:x + w] = 255
        v.stab_mask = sm
    return cfg


def load_meta(args):
    """Fill CLI args left unset (fps, saw-cut length, column length, cut-from) from a
    per-video metadata registry, so per-experiment facts live in one place instead of
    being re-typed for every run. Explicit CLI flags always take precedence.

    The registry is JSON keyed by video name:
        {"<video stem>": {"saw_cut_cm", "fps", "column_length_cm", "cut_from", ...}}
    Path is --meta, else "metadata.json" alongside the video. ROI/scale/static stay in
    the per-video config (they can have variants and are machine-generated).
    """
    path = args.meta or os.path.join(os.path.dirname(args.path) or ".", "metadata.json")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        reg = json.load(fh)
    stem = os.path.splitext(os.path.basename(args.path))[0]
    m = reg.get(stem)
    if not m:
        print(f"Note: no metadata entry for '{stem}' in {path}.")
        return
    filled = {}
    if args.fps is None and m.get("fps") is not None:
        args.fps = float(m["fps"]); filled["fps"] = args.fps
    if args.cut_length_cm is None and m.get("saw_cut_cm") is not None:
        args.cut_length_cm = float(m["saw_cut_cm"]); filled["cut_length_cm"] = args.cut_length_cm
    if args.column_length_cm is None and m.get("column_length_cm") is not None:
        args.column_length_cm = float(m["column_length_cm"]); filled["column_length_cm"] = args.column_length_cm
    if args.cut_from is None and m.get("cut_from"):
        args.cut_from = m["cut_from"]; filled["cut_from"] = args.cut_from
    if filled:
        print(f"Loaded metadata for '{stem}' from {path}: "
              + ", ".join(f"{k}={v}" for k, v in filled.items()))


def run(args):
    # Per-experiment defaults (saw cut, fps, column length, cut-from) from the
    # metadata registry; explicit CLI flags already on `args` win.
    load_meta(args)
    if args.cut_from is None:
        args.cut_from = "left"
    if args.column_length_cm is None:
        raise SystemExit("error: column length is required - pass --column-length-cm, or "
                         "add column_length_cm to this video's entry in the metadata file.")

    v = VideoUtil(args.path, interactive=False)

    # Results go in a per-video subdirectory (named after the clip) so runs on
    # different videos don't overwrite each other.
    video_stem = os.path.splitext(os.path.basename(args.path))[0]
    outdir = os.path.join(args.out, video_stem)
    prefix = args.prefix if args.prefix else video_stem
    config_path = os.path.join(outdir, f"{prefix}_config.json")
    load_path = args.config or (config_path if args.reuse else None)

    if load_path and os.path.exists(load_path):
        # Reuse a saved setup - no interactive ROI/scale/static steps.
        cfg = load_config(v, load_path)
        print(f"Loaded ROI/scale/static from {load_path}: "
              f"scale {cfg['pix_width_m_per_px'] * 1000:.3f} mm/px, "
              f"{len(cfg.get('static_zones_xywh') or [])} static zone(s)")
    else:
        # 1) Region of interest = the slab (where markers/features live)
        if args.roi_poly:
            _set_roi_polygon_from_arg(v, args.roi_poly)
        elif args.roi:
            _set_roi_from_arg(v, args.roi)
        elif args.roi_shape == "polygon":
            print("Trace a POLYGON around the slab (click vertices), then Enter/Esc.")
            v.set_roi_polygon()
        else:
            print("Draw a box around the slab markers (the part that collapses), then Esc.")
            v.set_roi()

        # 2) Spatial scale
        if args.mm_per_px:
            v.pix_width = v.pix_height = args.mm_per_px / 1000.0
            print(f"Scale set: {args.mm_per_px} mm/px")
        elif args.scale_from_column:
            print(f"Calibrate scale: drag a line across the {args.column_length_cm:.0f} cm "
                  f"column (end to end). Tilt doesn't matter.")
            v.set_scale_from_length(args.column_length_cm / 100.0)
        else:
            print("Calibrate scale: draw a rectangle of known size, then enter its size.")
            v.set_pxl_size()

        # 2b) Static zone for camera-motion (noise) compensation
        if args.stabilize:
            if args.static_roi:
                x, y, w, h = (int(s) for s in args.static_roi.split(","))
                sm = np.zeros(v.video_buffer[0].shape[:2], dtype=np.uint8)
                sm[y:y + h, x:x + w] = 255
                v.stab_mask = sm
            elif args.draw_static:
                print("Draw the STATIC reference zone(s) for stabilization, then Esc.")
                v.set_static_area()

        # Save the setup so the next run can reuse it with --reuse / --config.
        os.makedirs(outdir, exist_ok=True)
        save_config(v, config_path)
        print(f"Saved ROI/scale/static to {config_path} (re-run with --reuse to skip setup)")

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
    analyzer.report(outdir=outdir, prefix=prefix,
                    full_trajectories=args.full_trajectories)

    # 5) Annotated event video (markers colored by along-column distance)
    if not args.no_video:
        frames = v.video_buffer[track['start']:track['end']]
        out_mp4 = os.path.join(outdir, f"{prefix}_event.mp4")
        if analyzer.save_event_video(frames, out_mp4, fps_out=args.video_fps):
            print(f"Wrote annotated event video: {out_mp4}")


def build_parser():
    p = argparse.ArgumentParser(description="PST fracture-propagation analysis from video.")
    p.add_argument("-v", "--path", required=True, help="Path to the PST video file.")
    p.add_argument("--meta", default=None,
                   help="Per-video metadata registry JSON (saw cut, fps, column length, "
                        "cut-from), keyed by video name. Defaults to 'metadata.json' next "
                        "to the video. Supplies any of those values not given on the CLI.")
    p.add_argument("--column-length-cm", type=float, default=None,
                   help="Isolated column length in cm (used to judge arrest). Required "
                        "unless supplied by the metadata file.")
    p.add_argument("--cut-from", choices=["left", "right"], default=None,
                   help="Which image end the saw cut starts from (along-column origin). "
                        "From the metadata file if unset, else 'left'.")
    p.add_argument("--mm-per-px", type=float, default=None,
                   help="Spatial scale in mm/pixel. If omitted, calibrate interactively.")
    p.add_argument("--scale-from-column", action="store_true",
                   help="Derive mm/px from the known column length: click the two column "
                        "ends (tilt-independent), instead of the rectangle calibration. "
                        "Uses --column-length-cm / the metadata file.")
    p.add_argument("--roi", default=None,
                   help="Slab ROI as 'x,y,w,h' in pixels. If omitted, draw it interactively.")
    p.add_argument("--roi-shape", choices=["rect", "polygon"], default="rect",
                   help="Interactive ROI shape. 'polygon' traces an arbitrary outline "
                        "(for tilted/non-rectangular blocks); 'rect' (default) draws a box.")
    p.add_argument("--roi-poly", default=None,
                   help="Polygon ROI as '>=3 x,y vertex pairs (e.g. '10,20 300,40 280,200'). "
                        "Non-interactive; overrides --roi/--roi-shape.")
    p.add_argument("--reuse", action="store_true",
                   help="Reuse the saved ROI/scale/static config for this video (skip the "
                        "interactive setup). The config is auto-saved on the first run.")
    p.add_argument("--config", default=None,
                   help="Path to a saved config JSON to load (ROI/scale/static).")
    p.add_argument("--fps", type=float, default=None,
                   help="Override video fps. From the metadata file if unset.")
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
                   help="Camera-motion (noise) compensation from background features. "
                        "Use for handheld footage; needs good background outside the ROI.")
    p.add_argument("--draw-static", action="store_true",
                   help="With --stabilize, interactively draw the static reference zone(s) "
                        "instead of using everything outside the ROI.")
    p.add_argument("--static-roi", default=None,
                   help="With --stabilize, static reference zone as 'x,y,w,h' in pixels.")
    p.add_argument("--no-video", action="store_true",
                   help="Skip the annotated event video.")
    p.add_argument("--video-fps", type=float, default=30.0,
                   help="Playback fps of the annotated event video.")
    p.add_argument("--full-trajectories", action="store_true",
                   help="Export per-frame trajectories for the whole record, not just "
                        "the propagation window.")
    p.add_argument("--onset-frac", type=float, default=0.10,
                   help="Fraction of a marker's collapse used as the onset threshold.")
    p.add_argument("--min-collapse-mm", type=float, default=0.25,
                   help="Collapse below this (mm) counts as 'not collapsed'.")
    p.add_argument("--cut-length-cm", type=float, default=None,
                   help="Manual critical-cut length in cm; markers within it of the cut "
                        "end are excluded as saw-cut (overrides auto detection). From the "
                        "metadata file (saw_cut_cm) if unset.")
    p.add_argument("--no-exclude-saw", action="store_true",
                   help="Disable saw-cut exclusion (analyze all collapsed markers).")
    p.add_argument("--out", default="pst_results",
                   help="Base output directory; results go in <out>/<video_name>/.")
    p.add_argument("--prefix", default=None,
                   help="Output file prefix (defaults to the video name).")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
