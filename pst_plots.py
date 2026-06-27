# -*- coding: utf-8 -*-
"""
PST plotting - every figure in one place
========================================

Single module for all PST figures, per-video and cross-video, so they draw from
**one** collapse-data definition (`collapse_sample`) and one cut origin (the
measured critical cut, `critical_cut_length_m`). This replaces the old
``compare_collapse.py``, ``compare_collapse_box.py`` and ``pst_compare.py``
scripts and the plotting that used to live inside ``pst_analysis.py``.

Per-video figures are produced during analysis via ``per_video_figures(analyzer,
...)``, called from ``PSTAnalyzer.report()``. Cross-video figures have a CLI:

    python pst_plots.py compare --base pst_results
    python pst_plots.py box --collapse-time 2026-03-17T09:40 --reference Beehive_0

@author: Ron Simenhois
"""

import os
import glob
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# Box-plot fills: post-collapse boxes share one neutral fill; the reference is darker.
POST_FILL = "0.85"          # light gray
REF_FILL = "0.55"           # medium gray, still clearly distinct


# --------------------------------------------------------------------------- #
# Shared data layer - one definition of "the collapse data" for every figure
# --------------------------------------------------------------------------- #
def round_or_none(v, n=3):
    """Round to n places, or None for missing/non-finite values (JSON/CSV safe)."""
    try:
        if v is None or not np.isfinite(v):
            return None
    except TypeError:
        return v
    return round(float(v), n)


def collapse_sample(df, critical_cut_m, usable_m=None, trim_outliers=False, whis=1.5):
    """The single keep-mask + collapse arrays that every figure consumes.

    Keeps valid, in-column, collapsed markers past the measured critical cut and
    outside the excluded saw region. Returns ``(dist_from_cut_m, collapse_mm,
    kept_df)``, with distance measured from the cut. An optional Tukey trim drops
    amplitude outliers beyond ``whis`` * IQR (used by the box plot).
    """
    along = df["along_column_m"].to_numpy()
    keep = (df["collapsed"].to_numpy(bool)
            & df["in_column"].to_numpy(bool)
            & ~df["bad_track"].to_numpy(bool)
            & ~df["saw_excluded"].to_numpy(bool)
            & (along >= critical_cut_m))
    if usable_m:
        keep &= along <= usable_m
    sub = df[keep].copy()
    dist = sub["along_column_m"].to_numpy() - critical_cut_m
    amp = sub["collapse_mm"].to_numpy()
    if trim_outliers and amp.size >= 4:
        q1, q3 = np.percentile(amp, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - whis * iqr, q3 + whis * iqr
        m = (amp >= lo) & (amp <= hi)
        dist, amp, sub = dist[m], amp[m], sub[m]
    return dist, amp, sub


def binned_profile(dist, amp, bin_m=0.10, smooth_win=1):
    """Median collapse per distance bin - one routine for per-video and cross-video.

    Bins ``amp`` over ``dist`` in ``bin_m`` steps, taking the median per populated
    bin, then an optional centered moving average of ``smooth_win`` bins. Returns
    ``(bin_centers, values)`` for populated bins only.
    """
    dist = np.asarray(dist, float)
    amp = np.asarray(amp, float)
    if dist.size == 0:
        return np.array([]), np.array([])
    lo, hi = float(dist.min()), float(dist.max())
    if hi <= lo:
        return np.array([lo]), np.array([float(np.median(amp))])
    nb = max(1, int(np.ceil((hi - lo) / bin_m)))
    edges = np.linspace(lo, hi, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bid = np.clip(np.digitize(dist, edges) - 1, 0, len(centers) - 1)
    vals = np.full(len(centers), np.nan)
    for b in range(len(centers)):
        sel = bid == b
        if sel.any():
            vals[b] = np.median(amp[sel])
    good = np.isfinite(vals)
    centers, vals = centers[good], vals[good]
    if smooth_win > 1 and vals.size >= smooth_win:
        k = np.ones(smooth_win) / smooth_win
        vals = np.convolve(vals, k, mode="same")
    return centers, vals


def find_runs(paths, base):
    """Locate result dirs (those holding *_summary.json and *_markers.csv)."""
    candidates = list(paths)
    if base:
        candidates += sorted(os.path.join(base, d) for d in os.listdir(base)
                             if os.path.isdir(os.path.join(base, d)))
    runs, seen = [], set()
    for d in candidates:
        d = os.path.normpath(d)
        if d in seen:
            continue
        seen.add(d)
        summ = glob.glob(os.path.join(d, "*_summary.json"))
        mark = glob.glob(os.path.join(d, "*_markers.csv"))
        if summ and mark:
            runs.append((os.path.basename(d), summ[0], mark[0]))
    return runs


def load_run(name, summary_path, markers_path, meta=None, collapse_time=None):
    """Standardized per-video record consumed by every cross-video figure."""
    with open(summary_path) as fh:
        summary = json.load(fh)
    df = pd.read_csv(markers_path)
    meta = meta or {}
    cut = summary.get("critical_cut_length_m") or 0.0
    usable_m = (float(meta["usable_extent_cm"]) / 100.0
                if meta.get("usable_extent_cm") else None)
    dt_h = None
    if collapse_time and meta.get("date"):
        t0 = datetime.fromisoformat(collapse_time)
        dt_h = (datetime.fromisoformat(meta["date"]) - t0).total_seconds() / 3600.0
    return {"name": name, "summary": summary, "df": df,
            "critical_cut_m": float(cut), "usable_m": usable_m, "dt_h": dt_h}


def load_runs(dirs, base, meta_path=None, collapse_time=None):
    """Discover result dirs and load each into a standardized record."""
    found = find_runs(dirs, base)
    reg = {}
    if meta_path and os.path.exists(meta_path):
        with open(meta_path) as fh:
            reg = json.load(fh)
    return [load_run(n, s, m, reg.get(n), collapse_time) for n, s, m in found]


# --------------------------------------------------------------------------- #
# Per-video figures (called from PSTAnalyzer.report via per_video_figures)
# --------------------------------------------------------------------------- #
def _fig_collapse_curves(a, outdir, prefix):
    """Collapse-vs-time curves, colored by along-column position so the
    left-to-right propagation delay is visible. The window is shaded."""
    fig, ax = plt.subplots(figsize=(8, 5))
    idx = np.where(a.collapsed & a._valid())[0]
    if idx.size:
        order = idx[np.argsort(a.X[idx])]
        xmin, xmax = a.X[order].min(), a.X[order].max()
        cmap = plt.get_cmap("viridis")
        for i in order:
            frac = (a.X[i] - xmin) / (xmax - xmin) if xmax > xmin else 0.5
            ax.plot(a.t, a.w[:, i], lw=1, alpha=0.85, color=cmap(frac))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(xmin, xmax))
        fig.colorbar(sm, ax=ax, label="along-column position (m)")
    ax.axvspan(a.t[a.win0], a.t[a.win1], color="orange", alpha=0.15,
               label="propagation window")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("collapse (mm, downward +)")
    ax.set_title("Slab marker collapse vs time (color = position)")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{prefix}_collapse_curves.png"), dpi=130)
    plt.close(fig)


def _fig_kymograph(a, outdir, prefix):
    """Collapse over (along-column position, time); the propagation front shows as
    a tilted edge and the window is marked."""
    fig, ax = plt.subplots(figsize=(9, 5))
    incol = a._valid() & np.isfinite(a.X)
    if incol.sum() >= 2:
        nb = min(30, max(6, int(incol.sum())))
        Xi = a.X[incol]
        edges = np.linspace(Xi.min(), Xi.max(), nb + 1)
        binid = np.clip(np.digitize(a.X, edges) - 1, 0, nb - 1)
        ky = np.full((nb, a.w.shape[0]), np.nan)
        for b in range(nb):
            sel = incol & (binid == b)
            if sel.any():
                ky[b] = np.nanmean(a.w[:, sel], axis=1)
        im = ax.imshow(ky, aspect="auto", origin="lower", cmap="viridis",
                       extent=[a.t[0], a.t[-1], Xi.min(), Xi.max()])
        fig.colorbar(im, ax=ax, label="collapse (mm)")
        ax.axvline(a.t[a.win0], color="white", ls="--", lw=1)
        ax.axvline(a.t[a.win1], color="white", ls="--", lw=1)
    ax.set_xlabel("time (s)"); ax.set_ylabel("along-column position (m)")
    ax.set_title("Collapse kymograph (dashed = propagation window)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{prefix}_kymograph.png"), dpi=130)
    plt.close(fig)


def _fig_crack_speed(a, outdir, prefix):
    """Onset time vs along-column position, with the crack-speed fit."""
    fig, ax = plt.subplots(figsize=(8, 5))
    prop = a._propagation_mask()
    saw = a.saw_excluded & np.isfinite(a.onset)
    ax.scatter(a.X[prop], a.onset[prop], s=25, c="tab:red", label="propagation")
    if np.any(saw):
        ax.scatter(a.X[saw], a.onset[saw], s=22, c="tab:gray", marker="x",
                   label="saw-cut (excluded)")
    if np.isfinite(a.speed) and hasattr(a, "_speed_fit"):
        slope, intercept = a._speed_fit
        xs = np.linspace(a.X[prop].min(), a.X[prop].max(), 50)
        ax.plot(xs, slope * xs + intercept, "k--",
                label=f"fit: c = {a.speed:.1f} m/s  (R²={a.speed_r2:.2f})")
    if a.critical_cut_length > 0:
        ax.axvline(a.critical_cut_length, color="tab:purple", ls="-.",
                   label=f"critical cut = {a.critical_cut_length:.2f} m")
    ax.legend()
    ax.set_xlabel("along-column position (m)")
    ax.set_ylabel("collapse-onset time (s)")
    ax.set_title("Crack propagation: onset time vs position")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{prefix}_crack_speed.png"), dpi=130)
    plt.close(fig)


def _fig_collapse_profile(a, outdir, prefix):
    """Collapse amplitude vs position, with propagation distance / column end."""
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = a._valid()
    coll = a.collapsed & valid
    intact = (~a.collapsed) & valid
    ax.scatter(a.X[coll], a.amp[coll], s=25, c="tab:blue", label="collapsed")
    if np.any(intact):
        ax.scatter(a.X[intact], a.amp[intact], s=18, c="lightgray", label="not collapsed")
    ax.axvline(a.prop_distance, color="tab:green", ls="--",
               label=f"propagation = {a.prop_distance:.2f} m")
    ax.axvline(a.marker_span, color="tab:orange", ls="-.",
               label=f"tracked extent = {a.marker_span:.2f} m")
    ax.axvline(a.column_length, color="k", ls=":",
               label=f"column end = {a.column_length:.2f} m")
    ax.set_xlabel("along-column position (m)")
    ax.set_ylabel("collapse amplitude (mm)")
    ax.set_title("Collapse profile along the column")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{prefix}_collapse_profile.png"), dpi=130)
    plt.close(fig)


def _fig_collapse_line(a, outdir, prefix, include_sawcut):
    """Collapse magnitude vs along-column distance as a connecting line.

    The line is the binned median of the *collapsed* markers (via the shared
    `binned_profile`, so it matches the cross-video profile), with every marker
    drawn as a faint scatter. ``include_sawcut`` keeps the saw-cut region as a
    separate orange line; otherwise saw-cut markers are omitted.
    """
    valid = a._valid()
    X, A = a.X[valid], a.amp[valid]
    saw = a.saw_excluded[valid]
    coll = a.collapsed[valid]
    if X.size < 2:
        return

    def draw_group(ax, m, color, line_kw, scatter_label=False):
        mc, mi = m & coll, m & ~coll
        if np.any(mi):
            ax.scatter(X[mi], A[mi], s=12, c="lightgray", alpha=0.6,
                       label="not collapsed" if scatter_label else None)
        if np.any(mc):
            ax.scatter(X[mc], A[mc], s=12, c=color, alpha=0.30)
            bx, by = binned_profile(X[mc], A[mc], bin_m=0.10)
            ax.plot(bx, by, lw=2, ms=4, color=color, **line_kw)

    prop = ~saw
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_group(ax, prop, "tab:blue",
               dict(marker="o", label="collapse (propagation)" if include_sawcut else "collapse"),
               scatter_label=True)
    if include_sawcut:
        if np.any(saw):
            draw_group(ax, saw, "tab:orange", dict(marker="s", label="saw-cut region"))
        if a.critical_cut_length > 0:
            ax.axvline(a.critical_cut_length, color="tab:purple", ls="-.",
                       label=f"critical cut = {a.critical_cut_length:.2f} m")
    ax.axvline(a.column_length, color="k", ls=":",
               label=f"column end = {a.column_length:.2f} m")
    ax.set_xlabel("along-column distance (m)")
    ax.set_ylabel("collapse magnitude (mm)")
    ax.set_title("Collapse profile along the column"
                 + ("" if include_sawcut else " (saw-cut omitted)"))
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    suffix = "collapse_line" if include_sawcut else "collapse_line_no_sawcut"
    fig.savefig(os.path.join(outdir, f"{prefix}_{suffix}.png"), dpi=130)
    plt.close(fig)


def _fig_markers_overlay(a, outdir, prefix):
    """First-frame image with each tracked marker colored by its classification."""
    if a.frame0 is None:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(a.frame0[:, :, ::-1])      # BGR -> RGB
    x0, y0 = a.pos[0, :, 0], a.pos[0, :, 1]
    valid = a._valid()
    cats = [
        (~a.in_column, "red", "out of column (excluded)"),
        (a.bad_track, "magenta", "bad track (excluded)"),
        (a.saw_excluded & valid, "orange", "saw-cut (excluded)"),
        (a.collapsed & valid & ~a.saw_excluded, "lime", "propagation"),
        (~a.collapsed & valid & ~a.saw_excluded, "deepskyblue", "not collapsed"),
    ]
    for m, color, label in cats:
        if np.any(m):
            ax.scatter(x0[m], y0[m], s=18, c=color, edgecolors="black",
                       linewidths=0.4, label=f"{label} ({int(m.sum())})")
    ax.set_title(f"Tracked markers (n={a.amp.size}) on first frame")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{prefix}_markers_overlay.png"), dpi=130)
    plt.close(fig)


def per_video_figures(analyzer, outdir, prefix):
    """Write every per-video figure for one analyzed PST clip."""
    _fig_collapse_curves(analyzer, outdir, prefix)
    _fig_kymograph(analyzer, outdir, prefix)
    _fig_crack_speed(analyzer, outdir, prefix)
    _fig_collapse_profile(analyzer, outdir, prefix)
    _fig_collapse_line(analyzer, outdir, prefix, include_sawcut=True)
    _fig_collapse_line(analyzer, outdir, prefix, include_sawcut=False)
    _fig_markers_overlay(analyzer, outdir, prefix)


# --------------------------------------------------------------------------- #
# Cross-video figure: collapse-profile comparison (+ stats table)
# --------------------------------------------------------------------------- #
def compare_profiles(runs, outdir, bin_m=0.10, near_window_m=0.15, color_by_time=False):
    """Two-panel comparison: absolute collapse profile from the cut, and the same
    normalized to the near-cut collapse. Also writes a per-video stats CSV."""
    os.makedirs(outdir, exist_ok=True)
    if color_by_time and any(r["dt_h"] is not None for r in runs):
        dts = [r["dt_h"] for r in runs if r["dt_h"] is not None]
        norm = Normalize(min(dts), max(dts))
        cmap = plt.get_cmap("cool")
        def line_color(r, k):
            return cmap(norm(r["dt_h"])) if r["dt_h"] is not None else "black"
    else:
        tab = plt.get_cmap("tab10")
        def line_color(r, k):
            return tab(k % 10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    stats = []
    for k, r in enumerate(runs):
        color = line_color(r, k)
        dist, amp, _ = collapse_sample(r["df"], r["critical_cut_m"], r["usable_m"])
        xs, ys = binned_profile(dist, amp, bin_m)

        ax1.scatter(dist, amp, s=10, alpha=0.20, color=color)
        if xs.size:
            ax1.plot(xs, ys, "-o", color=color, lw=2, ms=4,
                     label=f'{r["name"]} (cut {r["critical_cut_m"]:.2f} m, n={dist.size})')

        slope = (np.polyfit(dist, amp, 1)[0]
                 if (dist.size >= 3 and np.ptp(dist) > 0) else np.nan)
        near = far = np.nan
        if xs.size:
            near_sel = xs <= near_window_m
            near = float(np.median(ys[near_sel])) if np.any(near_sel) else float(ys[0])
            far = float(ys[-1])
            if near and np.isfinite(near) and near > 0:
                ax2.plot(xs, ys / near, "-o", color=color, lw=2, ms=4, label=r["name"])

        stats.append({
            "video": r["name"],
            "critical_cut_m": round_or_none(r["critical_cut_m"]),
            "n_markers": int(dist.size),
            "collapse_near_mm": round_or_none(near, 2),
            "collapse_far_mm": round_or_none(far, 2),
            "collapse_slope_mm_per_m": round_or_none(slope, 2),
            "far_over_near": round_or_none(far / near, 2) if near else None,
            "crack_speed_m_s": r["summary"].get("crack_speed_m_s"),
            "propagation_distance_m": r["summary"].get("propagation_distance_m"),
        })

    ax1.set_xlabel("distance from saw cut (m)")
    ax1.set_ylabel("collapse magnitude (mm)")
    ax1.set_title("Collapse profile from the saw cut, per video")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

    ax2.axhline(1.0, color="gray", ls=":")
    ax2.set_xlabel("distance from saw cut (m)")
    ax2.set_ylabel("collapse / near-cut collapse")
    ax2.set_title("Relative change in collapse (normalized at the cut)")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(outdir, "collapse_comparison.png")
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    table = pd.DataFrame(stats)
    csv_path = os.path.join(outdir, "collapse_comparison.csv")
    table.to_csv(csv_path, index=False)
    print("\n=== PST collapse comparison ===")
    print(table.to_string(index=False))
    print(f"\nWrote {fig_path} and {csv_path}")
    return table


# --------------------------------------------------------------------------- #
# Cross-video figure: collapse-amplitude box plot on a broken time axis
# --------------------------------------------------------------------------- #
def _draw_boxes(ax, group, reference, rng):
    """One box per item: light-gray fill, black lines, raw measurements jittered."""
    black = dict(color="black", lw=1.2)
    for it in group:
        is_ref = it["name"] == reference
        fill = REF_FILL if is_ref else POST_FILL
        ax.boxplot([it["vals"]], positions=[it["dt_h"]], widths=it["width"],
                   patch_artist=True, manage_ticks=False, showfliers=False,
                   medianprops=dict(color="black", lw=1.8),
                   whiskerprops=black, capprops=black,
                   boxprops=dict(facecolor=fill, edgecolor="black", lw=1.2))
        jx = rng.uniform(-0.3, 0.3, it["vals"].size) * it["width"]
        ax.scatter(it["dt_h"] + jx, it["vals"], s=12, color="0.2", alpha=0.5,
                   edgecolors="none", zorder=3)
        ax.text(it["dt_h"], 0.02, f"n={it['vals'].size}\ncut {it['cut_m'] * 100:.0f} cm",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7, color="0.3")


def compare_box(runs, outdir, collapse_time, reference="Beehive_0", break_gap_h=12.0,
                trim_outliers=True, iqr_whis=1.5):
    """Collapse amplitude per video as a box (+ jittered raw markers), placed on a
    time axis relative to the slope collapse. The far-back reference gets its own
    panel via a broken x-axis when the largest time gap exceeds ``break_gap_h``."""
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(0)

    items = []
    for r in runs:
        if r["dt_h"] is None:
            print(f"skip {r['name']}: no date/collapse-time for the time axis")
            continue
        _, amp, _ = collapse_sample(r["df"], r["critical_cut_m"], r["usable_m"],
                                    trim_outliers=trim_outliers, whis=iqr_whis)
        if amp.size == 0:
            print(f"skip {r['name']}: no collapsed markers past the cut")
            continue
        items.append(dict(name=r["name"], dt_h=r["dt_h"],
                          cut_m=r["critical_cut_m"], vals=amp))
    if not items:
        raise SystemExit("no usable videos for the box plot")
    items.sort(key=lambda d: d["dt_h"])

    # Split on the largest time gap so a far-back reference gets its own panel.
    t = [it["dt_h"] for it in items]
    gaps = np.diff(t)
    gi = int(np.argmax(gaps)) if gaps.size else -1
    do_break = gaps.size > 0 and gaps[gi] > break_gap_h
    left = items[:gi + 1] if do_break else []
    right = items[gi + 1:] if do_break else items

    def _limits(group, pad_frac=0.6, min_pad=0.5):
        xs = [it["dt_h"] for it in group]
        lo, hi = min(xs), max(xs)
        span = (hi - lo) or 1.0
        pad = max(span * pad_frac, min_pad)
        return lo - pad, hi + pad

    def _set_widths(group, lo, hi):
        for it in group:
            it["width"] = 0.06 * (hi - lo)

    xlabel = f"time relative to slope collapse (h)  —  0 = {collapse_time}"
    if do_break:
        fig, (axL, axR) = plt.subplots(
            1, 2, sharey=True, figsize=(10, 5.5),
            gridspec_kw={"width_ratios": [1, 2.6], "wspace": 0.06})
        lL = _limits(left, pad_frac=1.0)
        lR = _limits(right)
        _set_widths(left, *lL)
        _set_widths(right, *lR)
        _draw_boxes(axL, left, reference, rng)
        _draw_boxes(axR, right, reference, rng)
        axL.set_xlim(*lL)
        axR.set_xlim(*lR)
        axL.spines["right"].set_visible(False)
        axR.spines["left"].set_visible(False)
        axR.tick_params(left=False)
        d = 0.012
        kw = dict(transform=axL.transAxes, color="k", clip_on=False, lw=1)
        axL.plot((1 - d, 1 + d), (-d, d), **kw)
        axL.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
        kw.update(transform=axR.transAxes)
        dr = d * 2.6
        axR.plot((-dr, dr), (-d, d), **kw)
        axR.plot((-dr, dr), (1 - d, 1 + d), **kw)
        for ax in (axL, axR):
            ax.axvline(0, color="0.6", ls="--", lw=1, zorder=0)
            ax.grid(axis="y", alpha=0.3)
        axL.set_ylabel("collapse amplitude (mm)")
        fig.supxlabel(xlabel, y=0.02)
        ref_ax = axR
    else:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        lR = _limits(right)
        _set_widths(right, *lR)
        _draw_boxes(ax, right, reference, rng)
        ax.set_xlim(*lR)
        ax.axvline(0, color="0.6", ls="--", lw=1, zorder=0)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("collapse amplitude (mm)")
        ax.set_xlabel(xlabel)
        ref_ax = ax

    has_ref = any(it["name"] == reference for it in items)
    handles = []
    if has_ref:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=REF_FILL,
                       edgecolor="black", label="pre-slope reference"))
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=POST_FILL,
                   edgecolor="black", label="post-collapse PST"))
    handles.append(plt.Line2D([0], [0], marker="o", ls="none", color="0.2",
                   alpha=0.5, ms=5, label="individual markers"))
    ref_ax.legend(handles=handles, fontsize=8, loc="upper right")
    fig.suptitle("PST collapse amplitude across videos, over time", y=0.97)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.13)
    out = os.path.join(outdir, "collapse_comparison_box.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description="PST cross-video comparison plots.")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="Two-panel collapse-profile comparison + stats CSV.")
    c.add_argument("dirs", nargs="*", help="Result directories (each from pst_analysis).")
    c.add_argument("--base", default="pst_results",
                   help="Compare every video subdirectory under this base results dir.")
    c.add_argument("--meta", default="data/metadata.json",
                   help="Metadata registry (for usable_extent_cm and dates).")
    c.add_argument("--out", default=None,
                   help="Output dir (default: <base>/_comparison).")
    c.add_argument("--bin-m", type=float, default=0.10, help="Distance bin width (m).")
    c.add_argument("--near-window-m", type=float, default=0.15,
                   help="Distance (m) from the cut used as the 'near-cut' reference.")
    c.add_argument("--color-by-time", action="store_true",
                   help="Color each line by its time vs --collapse-time instead of tab10.")
    c.add_argument("--collapse-time", default="2026-03-17T09:40",
                   help="Slope-collapse time (ISO) used when --color-by-time is set.")

    b = sub.add_parser("box", help="Collapse-amplitude box plot on a broken time axis.")
    b.add_argument("dirs", nargs="*", help="Result directories (each from pst_analysis).")
    b.add_argument("--base", "--results-dir", dest="base", default="pst_results",
                   help="Box every video subdirectory under this base results dir.")
    b.add_argument("--meta", default="data/metadata.json",
                   help="Metadata registry (per-video date, usable_extent_cm).")
    b.add_argument("--out", default=None,
                   help="Output dir (default: <base>/_comparison).")
    b.add_argument("--collapse-time", default="2026-03-17T09:40",
                   help="Slope-collapse time (ISO); x=0 and Δt are measured from it.")
    b.add_argument("--reference", default="Beehive_0",
                   help="Video drawn in darker gray as the pre-event reference.")
    b.add_argument("--break-gap-h", type=float, default=12.0,
                   help="Break the x-axis when the largest gap between videos exceeds this.")
    b.add_argument("--keep-outliers", action="store_true",
                   help="Keep Tukey outliers (by default dropped from box and scatter).")
    b.add_argument("--iqr-whis", type=float, default=1.5,
                   help="Tukey whisker multiplier for outlier removal.")
    return p


def main():
    args = build_parser().parse_args()
    out = args.out or (os.path.join(args.base, "_comparison") if args.base else "pst_comparison")
    if args.cmd == "compare":
        ct = args.collapse_time if args.color_by_time else None
        runs = load_runs(args.dirs, args.base, args.meta, ct)
        if not runs:
            raise SystemExit("No PST result directories found (need *_summary.json + *_markers.csv).")
        compare_profiles(runs, out, bin_m=args.bin_m, near_window_m=args.near_window_m,
                         color_by_time=args.color_by_time)
    elif args.cmd == "box":
        runs = load_runs(args.dirs, args.base, args.meta, args.collapse_time)
        if not runs:
            raise SystemExit("No PST result directories found (need *_summary.json + *_markers.csv).")
        compare_box(runs, out, collapse_time=args.collapse_time, reference=args.reference,
                    break_gap_h=args.break_gap_h, trim_outliers=not args.keep_outliers,
                    iqr_whis=args.iqr_whis)


if __name__ == "__main__":
    main()
