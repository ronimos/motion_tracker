# -*- coding: utf-8 -*-
"""
Compare PST collapse profiles across videos
===========================================

Reads the per-video outputs written by ``pst_analysis.py``
(``<out>/<video>/<video>_markers.csv`` and ``..._summary.json``) and compares the
collapse magnitude along the column between videos, aligned to each video's own
saw-cut length.

Two panels:
1.  Collapse magnitude (mm) vs distance from the saw cut, one curve per video -
    the absolute profiles, all starting at 0 = each video's critical cut length.
2.  The same profiles normalized to the near-cut collapse (each starts ~1.0) -
    shows the *relative* change in collapse as the crack runs from the cut
    (rising = collapse grows with propagation, falling = damping toward arrest),
    so the change is comparable between videos regardless of absolute magnitude.

Also writes a per-video stats table: critical cut, near/far collapse, the collapse
slope (mm/m), and the far/near ratio.

Usage:
    # compare every video under a results directory
    python pst_compare.py --base pst_results

    # or name specific run directories
    python pst_compare.py pst_results/PST_01 pst_results/PST_02 --out pst_results/_compare

@author: Ron Simenhois
"""

import os
import glob
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _r(v, n=3):
    try:
        if v is None or not np.isfinite(v):
            return None
    except TypeError:
        return v
    return round(float(v), n)


def find_runs(paths, base):
    """Locate result directories (those holding *_summary.json and *_markers.csv)."""
    candidates = list(paths)
    if base:
        candidates += sorted(os.path.join(base, d) for d in os.listdir(base)
                             if os.path.isdir(os.path.join(base, d)))
    runs = []
    seen = set()
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


def load_run(name, summary_path, markers_path):
    with open(summary_path) as fh:
        summary = json.load(fh)
    df = pd.read_csv(markers_path)
    cut = summary.get("critical_cut_length_m") or 0.0
    keep = df["collapsed"].to_numpy(dtype=bool, copy=True)
    if "in_column" in df.columns:
        keep &= df["in_column"].to_numpy(dtype=bool)
    if "bad_track" in df.columns:
        keep &= ~df["bad_track"].to_numpy(dtype=bool)
    df = df[keep].copy()
    df["dist_from_cut_m"] = df["along_column_m"] - cut
    df = df[df["dist_from_cut_m"] >= 0]
    return {"name": name, "summary": summary, "cut": float(cut), "df": df}


def binned_profile(df, bin_m):
    """Median collapse per distance bin (smooths sparse markers into a curve)."""
    d = df["dist_from_cut_m"].to_numpy()
    c = df["collapse_mm"].to_numpy()
    if d.size == 0:
        return np.array([]), np.array([])
    edges = np.arange(0.0, d.max() + bin_m, bin_m)
    if edges.size < 2:
        return np.array([d.mean()]), np.array([np.median(c)])
    idx = np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)
    xs, ys = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if np.any(sel):
            xs.append(edges[b] + bin_m / 2)
            ys.append(float(np.median(c[sel])))
    return np.asarray(xs), np.asarray(ys)


def compare(runs, outdir, bin_m=0.1, near_window_m=0.15):
    os.makedirs(outdir, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    stats = []

    for k, r in enumerate(runs):
        color = cmap(k % 10)
        df = r["df"]
        xs, ys = binned_profile(df, bin_m)
        d = df["dist_from_cut_m"].to_numpy()
        c = df["collapse_mm"].to_numpy()

        # Panel 1: absolute collapse profile from the saw cut
        ax1.scatter(d, c, s=10, alpha=0.20, color=color)
        if xs.size:
            ax1.plot(xs, ys, "-o", color=color, lw=2, ms=4,
                     label=f'{r["name"]} (cut {r["cut"]:.2f} m, n={d.size})')

        # Trend slope (mm per m) and near/far collapse
        slope = np.polyfit(d, c, 1)[0] if (d.size >= 3 and np.ptp(d) > 0) else np.nan
        near = far = np.nan
        if xs.size:
            near_sel = xs <= near_window_m
            near = float(np.median(ys[near_sel])) if np.any(near_sel) else float(ys[0])
            far = float(ys[-1])
            # Panel 2: relative change, normalized to near-cut collapse
            if near and np.isfinite(near) and near > 0:
                ax2.plot(xs, ys / near, "-o", color=color, lw=2, ms=4, label=r["name"])

        stats.append({
            "video": r["name"],
            "critical_cut_m": _r(r["cut"]),
            "n_markers": int(d.size),
            "collapse_near_mm": _r(near, 2),
            "collapse_far_mm": _r(far, 2),
            "collapse_slope_mm_per_m": _r(slope, 2),
            "far_over_near": _r(far / near, 2) if near else None,
            "crack_speed_m_s": r["summary"].get("crack_speed_m_s"),
            "propagation_distance_m": r["summary"].get("propagation_distance_m"),
        })

    ax1.set_xlabel("distance from saw cut (m)")
    ax1.set_ylabel("collapse magnitude (mm)")
    ax1.set_title("Collapse profile from the saw cut, per video")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.axhline(1.0, color="gray", ls=":")
    ax2.set_xlabel("distance from saw cut (m)")
    ax2.set_ylabel("collapse / near-cut collapse")
    ax2.set_title("Relative change in collapse (normalized at the cut)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

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


def build_parser():
    p = argparse.ArgumentParser(description="Compare PST collapse profiles across videos.")
    p.add_argument("dirs", nargs="*", help="Result directories to compare (each from pst_analysis).")
    p.add_argument("--base", default=None,
                   help="Compare every video subdirectory under this base results directory.")
    p.add_argument("--out", default=None,
                   help="Output directory (default: <base>/_comparison or ./pst_comparison).")
    p.add_argument("--bin-m", type=float, default=0.10,
                   help="Distance bin width (m) for the smoothed profile.")
    p.add_argument("--near-window-m", type=float, default=0.15,
                   help="Distance (m) from the cut used as the 'near-cut' reference.")
    return p


def main():
    args = build_parser().parse_args()
    runs = find_runs(args.dirs, args.base)
    if not runs:
        raise SystemExit("No PST result directories found (need *_summary.json + *_markers.csv).")
    loaded = [load_run(*r) for r in runs]
    out = args.out or (os.path.join(args.base, "_comparison") if args.base else "pst_comparison")
    compare(loaded, out, bin_m=args.bin_m, near_window_m=args.near_window_m)


if __name__ == "__main__":
    main()
