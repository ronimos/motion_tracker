# -*- coding: utf-8 -*-
"""
Cross-video collapse comparison
===============================

Overlays the collapse profile of several PST videos in one plot to compare how
the slab collapse - and the saw-cut length needed to trigger it - change over
time relative to a reference event (e.g. a natural slope/weak-layer collapse).

For each video it reads the per-marker CSV written by ``pst_analysis.py``
(``<results>/<video>/<video>_markers.csv``) and the metadata registry
(``saw_cut.json``: per-video ``saw_cut_cm`` and ``date``), then draws one line:

- x-axis = distance from the column edge (m). Each line **starts at that video's
  saw-cut length**, so where a line begins shows its critical cut and the line's
  height shows the collapse - both in one axes.
- y-axis = collapse magnitude (mm), a smoothed binned median of *all* valid
  markers past the cut (collapsed and not), so the line falls to ~0 where the
  crack **arrested** and stays there - that arrest is kept on purpose.
- color = time relative to ``--collapse-time`` (hours/days). A chosen reference
  video (``--reference``) is drawn in black.

Usage:
    python compare_collapse.py                    # defaults: pst_results, saw_cut.json
    python compare_collapse.py --reference Beehive_0 --collapse-time 2026-03-17T08:30

@author: Ron Simenhois
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def _smoothed_profile(x, y, collapsed, x0, x1, bin_m, smooth_win):
    """Collapse profile over [x0, x1] in bin_m steps, then a centered moving
    average of `smooth_win` bins. Returns (bin centers, smoothed values) for the
    populated bins.

    Per bin the value is the median collapse of the *collapsed* markers there - so
    the magnitude reflects the slab and isn't diluted to zero by background/off-
    slab markers that never moved. A bin that has markers but none collapsed gives
    0, so the line still falls to zero where the crack **arrested**; a bin with no
    markers at all is left empty (a genuine tracking gap)."""
    if x.size < 2 or x1 <= x0:
        return np.array([]), np.array([])
    nb = max(2, int(np.ceil((x1 - x0) / bin_m)))
    edges = np.linspace(x0, x1, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    vals = np.full(nb, np.nan)
    bid = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    for b in range(nb):
        sel = bid == b
        if not sel.any():
            continue                       # no markers -> leave a gap
        cc = collapsed[sel]
        vals[b] = np.median(y[sel][cc]) if cc.any() else 0.0   # 0 = arrest, not gap
    good = np.isfinite(vals)
    centers, vals = centers[good], vals[good]
    if smooth_win > 1 and vals.size >= smooth_win:
        k = np.ones(smooth_win) / smooth_win
        vals = np.convolve(vals, k, mode="same")
    return centers, vals


def _fmt_dt(hours):
    """Human label for a time offset: days when large, else hours."""
    if abs(hours) >= 48:
        return f"{hours / 24:+.1f} d"
    return f"{hours:+.1f} h"


def run(args):
    with open(args.meta) as fh:
        reg = json.load(fh)
    t0 = datetime.fromisoformat(args.collapse_time)

    # Collect each video's profile + metadata, ordered by time.
    items = []
    for name, m in reg.items():
        csv = os.path.join(args.results_dir, name, f"{name}_markers.csv")
        if not os.path.exists(csv):
            print(f"skip {name}: no {csv}")
            continue
        df = pd.read_csv(csv)
        saw_m = float(m.get("saw_cut_cm", 0)) / 100.0
        col_m = float(m.get("column_length_cm", 0)) / 100.0 or df.along_column_m.max()
        dt_h = (datetime.fromisoformat(m["date"]) - t0).total_seconds() / 3600.0
        keep = df.in_column & ~df.bad_track & (df.along_column_m >= saw_m) \
            & (df.along_column_m <= col_m)
        cx, cy = _smoothed_profile(df.along_column_m[keep].to_numpy(),
                                   df.collapse_mm[keep].to_numpy(),
                                   df.collapsed[keep].to_numpy().astype(bool),
                                   saw_m, col_m, args.bin_m, args.smooth_win)
        items.append(dict(name=name, saw_m=saw_m, dt_h=dt_h, x=cx, y=cy))
    items.sort(key=lambda d: d["dt_h"])

    # Color post-reference videos by Δt; the reference is black.
    others = [it for it in items if it["name"] != args.reference]
    dts = [it["dt_h"] for it in others] or [0, 1]
    norm = Normalize(vmin=min(dts), vmax=max(dts))
    cmap = plt.get_cmap(args.cmap)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for it in items:
        if it["x"].size == 0:
            continue
        is_ref = it["name"] == args.reference
        color = "black" if is_ref else cmap(norm(it["dt_h"]))
        label = (f"{it['name']}  {_fmt_dt(it['dt_h'])}  cut {it['saw_m'] * 100:.0f} cm"
                 + ("  (reference)" if is_ref else ""))
        ax.plot(it["x"], it["y"], "-", color=color, lw=2.4 if is_ref else 2.0,
                marker="o", ms=3, label=label)

    if others:
        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        fig.colorbar(sm, ax=ax, label=f"hours after {args.collapse_time} (slope collapse)")
    ax.set_xlabel("distance from column edge (m)  —  line starts at the saw cut")
    ax.set_ylabel("collapse magnitude (mm)")
    ax.set_title("PST collapse profile vs saw cut, over time")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"Wrote {args.out}")


def build_parser():
    p = argparse.ArgumentParser(description="Overlay PST collapse profiles across videos over time.")
    p.add_argument("--results-dir", default="pst_results",
                   help="Base dir holding <video>/<video>_markers.csv.")
    p.add_argument("--meta", default="data/saw_cut.json",
                   help="Metadata registry (per-video saw_cut_cm, column_length_cm, date).")
    p.add_argument("--collapse-time", default="2026-03-17T08:30",
                   help="Reference event time (ISO) the color/Δt is measured from.")
    p.add_argument("--reference", default="Beehive_0",
                   help="Video drawn in black as the pre-event reference.")
    p.add_argument("--bin-m", type=float, default=0.10, help="Distance bin width (m).")
    p.add_argument("--smooth-win", type=int, default=3,
                   help="Moving-average window in bins (1 = no smoothing).")
    p.add_argument("--cmap", default="plasma", help="Colormap for the Δt coloring.")
    p.add_argument("--out", default="pst_results/collapse_comparison.png",
                   help="Output image path.")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
