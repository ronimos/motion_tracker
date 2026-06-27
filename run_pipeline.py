# -*- coding: utf-8 -*-
"""
PST pipeline executor
=====================

Runs the whole PST workflow start to finish from the existing config + metadata:

1. For every video in ``metadata.json`` (that has a saved per-video config), run
   ``pst_analysis.py`` with ``--reuse`` (reuse the saved ROI/scale/static, no
   interactive setup) and ``--stabilize`` (camera-motion compensation).
2. Build the cross-video figures with ``pst_plots.py``: the collapse-amplitude
   ``box`` plot (outliers removed) and the ``compare`` profile + stats CSV.

Per-video stabilization can be overridden from the metadata: add ``"stabilize":
false`` to a video's entry to analyze it without ``--stabilize`` (e.g. a clip the
stabilizer inverts), regardless of the global default.

Usage:
    python run_pipeline.py                       # analyze all + build comparisons
    python run_pipeline.py --only Beehive_3      # just one (or a few) videos
    python run_pipeline.py --skip-analysis       # only rebuild the comparison plots
    python run_pipeline.py --no-stabilize        # global default off (metadata still wins)

@author: Ron Simenhois
"""

import os
import sys
import json
import argparse
import subprocess


def _run(cmd):
    """Run a subprocess, streaming its output. Returns True on success."""
    print("\n$ " + " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def analyze_one(py, name, meta_entry, args, stabilize):
    """Run pst_analysis.py for one video with --reuse and (maybe) --stabilize."""
    video = os.path.join(args.data_dir, f"{name}.{args.ext}")
    config = os.path.join(args.results_dir, name, f"{name}_config.json")
    if not os.path.exists(video):
        print(f"skip {name}: no video at {video}")
        return None
    if not os.path.exists(config):
        print(f"skip {name}: no saved config at {config} (run the interactive setup "
              f"once, or drop --reuse)")
        return None
    cmd = [py, "pst_analysis.py", "-v", video, "--meta", args.meta,
           "--out", args.results_dir, "--reuse"]
    if stabilize:
        cmd.append("--stabilize")
    if args.no_video:
        cmd.append("--no-video")
    return _run(cmd)


def main():
    ap = argparse.ArgumentParser(description="Run the full PST analysis + comparison pipeline.")
    ap.add_argument("--meta", default="data/metadata.json",
                    help="Per-video metadata registry (drives which videos run).")
    ap.add_argument("--data-dir", default="data", help="Directory holding the source videos.")
    ap.add_argument("--results-dir", default="pst_results",
                    help="Base output dir (<results-dir>/<video>/ per video).")
    ap.add_argument("--ext", default="mp4", help="Source video extension.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only analyze these video names (default: all in metadata).")
    ap.add_argument("--no-stabilize", action="store_true",
                    help="Global default: analyze without --stabilize (a per-video "
                         "metadata 'stabilize' value still overrides this).")
    ap.add_argument("--no-video", action="store_true",
                    help="Skip writing the annotated event video (faster).")
    ap.add_argument("--skip-analysis", action="store_true",
                    help="Skip per-video analysis; only rebuild the comparison plots.")
    ap.add_argument("--skip-compare", action="store_true",
                    help="Skip the cross-video comparison plots.")
    ap.add_argument("--collapse-time", default="2026-03-17T09:40",
                    help="Slope-collapse time (ISO) for the box plot time axis.")
    ap.add_argument("--reference", default="Beehive_0",
                    help="Pre-event reference video for the box plot.")
    args = ap.parse_args()

    py = sys.executable
    with open(args.meta) as fh:
        reg = json.load(fh)
    names = args.only if args.only else list(reg)
    default_stab = not args.no_stabilize

    results = {}
    if not args.skip_analysis:
        print(f"=== Analyzing {len(names)} video(s) ===")
        for name in names:
            entry = reg.get(name, {})
            stabilize = bool(entry.get("stabilize", default_stab))
            results[name] = analyze_one(py, name, entry, args, stabilize)

    if not args.skip_compare:
        print("\n=== Building cross-video comparisons ===")
        common = ["--base", args.results_dir, "--meta", args.meta]
        _run([py, "pst_plots.py", "box", *common,
              "--collapse-time", args.collapse_time, "--reference", args.reference])
        _run([py, "pst_plots.py", "compare", *common])

    # Summary of the per-video analysis pass.
    done = [n for n, ok in results.items() if ok]
    failed = [n for n, ok in results.items() if ok is False]
    skipped = [n for n, ok in results.items() if ok is None]
    if results:
        print(f"\n=== Pipeline done: {len(done)} ok, {len(failed)} failed, "
              f"{len(skipped)} skipped ===")
        if failed:
            print("  failed:  " + ", ".join(failed))
        if skipped:
            print("  skipped: " + ", ".join(skipped))


if __name__ == "__main__":
    main()
