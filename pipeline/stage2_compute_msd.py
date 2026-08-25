#!/usr/bin/env python3
"""
compute_msd.py — Compute per-cell MSD values from a Trackmate spots CSV.

For each track in the input CSV, writes one output row containing three MSD
variants at a fixed lag (default 3 frames = 9 min at 180 s/frame):

  msd_overlap_partial   TAMSD(lag) averaged over every overlapping (j, j+lag)
                        pair in the track whose endpoints both exist. Computed
                        for any track with ≥1 valid lag-pair (partial tracks
                        included).

  msd_overlap_complete  Same calculation, but only for tracks meeting the
                        "complete" criterion: a run of ≥ --min-consec
                        consecutive frames (no gaps) inside the track.

  msd_single_complete   Squared displacement of a single window from the start
                        of the complete track: (r_{f0+lag} − r_{f0})². Complete
                        tracks only.

--um-per-pixel is REQUIRED and has no default: the two imaging channels
have different pixel sizes, so no single default is correct.
  phase contrast   --um-per-pixel 0.68626
  GFP              --um-per-pixel 1.02939
  raw pixels       --um-per-pixel 1.0
The Makefile supplies this automatically from PROFILE=phase|gfp.

Usage:
    python stage2_compute_msd.py INPUT.csv [-o OUTPUT.csv] [--lag 3]
                                 [--min-consec 18] [--um-per-pixel 1.0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_trackmate(path: Path) -> pd.DataFrame:
    """Load a Trackmate spots CSV. Rows 2-4 after the header are unit/label rows."""
    df = pd.read_csv(path, header=0, skiprows=[1, 2, 3], low_memory=False)
    for col in ("TRACK_ID", "FRAME", "POSITION_X", "POSITION_Y"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["TRACK_ID", "FRAME", "POSITION_X", "POSITION_Y"])
    df["TRACK_ID"] = df["TRACK_ID"].astype(int)
    df["FRAME"] = df["FRAME"].astype(int)
    df = df[df["TRACK_ID"] >= 0]
    return df


def longest_consecutive_run(frames: np.ndarray):
    """Return (length, start_index_in_array) of the longest consecutive integer run."""
    if frames.size == 0:
        return 0, 0
    diffs = np.diff(frames)
    splits = np.where(diffs != 1)[0]
    starts = np.concatenate(([0], splits + 1))
    ends = np.concatenate((splits, [frames.size - 1]))
    lengths = ends - starts + 1
    best = int(np.argmax(lengths))
    return int(lengths[best]), int(starts[best])


def per_cell_msd(track: pd.DataFrame, lag: int, scale: float):
    track = track.sort_values("FRAME")
    frames = track["FRAME"].to_numpy()
    x = track["POSITION_X"].to_numpy() * scale
    y = track["POSITION_Y"].to_numpy() * scale

    # Overlapping lag-k windows: only pair true (j, j+lag) frame neighbours.
    frame_to_idx = {int(f): i for i, f in enumerate(frames)}
    sq = []
    for j, fj in enumerate(frames):
        i = frame_to_idx.get(int(fj) + lag)
        if i is not None:
            sq.append((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
    msd_overlap = float(np.mean(sq)) if sq else np.nan
    n_pairs = len(sq)

    consec_len, run_start_idx = longest_consecutive_run(frames)

    msd_single = np.nan
    if consec_len >= lag + 1:
        s = run_start_idx
        e = s + lag
        msd_single = float((x[e] - x[s]) ** 2 + (y[e] - y[s]) ** 2)

    return msd_overlap, n_pairs, consec_len, msd_single


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="Trackmate spots CSV")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output per-cell CSV (default: <input_stem>_msd.csv beside input)")
    ap.add_argument("--lag", type=int, default=3,
                    help="Lag in frames (default 3 = 9 min at 180 s/frame)")
    ap.add_argument("--min-consec", type=int, default=18,
                    help="Min length of a consecutive-frame run to count as complete (default 18)")
    ap.add_argument("--um-per-pixel", type=float, required=True,
                    help="Pixel size in µm/px. REQUIRED -- no default, because "
                         "the channels differ: phase contrast 0.68626, "
                         "GFP 1.02939. Pass 1.0 to keep outputs in px².")
    args = ap.parse_args()

    df = load_trackmate(args.input)
    if df.empty:
        sys.exit(f"No usable rows in {args.input}")

    rows = []
    for tid, group in df.groupby("TRACK_ID", sort=True):
        msd_overlap, n_pairs, consec, msd_single = per_cell_msd(
            group, args.lag, args.um_per_pixel
        )
        is_complete = consec >= args.min_consec
        rows.append({
            "source_file": args.input.name,
            "track_id": int(tid),
            "n_spots": int(len(group)),
            "longest_consec_frames": consec,
            "is_complete": bool(is_complete),
            "n_lag_pairs": n_pairs,
            "msd_overlap_partial": msd_overlap,
            "msd_overlap_complete": msd_overlap if is_complete else np.nan,
            "msd_single_complete": msd_single if is_complete else np.nan,
        })

    out_df = pd.DataFrame(rows)
    out_path = args.output or args.input.with_name(args.input.stem + "_msd.csv")
    out_df.to_csv(out_path, index=False)
    print(f"{args.input.name}: wrote {len(out_df)} cells "
          f"({int(out_df['is_complete'].sum())} complete) → {out_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
