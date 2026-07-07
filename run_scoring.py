#!/usr/bin/env python3
"""
Command-line interface for auto-SR.

Usage
-----

Single participant (story and recall can be given as text or as paths to
.txt files):

    python run_scoring.py single \\
        --story ../examples/example_story.txt \\
        --recall ../examples/example_recall.txt \\
        --id P01 \\
        --outdir results/

Batch mode (score many participants against one story from a CSV):

    python run_scoring.py batch \\
        --story ../examples/example_story.txt \\
        --recall-csv ../examples/example_recalls.csv \\
        --outdir results/

The recall CSV must have an "id" column and a "recall_text" column, and
may optionally have a "group" column (e.g., "HA" / "PWA") to enable a
group-level serial-position plot.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auto_sr import LaBSEEmbedder, score_retelling
from auto_sr.similarity import compute_ors_oors
from auto_sr.visualize import plot_similarity_matrix, plot_serial_position


def _read_text_or_path(value: str) -> str:
    """Treat `value` as a file path if it exists on disk, else as raw text."""
    if os.path.isfile(value):
        with open(value, "r", encoding="utf-8") as f:
            return f.read().strip()
    return value


def run_single(args):
    story_text = _read_text_or_path(args.story)
    recall_text = _read_text_or_path(args.recall)

    os.makedirs(args.outdir, exist_ok=True)

    embedder = LaBSEEmbedder()
    result = score_retelling(
        story_text,
        recall_text,
        num_segments=args.num_segments,
        overlap_ratio=args.overlap_ratio,
        embedder=embedder,
        plot=True,
        plot_save_path=os.path.join(args.outdir, f"{args.id}_similarity_matrix.png"),
        plot_title=f"Semantic Similarity Matrix: {args.id}",
    )

    np.save(os.path.join(args.outdir, f"{args.id}_z_matrix.npy"), result["z_matrix"])
    pd.DataFrame(result["z_matrix"]).to_csv(
        os.path.join(args.outdir, f"{args.id}_z_matrix.csv"), index=False
    )

    summary = pd.DataFrame([{"id": args.id, "ORS": result["ORS"], "OORS": result["OORS"]}])
    summary.to_csv(os.path.join(args.outdir, f"{args.id}_scores.csv"), index=False)

    print(f"ID: {args.id}")
    print(f"  ORS  = {result['ORS']:.4f}")
    print(f"  OORS = {result['OORS']:.4f}")
    print(f"Saved outputs to: {args.outdir}")


def run_batch(args):
    story_text = _read_text_or_path(args.story)
    df = pd.read_csv(args.recall_csv)

    required_cols = {"id", "recall_text"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"--recall-csv must contain columns {required_cols}, got {list(df.columns)}")

    has_group = "group" in df.columns

    os.makedirs(args.outdir, exist_ok=True)
    matrix_dir = os.path.join(args.outdir, "similarity_matrices")
    os.makedirs(matrix_dir, exist_ok=True)

    embedder = LaBSEEmbedder()
    records = []
    diagonals = {}
    groups = {} if has_group else None

    for _, row in df.iterrows():
        pid = str(row["id"])
        recall_text = str(row["recall_text"])

        result = score_retelling(
            story_text,
            recall_text,
            num_segments=args.num_segments,
            overlap_ratio=args.overlap_ratio,
            embedder=embedder,
            plot=True,
            plot_save_path=os.path.join(matrix_dir, f"{pid}_similarity_matrix.png"),
            plot_title=f"Semantic Similarity Matrix: {pid}",
        )

        np.save(os.path.join(matrix_dir, f"{pid}_z_matrix.npy"), result["z_matrix"])

        record = {"id": pid, "ORS": result["ORS"], "OORS": result["OORS"]}
        if has_group:
            record["group"] = row["group"]
            groups[pid] = row["group"]
        records.append(record)
        diagonals[pid] = np.diag(result["z_matrix"])

        print(f"[{pid}] ORS={result['ORS']:.4f}  OORS={result['OORS']:.4f}")

    summary = pd.DataFrame(records)
    summary_path = os.path.join(args.outdir, "summary_scores.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary scores to: {summary_path}")

    serial_plot_path = os.path.join(args.outdir, "serial_position_plot.png")
    plot_serial_position(
        diagonals,
        groups=groups,
        title="Recall Accuracy Across Story Segments",
        save_path=serial_plot_path,
    )
    print(f"Saved serial-position plot to: {serial_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="auto-SR: Automated Scoring of Story Retelling")
    parser.add_argument("--num-segments", dest="num_segments", type=int, default=10,
                         help="Number of segments for story and recall (default: 10, as in the published study)")
    parser.add_argument("--overlap-ratio", dest="overlap_ratio", type=float, default=0.2,
                         help="Overlap ratio between adjacent segments (default: 0.2, as in the published study)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    single_parser = subparsers.add_parser("single", help="Score one participant")
    single_parser.add_argument("--story", required=True,
                                help="Write the path to your original stimuli story (.txt), or paste the story text directly")
    single_parser.add_argument("--recall", required=True,
                                help="Write the path to your participant's recall transcript (.txt), or paste the text directly")
    single_parser.add_argument("--id", default="participant", help="Participant ID (used for output filenames)")
    single_parser.add_argument("--outdir", default="results", help="Output directory")
    single_parser.set_defaults(func=run_single)

    batch_parser = subparsers.add_parser("batch", help="Score many participants from a CSV")
    batch_parser.add_argument("--story", required=True,
                               help="Write the path to your original stimuli story (.txt), or paste the story text directly")
    batch_parser.add_argument("--recall-csv", dest="recall_csv", required=True,
                               help="CSV with columns: id, recall_text, [group]")
    batch_parser.add_argument("--outdir", default="results", help="Output directory")
    batch_parser.set_defaults(func=run_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
