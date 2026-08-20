#!/usr/bin/env python3
"""Count FTW train, validation, and test samples by country."""

import argparse
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    default_root = (
        Path("/scratch")
        / os.environ.get("USER", "unknown")
        / "ftw_data"
        / "ftw"
    )
    parser = argparse.ArgumentParser(
        description="Summarize the predefined FTW dataset splits by country."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Directory containing the FTW country folders (default: {default_root})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist or is not a directory: {root}")

    parquet_files = sorted(root.glob("*/chips_*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No chips_*.parquet files found below: {root}")

    rows = []
    for parquet_file in parquet_files:
        country = parquet_file.parent.name
        table = pd.read_parquet(parquet_file, columns=["split"])
        counts = table["split"].value_counts()

        rows.append(
            {
                "country": country,
                "train": int(counts.get("train", 0)),
                "val": int(counts.get("val", 0)),
                "test": int(counts.get("test", 0)),
                "total": len(table),
            }
        )

    summary = pd.DataFrame(rows)
    print(f"Dataset root: {root}\n")
    print(summary.to_string(index=False))
    print("\nTotals")
    print(summary[["train", "val", "test", "total"]].sum().to_string())


if __name__ == "__main__":
    main()
