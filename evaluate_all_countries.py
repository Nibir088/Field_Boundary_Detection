#!/usr/bin/env python3
"""Evaluate an FTW checkpoint on every downloaded country's test split."""

import argparse
import gc
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from ftw_tools.training.eval import test


def scratch_path(*parts: str) -> Path:
    """Build a path below the current user's Rivanna scratch directory."""
    return Path("/sfs/weka/scratch") / os.environ.get("USER", "unknown") / Path(*parts)


def parse_args() -> argparse.Namespace:
    run_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a three-class FTW model on every downloaded country's "
            "predefined test split."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=scratch_path("ftw_data", "ftw"),
        help="Directory containing the FTW country folders.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=scratch_path("ftw_models", "FTW_PRUE_EFNET_B5.ckpt"),
        help="Path to the FTW model checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=scratch_path("ftw_results", f"job_{run_id}"),
        help="Directory in which metric CSV files will be written.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index; use -1 for CPU.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Calculate bootstrap confidence intervals (considerably slower).",
    )
    return parser.parse_args()


def discover_test_countries(data_dir: Path) -> list[tuple[str, int]]:
    """Return downloaded countries having at least one predefined test sample."""
    countries = []

    for parquet_file in sorted(data_dir.glob("*/chips_*.parquet")):
        table = pd.read_parquet(parquet_file, columns=["split"])
        test_count = int((table["split"] == "test").sum())
        if test_count:
            countries.append((parquet_file.parent.name, test_count))

    return countries


def evaluate(
    model: Path,
    data_dir: Path,
    countries: list[str],
    output_file: Path,
    gpu: int,
    num_workers: int,
    bootstrap: bool,
) -> None:
    """Run the repository's standard two-class evaluation for a v3 model."""
    test(
        model_path=str(model),
        dir=str(data_dir),
        gpu=gpu,
        countries=countries,
        iou_threshold=0.5,
        out=str(output_file),
        model_predicts_3_classes=True,
        test_on_3_classes=False,
        temporal_options="stacked",
        use_val_set=False,
        swap_order=False,
        norm_constant=None,
        resize_factor=1,
        num_workers=num_workers,
        bootstrap=bootstrap,
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    model = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not data_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {data_dir}")
    if not model.is_file():
        raise SystemExit(f"Model checkpoint not found: {model}")
    if args.gpu >= 0 and not torch.cuda.is_available():
        raise SystemExit("A GPU was requested, but PyTorch cannot access CUDA.")

    country_counts = discover_test_countries(data_dir)
    if not country_counts:
        raise SystemExit(f"No countries with test samples found in: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Discovered test data:")
    for country, count in country_counts:
        print(f"  {country}: {count} samples")
    print(f"  total: {sum(count for _, count in country_counts)} samples")

    country_names = [country for country, _ in country_counts]

    for index, country in enumerate(country_names, start=1):
        print(f"\n[{index}/{len(country_names)}] Evaluating {country}")
        evaluate(
            model=model,
            data_dir=data_dir,
            countries=[country],
            output_file=output_dir / f"{country}_metrics.csv",
            gpu=args.gpu,
            num_workers=args.num_workers,
            bootstrap=args.bootstrap,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nEvaluating all {len(country_names)} countries together")
    evaluate(
        model=model,
        data_dir=data_dir,
        countries=country_names,
        output_file=output_dir / "all_countries_metrics.csv",
        gpu=args.gpu,
        num_workers=args.num_workers,
        bootstrap=args.bootstrap,
    )

    print(f"\nEvaluation complete. Metrics saved in: {output_dir}")


if __name__ == "__main__":
    main()
