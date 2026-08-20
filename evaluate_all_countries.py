#!/usr/bin/env python3
"""Evaluate a 3-class FTW checkpoint on every downloaded test split.

The same native predictions are scored in three ways:

1. Native 3-class: background, field interior, and field boundary.
2. Standard binary: field interior versus background plus boundary. This matches
   the historical ``ftw model test --model_predicts_3_classes`` behavior.
3. Field-extent 2-class: field interior plus boundary versus background.

Pixels with ground-truth value 3 (unknown/nodata) are ignored.
"""

import argparse
import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from scipy import ndimage
from torch.utils.data import DataLoader
from tqdm import tqdm

from ftw_tools.training.datasets import FTW
from ftw_tools.training.trainers import CustomSemanticSegmentationTask


CLASS_NAMES = ("background", "field_interior", "field_boundary")


def scratch_path(*parts: str) -> Path:
    """Build a path below the current user's Rivanna scratch directory."""
    return Path("/sfs/weka/scratch") / os.environ.get("USER", "unknown") / Path(*parts)


def parse_args() -> argparse.Namespace:
    run_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser = argparse.ArgumentParser(
        description="Evaluate all downloaded FTW countries with classwise metrics."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=scratch_path("ftw_data", "ftw"),
        help="Directory containing FTW country folders.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=scratch_path("ftw_models", "FTW_PRUE_EFNET_B5.ckpt"),
        help="Path to a native 3-class FTW checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=scratch_path("ftw_results", f"job_{run_id}"),
        help="Destination for metric and confusion-matrix CSV files.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index; use -1 for CPU.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


@dataclass
class EvaluationCounts:
    """Sufficient statistics for all requested evaluation views."""

    confusion_3class: torch.Tensor
    object_tp: int = 0
    object_fp: int = 0
    object_fn: int = 0
    matched_object_ious: list[float] = field(default_factory=list)
    ground_truth_objects: int = 0
    predicted_objects: int = 0
    samples: int = 0

    @classmethod
    def empty(cls) -> "EvaluationCounts":
        return cls(confusion_3class=torch.zeros((3, 3), dtype=torch.int64))

    def add(self, other: "EvaluationCounts") -> None:
        self.confusion_3class += other.confusion_3class
        self.object_tp += other.object_tp
        self.object_fp += other.object_fp
        self.object_fn += other.object_fn
        self.matched_object_ious.extend(other.matched_object_ious)
        self.ground_truth_objects += other.ground_truth_objects
        self.predicted_objects += other.predicted_objects
        self.samples += other.samples


def discover_test_countries(data_dir: Path) -> list[tuple[str, int]]:
    """Return downloaded countries having at least one predefined test sample."""
    countries = []
    for parquet_file in sorted(data_dir.glob("*/chips_*.parquet")):
        table = pd.read_parquet(parquet_file, columns=["split"])
        test_count = int((table["split"] == "test").sum())
        if test_count:
            countries.append((parquet_file.parent.name, test_count))
    return countries


def update_confusion(
    confusion: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor
) -> None:
    """Accumulate a 3x3 confusion matrix, ignoring unknown target pixels."""
    valid = (targets >= 0) & (targets < 3)
    encoded = targets[valid] * 3 + predictions[valid]
    confusion += torch.bincount(encoded.cpu(), minlength=9).reshape(3, 3)


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def metrics_from_confusion(
    confusion: torch.Tensor, class_names: tuple[str, ...]
) -> list[dict[str, int | float | str]]:
    """Calculate IoU, precision, recall, F1, and support for each class."""
    rows = []
    for index, class_name in enumerate(class_names):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        support = int(confusion[index, :].sum())

        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        iou = safe_divide(
            true_positive, true_positive + false_positive + false_negative
        )
        f1 = safe_divide(2 * precision * recall, precision + recall)

        rows.append(
            {
                "class": class_name,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support_pixels": support,
            }
        )
    return rows


def collapse_confusion(
    confusion: torch.Tensor, mapping: tuple[int, int, int]
) -> torch.Tensor:
    """Collapse the native confusion matrix into a two-class matrix."""
    result = torch.zeros((2, 2), dtype=torch.int64)
    for target_class in range(3):
        for predicted_class in range(3):
            result[mapping[target_class], mapping[predicted_class]] += confusion[
                target_class, predicted_class
            ]
    return result


def model_outputs(
    model: torch.nn.Module, model_type: str, images: torch.Tensor
) -> torch.Tensor:
    """Return native class predictions for an FTW model batch."""
    if model_type in {"fcsiamdiff", "fcsiamconc", "fcsiamavg"}:
        images = rearrange(images, "b (t c) h w -> b t c h w", t=2)
    with torch.inference_mode():
        return model(images).argmax(dim=1)


@dataclass
class ObjectMatchResult:
    """One-to-one connected-object matching results for one image."""

    true_positive: int
    false_positive: int
    false_negative: int
    matched_ious: list[float]
    ground_truth_count: int
    predicted_count: int


def match_objects(
    target: np.ndarray, prediction: np.ndarray, iou_threshold: float
) -> ObjectMatchResult:
    """Match 4-connected binary objects one-to-one by descending mask IoU."""
    connectivity = ndimage.generate_binary_structure(2, 1)
    target_labels, target_count = ndimage.label(target.astype(bool), connectivity)
    prediction_labels, prediction_count = ndimage.label(
        prediction.astype(bool), connectivity
    )

    if target_count == 0 or prediction_count == 0:
        return ObjectMatchResult(
            true_positive=0,
            false_positive=prediction_count,
            false_negative=target_count,
            matched_ious=[],
            ground_truth_count=target_count,
            predicted_count=prediction_count,
        )

    stride = prediction_count + 1
    intersections = np.bincount(
        (target_labels.ravel() * stride + prediction_labels.ravel()),
        minlength=(target_count + 1) * stride,
    ).reshape(target_count + 1, stride)
    target_areas = np.bincount(target_labels.ravel(), minlength=target_count + 1)
    prediction_areas = np.bincount(
        prediction_labels.ravel(), minlength=prediction_count + 1
    )

    candidate_target, candidate_prediction = np.nonzero(intersections[1:, 1:])
    candidate_target += 1
    candidate_prediction += 1
    candidate_intersections = intersections[
        candidate_target, candidate_prediction
    ].astype(float)
    candidate_unions = (
        target_areas[candidate_target]
        + prediction_areas[candidate_prediction]
        - candidate_intersections
    )
    candidate_ious = candidate_intersections / candidate_unions

    order = np.argsort(candidate_ious)[::-1]
    used_targets: set[int] = set()
    used_predictions: set[int] = set()
    matched_ious = []

    for candidate_index in order:
        iou = float(candidate_ious[candidate_index])
        if iou < iou_threshold:
            break
        target_id = int(candidate_target[candidate_index])
        prediction_id = int(candidate_prediction[candidate_index])
        if target_id in used_targets or prediction_id in used_predictions:
            continue
        used_targets.add(target_id)
        used_predictions.add(prediction_id)
        matched_ious.append(iou)

    true_positive = len(matched_ious)
    return ObjectMatchResult(
        true_positive=true_positive,
        false_positive=prediction_count - true_positive,
        false_negative=target_count - true_positive,
        matched_ious=matched_ious,
        ground_truth_count=target_count,
        predicted_count=prediction_count,
    )


def evaluate_country(
    country: str,
    data_dir: Path,
    model: torch.nn.Module,
    model_type: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    iou_threshold: float,
) -> EvaluationCounts:
    """Run native 3-class inference once for a country's test split."""
    dataset = FTW(
        root=str(data_dir),
        countries=[country],
        split="test",
        load_boundaries=True,
        temporal_options="stacked",
        verbose=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    counts = EvaluationCounts.empty()

    for batch in tqdm(dataloader, desc=country):
        images = batch["image"].to(device, non_blocking=True) / 3000.0
        targets = batch["mask"].to(device, non_blocking=True)
        predictions = model_outputs(model, model_type, images)

        update_confusion(counts.confusion_3class, predictions, targets)

        # Match the repository's historical binary/object evaluation: only
        # class 1 is field; class 0 and class 2 are treated as non-field.
        binary_predictions = (predictions == 1).to(torch.uint8).cpu().numpy()
        binary_targets = (targets == 1).to(torch.uint8).cpu().numpy()
        valid_targets = (targets != 3).cpu().numpy()

        for prediction, target, valid in zip(
            binary_predictions, binary_targets, valid_targets
        ):
            prediction = prediction.copy()
            target = target.copy()
            prediction[~valid] = 0
            target[~valid] = 0
            object_result = match_objects(
                target=target,
                prediction=prediction,
                iou_threshold=iou_threshold,
            )
            counts.object_tp += object_result.true_positive
            counts.object_fp += object_result.false_positive
            counts.object_fn += object_result.false_negative
            counts.matched_object_ious.extend(object_result.matched_ious)
            counts.ground_truth_objects += object_result.ground_truth_count
            counts.predicted_objects += object_result.predicted_count

        counts.samples += len(targets)

    return counts


def metric_rows(scope: str, counts: EvaluationCounts) -> list[dict]:
    """Create long-form rows for native, standard-binary, and extent metrics."""
    rows = []

    for row in metrics_from_confusion(counts.confusion_3class, CLASS_NAMES):
        rows.append({"scope": scope, "evaluation": "native_3class", **row})

    # Historical repository convention: interior is field; background and
    # boundary are non-field.
    standard_binary = collapse_confusion(counts.confusion_3class, (0, 1, 0))
    for row in metrics_from_confusion(standard_binary, ("non_field", "field_interior")):
        rows.append({"scope": scope, "evaluation": "standard_binary", **row})

    # Field-extent convention: both interior and boundary are field.
    field_extent = collapse_confusion(counts.confusion_3class, (0, 1, 1))
    for row in metrics_from_confusion(field_extent, ("background", "field_extent")):
        rows.append({"scope": scope, "evaluation": "field_extent_2class", **row})

    return rows


def summary_row(scope: str, counts: EvaluationCounts, metric_rows: list[dict]) -> dict:
    """Create macro pixel, object-detection, and panoptic metrics for one scope."""
    native_rows = [
        row
        for row in metric_rows
        if row["scope"] == scope and row["evaluation"] == "native_3class"
    ]
    object_precision = safe_divide(
        counts.object_tp, counts.object_tp + counts.object_fp
    )
    object_recall = safe_divide(counts.object_tp, counts.object_tp + counts.object_fn)
    object_f1 = safe_divide(
        2 * object_precision * object_recall, object_precision + object_recall
    )
    matched_ious = np.asarray(counts.matched_object_ious, dtype=float)
    sum_matched_iou = float(matched_ious.sum())
    mean_matched_iou = (
        float(matched_ious.mean()) if len(matched_ious) else float("nan")
    )
    median_matched_iou = (
        float(np.median(matched_ious)) if len(matched_ious) else float("nan")
    )
    panoptic_denominator = (
        counts.object_tp + 0.5 * counts.object_fp + 0.5 * counts.object_fn
    )
    segmentation_quality = safe_divide(sum_matched_iou, counts.object_tp)
    recognition_quality = safe_divide(counts.object_tp, panoptic_denominator)
    panoptic_quality = safe_divide(sum_matched_iou, panoptic_denominator)
    return {
        "scope": scope,
        "samples": counts.samples,
        "native_macro_iou": sum(row["iou"] for row in native_rows) / 3,
        "native_macro_precision": sum(row["precision"] for row in native_rows) / 3,
        "native_macro_recall": sum(row["recall"] for row in native_rows) / 3,
        "native_macro_f1": sum(row["f1"] for row in native_rows) / 3,
        "standard_object_precision": object_precision,
        "standard_object_recall": object_recall,
        "standard_object_f1": object_f1,
        "object_true_positives": counts.object_tp,
        "object_false_positives": counts.object_fp,
        "object_false_negatives": counts.object_fn,
        "ground_truth_object_count": counts.ground_truth_objects,
        "predicted_object_count": counts.predicted_objects,
        "object_count_error": counts.predicted_objects - counts.ground_truth_objects,
        "mean_matched_object_iou": mean_matched_iou,
        "median_matched_object_iou": median_matched_iou,
        "segmentation_quality_sq": segmentation_quality,
        "recognition_quality_rq": recognition_quality,
        "panoptic_quality_pq": panoptic_quality,
    }


def write_confusion(path: Path, confusion: torch.Tensor) -> None:
    """Write a labeled native 3-class confusion matrix."""
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["target\\prediction", *CLASS_NAMES])
        for class_name, values in zip(CLASS_NAMES, confusion.tolist()):
            writer.writerow([class_name, *values])


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not data_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {data_dir}")
    if not model_path.is_file():
        raise SystemExit(f"Model checkpoint not found: {model_path}")
    if args.gpu >= 0 and not torch.cuda.is_available():
        raise SystemExit("A GPU was requested, but PyTorch cannot access CUDA.")
    if args.iou_threshold < 0.5 or args.iou_threshold > 1:
        raise SystemExit("--iou-threshold must be between 0.5 and 1.")

    device = (
        torch.device(f"cuda:{args.gpu}")
        if args.gpu >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    country_counts = discover_test_countries(data_dir)
    if not country_counts:
        raise SystemExit(f"No countries with test samples found in: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading model on {device}: {model_path}")
    task = CustomSemanticSegmentationTask.load_from_checkpoint(
        str(model_path), map_location="cpu"
    )
    if int(task.hparams.get("num_classes", 3)) != 3:
        raise SystemExit("This evaluator requires a checkpoint with 3 output classes.")
    model_type = task.hparams["model"]
    model = task.model.eval().to(device)

    print("Discovered test data:")
    for country, count in country_counts:
        print(f"  {country}: {count} metadata rows")

    all_counts = EvaluationCounts.empty()
    counts_by_scope: dict[str, EvaluationCounts] = {}

    for index, (country, _) in enumerate(country_counts, start=1):
        print(f"\n[{index}/{len(country_counts)}] Evaluating {country}")
        counts = evaluate_country(
            country=country,
            data_dir=data_dir,
            model=model,
            model_type=model_type,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            iou_threshold=args.iou_threshold,
        )
        counts_by_scope[country] = counts
        all_counts.add(counts)
        write_confusion(
            output_dir / f"{country}_confusion_3class.csv",
            counts.confusion_3class,
        )

    counts_by_scope["all_countries"] = all_counts
    write_confusion(
        output_dir / "all_countries_confusion_3class.csv",
        all_counts.confusion_3class,
    )

    metrics = []
    summaries = []
    for scope, counts in counts_by_scope.items():
        scope_metrics = metric_rows(scope, counts)
        metrics.extend(scope_metrics)
        summaries.append(summary_row(scope, counts, scope_metrics))

    metrics_frame = pd.DataFrame(metrics)
    summary_frame = pd.DataFrame(summaries)
    object_matches_frame = pd.DataFrame.from_records(
        [
            {"scope": scope, "matched_object_id": index, "iou": iou}
            for scope, counts in counts_by_scope.items()
            if scope != "all_countries"
            for index, iou in enumerate(counts.matched_object_ious, start=1)
        ],
        columns=["scope", "matched_object_id", "iou"],
    )
    metrics_frame.to_csv(output_dir / "classwise_metrics.csv", index=False)
    summary_frame.to_csv(output_dir / "summary_metrics.csv", index=False)
    object_matches_frame.to_csv(output_dir / "matched_object_ious.csv", index=False)

    print("\nCombined classwise metrics")
    print(
        metrics_frame[metrics_frame["scope"] == "all_countries"].to_string(
            index=False
        )
    )
    print("\nSummary")
    print(summary_frame.to_string(index=False))
    print(f"\nEvaluation complete. Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
