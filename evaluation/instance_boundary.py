#!/usr/bin/env python3
"""Rigorous FTW field-instance and boundary evaluation.

Ground-truth fields come from ``label_masks/instance``. Predicted fields are
4-connected components of the model's class-1 (field interior) prediction.
Objects are matched one-to-one at multiple IoU thresholds. Boundary metrics use
the native class-2 prediction and support exact, 1-pixel, and 2-pixel tolerance.
"""

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from einops import rearrange
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluation import confidence, metrics
from ftw_tools.training.datasets import FTW
from ftw_tools.training.trainers import CustomSemanticSegmentationTask


DEFAULT_THRESHOLDS = (0.25, 0.50, 0.75)
PRIMARY_THRESHOLD = 0.50


def scratch_path(*parts: str) -> Path:
    return Path("/sfs/weka/scratch") / os.environ.get("USER", "unknown") / Path(*parts)


def parse_args() -> argparse.Namespace:
    run_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FTW field instances and boundaries on downloaded test data."
        )
    )
    parser.add_argument(
        "--data-dir", type=Path, default=scratch_path("ftw_data", "ftw")
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=scratch_path("ftw_models", "FTW_PRUE_EFNET_B5.ckpt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=scratch_path("ftw_results", f"instance_job_{run_id}"),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--metres-per-pixel", type=float, default=10.0)
    parser.add_argument("--geometry-epsilon-px", type=float, default=1.0)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-sample-pdfs", action="store_true")
    parser.add_argument(
        "--association-threshold",
        type=float,
        default=0.10,
        help="Minimum overlap coefficient used to identify merge/split relations.",
    )
    return parser.parse_args()


class IndexedFTW(Dataset):
    """Attach chip IDs to samples returned by the existing FTW dataset."""

    def __init__(self, dataset: FTW) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        sample["chip_id"] = Path(self.dataset.filenames[index]["mask"]).stem
        return sample


@dataclass
class ObjectArrays:
    gt_ids: np.ndarray
    prediction_ids: np.ndarray
    intersections: np.ndarray
    gt_areas: np.ndarray
    prediction_areas: np.ndarray
    ious: np.ndarray


def model_outputs(
    model: torch.nn.Module, model_type: str, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if model_type in {"fcsiamdiff", "fcsiamconc", "fcsiamavg"}:
        images = rearrange(images, "b (t c) h w -> b t c h w", t=2)
    with torch.inference_mode():
        logits = model(images)
        probabilities = logits.softmax(dim=1)
        return logits.argmax(dim=1), probabilities


def build_object_arrays(
    instance_mask: np.ndarray, prediction_mask: np.ndarray
) -> ObjectArrays:
    """Build pairwise intersection and IoU matrices for one chip."""
    gt_ids = np.unique(instance_mask)
    gt_ids = gt_ids[gt_ids != 0]
    prediction_labels, prediction_count = ndimage.label(
        prediction_mask.astype(bool), ndimage.generate_binary_structure(2, 1)
    )
    prediction_ids = np.arange(1, prediction_count + 1, dtype=np.int64)

    gt_compact = np.zeros(instance_mask.shape, dtype=np.int64)
    if len(gt_ids):
        positive = instance_mask != 0
        gt_compact[positive] = np.searchsorted(gt_ids, instance_mask[positive]) + 1

    stride = prediction_count + 1
    intersections_with_background = np.bincount(
        gt_compact.ravel() * stride + prediction_labels.ravel(),
        minlength=(len(gt_ids) + 1) * stride,
    ).reshape(len(gt_ids) + 1, stride)
    intersections = intersections_with_background[1:, 1:]
    gt_areas = np.bincount(gt_compact.ravel(), minlength=len(gt_ids) + 1)[1:]
    prediction_areas = np.bincount(
        prediction_labels.ravel(), minlength=prediction_count + 1
    )[1:]

    if len(gt_ids) and prediction_count:
        unions = gt_areas[:, None] + prediction_areas[None, :] - intersections
        ious = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions != 0,
        )
    else:
        ious = np.zeros((len(gt_ids), prediction_count), dtype=float)

    return ObjectArrays(
        gt_ids=gt_ids,
        prediction_ids=prediction_ids,
        intersections=intersections,
        gt_areas=gt_areas,
        prediction_areas=prediction_areas,
        ious=ious,
    )


def match_at_threshold(arrays: ObjectArrays, threshold: float) -> list[tuple[int, int]]:
    """Maximum-cardinality, then maximum-IoU, one-to-one assignment."""
    if arrays.ious.size == 0:
        return []

    valid = arrays.ious >= threshold
    cardinality_weight = min(arrays.ious.shape) + 1.0
    scores = valid.astype(float) * cardinality_weight + arrays.ious
    gt_indices, prediction_indices = linear_sum_assignment(scores, maximize=True)
    return [
        (int(gt_index), int(prediction_index))
        for gt_index, prediction_index in zip(gt_indices, prediction_indices)
        if arrays.ious[gt_index, prediction_index] >= threshold
    ]


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def object_metric_row(
    scope: str,
    threshold: float,
    true_positive: int,
    false_positive: int,
    false_negative: int,
    matched_ious: list[float],
) -> dict:
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    matched = np.asarray(matched_ious, dtype=float)
    sum_iou = float(matched.sum())
    denominator = true_positive + 0.5 * false_positive + 0.5 * false_negative
    return {
        "scope": scope,
        "iou_threshold": threshold,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "object_precision": precision,
        "object_recall": recall,
        "object_f1": f1,
        "mean_matched_iou": float(matched.mean()) if len(matched) else float("nan"),
        "median_matched_iou": (
            float(np.median(matched)) if len(matched) else float("nan")
        ),
        "segmentation_quality_sq": safe_divide(sum_iou, true_positive),
        "recognition_quality_rq": safe_divide(true_positive, denominator),
        "panoptic_quality_pq": safe_divide(sum_iou, denominator),
    }


def binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    true_positive = int(np.count_nonzero(target & prediction))
    false_positive = int(np.count_nonzero(~target & prediction))
    false_negative = int(np.count_nonzero(target & ~prediction))
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    return {
        "iou": safe_divide(
            true_positive, true_positive + false_positive + false_negative
        ),
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall),
        "target_pixels": int(np.count_nonzero(target)),
        "predicted_pixels": int(np.count_nonzero(prediction)),
    }


def boundary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    """Calculate exact, tolerant, and distance-based boundary metrics."""
    row = {
        f"exact_{key}": value
        for key, value in binary_metrics(target, prediction).items()
    }

    for tolerance in (1, 2):
        structure = ndimage.generate_binary_structure(2, 2)
        target_dilated = ndimage.binary_dilation(
            target, structure=structure, iterations=tolerance
        )
        prediction_dilated = ndimage.binary_dilation(
            prediction, structure=structure, iterations=tolerance
        )
        precision = safe_divide(
            int(np.count_nonzero(prediction & target_dilated)),
            int(np.count_nonzero(prediction)),
        )
        recall = safe_divide(
            int(np.count_nonzero(target & prediction_dilated)),
            int(np.count_nonzero(target)),
        )
        row[f"tolerance_{tolerance}px_precision"] = precision
        row[f"tolerance_{tolerance}px_recall"] = recall
        row[f"tolerance_{tolerance}px_f1"] = safe_divide(
            2 * precision * recall, precision + recall
        )

    if target.any() and prediction.any():
        distance_to_target = ndimage.distance_transform_edt(~target)
        distance_to_prediction = ndimage.distance_transform_edt(~prediction)
        prediction_distances = distance_to_target[prediction]
        target_distances = distance_to_prediction[target]
        symmetric_distances = np.concatenate(
            [prediction_distances, target_distances]
        )
        row["average_symmetric_boundary_distance_px"] = float(
            symmetric_distances.mean()
        )
        row["hausdorff_95_px"] = float(np.percentile(symmetric_distances, 95))
    else:
        row["average_symmetric_boundary_distance_px"] = float("nan")
        row["hausdorff_95_px"] = float("nan")
    return row


def overlap_relations(
    arrays: ObjectArrays, association_threshold: float
) -> tuple[int, int]:
    """Count split GT fields and merged predicted fields by overlap coefficient."""
    if arrays.intersections.size == 0:
        return 0, 0
    minimum_areas = np.minimum(
        arrays.gt_areas[:, None], arrays.prediction_areas[None, :]
    )
    coefficients = np.divide(
        arrays.intersections,
        minimum_areas,
        out=np.zeros_like(arrays.intersections, dtype=float),
        where=minimum_areas != 0,
    )
    associated = coefficients >= association_threshold
    split_fields = int(np.count_nonzero(associated.sum(axis=1) >= 2))
    merged_predictions = int(np.count_nonzero(associated.sum(axis=0) >= 2))
    return split_fields, merged_predictions


def aggregate_object_rows(chip_rows: list[dict]) -> pd.DataFrame:
    rows = []
    frame = pd.DataFrame(chip_rows)
    scopes = sorted(frame["country"].unique()) + ["all_countries"]
    for scope in scopes:
        subset = frame if scope == "all_countries" else frame[frame["country"] == scope]
        for threshold in DEFAULT_THRESHOLDS:
            threshold_rows = subset[subset["iou_threshold"] == threshold]
            matched_ious = [
                value
                for values in threshold_rows["matched_ious"]
                for value in values
            ]
            row = object_metric_row(
                scope=scope,
                threshold=threshold,
                true_positive=int(threshold_rows["true_positives"].sum()),
                false_positive=int(threshold_rows["false_positives"].sum()),
                false_negative=int(threshold_rows["false_negatives"].sum()),
                matched_ious=matched_ious,
            )
            row["ground_truth_objects"] = int(threshold_rows["gt_count"].sum())
            row["predicted_objects"] = int(threshold_rows["prediction_count"].sum())
            row["object_count_error"] = (
                row["predicted_objects"] - row["ground_truth_objects"]
            )
            row["split_ground_truth_fields"] = int(
                threshold_rows["split_fields"].sum()
            )
            row["merged_predictions"] = int(
                threshold_rows["merged_predictions"].sum()
            )
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_boundary_rows(chip_rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(chip_rows)
    metric_columns = [
        column
        for column in frame.columns
        if column not in {"country", "chip_id"}
    ]
    rows = []
    for scope in sorted(frame["country"].unique()) + ["all_countries"]:
        subset = frame if scope == "all_countries" else frame[frame["country"] == scope]
        row = {"scope": scope, "chips": len(subset)}
        for column in metric_columns:
            row[column] = float(subset[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_country(
    country: str,
    data_dir: Path,
    output_dir: Path,
    model: torch.nn.Module,
    model_type: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    association_threshold: float,
    metres_per_pixel: float,
    geometry_epsilon_px: float,
    write_sample_pdfs: bool,
) -> tuple[list[dict], ...]:
    base_dataset = FTW(
        root=str(data_dir),
        countries=[country],
        split="test",
        load_boundaries=True,
        temporal_options="stacked",
        verbose=False,
    )
    dataloader = DataLoader(
        IndexedFTW(base_dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    object_rows = []
    gt_field_rows = []
    prediction_rows = []
    boundary_rows = []
    geometry_rows = []
    geometry_population_rows = []
    geometry_exclusions = []
    repair_rows = []
    closing_rows = []
    probability_rows = []
    pixel_records = []
    probability_records = []
    confidence_rows = []
    topology_rows = []
    ground_truth_count = 0

    for batch in tqdm(dataloader, desc=country):
        images = batch["image"].to(device, non_blocking=True) / 3000.0
        report_images = batch["image"][:, :3].numpy()
        semantic_targets = batch["mask"].numpy()
        predicted_classes, probability_tensors = model_outputs(
            model, model_type, images
        )
        predictions = predicted_classes.cpu().numpy()
        probabilities = probability_tensors.cpu().numpy()

        for chip_id, report_image, semantic_target, prediction, probability in zip(
            batch["chip_id"],
            report_images,
            semantic_targets,
            predictions,
            probabilities,
        ):
            boundary_probability = probability[2]
            valid = semantic_target != 3
            instance_path = (
                data_dir
                / country
                / "label_masks"
                / "instance"
                / f"{chip_id}.tif"
            )
            if not instance_path.is_file():
                raise FileNotFoundError(f"Instance mask not found: {instance_path}")
            with rasterio.open(instance_path) as source:
                instance_mask = source.read(1)
            if instance_mask.shape != semantic_target.shape:
                raise ValueError(
                    f"Shape mismatch for {country}/{chip_id}: "
                    f"instance={instance_mask.shape}, semantic={semantic_target.shape}"
                )

            # Preserve authoritative instance IDs, but compare like with like:
            # the repository polygonizes class-1 interiors and excludes class-2
            # boundary pixels from field objects.
            arrays, gt_labels, prediction_labels = metrics.semantic_object_arrays(
                semantic_target, instance_mask, prediction
            )
            pixel_record = metrics.semantic_pixel_record(
                country, chip_id, semantic_target, prediction
            )
            pixel_records.append(pixel_record)
            probability_record = confidence.probability_chip_record(
                country, chip_id, semantic_target, probability
            )
            probability_records.append(probability_record)
            instance_mask = gt_labels
            ground_truth_count += len(arrays.gt_ids)
            object_confidences = confidence.semantic_object_confidences(
                arrays,
                prediction_labels,
                prediction,
                probability,
            )
            topology_values = metrics.topology_metrics(
                instance_mask, prediction_labels, valid
            )
            topology_rows.append(
                {"country": country, "chip_id": chip_id, **topology_values}
            )

            primary_matches = metrics.match_at_threshold(
                arrays, metrics.PRIMARY_THRESHOLD
            )
            matched_gt = {
                gt_index: pred_index for gt_index, pred_index in primary_matches
            }
            matched_prediction = {
                pred_index: gt_index for gt_index, pred_index in primary_matches
            }

            object_rows.extend(metrics.object_chip_rows(country, chip_id, arrays))

            gt_masks = [instance_mask == value for value in arrays.gt_ids]
            predicted_masks = [
                prediction_labels == value for value in arrays.prediction_ids
            ]
            chip_geometry, excluded = metrics.geometry_pair_rows(
                country,
                chip_id,
                arrays,
                gt_masks,
                predicted_masks,
                epsilon=geometry_epsilon_px,
            )
            geometry_rows.extend(chip_geometry)
            geometry_population_rows.extend(
                metrics.geometry_population_rows(
                    country,
                    chip_id,
                    gt_masks,
                    predicted_masks,
                    epsilon=geometry_epsilon_px,
                )
            )
            chip_repair_rows = metrics.merge_repair_rows(
                country,
                chip_id,
                arrays,
                gt_masks,
                predicted_masks,
                boundary_probability=boundary_probability,
            )
            repair_rows.extend(chip_repair_rows)
            geometry_exclusions.append(
                {"country": country, "chip_id": chip_id, **excluded}
            )
            chip_closing_rows = metrics.closing_radius_rows(
                country, chip_id, semantic_target, instance_mask, prediction
            )
            closing_rows.extend(chip_closing_rows)
            chip_probability_rows = metrics.boundary_probability_sweep(
                country,
                chip_id,
                semantic_target,
                instance_mask,
                prediction,
                boundary_probability,
            )
            probability_rows.extend(chip_probability_rows)
            boundary_values = metrics.boundary_metrics(
                target=(semantic_target == 2) & valid,
                prediction=(prediction == 2) & valid,
                metres_per_pixel=metres_per_pixel,
            )
            if write_sample_pdfs:
                from evaluation.sample_reports import (
                    chip_metric_values,
                    save_semantic_sample_pdf,
                )

                save_semantic_sample_pdf(
                    output_dir=output_dir,
                    country=country,
                    chip_id=chip_id,
                    image=report_image,
                    semantic=semantic_target,
                    instances=instance_mask,
                    prediction=prediction,
                    probabilities=probability,
                    metric_values=chip_metric_values(
                        arrays,
                        boundary_values,
                        pixel_record,
                        chip_geometry,
                        chip_repair_rows,
                        closing_rows=chip_closing_rows,
                        probability_rows=chip_probability_rows,
                        topology=topology_values,
                        probability_record=probability_record,
                    ),
                )

            for gt_index, gt_id in enumerate(arrays.gt_ids):
                best_prediction_index = None
                if len(arrays.prediction_ids):
                    candidate_index = int(np.argmax(arrays.ious[gt_index]))
                    if arrays.ious[gt_index, candidate_index] > 0:
                        best_prediction_index = candidate_index
                matched_prediction_index = matched_gt.get(gt_index)
                gt_field_rows.append(
                    {
                        "country": country,
                        "chip_id": chip_id,
                        "gt_field_id": int(gt_id),
                        "best_prediction_id": (
                            int(arrays.prediction_ids[best_prediction_index])
                            if best_prediction_index is not None
                            else None
                        ),
                        "best_iou": (
                            float(arrays.ious[gt_index, best_prediction_index])
                            if best_prediction_index is not None
                            else 0.0
                        ),
                        "matched_prediction_id_iou50": (
                            int(arrays.prediction_ids[matched_prediction_index])
                            if matched_prediction_index is not None
                            else None
                        ),
                        "matched_iou_iou50": (
                            float(arrays.ious[gt_index, matched_prediction_index])
                            if matched_prediction_index is not None
                            else 0.0
                        ),
                        "detected_iou50": matched_prediction_index is not None,
                    }
                )

            for prediction_index, prediction_id in enumerate(arrays.prediction_ids):
                best_gt_index = None
                if len(arrays.gt_ids):
                    candidate_index = int(np.argmax(arrays.ious[:, prediction_index]))
                    if arrays.ious[candidate_index, prediction_index] > 0:
                        best_gt_index = candidate_index
                matched_gt_index = matched_prediction.get(prediction_index)
                confidence_values = object_confidences[prediction_index]
                candidates = [
                    (
                        int(arrays.gt_ids[gt_index]),
                        float(arrays.ious[gt_index, prediction_index]),
                    )
                    for gt_index in np.flatnonzero(
                        arrays.ious[:, prediction_index] > 0
                    )
                ]
                confidence_rows.append(
                    {
                        "country": country,
                        "chip_id": chip_id,
                        "prediction_id": int(prediction_id),
                        "correct_iou50": matched_gt_index is not None,
                        "best_iou": (
                            float(arrays.ious[best_gt_index, prediction_index])
                            if best_gt_index is not None
                            else 0.0
                        ),
                        "candidates": candidates,
                        **confidence_values,
                    }
                )
                prediction_rows.append(
                    {
                        "country": country,
                        "chip_id": chip_id,
                        "prediction_id": int(prediction_id),
                        "best_gt_field_id": (
                            int(arrays.gt_ids[best_gt_index])
                            if best_gt_index is not None
                            else None
                        ),
                        "best_iou": (
                            float(arrays.ious[best_gt_index, prediction_index])
                            if best_gt_index is not None
                            else 0.0
                        ),
                        "matched_gt_field_id_iou50": (
                            int(arrays.gt_ids[matched_gt_index])
                            if matched_gt_index is not None
                            else None
                        ),
                        "matched_iou_iou50": (
                            float(arrays.ious[matched_gt_index, prediction_index])
                            if matched_gt_index is not None
                            else 0.0
                        ),
                        "correct_iou50": matched_gt_index is not None,
                        **confidence_values,
                    }
                )

            boundary_row = {
                "country": country,
                "chip_id": chip_id,
                **boundary_values,
            }
            boundary_rows.append(boundary_row)

    return (
        object_rows,
        gt_field_rows,
        prediction_rows,
        boundary_rows,
        geometry_rows,
        geometry_population_rows,
        geometry_exclusions,
        repair_rows,
        closing_rows,
        probability_rows,
        pixel_records,
        probability_records,
        confidence_rows,
        ground_truth_count,
        topology_rows,
    )


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
        raise SystemExit("A GPU was requested, but CUDA is unavailable.")
    if not 0 < args.association_threshold <= 1:
        raise SystemExit("--association-threshold must be in (0, 1].")
    if args.metres_per_pixel <= 0 or args.geometry_epsilon_px < 0:
        raise SystemExit("Pixel scale must be positive and geometry epsilon nonnegative.")

    device = (
        torch.device(f"cuda:{args.gpu}")
        if args.gpu >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    task = CustomSemanticSegmentationTask.load_from_checkpoint(
        str(model_path), map_location="cpu"
    )
    if int(task.hparams.get("num_classes", 3)) != 3:
        raise SystemExit("This evaluator requires a native 3-class checkpoint.")
    model = task.model.eval().to(device)
    model_type = task.hparams["model"]

    country_files = sorted(data_dir.glob("*/chips_*.parquet"))
    countries = []
    for chips_file in country_files:
        table = pd.read_parquet(chips_file, columns=["split"])
        if (table["split"] == "test").any():
            countries.append(chips_file.parent.name)
    if not countries:
        raise SystemExit(f"No downloaded countries with test samples found: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_object_rows = []
    all_gt_field_rows = []
    all_prediction_rows = []
    all_boundary_rows = []
    all_geometry_rows = []
    all_geometry_population_rows = []
    all_geometry_exclusions = []
    all_repair_rows = []
    all_closing_rows = []
    all_probability_rows = []
    all_pixel_records = []
    all_probability_records = []
    all_confidence_rows = []
    all_topology_rows = []
    ground_truth_counts = {}

    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Countries: {', '.join(countries)}")
    for country in countries:
        (
            object_rows,
            gt_rows,
            prediction_rows,
            boundary_rows,
            geometry_rows,
            geometry_population_rows,
            geometry_exclusions,
            repair_rows,
            closing_rows,
            probability_rows,
            pixel_records,
            probability_records,
            confidence_rows,
            ground_truth_count,
            topology_rows,
        ) = evaluate_country(
            country=country,
            data_dir=data_dir,
            output_dir=output_dir,
            model=model,
            model_type=model_type,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            association_threshold=args.association_threshold,
            metres_per_pixel=args.metres_per_pixel,
            geometry_epsilon_px=args.geometry_epsilon_px,
            write_sample_pdfs=not args.skip_sample_pdfs,
        )
        all_object_rows.extend(object_rows)
        all_gt_field_rows.extend(gt_rows)
        all_prediction_rows.extend(prediction_rows)
        all_boundary_rows.extend(boundary_rows)
        all_geometry_rows.extend(geometry_rows)
        all_geometry_population_rows.extend(geometry_population_rows)
        all_geometry_exclusions.extend(geometry_exclusions)
        all_repair_rows.extend(repair_rows)
        all_closing_rows.extend(closing_rows)
        all_probability_rows.extend(probability_rows)
        all_pixel_records.extend(pixel_records)
        all_probability_records.extend(probability_records)
        all_confidence_rows.extend(confidence_rows)
        ground_truth_counts[country] = ground_truth_count
        all_topology_rows.extend(topology_rows)

    object_summary = metrics.aggregate_object_rows(all_object_rows)
    boundary_by_chip = pd.DataFrame(all_boundary_rows)
    boundary_summary = metrics.aggregate_macro_rows(all_boundary_rows)
    geometry_summary = metrics.aggregate_geometry_rows(all_geometry_rows)
    geometry_population_summary = metrics.aggregate_geometry_population(
        all_geometry_population_rows
    )
    repair_summary = metrics.aggregate_repair_rows(all_repair_rows)
    probability_summary, probability_best = metrics.aggregate_probability_sweep(
        all_probability_rows
    )
    pixel_summary, pixel_confusion = metrics.aggregate_pixel_records(
        all_pixel_records
    )
    calibration_summary, calibration_bins, calibration_distance = (
        confidence.aggregate_probability_records(all_probability_records)
    )
    confidence_summary = confidence.confidence_reliability_summary(
        all_confidence_rows,
        ("persistence", "mean_interior_probability", "mean_top_confidence"),
    )
    try:
        risk_coverage, risk_summary = confidence.risk_coverage_rows(
            all_confidence_rows,
            ground_truth_counts,
            ("persistence", "mean_interior_probability", "area_pixels"),
        )
    except Exception as error:
        print(f"WARNING: risk-coverage summary failed: {error}")
        risk_coverage, risk_summary = pd.DataFrame(), pd.DataFrame()
    topology_summary = metrics.aggregate_macro_rows(all_topology_rows)

    object_summary.to_csv(output_dir / "object_summary_by_threshold.csv", index=False)
    pd.DataFrame(all_gt_field_rows).to_csv(
        output_dir / "ground_truth_field_matches_iou50.csv", index=False
    )
    pd.DataFrame(all_prediction_rows).to_csv(
        output_dir / "prediction_field_matches_iou50.csv", index=False
    )
    boundary_by_chip.to_csv(output_dir / "boundary_metrics_by_chip.csv", index=False)
    boundary_summary.to_csv(output_dir / "boundary_summary.csv", index=False)
    pd.DataFrame(all_geometry_rows).to_csv(
        output_dir / "geometry_by_matched_field.csv", index=False
    )
    geometry_summary.to_csv(output_dir / "geometry_summary.csv", index=False)
    pd.DataFrame(all_geometry_population_rows).to_csv(
        output_dir / "geometry_population.csv", index=False
    )
    geometry_population_summary.to_csv(
        output_dir / "geometry_population_summary.csv", index=False
    )
    pd.DataFrame(all_geometry_exclusions).to_csv(
        output_dir / "geometry_border_exclusions.csv", index=False
    )
    pd.DataFrame(all_closing_rows).to_csv(
        output_dir / "closing_radius_sweep.csv", index=False
    )
    pd.DataFrame(all_repair_rows).to_csv(
        output_dir / "merge_repair_distance.csv", index=False
    )
    repair_summary.to_csv(output_dir / "merge_repair_summary.csv", index=False)
    pd.DataFrame(all_probability_rows).to_csv(
        output_dir / "boundary_probability_sweep_by_chip.csv", index=False
    )
    probability_summary.to_csv(
        output_dir / "boundary_probability_sweep.csv", index=False
    )
    probability_best.to_csv(
        output_dir / "boundary_probability_best_pq.csv", index=False
    )
    pixel_summary.to_csv(output_dir / "pixel_metrics.csv", index=False)
    pixel_confusion.to_csv(output_dir / "pixel_confusion.csv", index=False)
    confidence_frame = pd.DataFrame(all_confidence_rows)
    if "candidates" in confidence_frame:
        confidence_frame = confidence_frame.drop(columns="candidates")
    confidence_frame.to_csv(
        output_dir / "field_confidence_by_prediction.csv", index=False
    )
    confidence_summary.to_csv(
        output_dir / "field_confidence_summary.csv", index=False
    )
    risk_coverage.to_csv(output_dir / "risk_coverage.csv", index=False)
    risk_summary.to_csv(output_dir / "risk_coverage_summary.csv", index=False)
    calibration_summary.to_csv(
        output_dir / "probabilistic_metrics.csv", index=False
    )
    calibration_bins.to_csv(output_dir / "calibration_bins.csv", index=False)
    calibration_distance.to_csv(
        output_dir / "calibration_by_boundary_distance.csv", index=False
    )
    pd.DataFrame(all_topology_rows).to_csv(
        output_dir / "topology_metrics_by_chip.csv", index=False
    )
    topology_summary.to_csv(output_dir / "topology_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": str(model_path),
                "split": "test",
                "object_iou_thresholds": "0.25,0.50,0.75",
                "association_alpha_sweep": "0.05,0.10,0.20,0.30",
                "boundary_probability_sweep": "0.10:0.05:0.90",
                "metres_per_pixel": args.metres_per_pixel,
                "geometry_epsilon_px": args.geometry_epsilon_px,
                "sample_pdfs": not args.skip_sample_pdfs,
                "confidence_scores": (
                    "persistence,mean_interior_probability,"
                    "mean_top_confidence,area_baseline"
                ),
                "calibration_fine_bins": confidence.CALIBRATION_FINE_BINS,
                "calibration_adaptive_bins": confidence.ADAPTIVE_BINS,
                "temperature_scaling": "not_applied",
            }
        ]
    ).to_csv(output_dir / "run_settings.csv", index=False)
    if not args.skip_plots:
        from evaluation.plots import create_plots

        create_plots(output_dir)

    print("\nObject summary")
    print(object_summary.to_string(index=False))
    print("\nBoundary summary")
    print(boundary_summary.to_string(index=False))
    print(f"\nResults saved in: {output_dir}")


if __name__ == "__main__":
    main()
