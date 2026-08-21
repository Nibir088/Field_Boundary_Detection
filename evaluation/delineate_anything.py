#!/usr/bin/env python3
"""Evaluate Delineate Anything variants on FTW instance test masks."""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as functional
from scipy import ndimage
from tqdm import tqdm

from evaluation import confidence, metrics
from evaluation.metrics import (
    DEFAULT_THRESHOLDS,
    PRIMARY_THRESHOLD,
    ObjectArrays,
    aggregate_object_rows,
    boundary_metrics,
    match_at_threshold,
)
from ftw_tools.inference.models import DelineateAnything
from ftw_tools.settings import FULL_DATA_COUNTRIES
from ftw_tools.training.datasets import FTW


MODEL_NAMES = ("DelineateAnything-S", "DelineateAnything", "DelineateAnythingV2")
MODEL_FILENAMES = {
    "DelineateAnything-S": "DelineateAnything-S.pt",
    "DelineateAnything": "DelineateAnything.pt",
    "DelineateAnythingV2": "DelineateAnythingV2.pt",
}
AP_THRESHOLDS = tuple(float(value) for value in np.arange(0.50, 0.96, 0.05))


def scratch_path(*parts: str) -> Path:
    return Path("/sfs/weka/scratch") / os.environ.get("USER", "unknown") / Path(*parts)


def parse_args() -> argparse.Namespace:
    run_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser = argparse.ArgumentParser(
        description="Evaluate a Delineate Anything model on FTW test instances."
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="DelineateAnythingV2")
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Optional local .pt file. Otherwise the registered URL/cache is used.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=scratch_path("ftw_data", "ftw")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=scratch_path("ftw_results", f"delineate_job_{run_id}"),
    )
    parser.add_argument(
        "--window", choices=("window_a", "window_b"), default="window_b"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resize-factor", type=int, default=2)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.30)
    parser.add_argument("--association-threshold", type=float, default=0.10)
    parser.add_argument("--metres-per-pixel", type=float, default=10.0)
    parser.add_argument("--geometry-epsilon-px", type=float, default=1.0)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-sample-pdfs", action="store_true")
    return parser.parse_args()


def build_arrays_from_masks(
    instance_mask: np.ndarray, prediction_masks: np.ndarray
) -> ObjectArrays:
    """Build pairwise IoU arrays without discarding overlapping predictions."""
    gt_ids = np.unique(instance_mask)
    gt_ids = gt_ids[gt_ids != 0]
    prediction_ids = np.arange(1, len(prediction_masks) + 1, dtype=np.int64)
    gt_areas = np.asarray(
        [np.count_nonzero(instance_mask == gt_id) for gt_id in gt_ids], dtype=np.int64
    )
    if len(prediction_masks):
        prediction_areas = prediction_masks.reshape(len(prediction_masks), -1).sum(
            axis=1
        )
    else:
        prediction_areas = np.zeros(0, dtype=np.int64)
    intersections = np.zeros((len(gt_ids), len(prediction_masks)), dtype=np.int64)
    for gt_index, gt_id in enumerate(gt_ids):
        gt_mask = instance_mask == gt_id
        if len(prediction_masks):
            intersections[gt_index] = np.count_nonzero(
                prediction_masks & gt_mask[None, ...], axis=(1, 2)
            )
    unions = gt_areas[:, None] + prediction_areas[None, :] - intersections
    ious = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=float),
        where=unions != 0,
    )
    return ObjectArrays(
        gt_ids=gt_ids,
        prediction_ids=prediction_ids,
        intersections=intersections,
        gt_areas=gt_areas,
        prediction_areas=prediction_areas,
        ious=ious,
    )


def result_masks(
    result, output_shape: tuple[int, int], valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return native predicted instance masks and aligned confidence scores."""
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, *output_shape), dtype=bool), np.zeros(0, dtype=float)
    masks = result.masks.data.float().unsqueeze(1)
    masks = functional.interpolate(masks, size=output_shape, mode="nearest")
    masks = masks.squeeze(1).cpu().numpy() >= 0.5
    masks &= valid[None, ...]
    confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
    keep = masks.reshape(len(masks), -1).any(axis=1)
    return masks[keep], confidences[keep]


def prediction_boundaries(prediction_masks: np.ndarray) -> np.ndarray:
    """Create the union of one-pixel inner edges of predicted instances."""
    if not len(prediction_masks):
        return np.zeros(prediction_masks.shape[1:], dtype=bool)
    from scipy import ndimage

    boundaries = np.zeros(prediction_masks.shape[1:], dtype=bool)
    structure = ndimage.generate_binary_structure(2, 2)
    for mask in prediction_masks:
        boundaries |= mask & ~ndimage.binary_erosion(mask, structure=structure)
    return boundaries


def average_precision(
    detections: list[dict], ground_truth_count: int, threshold: float
) -> float:
    """Compute 101-point interpolated AP with confidence-ranked matching."""
    if ground_truth_count == 0:
        return float("nan")
    detections = sorted(detections, key=lambda row: row["confidence"], reverse=True)
    used: dict[tuple[str, str], set[int]] = {}
    true_positives = []
    false_positives = []
    for detection in detections:
        chip_key = (detection["country"], detection["chip_id"])
        used.setdefault(chip_key, set())
        candidates = sorted(
            detection["candidates"], key=lambda pair: pair[1], reverse=True
        )
        match = next(
            (
                gt_id
                for gt_id, iou in candidates
                if iou >= threshold and gt_id not in used[chip_key]
            ),
            None,
        )
        if match is None:
            true_positives.append(0)
            false_positives.append(1)
        else:
            used[chip_key].add(match)
            true_positives.append(1)
            false_positives.append(0)

    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recalls = cumulative_tp / ground_truth_count
    precisions = np.divide(
        cumulative_tp,
        cumulative_tp + cumulative_fp,
        out=np.zeros_like(cumulative_tp, dtype=float),
        where=(cumulative_tp + cumulative_fp) != 0,
    )
    return float(
        np.mean(
            [
                precisions[recalls >= recall_level].max()
                if np.any(recalls >= recall_level)
                else 0.0
                for recall_level in np.linspace(0, 1, 101)
            ]
        )
    )


def ap_summary(detection_rows: list[dict], gt_counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    countries = sorted(gt_counts)
    for scope in countries + ["all_countries"]:
        detections = (
            detection_rows
            if scope == "all_countries"
            else [row for row in detection_rows if row["country"] == scope]
        )
        gt_count = (
            sum(gt_counts.values()) if scope == "all_countries" else gt_counts[scope]
        )
        values = {
            threshold: average_precision(detections, gt_count, threshold)
            for threshold in AP_THRESHOLDS
        }
        row = {
            "scope": scope,
            "ground_truth_objects": gt_count,
            "predictions": len(detections),
            "presence_only_precision_warning": bool(
                scope == "all_countries"
                and any(country not in FULL_DATA_COUNTRIES for country in countries)
                or scope != "all_countries" and scope not in FULL_DATA_COUNTRIES
            ),
            "ap50": values[0.50],
            "map50_95": float(np.mean(list(values.values()))),
        }
        row.update(
            {
                f"ap{int(threshold * 100):02d}": value
                for threshold, value in values.items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def append_match_tables(
    country: str,
    chip_id: str,
    arrays: ObjectArrays,
    confidences: np.ndarray,
    gt_rows: list[dict],
    prediction_rows: list[dict],
) -> None:
    matches = match_at_threshold(arrays, PRIMARY_THRESHOLD)
    gt_to_prediction = dict(matches)
    prediction_to_gt = {prediction: gt for gt, prediction in matches}
    for gt_index, gt_id in enumerate(arrays.gt_ids):
        best_index = (
            int(np.argmax(arrays.ious[gt_index])) if arrays.ious.shape[1] else None
        )
        if best_index is not None and arrays.ious[gt_index, best_index] == 0:
            best_index = None
        matched_index = gt_to_prediction.get(gt_index)
        gt_rows.append(
            {
                "country": country,
                "chip_id": chip_id,
                "gt_field_id": int(gt_id),
                "best_prediction_id": (
                    best_index + 1 if best_index is not None else None
                ),
                "best_iou": (
                    float(arrays.ious[gt_index, best_index])
                    if best_index is not None
                    else 0.0
                ),
                "matched_prediction_id_iou50": (
                    matched_index + 1 if matched_index is not None else None
                ),
                "matched_iou_iou50": (
                    float(arrays.ious[gt_index, matched_index])
                    if matched_index is not None
                    else 0.0
                ),
                "detected_iou50": matched_index is not None,
            }
        )
    for prediction_index, prediction_id in enumerate(arrays.prediction_ids):
        best_index = (
            int(np.argmax(arrays.ious[:, prediction_index]))
            if arrays.ious.shape[0]
            else None
        )
        if best_index is not None and arrays.ious[best_index, prediction_index] == 0:
            best_index = None
        matched_index = prediction_to_gt.get(prediction_index)
        prediction_rows.append(
            {
                "country": country,
                "chip_id": chip_id,
                "prediction_id": int(prediction_id),
                "confidence": float(confidences[prediction_index]),
                "best_gt_field_id": (
                    int(arrays.gt_ids[best_index])
                    if best_index is not None
                    else None
                ),
                "best_iou": (
                    float(arrays.ious[best_index, prediction_index])
                    if best_index is not None
                    else 0.0
                ),
                "matched_gt_field_id_iou50": (
                    int(arrays.gt_ids[matched_index])
                    if matched_index is not None
                    else None
                ),
                "matched_iou_iou50": (
                    float(arrays.ious[matched_index, prediction_index])
                    if matched_index is not None
                    else 0.0
                ),
                "correct_iou50": matched_index is not None,
            }
        )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {data_dir}")
    if args.gpu >= 0 and not torch.cuda.is_available():
        raise SystemExit("A GPU was requested, but CUDA is unavailable.")
    if args.batch_size < 1 or args.resize_factor < 1 or args.max_detections < 1:
        raise SystemExit(
            "Batch size, resize factor, and maximum detections must be positive."
        )
    if not 0 <= args.conf_threshold <= 1 or not 0 <= args.nms_iou_threshold <= 1:
        raise SystemExit("Confidence and NMS IoU thresholds must be between 0 and 1.")
    if not 0 <= args.association_threshold <= 1:
        raise SystemExit("Association threshold must be between 0 and 1.")
    if args.metres_per_pixel <= 0 or args.geometry_epsilon_px < 0:
        raise SystemExit("Pixel scale must be positive and geometry epsilon nonnegative.")

    device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    if args.model_path:
        model_path = args.model_path.expanduser().resolve()
    else:
        candidate = scratch_path("ftw_models", MODEL_FILENAMES[args.model])
        model_path = candidate if candidate.is_file() else None
    if model_path:
        if not model_path.is_file():
            raise SystemExit(f"Model file not found: {model_path}")
        DelineateAnything.checkpoints[args.model] = str(model_path)

    try:
        model = DelineateAnything(
            model=args.model,
            patch_size=256,
            resize_factor=args.resize_factor,
            max_detections=args.max_detections,
            iou_threshold=args.nms_iou_threshold,
            conf_threshold=args.conf_threshold,
            device=device,
        )
    except ModuleNotFoundError as error:
        if error.name == "ultralytics":
            raise SystemExit(
                "Missing Delineate Anything dependencies. Install them with "
                "`python -m pip install -r requirements.txt` or run "
                "`uv sync --extra delineate-anything`."
            ) from error
        raise
    output_dir.mkdir(parents=True, exist_ok=True)
    temporal_option = "windowA" if args.window == "window_a" else "windowB"

    object_rows = []
    gt_rows = []
    prediction_rows = []
    boundary_rows = []
    geometry_rows = []
    geometry_population_rows = []
    geometry_exclusions = []
    repair_rows = []
    detection_rows = []
    gt_counts: dict[str, int] = {}
    pixel_records = []
    topology_rows = []

    countries = sorted(path.parent.name for path in data_dir.glob("*/chips_*.parquet"))
    for country in countries:
        dataset = FTW(
            root=str(data_dir),
            countries=[country],
            split="test",
            load_boundaries=True,
            temporal_options=temporal_option,
            verbose=False,
        )
        if not len(dataset):
            continue
        gt_counts[country] = 0
        for start in tqdm(range(0, len(dataset), args.batch_size), desc=country):
            indices = range(start, min(start + args.batch_size, len(dataset)))
            samples = [dataset[index] for index in indices]
            images = torch.stack([sample["image"] for sample in samples])
            results = model(images)
            for index, sample, result in zip(indices, samples, results):
                chip_id = Path(dataset.filenames[index]["mask"]).stem
                semantic_target = sample["mask"].numpy()
                valid = semantic_target != 3
                instance_path = (
                    data_dir
                    / country
                    / "label_masks"
                    / "instance"
                    / f"{chip_id}.tif"
                )
                with rasterio.open(instance_path) as source:
                    instance_mask = source.read(1)
                masks, confidences = result_masks(result, semantic_target.shape, valid)
                arrays = metrics.mask_object_arrays(
                    semantic_target, instance_mask, masks
                )
                pixel_record = metrics.instance_pixel_record(
                    country, chip_id, semantic_target, masks
                )
                pixel_records.append(pixel_record)
                gt_counts[country] += len(arrays.gt_ids)
                object_rows.extend(metrics.object_chip_rows(country, chip_id, arrays))

                gt_interior = np.where(semantic_target == 1, instance_mask, 0)
                gt_masks = [gt_interior == value for value in arrays.gt_ids]
                predicted_union = (
                    masks.any(axis=0)
                    if len(masks)
                    else np.zeros(semantic_target.shape, dtype=bool)
                )
                predicted_topology_labels, _ = ndimage.label(
                    predicted_union, metrics.N4
                )
                topology_values = metrics.topology_metrics(
                    gt_interior, predicted_topology_labels, valid
                )
                topology_rows.append(
                    {"country": country, "chip_id": chip_id, **topology_values}
                )
                chip_geometry, excluded = metrics.geometry_pair_rows(
                    country,
                    chip_id,
                    arrays,
                    gt_masks,
                    list(masks),
                    epsilon=args.geometry_epsilon_px,
                )
                geometry_rows.extend(chip_geometry)
                geometry_population_rows.extend(
                    metrics.geometry_population_rows(
                        country,
                        chip_id,
                        gt_masks,
                        list(masks),
                        epsilon=args.geometry_epsilon_px,
                    )
                )
                chip_repair_rows = metrics.merge_repair_rows(
                    country, chip_id, arrays, gt_masks, list(masks)
                )
                repair_rows.extend(chip_repair_rows)
                geometry_exclusions.append(
                    {"country": country, "chip_id": chip_id, **excluded}
                )

                append_match_tables(
                    country, chip_id, arrays, confidences, gt_rows, prediction_rows
                )
                primary_matches = match_at_threshold(arrays, PRIMARY_THRESHOLD)
                matched_predictions = {prediction for _, prediction in primary_matches}
                for prediction_index, score in enumerate(confidences):
                    candidates = [
                        (
                            int(arrays.gt_ids[gt_index]),
                            float(arrays.ious[gt_index, prediction_index]),
                        )
                        for gt_index in np.flatnonzero(
                            arrays.ious[:, prediction_index] > 0
                        )
                    ]
                    detection_rows.append(
                        {
                            "country": country,
                            "chip_id": chip_id,
                            "confidence": float(score),
                            "native_confidence": float(score),
                            "area_pixels": int(
                                arrays.prediction_areas[prediction_index]
                            ),
                            "best_iou": (
                                float(arrays.ious[:, prediction_index].max())
                                if len(arrays.gt_ids)
                                else 0.0
                            ),
                            "correct_iou50": prediction_index in matched_predictions,
                            "candidates": candidates,
                        }
                    )

                boundary_values = boundary_metrics(
                    target=(semantic_target == 2) & valid,
                    prediction=prediction_boundaries(masks) & valid,
                    metres_per_pixel=args.metres_per_pixel,
                )
                boundary_rows.append(
                    {"country": country, "chip_id": chip_id, **boundary_values}
                )
                if not args.skip_sample_pdfs:
                    from evaluation.sample_reports import (
                        chip_metric_values,
                        save_instance_sample_pdf,
                    )

                    save_instance_sample_pdf(
                        output_dir=output_dir,
                        country=country,
                        chip_id=chip_id,
                        image=sample["image"].numpy(),
                        semantic=semantic_target,
                        instances=instance_mask,
                        prediction_masks=masks,
                        confidences=confidences,
                        metric_values=chip_metric_values(
                            arrays,
                            boundary_values,
                            pixel_record,
                            chip_geometry,
                            chip_repair_rows,
                            confidences=confidences,
                            topology=topology_values,
                        ),
                    )

    if not object_rows:
        raise SystemExit("No downloaded countries with usable test samples were found.")
    object_summary = aggregate_object_rows(object_rows)
    boundary_summary = metrics.aggregate_macro_rows(boundary_rows)
    ap_frame = ap_summary(detection_rows, gt_counts)
    geometry_summary = metrics.aggregate_geometry_rows(geometry_rows)
    geometry_population_summary = metrics.aggregate_geometry_population(
        geometry_population_rows
    )
    repair_summary = metrics.aggregate_repair_rows(repair_rows)
    pixel_summary, pixel_confusion = metrics.aggregate_pixel_records(pixel_records)
    confidence_summary = confidence.confidence_reliability_summary(
        detection_rows, ("native_confidence",)
    )
    risk_coverage, risk_summary = confidence.risk_coverage_rows(
        detection_rows,
        gt_counts,
        ("native_confidence", "area_pixels"),
    )
    topology_summary = metrics.aggregate_macro_rows(topology_rows)

    object_summary.to_csv(output_dir / "object_summary_by_threshold.csv", index=False)
    ap_frame.to_csv(output_dir / "average_precision_summary.csv", index=False)
    pd.DataFrame(gt_rows).to_csv(
        output_dir / "ground_truth_field_matches_iou50.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "prediction_field_matches_iou50.csv", index=False
    )
    pd.DataFrame(boundary_rows).to_csv(
        output_dir / "boundary_metrics_by_chip.csv", index=False
    )
    boundary_summary.to_csv(output_dir / "boundary_summary.csv", index=False)
    pd.DataFrame(geometry_rows).to_csv(
        output_dir / "geometry_by_matched_field.csv", index=False
    )
    geometry_summary.to_csv(output_dir / "geometry_summary.csv", index=False)
    pd.DataFrame(geometry_population_rows).to_csv(
        output_dir / "geometry_population.csv", index=False
    )
    geometry_population_summary.to_csv(
        output_dir / "geometry_population_summary.csv", index=False
    )
    pd.DataFrame(geometry_exclusions).to_csv(
        output_dir / "geometry_border_exclusions.csv", index=False
    )
    pd.DataFrame(repair_rows).to_csv(
        output_dir / "merge_repair_distance.csv", index=False
    )
    repair_summary.to_csv(output_dir / "merge_repair_summary.csv", index=False)
    pixel_summary.to_csv(output_dir / "pixel_metrics.csv", index=False)
    pixel_confusion.to_csv(output_dir / "pixel_confusion.csv", index=False)
    confidence_frame = pd.DataFrame(detection_rows)
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
    pd.DataFrame(topology_rows).to_csv(
        output_dir / "topology_metrics_by_chip.csv", index=False
    )
    topology_summary.to_csv(output_dir / "topology_summary.csv", index=False)

    settings = pd.DataFrame(
        [
            {
                "model": args.model,
                "model_path": (
                    str(model_path) if model_path else "registered_url_or_cache"
                ),
                "window": args.window,
                "resize_factor": args.resize_factor,
                "max_detections": args.max_detections,
                "confidence_threshold": args.conf_threshold,
                "nms_iou_threshold": args.nms_iou_threshold,
                "association_threshold": args.association_threshold,
                "association_alpha_sweep": "0.05,0.10,0.20,0.30",
                "metres_per_pixel": args.metres_per_pixel,
                "geometry_epsilon_px": args.geometry_epsilon_px,
                "sample_pdfs": not args.skip_sample_pdfs,
                "confidence_scores": "native_confidence,area_baseline",
                "temperature_scaling": "not_applicable_no_pixel_logits",
            }
        ]
    )
    settings.to_csv(output_dir / "run_settings.csv", index=False)
    if not args.skip_plots:
        from evaluation.plots import create_plots

        create_plots(output_dir)
    print("\nObject summary")
    print(object_summary.to_string(index=False))
    print("\nAverage precision")
    print(ap_frame.to_string(index=False))
    print(f"\nResults saved in: {output_dir}")


if __name__ == "__main__":
    main()
