"""Per-chip PDF diagnostics for semantic and instance evaluators."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage

from evaluation import metrics
from evaluation.metrics import N4, N8


SEMANTIC_COLORS = ListedColormap(("#202020", "#4daf4a", "#e41a1c", "#bdbdbd"))
SEMANTIC_NORM = BoundaryNorm((-0.5, 0.5, 1.5, 2.5, 3.5), 4)


def _rgb(image: np.ndarray) -> np.ndarray:
    rgb = np.moveaxis(image[:3].astype(float), 0, -1)
    rgb = np.clip(rgb / 3000.0, 0, 1)
    return np.power(rgb, 0.8)


def _instance_display(labels: np.ndarray) -> np.ndarray:
    values = np.unique(labels)
    values = values[values != 0]
    compact = np.zeros(labels.shape, dtype=np.int32)
    if len(values):
        positive = labels != 0
        compact[positive] = np.searchsorted(values, labels[positive]) + 1
    return compact


def _predicted_mask_labels(masks: np.ndarray) -> np.ndarray:
    labels = np.zeros(masks.shape[1:], dtype=np.int32)
    for index, mask in enumerate(masks, start=1):
        labels[mask] = index
    return labels


def _show_semantic(axis, values: np.ndarray, title: str) -> None:
    axis.imshow(values, cmap=SEMANTIC_COLORS, norm=SEMANTIC_NORM)
    axis.set_title(title)
    axis.axis("off")


def _show_instances(axis, labels: np.ndarray, title: str) -> None:
    display = _instance_display(labels)
    masked = np.ma.masked_where(display == 0, display)
    axis.imshow(masked, cmap="nipy_spectral", interpolation="nearest")
    axis.set_facecolor("#202020")
    axis.set_title(title)
    axis.axis("off")


def _prepare_path(output_dir: Path, country: str, chip_id: str) -> Path:
    country_dir = output_dir / "sample_pdfs" / country
    country_dir.mkdir(parents=True, exist_ok=True)
    return country_dir / f"{chip_id}.pdf"


def chip_metric_values(
    arrays: metrics.ObjectArrays,
    boundary: dict,
    pixel_record: dict,
    geometry_rows: list[dict],
    repair_rows: list[dict],
    closing_rows: list[dict] | None = None,
    probability_rows: list[dict] | None = None,
    confidences: np.ndarray | None = None,
    topology: dict | None = None,
    probability_record: dict | None = None,
) -> dict[str, object]:
    """Flatten every available per-chip metric for PDF table pages."""
    values: dict[str, object] = {}
    for row in metrics.object_chip_rows("sample", "sample", arrays):
        threshold = row["iou_threshold"]
        suffix = f"iou{int(threshold * 100):02d}"
        tp, fp, fn = row["true_positives"], row["false_positives"], row["false_negatives"]
        precision = metrics.safe_divide(tp, tp + fp)
        recall = metrics.safe_divide(tp, tp + fn)
        denominator = tp + 0.5 * fp + 0.5 * fn
        matched = np.asarray(row["matched_ious"], dtype=float)
        values.update(
            {
                f"objects/{suffix}/TP": tp,
                f"objects/{suffix}/FP": fp,
                f"objects/{suffix}/FN": fn,
                f"objects/{suffix}/precision": precision,
                f"objects/{suffix}/recall": recall,
                f"objects/{suffix}/F1": metrics.safe_divide(
                    2 * precision * recall, precision + recall
                ),
                f"objects/{suffix}/mean_matched_IoU": (
                    float(matched.mean()) if len(matched) else np.nan
                ),
                f"objects/{suffix}/SQ": metrics.safe_divide(matched.sum(), tp),
                f"objects/{suffix}/RQ": metrics.safe_divide(tp, denominator),
                f"objects/{suffix}/PQ": metrics.safe_divide(matched.sum(), denominator),
            }
        )
        if threshold == metrics.PRIMARY_THRESHOLD:
            for key, value in row.items():
                if key.startswith(("split_", "merged_", "fields_lost_")):
                    values[f"topology/{key}"] = value
    values["objects/ground_truth_count"] = len(arrays.gt_ids)
    values["objects/prediction_count"] = len(arrays.prediction_ids)
    values["objects/count_error"] = len(arrays.prediction_ids) - len(arrays.gt_ids)
    for key, value in boundary.items():
        values[f"boundary/{key}"] = value
    pixel_metrics, _ = metrics.aggregate_pixel_records([pixel_record])
    for row in pixel_metrics.to_dict("records"):
        for key in ("iou", "precision", "recall", "f1", "support_pixels"):
            values[f"pixels/{row['view']}/{row['class']}/{key}"] = row[key]
    if geometry_rows:
        for key in (
            "perimeter_ratio",
            "turning_distance_radians",
            "delta_area",
            "delta_perimeter",
            "delta_polsby_popper",
            "delta_solidity",
            "delta_rectangularity",
            "delta_elongation",
            "delta_holes",
            "delta_vertices",
            "delta_right_angle_fraction",
        ):
            data = np.asarray([row[key] for row in geometry_rows], dtype=float)
            values[f"geometry/median_{key}"] = float(np.nanmedian(data))
        values["geometry/valid_matched_pairs"] = len(geometry_rows)
    repair_costs = np.asarray(
        [row["repair_upper_bound_px"] for row in repair_rows], dtype=float
    )
    repair_costs = repair_costs[np.isfinite(repair_costs)]
    values["repair/breached_field_pairs"] = len(repair_rows)
    values["repair/valid_costs"] = len(repair_costs)
    values["repair/all_joint_multiway_exact"] = bool(
        repair_rows
        and all(row.get("joint_multiway_cut_exact", False) for row in repair_rows)
    )
    for radius in (1, 2, 3):
        values[f"repair/fraction_le_{radius}px"] = (
            float((repair_costs <= radius).mean()) if len(repair_costs) else np.nan
        )
    for key in (
        "breach_confidence_max",
        "breach_confidence_min",
        "breach_confidence_mean",
        "breach_confidence_p10",
    ):
        data = np.asarray(
            [row.get(key, np.nan) for row in repair_rows], dtype=float
        )
        data = data[np.isfinite(data)]
        values[f"repair/{key}_median"] = (
            float(np.median(data)) if len(data) else np.nan
        )
    for key in (
        "exact_pairwise_min_cut_pixels",
        "confidence_weighted_min_cut_pixels",
        "confidence_weighted_min_cut_cost",
    ):
        data = np.asarray(
            [row.get(key, np.nan) for row in repair_rows], dtype=float
        )
        data = data[np.isfinite(data)]
        values[f"repair/{key}_median"] = (
            float(np.median(data)) if len(data) else np.nan
        )
    for row in closing_rows or []:
        radius = int(row["closing_radius_px"])
        for key in ("true_positives", "false_positives", "false_negatives", "object_f1"):
            values[f"closure/radius_{radius}px/{key}"] = row[key]
    for row in probability_rows or []:
        threshold = row["boundary_probability_threshold"]
        suffix = f"p2_{threshold:.2f}"
        for key in (
            "panoptic_quality_pq",
            "split_fields_a10",
            "merged_predictions_a10",
            "fields_lost_to_merge_a10",
        ):
            values[f"boundary_sweep/{suffix}/{key}"] = row[key]
    if confidences is not None:
        values["confidence/predictions"] = len(confidences)
        for name, function in (
            ("minimum", np.min),
            ("median", np.median),
            ("mean", np.mean),
            ("maximum", np.max),
        ):
            values[f"confidence/{name}"] = (
                float(function(confidences)) if len(confidences) else np.nan
            )
    for key, value in (topology or {}).items():
        values[f"topology/{key}"] = value
    if probability_record is not None:
        summary, _, distance = metrics_for_probability_record(
            probability_record
        )
        for key, value in summary.items():
            values[f"probabilistic/{key}"] = value
        for row in distance:
            for key in (
                "pixels",
                "adaptive_ece",
                "mean_entropy",
                "mean_entropy_incorrect",
                "mean_entropy_correct",
            ):
                values[
                    f"probabilistic/{row['distance_stratum']}/{key}"
                ] = row[key]
    return values


def metrics_for_probability_record(
    probability_record: dict,
) -> tuple[dict, list[dict], list[dict]]:
    from evaluation.confidence import aggregate_probability_records

    summary, bins, distance = aggregate_probability_records(
        [probability_record]
    )
    country = probability_record["country"]
    summary_row = summary[summary.scope == country].iloc[0].to_dict()
    bin_rows = bins[bins.scope == country].to_dict("records")
    distance_rows = distance[distance.scope == country].to_dict("records")
    return summary_row, bin_rows, distance_rows


def _format_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return "—" if not np.isfinite(value) else f"{value:.6g}"
    return str(value)


def _write_report(
    path: Path,
    visual_figure: plt.Figure,
    title: str,
    metric_values: dict[str, object],
) -> None:
    with PdfPages(path) as pdf:
        pdf.savefig(visual_figure, bbox_inches="tight", facecolor="white")
        plt.close(visual_figure)
        items = sorted(metric_values.items())
        per_page = 32
        for start in range(0, len(items), per_page):
            page_items = items[start : start + per_page]
            figure, axis = plt.subplots(figsize=(11.7, 8.3))
            axis.axis("off")
            table = axis.table(
                cellText=[[key, _format_value(value)] for key, value in page_items],
                colLabels=("Metric", "Value"),
                colWidths=(0.78, 0.18),
                cellLoc="left",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.25)
            axis.set_title(
                f"{title} — per-chip metrics ({start + 1}–{start + len(page_items)})",
                fontsize=14,
                fontweight="bold",
                pad=16,
            )
            pdf.savefig(figure, bbox_inches="tight", facecolor="white")
            plt.close(figure)


def save_semantic_sample_pdf(
    output_dir: Path,
    country: str,
    chip_id: str,
    image: np.ndarray,
    semantic: np.ndarray,
    instances: np.ndarray,
    prediction: np.ndarray,
    probabilities: np.ndarray,
    metric_values: dict[str, object],
) -> Path:
    """Save a six-panel semantic-model report named after the image chip."""
    valid = semantic != 3
    gt_instances = np.where(semantic == 1, instances, 0)
    predicted_instances, _ = ndimage.label(prediction == 1, N4)
    predicted_instances *= valid
    boundary_probability = probabilities[2]
    confidence = probabilities.max(axis=0)
    entropy = -(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1))
    ).sum(axis=0)
    distance = ndimage.distance_transform_edt(~((semantic == 2) & valid))
    strata = np.digitize(distance, (1, 2, 3, 5, 10), right=False)
    expected_entropy = np.zeros(entropy.shape, dtype=float)
    for stratum in range(6):
        selected = (strata == stratum) & valid
        if selected.any():
            expected_entropy[selected] = entropy[selected].mean()
    entropy_residual = np.where(valid, entropy - expected_entropy, np.nan)
    path = _prepare_path(output_dir, country, chip_id)
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(_rgb(image))
    axes[0, 0].set_title("Sentinel-2 RGB")
    axes[0, 0].axis("off")
    _show_semantic(axes[0, 1], semantic, "Reference semantic")
    _show_semantic(axes[0, 2], prediction, "Predicted semantic")
    confidence_image = axes[0, 3].imshow(
        np.where(valid, confidence, np.nan), vmin=0, vmax=1, cmap="viridis"
    )
    axes[0, 3].set_title("Maximum class confidence")
    axes[0, 3].axis("off")
    figure.colorbar(confidence_image, ax=axes[0, 3], fraction=0.046, pad=0.04)
    _show_instances(axes[1, 0], gt_instances, "Reference field instances")
    _show_instances(axes[1, 1], predicted_instances, "Predicted field instances")
    probability = axes[1, 2].imshow(
        np.where(valid, boundary_probability, np.nan),
        vmin=0,
        vmax=1,
        cmap="magma",
    )
    axes[1, 2].set_title("Predicted boundary probability")
    axes[1, 2].axis("off")
    figure.colorbar(probability, ax=axes[1, 2], fraction=0.046, pad=0.04)
    residual_limit = np.nanmax(np.abs(entropy_residual))
    residual_limit = (
        residual_limit
        if np.isfinite(residual_limit) and residual_limit > 0
        else 1.0
    )
    residual = axes[1, 3].imshow(
        entropy_residual,
        vmin=-residual_limit,
        vmax=residual_limit,
        cmap="coolwarm",
    )
    axes[1, 3].set_title("Entropy residual given boundary distance")
    axes[1, 3].axis("off")
    figure.colorbar(residual, ax=axes[1, 3], fraction=0.046, pad=0.04)
    figure.suptitle(f"{country} / {chip_id}", fontsize=15, fontweight="bold")
    figure.tight_layout()
    _write_report(path, figure, f"{country} / {chip_id}", metric_values)
    return path


def save_instance_sample_pdf(
    output_dir: Path,
    country: str,
    chip_id: str,
    image: np.ndarray,
    semantic: np.ndarray,
    instances: np.ndarray,
    prediction_masks: np.ndarray,
    confidences: np.ndarray,
    metric_values: dict[str, object],
) -> Path:
    """Save a six-panel Delineate Anything report named after the image chip."""
    valid = semantic != 3
    gt_instances = np.where(semantic == 1, instances, 0)
    prediction_labels = _predicted_mask_labels(prediction_masks)
    prediction_extent = (
        prediction_masks.any(0)
        if len(prediction_masks)
        else np.zeros(semantic.shape, bool)
    )
    target_boundary = (semantic == 2) & valid
    predicted_boundary = np.zeros(semantic.shape, dtype=bool)
    for mask in prediction_masks:
        predicted_boundary |= mask & ~ndimage.binary_erosion(mask, N8)
    path = _prepare_path(output_dir, country, chip_id)
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    rgb = _rgb(image)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Sentinel-2 RGB")
    axes[0, 0].axis("off")
    _show_semantic(axes[0, 1], semantic, "Reference semantic")
    _show_instances(axes[0, 2], gt_instances, "Reference field instances")
    _show_instances(
        axes[1, 0],
        prediction_labels,
        f"Predicted instances (n={len(prediction_masks)})",
    )
    axes[1, 1].imshow(rgb)
    if target_boundary.any():
        axes[1, 1].contour(
            target_boundary, levels=(0.5,), colors=("yellow",), linewidths=0.7
        )
    if predicted_boundary.any():
        axes[1, 1].contour(
            predicted_boundary, levels=(0.5,), colors=("red",), linewidths=0.7
        )
    axes[1, 1].set_title("Boundaries: reference yellow, prediction red")
    axes[1, 1].axis("off")
    axes[1, 2].imshow(prediction_extent, cmap="Greens", vmin=0, vmax=1)
    mean_confidence = float(confidences.mean()) if len(confidences) else float("nan")
    axes[1, 2].set_title(f"Predicted extent; mean confidence={mean_confidence:.3f}")
    axes[1, 2].axis("off")
    figure.suptitle(f"{country} / {chip_id}", fontsize=15, fontweight="bold")
    figure.tight_layout()
    _write_report(path, figure, f"{country} / {chip_id}", metric_values)
    return path
