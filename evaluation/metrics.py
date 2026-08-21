"""Canonical FTW object, boundary, geometry, and closure metrics.

All evaluators use these functions so matching and aggregation definitions stay
identical across semantic and native instance-segmentation models.
"""

from dataclasses import dataclass
import math
from collections import deque

import numpy as np
import pandas as pd
from rasterio.features import shapes
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
from shapely.geometry import Point, shape

from ftw_tools.settings import FULL_DATA_COUNTRIES


DEFAULT_THRESHOLDS = (0.25, 0.50, 0.75)
ASSOCIATION_THRESHOLDS = (0.05, 0.10, 0.20, 0.30)
PRIMARY_THRESHOLD = 0.50
N4 = ndimage.generate_binary_structure(2, 1)
N8 = ndimage.generate_binary_structure(2, 2)


@dataclass
class ObjectArrays:
    gt_ids: np.ndarray
    prediction_ids: np.ndarray
    intersections: np.ndarray
    gt_areas: np.ndarray
    prediction_areas: np.ndarray
    ious: np.ndarray


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def arrays_from_labels(gt: np.ndarray, pred_lbl: np.ndarray) -> ObjectArrays:
    """Build intersections in one histogram pass, never by looping over pairs."""
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids != 0]
    prediction_ids = np.unique(pred_lbl)
    prediction_ids = prediction_ids[prediction_ids != 0]

    gt_compact = np.zeros_like(gt, dtype=np.int64)
    gt_pos = gt != 0
    gt_compact[gt_pos] = np.searchsorted(gt_ids, gt[gt_pos]) + 1
    pred_compact = np.zeros_like(pred_lbl, dtype=np.int64)
    pred_pos = pred_lbl != 0
    pred_compact[pred_pos] = (
        np.searchsorted(prediction_ids, pred_lbl[pred_pos]) + 1
    )

    m, n = len(gt_ids), len(prediction_ids)
    stride = n + 1
    intersections = np.bincount(
        gt_compact.ravel() * stride + pred_compact.ravel(),
        minlength=(m + 1) * stride,
    ).reshape(m + 1, stride)[1:, 1:]
    gt_areas = np.bincount(gt_compact.ravel(), minlength=m + 1)[1:]
    prediction_areas = np.bincount(pred_compact.ravel(), minlength=n + 1)[1:]
    unions = gt_areas[:, None] + prediction_areas[None, :] - intersections
    ious = np.divide(
        intersections,
        unions,
        out=np.zeros(intersections.shape, dtype=float),
        where=unions != 0,
    )
    return ObjectArrays(
        gt_ids, prediction_ids, intersections, gt_areas, prediction_areas, ious
    )


def semantic_object_arrays(
    semantic: np.ndarray, instances: np.ndarray, prediction: np.ndarray
) -> tuple[ObjectArrays, np.ndarray, np.ndarray]:
    """Construct spec objects; label predictions before applying valid mask."""
    valid = semantic != 3
    gt = instances.copy()
    gt[semantic != 1] = 0
    pred_lbl, _ = ndimage.label(prediction == 1, N4)
    pred_lbl = pred_lbl * valid
    return arrays_from_labels(gt, pred_lbl), gt, pred_lbl


def mask_object_arrays(
    semantic: np.ndarray, instances: np.ndarray, masks: np.ndarray
) -> ObjectArrays:
    """Build matrices for native, possibly overlapping predicted masks."""
    valid = semantic != 3
    gt = instances.copy()
    gt[semantic != 1] = 0
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids != 0]
    masks = masks.astype(bool) & valid[None, ...]
    prediction_ids = np.arange(1, len(masks) + 1, dtype=np.int64)
    gt_areas = np.asarray([(gt == value).sum() for value in gt_ids], dtype=np.int64)
    prediction_areas = (
        masks.reshape(len(masks), -1).sum(1)
        if len(masks)
        else np.zeros(0, dtype=np.int64)
    )
    intersections = np.zeros((len(gt_ids), len(masks)), dtype=np.int64)
    for index, value in enumerate(gt_ids):
        if len(masks):
            intersections[index] = np.count_nonzero(
                masks & (gt == value)[None, ...], axis=(1, 2)
            )
    unions = gt_areas[:, None] + prediction_areas[None, :] - intersections
    ious = np.divide(
        intersections,
        unions,
        out=np.zeros(intersections.shape, dtype=float),
        where=unions != 0,
    )
    return ObjectArrays(
        gt_ids, prediction_ids, intersections, gt_areas, prediction_areas, ious
    )


def match_at_threshold(
    arrays: ObjectArrays, threshold: float
) -> list[tuple[int, int]]:
    """Maximum-cardinality, then maximum-valid-IoU assignment."""
    ious = arrays.ious
    if ious.size == 0:
        return []
    valid = ious >= threshold
    scores = valid * (min(ious.shape) + 1.0) + ious * valid
    gt_indices, prediction_indices = linear_sum_assignment(scores, maximize=True)
    return [
        (int(i), int(j))
        for i, j in zip(gt_indices, prediction_indices)
        if ious[i, j] >= threshold
    ]


def overlap_relations(arrays: ObjectArrays, alpha: float) -> tuple[int, int, int]:
    if arrays.intersections.size == 0:
        return 0, 0, 0
    minimum = np.minimum(
        arrays.gt_areas[:, None], arrays.prediction_areas[None, :]
    )
    overlap = np.divide(
        arrays.intersections,
        minimum,
        out=np.zeros(arrays.intersections.shape, dtype=float),
        where=minimum != 0,
    )
    associated = overlap >= alpha
    merged_columns = associated.sum(axis=0) >= 2
    return (
        int((associated.sum(axis=1) >= 2).sum()),
        int(merged_columns.sum()),
        int(associated[:, merged_columns].any(axis=1).sum()),
    )


def object_chip_rows(country: str, chip_id: str, arrays: ObjectArrays) -> list[dict]:
    relation_sweep = {
        alpha: overlap_relations(arrays, alpha)
        for alpha in ASSOCIATION_THRESHOLDS
    }
    rows = []
    for threshold in DEFAULT_THRESHOLDS:
        matches = match_at_threshold(arrays, threshold)
        row = {
            "country": country,
            "chip_id": chip_id,
            "iou_threshold": threshold,
            "true_positives": len(matches),
            "false_positives": len(arrays.prediction_ids) - len(matches),
            "false_negatives": len(arrays.gt_ids) - len(matches),
            "matched_ious": [float(arrays.ious[i, j]) for i, j in matches],
            "gt_count": len(arrays.gt_ids),
            "prediction_count": len(arrays.prediction_ids),
        }
        for alpha, (splits, merges, lost) in relation_sweep.items():
            suffix = f"a{int(alpha * 100):02d}"
            row[f"split_fields_{suffix}"] = splits
            row[f"merged_predictions_{suffix}"] = merges
            row[f"fields_lost_to_merge_{suffix}"] = lost
        rows.append(row)
    return rows


def object_metric_row(scope: str, threshold: float, rows: pd.DataFrame) -> dict:
    tp = int(rows["true_positives"].sum())
    fp = int(rows["false_positives"].sum())
    fn = int(rows["false_negatives"].sum())
    matched = np.asarray([x for values in rows["matched_ious"] for x in values])
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    denominator = tp + 0.5 * fp + 0.5 * fn
    result = {
        "scope": scope,
        "iou_threshold": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "object_precision": precision,
        "object_recall": recall,
        "object_f1": safe_divide(2 * precision * recall, precision + recall),
        "mean_matched_iou": float(matched.mean()) if len(matched) else np.nan,
        "median_matched_iou": float(np.median(matched)) if len(matched) else np.nan,
        "segmentation_quality_sq": safe_divide(float(matched.sum()), tp),
        "recognition_quality_rq": safe_divide(tp, denominator),
        "panoptic_quality_pq": safe_divide(float(matched.sum()), denominator),
        "ground_truth_objects": int(rows["gt_count"].sum()),
        "predicted_objects": int(rows["prediction_count"].sum()),
        "presence_only_precision_warning": bool(
            any(country not in FULL_DATA_COUNTRIES for country in rows.country.unique())
        ),
    }
    result["object_count_error"] = (
        result["predicted_objects"] - result["ground_truth_objects"]
    )
    for alpha in ASSOCIATION_THRESHOLDS:
        suffix = f"a{int(alpha * 100):02d}"
        for prefix in ("split_fields", "merged_predictions", "fields_lost_to_merge"):
            result[f"{prefix}_{suffix}"] = int(rows[f"{prefix}_{suffix}"].sum())
    return result


def aggregate_object_rows(chip_rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(chip_rows)
    output = []
    for scope in sorted(frame.country.unique()) + ["all_countries"]:
        scoped = frame if scope == "all_countries" else frame[frame.country == scope]
        for threshold in DEFAULT_THRESHOLDS:
            output.append(
                object_metric_row(
                    scope, threshold, scoped[scoped.iou_threshold == threshold]
                )
            )
    return pd.DataFrame(output)


def binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    tp = int((target & prediction).sum())
    fp = int((~target & prediction).sum())
    fn = int((target & ~prediction).sum())
    precision, recall = safe_divide(tp, tp + fp), safe_divide(tp, tp + fn)
    return {
        "iou": safe_divide(tp, tp + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall),
        "target_pixels": int(target.sum()),
        "predicted_pixels": int(prediction.sum()),
    }


def boundary_metrics(
    target: np.ndarray, prediction: np.ndarray, metres_per_pixel: float = 10.0
) -> dict:
    """Chebyshev-tolerant and Euclidean-distance boundary metrics."""
    row = {f"exact_{key}": value for key, value in binary_metrics(target, prediction).items()}
    for tolerance in (1, 2):
        target_dilated = ndimage.binary_dilation(target, N8, iterations=tolerance)
        prediction_dilated = ndimage.binary_dilation(
            prediction, N8, iterations=tolerance
        )
        precision = safe_divide((prediction & target_dilated).sum(), prediction.sum())
        recall = safe_divide((target & prediction_dilated).sum(), target.sum())
        row[f"tolerance_{tolerance}px_precision"] = precision
        row[f"tolerance_{tolerance}px_recall"] = recall
        row[f"tolerance_{tolerance}px_f1"] = safe_divide(
            2 * precision * recall, precision + recall
        )
    if target.any() and prediction.any():
        d_prediction = ndimage.distance_transform_edt(~target)[prediction]
        d_target = ndimage.distance_transform_edt(~prediction)[target]
        combined = np.concatenate([d_prediction, d_target])
        row["asbd_pooled_px"] = float(combined.mean())
        row["asbd_balanced_px"] = float(
            0.5 * (d_prediction.mean() + d_target.mean())
        )
        row["hausdorff_95_px"] = float(np.percentile(combined, 95))
    else:
        row.update(
            asbd_pooled_px=np.nan, asbd_balanced_px=np.nan, hausdorff_95_px=np.nan
        )
    for key in ("asbd_pooled", "asbd_balanced", "hausdorff_95"):
        row[f"{key}_m"] = row[f"{key}_px"] * metres_per_pixel
    return row


def aggregate_macro_rows(chip_rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(chip_rows)
    metrics = [c for c in frame if c not in {"country", "chip_id"}]
    output = []
    for scope in sorted(frame.country.unique()) + ["all_countries"]:
        scoped = frame if scope == "all_countries" else frame[frame.country == scope]
        row = {"scope": scope, "chips": len(scoped)}
        for metric in metrics:
            numeric = pd.to_numeric(scoped[metric], errors="coerce")
            row[metric] = float(numeric.mean())
            row[f"{metric}_valid_count"] = int(numeric.notna().sum())
        output.append(row)
    return pd.DataFrame(output)


def touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def polygon_of(mask: np.ndarray, epsilon: float = 1.0):
    geometries = [shape(geometry) for geometry, value in shapes(mask.astype("uint8")) if value == 1]
    if not geometries:
        return None
    geometry = max(geometries, key=lambda item: item.area).simplify(epsilon)
    if geometry.geom_type == "MultiPolygon":
        geometry = max(geometry.geoms, key=lambda item: item.area)
    return geometry if geometry.geom_type == "Polygon" and not geometry.is_empty else None


def right_angle_fraction(geometry, tolerance_degrees: float = 10.0) -> float:
    coordinates = np.asarray(geometry.exterior.coords[:-1])
    if len(coordinates) < 3:
        return float("nan")
    vectors = np.roll(coordinates, -1, axis=0) - coordinates
    next_vectors = np.roll(vectors, -1, axis=0)
    denominator = np.linalg.norm(vectors, axis=1) * np.linalg.norm(next_vectors, axis=1)
    cosine = np.divide(
        (vectors * next_vectors).sum(axis=1),
        denominator,
        out=np.full(len(vectors), np.nan),
        where=denominator != 0,
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return float(np.nanmean(np.abs(angles - 90) < tolerance_degrees))


def geometry_descriptors(geometry) -> dict:
    rectangle = geometry.minimum_rotated_rectangle
    coordinates = list(rectangle.exterior.coords)
    edges = sorted(Point(coordinates[i]).distance(Point(coordinates[i + 1])) for i in range(4))
    short, long = edges[0], edges[-1]
    return {
        "area": geometry.area,
        "perimeter": geometry.length,
        "polsby_popper": safe_divide(4 * math.pi * geometry.area, geometry.length**2),
        "solidity": safe_divide(geometry.area, geometry.convex_hull.area),
        "rectangularity": safe_divide(geometry.area, rectangle.area),
        "elongation": safe_divide(short, long),
        "holes": len(geometry.interiors),
        "vertices": len(geometry.exterior.coords) - 1,
        "right_angle_fraction": right_angle_fraction(geometry),
    }


def geometry_pair_rows(
    country: str,
    chip_id: str,
    arrays: ObjectArrays,
    gt_masks: list[np.ndarray],
    prediction_masks: list[np.ndarray],
    epsilon: float = 1.0,
) -> tuple[list[dict], dict]:
    rows, excluded = [], {"gt": 0, "prediction": 0, "pairs": 0}
    for gt_index, prediction_index in match_at_threshold(arrays, PRIMARY_THRESHOLD):
        gt_mask, prediction_mask = gt_masks[gt_index], prediction_masks[prediction_index]
        gt_border, prediction_border = touches_border(gt_mask), touches_border(prediction_mask)
        excluded["gt"] += int(gt_border)
        excluded["prediction"] += int(prediction_border)
        if gt_border or prediction_border:
            excluded["pairs"] += 1
            continue
        gt_geometry = polygon_of(gt_mask, epsilon)
        prediction_geometry = polygon_of(prediction_mask, epsilon)
        if gt_geometry is None or prediction_geometry is None:
            continue
        gt_desc = geometry_descriptors(gt_geometry)
        prediction_desc = geometry_descriptors(prediction_geometry)
        row = {
            "country": country,
            "chip_id": chip_id,
            "gt_field_id": int(arrays.gt_ids[gt_index]),
            "prediction_id": int(arrays.prediction_ids[prediction_index]),
            "iou": float(arrays.ious[gt_index, prediction_index]),
            "perimeter_ratio": safe_divide(
                prediction_desc["perimeter"], gt_desc["perimeter"]
            ),
        }
        for key in gt_desc:
            row[f"gt_{key}"] = gt_desc[key]
            row[f"prediction_{key}"] = prediction_desc[key]
            row[f"delta_{key}"] = prediction_desc[key] - gt_desc[key]
        rows.append(row)
    return rows, excluded


def aggregate_geometry_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    descriptors = (
        "area", "perimeter", "polsby_popper", "solidity", "rectangularity",
        "elongation", "holes", "vertices", "right_angle_fraction"
    )
    output = []
    for scope in sorted(frame.country.unique()) + ["all_countries"]:
        scoped = frame if scope == "all_countries" else frame[frame.country == scope]
        row = {"scope": scope, "matched_pairs": len(scoped)}
        row["median_perimeter_ratio"] = float(scoped.perimeter_ratio.median())
        row["predicted_interior_rings_per_100_fields"] = float(
            100 * scoped.prediction_holes.sum() / len(scoped)
        )
        for descriptor in descriptors:
            row[f"median_delta_{descriptor}"] = float(scoped[f"delta_{descriptor}"].median())
            gt_values = scoped[f"gt_{descriptor}"].dropna()
            prediction_values = scoped[f"prediction_{descriptor}"].dropna()
            row[f"wasserstein_{descriptor}"] = (
                float(wasserstein_distance(gt_values, prediction_values))
                if len(gt_values) and len(prediction_values)
                else np.nan
            )
        row["gt_right_angle_fraction"] = float(scoped.gt_right_angle_fraction.mean())
        row["prediction_right_angle_fraction"] = float(
            scoped.prediction_right_angle_fraction.mean()
        )
        output.append(row)
    return pd.DataFrame(output)


def geometry_population_rows(
    country: str,
    chip_id: str,
    gt_masks: list[np.ndarray],
    prediction_masks: list[np.ndarray],
    epsilon: float = 1.0,
) -> list[dict]:
    rows = []
    for kind, masks in (("ground_truth", gt_masks), ("prediction", prediction_masks)):
        for object_index, mask in enumerate(masks, start=1):
            if touches_border(mask):
                continue
            geometry = polygon_of(mask, epsilon)
            if geometry is not None:
                rows.append({
                    "country": country,
                    "chip_id": chip_id,
                    "kind": kind,
                    "object_index": object_index,
                    **geometry_descriptors(geometry),
                })
    return rows


def aggregate_geometry_population(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    descriptors = (
        "area", "perimeter", "polsby_popper", "solidity", "rectangularity",
        "elongation", "holes", "vertices", "right_angle_fraction"
    )
    output = []
    for scope in sorted(frame.country.unique()) + ["all_countries"]:
        scoped = frame if scope == "all_countries" else frame[frame.country == scope]
        gt = scoped[scoped.kind == "ground_truth"]
        prediction = scoped[scoped.kind == "prediction"]
        row = {
            "scope": scope,
            "ground_truth_objects": len(gt),
            "predicted_objects": len(prediction),
        }
        for descriptor in descriptors:
            gt_values = gt[descriptor].dropna()
            prediction_values = prediction[descriptor].dropna()
            row[f"wasserstein_{descriptor}"] = (
                float(wasserstein_distance(gt_values, prediction_values))
                if len(gt_values) and len(prediction_values)
                else np.nan
            )
        output.append(row)
    return pd.DataFrame(output)


def _grid_shortest_path(mask: np.ndarray, source: np.ndarray, target: np.ndarray) -> float:
    starts = [tuple(value) for value in np.argwhere(mask & source)]
    destinations = {tuple(value) for value in np.argwhere(mask & target)}
    if not starts or not destinations:
        return float("nan")
    queue = deque((value, 0) for value in starts)
    visited = set(starts)
    height, width = mask.shape
    while queue:
        (row, column), distance = queue.popleft()
        if (row, column) in destinations:
            return float(distance)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate = (row + dr, column + dc)
            if (
                0 <= candidate[0] < height
                and 0 <= candidate[1] < width
                and mask[candidate]
                and candidate not in visited
            ):
                visited.add(candidate)
                queue.append((candidate, distance + 1))
    return float("nan")


def merge_repair_rows(
    country: str,
    chip_id: str,
    arrays: ObjectArrays,
    gt_masks: list[np.ndarray],
    prediction_masks: list[np.ndarray],
    alpha: float = 0.10,
) -> list[dict]:
    """Greedy upper-bound repair distance for every pair inside a merged object."""
    if arrays.intersections.size == 0:
        return []
    minimum = np.minimum(arrays.gt_areas[:, None], arrays.prediction_areas[None, :])
    overlap = np.divide(
        arrays.intersections,
        minimum,
        out=np.zeros(arrays.intersections.shape, dtype=float),
        where=minimum != 0,
    )
    associated = overlap >= alpha
    rows = []
    for prediction_index in np.flatnonzero(associated.sum(axis=0) >= 2):
        gt_indices = np.flatnonzero(associated[:, prediction_index])
        for left_pos, left in enumerate(gt_indices):
            for right in gt_indices[left_pos + 1:]:
                cost = _grid_shortest_path(
                    prediction_masks[prediction_index], gt_masks[left], gt_masks[right]
                )
                rows.append({
                    "country": country,
                    "chip_id": chip_id,
                    "prediction_id": int(arrays.prediction_ids[prediction_index]),
                    "gt_field_id_a": int(arrays.gt_ids[left]),
                    "gt_field_id_b": int(arrays.gt_ids[right]),
                    "breached": True,
                    "repair_upper_bound_px": cost,
                })
    return rows


def aggregate_repair_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    output = []
    for scope in sorted(frame.country.unique()) + ["all_countries"]:
        scoped = frame if scope == "all_countries" else frame[frame.country == scope]
        costs = scoped.repair_upper_bound_px.dropna()
        output.append({
            "scope": scope,
            "merge_field_pairs": len(scoped),
            "valid_repair_costs": len(costs),
            "repair_le_1px_fraction": float((costs <= 1).mean()) if len(costs) else np.nan,
            "repair_le_2px_fraction": float((costs <= 2).mean()) if len(costs) else np.nan,
            "repair_le_3px_fraction": float((costs <= 3).mean()) if len(costs) else np.nan,
            "repair_cost_median_px": float(costs.median()) if len(costs) else np.nan,
        })
    return pd.DataFrame(output)


def closing_radius_rows(
    country: str,
    chip_id: str,
    semantic: np.ndarray,
    instances: np.ndarray,
    prediction: np.ndarray,
) -> list[dict]:
    valid = semantic != 3
    boundary = (prediction == 2) & valid
    foreground = (prediction != 0) & valid
    rows = []
    for radius in (0, 1, 2, 3):
        closed = (
            ndimage.binary_closing(boundary, N8, iterations=radius)
            if radius
            else boundary
        )
        labels, _ = ndimage.label(foreground & ~closed, N4)
        arrays = arrays_from_labels(
            np.where(semantic == 1, instances, 0), labels * valid
        )
        matches = match_at_threshold(arrays, PRIMARY_THRESHOLD)
        tp = len(matches)
        fp = len(arrays.prediction_ids) - tp
        fn = len(arrays.gt_ids) - tp
        precision, recall = safe_divide(tp, tp + fp), safe_divide(tp, tp + fn)
        rows.append(
            {
                "country": country,
                "chip_id": chip_id,
                "closing_radius_px": radius,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "object_f1": safe_divide(
                    2 * precision * recall, precision + recall
                ),
            }
        )
    return rows


def boundary_probability_sweep(
    country: str,
    chip_id: str,
    semantic: np.ndarray,
    instances: np.ndarray,
    prediction: np.ndarray,
    boundary_probability: np.ndarray,
) -> list[dict]:
    valid = semantic != 3
    gt = np.where(semantic == 1, instances, 0)
    rows = []
    for threshold in np.arange(0.10, 0.95, 0.05):
        boundary = boundary_probability >= threshold
        labels, _ = ndimage.label((~boundary) & (prediction != 0), N4)
        arrays = arrays_from_labels(gt, labels * valid)
        matches = match_at_threshold(arrays, PRIMARY_THRESHOLD)
        tp = len(matches)
        fp = len(arrays.prediction_ids) - tp
        fn = len(arrays.gt_ids) - tp
        denominator = tp + 0.5 * fp + 0.5 * fn
        split, merge, lost = overlap_relations(arrays, 0.10)
        matched_iou_sum = float(sum(arrays.ious[i, j] for i, j in matches))
        rows.append(
            {
                "country": country,
                "chip_id": chip_id,
                "boundary_probability_threshold": float(threshold),
                "panoptic_quality_pq": safe_divide(
                    matched_iou_sum, denominator
                ),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "matched_iou_sum": matched_iou_sum,
                "split_fields_a10": split,
                "merged_predictions_a10": merge,
                "fields_lost_to_merge_a10": lost,
            }
        )
    return rows


def aggregate_probability_sweep(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(rows)
    summaries = []
    for scope in sorted(frame.country.unique()) + ["all_countries"]:
        scoped = frame if scope == "all_countries" else frame[frame.country == scope]
        for threshold, group in scoped.groupby("boundary_probability_threshold"):
            tp = int(group.true_positives.sum())
            fp = int(group.false_positives.sum())
            fn = int(group.false_negatives.sum())
            denominator = tp + 0.5 * fp + 0.5 * fn
            summaries.append({
                "scope": scope,
                "boundary_probability_threshold": threshold,
                "panoptic_quality_pq": safe_divide(group.matched_iou_sum.sum(), denominator),
                "split_fields_a10": int(group.split_fields_a10.sum()),
                "merged_predictions_a10": int(group.merged_predictions_a10.sum()),
            })
    summary = pd.DataFrame(summaries)
    best_rows = []
    for _, scoped in summary.groupby("scope"):
        valid = scoped.dropna(subset=["panoptic_quality_pq"])
        if len(valid):
            best_rows.append(valid.loc[valid.panoptic_quality_pq.idxmax()])
    best = pd.DataFrame(best_rows)
    return summary, best
