"""Confidence, calibration, persistence, and selective-risk evaluation."""

import math

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import rankdata, spearmanr

from evaluation.metrics import N4, ObjectArrays, safe_divide
from ftw_tools.settings import FULL_DATA_COUNTRIES


CALIBRATION_FINE_BINS = 200
ADAPTIVE_BINS = 15
DISTANCE_LABELS = ("0px", "1px", "2px", "3-4px", "5-9px", "10px+")
PERSISTENCE_THRESHOLDS = np.arange(0.10, 0.95, 0.05)


def probability_chip_record(
    country: str,
    chip_id: str,
    semantic: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """Return exact score sums plus mergeable fine-bin calibration counts."""
    valid = semantic != 3
    target = semantic[valid].astype(np.int64)
    probability = probabilities[:, valid].T.astype(np.float64)
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    correct = prediction == target
    entropy = -(probability * np.log(np.clip(probability, 1e-12, 1))).sum(1)
    one_hot = np.eye(3, dtype=float)[target]
    boundary = (semantic == 2) & valid
    distance = ndimage.distance_transform_edt(~boundary)[valid]
    strata = np.digitize(distance, (1, 2, 3, 5, 10), right=False)
    bins = np.minimum(
        (confidence * CALIBRATION_FINE_BINS).astype(int),
        CALIBRATION_FINE_BINS - 1,
    )
    shape = (len(DISTANCE_LABELS), CALIBRATION_FINE_BINS)
    counts = np.zeros(shape, dtype=np.int64)
    confidence_sums = np.zeros(shape, dtype=np.float64)
    correct_sums = np.zeros(shape, dtype=np.float64)
    np.add.at(counts, (strata, bins), 1)
    np.add.at(confidence_sums, (strata, bins), confidence)
    np.add.at(correct_sums, (strata, bins), correct.astype(float))
    stratum_count = np.bincount(strata, minlength=len(DISTANCE_LABELS))
    entropy_sum = np.bincount(
        strata, weights=entropy, minlength=len(DISTANCE_LABELS)
    )
    error_entropy_sum = np.bincount(
        strata,
        weights=entropy * (~correct),
        minlength=len(DISTANCE_LABELS),
    )
    error_count = np.bincount(
        strata, weights=(~correct), minlength=len(DISTANCE_LABELS)
    )
    return {
        "country": country,
        "chip_id": chip_id,
        "pixels": len(target),
        "nll_sum": float(
            -np.log(
                np.clip(probability[np.arange(len(target)), target], 1e-12, 1)
            ).sum()
        ),
        "brier_sum": float(np.square(probability - one_hot).sum()),
        "entropy_sum": float(entropy.sum()),
        "correct": int(correct.sum()),
        "counts": counts,
        "confidence_sums": confidence_sums,
        "correct_sums": correct_sums,
        "stratum_count": stratum_count,
        "stratum_entropy_sum": entropy_sum,
        "stratum_error_entropy_sum": error_entropy_sum,
        "stratum_error_count": error_count,
    }


def _adaptive_rows(
    scope: str,
    stratum: str,
    counts: np.ndarray,
    confidence_sums: np.ndarray,
    correct_sums: np.ndarray,
) -> tuple[list[dict], float]:
    total = int(counts.sum())
    if not total:
        return [], float("nan")
    target_mass = max(1, math.ceil(total / ADAPTIVE_BINS))
    rows, current = [], [0, 0.0, 0.0, None, None]
    for fine_bin in range(CALIBRATION_FINE_BINS):
        count = int(counts[fine_bin])
        if not count:
            continue
        if current[3] is None:
            current[3] = fine_bin / CALIBRATION_FINE_BINS
        current[0] += count
        current[1] += confidence_sums[fine_bin]
        current[2] += correct_sums[fine_bin]
        current[4] = (fine_bin + 1) / CALIBRATION_FINE_BINS
        if current[0] >= target_mass and len(rows) < ADAPTIVE_BINS - 1:
            rows.append(current)
            current = [0, 0.0, 0.0, None, None]
    if current[0]:
        rows.append(current)
    output, ece = [], 0.0
    for index, (count, confidence_sum, correct_sum, lower, upper) in enumerate(rows):
        mean_confidence = confidence_sum / count
        accuracy = correct_sum / count
        ece += count / total * abs(accuracy - mean_confidence)
        output.append(
            {
                "scope": scope,
                "distance_stratum": stratum,
                "adaptive_bin": index,
                "confidence_lower": lower,
                "confidence_upper": upper,
                "pixels": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "calibration_gap": accuracy - mean_confidence,
            }
        )
    return output, float(ece)


def aggregate_probability_records(
    records: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries, calibration_rows, distance_rows = [], [], []
    countries = sorted({record["country"] for record in records})
    for scope in countries + ["all_countries"]:
        selected = records if scope == "all_countries" else [
            record for record in records if record["country"] == scope
        ]
        pixels = sum(record["pixels"] for record in selected)
        counts = np.sum([record["counts"] for record in selected], axis=0)
        confidence_sums = np.sum(
            [record["confidence_sums"] for record in selected], axis=0
        )
        correct_sums = np.sum(
            [record["correct_sums"] for record in selected], axis=0
        )
        stratum_counts = np.sum(
            [record["stratum_count"] for record in selected], axis=0
        )
        stratum_entropy = np.sum(
            [record["stratum_entropy_sum"] for record in selected], axis=0
        )
        stratum_error_entropy = np.sum(
            [record["stratum_error_entropy_sum"] for record in selected], axis=0
        )
        stratum_error_count = np.sum(
            [record["stratum_error_count"] for record in selected], axis=0
        )
        all_bins, overall_ece = _adaptive_rows(
            scope,
            "all",
            counts.sum(axis=0),
            confidence_sums.sum(axis=0),
            correct_sums.sum(axis=0),
        )
        calibration_rows.extend(all_bins)
        summaries.append(
            {
                "scope": scope,
                "pixels": pixels,
                "accuracy": safe_divide(
                    sum(record["correct"] for record in selected), pixels
                ),
                "negative_log_likelihood": safe_divide(
                    sum(record["nll_sum"] for record in selected), pixels
                ),
                "multiclass_brier_score": safe_divide(
                    sum(record["brier_sum"] for record in selected), pixels
                ),
                "mean_predictive_entropy": safe_divide(
                    sum(record["entropy_sum"] for record in selected), pixels
                ),
                "adaptive_ece": overall_ece,
                "presence_only_precision_warning": any(
                    record["country"] not in FULL_DATA_COUNTRIES
                    for record in selected
                ),
            }
        )
        for index, label in enumerate(DISTANCE_LABELS):
            bins, ece = _adaptive_rows(
                scope,
                label,
                counts[index],
                confidence_sums[index],
                correct_sums[index],
            )
            calibration_rows.extend(bins)
            correct_count = stratum_counts[index] - stratum_error_count[index]
            correct_entropy = stratum_entropy[index] - stratum_error_entropy[index]
            distance_rows.append(
                {
                    "scope": scope,
                    "distance_stratum": label,
                    "pixels": int(stratum_counts[index]),
                    "adaptive_ece": ece,
                    "mean_entropy": safe_divide(stratum_entropy[index], stratum_counts[index]),
                    "mean_entropy_incorrect": safe_divide(
                        stratum_error_entropy[index], stratum_error_count[index]
                    ),
                    "mean_entropy_correct": safe_divide(correct_entropy, correct_count),
                }
            )
    return (
        pd.DataFrame(summaries),
        pd.DataFrame(calibration_rows),
        pd.DataFrame(distance_rows),
    )


def semantic_object_confidences(
    arrays: ObjectArrays,
    prediction_labels: np.ndarray,
    prediction: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict]:
    """Measure baseline field persistence and probability summaries."""
    rows = []
    threshold_labels = []
    for threshold in PERSISTENCE_THRESHOLDS:
        boundary = probabilities[2] >= threshold
        labels, _ = ndimage.label((~boundary) & (prediction != 0), N4)
        areas = np.bincount(labels.ravel())
        threshold_labels.append((labels, areas))
    for index, prediction_id in enumerate(arrays.prediction_ids):
        mask = prediction_labels == prediction_id
        mask_area = int(mask.sum())
        existence = []
        for labels, areas in threshold_labels:
            overlapping = labels[mask]
            overlapping = overlapping[overlapping != 0]
            if not len(overlapping):
                existence.append(False)
                continue
            values, counts = np.unique(overlapping, return_counts=True)
            maximum = int(np.argmax(counts))
            component = int(values[maximum])
            intersection = int(counts[maximum])
            union = int(areas[component] + mask_area - intersection)
            existence.append(safe_divide(intersection, union) >= 0.50)
        active = np.flatnonzero(existence)
        runs = np.split(active, np.flatnonzero(np.diff(active) != 1) + 1)
        longest = max(runs, key=len) if len(active) else np.asarray([], dtype=int)
        persistence = (
            float(
                PERSISTENCE_THRESHOLDS[longest[-1]]
                - PERSISTENCE_THRESHOLDS[longest[0]]
                + 0.05
            )
            if len(longest)
            else 0.0
        )
        rows.append(
            {
                "prediction_id": int(prediction_id),
                "persistence": persistence,
                "persistence_steps": int(len(longest)),
                "mean_interior_probability": float(probabilities[1][mask].mean()),
                "mean_top_confidence": float(probabilities[:, mask].max(axis=0).mean()),
                "area_pixels": mask_area,
            }
        )
    return rows


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive, negative = labels == 1, labels == 0
    if not positive.any() or not negative.any():
        return float("nan")
    ranks = rankdata(scores)
    positives, negatives = int(positive.sum()), int(negative.sum())
    return float(
        (ranks[positive].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _partial_spearman(confidence: np.ndarray, iou: np.ndarray, area: np.ndarray) -> float:
    if len(confidence) < 3:
        return float("nan")
    x = rankdata(confidence)
    y = rankdata(iou)
    control = np.column_stack((np.ones(len(area)), rankdata(np.log1p(area))))
    x_residual = x - control @ np.linalg.lstsq(control, x, rcond=None)[0]
    y_residual = y - control @ np.linalg.lstsq(control, y, rcond=None)[0]
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def confidence_reliability_summary(
    rows: list[dict], score_columns: tuple[str, ...]
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()
    output = []
    for scope in sorted(frame.country.unique()) + ["all_countries"]:
        selected = frame if scope == "all_countries" else frame[frame.country == scope]
        labels = selected.correct_iou50.astype(int).to_numpy()
        ious = selected.best_iou.to_numpy(float)
        area = selected.area_pixels.to_numpy(float)
        for score_column in score_columns:
            scores = selected[score_column].to_numpy(float)
            correlation = spearmanr(scores, ious, nan_policy="omit").statistic
            output.append(
                {
                    "scope": scope,
                    "confidence_score": score_column,
                    "predictions": len(selected),
                    "true_positives_iou50": int(labels.sum()),
                    "auroc_tp_vs_fp": _auroc(scores, labels),
                    "spearman_confidence_vs_best_iou": float(correlation),
                    "partial_spearman_controlling_log_area": _partial_spearman(scores, ious, area),
                    "area_auroc_baseline": _auroc(area, labels),
                    "area_spearman_baseline": float(
                        spearmanr(area, ious, nan_policy="omit").statistic
                    ),
                }
            )
    return pd.DataFrame(output)


def risk_coverage_rows(
    rows: list[dict],
    ground_truth_counts: dict[str, int],
    score_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    curve_rows, summary_rows = [], []
    countries = sorted(ground_truth_counts)
    for scope in countries + ["all_countries"]:
        selected = frame if scope == "all_countries" else frame[frame.country == scope]
        gt_total = (
            sum(ground_truth_counts.values())
            if scope == "all_countries"
            else ground_truth_counts[scope]
        )
        for score_column in score_columns:
            ranked = selected.sort_values(score_column, ascending=False)
            scope_rows = []
            for coverage in np.linspace(0.05, 1.0, 20):
                retained = ranked.head(max(1, math.ceil(len(ranked) * coverage)))
                used: dict[tuple[str, str], set[int]] = {}
                tp = 0
                for row in retained.itertuples():
                    key = (row.country, row.chip_id)
                    used.setdefault(key, set())
                    candidates = sorted(row.candidates, key=lambda item: item[1], reverse=True)
                    match = next(
                        (
                            gt_id
                            for gt_id, iou in candidates
                            if iou >= 0.50 and gt_id not in used[key]
                        ),
                        None,
                    )
                    if match is not None:
                        used[key].add(match)
                        tp += 1
                fp, fn = len(retained) - tp, gt_total - tp
                precision, recall = safe_divide(tp, tp + fp), safe_divide(tp, tp + fn)
                f1 = safe_divide(2 * precision * recall, precision + recall)
                scope_rows.append(
                    {
                        "scope": scope,
                        "confidence_score": score_column,
                        "coverage": float(coverage),
                        "retained_predictions": len(retained),
                        "object_f1": f1,
                        "risk_one_minus_f1": 1 - f1,
                    }
                )
            curve_rows.extend(scope_rows)
            summary_rows.append(
                {
                    "scope": scope,
                    "confidence_score": score_column,
                    "aurc": float(
                        np.trapz(
                            [row["risk_one_minus_f1"] for row in scope_rows],
                            [row["coverage"] for row in scope_rows],
                        )
                    ),
                    "object_f1_at_70pct_coverage": min(
                        scope_rows, key=lambda row: abs(row["coverage"] - 0.70)
                    )["object_f1"],
                    "object_f1_at_full_coverage": scope_rows[-1]["object_f1"],
                }
            )
    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)
