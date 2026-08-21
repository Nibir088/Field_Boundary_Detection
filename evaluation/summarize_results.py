"""Generate a readable Markdown report from an FTW evaluation directory."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


def read_csv(results_dir: Path, name: str) -> list[dict[str, str]]:
    path = results_dir / name
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str] | None, key: str) -> float:
    if not row:
        return math.nan
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return math.nan


def finite(value: float) -> bool:
    return math.isfinite(value)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:,.{digits}f}" if finite(value) else "not available"


def count(value: float) -> str:
    return f"{int(value):,}" if finite(value) else "not available"


def percent(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%" if finite(value) else "not available"


def combined(rows: Iterable[dict[str, str]]) -> dict[str, str] | None:
    return next((row for row in rows if row.get("scope") == "all_countries"), None)


def threshold_row(rows: Iterable[dict[str, str]], threshold: float) -> dict[str, str] | None:
    candidates = [row for row in rows if row.get("scope") == "all_countries"]
    return min(candidates, key=lambda row: abs(number(row, "iou_threshold") - threshold), default=None)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return output


def field_summary(results_dir: Path) -> tuple[list[str], dict[str, float]]:
    rows = read_csv(results_dir, "object_summary_by_threshold.csv")
    output = ["## 1. Field-level detection", ""]
    headline: dict[str, float] = {}
    if rows:
        table = []
        for threshold in (0.25, 0.50, 0.75):
            row = threshold_row(rows, threshold)
            if not row:
                continue
            table.append([
                fmt(threshold, 2), count(number(row, "true_positives")),
                count(number(row, "false_positives")), count(number(row, "false_negatives")),
                fmt(number(row, "object_precision")), fmt(number(row, "object_recall")),
                fmt(number(row, "object_f1")), fmt(number(row, "mean_matched_iou")),
                fmt(number(row, "panoptic_quality_pq")),
            ])
            if abs(threshold - 0.50) < 1e-6:
                headline = {key: number(row, key) for key in row}
        output += markdown_table(
            ["IoU threshold", "TP", "FP", "FN", "Precision", "Recall", "F1", "Matched IoU", "PQ"], table
        )
        row = threshold_row(rows, 0.50)
        if row:
            gt, pred = number(row, "ground_truth_objects"), number(row, "predicted_objects")
            error = number(row, "object_count_error")
            output += [
                "",
                f"At IoU ≥ 0.50, the evaluation contains **{count(gt)} reference fields** and "
                f"**{count(pred)} predicted fields**. The count error is **{count(error)}** "
                f"({percent(error / gt) if gt else 'not available'}).",
                "",
                "A high matched IoU with lower recall means successfully detected fields have good shapes, but many reference fields are missed or merged.",
            ]
    else:
        gt_rows = read_csv(results_dir, "ground_truth_field_matches_iou50.csv")
        pred_rows = read_csv(results_dir, "prediction_field_matches_iou50.csv")
        if gt_rows and pred_rows:
            tp = sum(row.get("detected_iou50", "").lower() == "true" for row in gt_rows)
            fp, fn = len(pred_rows) - tp, len(gt_rows) - tp
            precision = tp / len(pred_rows) if pred_rows else math.nan
            recall = tp / len(gt_rows) if gt_rows else math.nan
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
            matched = [number(row, "matched_iou_iou50") for row in gt_rows if row.get("detected_iou50", "").lower() == "true"]
            mean_iou = sum(matched) / len(matched) if matched else math.nan
            denominator = tp + 0.5 * fp + 0.5 * fn
            pq = sum(matched) / denominator if denominator else math.nan
            headline = {"true_positives": tp, "false_positives": fp, "false_negatives": fn, "object_precision": precision, "object_recall": recall, "object_f1": f1, "mean_matched_iou": mean_iou, "panoptic_quality_pq": pq}
            output += markdown_table(
                ["IoU threshold", "TP", "FP", "FN", "Precision", "Recall", "F1", "Matched IoU", "PQ"],
                [["0.50", count(tp), count(fp), count(fn), fmt(precision), fmt(recall), fmt(f1), fmt(mean_iou), fmt(pq)]],
            )
            output += ["", "This row was reconstructed from the two per-field match files."]
        else:
            risk_rows = [row for row in read_csv(results_dir, "risk_coverage_summary.csv") if row.get("scope") == "all_countries"]
            preferred = next((row for row in risk_rows if row.get("confidence_score") == "mean_interior_probability"), risk_rows[0] if risk_rows else None)
            full_f1 = number(preferred, "object_f1_at_full_coverage")
            output += [
                "The main `object_summary_by_threshold.csv` and per-field match files are missing.", "",
                f"The available risk-coverage file reports full-coverage object F1 = **{fmt(full_f1)}**, but TP/FP/FN, recall, precision, matched IoU, and per-field outcomes cannot be reconstructed from this folder.",
            ]
    output.append("")
    return output, headline


def boundary_summary(results_dir: Path) -> list[str]:
    row = combined(read_csv(results_dir, "boundary_summary.csv"))
    output = ["## 2. Boundary localization", ""]
    if not row:
        return output + ["`boundary_summary.csv` is missing.", ""]
    output += markdown_table(["Metric", "Value", "Valid chips"], [
        ["Exact boundary F1", fmt(number(row, "exact_f1")), count(number(row, "exact_f1_valid_count"))],
        ["F1 within 1 px", fmt(number(row, "tolerance_1px_f1")), count(number(row, "tolerance_1px_f1_valid_count"))],
        ["F1 within 2 px", fmt(number(row, "tolerance_2px_f1")), count(number(row, "tolerance_2px_f1_valid_count"))],
        ["Balanced ASBD", f"{fmt(number(row, 'asbd_balanced_px'), 2)} px / {fmt(number(row, 'asbd_balanced_m'), 1)} m", count(number(row, "asbd_balanced_px_valid_count"))],
        ["HD95", f"{fmt(number(row, 'hausdorff_95_px'), 2)} px / {fmt(number(row, 'hausdorff_95_m'), 1)} m", count(number(row, "hausdorff_95_px_valid_count"))],
    ])
    exact, tolerant = number(row, "exact_f1"), number(row, "tolerance_2px_f1")
    if finite(exact) and finite(tolerant):
        output += ["", f"Allowing two pixels raises boundary F1 by **{fmt(tolerant - exact)}**. This indicates substantial near-miss localization, while ASBD and HD95 expose larger failures."]
    output.append("")
    return output


def repair_topology(results_dir: Path) -> list[str]:
    repair = combined(read_csv(results_dir, "merge_repair_summary.csv"))
    topology = combined(read_csv(results_dir, "topology_summary.csv"))
    output = ["## 3. Merges, repair, and topology", ""]
    if repair:
        output += markdown_table(["Repair measure", "Result"], [
            ["Merged reference-field pairs", count(number(repair, "merge_field_pairs"))],
            ["Median minimum repair", f"{fmt(number(repair, 'repair_cost_median_px'), 1)} px"],
            ["Repairable within 1 px", percent(number(repair, "repair_le_1px_fraction"))],
            ["Repairable within 2 px", percent(number(repair, "repair_le_2px_fraction"))],
            ["Repairable within 3 px", percent(number(repair, "repair_le_3px_fraction"))],
            ["Repair cut below p(boundary)=0.10", percent(number(repair, "breach_max_below_010_fraction"))],
        ])
        output += ["", "`merge_field_pairs` counts interacting reference-field pairs, not merged prediction objects; one prediction containing three fields may contribute several pairs.", ""]
    else:
        output += ["`merge_repair_summary.csv` is missing.", ""]
    if topology:
        output += markdown_table(["Topology measure (chip macro)", "Result"], [
            ["Variation of Information", f"{fmt(number(topology, 'variation_of_information_bits'))} bits"],
            ["Reference connected components", fmt(number(topology, "gt_betti0"), 2)],
            ["Predicted connected components", fmt(number(topology, "prediction_betti0"), 2)],
            ["Component-count error", fmt(number(topology, "betti0_error"), 2)],
        ])
        if number(topology, "betti0_error") < 0:
            output += ["", "The negative component-count error supports an **under-segmentation / merging** pattern."]
    else:
        output.append("`topology_summary.csv` is missing.")
    output.append("")
    return output


def geometry_confidence(results_dir: Path) -> list[str]:
    geometry = combined(read_csv(results_dir, "geometry_summary.csv"))
    population = combined(read_csv(results_dir, "geometry_population_summary.csv"))
    confidence_rows = [row for row in read_csv(results_dir, "field_confidence_summary.csv") if row.get("scope") == "all_countries"]
    confidence = next((row for row in confidence_rows if row.get("confidence_score") == "mean_interior_probability"), None)
    risk_rows = [row for row in read_csv(results_dir, "risk_coverage_summary.csv") if row.get("scope") == "all_countries"]
    risk = next((row for row in risk_rows if row.get("confidence_score") == "mean_interior_probability"), None)
    output = ["## 4. Geometry of matched fields", ""]
    if geometry:
        output += markdown_table(["Measure", "Result"], [
            ["Eligible matched pairs", count(number(geometry, "matched_pairs"))],
            ["Median predicted/reference perimeter", fmt(number(geometry, "median_perimeter_ratio"))],
            ["Median area delta", f"{fmt(number(geometry, 'median_delta_area'), 1)} px²"],
            ["Median perimeter delta", f"{fmt(number(geometry, 'median_delta_perimeter'), 2)} px"],
            ["Predicted rings per 100 fields", fmt(number(geometry, "predicted_interior_rings_per_100_fields"), 1)],
        ])
        output += ["", "Geometry is conditional on IoU-0.50 matches and excludes border-touching/invalid polygons; it must not be interpreted as performance over every field.", ""]
    else:
        output += ["`geometry_summary.csv` is missing.", ""]
    if population:
        output += [
            f"The population comparison includes **{count(number(population, 'ground_truth_objects'))} reference** "
            f"and **{count(number(population, 'predicted_objects'))} predicted** eligible geometries. "
            f"Its Wasserstein distances are area={fmt(number(population, 'wasserstein_area'), 2)}, "
            f"perimeter={fmt(number(population, 'wasserstein_perimeter'), 2)}, and "
            f"right-angle fraction={fmt(number(population, 'wasserstein_right_angle_fraction'))}.",
            "",
            "Wasserstein values retain each descriptor's units, so area, perimeter, and unitless shape distances must not be compared directly.",
            "",
        ]
    output += ["## 5. Field confidence", ""]
    if confidence:
        output += markdown_table(["Measure", "Result"], [
            ["TP-vs-FP AUROC", fmt(number(confidence, "auroc_tp_vs_fp"))],
            ["Confidence vs best-IoU Spearman", fmt(number(confidence, "spearman_confidence_vs_best_iou"))],
            ["Partial Spearman controlling log-area", fmt(number(confidence, "partial_spearman_controlling_log_area"))],
            ["Area-only AUROC", fmt(number(confidence, "area_auroc_baseline"))],
        ])
    else:
        output.append("A mean-interior-probability confidence summary is not available.")
    if risk:
        output += ["", f"Keeping the top 70% by confidence gives object F1 **{fmt(number(risk, 'object_f1_at_70pct_coverage'))}**, compared with **{fmt(number(risk, 'object_f1_at_full_coverage'))}** at full coverage."]
    output.append("")
    return output


def threshold_and_country(results_dir: Path) -> list[str]:
    best = combined(read_csv(results_dir, "boundary_probability_best_pq.csv"))
    sweep = [row for row in read_csv(results_dir, "boundary_probability_sweep.csv") if row.get("scope") == "all_countries"]
    objects = read_csv(results_dir, "object_summary_by_threshold.csv")
    boundaries = read_csv(results_dir, "boundary_summary.csv")
    output = ["## 6. Boundary threshold", ""]
    if best:
        output += [
            f"Best pooled threshold: **{fmt(number(best, 'boundary_probability_threshold'), 2)}**, with PQ **{fmt(number(best, 'panoptic_quality_pq'))}**, {count(number(best, 'merged_predictions_a10'))} merged predictions, and {count(number(best, 'split_fields_a10'))} split reference fields at alpha=0.10.",
            "", "Choose thresholds using validation data, not this test-set optimum.",
        ]
        at_default = min(
            sweep,
            key=lambda row: abs(number(row, "boundary_probability_threshold") - 0.50),
            default=None,
        )
        if at_default:
            gain = number(best, "panoptic_quality_pq") - number(at_default, "panoptic_quality_pq")
            output += [
                "",
                f"At threshold 0.50, pooled PQ is **{fmt(number(at_default, 'panoptic_quality_pq'))}**; "
                f"the test-optimal threshold improves it by only **{fmt(gain, 4)}**.",
            ]
    else:
        output.append("No semantic boundary-probability optimum is available (expected for Delineate Anything).")
    output += ["", "## 7. Country comparison", ""]
    if objects:
        object_by_country = {row["scope"]: row for row in objects if row.get("scope") != "all_countries" and abs(number(row, "iou_threshold") - 0.5) < 1e-6}
        boundary_by_country = {row["scope"]: row for row in boundaries if row.get("scope") != "all_countries"}
        candidates = []
        for country, row in object_by_country.items():
            boundary = boundary_by_country.get(country, {})
            candidates.append((country, number(row, "object_f1"), number(row, "panoptic_quality_pq"), number(boundary, "tolerance_2px_f1"), number(boundary, "chips")))
        candidates.sort(key=lambda item: item[1], reverse=True)
        selected = candidates[:5] + candidates[-5:]
        output += markdown_table(["Group", "Country", "Object F1", "PQ", "Boundary F1 (2 px)", "Chips"], [
            ["Top" if i < 5 else "Bottom", c, fmt(f1), fmt(pq), fmt(bf1), count(chips)]
            for i, (c, f1, pq, bf1, chips) in enumerate(selected)
        ])
    elif boundaries:
        candidates = [(row["scope"], number(row, "tolerance_2px_f1"), number(row, "chips")) for row in boundaries if row.get("scope") != "all_countries" and finite(number(row, "tolerance_2px_f1"))]
        candidates.sort(key=lambda item: item[1], reverse=True)
        selected = candidates[:5] + candidates[-5:]
        output += markdown_table(["Group", "Country", "Boundary F1 (2 px)", "Chips"], [
            ["Top" if i < 5 else "Bottom", country, fmt(score), count(chips)]
            for i, (country, score, chips) in enumerate(selected)
        ])
        output += ["", "Object-level country ranking is unavailable because `object_summary_by_threshold.csv` is missing."]
    output += ["", "Treat countries with few chips or low `*_valid_count` as uncertain; do not rank them as if sample sizes were equal.", ""]
    return output


def inventory(results_dir: Path) -> list[str]:
    expected = [
        "object_summary_by_threshold.csv", "ground_truth_field_matches_iou50.csv",
        "prediction_field_matches_iou50.csv", "pixel_metrics.csv", "boundary_summary.csv",
        "merge_repair_summary.csv", "topology_summary.csv", "geometry_summary.csv",
        "field_confidence_summary.csv", "risk_coverage_summary.csv",
        "boundary_probability_best_pq.csv", "boundary_probability_sweep.csv",
        "geometry_population_summary.csv",
    ]
    present = [name for name in expected if (results_dir / name).is_file()]
    missing = [name for name in expected if name not in present]
    return [
        "## File completeness", "",
        f"Found **{len(list(results_dir.glob('*.csv')))} CSV files**. Key files found: {', '.join(f'`{name}`' for name in present) or 'none'}.", "",
        f"Key files missing: {', '.join(f'`{name}`' for name in missing) or 'none'}.", "",
    ]


def build_report(results_dir: Path) -> str:
    field, headline = field_summary(results_dir)
    conclusion = ["## Interpretation", ""]
    recall = headline.get("object_recall", math.nan)
    iou = headline.get("mean_matched_iou", math.nan)
    if finite(recall) and finite(iou) and iou > recall:
        conclusion.append("The dominant field-level pattern is stronger shape quality among matched fields than field recovery: the model delineates many successful matches well but fails to recover enough individual fields.")
    elif headline:
        conclusion.append("Interpret field recovery and matched-field geometry together; neither metric alone describes complete performance.")
    else:
        conclusion.append("A complete field-level conclusion requires the missing object summary or per-field match tables.")
    conclusion += ["", "Confirm aggregate findings by inspecting correct, missed, false-positive, merged, and split examples in `sample_pdfs/<country>/`.", ""]
    lines = ["# FTW evaluation summary", "", f"Results directory: `{results_dir.resolve()}`", ""]
    lines += inventory(results_dir) + field + boundary_summary(results_dir)
    lines += repair_topology(results_dir) + geometry_confidence(results_dir)
    lines += threshold_and_country(results_dir) + conclusion
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Directory containing evaluation CSV files")
    parser.add_argument("--output", type=Path, help="Write Markdown here; otherwise print to stdout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {args.results_dir}")
    report = build_report(args.results_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Summary written to: {args.output}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
