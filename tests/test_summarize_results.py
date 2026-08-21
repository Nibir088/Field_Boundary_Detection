import csv

from evaluation.summarize_results import build_report


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_complete_object_summary_is_reported(tmp_path):
    write_csv(
        tmp_path / "object_summary_by_threshold.csv",
        [
            {
                "scope": "all_countries",
                "iou_threshold": "0.5",
                "true_positives": "6",
                "false_positives": "4",
                "false_negatives": "2",
                "object_precision": "0.6",
                "object_recall": "0.75",
                "object_f1": "0.6666667",
                "mean_matched_iou": "0.8",
                "panoptic_quality_pq": "0.48",
                "ground_truth_objects": "8",
                "predicted_objects": "10",
                "object_count_error": "2",
            }
        ],
    )
    report = build_report(tmp_path)
    assert "## 1. Field-level detection" in report
    assert "**8 reference fields**" in report
    assert "0.667" in report
    assert "Key files missing" in report


def test_partial_results_explain_missing_object_files(tmp_path):
    write_csv(
        tmp_path / "risk_coverage_summary.csv",
        [
            {
                "scope": "all_countries",
                "confidence_score": "mean_interior_probability",
                "object_f1_at_full_coverage": "0.516",
                "object_f1_at_70pct_coverage": "0.534",
            }
        ],
    )
    report = build_report(tmp_path)
    assert "main `object_summary_by_threshold.csv`" in report
    assert "full-coverage object F1 = **0.516**" in report
    assert "cannot be reconstructed" in report
