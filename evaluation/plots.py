"""Create presentation-ready figures from either shared evaluator."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")


def _read(results: Path, name: str) -> pd.DataFrame | None:
    path = results / name
    if not path.is_file() or not path.stat().st_size:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _save(figure: plt.Figure, path: Path, dpi: int) -> None:
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Created: {path}")


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "legend.frameon": False,
        }
    )


def plot_pixel_metrics(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    combined = frame[frame.scope == "all_countries"].copy()
    positive = combined[
        ((combined["view"] == "native_3class"))
        | combined["class"].isin(("field_interior", "field_extent"))
    ]
    positive["label"] = positive["view"] + ": " + positive["class"]
    figure, axis = plt.subplots(figsize=(11, max(4.5, len(positive) * 0.75)))
    positive.set_index("label")[["iou", "precision", "recall", "f1"]].plot.barh(
        ax=axis, color=COLORS[:4]
    )
    axis.set_xlim(0, 1)
    axis.set_xlabel("Score")
    axis.set_ylabel("")
    axis.set_title("Combined pixel-level performance")
    _save(figure, output / "pixel_metrics.png", dpi)


def plot_confusion(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    combined = frame[frame.scope == "all_countries"]
    view = "native_3class" if "native_3class" in set(combined.view) else "field_extent_binary"
    selected = combined[combined.view == view]
    matrix = selected.pivot(
        index="target_class", columns="predicted_class", values="pixels"
    )
    values = matrix.to_numpy(float)
    normalized = np.divide(
        values,
        values.sum(axis=1, keepdims=True),
        out=np.zeros_like(values),
        where=values.sum(axis=1, keepdims=True) != 0,
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Reference")
    axis.set_title(f"Normalized pixel confusion: {view}")
    for row, column in np.ndindex(normalized.shape):
        value = normalized[row, column]
        axis.text(
            column,
            row,
            f"{value:.1%}\n{int(values[row, column]):,}",
            ha="center",
            va="center",
            color="white" if value > 0.55 else "black",
            fontsize=8,
        )
    figure.colorbar(image, ax=axis, label="Reference-row fraction")
    _save(figure, output / "pixel_confusion.png", dpi)


def plot_object_thresholds(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    combined = frame[frame.scope == "all_countries"].sort_values("iou_threshold")
    figure, axis = plt.subplots(figsize=(8, 5))
    for metric, label, color in (
        ("object_precision", "Precision", COLORS[1]),
        ("object_recall", "Recall", COLORS[2]),
        ("object_f1", "F1", COLORS[3]),
        ("panoptic_quality_pq", "PQ", COLORS[0]),
    ):
        axis.plot(combined.iou_threshold, combined[metric], "o-", label=label, color=color)
    axis.set_xticks(combined.iou_threshold)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Matching IoU threshold")
    axis.set_ylabel("Score")
    axis.set_title("Field detection sensitivity to matching IoU")
    axis.legend(ncol=2)
    _save(figure, output / "object_threshold_sweep.png", dpi)


def plot_country_objects(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    selected = frame[
        (frame.scope != "all_countries") & np.isclose(frame.iou_threshold, 0.50)
    ].sort_values("object_f1")
    if selected.empty:
        return
    figure, axis = plt.subplots(figsize=(10, max(5, len(selected) * 0.38)))
    positions = np.arange(len(selected))
    axis.barh(positions - 0.17, selected.object_f1, 0.34, label="Object F1", color=COLORS[4])
    axis.barh(
        positions + 0.17,
        selected.panoptic_quality_pq,
        0.34,
        label="Panoptic Quality",
        color=COLORS[0],
    )
    axis.set_yticks(positions, selected.scope.str.replace("_", " ").str.title())
    axis.set_xlim(0, 1)
    axis.set_xlabel("Score at IoU ≥ 0.50")
    axis.set_title("Field-level performance by country")
    axis.legend()
    _save(figure, output / "country_object_performance.png", dpi)


def plot_boundary(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    row = frame[frame.scope == "all_countries"]
    if row.empty:
        return
    row = row.iloc[0]
    labels = ("Exact", "1 px Chebyshev", "2 px Chebyshev")
    values = (
        row.get("exact_f1", np.nan),
        row.get("tolerance_1px_f1", np.nan),
        row.get("tolerance_2px_f1", np.nan),
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, values, color=COLORS[:3])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Boundary F1")
    axis.set_title("Combined boundary agreement")
    axis.bar_label(bars, fmt="%.3f", padding=3)
    _save(figure, output / "boundary_tolerance.png", dpi)


def plot_topology(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    row = frame[
        (frame.scope == "all_countries") & np.isclose(frame.iou_threshold, 0.50)
    ]
    if row.empty:
        return
    row = row.iloc[0]
    alphas = (0.05, 0.10, 0.20, 0.30)
    merges = [row[f"merged_predictions_a{int(value * 100):02d}"] for value in alphas]
    splits = [row[f"split_fields_a{int(value * 100):02d}"] for value in alphas]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(alphas, merges, "o-", label="Merged predictions", color=COLORS[4])
    axis.plot(alphas, splits, "o-", label="Split reference fields", color=COLORS[0])
    axis.set_xlabel("Overlap association α")
    axis.set_ylabel("Count")
    axis.set_title("Merge/split sensitivity")
    axis.legend()
    _save(figure, output / "merge_split_alpha_sweep.png", dpi)


def plot_geometry(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    row = frame[frame.scope == "all_countries"]
    if row.empty:
        return
    row = row.iloc[0]
    labels = ("Perimeter ratio", "GT right angles", "Pred right angles")
    values = (
        row.get("median_perimeter_ratio", np.nan),
        row.get("gt_right_angle_fraction", np.nan),
        row.get("prediction_right_angle_fraction", np.nan),
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, values, color=COLORS[:3])
    axis.set_ylabel("Value")
    axis.set_title("Matched-field geometry diagnostics")
    axis.tick_params(axis="x", rotation=15)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    _save(figure, output / "geometry_headlines.png", dpi)


def plot_closure(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    rows = []
    for radius, group in frame.groupby("closing_radius_px"):
        tp = group.true_positives.sum()
        fp = group.false_positives.sum()
        fn = group.false_negatives.sum()
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * precision * recall / (precision + recall)
        rows.append((radius, f1))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(*zip(*rows), "o-", color=COLORS[2])
    axis.set_ylim(0, 1)
    axis.set_xticks([row[0] for row in rows])
    axis.set_xlabel("Boundary closing radius (pixels)")
    axis.set_ylabel("Pooled object F1")
    axis.set_title("Closure sensitivity")
    _save(figure, output / "closing_radius_sweep.png", dpi)


def plot_repair_distribution(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    values = np.sort(frame.repair_upper_bound_px.dropna().to_numpy(float))
    if not len(values):
        return
    cumulative = np.arange(1, len(values) + 1) / len(values)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.step(values, cumulative, where="post", color=COLORS[4])
    axis.set_ylim(0, 1)
    axis.set_xlabel("Exact pairwise minimum separator (pixels)")
    axis.set_ylabel("Fraction of breached field pairs")
    axis.set_title("Pairwise minimum-cut repair distribution")
    _save(figure, output / "merge_repair_cdf.png", dpi)


def plot_reliability(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    selected = frame[
        (frame.scope == "all_countries") & (frame.distance_stratum == "all")
    ]
    if selected.empty:
        return
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot((0, 1), (0, 1), "--", color="#777777", label="Perfect calibration")
    axis.plot(
        selected.mean_confidence,
        selected.accuracy,
        "o-",
        color=COLORS[0],
        label="Model",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Accuracy")
    axis.set_title("Adaptive-bin reliability diagram")
    axis.legend()
    _save(figure, output / "calibration_reliability.png", dpi)


def plot_distance_calibration(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    selected = frame[frame.scope == "all_countries"]
    if selected.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(selected.distance_stratum, selected.adaptive_ece, color=COLORS[0])
    axis.set_ylabel("Adaptive ECE")
    axis.set_xlabel("Distance to reference boundary")
    axis.set_title("Calibration stratified by boundary distance")
    axis.bar_label(bars, fmt="%.3f", padding=3)
    _save(figure, output / "calibration_by_boundary_distance.png", dpi)


def plot_risk_coverage(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    selected = frame[frame.scope == "all_countries"]
    if selected.empty:
        return
    figure, axis = plt.subplots(figsize=(8, 5))
    for index, (score, group) in enumerate(selected.groupby("confidence_score")):
        axis.plot(
            group.coverage,
            group.object_f1,
            "o-",
            label=score.replace("_", " "),
            color=COLORS[index % len(COLORS)],
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Prediction coverage")
    axis.set_ylabel("Object F1 after rematching")
    axis.set_title("Field confidence performance–coverage")
    axis.legend()
    _save(figure, output / "risk_coverage.png", dpi)


def plot_confidence_iou(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    score = (
        "persistence"
        if "persistence" in frame
        else "native_confidence"
        if "native_confidence" in frame
        else None
    )
    if score is None or frame.empty:
        return
    sample = frame.sample(min(len(frame), 10000), random_state=42)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.hexbin(
        sample[score],
        sample.best_iou,
        gridsize=35,
        mincnt=1,
        cmap="viridis",
    )
    axis.set_xlabel(score.replace("_", " ").title())
    axis.set_ylabel("Best reference-field IoU")
    axis.set_title("Field confidence versus spatial agreement")
    _save(figure, output / "field_confidence_vs_iou.png", dpi)


def plot_breach_confidence(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    if "breach_confidence_max" not in frame:
        return
    values = frame.breach_confidence_max.dropna()
    if values.empty:
        return
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=np.linspace(0, 1, 21), color=COLORS[4], edgecolor="white")
    axis.axvspan(0.30, 0.49, color=COLORS[1], alpha=0.18, label="Decoding candidate")
    axis.axvspan(0, 0.10, color=COLORS[0], alpha=0.15, label="Representation candidate")
    axis.set_xlabel("Maximum boundary probability along breach path")
    axis.set_ylabel("Merged field pairs")
    axis.set_title("Breach confidence distribution")
    axis.legend()
    _save(figure, output / "breach_confidence.png", dpi)


def plot_topology_summary(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    selected = frame[frame.scope != "all_countries"].sort_values(
        "variation_of_information_bits"
    )
    if selected.empty:
        return
    figure, axis = plt.subplots(figsize=(10, max(5, len(selected) * 0.38)))
    axis.barh(
        selected.scope.str.replace("_", " ").str.title(),
        selected.variation_of_information_bits,
        color=COLORS[0],
    )
    axis.set_xlabel("Variation of information (bits; lower is better)")
    axis.set_title("Partition topology disagreement by country")
    _save(figure, output / "topology_variation_of_information.png", dpi)


def plot_model_sweep(results: Path, output: Path, dpi: int) -> None:
    ap = _read(results, "average_precision_summary.csv")
    probability = _read(results, "boundary_probability_sweep.csv")
    if ap is not None:
        row = ap[ap.scope == "all_countries"]
        if row.empty:
            return
        columns = sorted(
            (
                column
                for column in ap
                if column.startswith("ap") and column[2:].isdigit()
            ),
            key=lambda value: int(value[2:]),
        )
        thresholds = [int(column[2:]) / 100 for column in columns]
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(thresholds, row.iloc[0][columns], "o-", color=COLORS[0])
        axis.set_ylim(0, 1)
        axis.set_xlabel("Mask IoU threshold")
        axis.set_ylabel("Average Precision")
        axis.set_title("Confidence-ranked AP threshold sweep")
        _save(figure, output / "average_precision_sweep.png", dpi)
    elif probability is not None:
        combined = probability[probability.scope == "all_countries"]
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(
            combined.boundary_probability_threshold,
            combined.panoptic_quality_pq,
            "o-",
            color=COLORS[0],
        )
        axis.set_xlabel("Boundary probability threshold")
        axis.set_ylabel("Panoptic Quality")
        axis.set_title("Boundary threshold sensitivity")
        _save(figure, output / "boundary_probability_sweep.png", dpi)


def create_plots(results_dir: Path, output_dir: Path | None = None, dpi: int = 200) -> Path:
    results = results_dir.expanduser().resolve()
    output = (output_dir or results / "plots").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _style()
    pixel = _read(results, "pixel_metrics.csv")
    confusion = _read(results, "pixel_confusion.csv")
    objects = _read(results, "object_summary_by_threshold.csv")
    boundary = _read(results, "boundary_summary.csv")
    geometry = _read(results, "geometry_summary.csv")
    closing = _read(results, "closing_radius_sweep.csv")
    repair = _read(results, "merge_repair_distance.csv")
    calibration = _read(results, "calibration_bins.csv")
    calibration_distance = _read(
        results, "calibration_by_boundary_distance.csv"
    )
    risk_coverage = _read(results, "risk_coverage.csv")
    field_confidence = _read(results, "field_confidence_by_prediction.csv")
    topology = _read(results, "topology_summary.csv")
    if pixel is not None:
        plot_pixel_metrics(pixel, output, dpi)
    if confusion is not None:
        plot_confusion(confusion, output, dpi)
    if objects is not None:
        plot_object_thresholds(objects, output, dpi)
        plot_country_objects(objects, output, dpi)
        plot_topology(objects, output, dpi)
    if boundary is not None:
        plot_boundary(boundary, output, dpi)
    if geometry is not None:
        plot_geometry(geometry, output, dpi)
    if closing is not None:
        plot_closure(closing, output, dpi)
    if repair is not None:
        plot_repair_distribution(repair, output, dpi)
        plot_breach_confidence(repair, output, dpi)
    if calibration is not None:
        plot_reliability(calibration, output, dpi)
    if calibration_distance is not None:
        plot_distance_calibration(calibration_distance, output, dpi)
    if risk_coverage is not None:
        plot_risk_coverage(risk_coverage, output, dpi)
    if field_confidence is not None:
        plot_confidence_iou(field_confidence, output, dpi)
    if topology is not None:
        plot_topology_summary(topology, output, dpi)
    plot_model_sweep(results, output, dpi)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    output = create_plots(args.results_dir, args.output_dir, args.dpi)
    print(f"Plots saved in: {output}")


if __name__ == "__main__":
    main()
