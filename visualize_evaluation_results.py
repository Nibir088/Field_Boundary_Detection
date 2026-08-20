#!/usr/bin/env python3
"""Create presentation-ready plots from FTW evaluation CSV outputs."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "iou": "#0072B2",
    "precision": "#E69F00",
    "recall": "#009E73",
    "f1": "#CC79A7",
    "object_f1": "#D55E00",
    "pq": "#56B4E9",
}
CLASS_LABELS = {
    "background": "Background",
    "field_interior": "Field interior",
    "field_boundary": "Field boundary",
    "non_field": "Non-field",
    "field_extent": "Field extent",
}


def scratch_results_root() -> Path:
    return Path("/sfs/weka/scratch") / os.environ.get("USER", "unknown") / "ftw_results"


def latest_results_directory(root: Path) -> Path:
    """Return the newest results folder containing both required summary files."""
    candidates = [
        path
        for path in root.glob("job_*")
        if (path / "classwise_metrics.csv").is_file()
        and (path / "summary_metrics.csv").is_file()
    ]
    if not candidates:
        raise SystemExit(
            f"No completed evaluation result directories found below: {root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize results produced by evaluate_all_countries.py."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Evaluation directory. Defaults to the newest job under Rivanna scratch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Plot destination. Defaults to RESULTS_DIR/plots.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def annotate_bars(axis: plt.Axes, decimals: int = 2) -> None:
    """Add compact numeric values above vertical bars."""
    for container in axis.containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            labels.append("" if not np.isfinite(height) else f"{height:.{decimals}f}")
        axis.bar_label(container, labels=labels, padding=3, fontsize=8)


def combined_native_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics[
        (metrics["scope"] == "all_countries")
        & (metrics["evaluation"] == "native_3class")
    ].copy()
    if frame.empty:
        raise SystemExit("Combined native three-class metrics are missing.")
    frame["class"] = frame["class"].map(CLASS_LABELS).fillna(frame["class"])
    return frame.set_index("class")


def plot_native_classwise(axis: plt.Axes, metrics: pd.DataFrame) -> None:
    frame = combined_native_metrics(metrics)
    columns = ["iou", "precision", "recall", "f1"]
    frame[columns].plot(
        kind="bar",
        ax=axis,
        color=[COLORS[column] for column in columns],
        width=0.78,
    )
    axis.set_title("Native 3-class performance")
    axis.set_xlabel("")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.12)
    axis.tick_params(axis="x", rotation=0)
    axis.legend(["IoU", "Precision", "Recall", "F1"], ncol=2)
    annotate_bars(axis)


def positive_binary_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    selection = metrics[
        (metrics["scope"] == "all_countries")
        & (
            (
                (metrics["evaluation"] == "standard_binary")
                & (metrics["class"] == "field_interior")
            )
            | (
                (metrics["evaluation"] == "field_extent_2class")
                & (metrics["class"] == "field_extent")
            )
        )
    ].copy()
    selection["view"] = selection["evaluation"].map(
        {
            "standard_binary": "Interior vs rest",
            "field_extent_2class": "Interior + boundary vs background",
        }
    )
    return selection.set_index("view")


def plot_binary_comparison(axis: plt.Axes, metrics: pd.DataFrame) -> None:
    frame = positive_binary_metrics(metrics)
    columns = ["iou", "precision", "recall", "f1"]
    frame[columns].plot(
        kind="bar",
        ax=axis,
        color=[COLORS[column] for column in columns],
        width=0.72,
    )
    axis.set_title("Positive-class performance under two binary views")
    axis.set_xlabel("")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.12)
    axis.tick_params(axis="x", rotation=0)
    axis.legend(["IoU", "Precision", "Recall", "F1"], ncol=2)
    annotate_bars(axis)


def plot_country_performance(axis: plt.Axes, summary: pd.DataFrame) -> None:
    frame = summary[summary["scope"] != "all_countries"].copy()
    if frame.empty:
        frame = summary.copy()
    frame = frame.sort_values("native_macro_iou")
    y_positions = np.arange(len(frame))
    height = 0.25

    axis.barh(
        y_positions - height,
        frame["native_macro_iou"],
        height,
        label="Native macro IoU",
        color=COLORS["iou"],
    )
    axis.barh(
        y_positions,
        frame["standard_object_f1"],
        height,
        label="Object F1",
        color=COLORS["object_f1"],
    )
    axis.barh(
        y_positions + height,
        frame["panoptic_quality_pq"],
        height,
        label="Panoptic Quality",
        color=COLORS["pq"],
    )
    axis.set_yticks(y_positions, frame["scope"].str.replace("_", " ").str.title())
    axis.set_xlim(0, 1)
    axis.set_xlabel("Score")
    axis.set_title("Performance by country")
    axis.legend(loc="lower right")


def plot_object_quality(axis: plt.Axes, summary: pd.DataFrame) -> None:
    """Plot combined detection, overlap, and panoptic object metrics."""
    combined = summary[summary["scope"] == "all_countries"]
    if combined.empty:
        combined = summary.tail(1)
    row = combined.iloc[0]
    labels = ["Precision", "Recall", "F1", "Matched IoU", "SQ", "RQ", "PQ"]
    values = [
        row["standard_object_precision"],
        row["standard_object_recall"],
        row["standard_object_f1"],
        row["mean_matched_object_iou"],
        row["segmentation_quality_sq"],
        row["recognition_quality_rq"],
        row["panoptic_quality_pq"],
    ]
    bars = axis.bar(
        labels,
        values,
        color=[
            COLORS["precision"],
            COLORS["recall"],
            COLORS["f1"],
            COLORS["iou"],
            "#8A2BE2",
            "#999999",
            COLORS["pq"],
        ],
    )
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Score")
    axis.set_title("Combined object-level performance")
    axis.tick_params(axis="x", rotation=25)
    axis.bar_label(
        bars,
        labels=["" if not np.isfinite(value) else f"{value:.2f}" for value in values],
        padding=3,
        fontsize=9,
    )


def load_confusion(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "all_countries_confusion_3class.csv"
    if not path.is_file():
        raise SystemExit(f"Combined confusion matrix not found: {path}")
    return pd.read_csv(path, index_col=0)


def plot_confusion(axis: plt.Axes, confusion: pd.DataFrame) -> None:
    values = confusion.to_numpy(dtype=float)
    row_totals = values.sum(axis=1, keepdims=True)
    normalized = np.divide(
        values,
        row_totals,
        out=np.zeros_like(values),
        where=row_totals != 0,
    )
    image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")

    labels = [CLASS_LABELS.get(name, name) for name in confusion.columns]
    row_labels = [CLASS_LABELS.get(name, name) for name in confusion.index]
    axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Ground-truth class")
    axis.set_title("Normalized native 3-class confusion matrix")

    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{value:.1%}\n(n={int(values[row, column]):,})",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.55 else "black",
            )
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row proportion")


def save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Created: {path}")


def create_individual_plots(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    confusion: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 6))
    plot_native_classwise(axis, metrics)
    save_figure(figure, output_dir / "native_3class_metrics.png", dpi)

    figure, axis = plt.subplots(figsize=(11, 6))
    plot_binary_comparison(axis, metrics)
    save_figure(figure, output_dir / "binary_view_comparison.png", dpi)

    country_count = max(1, int((summary["scope"] != "all_countries").sum()))
    figure, axis = plt.subplots(figsize=(10, max(4.5, country_count * 0.42)))
    plot_country_performance(axis, summary)
    save_figure(figure, output_dir / "country_performance.png", dpi)

    figure, axis = plt.subplots(figsize=(8, 7))
    plot_confusion(axis, confusion)
    save_figure(figure, output_dir / "confusion_matrix_3class.png", dpi)

    figure, axis = plt.subplots(figsize=(11, 6))
    plot_object_quality(axis, summary)
    save_figure(figure, output_dir / "object_level_metrics.png", dpi)


def create_dashboard(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    confusion: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(18, 13))
    plot_native_classwise(axes[0, 0], metrics)
    plot_binary_comparison(axes[0, 1], metrics)
    plot_country_performance(axes[1, 0], summary)
    plot_confusion(axes[1, 1], confusion)
    figure.suptitle(
        "Fields of the World — Test-Set Evaluation", fontsize=20, weight="bold"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(figure, output_dir / "evaluation_dashboard.png", dpi)

    # PDF is convenient for slides, papers, and lossless enlargement.
    figure, axes = plt.subplots(2, 2, figsize=(18, 13))
    plot_native_classwise(axes[0, 0], metrics)
    plot_binary_comparison(axes[0, 1], metrics)
    plot_country_performance(axes[1, 0], summary)
    plot_confusion(axes[1, 1], confusion)
    figure.suptitle(
        "Fields of the World — Test-Set Evaluation", fontsize=20, weight="bold"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(figure, output_dir / "evaluation_dashboard.pdf", dpi)


def main() -> None:
    args = parse_args()
    results_dir = (
        args.results_dir.expanduser().resolve()
        if args.results_dir
        else latest_results_directory(scratch_results_root())
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else results_dir / "plots"
    )

    metrics_path = results_dir / "classwise_metrics.csv"
    summary_path = results_dir / "summary_metrics.csv"
    if not metrics_path.is_file() or not summary_path.is_file():
        raise SystemExit(
            "The results directory must contain classwise_metrics.csv and "
            f"summary_metrics.csv: {results_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    metrics = pd.read_csv(metrics_path)
    summary = pd.read_csv(summary_path)
    confusion = load_confusion(results_dir)

    create_individual_plots(metrics, summary, confusion, output_dir, args.dpi)
    create_dashboard(metrics, summary, confusion, output_dir, args.dpi)
    print(f"\nAll visualizations saved in: {output_dir}")


if __name__ == "__main__":
    main()
