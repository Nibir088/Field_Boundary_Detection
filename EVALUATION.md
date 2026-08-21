# FTW Evaluation Guide

The canonical formula-to-code implementation is documented in
[`evaluation/README.md`](evaluation/README.md) and implemented once in
`evaluation/metrics.py`. Both model runners import that shared module.

This project provides three complementary evaluators:

- `evaluate_all_countries.py` reports semantic pixel metrics and a lightweight
  connected-component object summary.
- `evaluate_instance_boundaries.py` is the rigorous field-level evaluator. It
  uses FTW instance-ID masks as the reference and separately measures field
  detection, field shape, merges/splits, and boundary quality.
- `evaluate_delineate_anything.py` applies the same field-level definitions to
  the native instance masks produced by Delineate Anything and also reports
  confidence-ranked AP50 and mAP50:95.

## Evaluation reference

The rigorous evaluator does not use a point, centroid, or bounding box as its
reference. Each reference field is the complete ground-truth instance stored in:

```text
label_masks/instance/<chip_id>.tif
```

Every nonzero raster value is a ground-truth field ID. Instance identities are
authoritative, while comparison geometry is restricted to pixels labeled class
1 in the semantic mask. This is a like-for-like comparison with the repository's
polygonization behavior, which treats class-1 interiors as field objects and
class-2 boundaries as separators. Predictions are produced by the native
three-class model:

```text
0 = background
1 = field interior
2 = field boundary
3 = unknown/nodata in the ground truth (ignored)
```

Predicted fields are the 4-connected components of class 1. The predicted
boundary class separates neighboring interiors, so each connected interior is a
candidate field object.

## Field matching

For every ground-truth field `G` and predicted field `P`, mask IoU is:

```text
IoU(G, P) = area(G intersect P) / area(G union P)
```

The evaluator builds the complete ground-truth-by-prediction IoU matrix for each
chip. It then performs maximum-cardinality, maximum-IoU bipartite assignment.
This enforces:

- one prediction can match at most one ground-truth field;
- one ground-truth field can match at most one prediction;
- the number of valid matches is maximized first;
- total IoU is maximized among assignments with that match count.

Results are reported at IoU thresholds 0.25, 0.50, and 0.75:

| Threshold | Practical interpretation |
|---|---|
| 0.25 | Approximate localization |
| 0.50 | Standard successful field detection |
| 0.75 | Accurate field delineation |

The threshold is always part of the definition of a correct detection. A field
with best IoU 0.31 is detected at IoU 0.25 but missed at IoU 0.50 and 0.75.

## Field detection metrics

For a selected IoU threshold:

```text
TP = one-to-one matched fields meeting the IoU threshold
FP = predicted fields without a valid match
FN = ground-truth fields without a valid match

Object precision = TP / (TP + FP)
Object recall    = TP / (TP + FN)
Object F1        = 2 * precision * recall / (precision + recall)
```

These support statements such as:

```text
800 known fields were detected correctly at IoU >= 0.50.
200 known fields were missed.
150 predictions were unmatched or insufficiently overlapping.
```

The evaluator also reports ground-truth count, prediction count, and signed
count error:

```text
count error = predicted objects - ground-truth objects
```

## Field shape and panoptic metrics

For correctly matched fields, the evaluator reports mean and median mask IoU.
It also reports:

```text
SQ = sum of matched IoUs / TP
RQ = TP / (TP + 0.5 FP + 0.5 FN)
PQ = sum of matched IoUs / (TP + 0.5 FP + 0.5 FN)
PQ = SQ * RQ
```

- Segmentation Quality (SQ) measures the shape quality of matched fields.
- Recognition Quality (RQ) measures whether fields were found without misses or
  spurious detections. Under one-to-one matching, RQ equals object F1.
- Panoptic Quality (PQ) combines detection and delineation quality.

A high SQ with low RQ means detected fields have good shapes, but many fields
are missed or spuriously predicted. A low SQ with high RQ means most fields are
found, but their shapes are inaccurate.

## Per-field tables

At the primary IoU threshold of 0.50, the evaluator writes a ground-truth-first
table:

```text
ground_truth_field_matches_iou50.csv
```

Important columns are:

| Column | Meaning |
|---|---|
| `gt_field_id` | Original instance ID from the FTW mask |
| `best_prediction_id` | Prediction having the largest IoU, even below 0.50 |
| `best_iou` | Best overlap available; zero when there is no overlap |
| `matched_prediction_id_iou50` | One-to-one assigned prediction at IoU >= 0.50 |
| `matched_iou_iou50` | Assigned IoU, or zero if missed |
| `detected_iou50` | Whether the field is a TP at IoU >= 0.50 |

The prediction-first table is:

```text
prediction_field_matches_iou50.csv
```

It distinguishes correct predictions from false or insufficiently overlapping
predictions. Numeric IoU is retained; subjective labels such as "excellent" or
"poor" are not assigned.

## Merge and split errors

A merge occurs when one predicted component substantially overlaps at least two
ground-truth fields. A split occurs when one ground-truth field substantially
overlaps at least two predictions.

The association test uses overlap coefficient:

```text
intersection / min(ground-truth area, prediction area) >= 0.10
```

The canonical report sweeps association alpha over 0.05, 0.10, 0.20, and 0.30
rather than silently relying on one value. Merge and split counts explain
topology failures that pixel IoU can hide. The `--association-threshold` option
is retained for command compatibility.

## Boundary metrics

Boundary pixels are evaluated separately from field objects. A connected
boundary network is not treated as an agricultural object because boundaries
branch, intersect, and are shared between neighboring fields.

For every chip, the evaluator reports:

- exact boundary IoU, precision, recall, and F1;
- boundary precision, recall, and F1 within a 1-pixel tolerance;
- boundary precision, recall, and F1 within a 2-pixel tolerance;
- average symmetric boundary distance in pixels;
- symmetric 95th-percentile Hausdorff distance in pixels.

FTW uses 10 m Sentinel-2 pixels, so one and two pixels correspond approximately
to 10 m and 20 m. Tolerant boundary F1 is important because an edge shifted by
one pixel may have almost no exact overlap while still being geographically
close.

Boundary CSV summaries are chip-macro averages: every chip contributes equally.
The per-chip CSV is retained for distributions and further statistical analysis.

## Presence-only labels

Belgium has presence/absence labels, so background and false-positive metrics
are meaningful. Rwanda is presence-only: outside known polygons, class 3 marks
unknown pixels that are excluded from evaluation.

Even after excluding unknown pixels, Rwanda precision must be interpreted with
caution near incomplete annotations. Recall on known fields and matched-field
shape metrics are generally more defensible than treating every unmatched
prediction as evidence of a nonexistent field.

## Running on Rivanna Open OnDemand

Start an Open OnDemand session with one GPU, open a terminal, and run:

```bash
cd /path/to/Field_Boundary_Detection
module purge
module load miniforge/24.11.3-py3.12
source .Field_Boundary/bin/activate

uv run python evaluate_instance_boundaries.py
```

Default paths are:

```text
Dataset: /sfs/weka/scratch/$USER/ftw_data/ftw
Model:   /sfs/weka/scratch/$USER/ftw_models/FTW_PRUE_EFNET_B5.ckpt
Output:  /sfs/weka/scratch/$USER/ftw_results/instance_job_<job-or-time-id>
```

Custom paths can be supplied:

```bash
uv run python evaluate_instance_boundaries.py \
  --data-dir /path/to/ftw \
  --model /path/to/model.ckpt \
  --output-dir /path/to/results \
  --gpu 0 \
  --batch-size 32 \
  --num-workers 4
```

## Output files

| File | Contents |
|---|---|
| `object_summary_by_threshold.csv` | Country and combined TP/FP/FN, detection, IoU, SQ/RQ/PQ, counts, merges, and splits at 0.25/0.50/0.75 |
| `ground_truth_field_matches_iou50.csv` | One row per known field, including best and assigned IoU |
| `prediction_field_matches_iou50.csv` | One row per predicted field, including best and assigned IoU |
| `boundary_metrics_by_chip.csv` | Exact, tolerant, and distance boundary metrics for each chip |
| `boundary_summary.csv` | Country and combined macro-average boundary results |
| `geometry_by_matched_field.csv` | Paired geometry descriptors/deltas for IoU-0.50 TP pairs |
| `geometry_population_summary.csv` | Population one-Wasserstein geometry distances |
| `geometry_border_exclusions.csv` | Truncated objects excluded from geometry |
| `closing_radius_sweep.csv` | Semantic object F1 after boundary closing at 0–3 px |
| `merge_repair_distance.csv` | Per-breached-pair greedy repair upper bounds |
| `merge_repair_summary.csv` | Repair-cost distribution and fractions within 1/2/3 px |
| `boundary_probability_sweep.csv` | Semantic pooled PQ and merge/split balance by p2 threshold |
| `boundary_probability_best_pq.csv` | Maximum semantic PQ and its boundary threshold |

## Recommended presentation

For every country and combined, present:

```text
Known fields, predicted fields, count error
TP, FP, FN at IoU >= 0.50
Object precision, recall, and F1 at IoU >= 0.50
Mean and median matched-field IoU
Detection rates at IoU >= 0.25, 0.50, and 0.75
SQ, RQ, and PQ
Merge and split counts
Exact boundary F1
Boundary F1 at 1-pixel and 2-pixel tolerance
Average symmetric boundary distance and Hausdorff-95
```

Always state the matching threshold, boundary tolerance, reference label type,
and whether country results are pooled or macro-averaged.

## Delineate Anything evaluation

The FTW command-line interface can run Delineate Anything inference, but it does
not provide the detailed FTW test-set breakdown described above. Use
`evaluate_delineate_anything.py` to evaluate any of the three registered
variants:

- `DelineateAnything-S`: smaller version 1 model;
- `DelineateAnything`: standard version 1 model;
- `DelineateAnythingV2`: standard version 2 model.

Unlike the PRUE semantic model, Delineate Anything directly predicts a set of
possibly overlapping field masks with a confidence score for each mask. The
evaluator preserves these native instances instead of converting their union
into connected components. It uses the complete nonzero shapes in
`label_masks/instance/<chip_id>.tif` as the ground-truth fields and ignores only
class-3 unknown/nodata pixels from the semantic mask.

The model accepts one temporal window and uses its first three RGB channels.
Choose `--window window_a` or `--window window_b`; the default is `window_b`.
This is an input difference from the default PRUE model, which uses both
temporal windows. Report the selected window in every comparison.

### Run interactively on a Rivanna GPU

In an Open OnDemand GPU session:

```bash
cd /path/to/Field_Boundary_Detection
module purge
module load miniforge/24.11.3-py3.12
source .Field_Boundary/bin/activate

# Run once after cloning or whenever the dependency files change.
uv sync --extra delineate-anything

uv run --extra delineate-anything python evaluate_delineate_anything.py \
  --model DelineateAnythingV2 \
  --window window_b \
  --gpu 0 \
  --batch-size 16
```

The defaults use these Rivanna locations:

```text
Dataset: /sfs/weka/scratch/$USER/ftw_data/ftw
Models:  /sfs/weka/scratch/$USER/ftw_models
Output:  /sfs/weka/scratch/$USER/ftw_results/delineate_job_<job-or-time-id>
```

If the selected checkpoint exists in the models directory under its expected
name, it is loaded locally. Otherwise, the registered model URL and normal
Ultralytics cache behavior are used. An explicit checkpoint always takes
precedence:

```bash
uv run --extra delineate-anything python evaluate_delineate_anything.py \
  --model DelineateAnything \
  --model-path /sfs/weka/scratch/$USER/ftw_models/DelineateAnything.pt \
  --output-dir /sfs/weka/scratch/$USER/ftw_results/delineate_v1
```

To compare all variants, run each in a distinct output directory:

```bash
uv run --extra delineate-anything python evaluate_delineate_anything.py \
  --model DelineateAnything-S \
  --output-dir /sfs/weka/scratch/$USER/ftw_results/delineate_s

uv run --extra delineate-anything python evaluate_delineate_anything.py \
  --model DelineateAnything \
  --output-dir /sfs/weka/scratch/$USER/ftw_results/delineate_v1

uv run --extra delineate-anything python evaluate_delineate_anything.py \
  --model DelineateAnythingV2 \
  --output-dir /sfs/weka/scratch/$USER/ftw_results/delineate_v2
```

### Delineate Anything metric breakdown

The evaluator reports two related field-detection views.

The fixed-threshold view uses the same maximum-cardinality, maximum-IoU
one-to-one assignment described in [Field matching](#field-matching). It reports
TP, FP, FN, object precision, recall, F1, matched-mask IoU, SQ, RQ, and PQ at IoU
0.25, 0.50, and 0.75. The per-field tables retain the best numeric IoU even
when a field fails the IoU 0.50 detection rule.

The ranking view sorts predictions by model confidence. At every IoU threshold,
each prediction greedily claims its best unmatched ground-truth field on the
same chip. It reports:

```text
AP50       = 101-point interpolated average precision at IoU >= 0.50
mAP50:95   = mean AP over IoU 0.50, 0.55, ..., 0.95
```

AP50 emphasizes whether fields are detected. mAP50:95 is stricter because high
IoU thresholds require accurate shapes. These values are computed from
predictions retained after `--conf-threshold`; keep that threshold low, and
constant across models, for a meaningful precision-recall curve. The default is
0.05. `--max-detections` defaults to 300 so chips containing many fields are
less likely to be artificially truncated.

Merge and split counts use the same 0.05/0.10/0.20/0.30 association sweep as
the semantic evaluator. The evaluator also compares the union of one-pixel
inner edges of predicted instance masks to the FTW semantic boundary class.
Delineate Anything has no separately predicted boundary class, so this boundary
score measures boundaries derived from its field masks and is not identical in
meaning to PRUE's native class-2 boundary score.

### Delineate Anything output files

| File | Contents |
|---|---|
| `object_summary_by_threshold.csv` | Per-country and pooled counts, TP/FP/FN, precision/recall/F1, matched IoU, SQ/RQ/PQ, merges, and splits |
| `average_precision_summary.csv` | Per-country and pooled AP50, AP50:95 components, and mAP50:95 |
| `ground_truth_field_matches_iou50.csv` | Every known field, its best IoU, assigned prediction, and detected/missed result |
| `prediction_field_matches_iou50.csv` | Every prediction, confidence, best IoU, assigned field, and correct/incorrect result |
| `boundary_metrics_by_chip.csv` | Exact, tolerant, and distance-based boundary metrics for every chip |
| `boundary_summary.csv` | Per-country and combined chip-macro boundary averages |
| `run_settings.csv` | Model, checkpoint source, temporal window, and inference thresholds |

### Fair comparison and parameter selection

Use identical FTW countries, the predefined `test` split, valid-pixel masking,
and IoU thresholds for all models. Compare fixed-IoU object metrics directly;
compare AP only when predictions from every model have confidence scores and use
the same AP procedure. State that Delineate Anything uses one RGB window while
the default PRUE model uses two four-band windows.

Do not select the temporal window, confidence threshold, NMS threshold, resize
factor, or maximum detections by looking at test performance. Select them on the
validation split, freeze them, and then run the test evaluator once. Rwanda is
presence-only, so its unmatched-prediction precision and AP remain less certain
than recall and matched-field shape quality, as described above.
