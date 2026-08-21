# Evaluation implementation

Both evaluation pipelines use `evaluation/metrics.py`; model runners only load
data and predictions. The legacy top-level scripts remain as compatible entry
points. New commands are:

```bash
python -m evaluation.instance_boundary
python -m evaluation.delineate_anything
```

Both runners now create a `plots/` directory automatically. Disable this with
`--skip-plots`, or regenerate figures from any compatible result directory:

```bash
python -m evaluation.plots --results-dir /path/to/results
```

They also create one multi-page diagnostic PDF per test chip:

```text
RESULTS_DIR/sample_pdfs/<country>/<image-file-stem>.pdf
```

Page 1 contains Sentinel-2 RGB, semantic reference, instance reference,
prediction, and boundary/extent diagnostics. Following pages contain that
sample's object metrics at every IoU threshold, alpha-sweep topology counts,
pixel metrics, exact/tolerant/distance boundary metrics, geometry summaries,
closure and boundary-probability sweeps where available, repair distances, and
Delineate Anything confidence summaries. Use `--skip-sample-pdfs` if storage or
runtime is more important; a full-data run can create thousands of PDF files.

## Rivanna SLURM job

Edit the configuration block near the top of `evaluation/run_evaluation.slurm`.
Select the runner with one line:

```bash
EVALUATOR="delineate_anything"
# or
EVALUATOR="instance_boundary"
```

Also verify the checkpoint paths, temporal window, allocation, wall time, and
batch size. Submit from the repository root so `SLURM_SUBMIT_DIR` resolves to
the correct checkout:

```bash
cd /path/to/Field_Boundary_Detection
sbatch evaluation/run_evaluation.slurm
```

Monitor and inspect the job with:

```bash
squeue -u "$USER"
tail -f ftw-evaluation-JOB_ID.out
```

Each job writes CSV files and figures below
`/sfs/weka/scratch/$USER/ftw_results/<evaluator>_job_<job-id>`.

## Shared symbols and object construction

`Y` is the `{0,1,2,3}` semantic reference, `L` contains reference instance IDs,
`hatY` is the predicted semantic class, `p2` is boundary probability, `V = Y !=
3`, and the Sentinel-2 scale is 10 metres per pixel. Four-connectivity defines
field components; eight-connectivity defines Chebyshev boundary tolerance.

Reference semantic objects are `L ∩ {Y=1}`. Semantic prediction components are
labeled before applying `V`, preventing unknown regions from acting as false
separators. Delineate Anything retains its native, possibly overlapping masks.

Intersections use one histogram pass. IoU is `C / (a + b - C)`. One-to-one
assignment maximizes cardinality first and valid-pair IoU second:

```text
S = 1[J >= tau] * (min(m,n)+1) + J * 1[J >= tau]
```

The second indicator is intentional: discarded pairs do not affect the
tie-break. Detection and pooled SQ/RQ/PQ are reported at IoU 0.25, 0.50, and
0.75. TP/FP/FN are summed across chips before ratios are recomputed.

## Merge, split, and closure

Merge/split association is `C / min(a,b) >= alpha`. Alpha is swept over 0.05,
0.10, 0.20, and 0.30. Each result includes split fields, merged predictions,
and reference fields lost to a merge.

The semantic runner sweeps boundary closing radii 0–3 pixels and recomputes
object F1. Both runners identify breached field pairs sharing a merged predicted
object and solve an exact pairwise minimum node cut on the cropped four-connected
raster graph. Pixel nodes are split into in/out nodes and solved by maximum
flow; eroded field cores are protected with effectively infinite capacity.
The output reports the exact minimum number of pixels needed to separate each
field pair and the fractions repairable within 1, 2, and 3 pixels.

For semantic predictions, a second exact pairwise cut uses integer-scaled
`-log(p2 + epsilon)` node costs, favoring separators supported by boundary
probability. Pairwise optimality does not imply that independently combining
cuts is the globally optimal joint repair for a prediction merging three or
more fields. `joint_multiway_cut_exact` is true only for two-field merges;
multi-terminal joint repair remains a separate multiway-cut problem.

## Boundary metrics

Exact IoU/precision/recall/F1 and tolerant precision/recall/F1 are chip-macro
averaged. Tolerance uses an eight-connected Chebyshev dilation at 1 and 2 pixels.
ASBD and HD95 use Euclidean distance transforms. Both cardinality-pooled and
direction-balanced ASBD are reported in pixels and metres. Every macro metric
has a `_valid_count` column because NaN values are skipped.

The semantic model additionally sweeps `p2` thresholds from 0.10 through 0.90
in steps of 0.05, rebuilding fields and reporting PQ, merge, and split behavior.
The maximum PQ and its threshold are saved separately. Delineate Anything has no
native semantic boundary probability, so this sweep is not fabricated for it.

## Pixel metrics

The semantic runner reports a native three-class confusion matrix plus two
binary views: field interior versus everything else, and complete field extent
(interior + boundary) versus non-field. Delineate Anything has no background,
interior, and boundary class logits, so it is evaluated only on the defensible
binary field-extent view formed by the union of its native instance masks.

Pixel IoU, precision, recall, F1, support, and raw confusion counts are pooled
within each country and across all countries. Unknown class-3 pixels are always
excluded. Do not present Delineate Anything as having a native three-class or
boundary-class pixel score.

## Confidence-aware evaluation

The semantic runner retains the complete softmax tensor `(p0, p1, p2)` from the
same forward pass. It reports exact pooled multiclass NLL and Brier score,
predictive entropy, and top-label ECE. ECE uses a mergeable 200-bin histogram
and reconstructs 15 approximately equal-mass bins, avoiding storage of every
pixel probability. Calibration is reported overall and separately at reference
boundary distances `0`, `1`, `2`, `3–4`, `5–9`, and `10+` pixels. Every stratum
includes its pixel count.

Semantic field confidence includes boundary-threshold persistence, mean
interior probability, and mean top-class confidence. Persistence is the longest
contiguous threshold interval over which a baseline field retains IoU at least
0.50 with a component decoded at `p2 >= lambda`. Reliability outputs include:

- AUROC for predicting IoU-0.50 TP versus FP;
- Spearman correlation with best matched IoU;
- partial Spearman correlation controlling for ranked log area; and
- area-only AUROC and correlation baselines.

Delineate Anything uses its native YOLO instance confidence for the same
ranking tests. It does not receive semantic calibration, entropy, persistence,
or `p2` metrics because those quantities are unavailable.

Risk–coverage retains the highest-scored predictions, rematches at every
coverage, and recomputes object F1 against all reference fields. Risk is
`1 - object F1`; lower AURC is better. Area is included as a mandatory selection
baseline. Ranking metrics are invariant to monotone recalibration, while ECE,
Brier, NLL, and breach probability require meaningful absolute probabilities.

For every semantic merge, the unweighted exact minimum separator is augmented
with the maximum, minimum, mean, and 10th-percentile `p2` along its cut pixels.
Maximum `p2` in `[0.30, 0.49]` is labeled a decoding-failure candidate and below
`0.10` a representation-failure candidate. These are diagnostic labels, not
causal proof. Confidence is measured on the unweighted cut to avoid selecting a
high-confidence separator and then using the same confidence as validation.

No temperature scaling is applied automatically. A positive scalar temperature
would leave multiclass argmax unchanged but would change probability calibration
and every explicit `p2` threshold analysis; any calibrated experiment must save
and report its temperature separately.

## Partition topology

Each chip reports variation of information between reference and predicted
field partitions, plus foreground Betti-0 and Betti-1 values and signed errors.
For overlapping Delineate Anything masks, topology is measured on the connected
components of their union; native instance counts remain in the object tables.

## Figures

The shared plotter creates, when the required CSV is available:

- pixel class metrics and a row-normalized confusion matrix;
- object precision/recall/F1/PQ versus matching IoU;
- country object F1 and PQ at IoU 0.50;
- exact versus 1 px and 2 px tolerant boundary F1;
- merge/split counts across association alpha;
- matched-field geometry headline diagnostics;
- closure-radius sensitivity and the merge repair-distance CDF;
- reliability diagrams, boundary-distance ECE, confidence–IoU, risk–coverage,
  breach-confidence, and topology plots;
- AP versus mask-IoU threshold for Delineate Anything; or
- PQ versus boundary-probability threshold for the semantic model.

Figures use pooled results only where the underlying metric is micro and the
chip-macro boundary summary where boundary metrics are macro. Raw CSV files
remain the authoritative numerical result.

## Geometry

Objects are polygonized and both sides are simplified identically at one pixel.
Border-touching objects are excluded and counted. For IoU-0.50 TP pairs the
outputs contain paired prediction-minus-reference deltas for area, perimeter,
Polsby–Popper compactness, solidity, rectangularity, elongation, holes, vertices,
and right-angle fraction, plus prediction/reference perimeter ratio.

Population-level one-Wasserstein distance is computed separately using every
non-border object. Headline geometry outputs are median perimeter ratio,
right-angle fraction for predictions and references, and predicted interior
rings per 100 matched fields.

## Interpretation

Object results are pooled (micro), boundary results are chip-macro, and geometry
results are per-pair medians plus population distributions. For presence-only
countries, precision, F1, RQ, PQ, and AP can be downward-biased; emphasize recall
and SQ. Always report IoU threshold, Chebyshev tolerance, Euclidean distance,
alpha sweep, border exclusion count, aggregation scheme, and valid counts.
