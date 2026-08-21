# Evaluation implementation

Both evaluation pipelines use `evaluation/metrics.py`; model runners only load
data and predictions. The legacy top-level scripts remain as compatible entry
points. New commands are:

```bash
python -m evaluation.instance_boundary
python -m evaluation.delineate_anything
```

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
object and report a four-connected shortest-path repair upper bound, including
the fractions repairable within 1, 2, and 3 pixels. This is a greedy diagnostic,
not the intractable exact topology-edit minimum.

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
