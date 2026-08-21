"""Regression tests for the shared evaluation contract."""

import numpy as np

from evaluation import metrics


def test_unknown_region_does_not_split_prediction_component():
    semantic = np.array([[1, 3, 1]])
    instances = np.array([[7, 0, 7]])
    prediction = np.array([[1, 1, 1]])
    arrays, _, labels = metrics.semantic_object_arrays(
        semantic, instances, prediction
    )
    assert len(arrays.prediction_ids) == 1
    assert labels.tolist() == [[1, 0, 1]]


def test_match_score_masks_below_threshold_iou():
    arrays = metrics.ObjectArrays(
        gt_ids=np.array([1, 2]),
        prediction_ids=np.array([1, 2]),
        intersections=np.zeros((2, 2)),
        gt_areas=np.ones(2),
        prediction_areas=np.ones(2),
        ious=np.array([[0.51, 0.49], [0.99, 0.51]]),
    )
    assert metrics.match_at_threshold(arrays, 0.50) == [(0, 0), (1, 1)]


def test_native_instance_builder_accepts_zero_predictions():
    semantic = np.array([[0, 1], [1, 1]])
    instances = np.array([[0, 5], [5, 5]])
    masks = np.zeros((0, 2, 2), dtype=bool)
    arrays = metrics.mask_object_arrays(semantic, instances, masks)
    assert arrays.ious.shape == (1, 0)
    assert arrays.prediction_areas.shape == (0,)


def test_overlap_sweep_counts_merge_and_lost_fields():
    arrays = metrics.ObjectArrays(
        gt_ids=np.array([1, 2]),
        prediction_ids=np.array([1]),
        intersections=np.array([[4], [4]]),
        gt_areas=np.array([4, 4]),
        prediction_areas=np.array([8]),
        ious=np.array([[0.5], [0.5]]),
    )
    splits, merges, lost = metrics.overlap_relations(arrays, 0.10)
    assert (splits, merges, lost) == (0, 1, 2)


def test_boundary_metrics_report_balanced_distance_and_metres():
    target = np.zeros((5, 5), dtype=bool)
    prediction = np.zeros((5, 5), dtype=bool)
    target[2, 1] = True
    prediction[2, 2] = True
    row = metrics.boundary_metrics(target, prediction, metres_per_pixel=10)
    assert row["asbd_balanced_px"] == 1
    assert row["asbd_balanced_m"] == 10
