"""Regression tests for the shared evaluation contract."""

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon

from evaluation import confidence, metrics


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


def test_semantic_pixel_record_ignores_unknown_pixels():
    semantic = np.array([[0, 1], [2, 3]])
    prediction = np.array([[0, 1], [0, 0]])
    record = metrics.semantic_pixel_record(
        "belgium", "chip", semantic, prediction
    )
    assert record["native_3class"].sum() == 3
    assert record["native_3class"].tolist() == [
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
    ]


def test_instance_pixel_record_uses_union_as_field_extent():
    semantic = np.array([[0, 1], [2, 3]])
    masks = np.array(
        [
            [[False, True], [False, False]],
            [[False, False], [True, False]],
        ]
    )
    record = metrics.instance_pixel_record("belgium", "chip", semantic, masks)
    assert record["field_extent_binary"].tolist() == [[1, 0], [0, 2]]


def test_probability_record_excludes_unknown_and_scores_perfect_prediction():
    semantic = np.array([[0, 1], [2, 3]])
    probabilities = np.full((3, 2, 2), 0.01, dtype=float)
    probabilities[0, 0, 0] = 0.98
    probabilities[1, 0, 1] = 0.98
    probabilities[2, 1, 0] = 0.98
    probabilities[:, 1, 1] = (0.2, 0.3, 0.5)
    record = confidence.probability_chip_record(
        "belgium", "chip", semantic, probabilities
    )
    assert record["pixels"] == 3
    assert record["correct"] == 3
    assert record["brier_sum"] < 0.01


def test_topology_metrics_detect_component_count_error():
    gt = np.array([[1, 0, 2]])
    prediction = np.array([[1, 1, 1]])
    row = metrics.topology_metrics(gt, prediction, np.ones_like(gt, dtype=bool))
    assert row["gt_betti0"] == 2
    assert row["prediction_betti0"] == 1
    assert row["betti0_error"] == -1


def test_exact_minimum_vertex_cut_finds_smallest_raster_separator():
    component = np.ones((3, 5), dtype=bool)
    source = np.zeros_like(component)
    target = np.zeros_like(component)
    source[:, 0] = True
    target[:, -1] = True
    cut, cost = metrics._minimum_vertex_cut(
        component, source, target, np.ones(component.shape)
    )
    assert cut.sum() == 3
    assert cost == 3
    remaining = component & ~cut
    labels, _ = metrics.ndimage.label(remaining, metrics.N4)
    source_labels = set(labels[source]) - {0}
    target_labels = set(labels[target]) - {0}
    assert not source_labels & target_labels


def test_turning_distance_is_invariant_to_translation_rotation_and_scale():
    reference = Polygon([(0, 0), (4, 0), (4, 1), (2, 3), (0, 1)])
    transformed = affinity.translate(
        affinity.rotate(affinity.scale(reference, 2.5, 2.5), 37),
        xoff=20,
        yoff=-8,
    )
    assert metrics.turning_distance(reference, transformed) < 1e-6


def test_turning_distance_separates_different_forms():
    square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    triangle = Polygon([(0, 0), (2, 0), (1, 2)])
    assert metrics.turning_distance(square, triangle) > 0.1
