"""Backward-compatible re-export of the metrics from the ``tsseg_eval`` package.

This module used to contain a (slightly drifted) copy of the metric
implementations. To avoid divergence, it now simply re-exports the canonical
implementations from :mod:`tsseg_eval.metrics`.
"""

from sklearn.metrics import (  # noqa: F401
    adjusted_rand_score,
    mutual_info_score,
    normalized_mutual_info_score,
)

from tsseg_eval.metrics import (  # noqa: F401
    DEFAULT_SMS_WEIGHTS,
    compute_boundaries_symmetrical,
    compute_boundary_distances,
    compute_segments,
    covering,
    f_score,
    labels_to_change_points,
    linear_distance,
    map_predicted_labels,
    state_matching_score,
    true_positives,
    weighted_adjusted_rand_score,
    weighted_contingency_matrix,
    weighted_normalized_mutual_info_score,
    weighted_pair_confusion_matrix,
    weighted_rand_score,
)
