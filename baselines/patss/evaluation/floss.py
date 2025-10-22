
import numpy as np
from sklearn.metrics.pairwise import paired_euclidean_distances


def floss_evaluation(cps_pred, cps_true, ts_len):
    """
    Compute the floss evaluation score:
        https://github.com/ermshaua/time-series-segmentation-benchmark/blob/95a62b8e1e4e380313f187544c38f3400c1773e5/tssb/evaluation.py#L6

    :param cps_pred: The predicted change points (segment boundaries)
    :param cps_true: The ground truth change points
    :param ts_len: The total length of the time series

    :return: The FLOSS evaluation score, as described in 'Domain agnostic online semantic segmentation for multi-dimensional time series'
    """
    if len(cps_pred) == 0 or len(cps_true) == 0:
        return 1.5

    differences = 0

    for cp_pred in cps_pred:
        distances = paired_euclidean_distances(
            np.array([cp_pred]*len(cps_true)).reshape(-1, 1),
            cps_true.reshape(-1, 1)
        )
        cp_true_idx = np.argmin(distances, axis=0)
        cp_true = cps_true[cp_true_idx]
        differences += np.abs(cp_pred-cp_true)

    return differences / (len(cps_pred) * ts_len)
