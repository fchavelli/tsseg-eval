import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster._supervised import mutual_info_score, entropy, _generalized_average
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from typing import List, Tuple, Dict, Union, Optional, Set, Any, Iterable

'''
F1 & Covering scores
Code adapted from TSSB (time-series-segmentation-benchmark)
https://github.com/ermshaua/time-series-segmentation-benchmark/blob/95a62b8e1e4e380313f187544c38f3400c1773e5/tssb/evaluation.py
Authors: Van den Burg, G.J.J. and Williams, C.K.I. from The Alan Turing Institute
'''

#------------------------------------------------------
# F1 score with margin, adapted from TSSB code
#------------------------------------------------------

def true_positives(T: Set[int], X: Set[int], margin: int = 5) -> Set[int]:
    """
    Compute true positives without double counting.
    
    Parameters
    ----------
    T : set
        True change-point locations.
    X : set
        Predicted change-point locations.
    margin : int, optional
        Maximum allowed distance between a true and a predicted change-point.
        
    Returns
    -------
    set
        A set of true change-points that have been matched.
    """
    X = set(X)  # make a copy so we don't modify the original
    TP = set()
    for tau in T:
        # Find all predictions x that are within the margin of tau
        close = [(abs(tau - x), x) for x in X if abs(tau - x) <= margin]
        close.sort()  # choose the closest one
        if not close:
            continue
        _, xstar = close[0]
        TP.add(tau)
        X.remove(xstar)  # remove the matched prediction to avoid double counting
    return TP

def f_score(annotation: Iterable[int], predictions: Iterable[int], margin: Optional[int] = None, alpha: float = 0.5, return_PR: bool = False) -> Union[float, Tuple[float, float, float]]:
    """
    Compute the F-measure for a single set of annotations.
    
    Both the annotation and predictions should be iterables of change-point 
    locations (0-based indices). The value 0 is always included by default.
    
    Parameters
    ----------
    annotation : iterable
        An iterable of true change-point locations.
    predictions : iterable
        An iterable of predicted change-point locations.
    margin : int, optional
        Maximum allowed distance between a true and predicted change-point.
    alpha : float, optional
        Weighting factor; alpha=0.5 corresponds to the F1-measure.
    return_PR : bool, optional
        If True, return a tuple (F, precision, recall). Otherwise, only F.
        
    Returns
    -------
    float or tuple
        The F-measure (and optionally precision and recall).
    """

    annotation_cps = labels_to_change_points(annotation)
    predictions_cps = labels_to_change_points(predictions)

    if margin is None:
        margin = int(0.01 * len(annotation_cps))

    # Ensure that 0 is included in both true annotations and predictions
    T = set(annotation_cps)
    T.add(0)
    
    X = set(predictions_cps)
    X.add(0)
    
    # Compute true positives
    TP = true_positives(T, X, margin=margin)
    
    # Compute precision and recall
    P = len(TP) / len(X) if len(X) > 0 else 0.0
    R = len(TP) / len(T) if len(T) > 0 else 0.0
    
    # Compute the F-measure using the weighted harmonic mean
    if P + R == 0:
        F = 0.0
    else:
        F = P * R / (alpha * R + (1 - alpha) * P)
    
    if return_PR:
        return F, P, R
    return F

#------------------------------------------------------
# Covering score, adaptation from TSSB & optimized version
#------------------------------------------------------

def labels_to_change_points(labels: Iterable[int]) -> List[int]:
    """
    Convert label sequence into change points (CPs).
    
    Parameters:
        labels (list or np.array): Label sequence.
    
    Returns:
        list: Change points (CPs) including start (0) and end (n+1).
    """
    labels_arr = np.asarray(labels)
    n = len(labels_arr)
    cp_indices = [0]  # Start at index 0
    for i in range(1, n):
        if labels_arr[i] != labels_arr[i - 1]:  # Detect change in labels
            cp_indices.append(i)
    cp_indices.append(n)  # Include last segment boundary
    return cp_indices

def compute_segments(cp_indices: List[int]) -> List[Tuple[int, int]]:
    """
    Convert change points to segment intervals.
    
    Parameters:
        cp_indices (list): List of change points.
    
    Returns:
        list: List of segment tuples (start, end).
    """
    return [(cp_indices[i], cp_indices[i + 1]) for i in range(len(cp_indices) - 1)]

def covering(ground_truth_labels: Iterable[int], predicted_labels: Iterable[int]) -> float:
    """
    Compute the covering score for segmentation using an optimized approach.
    
    Parameters:
        ground_truth_labels (list or np.array): Ground truth labels.
        predicted_labels (list or np.array): Predicted labels.
    
    Returns:
        float: Covering score.
    """
    n = len(list(ground_truth_labels))
    
    # Convert labels to change points and then to segments
    gt_cps = labels_to_change_points(ground_truth_labels)
    pred_cps = labels_to_change_points(predicted_labels)
    
    gt_segments = compute_segments(gt_cps)
    pred_segments = compute_segments(pred_cps)
    
    covering_score = 0.0
    p_idx = 0
    p_len = len(pred_segments)
    
    # Iterate over ground truth segments (each represented as [gt_start, gt_end))
    for gt_start, gt_end in gt_segments:
        segment_size = gt_end - gt_start
        max_iou = 0.0
        
        # Advance the pointer in predicted segments until the current segment might overlap
        while p_idx < p_len and pred_segments[p_idx][1] <= gt_start:
            p_idx += 1
        
        # Use a temporary pointer to check all predicted segments that overlap with the ground truth segment
        temp_idx = p_idx
        while temp_idx < p_len and pred_segments[temp_idx][0] < gt_end:
            pred_start, pred_end = pred_segments[temp_idx]
            
            # Calculate intersection and union without constructing sets
            inter = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
            union = (gt_end - gt_start) + (pred_end - pred_start) - inter
            iou = inter / union if union > 0 else 0.0
            
            max_iou = max(max_iou, iou)
            temp_idx += 1
        
        # Weight the IoU by the segment's length
        covering_score += segment_size * max_iou
    
    return covering_score / n


'''
WARI & WNMI scores
Code adapted from scikit-learn original implementation
https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73a86f5f11781a0e21f24c8f47979ec67/sklearn/metrics/cluster/_supervised.py
Authors: The scikit-learn developers
SPDX-License-Identifier: BSD-3-Clause
'''

#------------------------------------------------------
# Weighted Adjusted Rand Index (WARI)
#------------------------------------------------------

def compute_boundary_distances(labels: np.ndarray) -> np.ndarray:
    """
    Compute the distance from each point to the nearest change point (boundary)
    in the 1D label sequence.
    """
    n = len(labels)
    # Identify change points where label changes
    boundaries = np.where(np.diff(labels) != 0)[0]  # Identify change points
    boundaries = np.concatenate([boundaries, boundaries + 1])  # Add the next point for each change point
    boundaries = np.sort(boundaries)
    boundaries = np.concatenate([[0], boundaries, [n-1]])  # Add start and end points
    
    indices = np.arange(n)
    pos = np.searchsorted(boundaries, indices)
    distances = np.empty(n, dtype=float)
    
    # For indices before the first boundary:
    mask = (pos == 0)
    distances[mask] = boundaries[0] - indices[mask]
    
    # For indices after the last boundary:
    mask = (pos == len(boundaries))
    distances[mask] = indices[mask] - boundaries[-1]
    
    # For indices in between boundaries:
    mask = (pos > 0) & (pos < len(boundaries))
    left = indices[mask] - boundaries[pos[mask] - 1]
    right = boundaries[pos[mask]] - indices[mask]
    distances[mask] = np.minimum(left, right)
    
    return distances

def linear_distance(distances: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Compute the linear distance transformation for boundary distances.
    """
    return 1 + alpha * distances

def weighted_contingency_matrix(labels_true: np.ndarray, labels_pred: np.ndarray, distance: callable, *, eps: Optional[float] = None, sparse: bool = False, dtype: type = np.float64) -> Tuple[Union[np.ndarray, sp.spmatrix], np.ndarray]:
    """
    Build a weighted contingency matrix that describes the relationship between two partitions,
    weighting each sample by the associated weight (e.g., based on its distance to the boundary).
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    
    # Compute boundary distances for both clusterings
    d_true = compute_boundary_distances(labels_true)
    
    weights = distance(d_true)
    
    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")
    
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    
    # Build the weighted contingency matrix using coo_matrix.
    contingency = sp.coo_matrix(
        (weights, (class_idx, cluster_idx)),
        shape=(n_classes, n_clusters),
        dtype=dtype,
    )
    
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            contingency = contingency + eps
    return contingency, weights

def weighted_pair_confusion_matrix(labels_true: np.ndarray, labels_pred: np.ndarray, distance: callable) -> np.ndarray:
    """
    Compute the weighted pair confusion matrix from the weighted contingency matrix.
    """
    # Build the weighted contingency matrix (sparse for faster computation)
    contingency, weights = weighted_contingency_matrix(labels_true, labels_pred, distance, sparse=True, dtype=np.float64)
    
    # Sum of total weights
    total_weight = np.sum(weights)
    
    # Weighted sums by row and column
    row_sum = np.ravel(contingency.sum(axis=1))  # sum for each true class
    col_sum = np.ravel(contingency.sum(axis=0))  # sum for each predicted cluster
    
    # Calculate the sum of squares of the entries in the contingency matrix
    sum_squares = np.sum(contingency.data**2)
    
    # Build the weighted pair confusion matrix
    C = np.empty((2, 2), dtype=np.float64)
    C[1, 1] = sum_squares - total_weight
    C[0, 1] = contingency.dot(col_sum).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(row_sum).sum() - sum_squares
    C[0, 0] = total_weight**2 - C[0, 1] - C[1, 0] - sum_squares
    
    return C

def weighted_rand_score(labels_true: Iterable[int], labels_pred: Iterable[int], weights: callable) -> float:
    """
    Compute the weighted Rand Index (unadjusted) from the weighted pair confusion matrix.
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    C = weighted_pair_confusion_matrix(labels_true, labels_pred, weights)
    numerator = np.trace(C)
    denominator = np.sum(C)
    
    # Special cases: if no error is made or if there are no pairs (denom=0)
    if numerator == denominator or denominator == 0:
        return 1.0
    
    return numerator / denominator

def weighted_adjusted_rand_score(labels_true: Iterable[int], labels_pred: Iterable[int], distance: callable = linear_distance) -> float:
    """
    Compute the Weighted Adjusted Rand Index (WARI).
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    (tn, fp), (fn, tp) = weighted_pair_confusion_matrix(labels_true, labels_pred, distance)
    # convert to Python integer types, to avoid overflow or underflow
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0

    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))

#------------------------------------------------------
# Weighted Normalized Mutual Info Score (WNMI)
#------------------------------------------------------

def weighted_normalized_mutual_info_score(labels_true: Iterable[int], labels_pred: Iterable[int], distance: callable = linear_distance, *, average_method: str = "arithmetic") -> float:
    
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Special limit cases: no clustering since the data is not split.
    # It corresponds to both labellings having zero entropy.
    # This is a perfect match hence return 1.0.
    if (
        classes.shape[0] == clusters.shape[0] == 1
        or classes.shape[0] == clusters.shape[0] == 0
    ):
        return 1.0

    contingency, _ = weighted_contingency_matrix(labels_true, labels_pred, distance=distance, sparse=True)
    contingency = contingency.astype(np.float64, copy=False)
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)

    # At this point mi = 0 can't be a perfect match (the special case of a single
    # cluster has been dealt with before). Hence, if mi = 0, the nmi must be 0 whatever
    # the normalization.
    if mi == 0:
        return 0.0

    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)

    normalizer = _generalized_average(h_true, h_pred, average_method)
    return mi / normalizer


#------------------------------------------------------
# State Matching Score (SMS)
#------------------------------------------------------

def compute_boundaries_symmetrical(labels: np.ndarray) -> List[int]:
    """
    Computes the indices of the boundaries in the label sequence.
    
    Parameters:
        labels: numpy array of labels.
    
    Returns:
        List of indices where the label changes.
    """
    boundaries = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            boundaries.append(i)
    return boundaries

#------------------------------------------------------
# SMS
#------------------------------------------------------

def map_predicted_labels(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """
    Maps predicted labels to match true labels using the Hungarian algorithm,
    ensuring the number of unique output labels equals the number of unique
    input predicted labels.

    Parameters:
        labels_true: numpy array of ground truth labels.
        labels_pred: numpy array of predicted labels.

    Returns:
        mapped_pred: numpy array of predicted labels after mapping.
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    original_shape = labels_pred.shape

    if labels_pred.size == 0:
        return np.array([])

    all_unique_pred = np.unique(labels_pred)

    # Handle cases where true labels are empty or comparison is not possible
    if labels_true.size == 0:
        mapping = {label: i for i, label in enumerate(all_unique_pred)}
        mapped_pred_flat = np.array([mapping[val] for val in labels_pred.ravel()])
        return mapped_pred_flat.reshape(original_shape)

    # Determine common length for comparison
    compare_len = min(len(labels_pred.ravel()), len(labels_true.ravel()))
    if compare_len == 0: # No overlap to compare
        mapping = {label: i for i, label in enumerate(all_unique_pred)}
        mapped_pred_flat = np.array([mapping[val] for val in labels_pred.ravel()])
        return mapped_pred_flat.reshape(original_shape)

    true_comp = labels_true.ravel()[:compare_len]
    pred_comp = labels_pred.ravel()[:compare_len]

    unique_pred_comp = np.unique(pred_comp)
    unique_true_comp = np.unique(true_comp)

    # Cost matrix: negative overlap
    cost_matrix = np.zeros((len(unique_pred_comp), len(unique_true_comp)))
    for i, p_label in enumerate(unique_pred_comp):
        for j, t_label in enumerate(unique_true_comp):
            overlap = np.sum((pred_comp == p_label) & (true_comp == t_label))
            cost_matrix[i, j] = -overlap

    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    hungarian_map = {unique_pred_comp[i]: unique_true_comp[j] for i, j in zip(row_ind, col_ind)}

    # Build the final mapping ensuring unique outputs for all unique predicted labels
    final_map = {}
    used_targets = set()

    # 1. Apply Hungarian assignments where the target is available
    for p_label, t_label in hungarian_map.items():
        if t_label not in used_targets:
            final_map[p_label] = t_label
            used_targets.add(t_label)
        # else: Target taken, p_label will be handled in step 2

    # 2. Assign remaining unique predicted labels
    next_new_label = 0
    for p_label in all_unique_pred:
        if p_label not in final_map: # If not assigned yet
            # Try assigning p_label to itself if available
            if p_label not in used_targets:
                final_map[p_label] = p_label
                used_targets.add(p_label)
            # Otherwise, find the next available non-negative integer
            else:
                while next_new_label in used_targets:
                    next_new_label += 1
                final_map[p_label] = next_new_label
                used_targets.add(next_new_label)

    # Apply the final map to the original full predicted sequence
    mapped_pred_flat = np.array([final_map[val] for val in labels_pred.ravel()])
    mapped_pred = mapped_pred_flat.reshape(original_shape)

    return mapped_pred

def error_type(error_label: int, true_atomicity: int, p0: Optional[int], p1: Optional[int], t0: Optional[int], t1: Optional[int]) -> str:
    if true_atomicity == 1:
        # Delay
        # Check if the predicted error label matches either the preceding or succeeding true label.
        if (t0 is not None and p0 is not None and t0 == error_label and p0 == error_label) or (t1 is not None and p1 is not None and t1 == error_label and p1 == error_label):
            return "delay"
        
        # Isolation
        else:
            return "isolation"

    elif true_atomicity == 2:
    # Transition
    # True labels contains two different labels.
        return "transition"

    elif true_atomicity > 2:
        # Missing
        # True labels contain more than two different labels.
        return "missing"
    
    return "unknown"

def nearest_block_boundary_distance(true_boundaries: np.ndarray, block_start: int, block_end: int) -> float:
    """
    Computes the distance from the center of an error block to the nearest true boundary.

    Parameters:
        true_boundaries: List or numpy array of indices where true labels change.
        block_start: Start index of the error block.
        block_end: End index of the error block (inclusive).

    Returns:
        float: The minimum distance from the block center to a true boundary.
    """
    true_boundaries = np.asarray(true_boundaries)
    center = (block_start + block_end) / 2.0
    distances = np.abs(true_boundaries - center)
    min_dist = np.min(distances)

    return min_dist

def atomicity(sequence: np.ndarray) -> int:
    # Remove consecutive duplicates while preserving order
    if len(sequence) > 0:
        reduced_segment = [sequence[0]]
        for x in sequence[1:]:
            if x != reduced_segment[-1]:
                reduced_segment.append(x)
        sequence_atomicity = len(reduced_segment)
    else:
        sequence_atomicity = 0
    return sequence_atomicity

def state_matching_score(labels_true: Iterable[int], labels_pred: Iterable[int], weights: Dict[str, float] = {'delay': 0.1, 'transition': 0.3, 'isolation': 0.8, 'missing': 0.5}, return_mapped: bool = False, return_errors: bool = False) -> Union[float, Tuple[float, np.ndarray], Tuple[float, List[Dict[str, Any]]], Tuple[float, np.ndarray, List[Dict[str, Any]]], None]:
    """
    Computes a new State Matching Score (SMS) based on identifying and classifying
    error segments between true and predicted label sequences after optimal mapping.

    The function first maps predicted labels to true labels using an optimal
    assignment strategy. It then iterates through the sequences, identifying
    contiguous segments where the mapped predicted label differs from the true label
    and the predicted label within the segment is constant.

    Each error segment is classified into types ('delay', 'transition', 'isolation',
    'missing'). A weighted error score is calculated based on
    segment size, error type, distance to true boundaries, and segment atomicity.
    The penalty for each error type is determined by the input `weights` dictionary.

    The intended severity hierarchy for error types, from least to most severe, is
    typically: delay, transition, missing, isolation.
    The final score is normalized, where higher values indicate better agreement.

    Parameters:
        labels_true (array-like): Ground truth labels.
        labels_pred (array-like): Predicted labels.
        weights (dict): A dictionary mapping error type strings
                        ('delay', 'transition', 'isolation', 'missing')
                        to their respective penalty weights, reflecting their severity.
        return_mapped (bool, optional): If True, return the mapped predicted labels. Defaults to False.
        return_errors (bool, optional): If True, return a list of detected errors with details. Defaults to False.

    Returns:
        float or tuple:
            - If return_mapped is False and return_errors is False: The normalized State Matching Score.
            - If return_mapped is True and return_errors is False: (score, mapped_pred).
            - If return_mapped is False and return_errors is True: (score, errors_list).
            - If return_mapped is True and return_errors is True: (score, mapped_pred, errors_list).
            Returns None if input is empty.
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)
    if n == 0:
        return None # Return None for empty input

    # Map predicted labels to true labels
    mapped_pred = map_predicted_labels(labels_true, labels_pred)
    true_boundaries = np.unique([0] + compute_boundaries_symmetrical(labels_true) + [n])

    i = 0
    total_penalty = 0.0 # Renamed from error_score for clarity, represents the sum of penalties
    total_error_length = 0
    errors_list = [] if return_errors else None

    while i < n:
        if labels_true[i] != mapped_pred[i]:
            # Start of a potential error segment
            start_index = i
            error_label = mapped_pred[i]
            j = i + 1
            # Extend the segment as long as the error persists with the same predicted label
            while j < n and labels_true[j] != mapped_pred[j] and mapped_pred[j] == error_label:
                j += 1
            end_index = j - 1

            # Extract the segment of true labels corresponding to the error segment
            true_labels_segment = np.asarray(labels_true[start_index : end_index + 1])
            true_atomicity = atomicity(true_labels_segment)

            # Calculate segment size
            segment_size = end_index - start_index + 1
            total_error_length += segment_size

            # Determine neighbor values for predicted labels
            p0 = mapped_pred[start_index - 1] if start_index > 0 else None
            p1 = mapped_pred[end_index + 1] if end_index < n - 1 else None

            # Determine neighbor values for true labels
            t0 = labels_true[start_index - 1] if start_index > 0 else None
            t1 = labels_true[end_index + 1] if end_index < n - 1 else None

            # Call the error_type function with the required arguments
            err_type = error_type(error_label, true_atomicity, p0, p1, t0, t1)

            segment_penalty = 0.0
            weight = weights.get(err_type, 1.0) # Default weight if type not in dict

            if err_type == "delay":
                segment_penalty = weight * segment_size

            elif err_type == "transition" or err_type == "isolation":
                # Adjust the score based on the distance to the nearest true boundary
                # Nearest distance is at most 1/2 of the segment size
                min_dist = 2 * nearest_block_boundary_distance(true_boundaries, start_index, end_index) / n
                segment_penalty = weight * min_dist * segment_size

            elif err_type == "missing":
                
                #atomicity_penalty = 1 + (weight - 1) * (1 - 1 / true_atomicity) if weight != 1 else 1

                # Adjust the score based on the number of missed true labels. We want the penalty to:
                # - increase as true_atomicity and weight increase
                # - be equal to zero when weight is zero
                # - be equal to one when weight is one
                atomicity_penalty = 1.0 + (weight - 1.0) * (3.0 / true_atomicity)
                segment_penalty = weight * segment_size * atomicity_penalty

            total_penalty += segment_penalty

            if return_errors:
                errors_list.append({
                    'type': err_type,
                    'start': start_index,
                    'end': end_index,
                    'size': segment_size,
                    'penalty': segment_penalty # Store the calculated penalty for this segment
                })

            # Move the main loop index past this segment
            i = j
        else:
            # No error at this index, move to the next
            i += 1

    # Calculate the final score
    # The total error is the length of incorrect points plus the weighted penalty
    total_weighted_error = total_error_length + total_penalty
    score = 1.0 - total_weighted_error / n

    # Return based on flags
    if return_mapped and return_errors:
        return score, mapped_pred, errors_list
    elif return_mapped:
        return score, mapped_pred
    elif return_errors:
        return score, errors_list
    else:
        return score