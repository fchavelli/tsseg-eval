
import numpy as np


def convert_to_borders(segmentation, multivariate_time_series):
    """
    Convert the given segmentation (discrete labels for each time unit or a probability
    for each time unit) to a segment boundaries.

    :param segmentation: The segmentation to convert
    :param multivariate_time_series: The multivariate time series data

    :return: The segment boundaries
    """
    # From probabilistic or discrete segmentation to borders
    if len(segmentation.shape) == 2:
        segmentation_labels = np.argmax(segmentation, axis=0)
    elif segmentation.shape[0] < multivariate_time_series[0].shape[0]:
        return segmentation
    else:
        segmentation_labels = segmentation

    borders = []
    for i in range(1, len(segmentation_labels)):
        if segmentation_labels[i] != segmentation_labels[i-1]:
            borders.append(i)

    return np.array(borders)


def convert_to_probabilistic_segmentation(segmentation, multivariate_time_series):
    """
    Convert the given segmentation (boundaries or discrete labels) to a probability
    distribution in a naive way. This is by having probability either 0 or 1 for each
    segment.

    This method is not used for the final evaluation.

    :param segmentation: The segmentation to convert
    :param multivariate_time_series: The multivariate time series

    :return: The probability distribution over the various semantic segments
    """
    if len(segmentation.shape) == 2:
        return segmentation
    elif segmentation.shape[0] < multivariate_time_series[0].shape[0]:
        if len(segmentation) == 0:
            return np.ones((1, multivariate_time_series[0].shape[0]))
        discrete_segmentation = np.zeros(multivariate_time_series[0].shape[0], dtype=int)
        borders = sorted(segmentation)
        previous = borders[0]
        counter = 1
        for border in borders[1:]:
            discrete_segmentation[previous:border] = counter
            previous = border
            counter += 1
        discrete_segmentation[previous:] = counter
    else:
        discrete_segmentation = segmentation

    probabilistic_segmentation = np.zeros((np.max(discrete_segmentation)+1, discrete_segmentation.shape[0]))
    for t in range(discrete_segmentation.shape[0]):
        probabilistic_segmentation[discrete_segmentation[t], t] = 1.0

    return probabilistic_segmentation
