from typing import Union

import pandas as pd
import numpy as np


def format_raw_values(data: Union[pd.DataFrame, np.array], ground_truth):
    """
    Format the raw values in the time series data

    :param data: The data that has been read from disk
    :param ground_truth: The corresponding ground truth

    :return: A formatted version of the time series. That is a pandas DataFrame for each
             attribute in the time series with the columns 'time' and 'average_value'. For
             time series that have no direct time measure (i.e., the i^th measurement).
             Further, the ground truth labels are converted to the same index.
    """
    if type(data) == pd.DataFrame:
        formatted_data = [
            pd.DataFrame(
                {
                    'average_value': data[column],
                    'time': data['time']
                }
            )
            for column in data.columns[1:]
        ]
    elif len(data.shape) == 1:
        formatted_data = [
            pd.DataFrame(
                {
                    'average_value': data,
                    'time': [np.datetime64(0, 'Y') + i * np.timedelta64(1, 'h') for i in range(len(data))]
                }
            )
        ]
    else:
        formatted_data = [
            pd.DataFrame(
                {
                    'average_value': data[row, :],
                    'time': [np.datetime64(0, 'Y') + i * np.timedelta64(1, 'h') for i in range(data.shape[1])]
                }
            )
            for row in range(data.shape[0])
        ]
    formatted_ground_truth = ground_truth
    if type(ground_truth['segmentation']) == list:
        formatted_ground_truth['segmentation'] = [formatted_data[0].loc[b, 'time'] for b in ground_truth['segmentation']]

    return formatted_data, formatted_ground_truth
