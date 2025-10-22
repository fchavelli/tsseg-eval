
import numpy as np
from typing import Dict, Any, Tuple
from baselines.patss.data_handling.read_data import read_data
from baselines.patss.data_handling.format_data import format_raw_values


def read_time_series(time_series_name) -> Tuple[np.ndarray, Dict[str, Any], np.array]:
    data, ground_truth = format_raw_values(*read_data(time_series_name))
    time_series = np.empty((data[0].shape[0], len(data)))
    time_steps = None
    for i, attribute in enumerate(data):
        time_series[:, i] = attribute['average_value']
        time_steps = attribute['time']
    return time_series, ground_truth, time_steps
