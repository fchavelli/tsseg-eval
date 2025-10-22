
import os
import numpy as np
import pandas as pd

DATA_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/data/'
GROUND_TRUTH_DEFAULT_VALUE = {
    'segmentation': np.array([]),
    'segment_length': -1,
    'is_anomaly': [],
    'transition_areas': np.array([])
}


def set_data_directory(path):
    """
    Set the path of the directory containing the data to the given path.

    :param path: The path containing all the data.
    """
    global DATA_DIRECTORY
    DATA_DIRECTORY = path


def get_all_data_set_names(data_set):
    """
    Extract all the names within the given data set

    :param data_set: The name of the data set. This is the name of the directory (within the global
                     data directory) that contains all the single time series files.

    :return: A list of time series names. That is all the time series that are present in the directory
             given by the data set name. If this name has suffix '_all', then the time series will be
             extracted recursively in subdirectories.
    """
    path = DATA_DIRECTORY + data_set
    search_recursively = data_set.endswith('_all')
    if search_recursively:
        path = DATA_DIRECTORY + data_set[:-len('_all')]
        data_set = data_set[:-len('_all')]

    all_names = [data_set + '/' + file_name
            for file_name in os.listdir(path)
            if os.path.isfile(os.path.join(path, file_name)) and not file_name.endswith('.json')]

    if search_recursively:
        for directory in os.listdir(path):
            if os.path.isdir(os.path.join(path, directory)):
                all_names += get_all_data_set_names(data_set + '/' + directory + '_all')

    return all_names


def read_data(data_set: str):
    """
    Read the give data set

    :param data_set: The name (path within DATA_DIRECTORY) of the data set to read

    :return: The content of the file located at the given data_set name
    """
    path = DATA_DIRECTORY + data_set

    # Read the data according to its origin
    if data_set.startswith('floss/'):
        time_series, ground_truth = read_floss_dataset(path, True)
    elif data_set.startswith('autoplait/'):
        time_series, ground_truth = read_autoplait_dataset(path)
    elif data_set.startswith('google_trends/'):
        time_series, ground_truth = read_google_trends_dataset(path)
    elif data_set.startswith('clasp/'):
        time_series, ground_truth = read_clasp_dataset(path)
    elif data_set.startswith('synthetic/'):
        time_series, ground_truth = read_synthetic_data(path)
    elif data_set.startswith('icare/'):
        time_series, ground_truth = read_icare_data(path)
    else:
        raise NotImplementedError("No function implemented to read the dataset '%s'!" % data_set)

    # If some ground truth labels are not available in the given time series, fill those values in
    # with defaults indicating that there is not ground truth (e.g., an empty list if no ground truth
    # segmentation exists)
    for ground_truth_attribute in GROUND_TRUTH_DEFAULT_VALUE.keys():
        if ground_truth_attribute not in ground_truth:
            ground_truth[ground_truth_attribute] = GROUND_TRUTH_DEFAULT_VALUE[ground_truth_attribute]

    return time_series, ground_truth


def read_floss_dataset(path, allow_test_files=False):
    """
    Read a file from the FLOSS dataset, which is useful for segmenting time series.

    :param path: The path of the data file
    :param allow_test_files: Whether test files are allowed to be read and returned by this method.

    :return: A tuple of two elements
                - The time series data
                - A dictionary with the ground truths: the (estimated) segment length and border points
    """
    concrete_file_name = path[path.rindex('/') + 1:]

    if not allow_test_files and '/custom/' not in path:
        floss_path = path[:path.rindex('/') + 1]
        with open(floss_path + 'development_test_partition/test_set', 'r') as test_set_file:
            for test_file in test_set_file.readlines():
                if concrete_file_name == test_file.strip():
                    raise Exception('Trying to open a test file {0} from the FLOSS dataset, while not being allowed!', concrete_file_name)

    (_, segment_length, *borders_str) = concrete_file_name.split('_')
    ground_truth = {
        'segment_length': int(segment_length),
        'segmentation': np.array([int(border.split('.')[0]) for border in borders_str])
    }
    t = np.loadtxt(path)
    return t, ground_truth


def read_autoplait_dataset(path):
    """
    Read an AutoPlait data set

    :param path: The name of the data set

    :return: The data in the file and an empty ground truth
    """
    return np.loadtxt(path).transpose(), {}


def read_google_trends_dataset(path):
    """
    Read a Google trends data set

    :param path: The name of the data set

    :return: The data in the file and an empty ground truth
    """
    data = pd.read_csv(path)
    # Format the time column
    data[data.columns[0]] = pd.to_datetime(data[data.columns[0]])
    data.rename(columns={data.columns[0]: 'time'}, inplace=True)
    # Set type of values to integer
    data.replace(to_replace='<1', value=0, inplace=True)
    for column in data.columns[1:]:
        data[column] = data[column].astype(np.float)
    return data, {}


def read_clasp_dataset(path):
    """
    Read a ClaSP data set

    :param path: The name of the data set

    :return: The data in the file and ground truth segmentation and window length
    """
    dataset_name = os.path.basename(path)[:-4]  # Drop extension
    with open(os.path.join(DATA_DIRECTORY, "clasp", "util", "desc.txt"), 'r') as file:
        for line in file.readlines():
            line = line.split(",")

            if line[0] == dataset_name:
                (_, window_size), change_points = line[:2], line[2:]

                return np.loadtxt(fname=path, dtype=np.float64), {
                    'segment_length': int(window_size),
                    'segmentation': np.array([int(x) for x in change_points])
                }

    raise Exception("Clasp dataset '%s' was not found!" % dataset_name)


def read_synthetic_data(path):
    """
    Read a synthetic data set

    :param path: The name of the data set

    :return: The data in the file and ground truth transition area, ground truth
             segmentation probabilities, and ground truth window length
    """
    file_content = pd.read_csv(path)
    state_probabilities = []
    for col in file_content.columns:
        if col.startswith('state_probabilities'):
            state_probabilities.append(np.expand_dims(file_content[col], axis=0))
    window_size = int(path.split('_')[-1][:-4])
    return file_content['data'], {'transition_areas': file_content['transition_areas'], 'segmentation': np.concatenate(state_probabilities), 'segment_length': window_size}
