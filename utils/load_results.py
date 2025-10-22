import os
import numpy as np
import pandas as pd

RESULTS = 'results'

def load_clustering(experiment_type, algorithm_name, dataset_name, params, verbose=True):

    if dataset_name == 'PAMAP2':
        subject_number = params[0]
        if not (1 <= subject_number < 10):
            raise ValueError("Subject number must be between 1 and 10")

    elif dataset_name == 'MoCap':
        subject_number = params[0]
        if subject_number not in [1, 2, 3, 7, 8, 9, 10, 11, 14]:
            raise ValueError("Subject number must be in [1, 2, 3, 7, 8, 9, 10, 11, 14]")

    elif dataset_name == 'ActRecTut':
        subject_number = params[0]
        if subject_number not in [1, 2, 3, 4, 5, 6]:
            raise ValueError("Subject number must be in [1, 2, 3, 4, 5, 6]")
        
    elif dataset_name == 'UCRSEG':
        ts_name = params[0]
        if verbose:
            print('Loading file: UCRSEG', ts_name)

    elif dataset_name == 'USC-HAD':
        subject_number = params[0]
        if subject_number not in range(1, 15):
            raise ValueError("Subject number must be in between 1 and 14")
        target_number = params[1]
        if target_number not in range(1, 6):
            raise ValueError("Target number must be in between 1 and 6")
        if verbose:
            print('Loading file: USC-HAD subject', subject_number, 'target', target_number)

    elif dataset_name in ['Suturing', 'Needle_Passing', 'Knot_Tying']:
        subject = params[0]
        if subject not in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
            raise ValueError("Subject must be in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']")
        trial = params[1]
        if trial not in range(1, 6):
            raise ValueError("Trial must be in between 1 and 5")
        if verbose:
            print('Loading file:', dataset_name, 'subject', subject, 'trial', trial)

    base_path = os.path.join(RESULTS, experiment_type, 'clustering', algorithm_name)
    file_path = os.path.join(base_path, dataset_name + '_' + '_'.join(map(str, params)) + '.npy')
    if os.path.exists(file_path):
        data = np.load(file_path)
    else:
        print(f"File {file_path} does not exist.")
        return None
    return data

def load_score(experiment_type, algorithm_name, dataset_name, params, metrics=None):
    file_path = os.path.join(RESULTS, experiment_type, 'score', algorithm_name, dataset_name + '.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")
    
    df = pd.read_csv(file_path)
    ts_name = dataset_name + '_' + '_'.join(map(str, params))
    df = df[df['dataset'] == ts_name]
    
    if metrics is None:
        return df
    elif isinstance(metrics, str):
        if metrics not in df.columns:
            raise ValueError(f"Metric '{metrics}' not found in the file")
        return df[['dataset', metrics]]
    elif isinstance(metrics, list):
        for metric in metrics:
            if metric not in df.columns:
                raise ValueError(f"Metric '{metric}' not found in the file")
        return df[['dataset'] + metrics]
    else:
        raise TypeError("Metrics should be a string or a list of strings")
    
def load_metric(experiment_type, algorithm_name, dataset_name, params, metric):
    df = load_score(experiment_type, algorithm_name, dataset_name, params, metrics=metric)
    return df[metric].values[0]

def load_metrics(experiment_type, algorithm_name, dataset_name, metric, folder='score'):
    file_path = os.path.join(RESULTS, experiment_type, folder, algorithm_name, dataset_name + '.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")
    
    df = pd.read_csv(file_path)
    
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in the file")
    
    return df[metric].tolist()
