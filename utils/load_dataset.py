import os
import random
import scipy.io
import numpy as np
import pandas as pd

from baselines.E2USD.utils import *
from baselines.E2USD.params import *

DATA_PATH = 'data/'

def load_ts(dataset_name, params, verbose=True, return_params=False):
    
    if dataset_name == 'PAMAP2':

        subject_number = params[0]
        if not (1 <= subject_number < 10):
            raise ValueError("Subject number must be between 1 and 10")
        
        ts_path = os.path.join(DATA_PATH, 'PAMAP2/Protocol/subject10' + str(subject_number) + '.dat')

        if verbose:
            print('Loading file: PAMAP2 subject', str(subject_number))

        df = pd.read_csv(ts_path, sep=' ', header=None)
        data = df.to_numpy()
        groundtruth = np.array(data[:,1], dtype=int)
        hand_acc = data[:, 4:7]
        chest_acc = data[:, 21:24]
        ankle_acc = data[:, 38:41]
        data = np.hstack([hand_acc, chest_acc, ankle_acc])
        data = fill_nan(data)
        data = normalize(data)
    
    elif dataset_name == 'MoCap':

        subject_number = params[0]
        if subject_number not in [1, 2, 3, 7, 8, 9, 10, 11, 14]:
            raise ValueError("Subject number must be in [1, 2, 3, 7, 8, 9, 10, 11, 14]")
        
        subject_number = str(subject_number).zfill(2)
        ts_name = 'amc_86_' + subject_number + '.4d'
        ts_path = os.path.join(DATA_PATH, dataset_name, '4d/', ts_name)

        if verbose:
            print('Loading file: MoCap subject', subject_number)
        
        df = pd.read_csv(ts_path, sep=' ', usecols=range(0, 4))
        data = df.to_numpy()
        groundtruth = seg_to_label(dataset_info[ts_name]['label'])[:-1]

    elif dataset_name == 'ActRecTut':

        subject_number = params[0]
        if subject_number not in [1, 2, 3, 4, 5, 6]:
            raise ValueError("Subject number must be in [1, 2, 3, 4, 5, 6]")
        
        if verbose:
            print('Loading file: ActRecTut subject', subject_number)
    
        ts_path = os.path.join(DATA_PATH, dataset_name, f'subject{subject_number}_walk', 'data.mat')
        data = scipy.io.loadmat(ts_path)
        groundtruth = data['labels'].flatten()
        groundtruth = reorder_label(groundtruth)
        data = data['data'][:,0:10]
        data = normalize(data)

    elif dataset_name == 'UCRSEG':

        dataset_path = os.path.join(DATA_PATH, 'UCRSEG/')
        ts_name = params[0]

        if verbose:
            print('Loading file: UCRSEG', ts_name)

        info_list = ts_name[:-4].split('_')
        seg_info = {}
        i = 0
        for seg in info_list[2:]:
            seg_info[int(seg)] = i
            i += 1
        seg_info[len_of_file(dataset_path + ts_name)] = i
        n_states = len(seg_info)
        df = pd.read_csv(dataset_path + ts_name)
        data = df.to_numpy()
        data = normalize(data)
        groundtruth = seg_to_label(seg_info)

    elif dataset_name == 'USC-HAD':

        subject_number = params[0]
        if subject_number not in range(1, 15):
            raise ValueError("Subject number must be in between 1 and 14")
        
        target_number = params[1]
        if target_number not in range(1, 6):
            raise ValueError("Target number must be in between 1 and 6")
        
        if verbose:
            print('Loading file: USC-HAD subject', subject_number, 'target', target_number)

        data, groundtruth = load_USC_HAD(subject_number, target_number)
        data = normalize(data)

    if return_params:
        return data, groundtruth, params
    else:
        return data, groundtruth

def load_USC_HAD(subject, target):
    prefix = os.path.join('data/USC-HAD/Subject' + str(subject) + '/')
    fname_prefix = 'a'
    fname_postfix = 't' + str(target) + '.mat'
    data_list = []
    label_json = {}
    total_length = 0
    for i in range(1,13):
        data = scipy.io.loadmat(prefix + fname_prefix + str(i) + fname_postfix)
        data = data['sensor_readings']
        data_list.append(data)
        total_length += len(data)
        label_json[total_length] = i
    label = seg_to_label(label_json)
    return np.vstack(data_list), label

def load_random_ts(dataset_name, verbose=False):
    if dataset_name == 'PAMAP2':
        subject_number = random.randint(1, 9)
        return load_ts(dataset_name, [subject_number], verbose=verbose, return_params=True)

    if dataset_name == 'MoCap':
        subject_number = random.choice([1, 2, 3, 7, 8, 9, 10, 11, 14])
        return load_ts(dataset_name, [subject_number], verbose=verbose, return_params=True)

    if dataset_name == 'ActRecTut':
        subject_number = random.choice([1, 2, 3, 4, 5, 6])
        return load_ts(dataset_name, [subject_number], verbose=verbose, return_params=True)

    if dataset_name == 'UCRSEG':
        dataset_path = os.path.join(DATA_PATH, 'UCRSEG/')
        ts_name = random.choice(os.listdir(dataset_path))
        return load_ts(dataset_name, [ts_name], verbose=verbose, return_params=True)

    if dataset_name == 'USC-HAD':
        subject_number = random.randint(1, 14)
        target_number = random.randint(1, 5)
        return load_ts(dataset_name, [subject_number, target_number], verbose=verbose, return_params=True)

def load_data(dataset_name, verbose=True):

    if dataset_name == 'PAMAP2':
        for subject_number in range(1,10):
            data, groundtruth = load_ts(dataset_name, [subject_number], verbose=verbose)
            yield data, groundtruth, [subject_number]
    
    if dataset_name == 'MoCap':
        for subject_number in [1, 2, 3, 7, 8, 9, 10, 11, 14]:
            data, groundtruth = load_ts(dataset_name, [subject_number], verbose=verbose)
            yield data, groundtruth, [subject_number]

    if dataset_name == 'ActRecTut':
        for subject_number in [1, 2, 3, 4, 5, 6]:
            data, groundtruth = load_ts(dataset_name, [subject_number], verbose=verbose)
            yield data, groundtruth, [subject_number]

    if dataset_name == 'UCRSEG':
        dataset_path = os.path.join(DATA_PATH, 'UCRSEG/')
        for ts_name in os.listdir(dataset_path):
            data, groundtruth = load_ts(dataset_name, [ts_name], verbose=verbose)
            yield data, groundtruth, [ts_name]

    if dataset_name == 'USC-HAD':
        for subject_number in range(1,15):
            for target_number in range(1,6):
                data, groundtruth = load_ts(dataset_name, [subject_number, target_number], verbose=verbose)
                yield data, groundtruth, [subject_number, target_number]

def load_ts_info(dataset_name, params, verbose=False):
    data, groundtruth = load_ts(dataset_name, params, verbose=False)
    if verbose:
        print('Data shape:', data.shape)
        print('Groundtruth:', len(set(groundtruth)))
    return data.shape, len(set(groundtruth))