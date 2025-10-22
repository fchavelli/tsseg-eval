import os
from TSpy.label import seg_to_label
from TSpy.utils import remove_constant_col
import scipy.io
import numpy as np
import json
import pandas as pd

def load_SMD(data_path, use_data, remove_const=True):
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    label_path = os.path.join(data_path, 'test_label')
    df_train = pd.read_csv(os.path.join(train_path, use_data+'.txt'), header=None)
    df_test = pd.read_csv(os.path.join(test_path, use_data+'.txt'), header=None)
    if remove_const:
        df = remove_constant_col(pd.concat([df_train, df_test]))
        df_train = remove_constant_col(df_train)
        df_test = remove_constant_col(df_test)
    label = pd.read_csv(os.path.join(label_path, use_data+'.txt'), header=None)
    data = df.to_numpy()
    train_data = df_train.to_numpy()
    test_data = df_test.to_numpy()
    train_label = np.zeros(train_data.shape[0])
    test_label = label.to_numpy().flatten()
    label = np.concatenate([train_label, test_label])
    return data, label, train_data, train_label, test_data, test_label

def load_UCR(dataset, path):
    train_file = os.path.join(path+'/UCRArchive_2018', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join(path+'/UCRArchive_2018', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels

def load_UEA(dataset, path):
    path = os.path.join(path, dataset+'/'+dataset)
    train_data = np.load(path+'_TRAIN.npy')
    test_data = np.load(path+'_TEST.npy')
    train_labels = np.load(path+'_TRAIN_labels.npy')
    test_labels = np.load(path+'_TEST_labels.npy')
    with open(path+'_map.json') as f:
        label_map = json.load(f)
        f.close()
    return train_data, train_labels, test_data, test_labels, label_map

def load_general_seg_dagaset(path):
    data = np.load(path)
    mts = data[:,:-1]
    label = data[:,-1]
    # print(mts.shape, label.shape)
    return mts, label

def load_USC_HAD(subject, target, dataset_path):
    prefix = os.path.join(dataset_path,'USC-HAD/Subject'+str(subject)+'/')
    fname_prefix = 'a'
    fname_postfix = 't'+str(target)+'.mat'
    data_list = []
    label_json = {}
    total_length = 0
    for i in range(1,13):
        data = scipy.io.loadmat(prefix+fname_prefix+str(i)+fname_postfix)
        data = data['sensor_readings']
        data_list.append(data)
        total_length += len(data)
        label_json[total_length]=i
    label = seg_to_label(label_json)
    return np.vstack(data_list), label