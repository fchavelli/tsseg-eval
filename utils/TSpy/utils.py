# Created by Chengyu on 2021/12/8.
# TSpy util collection.

import numpy as np
import math
import pandas as pd

def remove_constant_col(df):    
    data = df.to_numpy()
    sum = np.sum(data, axis=0)
    data = np.squeeze(data[:, np.argwhere(sum!=0)], 2)
    return pd.DataFrame(data)

def len_of_file(path):
    return len(open(path,'rU').readlines())

def batch_z_normalize(data_tensor):
    result = np.empty(shape=data_tensor.shape)
    num_batch, _, _ = data_tensor.shape
    for i in range(num_batch):
        w = data_tensor[i,:,:]
        _range = np.max(w) - np.min(w)
        w = (w - np.min(w)) / _range
        result[i] = w
    return result

def normalize(X, mode='channel'):
    if mode == 'channel':
        for i in range(X.shape[1]):
            max = np.max(X[:,i])
            min = np.min(X[:,i])
            X[:,i] = (X[:,i]-min)/(max-min)
    elif mode == 'all':
        max = np.max(X)
        min = np.min(X)
        X = (X-min)/(max-min)
        
    return X

def all_normalize(data_tensor):
    mean = np.mean(data_tensor)
    var = np.var(data_tensor)
    i = 0
    for channel in data_tensor[0]:
        data_tensor[0][i] = (channel - mean)/math.sqrt(var)
        i += 1
    return data_tensor

def z_normalize(array):
    _range = np.max(array) - np.min(array)
    return (array - np.min(array)) / _range

# return the index of elems inside the interval (start,end]
def find(array, start, end):
    pos_min = array > start
    pos_max = array <= end
    return np.argwhere(pos_min & pos_max == True)
    
def calculate_density_matrix(feature_list, n=100):
    # convert to np array
    feature_list = np.array(feature_list)
    x = feature_list[:,0]
    y = feature_list[:,1]

    h_start = np.min(y)
    h_end = np.max(y)
    h_step = (h_end-h_start)/n
    w_start = np.min(x)
    w_end = np.max(x)
    w_step = (w_end-w_start)/n

    row_partition = []
    for i in range(n):
        row_partition.append(find(y,h_start+i*h_step,h_start+(i+1)*h_step))
    # print(len(row_partition))
    
    row_partition = list(reversed(row_partition))

    density_matrix = []
    
    for row_idx in row_partition:
        row = x[row_idx]
        row_densities = []
        for i in range(n):
            density = len(find(row,w_start+i*w_step,w_start+(i+1)*w_step))
            row_densities.append(density)
        density_matrix.append(row_densities)
    density_matrix = np.array(density_matrix)
    return density_matrix, w_start, w_end, h_start, h_end,

def calculate_scalar_velocity_list(feature_list, interval=1):
    velocity_list = []
    
    for pre_pos,pos in zip(feature_list[:-interval],feature_list[interval:]):
        # calculate displacement
        velocity_list.append(np.linalg.norm(pos-pre_pos))
    for i in range(interval):
        velocity_list.append(0)
    velocity_list = np.array(velocity_list)

    return velocity_list

def calculate_velocity_list(feature_list, interval=500):
    velocity_list_x = []
    velocity_list_y = []
    
    for pre_pos,pos in zip(feature_list[:-interval],feature_list[interval:]):
        # calculate displacement for x, y, respectively
        displacement_x = (pos[0]-pre_pos[0])/interval
        displacement_y = (pos[1]-pre_pos[1])/interval
        velocity_list_x.append(displacement_x)
        velocity_list_y.append(displacement_y)
    for i in range(interval):
        velocity_list_y.append(0)
        velocity_list_x.append(0)
    velocity_list_x = np.array(velocity_list_x)
    velocity_list_y = np.array(velocity_list_y)

    return velocity_list_x, velocity_list_y