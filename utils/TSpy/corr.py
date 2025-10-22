'''
Created by Chengyu on 2022/2/7.
'''

from sklearn import metrics
import numpy as np
import scipy.cluster.hierarchy as sch
import pandas as pd

def add_lag(X, Y, lag):
    if lag>0:
        seq1 = X[lag:]
        seq2 = Y[:-lag]
    elif lag<0:
        seq1 = X[:lag]
        seq2 = Y[-lag:]
    else:
        seq1 = X
        seq2 = Y
    return seq1, seq2

def decompose_state_seq(X):
    state_set = set(X)
    # return state_set
    # print(state_set)
    single_state_seq_list = []
    for state in list(state_set):
        single_state_seq = np.zeros(X.shape, dtype=int)
        single_state_seq[np.argwhere(X==state)]=1
        single_state_seq_list.append(single_state_seq)
    return np.array(single_state_seq_list)

def score(X,Y):
    # length = len(X)
    # p_x = np.count_nonzero(X)/length
    # p_xy = np.sum((X+Y)==2)/length
    # new = Y[np.argwhere(X==1)]
    # p_y_given_x = np.count_nonzero(new)/len(new)
    # # scores = p_xy*np.log(p_y_given_x/p_x)
    # scores = p_xy*p_y_given_x/p_x
    # # print(p_x, p_xy, p_y_given_x, scores)
    # return scores
    len_x_or_y = np.count_nonzero(X+Y)
    len_x_and_y = np.sum((X+Y)==2)  
    return len_x_and_y/len_x_or_y

def lagged_partial_state_corr(X, Y, atom_step=0.001, max_ratio=0.05):
    length = len(X)
    k = int(max_ratio/atom_step)
    listX = decompose_state_seq(X)
    listY = decompose_state_seq(Y)
    score_matrix = np.zeros((len(listX),len(listY)))
    lag_matrix = np.zeros((len(listX),len(listY)))
    for i in range(len(listX)):
        for j in range(len(listY)):
            sssX = listX[i]
            sssY = listY[j]
            max_score = -1
            lag = 0
            for l in range(-k,k+1):
                lag_len = int(l*atom_step*length)
                sX, sY = add_lag(sssX, sssY, lag_len)
                Jscore = score(sX, sY)
                if Jscore>=max_score:
                    max_score=Jscore
                    lag=lag_len
            score_matrix[i,j] = max_score
            lag_matrix[i,j] = lag
    return score_matrix, lag_matrix

def partial_state_corr(X,Y):
    listX = decompose_state_seq(X)
    listY = decompose_state_seq(Y)
    score_matrix = np.zeros((len(listX),len(listY)))
    for i in range(len(listX)):
        for j in range(len(listY)):
            sssX = listX[i]
            sssY = listY[j]
            Jscore = score(sssX, sssY)
            score_matrix[i,j] = Jscore
    # print(score_matrix)
    return score_matrix

def find_unique_best_match(X, Y, score_matrix):
    # print(score_matrix)
    matched_pair = []
    height, width = score_matrix.shape
    for i in range(min(height, width)):
        row, col = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        matched_pair.append((row, col))
        # print(score_matrix, row, col)
        score_matrix[row,:] = 0
        score_matrix[:,col] = 0
        if np.sum(score_matrix)==0:
            break
    # print(set(X), set(Y))
    # print(compact(X), compact(Y))
    # print(matched_pair)
    new_X = X.copy()
    new_Y = Y.copy()+10
    # color = 0
    for i,j in matched_pair:
        # new_X[np.argwhere(X==i)]=color
        new_Y[np.argwhere(Y==j)]=i
        # color+=1
    new_Y[new_Y>=10]=-1
    # print(set(new_X), set(new_Y))
    # print(compact(new_X), compact(new_Y))
    # print('========================')
    return X, new_Y

# Find best match for all states.
def find_best_match(X, Y, score_matrix):
    print(score_matrix)
    height, width = score_matrix.shape
    new_Y = np.zeros(Y.shape)
    for i in range(height):
        idx = np.argmax(score_matrix[i,:])
        adjust_idx = np.argwhere(Y==idx)
        new_Y[adjust_idx] = i
    return X, new_Y

def lagged_NMI(seq1, seq2, ratio, atom_step=0.001):
    length = len(seq1)
    k = int(ratio/atom_step)
    max_score = -1
    lag = 0
    for i in range(-k, k+1):
        lag_len = int(i*atom_step*length)
        if lag_len>0:
            NMI_score = metrics.normalized_mutual_info_score(seq1[lag_len:],seq2[:-lag_len])
        elif lag_len<0:
            NMI_score = metrics.normalized_mutual_info_score(seq1[:lag_len],seq2[-lag_len:])
        else:
            NMI_score = metrics.normalized_mutual_info_score(seq1,seq2)
        # print(i, lag_len, NMI_score)
        if NMI_score >= max_score:
            max_score = NMI_score
            lag = lag_len
    return max_score, lag

def lagged_state_correlation(seq_list, ratio=0.05):
    '''
    Lagged state correlation.
    @Params:
        seq_list: list of state sequences.
        ratio: maximum lag ratio.
    @return:
        correlation_matrix: state correlation matrix.
        lag_matrix: lag matrix.
    '''
    num_instance = len(seq_list)
    correlation_matrix = np.ones((num_instance,num_instance))
    lag_matrix = np.ones((num_instance, num_instance))
    for i in range(num_instance):
        for j in range(num_instance):
            if i < j:
                NMI_score, lag = lagged_NMI(seq_list[i],seq_list[j], ratio)
                correlation_matrix[i,j] = NMI_score
                correlation_matrix[j,i] = NMI_score
                lag_matrix[i,j] = lag
                lag_matrix[j,i] = -lag
            else:
                continue
    return correlation_matrix, lag_matrix

def state_correlation(seq_list):
    length = len(seq_list)
    correlation_matrix = np.ones((length,length))
    for i in range(length):
        for j in range(length):
            if i < j:
                NMI_score = metrics.normalized_mutual_info_score(seq_list[i],seq_list[j])
                correlation_matrix[i,j] = NMI_score
                correlation_matrix[j,i] = NMI_score
            else:
                continue
    return correlation_matrix

def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx], idx_to_cluster_array, idx