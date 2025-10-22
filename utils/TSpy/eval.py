# Created by Chengyu on 2022/1/26.

from sklearn import metrics
import numpy as np
from TSpy.label import reorder_label

def __adjusted_macro_score(groundtruth, prediction, score_type):
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
        return

    length = len(groundtruth)

    # convert to numpy array.
    groundtruth = np.array(groundtruth, dtype=int)
    prediction = np.array(prediction, dtype=int)

    groundtruth_set = set(groundtruth)
    prediction_set = set(prediction)
    used_label_set = set()
    n = len(groundtruth_set)

    total = 0
    for i in groundtruth_set:
        used_j = 0
        max_score = 0
        for j in prediction_set:
            if j in used_label_set:
                continue

            # convert to binary array of positive label.
            true = np.zeros(length, dtype=int)
            pred = np.zeros(length, dtype=int)
            true[np.argwhere(groundtruth == i)]=1
            pred[np.argwhere(prediction == j)]=1

            if score_type == 'f1':
                score = metrics.f1_score(true, pred, average='binary', pos_label=1, zero_division=0)
            elif score_type == 'precision':
                score = metrics.precision_score(true, pred, average='binary', pos_label=1, zero_division=0)
            elif score_type == 'recall':
                score = metrics.recall_score(true, pred, average='binary', pos_label=1, zero_division=0)
            else:
                print('Error: Score type does not exists.')

            if score > max_score:
                max_score = score
                used_j = j
                
        used_label_set.add(used_j)
        total += max_score
    return total/n

def adjusted_macro_recall(groundtruth, prediction):
    return __adjusted_macro_score(groundtruth, prediction, score_type='recall')

def adjusted_macro_precision(groundtruth, prediction):
    return __adjusted_macro_score(groundtruth, prediction, score_type='precision')

def adjusted_macro_f1score(groundtruth, prediction):
    return __adjusted_macro_score(groundtruth, prediction, score_type='f1')

def macro_recall(groundtruth, prediction, if_reorder_label=False):
    '''
    calculate macro precision
    '''
    groundtruth = np.array(groundtruth, dtype=int)
    prediction = np.array(prediction, dtype=int)
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
    else:
        if if_reorder_label:
            groundtruth = reorder_label(groundtruth)
            prediction = reorder_label(prediction)
        return metrics.recall_score(groundtruth, prediction, average='macro', zero_division=0)

def macro_precision(groundtruth, prediction, if_reorder_label=False):
    '''
    calculate macro precision
    '''
    groundtruth = np.array(groundtruth, dtype=int)
    prediction = np.array(prediction, dtype=int)
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
    else:
        if if_reorder_label:
            groundtruth = reorder_label(groundtruth)
            prediction = reorder_label(prediction)
        return metrics.precision_score(groundtruth, prediction, average='macro', zero_division=0)

def macro_f1score(groundtruth, prediction, if_reorder_label=False):
    '''
    calculate macro f1 score
    '''
    groundtruth = np.array(groundtruth, dtype=int)
    prediction = np.array(prediction, dtype=int)
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
    else:
        if if_reorder_label:
            groundtruth = reorder_label(groundtruth)
            prediction = reorder_label(prediction)
        return metrics.f1_score(groundtruth, prediction, average='macro', zero_division=0)

def find_cut_points_from_label(label):
    pre = 0
    current = 0
    length = len(list(label))
    result = []
    while current < length:
        if label[current] == label[pre]:
            current += 1
            continue
        else:
            result.append(current)
            pre = current
    return result

def evaluate_cut_point(groundtruth, prediction, d):
    list_true_pos_cut = find_cut_points_from_label(groundtruth)
    list_pred_pos_cut = find_cut_points_from_label(prediction)
    # print(list_true_pos_cut, list_pred_pos_cut)
    tp = 0
    fn = 0
    for pos_true in list_true_pos_cut:
        flag = False
        list_elem_to_be_removed = []
        for pos_pred in list_pred_pos_cut:
            if pos_pred >= pos_true-d and pos_pred <= pos_true+d-1:
                # print('current elem %d, internal[%d,%d],pop%d'%(pos_pred, pos_true-d, pos_true+d-1, pos_pred))
                # list_pred_pos_cut.remove(pos_pred)
                list_elem_to_be_removed.append(pos_pred)
                flag = True
        if not flag:
            fn += 1
        else:
            tp += 1
        # remove
        for e in list_elem_to_be_removed:
            list_pred_pos_cut.remove(e)
        # print(list_pred_pos_cut)

    fp = len(list_pred_pos_cut)
    
    if tp+fp==0:
        precision = 0
    else:
        precision = tp/(tp+fp)
    
    if tp+fn==0:
        recall = 0
    else:
        recall = tp/(tp+fn)
    
    if precision+recall==0:
        fscore = 0
    else:
        fscore = 2*precision*recall/(precision+recall)
    return fscore, precision, recall

# groundtruth = [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,2,2,5,5,5,5,5]
# prediction =  [1,1,1,1,0,1,0,0,0,1,1,1,1,1,2,2,2,2,2,2,5,5,5,3,3]
# print(evaluate_cut_point(groundtruth, prediction, 2))

def ARI(prediction, groundtruth):
    return metrics.adjusted_rand_score(groundtruth, prediction)

def ANMI(prediction, groundtruth):
    return metrics.adjusted_mutual_info_score(groundtruth, prediction)

def NMI(groundtruth, prediction):
    return metrics.normalized_mutual_info_score(groundtruth, prediction)

def evaluate_clustering(groundtruth, prediction):
    ari = ARI(groundtruth, prediction)
    anmi = ANMI(groundtruth, prediction)
    nmi = NMI(groundtruth, prediction)
    return ari, anmi, nmi

def evaluation(groundtruth, prediction):
    ari = ARI(groundtruth, prediction)
    ami = ANMI(groundtruth, prediction)
    f1 = adjusted_macro_f1score(groundtruth, prediction)
    precision = adjusted_macro_precision(groundtruth, prediction)
    recall = adjusted_macro_recall(groundtruth, prediction)
    return f1, precision, recall, ari, ami

def adjusted_macro_F_measure(groundtruth, prediction):
    f1 = adjusted_macro_f1score(groundtruth, prediction)
    precision = adjusted_macro_precision(groundtruth, prediction)
    recall = adjusted_macro_recall(groundtruth, prediction)
    return f1, precision, recall
