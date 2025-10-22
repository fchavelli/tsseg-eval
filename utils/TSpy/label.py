import numpy as np

def compact(series):
    '''
    series = [1, 1, 2, 2, 2, 3, 3, 1]
    result = compact(series)
    print(result)  # Output: [1, 2, 3, 1]
    '''
    compacted = []
    pre = series[0]
    compacted.append(pre)
    for e in series[1:]:
        if e != pre:
            pre = e
            compacted.append(e)
    return compacted

def remove_duplication(series):
    '''
    series = [1, 2, 2, 3, 1, 4]
    result = remove_duplication(series)
    print(result)  # Output: [1, 2, 3, 4]
    '''
    result = []
    for e in series:
        if e not in result:
            result.append(e)
    return result
    
def seg_to_label(label):
    '''
    label = {0: 3, 3: 5, 5: 2}  # Keys : positions, Values : labels
    result = seg_to_label(label)
    print(result)  # Output: [3, 3, 3, 5, 5, 2, 2]
    '''
    pre = 0
    seg = []
    for l in label:
        seg.append(np.ones(l-pre,dtype=int)*label[l])
        pre = l
    result = np.concatenate(seg)
    return result

def reorder_label(label):
    '''
    label = [3, 3, 5, 5, 2, 2]
    result = reorder_label(label)
    print(result)  # Output: [0, 0, 1, 1, 2, 2]
    '''
    # Start from 0.
    label = np.array(label)
    ordered_label_set = remove_duplication(compact(label))
    idx_list = [np.argwhere(label==e) for e in ordered_label_set]
    for i, idx in enumerate(idx_list):
        label[idx] = i
    return label

def adjust_label(label):
    '''
   label = [3, 3, 5, 5, 2, 2]
    result = adjust_label(label)
    print(result)  # Output: [3, 3, 5, 5, 2, 2]
    '''
    label = np.array(label)
    compacted_label = compact(label)
    ordered_label_set = remove_duplication(compacted_label)
    label_set = set(label)
    idx_list = [np.argwhere(label==e) for e in ordered_label_set]
    for idx, elem in zip(idx_list,label_set):
        label[idx] = elem
    return label

def bucket_vote(bucket):
    '''
    bucket = [1, 2, 2, 3, 3, 3]
    result = bucket_vote(bucket)
    print(result)  # Output: 3
    '''
    vote_vector = np.zeros(len(set(bucket)), dtype=int)
    
    # create symbol table
    symbol_table = {}
    symbol_list = []
    for i, s in enumerate(set(bucket)):
        symbol_table[s] = i
        symbol_list.append(s)

    # do vote
    for e in bucket:
        vote_vector[symbol_table[e]] += 1

    symbol_idx = np.argmax(vote_vector)
    return symbol_list[symbol_idx]

def smooth(X, bucket_size):
    '''
    X = [1, 1, 2, 2, 3, 3, 3, 1]
    result = smooth(X, bucket_size=3)
    print(result)  # Output: [1, 1, 1, 3, 3, 3, 3, 3]
    '''
    for i in range(0,len(X), bucket_size):
        s = bucket_vote(X[i:i+bucket_size])
        true_size = len(X[i:i+bucket_size])
        X[i:i+bucket_size] = s*np.ones(true_size,dtype=int)
    return X

def dilate_label(label, f, max_len):
    '''
    label = [1, 2, 3]
    result = dilate_label(label, f=3, max_len=8)
    print(result)  # Output: [1, 1, 1, 2, 2, 2, 3, 3]
    '''
    slice_list = []
    for e in label:
        slice_list.append(e*np.ones(f, dtype=int))
    return np.concatenate(slice_list)[:max_len]

def str_list_to_label(label):
    '''
    label = ["A", "B", "A", "C"]
    result = str_list_to_label(label)
    print(result)  # Output: [0, 1, 0, 2]
    '''
    label_set = remove_duplication(label)
    label = np.array(label)
    new_label = np.array(np.ones(len(label)))
    for i, l in enumerate(label_set):
        idx = np.argwhere(label==l)
        new_label[idx] = i
    return new_label.astype(int)