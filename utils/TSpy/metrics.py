import sklearn.metrics as metrics
import numpy as np

def calculate_NMI_matrix(seq_list):
    width = len(seq_list)
    result = np.zeros(shape=(width,width))
    for i in range(width):
        for j in range(width):
            nmi = metrics.normalized_mutual_info_score(seq_list[i], seq_list[j])
            result[i,j] = nmi
    return result