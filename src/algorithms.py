import os
import sys
import time
import json
import logging
import warnings

import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is in sys.path for absolute imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

os.environ['NUMPY_EXPERIMENTAL_DTYPE_API'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore", message="using slow sample_crp_tablecounts")
np.seterr(all='warn')

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from utils.TSpy.utils import *
from utils.TSpy.label import *

from utils.metrics import *
from utils import load_dataset
from claspy.segmentation import BinaryClaSPSegmentation
from baselines.clasp.test import run_clasp_multivariate
from baselines.ticc.TICC_solver import TICC
from baselines.hdp_hsmm.hdp_hsmm import HDP_HSMM
from baselines.patss.algorithms import PaTSS_perso
from baselines.time2state.src.time2state import Time2State
from baselines.time2state.src.adapers import *
from baselines.time2state.src.clustering import *
from baselines.time2state.src.default_params import *

from baselines.E2USD.e2usd import E2USD
from baselines.E2USD.adapers import *
from baselines.E2USD.utils import *
from baselines.E2USD.clustering import *
from baselines.E2USD.params import *
from baselines.E2USD.networks import *

def save_prediction(analysis_type, algorithm_name, dataset_name, ts_info, prediction):
    ts_info = '_' + '_'.join(map(str, ts_info)) + '.npy'
    results_dir = Path(f'results/{analysis_type}/clustering/{algorithm_name}')
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / (dataset_name + ts_info)
    np.save(results_file, np.array(prediction, dtype=int))

def save_results(analysis_type, results_df, dataset_name, ts_info, evaluate, groundtruth, prediction, elapsed_time):
    ts =  dataset_name + '_' + '_'.join(map(str, ts_info)) + '.npy'
    if evaluate:
        if analysis_type == 'univariate':
            f1, cov = run_evaluation(analysis_type, groundtruth, prediction)
            results_df = pd.concat([results_df, pd.DataFrame([{'dataset': ts, 'time': elapsed_time, 'f1': f1, 'covering': cov}])], ignore_index=True)
        else:
            f1, cov, nmi, ari, wari, sms = run_evaluation(analysis_type, groundtruth, prediction)
            results_df = pd.concat([results_df, pd.DataFrame([{'dataset': ts, 'time': elapsed_time, 'f1': f1, 'covering': cov, 'nmi': nmi, 'ari': ari, 'wari': wari, 'sms': sms}])], ignore_index=True)
    else:
        results_df = pd.concat([results_df, pd.DataFrame([{'dataset': ts, 'time': elapsed_time}])], ignore_index=True)
    return results_df


def run_experiment(analysis_type, algorithm_name, dataset_name, data, evaluate=False):
    
    results_df = pd.DataFrame()

    if dataset_name == 'tssb':

        results_df = pd.DataFrame(columns=['dataset', 'change_points', 'time'])
        for index, row in data.iterrows():
            # if index >= 1:
            #     break
            dataset, window_size, true_cps, data = row['dataset'], row['window_size'], row['change_points'], row['time_series']
            # window_size = np.int64(window_size)
            # change_points = np.asarray(change_points, dtype=np.int64)
            data = np.fromstring(data.strip('[]'), sep=',', dtype=np.float64)
            logging.info(f"Running algorithm: {algorithm_name} on TS: {dataset}")

            change_points, elapsed_time = run_algorithm(analysis_type, algorithm_name, data)

            results_df = pd.concat([results_df, pd.DataFrame([{'dataset': dataset, 'change_points': change_points, 'time': elapsed_time}])], ignore_index=True)
    
    else:

        config_path = Path(f"config/{algorithm_name}.json")
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        globals().update(config[dataset_name])

        for data, groundtruth, infos in load_dataset.load_data(dataset_name):

            if algorithm_name == 'ticc':
                if dataset_name in ['ActRecTut', 'PAMAP2', 'USC-HAD', 'Suturing', 'Knot_Tying', 'Needle_Passing']:
                    groundtruth = groundtruth[:-2]

                elif dataset_name == 'MoCap':
                    groundtruth = groundtruth[:-4]

            if dataset_name == 'UCRSEG':
                if algorithm_name == 'ticc':
                    groundtruth = groundtruth[win_size:]
                else:
                    groundtruth = groundtruth[:-1]
            
            n_states = len(set(groundtruth)) if groundtruth is not None else None

            prediction, elapsed_time = run_algorithm(analysis_type, algorithm_name, data, n_states=n_states)
            save_prediction(analysis_type, algorithm_name, dataset_name, infos, prediction)
            results_df = save_results(analysis_type, results_df, dataset_name, infos, evaluate, groundtruth, prediction, elapsed_time)

    return results_df

def run_algorithm(analysis_type, algorithm_name, data, n_states=None):

    if algorithm_name == 'clasp':
        if analysis_type == 'univariate':
            prediction, elapsed_time = run_clasp(data)
        else:
            prediction, elapsed_time = run_clasp_multi(data, window_size=win_size, num_cps=num_cps, n_states=n_states, offset=offset)
    elif algorithm_name == 'patss':
        _, prediction, elapsed_time = run_patss(data)
    elif algorithm_name == 'fluss':
        prediction, elapsed_time = run_fluss(data)
    elif algorithm_name == 'pelt':
        prediction, elapsed_time = run_pelt(data)
    elif algorithm_name == 'binseg':
        prediction, elapsed_time = run_binseg(data)
    elif algorithm_name == 'window':
        prediction, elapsed_time = run_window(data)
    elif algorithm_name == 'hidalgo':
        prediction, elapsed_time = run_hidalgo(data)
    elif algorithm_name == 'bocd':
        prediction, elapsed_time = run_bocd(data)
    elif algorithm_name == 'autoplait':
        prediction, elapsed_time = run_autoplait(data)
    elif algorithm_name == 'ticc':
        prediction, elapsed_time = run_ticc(data, window_size=win_size, number_of_clusters=n_states, lambda_parameter=lambda_parameter, beta=beta, threshold=threshold)
    elif algorithm_name == 'hvgh':
        prediction, elapsed_time = run_hvgh(data, window_size=100)
    elif algorithm_name == 'hdp_hsmm':
        prediction, elapsed_time = run_hdp_hsmm(data, alpha, beta, n_iter)
    elif algorithm_name == 'time2state':
        prediction, elapsed_time = run_time2state(data, in_channels, out_channels, win_size, step, M, N, nb_steps)
    elif algorithm_name == 'e2usd':
        prediction, elapsed_time = run_e2usd(data, in_channels, out_channels, win_size, step)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    return prediction, elapsed_time

def run_evaluation(analysis_type, groundtruth, prediction):

    if analysis_type == 'univariate':
        f1 = f_score(groundtruth, prediction)
        cov = covering(groundtruth, prediction)
        return f1, cov
    else:
        groundtruth = groundtruth[:len(prediction)]
        f1 = f_score(groundtruth, prediction)
        cov = covering(groundtruth, prediction)
        nmi = normalized_mutual_info_score(groundtruth, prediction)
        ari = adjusted_rand_score(groundtruth, prediction)
        wari = weighted_adjusted_rand_score(groundtruth, prediction)
        sms = state_matching_score(groundtruth, prediction)
        return f1, cov, nmi, ari, wari, sms


def run_patss(time_series):
    start_time = time.time()
    multivariate_time_series = PaTSS_perso.transform_to_dfs(time_series)
    length_time_series = multivariate_time_series[0].shape[0]
    segmentation, individual_embeddings, all_patterns_combined = PaTSS_perso.run_patss('baselines/patss/temp/', multivariate_time_series, length_time_series)
    single_array, change_points = PaTSS_perso.probas_to_segments_and_change_points(segmentation)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return change_points, single_array, elapsed_time

def run_clasp(time_series):
    start_time = time.time()
    clasp = BinaryClaSPSegmentation()
    clasp.fit_predict(time_series)
    change_points = clasp.change_points.tolist()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return change_points, elapsed_time

def run_clasp_multi(time_series, window_size, num_cps, n_states, offset):
    start_time = time.time()
    # n_states only used for clustering
    prediction = run_clasp_multivariate(time_series, window_size, num_cps, n_states, offset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return prediction, elapsed_time

def run_hdp_hsmm(time_series, alpha, beta, n_iter):
    start_time = time.time()
    time_series = normalize(time_series)
    prediction = HDP_HSMM(alpha, beta, n_iter).fit(time_series)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return prediction, elapsed_time

def run_ticc(time_series, window_size, number_of_clusters, lambda_parameter, beta, threshold):
    start_time = time.time()
    ticc = TICC(window_size=window_size, number_of_clusters=number_of_clusters, lambda_parameter=lambda_parameter, beta=beta, maxIters=10, threshold=threshold,
            write_out_file=False, prefix_string="output_folder/", num_proc=10)
    prediction, _ = ticc.fit_transform(time_series)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return prediction, elapsed_time

def run_time2state(time_series, in_channels, out_channels, window_size, step, M, N, nb_steps):
    start_time = time.time()
    params_LSE['in_channels'] = in_channels
    params_LSE['compared_length'] = window_size
    params_LSE['out_channels'] = out_channels
    params_LSE['win_size'] = window_size
    params_LSE['M'] = M
    params_LSE['N'] = N
    params_LSE['nb_steps'] = nb_steps
    t2s = Time2State(window_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(time_series, window_size, step)
    prediction = t2s.state_seq
    end_time = time.time()
    elapsed_time = end_time - start_time
    return prediction, elapsed_time

def run_time2state_fit(time_series, in_channels, out_channels, window_size, step, M, N, nb_steps):
    start_time = time.time()
    params_LSE['in_channels'] = in_channels
    params_LSE['compared_length'] = window_size
    params_LSE['out_channels'] = out_channels
    params_LSE['win_size'] = window_size
    params_LSE['M'] = M
    params_LSE['N'] = N
    params_LSE['nb_steps'] = nb_steps
    t2s = Time2State(window_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(time_series, window_size, step)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return t2s, elapsed_time

def run_time2state_predict(t2s, time_series, window_size, step):
    start_time = time.time()
    t2s.predict(time_series, window_size, step)
    prediction = t2s.state_seq
    end_time = time.time()
    elapsed_time = end_time - start_time
    return prediction, elapsed_time

def run_e2usd(time_series, in_channels, out_channels, window_size, step):
    start_time = time.time()
    params['in_channels'] = in_channels
    params['compared_length'] = window_size
    params['out_channels'] = out_channels
    e2usd = E2USD(window_size, step, E2USD_Adaper(params), DPGMM(None)).fit(time_series, window_size, step)
    prediction = e2usd.state_seq
    end_time = time.time()
    elapsed_time = end_time - start_time
    return prediction, elapsed_time

def run_e2usd_fit(time_series, in_channels, out_channels, window_size, step):
    start_time = time.time()
    params['in_channels'] = in_channels
    params['compared_length'] = window_size
    params['out_channels'] = out_channels
    e2usd = E2USD(window_size, step, E2USD_Adaper(params), DPGMM(None)).fit(time_series, window_size, step)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return e2usd, elapsed_time

def run_e2usd_predict(e2usd, time_series, window_size, step):
    start_time = time.time()
    e2usd.predict(time_series, window_size, step)
    prediction = e2usd.state_seq
    end_time = time.time()
    elapsed_time = end_time - start_time
    return prediction, elapsed_time