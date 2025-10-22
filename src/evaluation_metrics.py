import os
import sys
import json
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.load_dataset import load_data
from utils.load_results import load_clustering

from utils.metrics import *
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


np.random.seed(42)

n_experiments = 100

def random_weights():
    """
    Generate random weights for the state matching score.

    Returns:
        dict: A dictionary containing random weights for 'delay', 'transition', 'isolation', and 'missing'.
    """
    return {
        'delay': np.random.uniform(0, 1),
        'transition': np.random.uniform(0, 1),
        'isolation': np.random.uniform(0, 1),
        'missing': np.random.uniform(0, 1)
    }

metrics = {
    **{f"sms_{i}": lambda groundtruth, clustering, weights=random_weights(): state_matching_score(groundtruth, clustering, weights) for i in range(n_experiments)},
}

def main():
    """
    Main function to evaluate clustering algorithms on multiple datasets.

    This function performs the following steps:
    1. Sets the experiment type.
    2. Loads the configuration file for the specified experiment.
    3. Creates the results directory if it does not exist.
    4. Iterates over the algorithms specified in the configuration file.
    5. For each algorithm, iterates over the datasets specified in the configuration file.
    6. Loads the ground truth and clustering results for each dataset.
    7. Evaluates the clustering results using predefined metrics.
    8. Saves the evaluation results to a CSV file in the results directory.

    Raises:
        SystemExit: If the configuration file is not found.

    Prints:
        Error messages if the configuration file is not found.
        Progress messages indicating the current algorithm and dataset being evaluated.
        A message indicating where the results are saved.
    """

    experiment = "multivariate"

    config_path = os.path.join("config", f"config_{experiment}.json")
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

        results_dir = os.path.join("results", experiment, "score_sms_weights")
        os.makedirs(results_dir, exist_ok=True)

        datasets = config["dataset_names"]
        algorithms = config["algorithms"]

    # datasets = ['USC-HAD']
    # algorithms = ['hdp_hsmm']

    for algorithm in algorithms:

        print(f"Evaluating {algorithm} algorithm")
        scores_dir = os.path.join(results_dir, algorithm)
        os.makedirs(scores_dir, exist_ok=True)

        for dataset in datasets:

            print(f"Evaluating {dataset} dataset")
            results_df = pd.DataFrame()

            for _, groundtruth, params in load_data(dataset):

                ts_name = f"{dataset}_{'_'.join(map(str, params))}"
                clustering = load_clustering(experiment, algorithm, dataset, params, verbose=False)
                results = []

                if clustering is not None:
                    
                    for metric_name in metrics.keys():
                        print('Computing metric', metric_name)
                        results.append(metrics[metric_name](groundtruth[:len(clustering)], clustering))
                else:
                    score = "x"

                results_df = pd.concat([results_df, pd.DataFrame([{"dataset": ts_name, **{metric_name: result for metric_name, result in zip(metrics.keys(), results)}
                }])], ignore_index=True)

            results_file = os.path.join(scores_dir, f"{dataset}.csv")
            results_df.to_csv(results_file, index=False)

            print(f"Results saved in {scores_dir} directory")

if __name__ == "__main__":
    main()