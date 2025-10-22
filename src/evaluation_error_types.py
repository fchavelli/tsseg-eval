import os
import sys
import json
import pandas as pd
from collections import Counter

project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.load_dataset import load_data
from utils.load_results import load_clustering

from utils.metrics import state_matching_score

ERROR_TYPES = ['delay', 'transition', 'isolation', 'missing']

def main():
    """
    Main function to evaluate clustering algorithms on multiple datasets using
    State Matching Score (SMS) and its error type counts.

    This function performs the following steps:
    1. Sets the experiment type.
    2. Loads the configuration file for the specified experiment.
    3. Creates the results directory if it does not exist.
    4. Iterates over the algorithms specified in the configuration file.
    5. For each algorithm, iterates over the datasets specified in the configuration file.
    6. Loads the ground truth and clustering results for each dataset.
    7. Evaluates the clustering results using state_matching_score, extracting the score
       and the counts of different error types ('delay', 'transition', 'isolation', 'missing').
    8. Saves the evaluation results (score and error counts) to a CSV file in the results directory.

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

        results_dir = os.path.join("results", experiment, "score_sms")
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
                results_data = {"dataset": ts_name} # Initialize results dict

                if clustering is not None:
                    # Ensure groundtruth and clustering have the same length for comparison
                    min_len = min(len(groundtruth), len(clustering))
                    groundtruth_trimmed = groundtruth[:min_len]
                    clustering_trimmed = clustering[:min_len]

                    # Call state_matching_score requesting errors list
                    # Assuming the function returns (score, errors_list) when return_errors=True
                    try:
                        score, errors_list = state_matching_score(
                            groundtruth_trimmed,
                            clustering_trimmed,
                            return_errors=True
                        )

                        # Initialize dictionaries/counters for aggregation
                        error_counts = Counter()
                        error_size_sums = Counter()
                        # Use Counter for summing penalties; assumes penalties are numeric
                        error_penalty_sums = Counter()

                        # Aggregate counts, sizes, and penalties per error type
                        for error in errors_list:
                            err_type = error['type']
                            if err_type in ERROR_TYPES: # Process only defined error types
                                error_counts[err_type] += 1
                                # Use .get with default 0 in case 'size' or 'penalty' keys are missing
                                error_size_sums[err_type] += error.get('size', 0)
                                error_penalty_sums[err_type] += error.get('penalty', 0.0)

                        # Populate results dictionary
                        results_data['sms_score'] = score
                        for err_type in ERROR_TYPES:
                            # Store count, sum of sizes, and sum of penalties for each error type
                            results_data[f'sms_{err_type}_count'] = error_counts.get(err_type, 0)
                            results_data[f'sms_{err_type}_size_sum'] = error_size_sums.get(err_type, 0)
                            # Ensure penalty sum is stored appropriately (e.g., as float)
                            results_data[f'sms_{err_type}_penalty_sum'] = float(error_penalty_sums.get(err_type, 0.0))

                    except Exception as e:
                        print(f"Error computing SMS for {ts_name}: {e}")
                        # Handle cases where metric computation fails
                        results_data['sms_score'] = 'error'
                        for err_type in ERROR_TYPES:
                            results_data[f'sms_{err_type}_count'] = 'error'
                            results_data[f'sms_{err_type}_size_sum'] = 'error'
                            results_data[f'sms_{err_type}_penalty_sum'] = 'error'

                else:
                    # Handle cases where clustering results couldn't be loaded
                    results_data['sms_score'] = 'x' # Placeholder for missing results
                    for err_type in ERROR_TYPES:
                        results_data[f'sms_{err_type}_count'] = 'x'
                        results_data[f'sms_{err_type}_size_sum'] = 'x'
                        results_data[f'sms_{err_type}_penalty_sum'] = 'x'

                # Append results for this time series to the DataFrame
                results_df = pd.concat([results_df, pd.DataFrame([results_data])], ignore_index=True)

            # Define column order for the CSV file
            error_columns = []
            for err_type in ERROR_TYPES:
                error_columns.extend([
                    f'sms_{err_type}_count',
                    f'sms_{err_type}_size_sum',
                    f'sms_{err_type}_penalty_sum'
                ])
            column_order = ['dataset', 'sms_score'] + error_columns
            results_file = os.path.join(scores_dir, f"{dataset}.csv")
            results_df[column_order].to_csv(results_file, index=False) # Ensure columns are in desired order

            print(f"Results saved in {scores_dir} directory for {dataset}")

if __name__ == "__main__":
    main()