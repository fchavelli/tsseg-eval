import sys
import json
import time
import logging
import pandas as pd
from pathlib import Path

from algorithms import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(dataset_path):
    # Placeholder function to prepare the dataset
    #logging.info(f"Preparing dataset: {dataset_path}")
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logging.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    
    data = None
    if dataset_path.suffix == ".csv":
        try:
            data = pd.read_csv(dataset_path)
        except Exception as e:
            logging.error(f"Failed to read dataset file: {e}")
            sys.exit(1)
    else:
        data = str(dataset_path)

    return data

def main():
    if len(sys.argv) not in [3, 5, 7]:
        logging.error(
            "Usage: python experiments.py --t <analysis_type> [--a <algorithm_name>] [--d <dataset_name>]\n"
            "Flags:\n"
            "  --t   Specify the analysis type (univariate or multivariate) [mandatory]\n"
            "  --a   Specify the algorithm name [optional]\n"
            "  --d   Specify the dataset name [optional]"
        )
        sys.exit(1)

    # if sys.argv[1] != "--t" or sys.argv[2] not in ["univariate", "multivariate"]:
    #     logging.error("Invalid analysis type. Must be 'univariate' or 'multivariate'.")
    #     sys.exit(1)

    analysis_type = sys.argv[2]

    algorithm_name = None
    dataset_name = None

    if len(sys.argv) >= 5:
        if sys.argv[3] == "--a":
            algorithm_name = sys.argv[4]
        elif sys.argv[3] == "--d":
            dataset_name = sys.argv[4]
        else:
            logging.error("Invalid flag. Must be '--a' or '--d'.")
            sys.exit(1)

    if len(sys.argv) == 7:
        if sys.argv[5] == "--a":
            algorithm_name = sys.argv[6]
        elif sys.argv[5] == "--d":
            dataset_name = sys.argv[6]
        else:
            logging.error("Invalid flag. Must be '--a' or '--d'.")
            sys.exit(1)

    config_path = Path(f"config/config_{analysis_type}.json")
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    algorithms = config.get("algorithms", [])
    dataset_path = config.get("dataset_path", {})
    dataset_names = config.get("dataset_names", {})

    # Create results directory if it doesn't exist
    results_base_dir = Path("results")
    results_base_dir.mkdir(exist_ok=True)

    # Create analysis type directory if it doesn't exist
    analysis_type_dir = results_base_dir / analysis_type
    analysis_type_dir.mkdir(exist_ok=True)
    start_time = time.time()
    
    logging.info("Starting experiment...")

    data = load_dataset(dataset_path)

    if algorithm_name:
        if algorithm_name not in algorithms:
            logging.error(f"Algorithm '{algorithm_name}' not found in config file.")
            sys.exit(1)
        algorithms = [algorithm_name]

    for algo in algorithms:
        logging.info(f"Algorithm: {algo}")
        if dataset_name:
            dataset_names = [dataset_name]
        for dataset in dataset_names:
            logging.info(f"Dataset: {dataset}")
            results_df = run_experiment(analysis_type, algo, dataset, data, evaluate=False)
            results_dir = Path(f"results/{analysis_type}/{algo}/time")
            results_dir.mkdir(parents=True, exist_ok=True)
            results_file = results_dir / (dataset + ".csv")
            results_df.to_csv(results_file, index=False)
            logging.info(f"Results for {algo} saved to {results_file}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Experiment finished in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()