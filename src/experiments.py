import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from algorithms import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_dataset(dataset_path):
    """Return either the dataset path (for folder-based datasets) or a DataFrame."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logging.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    if dataset_path.suffix == ".csv":
        try:
            return pd.read_csv(dataset_path)
        except Exception as e:
            logging.error(f"Failed to read dataset file: {e}")
            sys.exit(1)
    return str(dataset_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run time-series segmentation experiments.")
    parser.add_argument("--t", dest="analysis_type", required=True,
                        choices=["univariate", "multivariate"],
                        help="Analysis type.")
    parser.add_argument("--a", dest="algorithm_name", default=None,
                        help="Restrict to a single algorithm (optional).")
    parser.add_argument("--d", dest="dataset_name", default=None,
                        help="Restrict to a single dataset (optional).")
    parser.add_argument("--no-eval", dest="evaluate", action="store_false",
                        help="Skip metric computation (only record runtime).")
    parser.set_defaults(evaluate=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    analysis_type = args.analysis_type
    algorithm_name = args.algorithm_name
    dataset_name = args.dataset_name

    config_path = Path(f"config/config_{analysis_type}.json")
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    algorithms = config.get("algorithms", [])
    dataset_path = config.get("dataset_path", {})
    dataset_names = config.get("dataset_names", {})

    # The score subdirectory is the one consumed by src/evaluation.py.
    results_base_dir = Path("results") / analysis_type
    results_base_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    logging.info("Starting experiment...")

    data = load_dataset(dataset_path)

    if algorithm_name:
        if algorithm_name not in algorithms:
            logging.error(f"Algorithm '{algorithm_name}' not found in config file.")
            sys.exit(1)
        algorithms = [algorithm_name]

    if dataset_name:
        dataset_names = [dataset_name]

    for algo in algorithms:
        logging.info(f"Algorithm: {algo}")
        for dataset in dataset_names:
            logging.info(f"Dataset: {dataset}")
            results_df = run_experiment(analysis_type, algo, dataset, data,
                                        evaluate=args.evaluate)
            subdir = "score" if args.evaluate else "time"
            results_dir = results_base_dir / subdir / algo
            results_dir.mkdir(parents=True, exist_ok=True)
            results_file = results_dir / f"{dataset}.csv"
            results_df.to_csv(results_file, index=False)
            logging.info(f"Results for {algo} saved to {results_file}")

    elapsed_time = time.time() - start_time
    logging.info(f"Experiment finished in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
