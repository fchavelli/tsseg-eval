"""Aggregate per-time-series metric CSVs produced by ``experiments.py``.

For each algorithm listed in ``config/config_<experiment>.json`` this script
reads ``results/<experiment>/score/<algorithm>/<dataset>.csv`` and writes the
per-metric mean/std summary to ``results/<experiment>/scores/<algorithm>.csv``.
"""

import argparse
import json
import os
import sys

import pandas as pd


METRIC_COLUMNS = ("time", "f1", "covering", "nmi", "ari", "wari", "wnmi", "sms")


def _mean_std(df: pd.DataFrame, col: str):
    if col in df.columns:
        return df[col].mean(), df[col].std()
    return "x", "x"


def aggregate(experiment: str) -> None:
    config_path = os.path.join("config", f"config_{experiment}.json")
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)

    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    scores_dir = os.path.join("results", experiment, "scores")
    os.makedirs(scores_dir, exist_ok=True)

    for algorithm in config["algorithms"]:
        results = []
        for dataset in config["dataset_names"]:
            results_path = os.path.join(
                "results", experiment, "score", algorithm, dataset + ".csv"
            )
            if not os.path.exists(results_path):
                print(
                    f"Missing results: '{results_path}' not found for "
                    f"algorithm '{algorithm}' / dataset '{dataset}'."
                )
                continue

            results_df = pd.read_csv(results_path)
            if results_df.empty:
                continue

            row = {"dataset": dataset}
            for col in METRIC_COLUMNS:
                mean, std = _mean_std(results_df, col)
                row[col] = mean
                row[f"{col}_std"] = std
            results.append(row)

        metrics_df = pd.DataFrame(results)
        metrics_df.to_csv(os.path.join(scores_dir, f"{algorithm}.csv"), index=False)

    print(f"Results saved in {scores_dir} directory")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics.")
    parser.add_argument(
        "experiment",
        choices=["univariate", "multivariate"],
        help="Which experiment to aggregate.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    aggregate(args.experiment)


if __name__ == "__main__":
    main()
