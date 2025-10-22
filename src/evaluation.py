import sys
import os
import json
import pandas as pd

def str_to_list(string):
    if string == '[]':
        return []
    else:
        return list(map(int, string.strip('[]').split(',')))

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluation.py <experiment>")
        sys.exit(1)

    experiment = sys.argv[1]
    valid_experiments = ["univariate", "multivariate"]

    if experiment not in valid_experiments:
        print(f"Error: Invalid experiment '{experiment}'. Valid options are {valid_experiments}.")
        sys.exit(1)

    config_path = os.path.join("config", f"config_{experiment}.json")
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    if experiment == 'multivariate':
        scores_dir = os.path.join("results", experiment, "scores")
        if not os.path.exists(scores_dir):
            os.makedirs(scores_dir)
        for algorithm in config["algorithms"]:
            if algorithm in ["e2usd", "time2state", "ticc", "clasp", "hdp_hsmm", "patss"]:
                results = []
                for dataset in config["dataset_names"]:
                    results_path = os.path.join("results", experiment, "score", algorithm, dataset + ".csv")
                    if not os.path.exists(results_path):
                        print(f"Missing results: Results file '{results_path}' not found for algorithm '{algorithm}' and dataset '{dataset}'.")
                        continue

                    results_df = pd.read_csv(results_path)
                    if not results_df.empty:
                        results.append({
                            "dataset": dataset,
                            "time": results_df['time'].mean() if 'time' in results_df.columns else 'x',
                            "time_std": results_df['time'].std() if 'time' in results_df.columns else 'x',
                            "f1": results_df['f1'].mean() if 'f1' in results_df.columns else 'x',
                            "f1_std": results_df['f1'].std() if 'f1' in results_df.columns else 'x',
                            "covering": results_df['covering'].mean() if 'covering' in results_df.columns else 'x',
                            "covering_std": results_df['covering'].std() if 'covering' in results_df.columns else 'x',
                            "nmi": results_df['nmi'].mean() if 'nmi' in results_df.columns else 'x',
                            "nmi_std": results_df['nmi'].std() if 'nmi' in results_df.columns else 'x',
                            "ari": results_df['ari'].mean() if 'ari' in results_df.columns else 'x',
                            "ari_std": results_df['ari'].std() if 'ari' in results_df.columns else 'x',
                            "wari": results_df['wari'].mean() if 'wari' in results_df.columns else 'x',
                            "wari_std": results_df['wari'].std() if 'wari' in results_df.columns else 'x',
                            "wnmi": results_df['wnmi'].mean() if 'wnmi' in results_df.columns else 'x',
                            "wnmi_std": results_df['wnmi'].std() if 'wnmi' in results_df.columns else 'x',
                            "sms": results_df['sms'].mean() if 'sms' in results_df.columns else 'x',
                            "sms_std": results_df['sms'].std() if 'sms' in results_df.columns else 'x'
                        })

                metrics_df = pd.DataFrame(results)
                metrics_df.to_csv(os.path.join(scores_dir, f'{algorithm}.csv'), index=False)

    if experiment == 'UCRSEG':
        scores_dir = os.path.join("results", "univariate", "scores_std")
        if not os.path.exists(scores_dir):
            os.makedirs(scores_dir)
        for algorithm in ["clasp", "fluss", "pelt", "window", "bocd", "binseg"]:
            results = []
            for dataset in ["UCRSEG"]:
                results_path = os.path.join("results", "univariate", algorithm, dataset + ".csv")
                if not os.path.exists(results_path):
                    print(f"Error: Results file '{results_path}' not found for algorithm '{algorithm}' and dataset '{dataset}'.")
                    continue

                results_df = pd.read_csv(results_path)
                results.append({
                    "dataset": dataset,
                    "time": results_df['time'].mean(),
                    # "ari": results_df['ari'].mean(),
                    # "anmi": results_df['anmi'].mean(),
                    # "nmi": results_df['nmi'].mean(),
                    "f1": results_df['f1'].mean(),
                    "covering": results_df['covering'].mean()
                })

            metrics_df = pd.DataFrame(results)
            metrics_df.to_csv(os.path.join(scores_dir, f'{algorithm}.csv'), index=False)

    print(f"Results saved in {scores_dir} directory")

if __name__ == "__main__":
    main()