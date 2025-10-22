import os
import sys
import pandas as pd

def escape_latex(text):
    return text.replace('_', '\\_')

def main():
    if len(sys.argv) < 3:
        print("Usage: python export_scores.py <experiment_type> <metric1> <metric2> ...")
        sys.exit(1)

    experiment_type = sys.argv[1]
    metrics = [metric.lower() for metric in sys.argv[2:]]

    results_dir = os.path.join("results", experiment_type, "scores")
    latex_dir = "latex"
    os.makedirs(latex_dir, exist_ok=True)
    metrics_str = "_".join(metrics)
    output_file = os.path.join(latex_dir, f"{experiment_type.lower()}_{metrics_str.lower()}_std.tex")

    # Read all CSV files in the results directory
    data_frames = {}
    for file_name in os.listdir(results_dir):
        if file_name.endswith(".csv"):
            algorithm_name = file_name[:-4].upper()
            file_path = os.path.join(results_dir, file_name)
            df = pd.read_csv(file_path)
            data_frames[algorithm_name] = df

    # Create a list of datasets
    datasets = data_frames[next(iter(data_frames))]['dataset'].tolist()
    datasets = ["ActRecTut", "MoCap", "PAMAP2", "UCRSEG", "USC-HAD", "Suturing", "Knot_Tying", "Needle_Passing"]
    
    # Create the LaTeX table
    with open(output_file, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{l" + "c" * len(datasets) * len(metrics) + "}\n")
        f.write("\\hline\n")
        f.write("Algorithm")

        for dataset in datasets:
            f.write(f" & \\multicolumn{{{len(metrics)}}}{{c}}{{{escape_latex(dataset)}}}")
        f.write(" \\\\\n")

        f.write(" ")
        for _ in datasets:
            for metric in metrics:
                f.write(f" & {metric}")
        f.write(" \\\\\n")
        f.write("\\hline\n")

        for algorithm, df in data_frames.items():
            f.write(escape_latex(algorithm))
            for dataset in datasets:
                for metric in metrics:
                    if 'dataset' in df.columns:
                        value = df.loc[df['dataset'] == dataset, metric].values
                        value = value[0] if len(value) > 0 else "x"
                    else:
                        value = "x"
                    if len(str(value)) > 5:
                        value = f"{float(value):.2f}"
                    f.write(f" & {value}")
            f.write(" \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\caption{Results for " + experiment_type + "}\n")
        f.write("\\end{table}\n")

    print(f"Results exported to {output_file}")

if __name__ == "__main__":
    main()
