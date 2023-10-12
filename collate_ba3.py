import sys 
import os

import argparse

import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ttest_ind
import numpy as np
from statsmodels.tsa.stattools import acf


def main():
    args = parse_args()

    # Check if directory exists
    if not os.path.exists(args.outdir):
        print(f"Error: Output directory {args.outdir} does not exist!")
        return

    all_logL_data = {}  # Using dictionary {run_name: data}

    pdf_path = f"{args.out}_mcmc.pdf"
    print("Evaluating trace files for each run...")
    with PdfPages(pdf_path) as pdf:
        for filename in os.listdir(args.outdir):
            filepath = os.path.join(args.outdir, filename)
            if "trace.txt" in filename:
                run_name, table_df = parse_trace_file(filepath)

                if run_name and not table_df.empty:
                    # mean test for convergence from post burn-in samples 
                    burn_idx = int(args.burn * len(table_df))
                    post_burn_table_df = table_df[burn_idx:]
                    mean_test(post_burn_table_df['LogProb'], args.x, args.alpha)

                    fig = trace_and_autocorr_plot(run_name, table_df['LogProb'], burn_idx)
                    pdf.savefig(fig)
                    plt.close(fig)

                    all_logL_data[run_name] = post_burn_table_df['LogProb']

        # After all runs are processed, plot the grid of histograms
        run_names = list(all_logL_data.keys())
        num_runs = len(run_names)
        cols = max(int(np.sqrt(num_runs)), 1)
        rows = math.ceil(num_runs / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))

        for idx, run_name in enumerate(run_names):
            ax = axes[idx // cols, idx % cols] if rows > 1 else axes[idx]
            ax.hist(all_logL_data[run_name], bins=30, edgecolor="k", alpha=0.7)
            ax.set_title(run_name)
            ax.set_xlabel('LogProb')
            ax.set_ylabel('Frequency')

        # If there are empty subplots, hide them
        for idx in range(num_runs, rows * cols):
            ax = axes[idx // cols, idx % cols] if rows > 1 else axes[idx]
            ax.axis('off')
        
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        print()

    # compute gelman-rubin statistic for convergence across reps
    print("Checking for convergence across runs...")
    R_hat = compute_R_hat(list(all_logL_data.values()))
    print("R-hat for LogProb:", R_hat)

    # parse popmap
    pop_mapping = {}
    if args.popmap:
        pop_mapping = parse_popmap(args.popmap)

    mig_est_data = {}
    mig_err_data = {}

    # parse migration matrices 
    for filename in os.listdir(args.outdir):
        filepath = os.path.join(args.outdir, filename)
        if "trace.txt" not in filename:
            run_name = filename 
            mig_est, mig_err = parse_ba3_results(filepath)

            if args.popmap:
                # Update the row and column names using the pop_mapping
                mig_est.columns = [pop_mapping[i] for i in mig_est.columns]
                mig_est.index = [pop_mapping[i] for i in mig_est.index]
                mig_err.columns = [pop_mapping[i] for i in mig_err.columns]
                mig_err.index = [pop_mapping[i] for i in mig_err.index]

            mig_est_data[run_name] = mig_est
            mig_err_data[run_name] = mig_err

    # calculate deviance 
    bayesian_deviances = {}
    for run_name, log_prob_trace in all_logL_data.items():
        bayesian_deviances[run_name] = calculate_bayesian_deviance(log_prob_trace)

    # format as pd and print 
    deviance_df = pd.DataFrame(list(bayesian_deviances.items()), columns=['Run Name', 'Bayesian Deviance'])
    print("Bayesian Deviances:")
    print(deviance_df)

    best_run = deviance_df.sort_values(by='Bayesian Deviance', ascending=True).iloc[0]
    print("The best run by Bayesian Deviance is:", best_run['Run Name'], "with a deviance of:", best_run['Bayesian Deviance'])
    print()

    # formatted tables 
    # table for best (by bayesian deviance)
    # table for average (across all reps)

    # plot histogram of within-pop rates, with lines showing the prior 
    # bounds (2/3 and 1.0)

    # heatmap of rates 

    # graph of rates (with optional coordinates supplied)


def calculate_bayesian_deviance(log_prob_trace):
    return -2 * np.mean(log_prob_trace)


def parse_ba3_results(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # extract pop name map
    map_line_index = lines.index("Population Index -> Population Label:") + 1
    map_line = lines[map_line_index]
    mapping_pairs = map_line.split()
    mapping = {int(pair.split("->")[0]): pair.split("->")[1] for pair in mapping_pairs}

    # empty matrices to hold data
    n = len(mapping)
    mig_est_matrix = np.zeros((n, n))
    mig_err_matrix = np.zeros((n, n))
    
    # parse mig matrix
    mig_rates_index = lines.index('Migration Rates:')
    for line in lines[mig_rates_index+1:]:
        if not line.startswith("m"):
            break
        parts = line.split()
        for i in range(0, len(parts), 2):
            part_key = parts[i]
            part_value = parts[i+1]

            row, col = map(int, [part_key.split('[')[1].split(']')[0], part_key.split('[')[2].split(']')[0]])
            value, stdev = part_value.split('(')
            stdev = stdev.rstrip(')')
            mig_est_matrix[row, col] = float(value)
            mig_err_matrix[row, col] = float(stdev)
                
    # convert to pop names
    df_mig_est = pd.DataFrame(mig_est_matrix)
    df_mig_err = pd.DataFrame(mig_err_matrix)
    df_mig_est.columns = [mapping[i] for i in df_mig_est.columns]
    df_mig_est.index = [mapping[i] for i in df_mig_est.index]
    df_mig_err.columns = [mapping[i] for i in df_mig_err.columns]
    df_mig_err.index = [mapping[i] for i in df_mig_err.index]
    
    return df_mig_est, df_mig_err



def parse_trace_file(filepath):
    # Load tab-delimited trace file into a pandas dataframe
    table_df = pd.read_csv(filepath, sep="\t", skipinitialspace=True)

    # Extract run name from the filename
    run_name = os.path.basename(filepath).replace(".trace.txt", "")

    return run_name, table_df


def trace_and_autocorr_plot(run_name, logL_series, burn_in):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # Full trace plot with burn-in line
    axes[0].plot(logL_series, color="blue")
    axes[0].axvline(x=burn_in, color='red', linestyle='--')
    axes[0].set_title(f"Trace plot of logL for {run_name}")
    
    # Post burn-in trace plot
    axes[1].plot(logL_series[burn_in:], color="green")
    axes[1].set_title(f"Post burn-in trace plot of logL for {run_name}")
    
    # Autocorrelation plot
    pd.plotting.autocorrelation_plot(logL_series[burn_in:], ax=axes[2])
    axes[2].set_title(f"Autocorrelation of logL for {run_name}")
    
    fig.tight_layout()
    return fig


def parse_popmap(filename):
    """Parse the population map file and return the index-to-label mapping."""
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    mapping = {}
    for line in lines:
        parts = line.split()
        mapping[parts[0]] = parts[1]

    return mapping


def compute_R_hat(log_probs):
    M = len(log_probs)
    N = len(log_probs[0])

    # between-chain variance
    chain_means = [np.mean(chain) for chain in log_probs]
    global_mean = np.mean(chain_means)
    B = N * np.var(chain_means, ddof=1)

    # within-chain variance
    W = np.mean([np.var(chain, ddof=1) for chain in log_probs])

    # R-hat
    var_theta = (1 - 1/N) * W + (1/N) * B
    R_hat = np.sqrt(var_theta / W)

    return R_hat


def mean_test(series, x_prop, alpha):
    n = len(series)
    x_percent = x_prop * 100
    x_len = int(n * x_prop)
    first_x = series[:x_len]
    last_x = series[-x_len:]
    
    t_stat, p_val = ttest_ind(first_x, last_x, equal_var=False)  # Using Welch's t-test
    if p_val < alpha:
        print(f"The means of the first and last {x_percent}% are statistically different (p-value: {p_val:.5f}).")
    else:
        print(f"The means of the first and last {x_percent}% are not statistically different (p-value: {p_val:.5f}).")

    return p_val


def parse_args():
    parser = argparse.ArgumentParser(description="Parse output files from BayesAss")
    
    # Arguments
    parser.add_argument('--outdir', default="output", help="Directory with output files")
    parser.add_argument('--out', default="ba3_combined", help="Desired prefix of collated output")
    parser.add_argument('--popmap', default=None, help="Optional tsv file mapping integer to string population labels")
    parser.add_argument('--x', type=float, default=0.2, help="Proportion used for the mean test")
    parser.add_argument('--alpha', type=float, default=0.05, help="Significance level for the t-test")
    parser.add_argument("--burn", type=float, default=0.5, help="Proportion of samples which were discarded as burn-in")
    return parser.parse_args()

if __name__ == "__main__":
    main()
