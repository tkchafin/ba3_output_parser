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

    # Check if directories exist
    if not os.path.exists(args.logdir):
        print(f"Error: Log directory {args.logdir} does not exist!")
        return
    if not os.path.exists(args.outdir):
        print(f"Error: Output directory {args.outdir} does not exist!")
        return

    all_logL_data = []
    all_run_names = []

    pdf_path = f"{args.out}_mcmc.pdf"
    with PdfPages(pdf_path) as pdf:
        for filename in os.listdir(args.logdir):
            filepath = os.path.join(args.logdir, filename)
            run_name, table_df = parse_log_file(filepath)

            if run_name and not table_df.empty:

                # mean test for convergence from post burn-in samples 
                burn_idx = int(args.burn * len(table_df))
                post_burn_table_df = table_df[burn_idx:]
                mean_test(post_burn_table_df['logL'], args.x, args.alpha)

                check_autocorrelation_and_thinning(post_burn_table_df['logL'])

                fig = trace_and_autocorr_plot(run_name, table_df['logL'], burn_idx)
                pdf.savefig(fig)
                plt.close(fig)

                all_logL_data.append(post_burn_table_df['logL'])
                all_run_names.append(run_name)

        # After all runs are processed, plot the grid of histograms
        num_runs = len(all_run_names)
        cols = max(int(np.sqrt(num_runs)), 1)
        rows = math.ceil(num_runs / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))

        for idx, (logL, run_name) in enumerate(zip(all_logL_data, all_run_names)):
            ax = axes[idx // cols, idx % cols] if rows > 1 else axes[idx]
            ax.hist(logL, bins=30, edgecolor="k", alpha=0.7)
            ax.set_title(run_name)
            ax.set_xlabel('logL')
            ax.set_ylabel('Frequency')
        
        # If there are empty subplots, hide them
        for idx in range(num_runs, rows * cols):
            ax = axes[idx // cols, idx % cols] if rows > 1 else axes[idx]
            ax.axis('off')
        
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

def parse_log_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read().replace('\r', '\n')
        lines = content.split('\n')

    first_non_blank = next((line for line in lines if line.strip()), None)
    if not first_non_blank or "BA3-SNPS" not in first_non_blank:
        return None, None

    run_name = None
    table_data = {
        'logP_M': [],
        'logL_G': [],
        'logL': []
    }

    for line in lines:
        line = line.strip()
        # Extract run name
        if "Output file:" in line:
            run_name = line.split("Output file:")[1].strip()

        # Collect trace 
        if line.startswith("logP(M):"):
            parts = line.split()
            table_data['logP_M'].append(float(parts[1]))
            table_data['logL_G'].append(float(parts[3]))
            table_data['logL'].append(float(parts[5]))

    table_df = pd.DataFrame(table_data)

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
    parser.add_argument('--logdir', default="logs", help="Directory with log files")
    parser.add_argument('--outdir', default="output", help="Directory with output files")
    parser.add_argument('--out', default="ba3_combined", help="Desired prefix of collated output")
    parser.add_argument('--popmap', default=None, help="Optional tsv file mapping integer to string population labels")
    parser.add_argument('--x', type=float, default=0.2, help="Proportion used for the mean test")
    parser.add_argument('--alpha', type=float, default=0.05, help="Significance level for the t-test")
    parser.add_argument("--burn", type=float, default=0.5, help="Proportion of samples which were discarded as burn-in")
    return parser.parse_args()

if __name__ == "__main__":
    main()
