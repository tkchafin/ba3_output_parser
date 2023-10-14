import sys 
import os

import argparse

import math
import pymc3 as pm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ttest_ind
from scipy.stats import norm
import numpy as np
from statsmodels.tsa.stattools import acf
import networkx as nx
import matplotlib.gridspec as gridspec

plt.rcParams['figure.max_open_warning'] = 40


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
                    # test for convergence 
                    burn_idx = int(args.burn * len(table_df))
                    post_burn_table_df = table_df[burn_idx:]
                    print(run_name, end=": ")
                    geweke_test(post_burn_table_df['LogProb'])

                    # report effective sample size 
                    ess = pm.ess(post_burn_table_df['LogProb'].values)
                    print(f"{run_name}: Effective sample size = {ess}")

                    # make plots 
                    fig = trace_and_autocorr_plot(run_name, table_df['LogProb'], burn_idx)
                    pdf.savefig(fig)
                    plt.close(fig)
                    print()

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
        print("Plotted traces to",pdf_path)
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
    print(best_run['Run Name'])
    # table for best (by bayesian deviance)
    # table for average (across all reps)
    print("Writing combined output...")
    all_est, all_err = save_to_excel(mig_est_data, mig_err_data, best_run['Run Name'], f"{args.out}_combined.xlsx")
    if args.tsv:
        save_to_tsv(mig_est_data, mig_err_data, best_run['Run Name'], args.out)
    print()

    # get coords if they exist otherwise set to None
    if args.coords:
        coords = read_coords_to_dict(args.coords)
    else:
        coords = None

    pdf_path = f"{args.out}_plots.pdf"
    print("Generating plots...")
    with PdfPages(pdf_path) as pdf:
        # Stacked histogram of combined within-population migration rates from all runs
        fig = plot_combined_within_pop_histogram(mig_est_data)
        pdf.savefig(fig)
        plt.close(fig)

        # rates plots 
        data_dict = {**mig_est_data, 'Average': all_est, 'Best Run': mig_est_data[best_run["Run Name"]]}
        for dataset_name, data in data_dict.items():
            fig = plot_rates(data, 
                             dataset_name, 
                             node_size=500,
                             font_size=6,
                             threshold=args.threshold,
                             max_line_weight=10,
                             pos=coords,
                             scale_alpha=True)
            pdf.savefig(fig)
            plt.close(fig)

        print("Saved plots to", pdf_path)
        print()


def plot_rates(data, dataset_name, max_line_weight=5, max_arrowsize=25, 
               threshold=0.02, node_size=700, font_size=8, scale_alpha=True, pos=None):

    # Create a 2x2 grid of subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Heatmap
    sns.heatmap(data, ax=ax1, cmap="YlGnBu", annot=True, fmt=".2f")
    ax1.set_title(f"Heatmap: {dataset_name}")

    # Determine vmin and vmax for each dataset
    non_diagonal_values = data.values[~np.eye(data.shape[0], dtype=bool)]
    vmin = np.min(non_diagonal_values)
    vmax = np.max(non_diagonal_values)

    G = nx.DiGraph()  # Using a Directed Graph to represent arrows

    for pop in data.columns:
        G.add_node(pop)

    # Using min-max scaled data for weights
    def min_max_scale(df):
        non_diagonal_values = df.values[~np.eye(df.shape[0], dtype=bool)]
        min_val, max_val = np.min(non_diagonal_values), np.max(non_diagonal_values)
        df_scaled = (df - min_val) / (max_val - min_val)
        np.fill_diagonal(df_scaled.values, 0)  # setting diagonal values to 0
        return df_scaled

    scaled_data = min_max_scale(data)

    for i, source in enumerate(data.columns):
        for j, target in enumerate(data.columns[i+1:]):
            weight_from = scaled_data.at[source, target]
            weight_to = scaled_data.at[target, source]

            if scale_alpha:
                alpha_to = weight_to
                alpha_from = weight_from
            else:
                alpha_to = alpha_from = 1

            G.add_edge(source, target, alpha_from=alpha_from, alpha_to=alpha_to,
                        weight_from=weight_from * max_line_weight, 
                        weight_to=weight_to * max_line_weight,
                        rate_from=data.at[source, target], rate_to=data.at[target, source])

    # Helper function to draw the graph
    def draw_graph(G, pos, ax, dataset_name, arrow_scaling, vmin, vmax, font_size, node_size):
        if pos is None:
            pos = nx.spring_layout(G)
        for (source, target, data) in G.edges(data=True):
            # Same code as before to draw the edges
            if data['rate_from'] > threshold:
                color_from = plt.cm.viridis((data['rate_from'] - vmin) / (vmax - vmin))
                arrow_size_from = max(min(data['weight_from'] * arrow_scaling, max_arrowsize), 1)
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(source, target)], alpha=data['alpha_from'],
                                    width=data['weight_from'], edge_color=color_from, connectionstyle="arc3,rad=0.3", 
                                    arrowstyle='-|>', arrowsize=arrow_size_from)
            
            if data['rate_to'] > threshold:
                color_to = plt.cm.viridis((data['rate_to'] - vmin) / (vmax - vmin))
                arrow_size_to = max(min(data['weight_to'] * arrow_scaling, max_arrowsize), 1)
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(target, source)], alpha=data['alpha_to'],
                                    width=data['weight_to'], edge_color=color_to, connectionstyle="arc3,rad=0.3", 
                                    arrowstyle='<|-', arrowsize=arrow_size_to)
        
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="white", edgecolors="black", node_size=node_size)
        ax.set_title(f"Graph: {dataset_name}")
        ax.axis("off")

        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        plt.colorbar(sm, ax=ax, orientation='vertical')

    max_width = max_line_weight
    arrow_scaling = max_arrowsize / max_width

    # Plot 2: Graph with Auto Layout
    draw_graph(G, None, ax2, dataset_name, arrow_scaling, vmin, vmax, font_size, node_size)

    # Plot 3: Graph with Provided Coordinates (if available)
    if pos:
        draw_graph(G, pos, ax3, dataset_name, arrow_scaling, vmin, vmax, font_size, node_size)

    # Plot 4: Net Emigration Bar Chart
    net_emigration = data.sum(axis=1) - data.sum(axis=0)  # sum of outgoing rates - sum of incoming rates
    colors = ["red" if val < 0 else "blue" for val in net_emigration]
    ax4.bar(net_emigration.index, net_emigration, color=colors)
    ax4.axhline(0, color="black", linestyle="--")
    ax4.set_title("Net Emigration by Population")
    ax4.set_ylabel("Net Emigration Rate")

    fig.tight_layout()

    return fig


def plot_combined_within_pop_histogram(mig_est_data):
    diagonals = []
    for run_name, data in mig_est_data.items():
        diagonals.extend([data[pop][pop] for pop in data.columns])
    fig, ax = plt.subplots()
    sns.histplot(diagonals, ax=ax, kde=False, bins=30)
    ax.axvline(2/3, color='r', linestyle='--', label='Lower prior bound')
    ax.axvline(1.0, color='b', linestyle='--', label='Upper prior bound')
    ax.set_title('Combined Within-Population Migration Rates from All Runs')
    ax.set_xlabel('Migration Rate')
    ax.set_ylabel('Frequency')
    ax.legend()
    return fig


def read_coords_to_dict(coords_path):
    """
    Read the coordinate table and convert it into a dictionary mapping each sample 
    to a tuple containing its latitude and longitude.
    """
    df = pd.read_csv(coords_path, delim_whitespace=True)
    coords_dict = {row['sample']: (row['long'], row['lat']) for _, row in df.iterrows()}
    return coords_dict


def save_to_tsv(mig_est_data, mig_err_data, best_run, prefix):
    out_dir = f"{prefix}_tsv_files"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # runs
    for run_name, est in mig_est_data.items():
        est_file_name = os.path.join(out_dir, f"{prefix}_{run_name}_est.tsv")
        err_file_name = os.path.join(out_dir, f"{prefix}_{run_name}_err.tsv")
        
        est.to_csv(est_file_name, sep='\t', index=True, header=True)
        mig_err_data[run_name].to_csv(err_file_name, sep='\t', index=True, header=True)

    # average tables
    all_dfs = list(mig_est_data.values())
    avg_df = pd.concat(all_dfs).groupby(level=0).mean()
    std_df = pd.concat(all_dfs).groupby(level=0).std()

    avg_df.to_csv(os.path.join(out_dir, f"{prefix}_avg_est.tsv"), sep='\t', index=True, header=True)
    std_df.to_csv(os.path.join(out_dir, f"{prefix}_avg_err.tsv"), sep='\t', index=True, header=True)

    # best run
    mig_est_data[best_run].to_csv(os.path.join(out_dir, f"{prefix}_best_est.tsv"), sep='\t', index=True, header=True)
    mig_err_data[best_run].to_csv(os.path.join(out_dir, f"{prefix}_best_err.tsv"), sep='\t', index=True, header=True)

    print(f"TSV files saved in directory: {out_dir}")
    return avg_df, std_df


def save_to_excel(mig_est_data, mig_err_data, best_run, output_name):
    # For each run, combine the estimate and error 
    combined_data = {}
    for run_name, est in mig_est_data.items():
        err = mig_err_data[run_name]
        combined = est.applymap('{:.5f}'.format) + " (" + err.applymap('{:.5f}'.format) + ")"
        combined_data[run_name] = combined

    # across-run average and standard deviation 
    all_dfs = list(mig_est_data.values())
    avg_df = pd.concat(all_dfs).groupby(level=0).mean()
    std_df = pd.concat(all_dfs).groupby(level=0).std()
    avg_combined = avg_df.applymap('{:.5f}'.format) + " (" + std_df.applymap('{:.5f}'.format) + ")"

    # write to excel
    with pd.ExcelWriter(output_name) as writer:
        for run_name, data in combined_data.items():
            data.to_excel(writer, sheet_name=run_name)
        avg_combined.to_excel(writer, sheet_name="Average_Across_Runs")
        best_run_combined = mig_est_data[best_run].applymap('{:.5f}'.format) + " (" + mig_err_data[best_run].applymap('{:.5f}'.format) + ")"
        best_run_combined.to_excel(writer, sheet_name="Best_Run")
    print(f"Excel file saved as {output_name}")
    return avg_df, std_df


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
    table_df = pd.read_csv(filepath, sep="\t", skipinitialspace=True)
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


def geweke_test(series, first=0.1, last=0.5, alpha=0.05):
    n = len(series)
    
    first_slice = series[:int(first * n)]
    last_slice = series[-int(last * n):]

    mean_first, var_first = first_slice.mean(), first_slice.var()
    mean_last, var_last = last_slice.mean(), last_slice.var()

    z = (mean_first - mean_last) / np.sqrt(var_first / len(first_slice) + var_last / len(last_slice))

    p = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed p-value

    if p < alpha:
        print(f"The means of the first {first*100:.0f}% and last {last*100:.0f}% are statistically different (Z-score: {z:.5f}, p-value: {p:.5f}).")
    else:
        print(f"The means of the first {first*100:.0f}% and last {last*100:.0f}% are not statistically different (Z-score: {z:.5f}, p-value: {p:.5f}).")

    return p


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
    parser.add_argument("--burn", type=float, default=0.5, help="Proportion of samples which were discarded as burn-in")
    parser.add_argument("--coords", default=None, type=str, help="Optional tsv file with population coordinates")
    parser.add_argument("--tsv", action='store_true', help="Toggle on to write outputs formatted as tsv")
    parser.add_argument("--threshold", default=0.01, type=float, help="Threshold to show migration edges in graph plots")
    return parser.parse_args()

if __name__ == "__main__":
    main()
