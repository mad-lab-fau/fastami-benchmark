from benchmark import SyntheticBenchmark
from fastami import standardized_mutual_information_separate_mc, standardized_mutual_information_direct_mc, standardized_mutual_information_exact, expected_mutual_information_mc, expected_mutual_information_pairwise
from sklearn.metrics.cluster import expected_mutual_information
from numpy.random import SeedSequence
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np

from pathlib import Path


def q95(x):
    return x.quantile(0.95)


def q5(x):
    return x.quantile(0.05)


def plot_emi_runtime(results, filename):
    # Plot EMI for different values of N
    r_max = results['R'].max()
    n_total_max = results['N'].max()
    ndf = results[(results['R'] == r_max) | (results['N'] != n_total_max)]
    ndf = ndf.groupby(['N', 'R', 'C']).agg(
        time_exact=('exact_time', 'mean'),
        time_exact_err=('exact_time', 'std'),
        time_exact_err_min=('exact_time', q5),
        time_exact_err_max=('exact_time', q95),
        time_pair=('pairwise_time', 'mean'),
        time_pair_err=('pairwise_time', 'std'),
        time_pair_err_min=('pairwise_time', q5),
        time_pair_err_max=('pairwise_time', q95),
        time_mc=('mc_time', 'mean'),
        time_mc_err=('mc_time', 'std'),
        time_mc_err_min=('mc_time', q5),
        time_mc_err_max=('mc_time', q95)
    ).reset_index()

    df = deepcopy(results[results['N'] == n_total_max])

    df['emi_rel_difference_mc'] = abs(
        (df['mc_emi'] - df['exact_emi']))/df['exact_emi']
    df['emi_rel_difference_pair'] = abs(
        (df['pairwise_emi'] - df['exact_emi']))/df['exact_emi']

    df = df.groupby(['N', 'R', 'C']).agg(
        time_exact=('exact_time', 'mean'),
        time_exact_err=('exact_time', 'std'),
        time_exact_err_min=('exact_time', q5),
        time_exact_err_max=('exact_time', q95),
        time_pair=('pairwise_time', 'mean'),
        time_pair_err=('pairwise_time', 'std'),
        time_pair_err_min=('pairwise_time', q5),
        time_pair_err_max=('pairwise_time', q95),
        time_mc=('mc_time', 'mean'),
        time_mc_err=('mc_time', 'std'),
        time_mc_err_min=('mc_time', q5),
        time_mc_err_max=('mc_time', q95),
        emi_rel_difference_mc=('emi_rel_difference_mc', 'mean'),
        emi_rel_difference_mc_std=('emi_rel_difference_mc', 'std'),
        emi_rel_difference_pair=('emi_rel_difference_pair', 'mean'),
        emi_rel_difference_pair_std=('emi_rel_difference_pair', 'std')
    ).reset_index()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(7.00697, 0.75*3.3374)
    ax1.text(-0.1, 1.1, '(a)', transform=ax1.transAxes)

    e1 = ax1.errorbar('R', 'time_exact', yerr=[df['time_exact_err_min'], df['time_exact_err_max']],
                      data=df, label='exact', ls='', marker='x', capsize=3)  # , capsize=3
    e2 = ax1.errorbar('R', 'time_mc', yerr=[df['time_mc_err_min'], df['time_mc_err_max']],
                      data=df, label='MC', ls='', marker='+', capsize=3)
    e3 = ax1.errorbar('R', 'time_pair', yerr=[df['time_pair_err_min'], df['time_pair_err_max']],
                      data=df, label='pairwise', ls='', marker='*', capsize=3)
    #ax1.legend(loc="upper left", mode="expand", ncol=3)
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$R, C$')
    ax1.set_ylabel(r'runtime [s]')

    ax1.text(0.03, 0.93, f'$N={n_total_max}$', ha='left',
             va='center', transform=ax1.transAxes)

    ax2.text(-0.1, 1.1, '(b)', transform=ax2.transAxes)

    precision_goal = results['precision_goal'].mean()
    ax2.axhline(precision_goal, color='C3', ls='--', alpha=0.9, zorder=0)
    ax2.errorbar('R', 'emi_rel_difference_mc', yerr='emi_rel_difference_mc_std',
                 data=df, label='MC', ls='', marker='+', capsize=3, color='C1')
    ax2.errorbar('R', 'emi_rel_difference_pair', yerr='emi_rel_difference_pair_std',
                 data=df, label='pairwise', ls='', marker='*', capsize=3, color='C2')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$R, C$')
    ax2.set_ylabel(r'relative error')

    ax2.text(0.97, 0.93, f'$N={n_total_max}$', ha='right',
             va='center', transform=ax2.transAxes)

    # Plot N varies
    ax3.text(-0.1, 1.1, '(c)', transform=ax3.transAxes)

    e1 = ax3.errorbar(ndf['N'], ndf['time_exact'], yerr=[ndf['time_exact_err_min'], ndf['time_exact_err_max']],
                      label='exact', capsize=3, ls='', marker='x')
    e2 = ax3.errorbar(ndf['N'], ndf['time_mc'], yerr=[ndf['time_mc_err_min'], ndf['time_mc_err_max']],
                      label='MC', capsize=3, ls='', marker='+')
    e3 = ax3.errorbar(ndf['N'], ndf['time_pair'], yerr=[ndf['time_pair_err_min'], ndf['time_pair_err_max']],
                      label='pairwise', capsize=3, ls='', marker='*')

    ax3.set_yscale('log')
    ax3.set_xlabel(r'$N$')
    ax3.set_ylabel(r'runtime [s]')
    ax3.text(0.97, 0.3, r'$R=C=\lfloor 0.9 N \rfloor$', ha='right',
             va='center', transform=ax3.transAxes)

    fig.legend(handles=[e1, e2, e3],
               labels=['exact', 'MC', 'pairwise'],
               loc="lower center",
               bbox_to_anchor=(0.5, 0.0),
               borderaxespad=0.1,
               ncol=3)

    plt.subplots_adjust(wspace=0.5, bottom=0.29)
    plt.savefig(str(filename.resolve()), format="pdf")


def plot_emi_values(results, filename):
    n_total_max = results['N'].max()
    df = deepcopy(results[results['N'] == n_total_max])

    df['exact_emi'] /= np.log(df['N'])
    df['mc_emi'] /= np.log(df['N'])
    df['mc_emi_err'] /= np.log(df['N'])
    df['pairwise_emi'] /= np.log(df['N'])

    df = df.groupby(['N', 'R', 'C']).agg(
        emi_exact=('exact_emi', 'mean'),
        emi_exact_std=('exact_emi', 'std'),
        emi_pair=('pairwise_emi', 'mean'),
        emi_pair_std=('pairwise_emi', 'std'),
        emi_mc=('mc_emi', 'mean'),
        emi_mc_std=('mc_emi', 'std')
    ).reset_index()

    # Rescale R,C
    df['R'] /= df['N']
    df['C'] /= df['N']

    # Add extreme values
    df = df.append({'R': 1.0/n_total_max, 'C': 1.0/n_total_max, 'emi_exact': 0.0, 'emi_exact_std': 0.0,
                    'emi_pair': 0.0, 'emi_pair_std': 0.0, 'emi_mc': 0.0, 'emi_mc_std': 0.0}, ignore_index=True)
    df = df.append({'N': n_total_max, 'R': 1.0, 'C': 1.0, 'emi_exact': 1.0, 'emi_exact_std': 0.0,
                    'emi_pair': 1.0, 'emi_pair_std': 0.0, 'emi_mc': 1.0, 'emi_mc_std': 0.0}, ignore_index=True)

    fig, ax = plt.subplots()
    fig.set_size_inches(3.3374, 0.75*3.3374)
    ax.errorbar('R', 'emi_exact', yerr='emi_exact_std',
                data=df, label='exact', ls='', marker='x')  # , capsize=3
    ax.errorbar('R', 'emi_mc', yerr='emi_mc_std',
                data=df, label='MC', ls='', marker='+')
    ax.errorbar('R', 'emi_pair', yerr='emi_pair_std',
                data=df, label='pairwise', ls='', marker='*')
    ax.text(0.03, 0.93, f'$N={n_total_max}$', ha='left',
            va='center', transform=ax.transAxes)
    ax.legend(loc="lower right")
    ax.set_xlabel(r'$R/N, C/N$')
    ax.set_ylabel(r'$\operatorname{EMI}/\log{N}$')
    plt.tight_layout()
    plt.savefig(filename, format="pdf")


def plot_smi_runtime(results, filename):
    df_grouped = results.groupby(['N', 'R', 'C']).agg(
        time_exact=('exact_time', 'mean'),
        time_exact_std=('exact_time', 'std'),
        time_exact_min=('exact_time', q5),
        time_exact_max=('exact_time', q95),
        time_mc=('separate_time', 'mean'),
        time_mc_err=('separate_time', 'std'),
        time_mc_min=('separate_time', q5),
        time_mc_max=('separate_time', q95),
        time_mc_std=('separate_time', 'std'),
        time_patefield=('direct_time', 'mean'),
        time_patefield_err=('direct_time', 'std'),
        time_patefield_min=('direct_time', q5),
        time_patefield_max=('direct_time', q95),
        time_patefield_std=('direct_time', 'std')
    ).reset_index()

    n_total = results['N'].max()

    # Get upper and lower bounds for error bars
    df_grouped['time_exact_err_lower'] = df_grouped['time_exact'] - \
        df_grouped['time_exact_min']
    df_grouped['time_exact_err_upper'] = df_grouped['time_exact_max'] - \
        df_grouped['time_exact']

    df_grouped['time_mc_err_lower'] = df_grouped['time_mc'] - \
        df_grouped['time_mc_min']
    df_grouped['time_mc_err_upper'] = df_grouped['time_mc_max'] - \
        df_grouped['time_mc']

    df_grouped['time_patefield_err_lower'] = df_grouped['time_patefield'] - \
        df_grouped['time_patefield_min']
    df_grouped['time_patefield_err_upper'] = df_grouped['time_patefield_max'] - \
        df_grouped['time_patefield']

    # Plot runtime over R,C
    fig, ax = plt.subplots(figsize=(3.3374, 0.75*3.3374))

    e1 = ax.errorbar(df_grouped['R'], df_grouped['time_exact'], yerr=[df_grouped['time_exact_err_lower'], df_grouped['time_exact_err_upper']],
                     ls='none', marker='x', label='exact SMI', capsize=3)
    e2 = ax.errorbar(df_grouped['R'], df_grouped['time_patefield'], yerr=[df_grouped['time_patefield_err_lower'], df_grouped['time_patefield_err_upper']],
                     ls='none', marker='+', label='direct MC', capsize=3)
    e3 = ax.errorbar(df_grouped['R'], df_grouped['time_mc'], yerr=[df_grouped['time_mc_err_lower'], df_grouped['time_mc_err_upper']],
                     ls='none', marker='*', label='separate MC', capsize=3)

    ax.text(0.03, 0.93, f'$N={n_total}$', ha='left',
            va='center', transform=ax.transAxes)

    # Set y-axis to log scale
    ax.set_yscale('log')

    # Set axis labels
    ax.set_xlabel('$R, C$')
    ax.set_ylabel('runtime [s]')

    ax.legend(loc='center right', bbox_to_anchor=(1.01, 0.72))
    fig.tight_layout()
    plt.savefig(filename)


if __name__ == "__main__":
    # Parameters
    seed = 1234
    output_path = Path('./results')

    # Create output directory if it does not exist
    output_path.mkdir(exist_ok=True)

    # Benchmark
    seed_sequence = SeedSequence(seed)
    synthetic_benchmark = SyntheticBenchmark(seed_sequence=seed_sequence)

    print("--- EMI benchmark ---\n")
    emi_algorithms = [('exact', expected_mutual_information),
                      ('mc', expected_mutual_information_mc),
                      ('pairwise', expected_mutual_information_pairwise)]
    result_synthetic_emi = synthetic_benchmark.benchmark_emi(algorithms=emi_algorithms,
                                                             n_total=5_000, precision_goal=0.01, partition_samples=200)

    result_synthetic_emi.to_csv(
        output_path / 'synthetic_emi_results.csv', index=False)
    plot_emi_runtime(result_synthetic_emi, output_path /
                     'synthetic_emi_runtime.pdf')
    plot_emi_values(result_synthetic_emi, output_path /
                    'synthetic_emi_values.pdf')

    print("\n--- SMI benchmark ---\n")
    smi_algorithms = [('direct', standardized_mutual_information_direct_mc),
                      ('separate', standardized_mutual_information_separate_mc),
                      ('exact', standardized_mutual_information_exact)]
    result_synthetic_smi = synthetic_benchmark.benchmark_smi(algorithms=smi_algorithms,
                                                             n_total=150, precision_goal=0.1, partition_samples=10)

    result_synthetic_smi.to_csv(
        output_path / 'synthetic_smi_results.csv', index=False)
    plot_smi_runtime(result_synthetic_smi, output_path /
                     'synthetic_smi_runtime.pdf')
