from time import sleep
from benchmark import SnapBenchmark, EmailCoreDataset, WikiDataset, AmazonDataset, FriendsterDataset, OrkutDataset, DblpDataset, LiveJournalDataset, YoutubeDataset, TestDataset
from fastami import adjusted_mutual_info_mc, adjusted_mutual_info_pairwise
from sklearn.metrics.cluster import adjusted_mutual_info_score
from numpy.random import SeedSequence
import pandas as pd
import numpy as np
from scipy.stats.stats import spearmanr

from pathlib import Path


def baseline(labels_true, labels_pred):
    sleep(0.1)
    return np.nan


if __name__ == "__main__":
    # Parameters
    seed = 12345
    snap_base_path = Path(
        './data/snap')
    output_path = Path('./results')

    # Benchmark
    seed_sequence = SeedSequence(seed)

    print("--- SNAP benchmark ---\n")
    snap_datasets = [
        TestDataset,
        EmailCoreDataset,
        AmazonDataset,
        DblpDataset,
        YoutubeDataset,
        LiveJournalDataset,
        WikiDataset,
        OrkutDataset,
        FriendsterDataset
    ]

    algorithms = [
        ('baseline', baseline),
        ('ami_exact', adjusted_mutual_info_score),
        ('ami_mc', adjusted_mutual_info_mc),
        ('ami_pairwise', adjusted_mutual_info_pairwise)
    ]

    snap_benchmark = SnapBenchmark(
        base_path=snap_base_path,
        datasets=snap_datasets,
        comparison_metrics=algorithms,
        seed_sequence=seed_sequence)

    timeout = 2_000
    result_snap = snap_benchmark.run(metric_timeout=timeout)

    result_snap.to_csv(output_path / 'snap_results.csv', index=False)

    # Pivot values
    df_values = result_snap.pivot(
        index=['dataset', 'clustering_1', 'clustering_2'], columns='metric', values='value')

    # Calculate absolute errrors
    df_error = df_values.drop(columns=['baseline'])
    df_error['error_pairwise'] = abs(
        df_error['ami_pairwise'] - df_error['ami_exact'])
    df_error['error_pairwise_est'] = abs(
        df_error['ami_pairwise'] - df_error['ami_mc'])
    df_error['error_mc'] = abs(
        df_error['ami_mc'] - df_error['ami_exact'])
    df_error = df_error[['error_mc',
                         'error_pairwise',
                         'error_pairwise_est']]

    result_snap.rename(columns={'error': 'error_mc_est'}, inplace=True)
    result_snap = result_snap.merge(df_error, how='left', left_on=[
        'dataset', 'clustering_1', 'clustering_2'], right_index=True)
    result_snap['absolute_error'] = np.nan
    result_snap['is_error_estimated'] = False

    mask = result_snap['metric'] == 'ami_pairwise'
    result_snap.loc[mask,
                    'absolute_error'] = result_snap.loc[mask, 'error_pairwise']
    mask &= (result_snap['absolute_error'].isna())
    result_snap.loc[mask, 'absolute_error'] = result_snap.loc[mask,
                                                              'error_pairwise_est']
    result_snap.loc[mask, 'is_error_estimated'] = True

    mask = result_snap['metric'] == 'ami_mc'
    result_snap.loc[mask, 'absolute_error'] = result_snap.loc[mask, 'error_mc']
    mask &= (result_snap['absolute_error'].isna())
    result_snap.loc[mask,
                    'absolute_error'] = result_snap.loc[mask, 'error_mc_est']
    result_snap.loc[mask, 'is_error_estimated'] = True

    mask = result_snap['metric'] == 'ami_exact'
    result_snap.loc[mask, 'absolute_error'] = 0.0

    result_snap['is_timeout'] = result_snap['comment'] == 'Timeout'
    result_snap['is_memory_error'] = result_snap['wall_time_seconds'].isna(
    ) & result_snap['process_time_seconds'].isna() & result_snap['max_memory_MiB'].notna()
    result_snap['success'] = result_snap['value'].notna()

    df_grouped = result_snap.groupby(['dataset', 'metric']).agg(total_process_time=(
        'process_time_seconds', 'sum'), total_wall_time=('wall_time_seconds', 'sum'), max_memory_MiB=('max_memory_MiB', 'max'), is_timeout=('is_timeout', 'any'), is_oom=('is_memory_error', 'any'), mean_absolute_error=('absolute_error', 'mean'), is_error_estimated=('is_error_estimated', 'any'), comparisons=('clustering_1', 'count'), result_returned=('success', 'sum'))

    # Subtract baseline memory
    baseline_memory = df_grouped.loc[df_grouped.index.get_level_values(
        'metric') == 'baseline', 'max_memory_MiB'].min()
    df_grouped.drop(index='baseline', level=1, inplace=True)
    df_grouped['max_memory_MiB'] -= baseline_memory

    df_grouped['spearman'] = np.nan
    df_grouped['is_spearman_complete'] = True

    for dataset in result_snap['dataset'].unique():
        # Calculate spearman correlation
        selected_df = df_values[df_values.index.get_level_values(
            'dataset') == dataset]

        df_grouped.at[(dataset, 'ami_pairwise'), 'spearman'] = spearmanr(
            selected_df['ami_exact'], selected_df['ami_pairwise'], nan_policy='omit').correlation
        df_grouped.at[(dataset, 'ami_mc'), 'spearman'] = spearmanr(
            selected_df['ami_exact'], selected_df['ami_mc'], nan_policy='omit').correlation
        df_grouped.at[(dataset, 'ami_exact'), 'spearman'] = 1.0

        if selected_df['ami_exact'].isna().any():
            df_grouped.loc[df_grouped.index.get_level_values(
                'dataset') == dataset, 'is_spearman_complete'] = False

    df_grouped.to_csv(output_path / 'snap_results_analysis.csv', index=True)
    print(df_grouped)
