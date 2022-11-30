from time import sleep
from benchmark import GagolewskiBenchmark, SnapBenchmark, SyntheticBenchmark, EmailCoreDataset, WikiDataset, AmazonDataset, FriendsterDataset, OrkutDataset, DblpDataset, LiveJournalDataset, YoutubeDataset, TestDataset
from fastami import adjusted_mutual_info_mc, adjusted_mutual_info_pairwise, standardized_mutual_information_separate_mc, standardized_mutual_information_direct_mc, standardized_mutual_information_exact, expected_mutual_information_mc, expected_mutual_information_pairwise
from sklearn.metrics.cluster import expected_mutual_information, adjusted_mutual_info_score
from numpy.random import SeedSequence, default_rng
import pandas as pd
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    # Parameters
    seed = 12345
    gagolewski_parquet_path = Path(
        './data/gagolewski/')
    output_path = Path('./results')

    # Benchmark
    seed_sequence = SeedSequence(seed)

    print("--- Gagolewski benchmark ---\n")

    algorithms = [
        ('ami_exact', adjusted_mutual_info_score),
        ('ami_mc', adjusted_mutual_info_mc),
        ('ami_pairwise', adjusted_mutual_info_pairwise),
        ('smi_exact', standardized_mutual_information_exact),
        ('smi_direct', standardized_mutual_information_direct_mc)
    ]

    gagolewski_benchmark = GagolewskiBenchmark(
        parquet_file=gagolewski_parquet_path,
        comparison_metrics=algorithms,
        seed_sequence=seed_sequence,
        timeout_seconds=20)

    result_gagolewski = gagolewski_benchmark.results

    result_gagolewski.to_csv(
        output_path / 'gagolewski_results.csv', index=False)

    # Analyze results
    result_gagolewski['smi_exact_value'] = result_gagolewski['smi_exact_value'].astype(
        complex).astype(float)
    # Set timeouts to nan
    for c in result_gagolewski.columns:
        if c.endswith('_time'):
            result_gagolewski.loc[result_gagolewski[c] == 20.0, c] = np.nan

    result_df = {'metric_family': [], 'metric': [], 'result_returned': [],
                 'mean_time': [], 'spearman': []}

    for metric, _ in algorithms:
        metric_family = metric.split('_')[0]
        result_df['metric_family'].append(metric_family)
        result_df['metric'].append(metric)
        returned_result = result_gagolewski[f'{metric}_value'].notna().sum()
        result_df['result_returned'].append(
            returned_result/len(result_gagolewski))
        result_df['mean_time'].append(
            result_gagolewski[f'{metric}_time'].sum())
        result_df['spearman'].append(result_gagolewski[f'{metric}_value'].corr(
            result_gagolewski[f'{metric_family}_exact_value'], method='spearman'))

    result_df = pd.DataFrame(result_df)

    result_df['metric'] = result_df['metric'].replace(
        {'ami_mc': 'FastAMI', 'ami_pairwise': 'pairwise AMI', 'ami_exact': 'AMI (sklearn)', 'smi_direct': 'FastSMI', 'smi_exact': 'SMI (exact)'})

    result_df.to_csv(
        output_path / 'gagolewski_results_analysis.csv', index=False)
    print(result_df)
