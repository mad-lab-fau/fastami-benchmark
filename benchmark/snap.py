import pandas as pd
import numpy as np
from numpy.random import default_rng, SeedSequence
import networkit as nk
from pathlib import Path
from tqdm import tqdm
from contextlib import redirect_stdout
from itertools import combinations
from memory_profiler import memory_usage
from typing import Callable, List, Tuple
from time import process_time, perf_counter
from inspect import signature

from .snap_datasets import SnapGroundTruthDataset
from .utils import Timeout, get_total_memory, set_memory_limit


class SnapBenchmark:
    def __init__(self, base_path: Path, comparison_metrics: List[Tuple[str, Callable]], datasets: List[SnapGroundTruthDataset], seed_sequence: SeedSequence) -> None:
        """ Benchmark object for comparison metrics on a set of SNAP datasets.

        Clusters are calculated for each dataset using:
        - PLM
        - PLP
        - LouvainMapEquation
        - LPDegreeOrdered
        - ParallelLeiden
        - ParallelConnectedComponents

        The benchmark consists then of pairwise comparing the results of each 
        algorithm on each dataset with the specified comparison metrics.

        Args:
            base_path: Path to the directory where the datasets are stored.
            comparison_metrics: List of clustering comparison metrics (name, metric)
            datasets: List of SNAP datasets to benchmark.
            seed_sequence: Seed sequence for the random number generator.
        """
        self._base_path = base_path
        self._datasets = datasets
        self._prng = default_rng(seed_sequence.spawn(1)[0])

        self._community_detectors = [
            ('plm', nk.community.PLM),
            ('plp', nk.community.PLP),
            ('map', nk.community.LouvainMapEquation),
            ('lpd', nk.community.LPDegreeOrdered),
            ('pld', nk.community.ParallelLeiden),
            ('pcc', nk.components.ParallelConnectedComponents)
        ]

        self._comparision_metrics = comparison_metrics
        self._memory_limit_bytes = get_total_memory() * 1024 - 1_048_576
        set_memory_limit(self._memory_limit_bytes)

    def _calculate_clusters(self, timeout_seconds: int) -> None:
        """ Calculate clusters for all datasets.

        Uses the following community detection algorithms from networkit:
        - PLM
        - PLP
        - LouvainMapEquation
        - LPDegreeOrdered
        - ParallelLeiden
        - ParallelConnectedComponents

        Args:
            timeout_seconds: Timeout for each clustering algorithm.
        """
        pbar_ds = tqdm(self._datasets)
        for dataset in pbar_ds:
            first_run = True
            ds = dataset(self._base_path)
            pbar_ds.set_description(ds.name)
            pbar_cm = tqdm(self._community_detectors, leave=False)
            for name, algo in pbar_cm:
                pbar_cm.set_description(name)

                output_path = self._base_path / f'{ds.name}_{name}.partition'
                if not output_path.is_file():
                    if first_run:
                        ds.graph
                        first_run = False
                    try:
                        with redirect_stdout(None):
                            with Timeout(seconds=timeout_seconds):
                                result = nk.community.detectCommunities(
                                    ds.graph, algo(ds.graph))
                            nk.community.writeCommunities(
                                result, str(output_path.resolve()))
                    except TimeoutError:
                        tqdm.write(
                            f"Skipped {name} for {ds.name}, since it took longer than {timeout_seconds} s.")
            del ds

    def _profile_metric(self, comparison_metric: Callable, y_1, y_2, timeout_seconds: int) -> Tuple[float, float, float, int, str]:
        """ Benchmark a comparison metric on a single pair of input labels.

        Args:
            comparison_metric: Comparison metric to profile.
            y_1: First set of clusters.
            y_2: Second set of clusters.
            timeout_seconds: Timeout for the comparison metric.

        Returns:
            Tuple of (result, error estimate, process time, walltime, comment).
        """
        kwargs = dict()
        if 'seed' in signature(comparison_metric).parameters.keys():
            kwargs['seed'] = self._prng
        start_time = process_time()
        start_time_wall = perf_counter()
        try:
            with Timeout(seconds=timeout_seconds):
                max_memory, ret_val = memory_usage(proc=(
                    comparison_metric, (y_1, y_2), kwargs), timeout=timeout_seconds, max_usage=True, retval=True)
        except MemoryError:
            return np.nan, np.nan, np.nan, np.nan, self._memory_limit_bytes/2**20, 'OOM'
        except Exception as e:
            return np.nan, np.nan, timeout_seconds, timeout_seconds, np.nan, str(e)
        stop_time_wall = perf_counter()
        stop_time = process_time()

        if isinstance(ret_val, tuple):
            val, val_err = ret_val
        else:
            val = ret_val
            val_err = 0
        return val, val_err, stop_time - start_time, stop_time_wall - start_time_wall, max_memory, 'Normal'

    def _benchmark_comparison_metrics(self, timeout_seconds) -> pd.DataFrame:
        """ Benchmark comparison metrics on all datasets.

        Args:
            timeout_seconds: Timeout for each comparison metric.

        Returns:
            Pandas DataFrame with the results.
        """
        results = {'metric': [], 'dataset': [], 'clustering_1': [], 'clustering_2': [
        ], 'value': [], 'error': [], 'process_time_seconds': [], 'wall_time_seconds': [], 'max_memory_MiB': [], 'comment': []}

        ds_pbar = tqdm(self._datasets)
        for ds in ds_pbar:
            ds_pbar.desc = ds.name
            metric_pbar = tqdm(self._comparision_metrics, leave=False)
            for metric_name, comparison_metric in metric_pbar:
                metric_pbar.desc = metric_name

                community_detectors = [name for name,
                                       _ in self._community_detectors if (self._base_path / f"{ds.name}_{name}.partition").is_file()]
                n_detectors = len(community_detectors)

                pairs_pbar = tqdm(combinations(community_detectors, 2),
                                  total=n_detectors*(n_detectors - 1)//2, leave=False)
                for clustering_1, clustering_2 in pairs_pbar:
                    results['metric'].append(metric_name)
                    results['dataset'].append(ds.name)
                    results['clustering_1'].append(clustering_1)
                    results['clustering_2'].append(clustering_2)
                    y_1 = np.loadtxt(self._base_path /
                                     f'{ds.name}_{clustering_1}.partition')
                    y_2 = np.loadtxt(self._base_path /
                                     f'{ds.name}_{clustering_2}.partition')
                    val, err, elapsed_process_time, elapsed_walltime, max_memory, comment = self._profile_metric(
                        comparison_metric, y_1, y_2, timeout_seconds=timeout_seconds)
                    results['value'].append(val)
                    results['error'].append(err)
                    results['process_time_seconds'].append(
                        elapsed_process_time)
                    results['wall_time_seconds'].append(elapsed_walltime)
                    results['max_memory_MiB'].append(max_memory)
                    results['comment'].append(comment)

        return pd.DataFrame(results)

    def run(self, clustering_timeout: int = 2_000, metric_timeout: int = 2_000) -> pd.DataFrame:
        """ Run the benchmark.

        Args:
            clustering_timeout: Timeout for each clustering algorithm.
            metric_timeout: Timeout for each comparison metric.

        Returns:
            Pandas DataFrame with the results.
        """
        print("\nCalculating Clusterings ...")
        self._calculate_clusters(clustering_timeout)

        print("\nBenchmarking comparison metrics ...")
        return self._benchmark_comparison_metrics(metric_timeout)
