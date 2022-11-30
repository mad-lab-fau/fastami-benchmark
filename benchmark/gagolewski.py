from random import seed
from typing import Optional, Sequence, Tuple, Callable
from time import perf_counter
import pandas as pd
import numpy as np
from numpy.random import SeedSequence, default_rng
from inspect import signature
from tqdm import tqdm

from .utils import Timeout

tqdm.pandas()


class GagolewskiBenchmark:
    def __init__(self, parquet_file, comparison_metrics: Optional[Sequence[Tuple[str, Callable]]], seed_sequence: SeedSequence, timeout_seconds: int = 20) -> None:
        """Cluster comparison benchmark.

        Args:
            parquet_file: Path to the parquet file containing the data.
            comparison_metrics: Metrics to use for benchmark, can also be appended later with run_benchmark.
            seed_sequence: Seed sequence to use for random number generation.
            timeout_seconds: Timeout for each comparison.

        Properties:
            results: Dataframe containing the results.
        """
        self._timeout_seconds = timeout_seconds
        self._prng = default_rng(seed_sequence.spawn(1)[0])
        self._data = pd.read_parquet(parquet_file)

        if comparison_metrics is not None:
            for metric_name, metric in comparison_metrics:
                print(f"Running Gagolewski Benchmark for {metric_name}")
                self.run_benchmark(metric_name, metric)

    def _compare(self, y1: Sequence[int], y2: Sequence[int], comparison_metric: Callable) -> Tuple[float, float, float, str]:
        """Compare two clusterings and measure the performance of the metric.

        Args:
            y1: First clustering.
            y2: Second clustering.
            comparison_metric: Metric to use for comparison.

        Returns:
            The metric value, the error and the average runtime of a single comparison.
        """
        try:
            with Timeout(seconds=self._timeout_seconds):
                if 'seed' in signature(comparison_metric).parameters.keys():
                    start = perf_counter()
                    val, val_err = comparison_metric(y1, y2, seed=self._prng)
                    stop = perf_counter()
                    return val, val_err, (stop - start), 'Normal'
                else:
                    start = perf_counter()
                    val = comparison_metric(y1, y2)
                    stop = perf_counter()
                    return val, 0.0, (stop - start), 'Normal'
        except TimeoutError as e:
            return np.nan, np.nan, float(self._timeout_seconds), str(e)
        except Exception as e:
            return np.nan, np.nan, np.nan, str(e)

    def run_benchmark(self, metric_name: str, metric: Callable) -> None:
        """Run the benchmark for a given metric.

        Args:
            metric_name: Name of the metric.
            metric: Metric to use for benchmark.
        """
        columns = [f"{metric_name}_value",
                   f"{metric_name}_error", f"{metric_name}_time", f"{metric_name}_comment"]
        self._data[columns] = self._data.progress_apply(
            lambda row: self._compare(row['pred_labels'], row['labels'], metric), axis=1, result_type='expand')

    @property
    def results(self) -> pd.DataFrame:
        """Return the results of the benchmark."""
        return self._data.drop(columns=['pred_labels', 'labels'])
