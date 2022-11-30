from typing import Callable, Dict, List, Tuple
from numpy.random import default_rng, SeedSequence
from multiprocessing import Pool
from psutil import cpu_count
import numpy as np
import pandas as pd
from time import perf_counter
from tqdm import tqdm
from sklearn.metrics.cluster import contingency_matrix
from inspect import signature, isbuiltin


class SyntheticBenchmark:
    def __init__(self, seed_sequence: SeedSequence) -> None:
        """Synthetic EMI and SMI benchmark.

        Args:
            seed_sequence: The seed sequence to use.
        """
        self._cpu_count = cpu_count(logical=False)
        self._seed_sequence = seed_sequence
        self._prng = default_rng(seed_sequence.spawn(1)[0])

    @staticmethod
    def _generate_random_partition(n: int, parts: int = 0, seed=None) -> List[int]:
        """Return a random partition of an integer via rejection sampling.

        The expected number of rejections is O(n^(3/4)) for unconstrained
        partitions. The corresponding expected runtime is O(n^5/4).
        This algorithm is originally due to Fristedt, The Structure of 
        Random Partitions of Large Integer (1993). A nice explanation can 
        be found in https://arxiv.org/abs/1110.3856 (Arratia, DeSalvo, 
        Probabilistic divide-and-conquer: a new exact simulation method, 
        with integer partitions as an example (2011) and on StackOverflow
        by the same author https://stackoverflow.com/questions/2161406/
        how-do-i-generate-a-uniform-random-integer-partition.


        Args:
            n: integer to be partitioned
            parts: number of parts (<= 0 for an unconstrained partition)
            seed: random seed to use same as numpy default_rng seed

        Returns:
            A 1-indexed list, where the index represents an integer
            and the corresponding list value is the number of occurences
            of that integer in the partition.

            Example:
            A partition of 13 is 5 + 4 + 2 + 2 and would be represented
            as [0,2,0,1,1].
        """
        prng = default_rng(seed=seed)
        if parts > 0:
            m = parts
        else:
            parts = 0
            m = n

        x = np.exp(-np.pi/np.sqrt(6*(n-parts)))
        s = parts
        while s != n:
            partition = np.empty(m, dtype=int)
            s = parts
            for i in range(1, m+1):
                z = prng.geometric(1-x**i) - 1
                partition[i-1] = z
                s += i*z

        if parts > 0:
            # Make sure there is at least one cluster of size parts
            partition[m - 1] += 1
            # Conjugate the young diagram
            conj = np.zeros(n, dtype=int)
            # initialize s to -1 for 0 indexing
            s = -1
            for z in reversed(partition):
                s += z
                conj[s] += 1
            partition = conj

        return partition

    @staticmethod
    def _get_random_clustering(partition: List[int], seed=None, shuffle: bool = True) -> List[int]:
        """Convert a partition to a clustering.

        Args:
            partition: A partition of an integer.
            seed: Random seed to use same as numpy default_rng seed
            shuffle: Whether to shuffle the clustering.

        Returns:
            A random instance of a set partition corresponding to the integer partition. 
        """
        prng = default_rng(seed=seed)
        labels = []
        max_label = 0
        for i, v in enumerate(partition, 1):
            for j in range(v):
                labels.append(np.full(i, j + max_label))
            max_label += v
        labels = np.concatenate(labels, axis=0)
        if shuffle:
            prng.shuffle(labels)
        return labels

    @staticmethod
    def _random_emi(parameters) -> Dict:
        """Benchmark the EMI algorithms on a single random pair.

        Args:
            parameters: tuple with algorithms, N, the number of clusters (R=C), prng, precision_goal.

        Returns:
            A dictionary with the results.
        """
        algorithms, n_total, number_of_clusters, prng, precision_goal = parameters
        result = {'N': n_total, 'R': number_of_clusters,
                  'C': number_of_clusters, 'precision_goal': precision_goal}

        true_partition = SyntheticBenchmark._generate_random_partition(
            n_total, parts=number_of_clusters, seed=prng)
        predict_partition = SyntheticBenchmark._generate_random_partition(
            n_total, parts=number_of_clusters, seed=prng)

        # For the EMI, the concrete labels should not make a difference
        labels_true = SyntheticBenchmark._get_random_clustering(
            true_partition, seed=prng, shuffle=False)
        labels_pred = SyntheticBenchmark._get_random_clustering(
            predict_partition, seed=prng, shuffle=False)

        contingency = contingency_matrix(labels_true, labels_pred)
        nrowt = np.ravel(contingency.sum(axis=1))
        ncolt = np.ravel(contingency.sum(axis=0))

        for name, algorithm in algorithms:
            if isbuiltin(algorithm):
                start = perf_counter()
                emi = algorithm(contingency, n_total)
                end = perf_counter()
                result[f'{name}_time'] = end - start
                result[f'{name}_emi'] = emi
            elif ('seed' in signature(algorithm).parameters.keys()):
                start = perf_counter()
                emi, emi_err = algorithm(
                    nrowt, ncolt, seed=prng, precision_goal=precision_goal)
                end = perf_counter()
                result[f'{name}_time'] = end - start
                result[f'{name}_emi'] = emi
                result[f'{name}_emi_err'] = emi_err
            else:
                start = perf_counter()
                emi = algorithm(contingency)
                end = perf_counter()
                result[f'{name}_time'] = end - start
                result[f'{name}_emi'] = emi

        return result

    @staticmethod
    def _random_smi(parameters) -> Dict:
        """Benchmark the SMI algorithms on a single random pair.

        Args:
            parameters: tuple with algorithms, N, the number of clusters (R=C), prng, precision_goal.

        Returns:
            A dictionary with the results.
        """
        algorithms, n_total, number_of_clusters, prng, precision_goal = parameters
        result = {'N': n_total, 'R': number_of_clusters,
                  'C': number_of_clusters, 'precision_goal': precision_goal}

        true_partition = SyntheticBenchmark._generate_random_partition(
            n_total, parts=number_of_clusters, seed=prng)
        predict_partition = SyntheticBenchmark._generate_random_partition(
            n_total, parts=number_of_clusters, seed=prng)

        labels_true = SyntheticBenchmark._get_random_clustering(
            true_partition, seed=prng)
        labels_pred = SyntheticBenchmark._get_random_clustering(
            predict_partition, seed=prng)

        for name, algorithm in algorithms:
            if 'seed' in signature(algorithm).parameters.keys():
                start = perf_counter()
                smi, smi_err = algorithm(
                    labels_true, labels_pred, seed=prng, precision_goal=precision_goal)
                end = perf_counter()
                result[f'{name}_time'] = end - start
                result[f'{name}_smi'] = smi
                result[f'{name}_smi_err'] = smi_err
            else:
                start = perf_counter()
                smi = algorithm(labels_true, labels_pred)
                end = perf_counter()
                result[f'{name}_time'] = end - start
                result[f'{name}_emi'] = smi

        return result

    def benchmark_emi(self, algorithms: List[Tuple[str, Callable]], n_total: int, precision_goal: float, partition_samples: int = 200) -> pd.DataFrame:
        """Benchmark the EMI algorithms.

        Args:
            algorithms: List of tuples with the name of the algorithm and the function to use.
            n_total: The number of data points N.
            precision_goal: The precision goal for the Monte Carlo algorithm.
            partition_samples: The number of partitions to sample.

        Returns:
            A pandas dataframe with the results.
        """
        parameters = [(algorithms, n_total, int(np.floor(x*n_total)), default_rng(seed=seed), precision_goal)
                      for x in np.arange(0.1, 0.91, 0.1)
                      for seed in self._seed_sequence.spawn(partition_samples)]

        parameters_n_varies = [(algorithms, n, int(np.floor(0.9*n)), default_rng(seed=seed), precision_goal)
                               for n in range(int(np.floor(0.1*n_total)), int(np.floor(0.9*n_total)) + 1, int(np.floor(0.1*n_total)))
                               for seed in self._seed_sequence.spawn(partition_samples)]
        parameters.extend(parameters_n_varies)

        with Pool(self._cpu_count) as p:
            results = list(tqdm(p.imap(SyntheticBenchmark._random_emi, parameters),
                                total=len(parameters), desc='EMI Benchmark'))
        return pd.DataFrame(results)

    def benchmark_smi(self, algorithms: List[Tuple[str, Callable]], n_total: int, precision_goal: float, partition_samples: int = 10) -> pd.DataFrame:
        """Benchmark the SMI algorithms.

        Args:
            algorithms: List of tuples with the name of the algorithm and the function to use.
            n_total: The number of data points N.
            precision_goal: The precision goal for the Monte Carlo algorithm.
            partition_samples: The number of partitions to sample.

        Returns:
            A pandas DataFrame with the results.
        """
        parameters = [(algorithms, n_total, int(np.floor(x*n_total)), default_rng(seed=seed), precision_goal)
                      for x in np.arange(0.1, 0.91, 0.1)
                      for seed in self._seed_sequence.spawn(partition_samples)]

        with Pool(self._cpu_count) as p:
            results = list(tqdm(p.imap(SyntheticBenchmark._random_smi, parameters),
                                total=len(parameters), desc='SMI Benchmark'))
        return pd.DataFrame(results)
