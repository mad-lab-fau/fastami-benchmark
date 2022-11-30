from typing import Sequence, Tuple
from numpy.random import default_rng
from sklearn.metrics.cluster import contingency_matrix, mutual_info_score
from sklearn.metrics.cluster import entropy as full_entropy
import numpy as np

from .utils import WalkerRandomSampling


def expected_mutual_information_mc(nrowt: Sequence[int], ncolt: Sequence[int], seed=None, precision_goal: float = 0.01, min_samples: int = 10_000) -> Tuple[float, float]:
    """Monte-Carlo approximation to the expected mutual information.

    The expected mutual information is calculated for fixed column and
    row totals of the contingency matrix.

    Args:
        nrowt: Array of row totals of the contingency matrix.
        ncolt: Array of column totals of the contingency matrix.
        seed: The random seed to use.
        precision_goal: Specify the target precision.
        min_samples: Specify the minimum number of MC samples to consider.

    Returns:
       A tuple of the expected mutual information and an error estimate.
    """
    # Preprocessing
    n_total = np.sum(nrowt)

    # Due to normalization of nrowt_counts and ncolt_counts and 1/N**2 after rescaling hypergeometric pdf
    normalization = len(nrowt)/n_total*len(ncolt)/n_total
    prng = default_rng(seed=seed)

    nrowt_vals, nrowt_counts = np.unique(nrowt, return_counts=True)
    ncolt_vals, ncolt_counts = np.unique(ncolt, return_counts=True)

    # Watch out! Here, we get the normalized distributions.
    nrowt_rng = WalkerRandomSampling(
        weights=nrowt_counts, keys=nrowt_vals, seed=prng)
    ncolt_rng = WalkerRandomSampling(
        weights=ncolt_counts, keys=ncolt_vals, seed=prng)

    # Sampling
    mc_samples = min_samples
    precision = precision_goal + 1
    emi_arr = []

    while precision > precision_goal:
        a_arr = nrowt_rng.random(mc_samples)
        b_arr = ncolt_rng.random(mc_samples)
        n_arr = prng.hypergeometric(
            ngood=a_arr-1, nbad=n_total-a_arr, nsample=b_arr-1) + 1

        # Watch out for potential underflows.
        emi_arr.extend((a_arr*b_arr)*np.log(n_total*n_arr/(a_arr*b_arr)))
        emi = normalization*np.mean(emi_arr)
        emi_std = normalization*np.std(emi_arr, ddof=1)
        emi_err = emi_std/np.sqrt(len(emi_arr))
        precision = emi_err/emi  # TODO: WATCH OUT FOR ZERO DIVISION ERRORS

        # Estimate samples needed to fulfill precision requirements
        mc_samples = int(
            np.ceil((emi_std/(precision_goal*emi))**2 - len(emi_arr)))
        mc_samples = max(mc_samples, min_samples)
        # Make sure that we don't overestimate too much
        mc_samples = min(mc_samples, 100*min_samples)

    return emi, emi_err


def expected_mutual_information_pairwise(contingency) -> float:
    """Calculate the expected mutual information under pairwise permutations.

    This algorithm for pairwise adjusted mutual information was developed
    by D. Lazarenko, T. Bonald (https://arxiv.org/abs/2103.12641) and the
    code was adapted from the reference implementation (https://github.com/
    denyslazarenko/Pairwise-Adjusted-Mutual-Information).

    Args:
        contingency: The contingency table.

    Returns:
        The pairwise expected mutual information.
    """
    k, l = contingency.shape
    a = contingency.sum(axis=1)
    b = contingency.sum(axis=0)
    c = contingency.ravel()
    n_samples = np.sum(a)

    # first term
    factor = c * (contingency - np.outer(a, np.ones(l)) -
                  np.outer(np.ones(k), b) + n_samples).ravel()
    entropy = np.zeros(len(c))
    entropy[c > 0] = c[c > 0] / n_samples * np.log(c[c > 0] / n_samples)
    entropy_ = np.zeros(len(c))
    entropy_[c > 1] = (c[c > 1] - 1) / n_samples * \
        np.log((c[c > 1] - 1) / n_samples)
    result = np.sum(factor * (entropy - entropy_)) / n_samples ** 2
    # second term
    factor = ((np.outer(a, np.ones(l)) - contingency) *
              (np.outer(np.ones(k), b) - contingency)).ravel()
    entropy_ = (c + 1) / n_samples * np.log((c + 1) / n_samples)
    result += np.sum(factor * (entropy - entropy_)) / n_samples ** 2

    return mutual_info_score(None, None, contingency=contingency) - result


def adjusted_mutual_info_mc(labels_true: Sequence[int], labels_pred: Sequence[int], seed=None, accuracy_goal: float = 0.01, min_samples: int = 10_000) -> Tuple[float, float]:
    """Approximate adjusted mutual information score for two clusterings.

    The ajusted mutual information score is calculated based on a Monte-Carlo
    estimate of the expected mutual information.

    Args:
        labels_true: A clustering of the data into disjoint subsets.
        labels_pred: Another clustering of the data into disjoint subsets.
        seed: The random seed to use.
        accuracy_goal: The desired accuracy of the Monte-Carlo estimate.
        min_samples: The minimum number of samples to use.

    Returns:
       A tuple of the adjusted mutual information and an error estimate.
    """
    n_total = len(labels_true)

    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)
    # Calculate the expected value for the mutual information
    nrowt = np.ravel(contingency.sum(axis=1))
    ncolt = np.ravel(contingency.sum(axis=0))

    # Calculate entropy for each labeling
    h_true, h_pred = full_entropy(labels_true), full_entropy(labels_pred)
    normalizer = 0.5*(h_true + h_pred)

    # Expected mutual information
    # Due to normalization of nrowt_counts and ncolt_counts and 1/N**2 after rescaling hypergeometric pdf
    normalization = len(nrowt)/n_total*len(ncolt)/n_total
    prng = default_rng(seed=seed)

    nrowt_vals, nrowt_counts = np.unique(nrowt, return_counts=True)
    ncolt_vals, ncolt_counts = np.unique(ncolt, return_counts=True)

    nrowt_rng = WalkerRandomSampling(
        weights=nrowt_counts, keys=nrowt_vals, seed=prng)
    ncolt_rng = WalkerRandomSampling(
        weights=ncolt_counts, keys=ncolt_vals, seed=prng)

    mc_samples = min_samples
    ami_err = accuracy_goal + 1
    emi_arr = []

    while ami_err > accuracy_goal:
        a_arr = nrowt_rng.random(mc_samples)
        b_arr = ncolt_rng.random(mc_samples)
        n_arr = prng.hypergeometric(
            ngood=a_arr-1, nbad=n_total-a_arr, nsample=b_arr-1) + 1

        # Watch out for potential underflows.
        emi_arr.extend((a_arr*b_arr)*np.log(n_total*n_arr/(a_arr*b_arr)))
        emi = normalization*np.mean(emi_arr)

        denominator = normalizer - emi
        # Avoid 0.0 / 0.0 when expectation equals maximum, i.e a perfect match.
        # normalizer should always be >= emi, but because of floating-point
        # representation, sometimes emi is slightly larger. Correct this
        # by preserving the sign.
        if denominator < 0:
            denominator = min(denominator, -np.finfo("float64").eps)
        else:
            denominator = max(denominator, np.finfo("float64").eps)

        ami = (mi - emi) / denominator

        # Avoid division by zero.
        second_denominator = max(abs(mi - emi), np.finfo("float64").eps)

        ami_std = normalization*np.std(emi_arr, ddof=1) * abs(ami) * abs(
            normalizer - mi)/(abs(denominator)*second_denominator)
        ami_err = ami_std/np.sqrt(len(emi_arr))

        mc_samples = int(np.ceil((ami_std/accuracy_goal)**2 - len(emi_arr)))

        mc_samples = max(mc_samples, min_samples)
        # Make sure that we don't overestimate too much
        mc_samples = min(mc_samples, 100*min_samples)

    return ami, ami_err


def adjusted_mutual_info_pairwise(labels_true: Sequence[int], labels_pred: Sequence[int]) -> float:
    """Pairwise adjusted mutual information score for two clusterings.

    This algorithm for pairwise adjusted mutual information was developed
    by D. Lazarenko, T. Bonald (https://arxiv.org/abs/2103.12641) and the
    code was adapted from their reference implementation (https://github.com/
    denyslazarenko/Pairwise-Adjusted-Mutual-Information).

    Args:
        labels_true: A clustering of the data into disjoint subsets.
        labels_pred: Another clustering of the data into disjoint subsets.

    Returns:
       The pairwise adjusted mutual information.
    """
    contingency = contingency_matrix(labels_true, labels_pred, sparse=False)
    k, l = contingency.shape
    a = contingency.sum(axis=1)
    b = contingency.sum(axis=0)
    c = contingency.ravel()
    n_samples = np.sum(a)

    # First term
    factor = c * (contingency - np.outer(a, np.ones(l)) -
                  np.outer(np.ones(k), b) + n_samples).ravel()
    entropy = np.zeros(len(c))
    entropy[c > 0] = c[c > 0] / n_samples * np.log(c[c > 0] / n_samples)
    entropy_ = np.zeros(len(c))
    entropy_[c > 1] = (c[c > 1] - 1) / n_samples * \
        np.log((c[c > 1] - 1) / n_samples)
    result = np.sum(factor * (entropy - entropy_)) / n_samples ** 2
    # Second term
    factor = ((np.outer(a, np.ones(l)) - contingency) *
              (np.outer(np.ones(k), b) - contingency)).ravel()
    entropy_ = (c + 1) / n_samples * np.log((c + 1) / n_samples)
    result += np.sum(factor * (entropy - entropy_)) / n_samples ** 2

    # Normalization
    emi = mutual_info_score(labels_true, labels_pred) - result
    h_true, h_pred = full_entropy(labels_true), full_entropy(labels_pred)
    normalizer = 0.5*(h_true + h_pred)

    return result/(normalizer - emi)
