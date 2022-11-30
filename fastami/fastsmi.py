from typing import Sequence, Tuple
import numpy as np
from numpy.random import default_rng
from sklearn.metrics.cluster import contingency_matrix, mutual_info_score
from time import perf_counter
import ctypes
from pathlib import Path
from oct2py import Oct2Py

from .utils import Welford
from .rcont2 import rcont2

OCTAVE_SCRIPTS_PATH = Path(__file__).resolve().parent / 'octave'


class StandardizedMutualInformationSampler():
    def __init__(self, labels_true: Sequence[int], labels_pred: Sequence[int], min_samples: int = 100_000, force_precision: bool = False, seed=None) -> None:
        """Sampler for the Standardized Mutual information.

        Args:
            labels_true (int array[n_total]): A clustering of the data into
                disjoint subsets.
            labels_pred (int array[n_total]): Another clustering of the data into
                disjoint subsets.
            min_samples: Specify the minimum number of MC samples to consider.
            force_precision: If false, the algorithm will terminate before the
                relative precision is reached, if the true value is close to zero,
                to guarantee convergence (default: False).
            seed: The random seed to use.
        """
        super().__init__()

        # Parameters
        self._min_samples = min_samples
        self._max_samples = max(10_000_000, 100*min_samples)
        self._force_precision = force_precision

        # Preprocessing
        self._N = len(labels_true)

        contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
        # Calculate the MI for the two clusterings
        self._mi = mutual_info_score(labels_true, labels_pred,
                                     contingency=contingency)
        # Calculate the expected value for the mutual information
        self._a = np.ravel(contingency.sum(axis=0))
        self._b = np.ravel(contingency.sum(axis=1))

        if min(len(self._a), len(self._b)) < 2:
            raise ValueError("Clusterings should have at least 2 clusters.")

        # Get probability of encountering 1
        mask = (self._a == 1)
        self._ap1 = mask.sum()/len(self._a)
        self._a1 = self._a[~mask]

        mask = (self._b == 1)
        self._bp1 = mask.sum()/len(self._b)
        self._b1 = self._b[~mask]

        self._r = len(self._a)
        self._c = len(self._b)

        loga = np.nan_to_num(np.log(self._a), neginf=0.0)
        logb = np.nan_to_num(np.log(self._b), neginf=0.0)

        self._x = self._N * np.log(self._N) - \
            np.sum(self._a*loga) - np.sum(self._b*logb)

        self._prng = default_rng(seed=seed)

        self._E_sum_nLogn = Welford(constant=((self._r*self._c)/self._N))

        self._ij = Welford(constant=((self._r*self._c)/self._N))
        self._ipj = Welford(constant=((self._r*self._c) /
                            self._N*(self._r-1)/(self._N-1)))
        self._ijp = Welford(constant=((self._r*self._c) /
                            self._N*(self._c-1)/(self._N-1)))
        self._ipjp = Welford(constant=((self._r*self._c) /
                                       self._N*((self._r-1)*(self._c-1))/(self._N-1)))

        t0 = perf_counter()
        self._update_ij(self._min_samples)
        t1 = perf_counter()
        self._update_ipj(self._min_samples)
        t2 = perf_counter()
        self._update_ijp(self._min_samples)
        t3 = perf_counter()
        self._update_ipjp(self._min_samples)
        t4 = perf_counter()
        self._update_E_sum_nLogn(self._min_samples)
        t5 = perf_counter()

        # Timings for optimal sample distribution
        self._tij = t1 - t0
        self._tipj = t2 - t1
        self._tijp = t3 - t2
        self._tipjp = t4 - t3
        self._tE_sum_nLogn = t5 - t4

    def _random_round(self, x: float) -> int:
        """Randomly round to integer based on float remainder."""
        y = int(x)
        if self._prng.random() < x-y:
            return y + 1
        else:
            return y

    def _round_samples(self, estimated_samples: float) -> int:
        """Round an estimated number of samples to an integer."""
        if estimated_samples < self._min_samples:
            return self._min_samples
        elif estimated_samples > self._max_samples:
            return self._max_samples
        else:
            return int(np.ceil(estimated_samples))

    def _choose_pairs_without_replacement(self, data, size: int):
        """Get size random pairs from data without replacement."""
        idx_1 = self._prng.integers(len(data), size=size)
        idx_2 = self._prng.integers(len(data)-1, size=size)
        idx_2 = idx_2 + (idx_1 <= idx_2)

        return data[np.vstack((idx_1, idx_2)).T]

    def _update_E_sum_nLogn(self, mc_samples: int) -> None:
        """Update the diagonal term where i=i' and j=j'.

        Args:
            mc_samples: Number of samples that should be added to the existing calculation.
        """
        a = self._prng.choice(self._a, size=mc_samples, replace=True)
        b = self._prng.choice(self._b, size=mc_samples, replace=True)
        n = self._prng.hypergeometric(
            ngood=a-1,
            nbad=self._N-a,
            nsample=b-1
        ) + 1
        self._E_sum_nLogn.add_batch(a*b*np.log(n))

    def _update_ij(self, mc_samples: int) -> None:
        """Update the diagonal term where i=i' and j=j'.

        Args:
            mc_samples: Number of samples that should be added to the existing calculation.
        """
        a = self._prng.choice(self._a, size=mc_samples, replace=True)
        b = self._prng.choice(self._b, size=mc_samples, replace=True)
        n = self._prng.hypergeometric(
            ngood=a-1,
            nbad=self._N-a,
            nsample=b-1
        ) + 1
        self._ij.add_batch(a*b*n*np.log(n)**2)
        # self._E_sum_nLogn.add_batch(a*b*np.log(n))

    def _update_ijp(self, mc_samples: int) -> None:
        """Update the diagonal term where i=i' and j!=j'.

        Args:
            mc_samples: Number of samples that should be added to the existing calculation.
        """
        # Add zeros for the singletons
        one_samples = self._random_round(self._ap1 * mc_samples)
        if one_samples > 0:
            self._ijp._merge(one_samples, 0, 0)

        a = self._prng.choice(self._a1, size=mc_samples -
                              one_samples, replace=True)
        b = self._choose_pairs_without_replacement(
            self._b, mc_samples - one_samples)

        nij = self._prng.hypergeometric(
            b[:, 0] - 1,
            self._N - b[:, 0] - 1,
            a - 2
        )
        # If there are no more samples to be drawn
        mask = ((a - 2) != nij)
        nijp = np.zeros_like(nij)
        nijp[mask] = self._prng.hypergeometric(
            b[mask, 1] - 1,
            self._N - b[mask, 0] - b[mask, 1],
            a[mask] - 2 - nij[mask]
        )
        nij = nij + 1
        nijp = nijp + 1

        self._ijp.add_batch(
            (b[:, 0]*a)*(b[:, 1]*(a-1)) * np.log(nij) * np.log(nijp)
        )

    def _update_ipj(self, mc_samples: int) -> None:
        """Update the diagonal term where i!=i' and j=j'.

        Args:
            mc_samples: Number of samples that should be added to the existing calculation.
        """
        # Add zeros for the singletons
        one_samples = self._random_round(self._bp1 * mc_samples)
        if one_samples > 0:
            self._ipj._merge(one_samples, 0, 0)

        b = self._prng.choice(self._b1, size=mc_samples -
                              one_samples, replace=True)

        a = self._choose_pairs_without_replacement(
            self._a, mc_samples - one_samples)

        nij = self._prng.hypergeometric(
            a[:, 0] - 1,
            self._N - a[:, 0] - 1,
            b - 2
        )
        # If there are no more samples to be drawn
        mask = ((b - 2) != nij)
        nipj = np.zeros_like(nij)
        nipj[mask] = self._prng.hypergeometric(
            a[mask, 1] - 1,
            self._N - a[mask, 0] - a[mask, 1],
            b[mask] - 2 - nij[mask]
        )
        nij = nij + 1
        nipj = nipj + 1

        self._ipj.add_batch(
            (a[:, 0]*b)*(a[:, 1]*(b-1)) * np.log(nij) * np.log(nipj)
        )

    def _update_ipjp(self, mc_samples: int) -> None:
        """Update the off-diagonal term where i!=i' and j!=j'.

        Args:
            mc_samples: Number of samples that should be added to the existing calculation.
        """
        a = self._choose_pairs_without_replacement(
            self._a, mc_samples)
        b = self._choose_pairs_without_replacement(
            self._b, mc_samples)

        # Make sure that nij > 0 and nijp < bp!
        nij = self._prng.hypergeometric(
            b[:, 0]-1, self._N - b[:, 0] - 1, a[:, 0] - 1)
        # If there are no more samples to be drawn
        mask = ((a[:, 0] - 1) != nij)
        nijp = np.zeros_like(nij)
        nijp[mask] = self._prng.hypergeometric(
            b[mask, 1] - 1,
            self._N - b[mask, 0] - b[mask, 1],
            a[mask, 0] - 1 - nij[mask]
        )
        nij = nij + 1

        nipjp = self._prng.hypergeometric(
            b[:, 1] - nijp - 1,
            self._N - a[:, 0] - b[:, 1] + nijp,
            a[:, 1] - 1
        )
        nipjp = nipjp + 1
        self._ipjp.add_batch(
            a[:, 0]*b[:, 0]*a[:, 1]*b[:, 1]*np.log(nij)*np.log(nipjp)
        )

    def _get_required_samples(self, precision_goal: float) -> Tuple[int, int, int, int, int]:
        """Get number of required samples for a precision.

        The total number of required samples is minimized using the Lagrange multipliers.

        Args:
            precision_goal: Targeted relative error of the approximation.

        Returns:
            Tuple (E_sum_nLogn, ij, ijp, ipj, ipjp).
        """
        # Get the current values for frequently used terms for convenience
        m = (self._N * self._mi - self._x)
        e = self._E_sum_nLogn.mean
        e2 = self._ij.mean + self._ipj.mean + self._ijp.mean + self._ipjp.mean

        smi_squared = abs((m-e)**2/(e2-e**2))

        beta = 0.5*abs(e-m)/abs(e2-e**2)**1.5
        gamma = abs(m*e-e2)/abs(e2-e**2)**1.5

        # Dimensionless standard deviations
        sigma_e = gamma * np.sqrt(self._E_sum_nLogn.var_s)
        sigma_ij = beta * np.sqrt(self._ij.var_s)
        sigma_ipj = beta * np.sqrt(self._ipj.var_s)
        sigma_ijp = beta * np.sqrt(self._ijp.var_s)
        sigma_ipjp = beta * np.sqrt(self._ipjp.var_s)

        # Calculate the lagrange multiplier
        sqrt_lambda = (
            np.sqrt(self._tE_sum_nLogn) * sigma_e +
            np.sqrt(self._tij) * sigma_ij +
            np.sqrt(self._tipj) * sigma_ipj +
            np.sqrt(self._tijp) * sigma_ijp +
            np.sqrt(self._tipjp) * sigma_ipjp
        ) / (precision_goal**2 * smi_squared)

        # Calculate estimates for required mc_samples subtracting the samples that were already taken
        sample_estimates = [
            sigma_e*sqrt_lambda /
            np.sqrt(self._tE_sum_nLogn) - self._E_sum_nLogn.count,
            sigma_ij*sqrt_lambda /
            np.sqrt(self._tij) - self._ij.count,
            sigma_ipj*sqrt_lambda /
            np.sqrt(self._tipj) - self._ipj.count,
            sigma_ijp*sqrt_lambda /
            np.sqrt(self._tijp) - self._ijp.count,
            sigma_ipjp*sqrt_lambda /
            np.sqrt(self._tipjp) - self._ipjp.count
        ]

        # Round and return estimates
        return tuple(self._round_samples(n) for n in sample_estimates)

    @property
    def smi(self) -> Tuple[float, float]:
        """The current estimate for the SMI with an error estimate."""
        m = (self._N * self._mi - self._x)
        e = self._E_sum_nLogn.mean
        e2 = self._ij.mean + self._ipj.mean + self._ijp.mean + self._ipjp.mean

        if e2 < e**2:
            return (0.0, np.inf)

        smi = (m-e)/np.sqrt(e2-e**2)

        smi_err = np.sqrt(
            ((m*e-e2)**2/(e2 - e**2)**3)*self._E_sum_nLogn.err**2 +
            0.25*((e-m)**2/(e2-e**2)**3)*(
                self._ij.err**2 +
                self._ijp.err**2 +
                self._ipj.err**2 +
                self._ipjp.err**2
            )
        )

        return smi, smi_err

    def update(self, precision_goal: float = 0.1) -> None:
        """Update the Monte-Carlo estimate of the SMI to fulfill the required precision.

        Args:
            precision_goal: Targeted relative error of the approximation.
        """
        smi, smi_err = self.smi
        while (abs(smi_err)/max(abs(smi), 1e-14) > precision_goal) and (self._force_precision or (smi_err > precision_goal)):
            n_e, n_ij, n_ipj, n_ijp, n_ipjp = self._get_required_samples(
                precision_goal)
            self._update_E_sum_nLogn(n_e)
            self._update_ij(n_ij)
            self._update_ipj(n_ipj)
            self._update_ijp(n_ijp)
            self._update_ipjp(n_ipjp)

            smi, smi_err = self.smi


def standardized_mutual_information_separate_mc(labels_true: Sequence[int], labels_pred: Sequence[int], seed=None, precision_goal: float = 0.1, min_samples=100_000) -> Tuple[float, float]:
    """Calculate the Monte-Carlo estimate of the standardized mutual 
    information using separate samples for each term.

    Args:
        labels_true: True labels.
        labels_pred: Predicted labels.
        precision_goal: Targeted relative error of the approximation.
        seed: Random seed.
        min_samples: Minimum number of samples to use.

    Returns:
        Tuple (smi, smi_err).
    """
    sampler = StandardizedMutualInformationSampler(
        labels_true, labels_pred, seed=seed)
    sampler.update(precision_goal)
    return sampler.smi


def standardized_mutual_information_direct_mc(labels_true: Sequence[int], labels_pred: Sequence[int], seed=None, precision_goal: float = 0.1, min_samples: int = 100) -> Tuple[float, float]:
    """Calculate the Monte-Carlo estimate of the standardized mutual 
    information sampling full contingency matrices directly.

    Args:
        labels_true: True labels.
        labels_pred: Predicted labels.
        seed: Random seed.
        precision_goal: Targeted relative error of the approximation.
        min_samples: Minimum number of samples to use.

    Returns:
        Tuple (smi, smi_err).
    """
    if not isinstance(seed, int):
        seed = np.random.default_rng(seed).integers(low=-32767, high=32767)
    # Wee need a cint here such that the seed can be updated in rcont2 code.
    c_seed = ctypes.c_int(seed)

    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)
    # Calculate the expected value for the mutual information
    nrowt = np.ravel(contingency.sum(axis=1))
    ncolt = np.ravel(contingency.sum(axis=0))

    mc_samples = min_samples
    precision = precision_goal + 1

    mi_arr = []

    while precision > precision_goal:
        for _ in range(mc_samples):
            mi_arr.append(mutual_info_score(
                _, _, contingency=rcont2(nrowt, ncolt, c_seed)))

        emi = np.mean(mi_arr)
        emi_std = np.std(mi_arr, ddof=1)

        smi = (mi - emi)/emi_std

        smi_err = np.sqrt(1/len(mi_arr) + (smi*smi) /
                          (2*(len(mi_arr) - 1)))
        precision = abs(smi_err/smi)

        # Estimate samples needed to fulfill precision requirements

        s2 = smi**2
        ps2 = precision_goal**2*s2

        mc_samples = np.ceil(
            (2+s2+2*ps2+np.sqrt(-16*ps2+(2+s2+2*ps2)**2))/(4*ps2)
        ) - len(mi_arr)
        if mc_samples == np.nan:
            mc_samples = min_samples
        mc_samples = max(mc_samples, min_samples)
        # Make sure that we don't overestimate too much
        mc_samples = min(mc_samples, 100*min_samples)
        mc_samples = int(mc_samples)

    return smi, smi_err


def standardized_mutual_information_exact(labels_true: Sequence[int], labels_pred: Sequence[int]) -> float:
    """Calculate the exact standardized mutual
    information using the implementation of Romano et al.

    The implementation was taken from here:
    https://sites.google.com/site/icml2014smi/SMI.zip

    S. Romano, J. Bailey, V. Nguyen, and K. Verspoor, “Standardized Mutual Information for Clustering Comparisons: One Step Further in Adjustment for Chance,” in Proceedings of the 31st International Conference on Machine Learning, Jun. 2014, pp. 1143–1151. Accessed: Dec. 08, 2021. [Online]. Available: https://proceedings.mlr.press/v32/romano14.html

    Args:
        labels_true: True labels.
        labels_pred: Predicted labels.

    Returns:
        The exact Standardized Mutual Information.
    """
    with Oct2Py() as oc:
        oc.addpath(str(OCTAVE_SCRIPTS_PATH))
        return oc.smi(labels_true + 1, labels_pred + 1)
