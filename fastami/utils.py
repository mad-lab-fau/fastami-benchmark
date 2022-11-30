from numpy import array, ndarray, ones, where
from numpy.random import default_rng
import numpy as np


class Welford(object):
    """Accumulator object for Welfords online / parallel variance algorithm.

    Implementation is adapted from https://github.com/a-mitani/welford under the 
    MIT license. Some modifications have been made.

    Attributes:
        count: The number of accumulated samples.
        mean: Mean of the accumulated samples.
        var_s: Sample variance of the accumulated samples.
        var_p: Population variance of the accumulated samples.
    """

    def __init__(self, elements=None, constant=1) -> None:
        """Initialize with optional data.

        For the calculation efficiency, Welford's method is not used in the initialization process.

        Args:
            elements: data samples.
            constant: A multiplicative constant to the mean.
        """
        super(Welford, self).__init__()
        # Initialize instance attributes
        self.__count = 0
        self.__m = 0
        self.__s = 0
        self.__constant = constant

        if elements != None:
            self.add_batch(elements)

    @property
    def count(self):
        return self.__count

    @property
    def mean(self):
        return self.__m * self.__constant

    @property
    def var_s(self):
        return self.__getvars(ddof=1) * self.__constant**2

    @property
    def var_p(self):
        return self.__getvars(ddof=0) * self.__constant**2

    @property
    def err(self):
        """Standard error of the mean"""
        return np.sqrt(self.var_s/self.count)

    def add(self, element) -> None:
        """Add one data sample.

        Args:
            element: data sample.
        """

        # Welford's algorithm
        self.__count += 1
        delta = element - self.__m
        self.__m += delta / self.__count
        self.__s += delta * (element - self.__m)

    def add_batch(self, elements) -> None:
        """Add observations as batch.

        For the calculation efficiency, numpy mean and std are used.

        Args:
            elements: data samples.
        """

        count = len(elements)
        m = np.mean(elements, axis=0)
        s = np.var(elements, axis=0, ddof=0) * count
        self._merge(count, m, s)

    def merge(self, other) -> None:
        """Merge this accumulator with another one.

        Args:
            other: The other accumulator to merge.
        """

        self._merge(other.__count, other.mean/self.__constant,
                    (other.var_p*other.__count)/self.__constant**2)

    def __getvars(self, ddof) -> float:
        min_count = ddof
        if self.__count <= min_count:
            raise RuntimeError(
                f"Variance (ddof = {ddof}) is not defined for {self.__count} observations.")
        else:
            return self.__s / (self.__count - ddof)

    def _merge(self, count, m, s):
        tot_count = self.__count + count
        delta = self.__m - m
        delta2 = delta * delta
        m = (self.__count * self.__m + count * m) / tot_count
        s = self.__s + s + delta2 * \
            (self.__count * count) / tot_count

        self.__count = tot_count
        self.__m = m
        self.__s = s


class WalkerRandomSampling(object):
    """Walker's alias method for random objects with different probablities.

    Based on the implementation of Denis Bzowy at the following URL:
    http://code.activestate.com/recipes/576564-walkers-alias-method-for-random-objects-with-diffe/
    """

    def __init__(self, weights, keys=None, seed=None):
        """Builds the Walker tables ``prob`` and ``inx`` for calls to `random()`.
        The weights (a list or tuple or iterable) can be in any order and they
        do not even have to sum to 1.

        Args:
            weights: Weights of the random variates.
            keys: Keys of the random variates.
            seed: Seed for the random number generator.

        Raises:
            ValueError: If the weights do not sum to 1.
        """
        n = self.n = len(weights)
        if keys is None:
            self.keys = keys
        else:
            self.keys = array(keys)

        self._rng = default_rng(seed)

        if isinstance(weights, (list, tuple)):
            weights = array(weights, dtype=float)
        elif isinstance(weights, ndarray):
            if weights.dtype != float:
                weights = weights.astype(float)
        else:
            weights = array(list(weights), dtype=float)

        if weights.ndim != 1:
            raise ValueError("weights must be a vector")

        weights = weights * n / weights.sum()

        inx = -ones(n, dtype=int)
        short = where(weights < 1)[0].tolist()
        long = where(weights > 1)[0].tolist()
        while short and long:
            j = short.pop()
            k = long[-1]

            inx[j] = k
            weights[k] -= (1 - weights[j])
            if weights[k] < 1:
                short.append(k)
                long.pop()

        self.prob = weights
        self.inx = inx

    def random(self, count=None):
        """Returns a given number of random integers or keys, with probabilities
        being proportional to the weights supplied in the constructor.
        When `count` is ``None``, returns a single integer or key, otherwise
        returns a NumPy array with a length given in `count`.

        Args:
            count: Number of random integers or keys to return.

        Returns:
            Random variates with probabilities being proportional to the weights supplied in the constructor.
        """
        if count is None:
            u = self._rng.random()
            j = self._rng.integers(self.n)
            k = j if u <= self.prob[j] else self.inx[j]
            return self.keys[k] if self.keys is not None else k

        u = self._rng.random(size=count)
        j = self._rng.integers(self.n, size=count)
        k = where(u <= self.prob[j], j, self.inx[j])
        return self.keys[k] if self.keys is not None else k
