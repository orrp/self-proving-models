import abc

import numpy as np

from spm.data.samplers import Sampler
from spm.data.samples import Samples, fcast, Transcript


class MSqrtSampler(Sampler):
    """
    Abstract class for samplers of the form a = k^2 mod b.
    """

    @abc.abstractmethod
    def b_k_sampler(self, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def sample(self, num_samples) -> Samples:
        b, k = self.b_k_sampler(num_samples)
        assert np.all(
            0 <= k < (b - 1) / 2)  # We want k to be the "positive root", which we arbitrarily define as "the one less than b/2".
        a = np.mod(np.square(k), b)
        return Samples(a, b, k)


class FixedBLogUniformKSampler(MSqrtSampler):
    def __init__(self, b: int, base: int = 10):
        assert b > 0
        self.b = b
        self.base = base
        self.log_ubound = np.emath.logn(self.base, (b - 1) / 2)

    def b_k_sampler(self, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        log_k = np.random.uniform(0, self.log_ubound, num_samples)
        k = fcast(np.power(self.base, log_k))
        return np.full(num_samples, self.b), k


