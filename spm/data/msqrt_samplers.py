import abc

import numpy as np

from spm.data.samplers import Sampler
from spm.data.samples import Samples

class MSqrtSampler(Sampler):
    @abc.abstractmethod
    def b_k_sampler(self, num_samples: int) -> Samples:
        raise NotImplementedError

    def sample(self, num_samples) -> Samples:
        b, k = self.b_k_sampler(num_samples)
        assert np.all(0 <= k < b - 1 / 2)  # We want k to be the "positive root", which we arbitrarily define as "the one less than b/2".
        a = np.mod(np.square(k), b)
        return Samples(a, b, k)