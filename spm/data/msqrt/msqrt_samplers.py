import numpy as np

from spm.data.msqrt.simulate_proof import simulate_proof
from spm.data.transcript_batch import TranscriptBatch, Sampler
from spm.data.msqrt import is_quadratic_residue, is_prime, BOT


class FixedX1LogUniformX0:
    def __init__(self, x1: int, base: int = 10):
        assert is_prime(x1), "For now, only prime x1 is supported"
        self.x1 = x1
        self.base = base
        self.log_ubound = np.emath.logn(self.base, x1 - 1)

    def sample_x_y(self, num_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_x0 = np.random.uniform(0, self.log_ubound, num_samples)
        x0 = np.power(self.base, log_x0)
        y = []
        for i, x0_ in enumerate(x0):
            if not is_quadratic_residue(x0_, self.x1):
                y.append(BOT)
                continue
            y_ = np.random.randint(0, (self.x1 - 1) // 2)  # Sample a random "positive square root"
            x0[i] = pow(y_, 2, self.x1)  # Replace x0 with the square of this square root
            y.append(y_)
        return x0, np.full(shape=(num_samples,), fill_value=self.x1), np.array(y)


class MSqrtTranscriptSampler(Sampler):
    def __init__(self, sampler):
        self.inner_sampler = sampler

    def sample(self, num_samples) -> TranscriptBatch:
        x0, x1, y = self.inner_sampler(num_samples)
        q, a = [], []
        for x0_, x1_, y_ in zip(x0, x1, y):
            q_, a_ = simulate_proof(x0_, x1_, y_)
            q.append(q_)
            a.append(a_)
        # stack q, a to be of shape (num_samples)
        q = np.stack(q, axis=0)
        a = np.stack(a, axis=0)
        # stack x0, x1 to be of shape (2, num_samples)
        x = np.stack((x0, x1), axis=0)
        # first query round is the input, first answer round is the output
        q = (x, q)
        a = (y, a)
        return TranscriptBatch(q, a, None)
