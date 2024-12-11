import numpy as np

from spm.data.msqrt.simulate_proof import simulate_proof
from spm.data.transcript_batch import TranscriptBatch, Sampler, fcast
from spm.data.msqrt import is_quadratic_residue, is_prime, BOT


class FixedX1LogUniformX0:
    _x1: int
    base: int
    log_ubound: float

    def __init__(self, x1: int, base: int = 10):
        assert is_prime(x1), "For now, only prime x1 is supported"
        self._x1 = x1
        self.base = base
        self.log_ubound = np.emath.logn(self.base, x1 - 1)

    def sample_x_y(self, num_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample x0 log-uniformly from [0, x1) for the (fixed) x1.

        Args:
            num_samples: Number of samples to generate

        Note: The current sampling algorithm is somewhat inefficient:
            1. Sample num_samples many x0, without keeping track of y.
            2. For each x0:
                If x0 is a quadratic residue (via the Euler criterion):
                    Sample a random y in [0, (x1 - 1) // 2) and replace x0 with y^2 mod x1.
                Otherwise:
                    Set y to BOT.
            This is somewhat inefficient because it discards x0 that are quadratic residues, rather than just generating
            only quadratic nonresidues and then adding the square of a random y's. But it has the following benefits:
            (a) I don't know how to generate quadratic nonresidues faithfully and efficiently.
            (b) It preserves the number of quadratic residues in the total sample.
            (b) Although the quadratic residue x0's are not necessarily log-uniform, they are
                bias towards being small (as in log-uniform).
        """
        log_x0 = np.random.uniform(0, self.log_ubound, num_samples)
        x0 = fcast(np.power(self.base, log_x0))
        y = np.full(shape=(num_samples,), fill_value=BOT)
        qr_indices = np.where(is_quadratic_residue(x0, self._x1))[0]
        y[qr_indices] = np.random.randint(0, (self._x1 - 1) // 2, size=len(qr_indices))  # Sample a random positive sqrt
        x0[qr_indices] = np.mod(np.power(y[qr_indices], 2), self._x1)  # Replace x0 with the square of this square root
        return x0, np.full(shape=(num_samples,), fill_value=self._x1), y


class MSqrtTranscriptSampler(Sampler):
    def __init__(self, sampler: FixedX1LogUniformX0):
        self.inner_sampler = sampler

    def sample(self, num_samples) -> TranscriptBatch:
        # Get all samples at once
        x0, x1, y = self.inner_sampler.sample_x_y(num_samples)

        # Call vectorized simulate_proof once
        q, a = simulate_proof(x0, x1, y)

        # stack x0, x1 to be of shape (2, num_samples)
        x = np.stack((x0, x1), axis=0)

        # first query round is the input, first answer round is the output
        q = (x, q)
        a = (y, a)

        return TranscriptBatch(q, a, None)

    def __call__(self, num_samples):
        return self.sample(num_samples)
