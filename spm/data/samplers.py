import abc
from pathlib import Path

import numpy as np

from spm import logging
from spm.data.samples import DTYPE, AnnotatedTranscript, Samples, Transcript, fcast
from spm.utils import egcd


def vectorized_egcd(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the extended GCD of two arrays of integers.
    NOTE: This is vectorized for convenience, it simply uses a for-loop.
    """
    assert a.shape == b.shape
    k, u, v = np.zeros_like(a), np.zeros_like(a), np.zeros_like(a)
    for i in range(a.shape[0]):
        k[i], u[i], v[i] = egcd(a[i].item(), b[i].item())
    # assert np.all(k == np.gcd(a, b))
    return k, u, v


class Sampler(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, num_samples) -> Samples:
        raise NotImplementedError

    def __call__(self, num_samples):
        return self.sample(num_samples)


class GCDSampler(Sampler):
    """Abstract class for samplers that sample a, b, k such that k = gcd(a, b). Convenient because
    the gcd is computed automatically once sample_a_b is implemented."""

    @abc.abstractmethod
    def sample_a_b(self, num_samples) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def sample(self, num_samples) -> Samples:
        """Sample a, b, k such that k = gcd(a, b).
        Returns:
            dict with keys 'a', 'b', 'k', each to a numpy array of shape (num_samples,).
        """
        if num_samples == 0:
            return Samples(
                np.array([], dtype=DTYPE),
                np.array([], dtype=DTYPE),
                np.array([], dtype=DTYPE),
            )
        a, b = self.sample_a_b(num_samples)
        k = np.gcd(a, b)
        return Samples(a, b, k)


class CSVSampler(GCDSampler):
    """Sampler that reads a CSV file with a, b values."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        # Check that CSV file has header a, b and is non-empty.
        with open(csv_path) as f:
            header = f.readline().strip()
            if header != "a,b":
                raise ValueError(f"CSV file should have header 'a,b', found '{header}'.")
            if not f.readline():
                raise ValueError("CSV file is empty.")

    def sample_a_b(self, num_samples) -> tuple[np.ndarray, np.ndarray]:
        """Sample the first num_samples entries from the CSV file."""
        a, b = [], []
        with open(self.csv_path) as f:
            f.readline()
            for i, line in enumerate(f):
                if num_samples is not None and i == num_samples:
                    break
                a_, b_ = map(int, line.strip().split(","))
                a.append(a_)
                b.append(b_)
        if num_samples is not None and len(a) < num_samples:
            raise ValueError(f"CSV file has only {len(a)} samples, requested {num_samples}.")
        return np.array(a, dtype=DTYPE), np.array(b, dtype=DTYPE)


class UniformGCDSampler(GCDSampler):
    """Sample two integers from 1 to ubound uniformly at random."""

    def __init__(self, ubound):
        self.ubound = ubound

    def sample_a_b(self, num_samples) -> tuple[np.ndarray, np.ndarray]:
        a, b = np.random.randint(1, self.ubound, size=(2, num_samples))
        return a, b


class LogUniformGCDSampler(GCDSampler):
    """
    Sample two integers from 1 to ubound with a log-uniform distribution.
    """

    def __init__(self, ubound, base: int = 10):
        self.base = base
        self.ubound = ubound

    def sample_a_b(self, num_samples):
        log_ubound = np.emath.logn(self.base, self.ubound)
        log_a, log_b = np.random.uniform(0, log_ubound, size=(2, num_samples))
        a, b = fcast(np.power(self.base, log_a)), fcast(np.power(self.base, log_b))
        return a, b


class ExhaustiveGCDSampler(GCDSampler):

    def __init__(self, ubound):
        self.ubound = ubound

    def sample_a_b(self, num_samples, assume_symmetry=True):
        # Sample a and b from 1 to ubound such that b < a.
        num_generated = 0
        as_: list[int] = []
        bs_: list[int] = []
        for a in range(1, self.ubound):
            for b in range(1, a + 1):
                if num_generated >= num_samples:
                    return np.array(as_), np.array(bs_)
                as_.append(a)
                bs_.append(b)
                num_generated += 1
        return np.array(as_, dtype=DTYPE), np.array(bs_, dtype=DTYPE)


class TranscriptSampler(Sampler):
    def __init__(self, sampler: Sampler):
        self.inner_sampler = sampler

    def sample(self, num_samples) -> Transcript:
        samples = self.inner_sampler(num_samples)
        a, b, k = samples.a, samples.b, samples.k
        kprime, u, v = vectorized_egcd(a, b)
        assert np.all(k == kprime)
        return Transcript(a, b, k, u, v)


class AnnotatedTranscriptSampler(TranscriptSampler):
    def __init__(self, sampler: Sampler, annot_len: int):
        super().__init__(sampler)  # self.inner_sampler = sampler
        self.cot_length = annot_len

    def sample(self, num_samples) -> AnnotatedTranscript:
        """Sample a, b, k, u, v, q such that u[-1] * a + v[-1] * b == k."""
        samples = self.inner_sampler(num_samples)
        u_chain_n, v_chain_n, q_chain_n = self.vectorized_chains(samples)
        u_chain, v_chain, q_chain = self.pad_stack(u_chain_n), self.pad_stack(v_chain_n), self.pad_stack(q_chain_n)
        u_chain, u = u_chain[:, :-1], u_chain[:, -1]
        v_chain, v = v_chain[:, :-1], v_chain[:, -1]
        q_chain = q_chain[:, 1:]  # remove the first element, which is always 0.
        return AnnotatedTranscript(samples.a, samples.b, samples.k, u_chain, v_chain, q_chain, u, v)

    def pad_stack(self, chain: list[np.ndarray]) -> np.ndarray:
        """Pad and stack a list of numpy arrays."""
        for j in range(len(chain)):
            # +1 to account for u and v, which are after the cot itself.
            pad_len = self.cot_length + 1 - len(chain[j])
            if pad_len <= 0:
                continue
            # mode = 'edge': repeat the last element of the sequence.
            chain[j] = np.pad(chain[j], (0, pad_len), mode="edge")
        return np.stack(chain, axis=0)  # shape (num_samples, self.cot_length + 1)

    def vectorized_chains(self, samples: Samples) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Returns a tuple of u and v sequences for each sample.
        Returns:
            u_n, v_n, q_n: numpy arrays with u[i][-1] * a[i] + v[i][-1] * b[i] == k[i].
        """
        u_n, v_n, q_n = [], [], []
        for i in range(len(samples)):
            a, b = samples.a[i].item(), samples.b[i].item()  # Here a and b are scalars.
            u_chain, v_chain, q_chain = self.chains(a, b)
            u_trimmed, v_trimmed, q_trimmed = self.trim(u_chain), self.trim(v_chain), self.trim(q_chain)
            assert len(u_trimmed) == len(v_trimmed) == len(q_trimmed) <= self.cot_length + 1
            assert u_trimmed[-1] * a + v_trimmed[-1] * b == samples.k[i].item()  # Bezout's identity
            u_n.append(u_trimmed)
            v_n.append(v_trimmed)
            q_n.append(q_trimmed)
        return u_n, v_n, q_n

    def trim(self, chain: list[int]) -> np.ndarray:
        """Trim the chain while ensuring the Bezout coefficient is at the end. Then put it in a numpy array."""
        chain, bezout = chain[:-1], chain[-1]
        chain = chain[: self.cot_length] + [bezout]
        return np.array(chain)

    @staticmethod
    def chains(a: int, b: int) -> tuple[list[int], list[int], list[int]]:
        """Returns an array of intermediate steps in the "optimized" extended Euclidean algorithm.
        See https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm#Pseudocode
        """
        r, s, q = [a, b], [1, 0], [0]  # init q to have a dummy element so it has the same length as r and s in the end.
        assert r[-1] != 0  # b should be non-zero, ensuring we enter the loop next.
        while r[-1] != 0:
            q.append(r[-2] // r[-1])
            r.append(r[-2] - q[-1] * r[-1])
            s.append(s[-2] - q[-1] * s[-1])
        r, s = r[:-1], s[:-1]  # remove the last element, which is trivial. now all have the same length.
        bezout_v = (r[-1] - a * s[-1]) // b  # We'll replace r[-1] with the Bezout coefficient.
        return s, r[:-1] + [bezout_v], q


class Rejector(Sampler):
    def __init__(self, sampler: Sampler):
        self.inner_sampler = sampler

    @abc.abstractmethod
    def reject(self, samples: Samples):
        """Remove samples that do not satisfy the rejection condition."""
        raise NotImplementedError

    def sample(self, num_samples) -> Samples:
        samples: Samples = self.inner_sampler(num_samples)
        self.reject(samples)
        while len(samples) < num_samples:
            logging.info(f"{type(self).__name__}: {num_samples - len(samples)} additional samples needed.")
            additional_samples = self.inner_sampler(num_samples - len(samples))
            self.reject(additional_samples)
            samples.add(additional_samples)
        return samples


class DisjointRejector(Rejector):
    """Wrapper to output samples disjoint from the exluded samples (e.g. for validation)."""

    def __init__(self, sampler: Sampler, excluded_samples: Samples):
        super().__init__(sampler)
        self.exclude_a_b = set(zip(excluded_samples.a, excluded_samples.b))

    def reject(self, samples: Samples):
        """Remove samples that are in the exclusion set."""
        a, b = samples.a, samples.b
        to_remove = []
        for i in range(len(a)):
            if (a[i], b[i]) in self.exclude_a_b:
                to_remove.append(i)
        samples.remove(to_remove)


class UpperBoundRejector(Rejector):
    def __init__(self, sampler: Sampler, rejection_ubound):
        super().__init__(sampler)
        self.rejection_ubound = rejection_ubound  # not to be confused with the ubound of the sampler!

    def reject(self, samples: Samples):
        """Remove samples where any of the components is greater than  rejection_ubound."""
        to_remove = []
        for i in range(len(samples)):
            for v in samples.to_dict().values():
                if v[i] > self.rejection_ubound:
                    to_remove.append(i)
                    break
        samples.remove(to_remove)


SAMPLERS = {
    "u": UniformGCDSampler,
    "lu": LogUniformGCDSampler,
    "e": ExhaustiveGCDSampler,
}


def euclidean_depths(
    depths: list[int], ubound=10**4, base=10, n_samples=100000, cot_length=13, rejection_ubound=210**3
):
    """
    Estimate the fraction of samples with at most a certain Euclidean depth.
    Args:
        depths: list of Euclidean depths to consider.
        ubound: upper bound on the integers whose GCD is to be computed.
        base: base in which the integers are represented.
        n_samples: number of samples to use for estimation
        cot_length: length of the Chain of Thought.
    Returns:
        list of fractions of samples with at most the corresponding Euclidean depth.
    """
    base_sampler = LogUniformGCDSampler(ubound=ubound, base=base)
    sampler = UpperBoundRejector(
        AnnotatedTranscriptSampler(base_sampler, annot_len=cot_length), rejection_ubound=rejection_ubound
    )
    np.random.seed(0)
    cot_samples: AnnotatedTranscript = sampler(n_samples)
    # Compute the actual amount of Chain of Thought needed for each sample
    max_depth = cot_samples.u_cot.shape[1] + 1
    cot_len = (
        np.ones(
            shape=(
                len(
                    cot_samples,
                )
            ),
            dtype=int,
        )
        * max_depth
    )
    for i in range(len(cot_samples)):
        for d in range(cot_samples.u_cot.shape[1]):
            if cot_samples.u_cot[i][d] == cot_samples.u[i]:
                cot_len[i] = d
                break
    # for each depth, count the fraction of samples with at most that depth
    depth_counts = [np.sum(cot_len <= d) / len(cot_len) for d in depths]
    print({d: c for d, c in zip(depths, depth_counts)})
    return depth_counts
