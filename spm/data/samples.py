import numpy as np

DTYPE = np.int64


def cast(arr):
    return arr.astype(DTYPE, casting="safe")


def fcast(float_arr):
    # round to nearest integer, then cast to DTYPE. Casting='unsafe' is fine because we know the result is an integer.
    return np.round(float_arr).astype(DTYPE, casting="unsafe")


class Labels:
    """Class to store labels for "columns" of the sampler"."""

    a: str = "a"
    b: str = "b"
    k: str = "k"

    aq: str = "aq"
    bq: str = "bq"
    kq: str = "kq"
    INPUTS = [a, b, aq, bq]

    @staticmethod
    def u(i: int | None = None) -> str:
        if i is None:
            return "u"
        return f"u{i}"

    @staticmethod
    def v(i: int | None = None) -> str:
        if i is None:
            return "v"
        return f"v{i}"

    @staticmethod
    def q(i: int | None = None) -> str:
        if i is None:
            return "q"
        return f"q{i}"


class Samples:

    def __init__(self, a: np.ndarray, b: np.ndarray, k: np.ndarray):
        self.a = a
        self.b = b
        self.k = k
        self._arr_attributes = ["a", "b", "k"]

    def to_dict(self):
        return {
            Labels.a: self.a,
            Labels.b: self.b,
            Labels.k: self.k,
        }

    def __len__(self):
        return self.a.shape[0]

    def remove(self, indices: list[int]):
        for attr in self._arr_attributes:
            arr = getattr(self, attr)
            setattr(self, attr, np.delete(arr, indices, axis=0))

    def add(self, other):
        for attr in self._arr_attributes:
            arr = getattr(self, attr)
            setattr(self, attr, np.concatenate((arr, getattr(other, attr))))

    def num_overlaps(self, other):
        other_a_b_set = set(zip(other.a, other.b))
        return sum(1 for a, b in zip(self.a, self.b) if (a, b) in other_a_b_set)


class Transcript(Samples):
    def __init__(self, a: np.ndarray, b: np.ndarray, k: np.ndarray, u: np.ndarray, v: np.ndarray):
        super().__init__(a, b, k)
        self.u = u
        self.v = v
        self._arr_attributes += ["u", "v"]

    def to_dict(self):
        d = super().to_dict()
        d[Labels.u()] = self.u
        d[Labels.v()] = self.v
        return d


class AnnotatedTranscript(Transcript):
    def __init__(
        self,
        a: np.ndarray,
        b: np.ndarray,
        k: np.ndarray,
        u_cot: np.ndarray,
        v_cot: np.ndarray,
        q_cot: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
    ):
        # Check shapes
        assert len(set(arr.shape[0] for arr in (a, b, k, u_cot, v_cot, q_cot))) == 1
        assert u_cot.shape[1] == v_cot.shape[1] == q_cot.shape[1]
        super().__init__(a, b, k, u, v)
        self.u_cot = u_cot
        self.v_cot = v_cot
        self.q_cot = q_cot
        self.len_cot = self.u_cot.shape[1]
        self._arr_attributes += ["u_cot", "v_cot", "q_cot"]

    #
    def to_dict(self):
        # Code repetition for clarity: this is the order of columns (a, b, k, u0, v0, q0, ..., ud, vd, qd u, v)
        d = {
            Labels.a: self.a,
            Labels.b: self.b,
            Labels.k: self.k,
            Labels.u(0): self.u_cot[:, 0],
            Labels.v(0): self.v_cot[:, 0],
        }
        for i in range(1, self.len_cot):
            d[Labels.q(i - 1)] = self.q_cot[:, i - 1]
            d[Labels.u(i)] = self.u_cot[:, i]
            d[Labels.v(i)] = self.v_cot[:, i]
        d[Labels.u()] = self.u
        d[Labels.v()] = self.v
        return d

    def shorten_annot(self):
        self.u_cot = self.u_cot[:, :-1]
        self.v_cot = self.v_cot[:, :-1]
        self.q_cot = self.q_cot[:, :-1]
        self.len_cot -= 1
