from typing import Callable

import numpy as np
import pandas as pd
from bidict import bidict
from emoji.unicode_codes import EMOJI_DATA

from spm.data.samples import Labels, Samples


class StrRepr:
    """
    A class used to represent a batch of samples (integers) as strings in a given base.
    """

    MAX_NP_REPR_BASE = 36
    # len(SINGLE_CHAR_EMOJIS) is 1386
    MAX_REPR_BASE = 1386

    MINUS = "-"
    PLUS = "+"

    SINGLE_CHAR_EMOJIS = [emoji for emoji in EMOJI_DATA.keys() if len(emoji) == 1]
    ALPHABET = bidict({i: emoji for i, emoji in enumerate(SINGLE_CHAR_EMOJIS)})

    def __init__(self, base):
        """
        Initialize the Representer with a specific base.
        """
        self.base = base
        self._vec_encode: Callable[[np.ndarray], np.ndarray] = np.vectorize(self.encode)

    def emoji_encode(self, x: int) -> str:
        """
        Encode x in the given base using emojis as digits, big endian.
        """
        assert x >= 0
        if x == 0:  # Special case, as it will not enter the while loop
            return self.ALPHABET[0]
        digits = []
        while x > 0:
            x, r = divmod(x, self.base)
            digits.append(self.ALPHABET[r])
        return "".join(reversed(digits))

    def emoji_decode(self, s: str) -> int:
        """
        Decode s in the given base using emojis as digits.
        """
        if self.base > len(self.ALPHABET):
            raise ValueError(f"Base {self.base} greater than number of symbols {len(self.ALPHABET)}")
        return sum((self.base**i) * self.ALPHABET.inverse[digit] for i, digit in enumerate(reversed(s)))

    def decode_digit(self, digit: str) -> int:
        """
        Decode a digit in the given base.
        """
        if self.base < self.MAX_NP_REPR_BASE:
            return int(digit, self.base)
        return self.ALPHABET.inverse[digit]

    def encode_digit(self, digit: int) -> str:
        """
        Encode a digit in the given base.
        """
        if self.base < self.MAX_NP_REPR_BASE:
            return np.base_repr(digit, self.base)
        return self.ALPHABET[digit]

    def encode_abs(self, x: int) -> str:
        """
        Encode the absolute value of x in the given base.
        """
        assert x >= 0
        if self.base < self.MAX_NP_REPR_BASE:
            return np.base_repr(x, self.base)
        if self.base > len(self.ALPHABET):
            raise ValueError(f"Base {self.base} greater than number of symbols {len(self.ALPHABET)}")
        return self.emoji_encode(x)

    def decode_abs(self, s: str) -> int:
        """
        Decode the absolute value of s in the given base.
        """
        if self.base < self.MAX_NP_REPR_BASE:
            return int(s, self.base)
        return self.emoji_decode(s)

    def encode(self, x: int) -> str:
        """
        Encode a signed int x as a string, in the given base.
        """
        sign = self.PLUS if x >= 0 else self.MINUS
        return sign + self.encode_abs(abs(x))

    def decode(self, s: str) -> int:
        """
        Decode a signed int s from a string, in the given base.
        """
        assert s[0] in [self.PLUS, self.MINUS]
        sign = +1 if s[0] == self.PLUS else -1
        return self.decode_abs(s[1:]) * sign

    def encode_arr(self, arr: np.ndarray) -> np.ndarray:
        """
        Encode an array of integers in the given base.
        """
        return self._vec_encode(arr)


class EncodedSamples:
    """Same as Samples, but with all values encoded as strings in a given base."""

    a: np.ndarray
    b: np.ndarray
    k: np.ndarray

    # optional
    u: np.ndarray
    v: np.ndarray
    u_cot: np.ndarray
    v_cot: np.ndarray
    a_prime: np.ndarray
    b_prime: np.ndarray
    k_prime: np.ndarray

    def __init__(self, samples: Samples, base: int):
        self.raw_samples = samples
        self.base = base
        self.strr = StrRepr(base)
        self._as_dict = {k: self.strr.encode_arr(v) for k, v in samples.to_dict().items()}
        self.input_labels = [k for k in self._as_dict.keys() if k in Labels.INPUTS]
        self.target_labels = [k for k in self._as_dict.keys() if k not in Labels.INPUTS]
        self.all_labels = list(self._as_dict.keys())
        for k, v in self._as_dict.items():
            setattr(self, k, v)

    def save(self, path: str):
        """Save to a CSV."""
        df = pd.DataFrame(self._as_dict)
        df.to_csv(path, index=False, header=True)

    def __len__(self):
        return len(self.raw_samples)

    def keys(self):
        return self._as_dict.keys()

    def items(self):
        return self._as_dict.items()

    def get_inputs(self, i) -> dict[str, str]:
        return {k: self._as_dict[k][i] for k in self.input_labels}

    def get_targets(self, i) -> dict[str, str]:
        return {k: self._as_dict[k][i] for k in self.target_labels}

    def max_len(self) -> int:
        """Return the length of the longest sample"""
        max_len = 0
        for i in range(len(self.raw_samples)):
            i_len = sum(len(v[i]) for v in self._as_dict.values())
            max_len = max(max_len, i_len)
        return max_len

    def max_target_len(self) -> int:
        """Return the length of the longest target"""
        max_len = 0
        for i in range(len(self.raw_samples)):
            i_len = sum(len(self._as_dict[k][i]) for k in self.target_labels)
            max_len = max(max_len, i_len)
        return max_len
