import abc

import numpy as np
from typing import Any, Optional

from jedi.inference.gradual.typing import Callable

DTYPE = np.int64


def cast(arr) -> np.ndarray:
    return arr.astype(DTYPE, casting="safe")


def fcast(float_arr) -> np.ndarray[DTYPE]:
    # round to nearest integer, then cast to DTYPE. Casting='unsafe' is fine because we know the result is an integer.
    return np.round(float_arr).astype(DTYPE, casting="unsafe")


class TranscriptBatch:
    """
    Class to store a batch of transcripts. Transcripts consist of queries, answers and (optional) annotations.
    Each of these is a tuple of rounds, each round is a 2d-array with dims [n_span, batch_size],
    where a span refers to a meaningful sequence of tokens (to be delimited when tokenizing), and
    batch_size is the number of samples in the batch.

    Number of rounds and batch_size must be the same for all three arrays and within each round.
    n_span can be different between query/answer/annot and diff rounds, but not within a round (in each).
    """
    # num_spans can be different between query/answer/annot and diff rounds, but not within a round in each.
    query: tuple[np.ndarray]  # [round, span, batch] # Sadly array dimensionality cannot be typechecked :/
    answer: tuple[np.ndarray]  # [round, span, batch]
    annot: tuple[np.ndarray]  # [round, span, batch]

    def __init__(self, query: tuple[np.ndarray], answer: tuple[np.ndarray], annot: Optional[tuple[np.ndarray]]):
        # if annot is None, let it be a copy of the queries but with no spans (in all rounds)
        if annot is None:
            annot = tuple([np.empty((0, q.shape[-1]), dtype=q.dtype) for q in query])
        self.n_rounds = len(query)
        assert self.n_rounds == len(answer) == len(annot)  # number of rounds
        self.batch_size = query[0].shape[-1]
        self.query = self._normalize_component(query)
        self.answer = self._normalize_component(answer)
        self.annot = self._normalize_component(annot)

    def _normalize_component(self, rounds: tuple[np.ndarray]) -> tuple[np.ndarray]:
        new_rounds = []
        for round in rounds:
            assert round.shape[-1] == self.batch_size
            # Next, if the array was 1d (i.e., a single span), unsqueeze it. And cast the result to DTYPE
            if round.ndim == 1:
                round = round[np.newaxis, :]
            new_rounds.append(cast(round))
        return tuple(new_rounds)


    def __str__(self):
        return f"TranscriptBatch(query={self.query}, answer={self.answer}, annot={self.annot})"


class Sampler(abc.ABC, Callable):
    @abc.abstractmethod
    def __call__(self, num_samples: int) -> TranscriptBatch:
        raise NotImplementedError
