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
    Each of these is a list (one for each round) of 2d-arrays with dims [n_span, batch_size],
    where a span refers to a meaningful sequence of tokens (to be delimited when tokenizing), and
    batch_size is the number of samples in the batch.

    Number of rounds and batch_size must be the same for all three arrays and within each round.
    n_span can be different between query/answer/annot and diff rounds, but not within a round (in each).
    """
    # num_spans can be different between query/answer/annot and diff rounds, but not within a round in each.
    query: tuple[np.ndarray]  # [round, span, batch] # Sadly array dimensionality cannot be typechecked :/
    answer: tuple[np.ndarray]  # [round, span, batch]
    annot: Optional[tuple[np.ndarray]]  # [round, span, batch]

    def __init__(self, query: tuple[np.ndarray], answer: tuple[np.ndarray], annot: Optional[tuple[np.ndarray]]):
        self.n_rounds = len(query)
        assert self.n_rounds == len(answer) == len(annot)  # number of rounds
        # check batch_size is the same for each array in each tuple
        self.batch_size = query[0].shape[-1]
        assert all(q.shape[-1] == ans.shape[-1] == self.batch_size for q, ans, ant in zip(query, answer))
        self.query = tuple([cast(q) for q in query])
        self.answer = (cast(ans) for ans in answer)
        self.annot = None if annot is None else (cast(ant) for ant in annot)


class Sampler(abc.ABC, Callable):
    @abc.abstractmethod
    def __call__(self, num_samples: int) -> TranscriptBatch:
        raise NotImplementedError
