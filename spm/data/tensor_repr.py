import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import numpy as np
from torch import Tensor

from spm import DATA_DIR, logging
from spm.data.samples import Labels
from spm.data.str_repr import EncodedSamples, StrRepr
from spm.utils import arr_split, log_save, np_to_torch

NO_LOSS = -1  # Token to ignore in loss, as per minGPT implementation

DTYPE = np.int16  # int16 is more than enough for encoding single digits in reasonable bases (<32k)


def block_size(samples: EncodedSamples) -> int:
    """Compute the block size needed to train on this dataset."""
    # Longest row, plus number of delimiters (one for each label), minus one because x,y are shifted
    return samples.max_len() + len(samples.all_labels) - 1


def generation_size(samples: EncodedSamples) -> int:
    """Compute the generation size needed to train on this dataset."""
    # Longest target, plus a delimiter for each target component
    return samples.max_target_len() + len(samples.target_labels)


class TargetComponent:  # Enum was being annoying so we'll just use a class with static attributes
    """The target components for the GCD problem."""

    # Value cannot include '_' because it's used to go to and from keys in the npz.
    ANNOTATED_TRANSCRIPT = "annotated"
    TRANSCRIPT = "transcript"
    OUTPUT = "output"

    @staticmethod
    def possible_values() -> list[str]:
        return [
            TargetComponent.__dict__[attr]
            for attr in TargetComponent.__dict__
            if not attr.startswith("__") and not callable(getattr(TargetComponent, attr))
        ]


class InvalidEncodingError(ValueError):
    """Raised when attempting to decode a tensor that is not a valid encoding."""

    pass


class Tokens:
    """Delimiters and signs for the GCD problem. The delimiter is named for the token that _precedes_ it.
    E.g. egcd(15,20) has k=5, u=-1, v=1; encoded in base 10 as <+>15<a><+>20<b><+>5<k><->1<u><+>1<v>
    """

    pad: int
    plus: int
    minus: int

    label_to_delim: dict[str, int]

    # input_delims: tuple[int]
    # target_delims: tuple[int]

    def __init__(self, base: int, labels: list[str]):
        self.base = base
        self.pad = base + 0
        self.plus = base + 1
        self.minus = base + 2
        self.label_to_delim = {col: base + 3 + i for i, col in enumerate(labels)}
        self.input_delims = tuple(v for k, v in self.label_to_delim.items() if k in Labels.INPUTS)
        self.target_delims = tuple(v for k, v in self.label_to_delim.items() if k not in Labels.INPUTS)

    def __len__(self):
        return len(self.label_to_delim) + 3  # +3 for pad, plus and minus

    def __getitem__(self, key: str) -> int:
        return self.label_to_delim[key]

    def __dict__(self):
        return {"base": self.base, "label_to_delim": self.label_to_delim}

    def __eq__(self, other) -> bool:
        if not isinstance(other, Tokens):
            raise NotImplementedError(f"Cannot compare Tokens to {type(other)}")
        return self.base == other.base and self.label_to_delim == other.label_to_delim

    @classmethod
    def from_dict(cls, d: dict) -> "Tokens":
        return cls(d["base"], d["label_to_delim"].keys())

    def __str__(self):
        return str(self.__dict__())


@dataclass
class Meta:
    """Metadata for a dataset of transcripts for the GCD problem.
    Contains base of representation, delimiters, sequence lengths, paths.
    """

    base: int
    input_labels: list[str]
    target_labels: list[str]
    block_size: int
    generation_size: int
    name: str
    num_samples: int
    # initialized in __post_init__
    labels: list[str] = field(init=False)
    vocab_size: int = field(init=False)
    root_path: Path = field(init=False)
    meta_path: Path = field(init=False)
    x_path: Path = field(init=False)
    y_path: Path = field(init=False)
    val_path: Path = field(init=False)
    # Need to keep track of these for saving and loading
    _post_init_fields: tuple[str, ...] = (
        "labels",
        "vocab_size",
        "root_path",
        "meta_path",
        "x_path",
        "y_path",
        "val_path",
    )

    def __post_init__(self):
        self.labels = self.input_labels + self.target_labels
        self.vocab_size = self.base + len(self.labels) + 3  # +3 for pad, plus, minus

        self.root_path = DATA_DIR / self.name
        self.meta_path = self.root_path / "meta.json"
        self.x_path = self.root_path / "x.bin"
        self.y_path = self.root_path / "y.bin"
        self.val_path = self.root_path / "eval.npz"

    def save(self):
        """Save metadata to a JSON file."""
        self.meta_path.parent.mkdir(exist_ok=True)
        d = self.__dict__.copy()
        # remove fields should not be passed to __init__ so that load() doesn't fail
        for field_name in self._post_init_fields:
            d.pop(field_name)
        d.pop("_post_init_fields")
        with open(self.meta_path, "w") as f:
            json.dump(d, f)
        log_save(self.meta_path)

    @classmethod
    def load(cls, name: str) -> "Meta":
        path = DATA_DIR / name / "meta.json"
        logging.info(f"Loading metadata from {path}")
        """Load metadata from a JSON file."""
        with open(path, "r") as f:
            loaded = json.load(f)
        return cls(**loaded)

    def delete(self):
        """Delete all files associated with this metadata."""
        if self.root_path.exists():
            shutil.rmtree(self.root_path)
            logging.info(f"Deleted dataset {self.name}")


class TensorRepr:
    """Transcripts for the GCD problem represented as tensors."""

    def __init__(self, meta: Meta):
        """Initialize the tensor representation from a metadata dictionary."""
        self.m = meta
        self.sr = StrRepr(self.m.base)
        self.toks = Tokens(self.m.base, self.m.labels)

        self.m.meta_path.parent.mkdir(exist_ok=True)

        # Python note: I prefer declaring and then checking with hassatr rather than initializing to None bc typesafety
        # If we are instantiated from_samples, that class method will set these
        self._train_samples: EncodedSamples
        self._val_samples: EncodedSamples

        # Set in load_val_batches
        self._val_inputs: dict[int, np.ndarray]
        self._val_masks: dict[int, dict[str, np.ndarray]]
        self._val_masked: dict[int, dict[str, np.ndarray]]

    @classmethod
    def from_name(cls, name: str) -> "TensorRepr":
        """Instantiate a TensorRepr from a name."""
        meta = Meta.load(name)
        return cls(meta)

    @classmethod
    def from_samples(cls, train_samples: EncodedSamples, val_samples: EncodedSamples, name: str) -> "TensorRepr":
        """Create metadata from samples and instantiate a TensorRepr from it."""
        meta = Meta(
            base=train_samples.base,
            input_labels=train_samples.input_labels,
            target_labels=train_samples.target_labels,
            block_size=max(block_size(train_samples), block_size(val_samples)),
            generation_size=generation_size(val_samples),  # only val samples matter for generation
            name=name,
            num_samples=len(train_samples),
        )
        # logging.info(f"Train/val overlap: {train_samples.raw_samples.num_overlaps(val_samples.raw_samples)}")
        tr = cls(meta)
        tr._train_samples = train_samples
        tr._val_samples = val_samples
        return tr

    @classmethod
    def from_rlvf_samples(cls, samples: EncodedSamples, model_block_size: int, acceptance_mask: str) -> "TensorRepr":
        """Create a temporary TensorRepr for RLVF transcript generation.
        Args:
            samples: EncodedSamples generated by the model
            model_block_size: block size of the model
            acceptance_mask: mask for testing whether a transcript was accepting (TRANSCRIPT or ANNOTATED_TRANSCRIPT)
                            TRANSCRIPT means that acceptance is iff the proof is correct,
                            ANNOTATED_TRANSCRIPT means that acceptance is iff the proof and its annotation are correct.
        Returns:
            TensorRepr: a temporary instance for generating X and Y for the RLVF gradient step.
        Requires:
            model_block_size >= block_size(samples). Otherwise, can't feed samples to the model.

        """
        # Similar to from_samples, but avoids having to save/load the samples.
        assert model_block_size >= block_size(samples)
        meta = Meta(
            base=samples.base,
            input_labels=samples.input_labels,
            target_labels=samples.target_labels,
            block_size=model_block_size,
            generation_size=generation_size(samples),
            name="tmp_rlvf",  # name doesn't matter because this will never be saved.
            num_samples=len(samples),
        )
        tr = cls(meta)
        # tr._train_samples is left unset because trainer will derive X,Y directly during eval.
        tr._val_samples = samples
        if acceptance_mask == TargetComponent.ANNOTATED_TRANSCRIPT:
            tr._val_inputs, tr._val_masks, tr._val_masked = tr.make_val([TargetComponent.ANNOTATED_TRANSCRIPT])
        elif acceptance_mask == TargetComponent.TRANSCRIPT:
            # We still need ANNOTATED_TRANSCRIPT for the loss mask.
            tr._val_inputs, tr._val_masks, tr._val_masked = tr.make_val(
                [TargetComponent.ANNOTATED_TRANSCRIPT, TargetComponent.TRANSCRIPT]
            )
        else:
            raise ValueError(f"Invalid acceptance mask {acceptance_mask}")
        return tr

    def __len__(self):
        return self.m.num_samples

    def save_train(self):
        """Save training data to path.x and path.y . They can then be memmaped for training.
        via get_train_batch().
        """
        assert hasattr(self, "_train_samples"), "Can only save training data if instantiated from_samples."
        self.m.save()

        x_n, y_n = [], []
        for i in range(len(self._train_samples)):
            inputs = self._train_samples.get_inputs(i)
            targets = self._train_samples.get_targets(i)
            x, y = self.sample_to_xy(inputs, targets)
            assert x.dtype == DTYPE and y.dtype == DTYPE, f"Expected {DTYPE}, got {x.dtype}, {y.dtype}"
            x_n.append(x)
            y_n.append(y)
        x = np.stack(x_n)
        y = np.stack(y_n)
        # We can't savez these because we need to memmap them in training (that's the whole point)
        x.tofile(self.m.x_path)
        log_save(self.m.x_path)
        y.tofile(self.m.y_path)
        log_save(self.m.y_path)

    def sample_to_xy(self, inputs: dict[str, str], targets: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
        """Convert a sample to x and y arrays (for training).
        Args:
            inputs (dict[str,str]): indexes component names to their values (encoded as strings).
                                    for example, {'a': '+15', 'b': '+20'}
            targets (dict[str,str]): indexes component names to their values (encoded as strings).
                                        for example, {'k': '+5', 'u': '-1', 'v': '+1'}
        Returns:
            x: array to be fed as input to a causally masked transformer
            y: target to be predicted. It's x shifted by one, with the loss masked out at the input and padding idxs.
        """
        input_arr = self.input_arr(inputs)
        target_arr = self.target_arr(targets)
        pad_length = self.m.block_size - input_arr.shape[0] - target_arr.shape[0] + 1  # +1 because block is row shifted
        pad_arr = self.make_pad_array(pad_length)
        # padding in the end, so it doesn't hint the model about the length of the solution in generation
        sample = np.concatenate([input_arr, target_arr, pad_arr], axis=0)
        assert sample.shape[0] == self.m.block_size + 1
        x = sample[:-1]
        y = sample[1:].copy()
        # we only want to predict at output locations, mask out the loss at the input locations
        # mask out the input, except for the <b> delimiter; we want the model to see it, so it knows to start generating
        y[: input_arr.shape[0] - 2] = NO_LOSS
        # mask out the padding. we check if pad_tensor is empty because in that case it y[-0:] is y[:]!
        if pad_arr.shape[0] > 0:
            y[-pad_arr.shape[0] :] = NO_LOSS
        return x, y

    def as_array(self, int_str: str, col_name: str) -> np.ndarray:
        """Encode a string (representing a signed integer) and delimiter as an ndarray."""
        if int_str[0] == self.sr.PLUS:
            sign_tok = self.toks.plus
        elif int_str[0] == self.sr.MINUS:
            sign_tok = self.toks.minus
        else:
            raise ValueError(f"Invalid integer string {int_str}: must start with {self.sr.PLUS} or {self.sr.MINUS}")
        return np.array(
            [sign_tok] + [self.sr.decode_digit(c) for c in int_str[1:]] + [self.toks[col_name]],
            dtype=DTYPE,
        )

    def cat_components(self, arrs: dict[str, str], labels) -> np.ndarray:
        """Concatenate components into a single array with delimiters specified by labels."""
        assert list(arrs.keys()) == labels, f"Expected {labels}, got {arrs.keys()}"
        return np.concatenate([self.as_array(arrs[col], col) for col in labels], axis=0)

    def input_arr(self, inputs: dict[str, str]) -> np.ndarray:
        return self.cat_components(inputs, self.m.input_labels)

    def target_arr(self, targets: dict[str, str]) -> np.ndarray:
        return self.cat_components(targets, self.m.target_labels)

    def make_pad_array(self, pad_length: int) -> np.ndarray:
        return np.array([self.toks.pad] * pad_length, dtype=DTYPE)

    def get_train_batch(self, batch_size, device, idxs=None) -> tuple[Tensor, Tensor]:
        """Get a batch of training data.
        Args:
            batch_size (int): number of samples in the batch
            device: torch device
            idxs (np.ndarray): indexes of samples to use. If None, will sample randomly.
        Returns:
            x: tensor of input sequences
            y: tensor of target sequences
        Requires:
            self.m.x_path and self.m.y_path exist and are memmappable.
            The correct way to create them is to create a TensorRepr.from_samples and call save_train().

        """
        # gptNano: recreate np.memmap every batch to avoid a memory leak
        x = np.memmap(self.m.x_path, dtype=DTYPE, mode="r", shape=(len(self), self.m.block_size))
        y = np.memmap(self.m.y_path, dtype=DTYPE, mode="r", shape=(len(self), self.m.block_size))
        if idxs is None:
            idxs = np.random.randint(low=0, high=len(self), size=batch_size)
        return np_to_torch((x[idxs]), device), np_to_torch((y[idxs]), device)

    def sample_to_val(
        self, inputs: dict[str, str], targets: dict[str, str], val_cols: list[str]
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Convert a sample to an input array, target arrays, and masks for evaluation.
        Args:
            inputs (dict[str,str]): indexes component names to their values (encoded as strings).
                                    for example, {'a': '+15', 'b': '+20'}
            targets (dict[str,str]): indexes component names to their values (encoded as strings).
                                     for example, {'k': '+5', 'u': '-1', 'v': '+1'}
            val_cols (list[str]): list of target components to evaluate.
                                  Subset of [TRANSCRIPT, OUTPUT, ANNOTATED_TRANSCRIPT]
        Returns:
            input: array to be fed as input to a causally masked transformer
            target: what we expect the model to generate (padded with zeros past the last target component)
            masks: dict of masks, one for each target component, with 1s where the loss is to be computed
        """
        inp = self.input_arr(inputs)
        target_short = self.target_arr(targets)
        target = np.zeros(self.m.generation_size, dtype=DTYPE)
        target[: len(target_short)] = target_short
        # We make masks with case analysis as a temporary solution.
        masks = {}
        for col in val_cols:
            if col == TargetComponent.ANNOTATED_TRANSCRIPT:
                # Mask out the padding (zeros) part of the target. We can do this a bit faster than make_mask
                all_mask = np.zeros(self.m.generation_size, dtype=DTYPE)
                all_mask[: len(target_short)] = 1
                masks[TargetComponent.ANNOTATED_TRANSCRIPT] = all_mask
            elif col == TargetComponent.TRANSCRIPT:
                assert (
                    Labels.u() in self.m.target_labels and Labels.v() in self.m.target_labels
                ), "transcript (u,v) specified, but u or v not in target_cols"
                masks[TargetComponent.TRANSCRIPT] = self.make_mask(target, start_col=Labels.u(), end_col=Labels.v())
                masks[TargetComponent.TRANSCRIPT] += self.make_mask(target, start_col=None, end_col=Labels.k)
            elif col == TargetComponent.OUTPUT:
                masks[TargetComponent.OUTPUT] = self.make_mask(target, start_col=None, end_col=Labels.k)
            else:
                raise ValueError(f"For now, {col} is not a valid val column.")
        return inp, target, masks

    def make_val(
        self, val_cols: list[str]
    ) -> tuple[dict[int, np.ndarray], dict[int, dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]]]:
        """
        Prepare entire dataset as batches for evaluation. No need to worry about padding
        """
        assert hasattr(self, "_val_samples"), "Can only save validation data if instantiated from_samples."
        samples = [
            self.sample_to_val(self._val_samples.get_inputs(i), self._val_samples.get_targets(i), val_cols)
            for i in range(len(self._val_samples))
        ]  # [(input, target, masks)]

        # batch samples by input length
        len_to_samples = defaultdict(list)
        for sample in samples:
            input_sample = sample[0]
            len_to_samples[len(input_sample)].append(sample)

        # stack samples by input length and split into batches of batch_size
        inputs: dict[int, np.ndarray] = {}
        masks: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
        masked: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
        for length, samples in len_to_samples.items():
            inputs[length] = np.stack([s[0] for s in samples])
            target_stacked = np.stack([s[1] for s in samples])
            for col in val_cols:
                masks[length][col] = np.stack([s[2][col] for s in samples])
                masked[length][col] = target_stacked * masks[length][col]
        return inputs, masks, masked

    def save_val(self, val_cols: list[str]):
        """
        Save validation data to path.val as npz
        """
        self.m.save()
        inputs, masks, masked = self.make_val(val_cols)
        val_data = {}
        for length in inputs:
            val_data[f"input_{length}"] = inputs[length]
            for col in masks[length].keys():  # for col in val_cols
                val_data[f"masks_{length}_{col}"] = masks[length][col]
                val_data[f"masked_{length}_{col}"] = masked[length][col]
        np.savez(self.m.val_path, **val_data)
        log_save(self.m.val_path)

    def _load_val_batches(self):
        """Load evaluation batches from npz."""
        # Hacky and confusing without the previous function. But let's not overengineer this.
        logging.info(f"Loading evaluation batches from {self.m.val_path}")
        val_data = np.load(self.m.val_path)
        inputs: dict[int, np.ndarray] = {}
        masks: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
        masked: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
        for k in val_data:
            if k.startswith("input"):
                inputs[int(k.split("_")[1])] = val_data[k]
            elif k.startswith("masks"):
                length, col = k.split("_")[1], k.split("_")[2]
                masks[int(length)][col] = val_data[k]
            elif k.startswith("masked"):
                length, col = k.split("_")[1], k.split("_")[2]
                masked[int(length)][col] = val_data[k]
            else:
                raise ValueError(f"Invalid key {k} in {self.m.val_path}")
        self._val_inputs, self._val_masks, self._val_masked = inputs, masks, masked

    def val_batches(
        self, device, batch_size
    ) -> Generator[tuple[int, Tensor, dict[str, Tensor], dict[str, Tensor]], None, None]:
        """Yield batches for evaluation.
        Yields:
            actual_length: length of the input (excluding delimiters and signs)
            inputs: tensor of input sequences
            masks: dictionary of masks, one for each target component
            masked: dictionary of target sequences, masked except for the target component.
                    it should be that gen(input[i]) * masks[i] == masked[i], where gen is a generation from the model
                                                                             (with the prompt trimmed from it)
        Requires:
            self.val_path exists and is a valid npz file.
            The correct way to create it is to call save_val().
        """
        if not hasattr(self, "_val_inputs"):
            self._load_val_batches()
        assert (
            hasattr(self, "_val_inputs") and hasattr(self, "_val_masks") and hasattr(self, "_val_masked")
        ), "val batches not loaded"
        for length in self._val_inputs:
            actual_length = length - 4
            inp, mask, maskd = (
                self._val_inputs[length],
                self._val_masks[length],
                self._val_masked[length],
            )
            # split into batches
            inp_n = arr_split(inp, batch_size)
            mk_n = {col: arr_split(mask[col], batch_size) for col in mask}
            mkd_n = {col: arr_split(maskd[col], batch_size) for col in maskd}
            for i in range(len(inp_n)):
                yield (
                    actual_length,
                    np_to_torch(inp_n[i], device),
                    {col: np_to_torch(mk_n[col][i], device) for col in mask},
                    {col: np_to_torch(mkd_n[col][i], device) for col in maskd},
                )

    def make_mask(self, arr: np.ndarray, start_col: str | None, end_col: str | None) -> np.ndarray:
        """Creates a mask that is 1 between start_col and end_col components, and 0 elsewhere."""
        # First, iterate through the elements of the tensor, and find the start and end of the target component
        as_list = arr.tolist()
        start, end = None, None
        # Handle the case where start_col or end_col are not specified
        if start_col is None:
            start = 0
        # It's not symmetric; end_col must be searched for because tensor might be padded past it
        if end_col is None:
            end_col = self.m.target_labels[-1]
        for i in range(len(as_list)):
            if start_col is not None and as_list[i] == self.toks[start_col]:
                assert start is None, f"Found two {start_col} delimiters in {as_list}"
                start = i
            if as_list[i] == self.toks[end_col]:
                assert start is not None, f"Found {end_col} delimiter before {start_col} in {as_list}"
                end = i + 1
            if start is not None and end is not None:
                break
        assert start is not None and end is not None, f"Could not find start and end for {start_col},{end_col}"
        mask = np.zeros(self.m.generation_size, dtype=DTYPE)
        mask[start:end] = 1
        return mask

    @staticmethod
    def split_at_token(array: np.ndarray, token: int) -> tuple[np.ndarray, np.ndarray]:
        """Splits array at first occurrence of a delimiter. The delimiter is included in the first part."""
        for i in range(array.shape[0]):
            if array[i] == token:
                return array[: i + 1], array[i + 1 :]
        raise InvalidEncodingError(f"Delimiter {token} not found in array {array}")

    def tok_to_sign(self, tok: int) -> str:
        """Convert a token to a positive or negative sign."""
        if tok == self.toks.plus:
            return self.sr.PLUS
        if tok == self.toks.minus:
            return self.sr.MINUS
        raise InvalidEncodingError(f"Token {tok} not recognized as a sign")

    def arr_to_int(self, array: np.ndarray) -> int:
        """
        Decode an array (encoding an integer) into an integer.
        Useful for testing.
        """
        if array[-1] not in self.toks.label_to_delim.values():
            raise InvalidEncodingError(f"{array}: delimiter {array[-1]} not recognized")
        array = array[:-1]  # remove the delimiter
        sign_str = self.tok_to_sign(array[0].item())
        array = array[1:]  # remove the sign
        abs_str = "".join(self.sr.encode_digit(tok) for tok in array)
        try:
            integer = self.sr.decode(sign_str + abs_str)
        except ValueError:
            raise InvalidEncodingError(f"sign: {sign_str}, abs: {abs_str}")
        return integer

    def _arr_to_ints(self, array: np.ndarray, delims) -> list[int]:
        ints = []
        for delim in delims:
            component, array = self.split_at_token(array, delim)
            ints.append(self.arr_to_int(component))
        return ints

    def input_arr_to_ints(self, input_array: np.ndarray) -> list[int]:
        """
        Decode an input array (encoding a and b) into a pair of integers.
        Useful for testing.
        """
        return self._arr_to_ints(input_array, self.toks.input_delims)

    def target_arr_to_ints(self, target_array: np.ndarray) -> list[int]:
        """
        Decode a target array (with components encoding self.target_cols) into a tuple of integers.
        """
        return self._arr_to_ints(target_array, self.toks.target_delims)

    def trim_padding(self, arr: np.ndarray) -> np.ndarray:
        """Trim padding from the end of an array."""
        for i in range(arr.shape[0]):
            if arr[i] == self.toks.pad:
                return arr[:i]
        return arr

    def x_y_to_ints(self, x: np.ndarray, y: np.ndarray) -> tuple[list[int], list[int]]:
        """Decode a training sample (x,y) into a, b, and the target components."""
        # Sanity check: assert that x and y are shifted versions of each other (except for NO_LOSS, which is ignored)
        for i in range(1, x.shape[0]):
            if x[i] != y[i - 1] and y[i - 1] != NO_LOSS:
                raise InvalidEncodingError(f"x and y are not shifted versions of each other: {x}, {y}")
        # reconstruct the original row
        original_row = np.concatenate([x, [y[-1]]], axis=0)
        original_row = self.trim_padding(original_row)
        last_input_delim = self.toks.input_delims[-1]
        input_arr, target_arr = self.split_at_token(original_row, last_input_delim)
        inputs = self.input_arr_to_ints(input_arr)
        targets = self.target_arr_to_ints(target_arr)
        return inputs, targets
