import abc

import numpy as np
import torch

from spm.data.samples import Labels
from spm.data.str_repr import StrRepr
from spm.data.tensor_repr import TensorRepr
from spm.utils import egcd


class SystematicModel(abc.ABC):
    """Abstract class for a systematic (not NN) model that generates solutions to the EGCD problem by
    applying a systematic algorithm ("pseudo_egcd"). Handles encoding/decoding of strings to ints, batching, etc.
    Useful for testing and comparing with NN models. When pseudo_egcd=egcd, should have perfect scores in evaluation.
    """

    def __init__(self, tr: TensorRepr):
        # NOTE: used only for encoding/decoding strings to ints! Not for retrieving the gcd itself.
        self.data = tr
        self.base = tr.m.base
        self.sr = StrRepr(self.base)

    @abc.abstractmethod
    def pseudo_egcd(self, a: int, b: int):
        raise NotImplementedError

    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) representing an input to the GCD
        problem and output the answer according to pseudo_gcd.
        """
        # move idx to cpu if needed
        orig_device = idx.device
        idx = idx.cpu()
        generated_tensor_n = []
        for input_tensor in idx:
            a, b = self.data.input_arr_to_ints(input_tensor)
            k, u, v = self.pseudo_egcd(a, b)
            k, u, v = self.sr.encode(k), self.sr.encode(u), self.sr.encode(v)
            target_arrs = [
                self.data.as_array(k, Labels.k),
                self.data.as_array(u, Labels.u()),
                self.data.as_array(v, Labels.v()),
            ]
            pad_len = max_new_tokens - sum(len(t) for t in target_arrs)
            pad_tensor = torch.full((pad_len,), self.data.toks.pad, dtype=torch.long)
            output_arr = np.concatenate([input_tensor] + target_arrs + [pad_tensor])
            generated_tensor_n.append(torch.from_numpy(output_arr))
        return torch.stack(generated_tensor_n, dim=0).to(orig_device)


class GroundTruth(SystematicModel):
    """Ground truth model for the GCD problem. Should have perfect scores in evaluation."""

    def pseudo_egcd(self, a: int, b: int):
        k, u, v = egcd(a, b)
        return k, u, v


class ErrsOnEvenGCD(SystematicModel):
    """Like the ground truth model, but errs as follows:
    If a is even, u=v=0. If b is even, k=0.
    """

    def pseudo_egcd(self, a: int, b: int):
        k, u, v = egcd(a, b)
        if a % 2 == 0:
            u, v = 0, 0
        if b % 2 == 0:
            k = 0
        return k, u, v
