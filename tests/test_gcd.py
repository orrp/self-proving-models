import math

import numpy as np
import pytest
import torch.random
import wandb

from spm.utils import is_egcd, egcd
from spm.data.samplers import SAMPLERS, TranscriptSampler, AnnotatedTranscriptSampler, \
    ExhaustiveGCDSampler, UniformGCDSampler
from spm.data.samples import Transcript
from spm.data.str_repr import StrRepr, EncodedSamples
from spm.data.tensor_repr import TensorRepr, DTYPE
from spm.data.tensor_repr import  TargetComponent as TC
from spm.gpt.config import TrainerConfig
from spm.gpt.trainer import evaluate, Trainer
from spm.gpt.rlvf_trainer import make_rlvf_batch
from spm.systematic_models import GroundTruth, ErrsOnEvenGCD

DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"


@pytest.fixture
def hardcoded_samples():
    a_n = [12, 18, 21, 5, 7, 10, 25, 30, 14, 1024]
    b_n = [15, 24, 36, 10, 21, 15, 35, 45, 7, 512]
    ret_n = [egcd(a, b) for a, b in zip(a_n, b_n)]
    k_n, u_n, v_n = zip(*ret_n)
    a, b = np.asarray(a_n), np.asarray(b_n)
    k, u, v = np.asarray(k_n), np.asarray(u_n), np.asarray(v_n)
    samples = Transcript(a, b, k, u, v)
    return samples


@pytest.fixture
def exhaustive_samples():
    """Samples all pairs of a,b in [1, 100] x [1, 100]"""
    sampler = TranscriptSampler(ExhaustiveGCDSampler(100))
    samples = sampler(1000000)
    assert len(samples) == 4950, f"Expected 4950 samples, got {len(samples)}."
    return samples


def test_repr():
    """Test that encoding and decoding in various bases work"""
    nums = [d for d in range(11)] + [30, 256, 1000, int(10e9)]
    nums = nums + [-x for x in nums]
    bases = [2, 3, 10, 30, 256]
    for base in bases:
        sr = StrRepr(base)
        for num in nums:
            num_str = sr.encode(num)
            assert num_str[0] in [sr.PLUS, sr.MINUS]
            assert sr.decode(num_str) == num
            expected_length = math.ceil(math.log(abs(num) + 1, base)) + 1
            if num != 0:
                assert len(num_str) == expected_length


def test_egcd(hardcoded_samples):
    """Test the extended Euclidean algorithm implementation."""
    samples = hardcoded_samples
    for a, b, k, u, v in zip(samples.a, samples.b, samples.k, samples.u, samples.v):
        assert is_egcd(a, b, k, u, v)


def test_tensor_to_ints(exhaustive_samples, request):
    """Test that the tensor_to_ints functions are inverses of the int_to_tensor functions on base 30"""
    name = request.node.name
    samples = exhaustive_samples
    encoded_samples = EncodedSamples(samples, 30)
    tr = TensorRepr.from_samples(encoded_samples, encoded_samples, name)
    assert len(encoded_samples) == 4950, f"Expected 4950 samples, got {len(encoded_samples)}."
    for i in range(len(encoded_samples)):
        inputs = encoded_samples.get_inputs(i)
        targets = encoded_samples.get_targets(i)
        input_arr = tr.input_arr(inputs)
        assert tr.input_arr_to_ints(input_arr) == [samples.a[i], samples.b[i]]
        targets_arr = tr.target_arr(targets)
        assert tr.target_arr_to_ints(targets_arr) == [samples.k[i], samples.u[i], samples.v[i]]


@pytest.mark.parametrize("sampler_name", SAMPLERS.keys())
@pytest.mark.parametrize("base", [10, 30, 210])
@pytest.mark.parametrize("mode", ["correctness", "transcript", "annotated_transcript"])
def test_data_gen(sampler_name, base, mode):
    num_samples = 100
    ubound = 1000
    args = {'ubound': ubound}
    sampler = SAMPLERS[sampler_name](**args)
    if mode == "transcript":
        sampler = TranscriptSampler(sampler)
    elif mode == "annotated_transcript":
        sampler = AnnotatedTranscriptSampler(sampler, annot_len=5)

    samples = sampler(num_samples)
    assert len(samples) == num_samples
    # Check that all rows correctly encode the egcd problem
    a, b, k = samples.a, samples.b, samples.k
    if mode == "correctness":
        assert np.all(np.gcd(a, b) == k)
        return
    u, v = samples.u, samples.v
    for a, b, k, u, v in zip(a, b, k, u, v):
        assert is_egcd(a, b, k, u, v)


def test_evaluate(hardcoded_samples, request):
    """Test evaluate function."""
    name = request.node.name
    samples = hardcoded_samples
    enc_samples = EncodedSamples(samples, 10)
    tr = TensorRepr.from_samples(enc_samples, enc_samples, name)
    tr.save_val([TC.ANNOTATED_TRANSCRIPT, TC.OUTPUT])
    # Check that the ground truth model has perfect scores
    ground_truth = GroundTruth(tr)
    scores = evaluate(ground_truth, tr, device=DEVICE, batch_size=4)
    total_all_score, total_k_score = scores["total"][TC.ANNOTATED_TRANSCRIPT], scores["total"][TC.OUTPUT]
    assert total_all_score == 1
    assert total_k_score == 1

    # Errs as follows: if a is even, u=v=0. if b is even, k = 0.
    errs_on_even_gcd = ErrsOnEvenGCD(tr)
    frac_target_errs = sum(
        a % 2 == 0 or b % 2 == 0
        for a, b in zip(samples.a, samples.b)
    ) / len(samples)
    frac_k_errs = sum(b % 2 == 0 for b in samples.b) / len(samples)
    scores = evaluate(errs_on_even_gcd, tr, device=DEVICE, batch_size=4)
    total_all_score, total_k_score = scores["total"][TC.ANNOTATED_TRANSCRIPT], scores["total"][TC.OUTPUT]
    assert np.isclose(total_all_score, 1 - frac_target_errs), np.isclose(
        total_k_score, 1 - frac_k_errs
    )


def test_make_rlvf_batch(hardcoded_samples, request):
    """Test make_rlvf_batch function."""
    name = request.node.name
    samples = hardcoded_samples
    enc_samples = EncodedSamples(samples, 10)
    tr = TensorRepr.from_samples(enc_samples, enc_samples, name)
    # Check that the ground truth model has perfect scores
    ground_truth = GroundTruth(tr)
    accepted_samples = make_rlvf_batch(enc_samples, ground_truth,
                                       device="cpu",  # because we are just manually looking at the tensors
                                       temp=1, block_size=tr.m.block_size, acceptance_mask=TC.TRANSCRIPT)
    assert accepted_samples[0].shape[0] == accepted_samples[1].shape[0] == len(samples)
    for x, y in zip(accepted_samples[0], accepted_samples[1]):
        (a, b), (k, u, v) = tr.x_y_to_ints(x.numpy().astype(DTYPE), y.numpy().astype(DTYPE))
        # find the corresponding sample in samples (the order may have changed due to batching)
        i = np.where((samples.a == a) & (samples.b == b))[0][0]
        assert i is not None
        assert (a, b, k, u, v) == (samples.a[i], samples.b[i], samples.k[i], samples.u[i], samples.v[i])
    # Check that the errs_on_even_gcd model has perfect scores
    errs_on_even_gcd = ErrsOnEvenGCD(tr)
    frac_target_errs = sum(
        a % 2 == 0 or b % 2 == 0
        for a, b in zip(samples.a, samples.b)
    ) / len(samples)
    accepted_samples = make_rlvf_batch(enc_samples, errs_on_even_gcd,
                                       device="cpu",  # because we are just manually looking at the tensors
                                       temp=1, block_size=tr.m.block_size,
                                       acceptance_mask=TC.TRANSCRIPT)
    assert accepted_samples[0].shape[0] == accepted_samples[1].shape[0]
    assert np.isclose(accepted_samples[0].shape[0] / len(samples), 1 - frac_target_errs)


def overfit_helper(tr, epochs):
    config = {
        "data": tr.m.name,
        "log_interval": 1000,
        "eval_interval": 1000,
        "seed": 0,
        "learning_rate": 0.001,
        "model": "nano",
        "device": DEVICE,
        "batch_size": 4,
        "epochs": epochs,
        "warmup_iters": 0,
    }
    trainer = Trainer(TrainerConfig.from_defaults(config))
    trainer.run()

    scores = evaluate(trainer.model, trainer.data, device=DEVICE, batch_size=4)
    return scores["total"][TC.ANNOTATED_TRANSCRIPT], scores["total"][TC.OUTPUT]


def overfit_trs(base_name):
    ubound = 1000
    num_samples = 10
    sampler = UniformGCDSampler(ubound)
    k_samples = sampler(num_samples)
    uv_samples = TranscriptSampler(sampler)(num_samples)
    cot_samples = AnnotatedTranscriptSampler(sampler, annot_len=10)(num_samples)
    names = [f"{base_name}_{name}" for name in ["k", "uv", "cot"]]
    trs = []
    for samples, name in zip([k_samples, uv_samples, cot_samples], names):
        encoded_samples = EncodedSamples(samples, 10)
        tr = TensorRepr.from_samples(train_samples=encoded_samples, val_samples=encoded_samples, name=name)
        tr.save_train()
        eval_cols = [TC.ANNOTATED_TRANSCRIPT, TC.OUTPUT]
        tr.save_val(eval_cols)
        trs.append(tr)
    return trs


def test_save_load(hardcoded_samples, request):
    name = request.node.name
    samples = hardcoded_samples
    enc_samples = EncodedSamples(samples, 10)
    tr = TensorRepr.from_samples(enc_samples, enc_samples, name)
    tr.save_train()
    tr2 = TensorRepr.from_name(name)
    assert tr.m == tr2.m
    # Check that train samples are the same
    for i in range(len(samples)):
        inputs = [samples.a[i], samples.b[i]]
        targets = [samples.k[i], samples.u[i], samples.v[i]]
        x, y = tr2.get_train_batch(batch_size=1, device='cpu', idxs=[i])
        x, y = x.squeeze(0), y.squeeze(0)
        assert (inputs, targets) == tr2.x_y_to_ints(x.numpy(), y.numpy())
    # TODO check that val samples are the same


def test_overfit(request):
    """Test that a transformer can overfit to a small dataset"""
    # seed
    np.random.seed(0)
    torch.random.manual_seed(0)
    # disable wandb
    wandb.init(mode="disabled")
    # set up tensors
    k_tr, uv_tr, cot_tr = overfit_trs(base_name=request.node.name)
    # Test that model that trains on k learns k
    all_score, k_score = overfit_helper(k_tr, 250)
    assert np.isclose(all_score, 1) and np.isclose(k_score, 1)

    # Test that model that trains on kmu,v learns them
    all_score, k_score = overfit_helper(uv_tr, 250)
    assert np.isclose(all_score, 1) and np.isclose(k_score, 1)

    # Test that model that trains on longer annotation fits k but struggles with target
    all_score, k_score = overfit_helper(cot_tr, 250)
    assert all_score <= 0.7 and 0.9 <= k_score
