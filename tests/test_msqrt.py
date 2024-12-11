import pytest

from spm.data.msqrt import is_prime
from spm.data.msqrt.msqrt_samplers import FixedX1LogUniformX0, MSqrtTranscriptSampler
from spm.data.transcript_batch import TranscriptBatch


@pytest.fixture
def fixed_sampler():
    return FixedX1LogUniformX0(x1=11, base=10)


@pytest.fixture
def msqrt_sampler(fixed_sampler):
    return MSqrtTranscriptSampler(fixed_sampler)


def test_msqrt_sampler_output_structure(msqrt_sampler):
    """Test that MSqrtTranscriptSampler produces correctly structured output."""
    batch_size = 10
    batch = msqrt_sampler(batch_size)

    # Verify it's a TranscriptBatch
    assert isinstance(batch, TranscriptBatch)

    # Verify query structure
    assert isinstance(batch.query, tuple)
    assert len(batch.query) == 2  # two rounds
    for query_round in batch.query:
        assert isinstance(query_round, np.ndarray)
        assert query_round.ndim == 2  # [span, batch]
        assert query_round.shape[1] == batch_size  # batch dimension

    # Verify answer structure
    assert isinstance(batch.answer, tuple)
    assert len(batch.answer) == 2  # two rounds
    for answer_round in batch.answer:
        assert isinstance(answer_round, np.ndarray)
        assert answer_round.ndim == 2  # [span, batch]
        assert answer_round.shape[1] == batch_size  # batch dimension

    # Verify annotation structure if present
    if batch.annot is not None:
        assert isinstance(batch.annot, tuple)
        assert len(batch.annot) == 2  # two rounds
        for annot_round in batch.annot:
            assert isinstance(annot_round, np.ndarray)
            assert annot_round.ndim == 2  # [span, batch]
            assert annot_round.shape[1] == batch_size  # batch dimension


def test_msqrt_sampler_output_values(msqrt_sampler):
    """Test that sampled values are valid."""
    batch_size = 10
    batch = msqrt_sampler(batch_size)

    # First round of query should contain x values 
    first_query_round = batch.query[0]
    assert first_query_round.shape[0] == 2  # two spans for x0, x1

    # Extract x0 and x1 values
    x0_values = first_query_round[0]  # first span
    x1_values = first_query_round[1]  # second span

    # First round of answer should contain y values
    first_answer_round = batch.answer[0]
    assert first_answer_round.shape[0] == 1  # one span for y
    y_values = first_answer_round[0]  # first span

    # Verify x1 values are constant and prime
    assert all(x1_values == x1_values[0])
    assert is_prime(x1_values[0])

    # Check y values are either BOT or valid square roots
    qr = y_values != BOT
    assert np.allclose(np.mod(np.power(y_values[qr], 2), x1_values[qr]), np.mod(x0_values[qr], x1_values[qr]))


def test_fixed_sampler_output(fixed_sampler):
    """Test FixedX1LogUniformX0 output properties."""
    num_samples = 20
    x0, x1, y = fixed_sampler.sample_x_y(num_samples)

    assert len(x0) == num_samples
    assert len(x1) == num_samples
    assert len(y) == num_samples
    assert all(x1_ == fixed_sampler._x1 for x1_ in x1)
    assert all(0 <= x0_ < fixed_sampler._x1 for x0_ in x0)


def test_fixed_sampler_prime_requirement():
    """Test that FixedX1LogUniformX0 requires prime x1."""
    with pytest.raises(AssertionError):
        FixedX1LogUniformX0(x1=4)  # 4 is not prime


def test_transcript_batch_initialization():
    """Test TranscriptBatch initialization with proper array structure."""
    batch_size = 5
    # Create arrays with proper [span, batch] dimensions
    query_round1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])  # [2 spans, batch_size]
    query_round2 = np.array([[11, 12, 13, 14, 15]])  # [1 span, batch_size]

    answer_round1 = np.array([[16, 17, 18, 19, 20]])  # [1 span, batch_size]
    answer_round2 = np.array([[21, 22, 23, 24, 25]])  # [1 span, batch_size]

    query = (query_round1, query_round2)
    answer = (answer_round1, answer_round2)
    annot = None

    batch = TranscriptBatch(query, answer, annot)

    assert batch.n_rounds == 2
    assert batch.batch_size == 5
    assert all(isinstance(q, np.ndarray) for q in batch.query)
    assert all(isinstance(a, np.ndarray) for a in batch.answer)
    assert all(q.shape[1] == batch_size for q in batch.query)
    assert all(a.shape[1] == batch_size for a in batch.answer)


import numpy as np
import pytest
from spm.data.msqrt import BOT
from spm.data.msqrt.simulate_proof import simulate_proof


def test_simulate_proof_scalar():
    """Test that simulate_proof works with single-element arrays."""
    # Case 1: y != BOT (has square root)
    x0 = np.array([4])
    x1 = np.array([31])
    y = np.array([2])  # 2^2 = 4 mod 31, and 2 < (31-1)//2
    q, a = simulate_proof(x0, x1, y)
    assert q[0] == BOT and a[0] == BOT

    # Case 2: y == BOT (no square root)
    x0 = np.array([3])
    x1 = np.array([31])
    y = np.array([BOT])  # 3 is not a quadratic residue mod 31
    q, a = simulate_proof(x0, x1, y)
    if a[0] == BOT:
        # mult_x0 was 1, verify q = x0 * r^2 mod x1 for some r
        assert 0 < q[0] < x1[0]
    else:
        # mult_x0 was 0, verify q = r^2 mod x1 and a = r
        assert 0 < a[0] < x1[0]
        assert q[0] == np.mod(np.power(a[0], 2), x1[0])


def test_simulate_proof_vectorized():
    """Test that vectorized version works with array inputs."""
    # Test case with mixture of having and not having square roots
    x0 = np.array([4, 3, 9, 2, 8])
    x1 = np.array([31, 31, 31, 31, 31])
    y = np.array([2, BOT, 3, BOT, 15])  # 2^2 = 4 mod 31, 3^2 = 9 mod 31, 15^2 = 8 mod 31

    q, a = simulate_proof(x0, x1, y)

    # Check shape
    assert q.shape == x0.shape
    assert a.shape == x0.shape

    # Check cases where y != BOT
    has_sqrt = y != BOT
    assert np.all(q[has_sqrt] == BOT)
    assert np.all(a[has_sqrt] == BOT)

    # Check cases where y == BOT
    no_sqrt = ~has_sqrt
    for i in np.where(no_sqrt)[0]:
        if a[i] == BOT:
            # mult_x0 was 1
            assert 0 < q[i] < x1[i]
        else:
            # mult_x0 was 0
            assert 0 < a[i] < x1[i]
            assert q[i] == np.mod(np.power(a[i], 2), x1[i])


def test_simulate_proof_validation():
    """Test that input validation works correctly."""
    # Test invalid y value (outside range)
    x0 = np.array([4, 4])
    x1 = np.array([7, 7])
    y = np.array([4, 2])  # 4 is too large (should be < (7-1)//2 = 3)

    with pytest.raises(AssertionError, match="y values must be in"):
        simulate_proof(x0, x1, y)

    # Test invalid square (y^2 != x0 mod x1)
    x0 = np.array([4, 5])
    x1 = np.array([7, 7])
    y = np.array([2, 2])  # 2^2 = 4 != 5 mod 7

    with pytest.raises(AssertionError):  # don't check max because regex is annoying
        simulate_proof(x0, x1, y)


def test_simulate_proof_empty():
    """Test that vectorized version works with empty arrays."""
    x0 = np.array([])
    x1 = np.array([])
    y = np.array([])

    q, a = simulate_proof(x0, x1, y)
    assert len(q) == 0
    assert len(a) == 0