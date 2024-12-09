import numpy as np
from pathlib import Path
from bisect import bisect_left

# read primes from primes.txt found in this directory
# Read all lines and convert to integers
PRIMES_PATH = Path(__file__).parent / "primes.txt"
with open(PRIMES_PATH) as f:
    primes = [int(line.strip()) for line in f]
PRIMES = np.array(primes, dtype=np.int64)
MAX_PRIME = PRIMES[-1]

BOT = -1  # Special value for \bot


def is_prime(x):
    """Check if x is prime using a binary search on a precomputed list of primes."""
    assert x <= MAX_PRIME, f"{x} > {MAX_PRIME}, need a longer list of primes (see generate_primes.py)"
    idx = bisect_left(PRIMES, x)  # Binary search for the index into which x should be inserted
    return idx != len(PRIMES) and PRIMES[idx] == x  # If x is not already in the list, it's not prime


def is_quadratic_residue(x: int, p: int):
    """Check if x is a quadratic residue modulo p where p is a prime.

    Args:
        x: The integer to check.
        p: The prime modulus.

    Returns:
        True if x is a quadratic residue modulo p, False otherwise.
    """
    assert p > 2 and is_prime(p)
    return pow(x, (p - 1) // 2, p) == 1  # Euler's criterion
