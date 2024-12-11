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


def is_prime(x: np.ndarray | int) -> bool:
    """
    Check if array consists only of prime numbers against a precomputed list of primes.

    Args:
        x: Array of integers.

    Returns:
        True if all elements in x are prime, False otherwise
    """
    x = np.atleast_1d(x)
    for elem in x.flatten():
        assert elem <= MAX_PRIME, f"{elem} > {MAX_PRIME}, need a longer list of primes (see generate_primes.py)"
        idx = bisect_left(PRIMES, elem)  # Binary search for the index into which elem should be inserted
        if idx == len(PRIMES) or PRIMES[idx] != elem:
            return False
    return True


def is_quadratic_residue(x: np.ndarray, p: np.ndarray | int):
    """Check if x is a quadratic residue modulo p where p is a prime.

    Args:
        x: Array of integers
        p: Array of primes or a single prime

    Returns:
        True where x is a quadratic residue modulo p, False otherwise.
    """
    assert np.all(p > 2) and is_prime(p)
    return np.isclose(a=np.mod(np.power(x, (p - 1) // 2), p), b=1)  # Euler's criterion

def is_modular_sqrt(x: np.ndarray, p: np.ndarray | int, y: np.ndarray) -> np.ndarray:
    """Check if y is a square root of x modulo p where p is a prime.

    Args:
        x: Array of integers
        p: Array of primes or a single prime
        y: Array of integers

    Returns:
        True where y is a square root of x modulo p, False otherwise.
    """
    assert np.all(p > 2) and is_prime(p)
    return np.isclose(a=np.mod(np.power(y, 2), p), b=x)
