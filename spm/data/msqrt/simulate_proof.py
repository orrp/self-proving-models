import numpy as np

from spm.data.msqrt import BOT, is_modular_sqrt


def simulate_proof(x0: np.ndarray, x1: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simulate proofs that x0 = y^2 mod x1 if y != BOT, else proofs that x0 is not a quadratic residue mod x1.

    Args:
        x0: Array of integers whose MSqrt is to be proven
        x1: Array of moduli
        y: Array of square roots of x0 modulo x1, or BOT if x0 is not a quadratic residue modulo x1

    Returns:
        q: Array of queries to the prover
        a: Array of answers from the (canonicalized) prover

    Note:
        All inputs must be numpy arrays of the same shape
    """
    # Create mask for y != BOT cases
    has_sqrt = y != BOT

    # Validate y values where sqrt exists
    positive_y = np.logical_and(0 <= y, y <= (x1 - 1) // 2)
    assert np.all(np.logical_or(~has_sqrt, positive_y)), "y values must be in [0, (x1-1)//2)"

    # Validate that y^2 = x0 mod x1 where sqrt exists
    assert is_modular_sqrt(x0[has_sqrt], x1[has_sqrt], y[has_sqrt]).all(), "y^2 must equal x0 mod x1"

    # Initialize output arrays
    q = np.full_like(x0, BOT)
    a = np.full_like(x0, BOT)

    # Handle y == BOT cases
    no_sqrt_indices = np.where(~has_sqrt)[0]
    if len(no_sqrt_indices) > 0:
        # Generate random r values for each no_sqrt case
        r = np.random.randint(1, x1[no_sqrt_indices], size=len(no_sqrt_indices))
        r_squared = np.mod(np.power(r, 2), x1[no_sqrt_indices])

        # Randomly decide whether to multiply by x0
        mult_x0 = np.random.randint(2, size=len(no_sqrt_indices))

        # Calculate queries
        q[no_sqrt_indices] = np.where(
            mult_x0,
            np.mod(x0[no_sqrt_indices] * r_squared, x1[no_sqrt_indices]),
            r_squared
        )

        # Set answers
        a[no_sqrt_indices] = np.where(mult_x0, BOT, r)

    return q, a
