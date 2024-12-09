import numpy as np

from spm.data.msqrt import BOT


def simulate_proof(x0, x1, y) -> tuple[int, int]:
    """Simulate a proof that x0 = y^2 mod x1 if y != BOT, else a proof that x0 is not a quadratic residue mod x1.

    Args:
        x0: The integer whose MSqrt is to be proven.
        x1: The modulus.
        y: The square root of x0 modulo x1, or BOT if x0 is not a quadratic residue modulo x1.

    Returns:
        q: The query to the prover.
        a: The answer of the (canonicalized) prover.
    """
    # This can probably be vectorized with some effort, if needed.
    if y != BOT:
        # y is the "positive root", which we arbitrarily define as "the one less than x1/2".
        assert 0 <= y < (x1 - 1) // 2
        assert x0 == pow(y, 2, x1)
        return BOT, BOT
    # we cannot assert that x0 is not a quadratic residue (efficiently), without knowing the factorization of x1...
    r = np.random.randint(1, x1 - 1)
    r_squared = pow(r, 2, x1)
    # sample a random bit
    mult_x0 = np.random.randint(2)
    if mult_x0:
        return x0 * r_squared % x1, BOT
    return r_squared, r
