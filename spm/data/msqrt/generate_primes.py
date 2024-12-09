from gmpy2 import next_prime

from spm.data.msqrt import PRIMES_PATH

MAX_PRIME = 10 ** 6


# Generate all primes up to n
def generate_primes(n):
    primes = []
    p = 2
    while p < n:
        primes.append(p)
        p = next_prime(p)
    return primes


primes = generate_primes(MAX_PRIME)
# Save to a file in the current directory with newline separated primes
with open(PRIMES_PATH, "w") as f:
    f.write("\n".join(map(str, primes)))
