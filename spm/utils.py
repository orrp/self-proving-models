import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from spm import ROOT_DIR


def arr_split(arr: np.ndarray, batch_size: int):
    """Split an array into batches of size batch_size (last batch may be smaller)."""
    num_batches = (arr.shape[0] + batch_size - 1) // batch_size
    s = np.array_split(arr, num_batches)
    return [a for a in s if a.size > 0]


def np_to_torch(arr: np.ndarray, device: str) -> Tensor:
    # We cast to np.int64 for consistency with nanoGPT
    t = torch.from_numpy(arr.astype(np.int64, casting="safe"))
    if device == "cuda":
        # nanoGPT: pinned, which allows us to move it to GPU asynchronously (non_blocking=True)
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def log_save(path: Path):
    logging.info(f"Saved {path.relative_to(ROOT_DIR)} size {human_readable_size(path)}")


def human_readable_size(path, decimal_places=2):
    size = path.stat().st_size
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if abs(size) < 1024.0 or unit == "PiB":
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def is_egcd(a, b, k, u, v):
    """Check that k is the GCD of a and b and that u, v are Bezout coefficients."""
    return a * u + b * v == k and a % k == 0 and b % k == 0


def egcd(a, b):
    """Extended Euclidean algorithm."""
    u0, u1 = 1, 0
    v0, v1 = 0, 1
    while b:
        q, a, b = a // b, b, a % b
        u0, u1 = u1, u0 - q * u1
        v0, v1 = v1, v0 - q * v1
    return a, u0, v0
