import numpy as np


def mse(pred, ref):
    return float(np.mean((pred - ref) ** 2))


def rel_l2(pred, ref):
    """Global relative L2 error ||pred-ref||_2 / ||ref||_2 over all entries."""
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def rel_l2_per_sample(pred, ref):
    """Relative L2 per leading-index sample (last axis = space), averaged over remaining axes
    by the caller. Shapes (..., N) -> (...)."""
    return np.linalg.norm(pred - ref, axis=-1) / np.linalg.norm(ref, axis=-1)
