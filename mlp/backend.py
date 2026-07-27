import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


def get_backend(device):
    if device == "cpu":
        return np

    elif device == "gpu":
        if cp is None:
            raise ImportError(
                "CuPy is not installed. Install CuPy to use device='gpu'."
            )
        return cp

    else:
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")


def get_array_module(x):
    if cp is not None and isinstance(x, cp.ndarray):
        return cp
    return np


def to_cpu(x):
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x
