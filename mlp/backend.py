import numpy as np
import cupy as cp


def get_backend(device):
  if device == "cpu":
    return np
  elif device == "gpu":
    return cp
  else:
    raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")


def get_array_module(x):
  if isinstance(x, cp.ndarray):
    return cp
  else:
    return np


def to_cpu(x):
  if isinstance(x, cp.ndarray):
    return cp.asnumpy(x)
