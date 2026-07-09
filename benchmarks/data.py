"""Test data generators for benchmarks."""

from __future__ import annotations

import random

import numpy as np


def make_numeric_array(n: int, dtype: np.dtype, seed: int = 42) -> np.ndarray:
    """Generate `n` random numeric values of `dtype`.

    Integers are drawn from [-10_000, 10_000); floats are standard-normal.
    """
    rng = np.random.default_rng(seed)
    if dtype in (np.int32, np.int64):
        return rng.integers(-10_000, 10_000, size=n, dtype=dtype)
    return rng.standard_normal(n).astype(dtype)


def make_numeric_keys(n: int, dtype: np.dtype, n_unique: int, seed: int = 42) -> np.ndarray:
    """Generate `n` groupby keys of `dtype`, drawn uniformly from `n_unique` distinct values."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_unique, size=n).astype(dtype)


def make_string_list(n: int, n_categories: int = 100, seed: int = 42) -> list[str]:
    """Generate `n` strings drawn uniformly from `n_categories` categories."""
    random.seed(seed)
    categories = [f"cat_{i:04d}" for i in range(n_categories)]
    return [random.choice(categories) for _ in range(n)]
