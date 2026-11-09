from __future__ import annotations

import platform
import random
from typing import Any

import numpy as np
import pandas as pd


def initialize_random_state(seed: int) -> dict[str, Any]:
    random.seed(seed)
    # NumPy's legacy global RNG accepts only unsigned 32-bit seeds. Preserve the
    # configured int64 seed for Python and record the normalized NumPy value.
    numpy_seed = seed % (2 ** 32)
    np.random.seed(numpy_seed)
    return {
        "seed": seed,
        "numpy_seed": numpy_seed,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
