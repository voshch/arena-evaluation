from __future__ import annotations

import numpy as np
import polars as pl


def null_float_series(values: np.ndarray, name: str) -> pl.Series:
    """Float64 series with every NaN replaced by an explicit null."""
    return pl.Series(name, values, dtype=pl.Float64).fill_nan(None)
