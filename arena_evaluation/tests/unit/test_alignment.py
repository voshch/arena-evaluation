from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("polars")


SOURCE = np.array([10, 20, 30], dtype=np.int64)
