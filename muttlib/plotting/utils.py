"""Plotting module utils"""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _base_10_tick_scaler(median_val: float) -> int:
    for i in reversed(range(1, 4)):
        if median_val / (10 ** i) >= 1:
            return i
    return 0


def _create_common_series(
    df: pd.DataFrame, ds_col: str, start_date=None, end_date=None
) -> Tuple[pd.DataFrame, np.ndarray]:  # pylint:disable=unused-argument
    """Get series data for start/end date range.

    Return filtered values and dates."""
    window = df
    if start_date:
        window = window.query(f"@start_date <= {ds_col}")
    if end_date:
        window = window.query(f"{ds_col} <= @end_date")
    window_dates = window[ds_col].dt.to_pydatetime()
    return window, window_dates
