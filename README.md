# Korea-Gomtang-Index
Korea-Gomtang-Index
# src/utils/rolling_score.py
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_percentile_last_winsorized(
    s: pd.Series,
    *,
    window: int = 1260,
    min_obs: int = 252,
    winsor_p: float = 0.01,
) -> pd.Series:
    """
    롤링 윈도우(기본 1260영업일)마다:
      1) 윈도우 내부에서 raw를 winsor_p ~ (1-winsor_p) 분위로 클리핑(winsorize)
      2) '마지막 값'의 percentile(0~100)을 계산

    반환: score 시리즈(0~100). min_obs 미만이면 NaN.
    """
    x = pd.to_numeric(s, errors="coerce").astype(float)

    def _pct_last(arr: np.ndarray) -> float:
        arr = arr.astype(float)
        arr = arr[np.isfinite(arr)]
        if arr.size < min_obs:
            return np.nan

        lo = np.quantile(arr, winsor_p)
        hi = np.quantile(arr, 1.0 - winsor_p)
        arr_w = np.clip(arr, lo, hi)

        last = arr_w[-1]
        return float(np.mean(arr_w <= last) * 100.0)

    return x.rolling(window=window, min_periods=min_obs).apply(_pct_last, raw=True)


def to_score_from_raw(
    raw: pd.Series,
    *,
    window: int = 1260,
    min_obs: int = 252,
    winsor_p: float = 0.01,
    invert: bool = False,
) -> pd.Series:
    """
    raw -> (winsorize + rolling percentile) -> score(0~100)
    invert=True이면 공포형 지표로 간주하여 100 - percentile 처리.
    """
    p = rolling_percentile_last_winsorized(raw, window=window, min_obs=min_obs, winsor_p=winsor_p)
    return (100.0 - p) if invert else p
