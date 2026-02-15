# src/utils/rolling_score.py
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_series(
    s: pd.Series,
    p: float = 0.01,
) -> pd.Series:
    """
    전체 시계열 기준 winsorize(클리핑). (디버그/보조용)
    """
    s = pd.to_numeric(s, errors="coerce")
    lo = s.quantile(p)
    hi = s.quantile(1.0 - p)
    return s.clip(lower=lo, upper=hi)


def rolling_percentile_last_winsorized(
    s: pd.Series,
    window: int = 1260,
    min_obs: int = 252,
    winsor_p: float = 0.01,
) -> pd.Series:
    """
    롤링 윈도우(기본 1260영업일)마다:
      1) raw를 윈도우 내부 분위수(winsor_p, 1-winsor_p)로 클리핑(winsorize)
      2) '마지막 값'이 윈도우 내에서 몇 percentile인지(0~100) 산출

    반환: score 시리즈(0~100), min_obs 미만이면 NaN.
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
        # percentile: (<= last)의 비율을 0~100으로
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
    raw -> winsorized rolling percentile score(0~100)
    invert=True이면 공포형(값이 클수록 공포) 지표로 간주하여 100 - percentile 처리.
    """
    p = rolling_percentile_last_winsorized(raw, window=window, min_obs=min_obs, winsor_p=winsor_p)
    if invert:
        return 100.0 - p
    return p
