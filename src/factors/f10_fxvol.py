# src/factors/f10_fxvol.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


def _rolling_percentile(series: pd.Series, window: int, min_obs: int) -> pd.Series:
    # 각 시점 t에서, 최근 window 구간에서 현재값의 percentile (0~100)
    def _pct(x: np.ndarray) -> float:
        cur = x[-1]
        if np.isnan(cur):
            return np.nan
        arr = x[~np.isnan(x)]
        if len(arr) < min_obs:
            return np.nan
        return float((arr <= cur).mean() * 100.0)

    return series.rolling(window=window, min_periods=min_obs).apply(_pct, raw=True)


def main():
    cache_path = Path(os.environ.get("USDKRW_CACHE_PATH", "data/usdkrw_level.parquet"))
    out_path = Path(os.environ.get("F10_OUT_PATH", "data/factors/f10.parquet"))

    vol_window = int(os.environ.get("VOL_WINDOW_DAYS", "20"))  # 확정: 20영업일
    rolling_days = int(os.environ.get("ROLLING_DAYS", "1260"))  # 5년
    min_obs = int(os.environ.get("MIN_OBS", "252"))            # 최소 1년

    if not cache_path.exists():
        raise RuntimeError(
            f"Missing {cache_path}. Run usdkrw_fetch.py (ECOS) first or set USDKRW_CACHE_PATH correctly."
        )

    df = pd.read_parquet(cache_path)
    if "date" not in df.columns:
        raise RuntimeError(f"[f10] Missing 'date' col in {cache_path}. cols={list(df.columns)}")

    # 컬럼 유연 처리
    price_col = "usdkrw" if "usdkrw" in df.columns else ("Close" if "Close" in df.columns else None)
    if price_col is None:
        raise RuntimeError(f"[f10] Missing usdkrw price col in {cache_path}. cols={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["date", price_col]).sort_values("date").reset_index(drop=True)

    # log return
    ret = np.log(df[price_col]).diff()

    # 20일 변동성(표준편차)
    fxvol_raw = ret.rolling(vol_window, min_periods=max(5, vol_window // 2)).std()

    # 퍼센타일(0~100): 값이 클수록 변동성↑(Fear)
    pct = _rolling_percentile(fxvol_raw, window=rolling_days, min_obs=min_obs)

    # “높을수록 Greed” 통일: Fear 지표는 뒤집기
    f10_score = 100.0 - pct

    out = pd.DataFrame({
        "date": df["date"],
        "f10_raw_fxvol": fxvol_raw,
        "f10_score": f10_score,
    }).dropna(subset=["f10_score"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[f10] OK rows={len(out)} vol_window={vol_window} -> {out_path}")


if __name__ == "__main__":
    main()
