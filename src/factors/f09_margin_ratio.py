# src/factors/f09_margin_ratio.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


def _rolling_percentile(series: pd.Series, window: int, min_obs: int) -> pd.Series:
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
    cache_path = Path(os.environ.get("F09_CACHE_PATH", "data/cache/f09_credit_deposit.parquet"))
    out_path = Path(os.environ.get("F09_OUT_PATH", "data/factors/f09.parquet"))

    rolling_days = int(os.environ.get("ROLLING_DAYS", "1260"))
    min_obs = int(os.environ.get("MIN_OBS", "252"))

    if not cache_path.exists():
        raise RuntimeError(f"[f09] Missing {cache_path}. Run cache_f09_credit_deposit_datago first.")

    df = pd.read_parquet(cache_path)
    if "date" not in df.columns or "margin_ratio_raw" not in df.columns:
        raise RuntimeError(f"[f09] missing cols in {cache_path}. cols={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["margin_ratio_raw"] = pd.to_numeric(df["margin_ratio_raw"], errors="coerce")
    df = df.dropna(subset=["date", "margin_ratio_raw"]).sort_values("date").reset_index(drop=True)

    score = _rolling_percentile(df["margin_ratio_raw"], window=rolling_days, min_obs=min_obs)

    out = pd.DataFrame({
        "date": df["date"],
        "f09_raw": df["margin_ratio_raw"],
        "f09_score": score,   # 높을수록 Greed
    }).dropna(subset=["f09_score"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[f09] OK rows={len(out)} -> {out_path}")


if __name__ == "__main__":
    main()
