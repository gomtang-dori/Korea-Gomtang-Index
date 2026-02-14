# src/factors/f01_momentum.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


def rolling_percentile(s: pd.Series, window: int, min_obs: int) -> pd.Series:
    def _pct(x):
        if len(x) < min_obs:
            return np.nan
        return float(pd.Series(x).rank(pct=True).iloc[-1] * 100.0)
    return s.rolling(window=window, min_periods=min_obs).apply(_pct, raw=False)


def main():
    level_path = Path(os.environ.get("KOSPI_LEVEL_CACHE_PATH", "data/cache/kospi_mcap_weighted_level.parquet"))
    out_path = Path(os.environ.get("F01_PATH", "data/factors/f01.parquet"))

    ma_days = int(os.environ.get("F01_MA_DAYS", "125"))
    window = int(os.environ.get("ROLLING_DAYS", str(252 * 5)))
    min_obs = int(os.environ.get("MIN_OBS", "252"))

    df = pd.read_parquet(level_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["kospi_level"] = pd.to_numeric(df["kospi_level"], errors="coerce")
    df = df.dropna(subset=["date", "kospi_level"]).sort_values("date").reset_index(drop=True)

    df["ma"] = df["kospi_level"].rolling(ma_days, min_periods=ma_days).mean()
    df["f01_raw"] = df["kospi_level"] / df["ma"] - 1.0
    df["f01_score"] = rolling_percentile(df["f01_raw"], window, min_obs)

    out = df[["date", "f01_raw", "f01_score"]].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[f01] OK rows={len(out)} ma_days={ma_days} -> {out_path}")


if __name__ == "__main__":
    main()
