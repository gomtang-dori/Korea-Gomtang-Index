# src/factors/f02_strength.py
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
    ohlcv_path = Path(os.environ.get("K200_OHLCV_CACHE_PATH", "data/cache/k200_ohlcv.parquet"))
    out_path = Path(os.environ.get("F02_PATH", "data/factors/f02.parquet"))

    window = int(os.environ.get("ROLLING_DAYS", str(252 * 5)))
    min_obs = int(os.environ.get("MIN_OBS", "252"))

    df = pd.read_parquet(ohlcv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "isu_cd", "high", "low"]).sort_values(["isu_cd", "date"]).reset_index(drop=True)

    # 251일(과거) 기준 max/min
    g = df.groupby("isu_cd", group_keys=False)
    df["hi_max_251"] = g["high"].shift(1).rolling(251, min_periods=200).max()
    df["lo_min_251"] = g["low"].shift(1).rolling(251, min_periods=200).min()

    df["is_nh"] = (df["high"] >= df["hi_max_251"]).astype(float)
    df["is_nl"] = (df["low"] <= df["lo_min_251"]).astype(float)

    daily = df.groupby("date").agg(
        nh=("is_nh", "sum"),
        nl=("is_nl", "sum"),
        eligible=("isu_cd", "count"),
    ).reset_index()

    daily["f02_raw"] = (daily["nh"] - daily["nl"]) / daily["eligible"].replace(0, np.nan)
    daily["f02_score"] = rolling_percentile(daily["f02_raw"], window, min_obs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily[["date", "f02_raw", "f02_score"]].to_parquet(out_path, index=False)
    print(f"[f02] OK rows={len(daily)} -> {out_path}")


if __name__ == "__main__":
    main()
