# src/factors/f03_breadth.py
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
    out_path = Path(os.environ.get("F03_PATH", "data/factors/f03.parquet"))
    window = int(os.environ.get("ROLLING_DAYS", str(252 * 5)))
    min_obs = int(os.environ.get("MIN_OBS", "252"))

    df = pd.read_parquet(ohlcv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["date", "isu_cd", "close", "volume"]).sort_values(["isu_cd", "date"]).reset_index(drop=True)

    g = df.groupby("isu_cd", group_keys=False)
    df["ret1"] = g["close"].pct_change()
    df["turnover"] = df["close"] * df["volume"]

    df["up_val"] = np.where(df["ret1"] > 0, df["turnover"], 0.0)
    df["dn_val"] = np.where(df["ret1"] < 0, df["turnover"], 0.0)

    daily = df.groupby("date").agg(
        up=("up_val", "sum"),
        dn=("dn_val", "sum"),
    ).reset_index()

    denom = (daily["up"] + daily["dn"]).replace(0, np.nan)
    daily["f03_raw"] = (daily["up"] - daily["dn"]) / denom
    daily["f03_score"] = rolling_percentile(daily["f03_raw"], window, min_obs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily[["date", "f03_raw", "f03_score"]].to_parquet(out_path, index=False)
    print(f"[f03] OK rows={len(daily)} -> {out_path}")


if __name__ == "__main__":
    main()
