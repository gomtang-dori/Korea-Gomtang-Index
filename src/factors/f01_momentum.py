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
    k200_path = Path(os.environ.get("K200_CACHE_PATH", "data/cache/k200_close.parquet"))
    out_path = Path(os.environ.get("F01_PATH", "data/factors/f01.parquet"))
    window = int(os.environ.get("ROLLING_DAYS", str(252 * 5)))
    min_obs = int(os.environ.get("MIN_OBS", "252"))

    df = pd.read_parquet(k200_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["k200_close"] = pd.to_numeric(df["k200_close"], errors="coerce")
    df = df.dropna(subset=["date", "k200_close"]).sort_values("date").reset_index(drop=True)

    df["f01_raw"] = df["k200_close"].pct_change(20)
    df["f01_score"] = rolling_percentile(df["f01_raw"], window, min_obs)

    out = df[["date", "f01_raw", "f01_score"]].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[f01] OK rows={len(out)} -> {out_path}")


if __name__ == "__main__":
    main()
