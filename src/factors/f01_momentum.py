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
    # ✅ 입력: FDR이 만든 KOSPI200 종가 캐시
    k200_path = Path(os.environ.get("K200_CACHE_PATH", "data/cache/k200_close.parquet"))
    out_path = Path(os.environ.get("F01_PATH", "data/factors/f01.parquet"))

    ma_days = int(os.environ.get("F01_MA_DAYS", "125"))
    window = int(os.environ.get("ROLLING_DAYS", str(252 * 5)))
    min_obs = int(os.environ.get("MIN_OBS", "252"))

    if not k200_path.exists():
        raise FileNotFoundError(f"[f01] missing K200_CACHE_PATH: {k200_path}")

    df = pd.read_parquet(k200_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 컬럼명 방어: cache_k200_close_fdr.py가 k200_close로 저장하도록 되어있음
    if "k200_close" not in df.columns:
        raise RuntimeError(f"[f01] missing column 'k200_close' in {k200_path}. cols={list(df.columns)}")

    df["k200_close"] = pd.to_numeric(df["k200_close"], errors="coerce")
    df = df.dropna(subset=["date", "k200_close"]).sort_values("date").reset_index(drop=True)

    df["ma"] = df["k200_close"].rolling(ma_days, min_periods=ma_days).mean()
    df["f01_raw"] = df["k200_close"] / df["ma"] - 1.0
    df["f01_score"] = rolling_percentile(df["f01_raw"], window, min_obs)

    out = df[["date", "f01_raw", "f01_score"]].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[f01] OK rows={len(out)} ma_days={ma_days} -> {out_path}")


if __name__ == "__main__":
    main()
