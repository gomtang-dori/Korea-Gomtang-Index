from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


def rolling_percentile_last(s: pd.Series, window: int, min_obs: int) -> pd.Series:
    # 마지막 값의 percentile만 필요하지만, rolling.apply로 유지
    def _pct(x):
        x = pd.Series(x).dropna()
        if len(x) < min_obs:
            return np.nan
        return float(x.rank(pct=True).iloc[-1] * 100.0)
    return s.rolling(window=window, min_periods=min_obs).apply(_pct, raw=False)


def main():
    ohlcv_path = Path(os.environ.get("K200_OHLCV_CACHE_PATH", "data/cache/k200_ohlcv.parquet"))
    out_path = Path(os.environ.get("F03_PATH", "data/factors/f03.parquet"))

    mode = os.environ.get("F03_MODE", "").strip().lower()  # "daily" or ""
    if mode == "daily":
        window = int(os.environ.get("ROLLING_DAYS", "60"))
        min_obs = int(os.environ.get("MIN_OBS", "20"))
    else:
        window = int(os.environ.get("ROLLING_DAYS", str(252 * 5)))
        min_obs = int(os.environ.get("MIN_OBS", "252"))

    if not ohlcv_path.exists():
        raise RuntimeError(f"Missing {ohlcv_path}. Run cache_k200_ohlcv first.")

    df = pd.read_parquet(ohlcv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["isu_cd"] = df["isu_cd"].astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["date", "isu_cd", "close", "volume"]).sort_values(["isu_cd", "date"]).reset_index(drop=True)

    # 1D return per stock
    g = df.groupby("isu_cd", group_keys=False)
    df["ret1"] = g["close"].pct_change()

    # turnover (close * volume)
    df["turnover"] = df["close"] * df["volume"]

    # ret1 NaN(첫날)은 제외하는 편이 깔끔 (초반 왜곡 방지)
    df = df.dropna(subset=["ret1", "turnover"])

    df["up_val"] = np.where(df["ret1"] > 0, df["turnover"], 0.0)
    df["dn_val"] = np.where(df["ret1"] < 0, df["turnover"], 0.0)

    daily = (
        df.groupby("date", as_index=False)
        .agg(up=("up_val", "sum"), dn=("dn_val", "sum"))
        .sort_values("date")
        .reset_index(drop=True)
    )

    denom = (daily["up"] + daily["dn"]).replace(0, np.nan)
    daily["f03_raw"] = (daily["up"] - daily["dn"]) / denom

    # score (percentile)
    daily["f03_score"] = rolling_percentile_last(daily["f03_raw"], window=window, min_obs=min_obs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily[["date", "f03_raw", "f03_score"]].to_parquet(out_path, index=False)

    n_score = int(daily["f03_score"].notna().sum())
    print(f"[f03] OK rows={len(daily)} scored={n_score} window={window} min_obs={min_obs} -> {out_path}")


if __name__ == "__main__":
    main()
