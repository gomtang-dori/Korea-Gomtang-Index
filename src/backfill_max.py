# src/backfill_max.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from lib.pykrx_factors import (
    factor1_momentum,
    factor2_strength,
    factor3_breadth,
    factor6_volatility,
    factor7_safe_haven,
    factor8_foreign_netbuy,
)
from lib.krx_putcall import fetch_putcall_ratio_by_date
from lib.krx_kospi_index import KRXKospiIndexAPI


@dataclass
class CFG:
    ROLLING_DAYS: int = 252 * 5
    MIN_OBS: int = 252
    DATA_DIR: str = "data"
    USDKRW_LEVEL_PATH: str = "data/usdkrw_level.parquet"
    K200_LOOKBACK_EXTRA_DAYS: int = 60  # holidays buffer in range calls
    W: dict = None

    def __post_init__(self):
        if self.W is None:
            self.W = {
                "f01_score": 0.10,
                "f02_score": 0.10,
                "f03_score": 0.10,
                "f04_score": 0.10,
                "f05_score": 0.05,
                "f06_score": 0.125,
                "f07_score": 0.10,
                "f08_score": 0.10,
                "f10_score": 0.10,
            }


cfg = CFG()


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def safe_to_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[col])
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def rolling_percentile(s: pd.Series, window: int, min_obs: int) -> pd.Series:
    def _pct(x):
        if len(x) < min_obs:
            return np.nan
        return float(pd.Series(x).rank(pct=True).iloc[-1] * 100.0)
    return s.rolling(window=window, min_periods=min_obs).apply(_pct, raw=False)


def forward_return(level: pd.Series, n: int) -> pd.Series:
    return level.shift(-n) / level - 1.0


def forward_win(level: pd.Series, n: int) -> pd.Series:
    return (forward_return(level, n) > 0).astype(float)


def renormalize_weights(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    score_cols = list(weights.keys())
    w = pd.Series(weights, dtype=float)
    avail = df[score_cols].notna().astype(float)
    w_mat = avail.mul(w, axis=1)
    w_sum = w_mat.sum(axis=1).replace(0, np.nan)
    return w_mat.div(w_sum, axis=0)


def load_existing_f05_f10(index_df: pd.DataFrame) -> pd.DataFrame:
    keep = ["date", "f05_raw", "f10_raw"]
    cols = [c for c in keep if c in index_df.columns]
    if not cols:
        return pd.DataFrame(columns=["date"])
    return index_df[cols].copy()


def main():
    ensure_dir(cfg.DATA_DIR)

    backfill_years = int(os.environ.get("BACKFILL_YEARS", "12"))
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = end - pd.DateOffset(years=backfill_years)

    start_str = pd.Timestamp(start).strftime("%Y%m%d")
    end_str = pd.Timestamp(end).strftime("%Y%m%d")

    # ---- USD/KRW level mandatory ----
    usd_path = Path(cfg.USDKRW_LEVEL_PATH)
    if not usd_path.exists():
        raise RuntimeError(f"Missing {usd_path}. Backfill workflow must run usdkrw_fetch.py first.")
    usdkrw = pd.read_parquet(usd_path)
    usdkrw = safe_to_datetime(usdkrw, "date")
    if "usdkrw" not in usdkrw.columns:
        raise RuntimeError("usdkrw_level.parquet missing 'usdkrw'")
    usdkrw["usdkrw"] = pd.to_numeric(usdkrw["usdkrw"], errors="coerce")
    usdkrw = usdkrw.dropna(subset=["date", "usdkrw"]).sort_values("date").reset_index(drop=True)

    # ---- KOSPI200 Close from KRX OpenAPI (stable) ----
    api = KRXKospiIndexAPI.from_env()
    # For backfill, call per-day across years (may take time). It's the most stable method.
    k200 = api.fetch_k200_close_range(start, end)
    k200 = safe_to_datetime(k200, "date")
    k200["k200_close"] = pd.to_numeric(k200.get("k200_close"), errors="coerce")
    k200 = k200.dropna(subset=["date", "k200_close"]).sort_values("date").reset_index(drop=True)

    if k200.empty:
        raise RuntimeError("K200 close series is empty. Check KRX_KOSPI_DD_TRD_URL / KRX_AUTH_KEY / API approval.")

    # ---- Factors ----
    f01 = factor1_momentum(k200)
    f02 = factor2_strength(start_str, end_str)
    f03 = factor3_breadth(start_str, end_str)
    f06 = factor6_volatility(k200)
    f07 = factor7_safe_haven(k200, usdkrw)
    f08 = factor8_foreign_netbuy(start_str, end_str)

    for df in [f01, f02, f03, f06, f07, f08]:
        safe_to_datetime(df, "date")

    # ---- Put/Call full window ----
    f04 = fetch_putcall_ratio_by_date(pd.to_datetime(start), pd.to_datetime(end))
    f04 = safe_to_datetime(f04, "date")

    # ---- Existing f05/f10 from existing index (optional) ----
    index_path = Path(cfg.DATA_DIR) / "index_daily.parquet"
    old = pd.read_parquet(index_path) if index_path.exists() else pd.DataFrame()
    old = safe_to_datetime(old, "date")
    f05f10 = load_existing_f05_f10(old)

    base = k200[["date", "k200_close"]].copy()
    for add in [f01, f02, f03, f04, f06, f07, f08, f05f10]:
        if add is None or add.empty:
            continue
        if "k200_close" in add.columns and "k200_close" in base.columns:
            add = add.drop(columns=["k200_close"])
        base = base.merge(add, on="date", how="left")
    base = base.sort_values("date").reset_index(drop=True)

    # ---- Derived K200 ----
    base["k200_ret_3d"] = base["k200_close"].pct_change(3)
    base["k200_ret_5d"] = base["k200_close"].pct_change(5)
    base["k200_ret_7d"] = base["k200_close"].pct_change(7)
    base["k200_fwd_10d_return"] = forward_return(base["k200_close"], 10)
    base["k200_fwd_10d_win"] = forward_win(base["k200_close"], 10)

    # ---- Scores ----
    for raw, score in [
        ("f01_raw", "f01_score"),
        ("f02_raw", "f02_score"),
        ("f03_raw", "f03_score"),
        ("f04_raw", "f04_score"),
        ("f05_raw", "f05_score"),
        ("f06_raw", "f06_score"),
        ("f07_raw", "f07_score"),
        ("f08_raw", "f08_score"),
        ("f10_raw", "f10_score"),
    ]:
        if raw in base.columns:
            base[raw] = pd.to_numeric(base[raw], errors="coerce")
            base[score] = rolling_percentile(base[raw], cfg.ROLLING_DAYS, cfg.MIN_OBS)
        else:
            base[score] = np.nan

    w_norm = renormalize_weights(base, cfg.W)
    score_cols = list(cfg.W.keys())
    base["index_score_total"] = (base[score_cols] * w_norm).sum(axis=1)
    base["bucket_5pt"] = (np.floor(base["index_score_total"] / 5.0) * 5.0).clip(0, 100)

    # ---- Upsert ----
    if not old.empty:
        out = pd.concat([old, base], ignore_index=True)
        out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    else:
        out = base

    out.to_parquet(index_path, index=False)
    print(f"[backfill_max] OK years={backfill_years} rows={len(out)} -> {index_path}")


if __name__ == "__main__":
    main()
