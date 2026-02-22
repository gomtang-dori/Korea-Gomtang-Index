#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch daily market cap table by date (KOSPI+KOSDAQ) and save as one parquet per trading day.

Output:
- data/stocks/raw/market_cap_by_date/YYYYMMDD.parquet
Schema (fixed):
- date (datetime64[ns])
- ticker (6-digit str)
- market_cap (float)
- shares (float)
- value (float)  # trading value (KRW)

Env:
- MARKETCAP_TRADING_DAYS (default 60): number of *trading days* to collect counting backward from today
- MARKETCAP_MAX_BACKTRACK_DAYS (default 200): max calendar days to scan backward
- MARKETCAP_THROTTLE_SEC (default 0.20): sleep between API calls
- MARKETCAP_OVERWRITE (default false): overwrite existing files
- MARKETCAP_OUT_DIR (default data/stocks/raw/market_cap_by_date)
- MARKETCAP_COMPRESSION (default zstd)
"""

import os
import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from pykrx import stock


def env_bool(k: str, default: bool = False) -> bool:
    v = os.getenv(k)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


OUT_DIR = Path(os.getenv("MARKETCAP_OUT_DIR", "data/stocks/raw/market_cap_by_date"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = int(os.getenv("MARKETCAP_TRADING_DAYS", "60"))
MAX_BACKTRACK = int(os.getenv("MARKETCAP_MAX_BACKTRACK_DAYS", "200"))
THROTTLE_SEC = float(os.getenv("MARKETCAP_THROTTLE_SEC", "0.20") or "0.20")
OVERWRITE = env_bool("MARKETCAP_OVERWRITE", False)

COMPRESSION = os.getenv("MARKETCAP_COMPRESSION", "zstd").strip() or "zstd"


def _fetch_one_market(date_yyyymmdd: str, market: str) -> pd.DataFrame:
    # get_market_cap_by_ticker returns all tickers for that market/date in one call
    df = stock.get_market_cap_by_ticker(date_yyyymmdd, market=market)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.index = df.index.astype(str).str.zfill(6)
    df.reset_index(inplace=True)
    df.rename(columns={"티커": "ticker"}, inplace=True)

    # Column names are Korean in PyKRX
    # Expected: 시가총액, 상장주식수, 거래대금 (and others)
    col_map = {
        "시가총액": "market_cap",
        "상장주식수": "shares",
        "거래대금": "value",
    }
    keep = ["ticker"] + [k for k in col_map.keys() if k in df.columns]
    df = df[keep].rename(columns=col_map)

    for c in ["market_cap", "shares", "value"]:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date"] = pd.to_datetime(date_yyyymmdd, format="%Y%m%d", errors="coerce")
    df = df[["date", "ticker", "market_cap", "shares", "value"]]
    return df


def _fetch_one_date(date_yyyymmdd: str) -> pd.DataFrame:
    # KOSPI + KOSDAQ
    time.sleep(THROTTLE_SEC)
    a = _fetch_one_market(date_yyyymmdd, "KOSPI")
    time.sleep(THROTTLE_SEC)
    b = _fetch_one_market(date_yyyymmdd, "KOSDAQ")
    if a.empty and b.empty:
        return pd.DataFrame()
    df = pd.concat([a, b], ignore_index=True)
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df = df.dropna(subset=["date", "ticker"]).drop_duplicates(subset=["date", "ticker"], keep="last")
    return df


def main():
    print("[marketcap] start")
    print(f"  out_dir={OUT_DIR}")
    print(f"  target_trading_days={TRADING_DAYS}, max_backtrack_days={MAX_BACKTRACK}")
    print(f"  throttle_sec={THROTTLE_SEC}, overwrite={OVERWRITE}, compression={COMPRESSION}")

    got = 0
    scanned = 0
    d = datetime.utcnow().date()

    while got < TRADING_DAYS and scanned < MAX_BACKTRACK:
        yyyymmdd = d.strftime("%Y%m%d")
        out_path = OUT_DIR / f"{yyyymmdd}.parquet"

        scanned += 1

        if out_path.exists() and not OVERWRITE:
            got += 1
            if got <= 3 or got % 10 == 0:
                print(f"  [{got}/{TRADING_DAYS}] exists -> {out_path.name}")
            d = d - timedelta(days=1)
            continue

        try:
            df = _fetch_one_date(yyyymmdd)
            if df.empty:
                # non-trading day or no data
                d = d - timedelta(days=1)
                continue

            df.to_parquet(out_path, index=False, compression=COMPRESSION)
            got += 1
            if got <= 3 or got % 10 == 0:
                print(f"  [{got}/{TRADING_DAYS}] saved {out_path.name} rows={len(df):,}")
        except Exception as e:
            print(f"  [warn] {yyyymmdd} fetch error: {e}")

        d = d - timedelta(days=1)

    print("[marketcap] done")
    print(f"  got_trading_days={got}, scanned_calendar_days={scanned}")
    if got < TRADING_DAYS:
        raise SystemExit(f"[marketcap] not enough trading days: got={got}, want={TRADING_DAYS}")


if __name__ == "__main__":
    main()
