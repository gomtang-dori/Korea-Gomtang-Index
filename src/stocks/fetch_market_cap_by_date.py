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
- MARKETCAP_MODE: "range" or "recent" (default recent)

Range mode:
- MARKETCAP_START_DATE (YYYYMMDD)
- MARKETCAP_END_DATE   (YYYYMMDD)

Recent mode:
- MARKETCAP_TRADING_DAYS (default 60): number of *trading days* to collect counting backward from today
- MARKETCAP_MAX_BACKTRACK_DAYS (default 200): max calendar days to scan backward

Common:
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


MODE = (os.getenv("MARKETCAP_MODE", "recent") or "recent").strip().lower()

OUT_DIR = Path(os.getenv("MARKETCAP_OUT_DIR", "data/stocks/raw/market_cap_by_date"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

THROTTLE_SEC = float(os.getenv("MARKETCAP_THROTTLE_SEC", "0.20") or "0.20")
OVERWRITE = env_bool("MARKETCAP_OVERWRITE", False)
COMPRESSION = (os.getenv("MARKETCAP_COMPRESSION", "zstd") or "zstd").strip()

# recent mode
TRADING_DAYS = int(os.getenv("MARKETCAP_TRADING_DAYS", "60"))
MAX_BACKTRACK = int(os.getenv("MARKETCAP_MAX_BACKTRACK_DAYS", "200"))

# range mode
START_DATE = (os.getenv("MARKETCAP_START_DATE", "") or "").strip()
END_DATE = (os.getenv("MARKETCAP_END_DATE", "") or "").strip()


def _fetch_one_market(date_yyyymmdd: str, market: str) -> pd.DataFrame:
    df = stock.get_market_cap_by_ticker(date_yyyymmdd, market=market)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.index = df.index.astype(str).str.zfill(6)
    df.reset_index(inplace=True)
    df.rename(columns={"티커": "ticker"}, inplace=True)

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
    return df[["date", "ticker", "market_cap", "shares", "value"]]


def _fetch_one_date(date_yyyymmdd: str) -> pd.DataFrame:
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


def _save_if_any(date_yyyymmdd: str) -> bool:
    out_path = OUT_DIR / f"{date_yyyymmdd}.parquet"
    if out_path.exists() and not OVERWRITE:
        return True

    df = _fetch_one_date(date_yyyymmdd)
    if df.empty:
        return False

    df.to_parquet(out_path, index=False, compression=COMPRESSION)
    return True


def run_recent():
    print("[marketcap] mode=recent")
    print(f"  out_dir={OUT_DIR}")
    print(f"  target_trading_days={TRADING_DAYS}, max_backtrack_days={MAX_BACKTRACK}")
    print(f"  throttle_sec={THROTTLE_SEC}, overwrite={OVERWRITE}, compression={COMPRESSION}")

    got = 0
    scanned = 0
    d = datetime.utcnow().date()

    while got < TRADING_DAYS and scanned < MAX_BACKTRACK:
        yyyymmdd = d.strftime("%Y%m%d")
        scanned += 1

        ok = _save_if_any(yyyymmdd)
        if ok:
            got += 1
            if got <= 3 or got % 10 == 0:
                print(f"  [{got}/{TRADING_DAYS}] ok: {yyyymmdd}")

        d = d - timedelta(days=1)

    print("[marketcap] done")
    print(f"  got_trading_days={got}, scanned_calendar_days={scanned}")
    if got < TRADING_DAYS:
        raise SystemExit(f"[marketcap] not enough trading days: got={got}, want={TRADING_DAYS}")


def run_range():
    if not START_DATE or not END_DATE:
        raise SystemExit("[marketcap] range mode requires MARKETCAP_START_DATE and MARKETCAP_END_DATE")

    print("[marketcap] mode=range")
    print(f"  start={START_DATE}, end={END_DATE}")
    print(f"  out_dir={OUT_DIR}")
    print(f"  throttle_sec={THROTTLE_SEC}, overwrite={OVERWRITE}, compression={COMPRESSION}")

    s = datetime.strptime(START_DATE, "%Y%m%d").date()
    e = datetime.strptime(END_DATE, "%Y%m%d").date()
    d = s

    saved = 0
    while d <= e:
        yyyymmdd = d.strftime("%Y%m%d")
        try:
            ok = _save_if_any(yyyymmdd)
            if ok:
                saved += 1
                if saved <= 3 or saved % 50 == 0:
                    print(f"  saved={saved:,} last={yyyymmdd}")
        except Exception as ex:
            print(f"  [warn] {yyyymmdd} error: {ex}")
        d = d + timedelta(days=1)

    print("[marketcap] done")
    print(f"  saved_files={saved:,}")


def main():
    if MODE == "range":
        run_range()
    else:
        run_recent()


if __name__ == "__main__":
    main()
