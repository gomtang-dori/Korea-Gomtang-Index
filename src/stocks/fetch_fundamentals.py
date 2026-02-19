#!/usr/bin/env python3
"""
PyKRX fundamentals (DIV/BPS/PER/EPS/PBR/DPS ...) backfill/incremental
RAW 저장: data/stocks/raw/fundamentals/{ticker}.parquet

env
- INCREMENTAL_MODE: true/false (default false)
- INCREMENTAL_DAYS: default 5
- START_DATE: default 20150101
- END_DATE: default today (YYYYMMDD)
- MAX_WORKERS_FUNDAMENTALS: default 10
- FUND_CHUNK_YEARS: default 2
- FUND_SAVE_FORMAT: parquet|csv (default parquet)
- FUND_RETRY: exception retry (default 3)
- FUND_EMPTY_RETRY: empty df retry (default 2)
- FUND_EMPTY_SLEEP_BASE: default 2.0
"""

import os
import time
import random
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from pykrx import stock
from concurrent.futures import ThreadPoolExecutor, as_completed


PROJECT_ROOT = Path.cwd()

INCREMENTAL_MODE = os.getenv("INCREMENTAL_MODE", "false").lower() == "true"
INCREMENTAL_DAYS = int(os.getenv("INCREMENTAL_DAYS", "5"))
START_DATE = os.getenv("START_DATE", "20150101")
END_DATE = os.getenv("END_DATE", datetime.now().strftime("%Y%m%d"))

MAX_WORKERS_FUND = int(os.getenv("MAX_WORKERS_FUNDAMENTALS", "10"))
FUND_CHUNK_YEARS = int(os.getenv("FUND_CHUNK_YEARS", "2"))

SAVE_FORMAT = os.getenv("FUND_SAVE_FORMAT", "parquet").lower()
RETRY = int(os.getenv("FUND_RETRY", "3"))
EMPTY_RETRY = int(os.getenv("FUND_EMPTY_RETRY", "2"))
EMPTY_SLEEP_BASE = float(os.getenv("FUND_EMPTY_SLEEP_BASE", "2.0"))

RAW_DIR = PROJECT_ROOT / "data/stocks/raw/fundamentals"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _date_range():
    end_date = END_DATE
    if INCREMENTAL_MODE:
        start_date = (datetime.now() - timedelta(days=INCREMENTAL_DAYS)).strftime("%Y%m%d")
    else:
        start_date = START_DATE
    return start_date, end_date


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index.name = "date"
    out.reset_index(inplace=True)
    out["date"] = pd.to_datetime(out["date"])
    # 컬럼 소문자 표준화(원본 컬럼명 유지 + 소문자 매핑)
    out.columns = [c.lower() for c in out.columns]
    return out


def _load_existing(out_base: Path) -> pd.DataFrame:
    p_parq = out_base.with_suffix(".parquet")
    p_csv = out_base.with_suffix(".csv")
    if not p_parq.exists() and not p_csv.exists():
        return pd.DataFrame()
    try:
        if p_parq.exists():
            df = pd.read_parquet(p_parq)
        else:
            df = pd.read_csv(p_csv, encoding="utf-8-sig")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()


def _save(df: pd.DataFrame, out_base: Path):
    df = df.sort_values("date")
    if SAVE_FORMAT == "parquet":
        df.to_parquet(out_base.with_suffix(".parquet"), index=False)
    else:
        df.to_csv(out_base.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def _fetch_with_retry(fromdate: str, todate: str, ticker: str) -> pd.DataFrame:
    last = None
    for k in range(RETRY):
        try:
            df = stock.get_market_fundamental_by_date(fromdate, todate, ticker)
            return df
        except Exception as e:
            last = e
            time.sleep(0.7 * (k + 1) + random.random() * 0.3)
    raise last


def _chunk_ranges(start_yyyymmdd: str, end_yyyymmdd: str, chunk_years: int):
    s = datetime.strptime(start_yyyymmdd, "%Y%m%d").date()
    e = datetime.strptime(end_yyyymmdd, "%Y%m%d").date()
    cur_y = s.year
    while cur_y <= e.year:
        y1 = cur_y
        y2 = min(cur_y + chunk_years - 1, e.year)
        c_start = max(s, datetime(y1, 1, 1).date())
        c_end = min(e, datetime(y2, 12, 31).date())
        yield c_start.strftime("%Y%m%d"), c_end.strftime("%Y%m%d")
        cur_y = y2 + 1


def fetch_one_ticker(ticker: str, start_date: str, end_date: str) -> str:
    out_base = RAW_DIR / ticker

    # 백필이면 파일 있으면 skip
    if (not INCREMENTAL_MODE) and (out_base.with_suffix(".parquet").exists() or out_base.with_suffix(".csv").exists()):
        return f"{ticker}: exists -> skip"

    frames = []
    try:
        for a, b in _chunk_ranges(start_date, end_date, FUND_CHUNK_YEARS):
            df = _fetch_with_retry(a, b, ticker)
            df = _standardize(df)
            if df.empty:
                # empty retry (ticker+range only)
                ok = False
                for n in range(1, EMPTY_RETRY + 1):
                    time.sleep(EMPTY_SLEEP_BASE * n + random.random())
                    df2 = _fetch_with_retry(a, b, ticker)
                    df2 = _standardize(df2)
                    if not df2.empty:
                        df = df2
                        ok = True
                        break
                if not ok:
                    continue
            frames.append(df)

        if not frames:
            return f"{ticker}: no data"

        df_all = pd.concat(frames, ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["date"], keep="last")

        if INCREMENTAL_MODE:
            old = _load_existing(out_base)
            if not old.empty:
                df_all = pd.concat([old, df_all], ignore_index=True)
                df_all = df_all.drop_duplicates(subset=["date"], keep="last")

        _save(df_all, out_base)
        return f"{ticker}: OK rows={len(df_all):,} cols={len(df_all.columns)}"

    except Exception as e:
        return f"{ticker}: ERROR ({e})"


def main():
    print("[fetch_fundamentals] start")
    print(f"  CWD={PROJECT_ROOT}")
    print(f"  INCREMENTAL_MODE={INCREMENTAL_MODE}, INCREMENTAL_DAYS={INCREMENTAL_DAYS}")
    print(f"  START_DATE={START_DATE}, END_DATE={END_DATE}")
    print(f"  MAX_WORKERS_FUNDAMENTALS={MAX_WORKERS_FUND}, CHUNK_YEARS={FUND_CHUNK_YEARS}")
    print(f"  SAVE_FORMAT={SAVE_FORMAT}, RETRY={RETRY}, EMPTY_RETRY={EMPTY_RETRY}")

    master = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    if not master.exists():
        raise FileNotFoundError(f"missing: {master}")
    df_master = pd.read_parquet(master)
    tickers = df_master["ticker"].astype(str).tolist()

    start_date, end_date = _date_range()
    print(f"  range: {start_date} ~ {end_date} | tickers={len(tickers)}")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_FUND) as ex:
        futures = [ex.submit(fetch_one_ticker, t, start_date, end_date) for t in tickers]
        for i, fut in enumerate(as_completed(futures), 1):
            msg = fut.result()
            results.append(msg)
            if i <= 20 or i % 200 == 0:
                print(f"  [{i}/{len(futures)}] {msg}")

    ok = sum(1 for r in results if ": OK" in r)
    skip = sum(1 for r in results if "skip" in r)
    nodata = sum(1 for r in results if ": no data" in r)
    err = sum(1 for r in results if ": ERROR" in r)
    print(f"[fetch_fundamentals] done | OK={ok} SKIP={skip} NO_DATA={nodata} ERROR={err}")
    print(f"[fetch_fundamentals] raw dir: {RAW_DIR}")


if __name__ == "__main__":
    main()
