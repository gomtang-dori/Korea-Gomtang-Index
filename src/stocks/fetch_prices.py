#!/usr/bin/env python3
"""
가격(OHLCV) 수집: 연도/기간 분할 + append + 진행 로그

env
- START_DATE: YYYYMMDD
- END_DATE:   YYYYMMDD (default today)
- INCREMENTAL_MODE: true/false
- INCREMENTAL_DAYS: default 5
- MAX_WORKERS: default 10
- PRICE_APPEND_IF_EXISTS: default true  (연도 분할 backfill에서 필수)
"""

import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from pykrx import stock
from concurrent.futures import ThreadPoolExecutor, as_completed


PROJECT_ROOT = Path.cwd()

INCREMENTAL_MODE = os.getenv("INCREMENTAL_MODE", "false").lower() == "true"
INCREMENTAL_DAYS = int(os.getenv("INCREMENTAL_DAYS", "5"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))

START_DATE = os.getenv("START_DATE", "20200101")
END_DATE = os.getenv("END_DATE", datetime.now().strftime("%Y%m%d"))

PRICE_APPEND_IF_EXISTS = os.getenv("PRICE_APPEND_IF_EXISTS", "true").lower() == "true"


def _date_range():
    if INCREMENTAL_MODE:
        end_date = END_DATE
        start_date = (datetime.now() - timedelta(days=INCREMENTAL_DAYS)).strftime("%Y%m%d")
        return start_date, end_date
    return START_DATE, END_DATE


def _load_existing(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()


def fetch_one_ticker_price(ticker: str, start_date: str, end_date: str, out_path: Path) -> tuple[str, str]:
    """
    returns (ticker, status) status in {"OK","NO_DATA","ERROR","SKIP"}
    """
    try:
        if (not INCREMENTAL_MODE) and out_path.exists() and (not PRICE_APPEND_IF_EXISTS):
            return ticker, "SKIP"

        df_ohlcv = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
        if df_ohlcv is None or df_ohlcv.empty:
            return ticker, "NO_DATA"

        df_ohlcv.reset_index(inplace=True)
        df_ohlcv.rename(
            columns={
                "날짜": "date",
                "종가": "close",
                "시가": "open",
                "고가": "high",
                "저가": "low",
                "거래량": "volume",
            },
            inplace=True,
        )
        df_ohlcv["date"] = pd.to_datetime(df_ohlcv["date"])

        if out_path.exists() and (INCREMENTAL_MODE or PRICE_APPEND_IF_EXISTS):
            old = _load_existing(out_path)
            if not old.empty:
                df_ohlcv = pd.concat([old, df_ohlcv], ignore_index=True)
                df_ohlcv = df_ohlcv.drop_duplicates(subset=["date"], keep="last")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_ohlcv.sort_values("date").to_csv(out_path, index=False, encoding="utf-8-sig")
        return ticker, "OK"

    except Exception:
        return ticker, "ERROR"


def fetch_prices():
    print("[fetch_prices] start")
    print(f"  INCREMENTAL_MODE={INCREMENTAL_MODE}, MAX_WORKERS={MAX_WORKERS}")
    print(f"  START_DATE={START_DATE}, END_DATE={END_DATE}, APPEND={PRICE_APPEND_IF_EXISTS}")

    master_path = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"master not found: {master_path}")

    df_master = pd.read_parquet(master_path)
    tickers = df_master["ticker"].astype(str).tolist()

    out_dir = PROJECT_ROOT / "data/stocks/raw/prices"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date, end_date = _date_range()
    total = len(tickers)

    ok = skip = nodata = err = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_one_ticker_price, t, start_date, end_date, out_dir / f"{t}.csv"): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            _, status = fut.result()
            if status == "OK":
                ok += 1
            elif status == "SKIP":
                skip += 1
            elif status == "NO_DATA":
                nodata += 1
            else:
                err += 1

            if i <= 20 or i % 200 == 0:
                print(f"  [{i}/{total}] {status}")

    print(f"[fetch_prices] done | OK={ok} SKIP={skip} NO_DATA={nodata} ERROR={err}")
    print(f"[fetch_prices] out_dir={out_dir}")


if __name__ == "__main__":
    fetch_prices()
