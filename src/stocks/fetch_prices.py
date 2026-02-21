#!/usr/bin/env python3
"""
가격(OHLCV) 안정 강화형 수집: 기간 분할 + append + no-data retry + throttle + progress

env
- INCREMENTAL_MODE, INCREMENTAL_DAYS
- START_DATE, END_DATE
- MAX_WORKERS
- PRICE_APPEND_IF_EXISTS (default true)
- PRICE_RETRY (default 3)
- PRICE_NO_DATA_RETRY (default 1)
- PRICE_NO_DATA_SLEEP_BASE (default 3.0)
- PRICE_THROTTLE_SEC (default 0.06)
- HEARTBEAT_SEC (default 60)
- PROGRESS_JSON (default docs/stocks/progress_prices.json)
"""

import os
import time
import random
import json
import threading
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

PRICE_RETRY = int(os.getenv("PRICE_RETRY", "3"))
PRICE_NO_DATA_RETRY = int(os.getenv("PRICE_NO_DATA_RETRY", "1"))
PRICE_NO_DATA_SLEEP_BASE = float(os.getenv("PRICE_NO_DATA_SLEEP_BASE", "3.0"))

PRICE_THROTTLE_SEC = float(os.getenv("PRICE_THROTTLE_SEC", "0.06").strip() or "0.06")

HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "60"))
PROGRESS_JSON = Path(os.getenv("PROGRESS_JSON", "docs/stocks/progress_prices.json"))


class GlobalThrottle:
    def __init__(self, min_interval_sec: float):
        self.min_interval_sec = float(min_interval_sec)
        self.lock = threading.Lock()
        self.last_ts = 0.0

    def wait(self):
        if self.min_interval_sec <= 0:
            return
        with self.lock:
            now = time.time()
            gap = now - self.last_ts
            wait_sec = self.min_interval_sec - gap
            if wait_sec > 0:
                time.sleep(wait_sec)
            self.last_ts = time.time()


throttle = GlobalThrottle(PRICE_THROTTLE_SEC)


def _date_range():
    if INCREMENTAL_MODE:
        end_date = END_DATE
        start_date = (datetime.now() - timedelta(days=INCREMENTAL_DAYS)).strftime("%Y%m%d")
        return start_date, end_date
    return START_DATE, END_DATE


def _parse_yyyymmdd(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, format="%Y%m%d")


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


def _covers_range(df_old: pd.DataFrame, start_yyyymmdd: str, end_yyyymmdd: str) -> bool:
    if df_old is None or df_old.empty or "date" not in df_old.columns:
        return False
    s = _parse_yyyymmdd(start_yyyymmdd)
    e = _parse_yyyymmdd(end_yyyymmdd)
    dmin = pd.to_datetime(df_old["date"]).min()
    dmax = pd.to_datetime(df_old["date"]).max()
    return (dmin <= s) and (dmax >= e)


def _write_progress(ok, skip, nodata, err, total, start_ts, start_date, end_date):
    PROGRESS_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "range": {"start": start_date, "end": end_date},
        "counts": {"ok": ok, "skip": skip, "no_data": nodata, "error": err, "done": ok+skip+nodata+err, "total": total},
        "elapsed_sec": int(time.time() - start_ts),
        "utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "throttle_sec": PRICE_THROTTLE_SEC,
        "workers": MAX_WORKERS,
    }
    PROGRESS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_with_retry(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    last_err = None
    for k in range(PRICE_RETRY):
        try:
            throttle.wait()
            return stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
        except Exception as e:
            last_err = e
            time.sleep(0.7 * (k + 1) + random.random() * 0.3)
    raise last_err


def fetch_one_ticker_price(ticker: str, start_date: str, end_date: str, out_path: Path) -> tuple[str, str]:
    """
    returns (ticker, status): OK / SKIP / NO_DATA / ERROR
    """
    try:
        if out_path.exists():
            old = _load_existing(out_path)
            if _covers_range(old, start_date, end_date):
                return ticker, "SKIP"

        df = _fetch_with_retry(ticker, start_date, end_date)
        if df is None or df.empty:
            # no-data retry
            ok = False
            for n in range(1, PRICE_NO_DATA_RETRY + 1):
                time.sleep(PRICE_NO_DATA_SLEEP_BASE * n + random.random())
                df2 = _fetch_with_retry(ticker, start_date, end_date)
                if df2 is not None and not df2.empty:
                    df = df2
                    ok = True
                    break
            if not ok:
                return ticker, "NO_DATA"

        df = df.copy()
        df.reset_index(inplace=True)
        df.rename(columns={"날짜": "date", "종가": "close", "시가": "open", "고가": "high", "저가": "low", "거래량": "volume"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])

        if (INCREMENTAL_MODE or PRICE_APPEND_IF_EXISTS) and out_path.exists():
            old = _load_existing(out_path)
            if not old.empty:
                df = pd.concat([old, df], ignore_index=True)
                df = df.drop_duplicates(subset=["date"], keep="last")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.sort_values("date").to_csv(out_path, index=False, encoding="utf-8-sig")
        return ticker, "OK"

    except Exception:
        return ticker, "ERROR"


def fetch_prices():
    print("[fetch_prices] start")
    print(f"  START_DATE={START_DATE}, END_DATE={END_DATE}")
    print(f"  workers={MAX_WORKERS}, throttle={PRICE_THROTTLE_SEC}s, append={PRICE_APPEND_IF_EXISTS}")
    print(f"  retry={PRICE_RETRY}, no_data_retry={PRICE_NO_DATA_RETRY}")

    master_path = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"master not found: {master_path}")

    df_master = pd.read_parquet(master_path)
    tickers = df_master["ticker"].astype(str).tolist()

    out_dir = PROJECT_ROOT / "data/stocks/raw/prices"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date, end_date = _date_range()
    total = len(tickers)
    start_ts = time.time()

    ok = skip = nodata = err = 0
    lock = threading.Lock()
    stop = {"v": False}

    def heartbeat():
        while not stop["v"]:
            time.sleep(HEARTBEAT_SEC)
            with lock:
                done = ok + skip + nodata + err
                print(f"[heartbeat] done={done}/{total} ok={ok} skip={skip} nodata={nodata} err={err} elapsed={int(time.time()-start_ts)}s")
                _write_progress(ok, skip, nodata, err, total, start_ts, start_date, end_date)

    threading.Thread(target=heartbeat, daemon=True).start()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fetch_one_ticker_price, t, start_date, end_date, out_dir / f"{t}.csv") for t in tickers]
        for i, fut in enumerate(as_completed(futures), 1):
            _, status = fut.result()
            with lock:
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
                if i % 50 == 0:
                    _write_progress(ok, skip, nodata, err, total, start_ts, start_date, end_date)

    stop["v"] = True
    _write_progress(ok, skip, nodata, err, total, start_ts, start_date, end_date)
    print(f"[fetch_prices] done | OK={ok} SKIP={skip} NO_DATA={nodata} ERROR={err}")
    print(f"[fetch_prices] progress: {PROGRESS_JSON}")


if __name__ == "__main__":
    fetch_prices()
