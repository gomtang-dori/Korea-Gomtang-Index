#!/usr/bin/env python3
"""
가격(OHLCV) 안정 강화형 수집: 기간 분할 + append + no-data retry + throttle + progress

추가(중요)
- PRICES_RAW_DIR env 지원: 출력 폴더를 바꿀 수 있음 (matrix 연도별 저장용)

env
- START_DATE, END_DATE (YYYYMMDD)
- MAX_WORKERS
- PRICE_APPEND_IF_EXISTS (default true)
- PRICE_RETRY (default 3)
- PRICE_NO_DATA_RETRY (default 1)
- PRICE_NO_DATA_SLEEP_BASE (default 3.0)
- PRICE_THROTTLE_SEC (default 0.06)
- HEARTBEAT_SEC (default 60)
- PROGRESS_JSON (default docs/stocks/progress_prices.json)
- PRICES_RAW_DIR (default data/stocks/raw/prices)
"""

import os
import time
import random
import json
import threading
from pathlib import Path
from datetime import datetime

import pandas as pd
from pykrx import stock
from concurrent.futures import ThreadPoolExecutor, as_completed


PROJECT_ROOT = Path.cwd()

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

PRICES_RAW_DIR = Path(os.getenv("PRICES_RAW_DIR", str(PROJECT_ROOT / "data/stocks/raw/prices")))


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


def _write_progress(ok, skip, nodata, err, total, start_ts):
    PROGRESS_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "range": {"start": START_DATE, "end": END_DATE},
        "counts": {"ok": ok, "skip": skip, "no_data": nodata, "error": err, "done": ok + skip + nodata + err, "total": total},
        "elapsed_sec": int(time.time() - start_ts),
        "utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "throttle_sec": PRICE_THROTTLE_SEC,
        "workers": MAX_WORKERS,
        "out_dir": str(PRICES_RAW_DIR),
    }
    PROGRESS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_with_retry(ticker: str) -> pd.DataFrame:
    last_err = None
    for k in range(PRICE_RETRY):
        try:
            throttle.wait()
            return stock.get_market_ohlcv_by_date(START_DATE, END_DATE, ticker)
        except Exception as e:
            last_err = e
            time.sleep(0.7 * (k + 1) + random.random() * 0.3)
    raise last_err


def fetch_one_ticker_price(ticker: str) -> tuple[str, str]:
    out_path = PRICES_RAW_DIR / f"{ticker}.csv"
    try:
        if out_path.exists():
            old = _load_existing(out_path)
            if _covers_range(old, START_DATE, END_DATE):
                return ticker, "SKIP"

        df = _fetch_with_retry(ticker)
        if df is None or df.empty:
            ok = False
            for n in range(1, PRICE_NO_DATA_RETRY + 1):
                time.sleep(PRICE_NO_DATA_SLEEP_BASE * n + random.random())
                df2 = _fetch_with_retry(ticker)
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

        if PRICE_APPEND_IF_EXISTS and out_path.exists():
            old = _load_existing(out_path)
            if not old.empty:
                df = pd.concat([old, df], ignore_index=True)
                df = df.drop_duplicates(subset=["date"], keep="last")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.sort_values("date").to_csv(out_path, index=False, encoding="utf-8-sig")
        return ticker, "OK"
    except Exception:
        return ticker, "ERROR"


def main():
    print("[fetch_prices] start")
    print(f"  range={START_DATE}~{END_DATE}")
    print(f"  out_dir={PRICES_RAW_DIR}")
    print(f"  workers={MAX_WORKERS}, throttle={PRICE_THROTTLE_SEC}s, retry={PRICE_RETRY}, nodata_retry={PRICE_NO_DATA_RETRY}, append={PRICE_APPEND_IF_EXISTS}")

    master_path = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"master not found: {master_path}")

    df_master = pd.read_parquet(master_path)
    tickers = df_master["ticker"].astype(str).tolist()
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
                _write_progress(ok, skip, nodata, err, total, start_ts)

    threading.Thread(target=heartbeat, daemon=True).start()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fetch_one_ticker_price, t) for t in tickers]
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
                    _write_progress(ok, skip, nodata, err, total, start_ts)

    stop["v"] = True
    _write_progress(ok, skip, nodata, err, total, start_ts)
    print(f"[fetch_prices] done | OK={ok} SKIP={skip} NO_DATA={nodata} ERROR={err}")
    print(f"[fetch_prices] progress: {PROGRESS_JSON}")


if __name__ == "__main__":
    main()
