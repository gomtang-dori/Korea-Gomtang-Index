#!/usr/bin/env python3
"""
PyKRX fundamentals(DIV/BPS/PER/EPS/PBR/DPS ...) 안정 강화형 수집
RAW 저장: data/stocks/raw/fundamentals/{ticker}.parquet

개선
- START_DATE / END_DATE env 존중
- FUND_APPEND_IF_EXISTS=true: 기간 슬라이스 append + date 기준 dedup
- 기존 파일이 이미 기간 커버하면 SKIP
- exception retry + empty(no data) retry
- throttle + progress json + heartbeat

env
- INCREMENTAL_MODE, INCREMENTAL_DAYS
- START_DATE, END_DATE
- MAX_WORKERS_FUNDAMENTALS
- FUND_CHUNK_YEARS (default 1 권장; 연도 분할 실행 시)
- FUND_SAVE_FORMAT parquet|csv
- FUND_RETRY (exception retry)
- FUND_EMPTY_RETRY, FUND_EMPTY_SLEEP_BASE
- FUND_NO_DATA_RETRY, FUND_NO_DATA_SLEEP_BASE
- FUND_THROTTLE_SEC
- FUND_APPEND_IF_EXISTS (default true)
- HEARTBEAT_SEC
- PROGRESS_JSON
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

START_DATE = os.getenv("START_DATE", "20150101")
END_DATE = os.getenv("END_DATE", datetime.now().strftime("%Y%m%d"))

MAX_WORKERS_FUND = int(os.getenv("MAX_WORKERS_FUNDAMENTALS", "8"))
FUND_CHUNK_YEARS = int(os.getenv("FUND_CHUNK_YEARS", "1"))

SAVE_FORMAT = os.getenv("FUND_SAVE_FORMAT", "parquet").lower()
RETRY = int(os.getenv("FUND_RETRY", "3"))

EMPTY_RETRY = int(os.getenv("FUND_EMPTY_RETRY", "2"))
EMPTY_SLEEP_BASE = float(os.getenv("FUND_EMPTY_SLEEP_BASE", "2.5"))

NO_DATA_RETRY = int(os.getenv("FUND_NO_DATA_RETRY", "1"))
NO_DATA_SLEEP_BASE = float(os.getenv("FUND_NO_DATA_SLEEP_BASE", "3.0"))

FUND_THROTTLE_SEC = float(os.getenv("FUND_THROTTLE_SEC", "0.08").strip() or "0.08")

FUND_APPEND_IF_EXISTS = os.getenv("FUND_APPEND_IF_EXISTS", "true").lower() == "true"

HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "60"))
PROGRESS_JSON = Path(os.getenv("PROGRESS_JSON", "docs/stocks/progress_fundamentals.json"))

RAW_DIR = PROJECT_ROOT / "data/stocks/raw/fundamentals"
RAW_DIR.mkdir(parents=True, exist_ok=True)


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


throttle = GlobalThrottle(FUND_THROTTLE_SEC)


def _date_range():
    if INCREMENTAL_MODE:
        end_date = END_DATE
        start_date = (datetime.now() - timedelta(days=INCREMENTAL_DAYS)).strftime("%Y%m%d")
        return start_date, end_date
    return START_DATE, END_DATE


def _parse_yyyymmdd(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, format="%Y%m%d")


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index.name = "date"
    out.reset_index(inplace=True)
    out["date"] = pd.to_datetime(out["date"])
    out.columns = [c.lower() for c in out.columns]
    return out


def _covers_range(df_old: pd.DataFrame, start_yyyymmdd: str, end_yyyymmdd: str) -> bool:
    if df_old is None or df_old.empty or "date" not in df_old.columns:
        return False
    s = _parse_yyyymmdd(start_yyyymmdd)
    e = _parse_yyyymmdd(end_yyyymmdd)
    dmin = pd.to_datetime(df_old["date"]).min()
    dmax = pd.to_datetime(df_old["date"]).max()
    return (dmin <= s) and (dmax >= e)


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
            throttle.wait()
            return stock.get_market_fundamental_by_date(fromdate, todate, ticker)
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


def _write_progress(ok, skip, nodata, err, total, start_ts, start_date, end_date):
    PROGRESS_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "range": {"start": start_date, "end": end_date},
        "counts": {"ok": ok, "skip": skip, "no_data": nodata, "error": err, "done": ok+skip+nodata+err, "total": total},
        "elapsed_sec": int(time.time() - start_ts),
        "utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "throttle_sec": FUND_THROTTLE_SEC,
        "workers": MAX_WORKERS_FUND,
        "chunk_years": FUND_CHUNK_YEARS,
    }
    PROGRESS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_one_ticker(ticker: str, start_date: str, end_date: str) -> tuple[str, str]:
    """
    returns (ticker, status): OK / SKIP / NO_DATA / ERROR
    """
    out_base = RAW_DIR / ticker

    try:
        if out_base.with_suffix(".parquet").exists() or out_base.with_suffix(".csv").exists():
            old = _load_existing(out_base)
            if _covers_range(old, start_date, end_date):
                return ticker, "SKIP"

        frames = []
        for a, b in _chunk_ranges(start_date, end_date, FUND_CHUNK_YEARS):
            df = _fetch_with_retry(a, b, ticker)
            df = _standardize(df)

            if df.empty:
                # empty retry (range 단위)
                ok = False
                for n in range(1, EMPTY_RETRY + 1):
                    time.sleep(EMPTY_SLEEP_BASE * n + random.random())
                    df2 = _standardize(_fetch_with_retry(a, b, ticker))
                    if not df2.empty:
                        df = df2
                        ok = True
                        break
                if not ok:
                    continue

            frames.append(df)

        if not frames:
            # no-data retry (ticker 단위)
            for n in range(1, NO_DATA_RETRY + 1):
                time.sleep(NO_DATA_SLEEP_BASE * n + random.random())
                frames2 = []
                for a, b in _chunk_ranges(start_date, end_date, FUND_CHUNK_YEARS):
                    df = _standardize(_fetch_with_retry(a, b, ticker))
                    if not df.empty:
                        frames2.append(df)
                if frames2:
                    frames = frames2
                    break

        if not frames:
            return ticker, "NO_DATA"

        df_all = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"], keep="last")

        if INCREMENTAL_MODE or FUND_APPEND_IF_EXISTS:
            old = _load_existing(out_base)
            if not old.empty:
                df_all = pd.concat([old, df_all], ignore_index=True)
                df_all = df_all.drop_duplicates(subset=["date"], keep="last")

        _save(df_all, out_base)
        return ticker, "OK"

    except Exception:
        return ticker, "ERROR"


def main():
    print("[fetch_fundamentals] start")
    print(f"  START_DATE={START_DATE}, END_DATE={END_DATE}")
    print(f"  workers={MAX_WORKERS_FUND}, throttle={FUND_THROTTLE_SEC}s, chunk_years={FUND_CHUNK_YEARS}")
    print(f"  retry={RETRY}, empty_retry={EMPTY_RETRY}, no_data_retry={NO_DATA_RETRY}, append={FUND_APPEND_IF_EXISTS}")

    master = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    if not master.exists():
        raise FileNotFoundError(f"missing: {master}")

    df_master = pd.read_parquet(master)
    tickers = df_master["ticker"].astype(str).tolist()

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

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_FUND) as ex:
        futures = [ex.submit(fetch_one_ticker, t, start_date, end_date) for t in tickers]
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

    print(f"[fetch_fundamentals] done | OK={ok} SKIP={skip} NO_DATA={nodata} ERROR={err}")
    print(f"[fetch_fundamentals] progress: {PROGRESS_JSON}")
    print(f"[fetch_fundamentals] raw dir: {RAW_DIR}")


if __name__ == "__main__":
    main()
