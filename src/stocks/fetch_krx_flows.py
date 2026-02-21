#!/usr/bin/env python3
"""
KRX 투자자별 매매(상세컬럼) 안정 강화형 수집

추가(중요)
- FLOWS_RAW_DIR env 지원: 출력 폴더 변경 가능 (matrix 연도별 저장용)
- _no_data_tickers.txt / _error_tickers.txt 를 출력 폴더에 생성 (retry2 호환)

env
- START_DATE, END_DATE
- MAX_WORKERS_FLOWS
- KRX_FLOWS_SAVE_FORMAT (parquet|csv)
- KRX_RETRY
- KRX_NO_DATA_RETRY, KRX_NO_DATA_SLEEP_BASE
- KRX_THROTTLE_SEC
- FLOW_APPEND_IF_EXISTS
- HEARTBEAT_SEC
- PROGRESS_JSON
- FLOWS_RAW_DIR (default data/stocks/raw/krx_flows)
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

MAX_WORKERS_FLOWS = int(os.getenv("MAX_WORKERS_FLOWS", "3"))

START_DATE = os.getenv("START_DATE", "20150101")
END_DATE = os.getenv("END_DATE", datetime.now().strftime("%Y%m%d"))

SAVE_FORMAT = os.getenv("KRX_FLOWS_SAVE_FORMAT", "parquet").lower()
RETRY = int(os.getenv("KRX_RETRY", "3"))

NO_DATA_RETRY = int(os.getenv("KRX_NO_DATA_RETRY", "1"))
NO_DATA_SLEEP_BASE = float(os.getenv("KRX_NO_DATA_SLEEP_BASE", "4.0"))

KRX_THROTTLE_SEC = float(os.getenv("KRX_THROTTLE_SEC", "0.20").strip() or "0.20")
FLOW_APPEND_IF_EXISTS = os.getenv("FLOW_APPEND_IF_EXISTS", "true").lower() == "true"

HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "60"))
PROGRESS_JSON = Path(os.getenv("PROGRESS_JSON", "docs/stocks/progress_flows.json"))

FLOWS_RAW_DIR = Path(os.getenv("FLOWS_RAW_DIR", str(PROJECT_ROOT / "data/stocks/raw/krx_flows")))
FLOWS_RAW_DIR.mkdir(parents=True, exist_ok=True)

ONS = ["순매수", "매수", "매도"]


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


throttle = GlobalThrottle(KRX_THROTTLE_SEC)


def _parse_yyyymmdd(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, format="%Y%m%d")


def _standardize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index.name = "date"
    out.reset_index(inplace=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def _rename_with_suffix(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if df.empty:
        return df
    rename = {c: f"{c}_{suffix}" for c in df.columns if c != "date"}
    return df.rename(columns=rename)


def _safe_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left is None or left.empty:
        return right
    if right is None or right.empty:
        return left
    return left.merge(right, on="date", how="outer")


def _out_base(ticker: str) -> Path:
    return FLOWS_RAW_DIR / ticker


def _exists_any(out_base: Path) -> bool:
    return out_base.with_suffix(".parquet").exists() or out_base.with_suffix(".csv").exists()


def _load_existing(out_base: Path) -> pd.DataFrame:
    p_parq = out_base.with_suffix(".parquet")
    p_csv = out_base.with_suffix(".csv")
    if not p_parq.exists() and not p_csv.exists():
        return pd.DataFrame()
    try:
        if p_parq.exists():
            df_old = pd.read_parquet(p_parq)
        else:
            df_old = pd.read_csv(p_csv, encoding="utf-8-sig")
        if "date" in df_old.columns:
            df_old["date"] = pd.to_datetime(df_old["date"])
        return df_old
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


def _save(df: pd.DataFrame, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("date")
    if SAVE_FORMAT == "parquet":
        df.to_parquet(out_base.with_suffix(".parquet"), index=False)
    else:
        df.to_csv(out_base.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def _fetch_with_retry(fetch_fn, *args, **kwargs) -> pd.DataFrame:
    last_err = None
    for k in range(RETRY):
        try:
            throttle.wait()
            return fetch_fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            time.sleep(0.7 * (k + 1) + random.random() * 0.3)
    raise last_err


def _fetch_frames_for_ticker(ticker: str) -> list[pd.DataFrame]:
    frames = []

    for on in ONS:
        df = _fetch_with_retry(
            stock.get_market_trading_value_by_date,
            START_DATE, END_DATE, ticker,
            detail=True, on=on
        )
        df = _standardize_date_index(df)
        if not df.empty:
            suffix = {"순매수": "value_net", "매수": "value_buy", "매도": "value_sell"}[on]
            frames.append(_rename_with_suffix(df, suffix))

    for on in ONS:
        df = _fetch_with_retry(
            stock.get_market_trading_volume_by_date,
            START_DATE, END_DATE, ticker,
            detail=True, on=on
        )
        df = _standardize_date_index(df)
        if not df.empty:
            suffix = {"순매수": "vol_net", "매수": "vol_buy", "매도": "vol_sell"}[on]
            frames.append(_rename_with_suffix(df, suffix))

    return frames


def _write_list(path: Path, items: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    uniq = sorted(set([x for x in items if x]))
    path.write_text("\n".join(uniq) + ("\n" if uniq else ""), encoding="utf-8")


def _write_progress(ok, skip, nodata, err, total, start_ts):
    PROGRESS_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "range": {"start": START_DATE, "end": END_DATE},
        "counts": {"ok": ok, "skip": skip, "no_data": nodata, "error": err, "done": ok + skip + nodata + err, "total": total},
        "elapsed_sec": int(time.time() - start_ts),
        "utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "throttle_sec": KRX_THROTTLE_SEC,
        "workers": MAX_WORKERS_FLOWS,
        "save_format": SAVE_FORMAT,
        "out_dir": str(FLOWS_RAW_DIR),
    }
    PROGRESS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_one_ticker(ticker: str) -> tuple[str, str]:
    out_base = _out_base(ticker)
    try:
        if _exists_any(out_base):
            old = _load_existing(out_base)
            if _covers_range(old, START_DATE, END_DATE):
                return ticker, "SKIP"

        frames = _fetch_frames_for_ticker(ticker)

        if not frames:
            for n in range(1, NO_DATA_RETRY + 1):
                time.sleep(NO_DATA_SLEEP_BASE * n + random.random() * 1.5)
                frames = _fetch_frames_for_ticker(ticker)
                if frames:
                    break

        if not frames:
            return ticker, "NO_DATA"

        df_all = pd.DataFrame()
        for df in frames:
            df_all = _safe_merge(df_all, df)

        if df_all is None or df_all.empty:
            return ticker, "NO_DATA"

        if FLOW_APPEND_IF_EXISTS and _exists_any(out_base):
            old = _load_existing(out_base)
            if not old.empty:
                df_all = pd.concat([old, df_all], ignore_index=True)
                df_all = df_all.drop_duplicates(subset=["date"], keep="last")

        _save(df_all, out_base)
        return ticker, "OK"
    except Exception:
        return ticker, "ERROR"


def main():
    print("[fetch_krx_flows] start")
    print(f"  range={START_DATE}~{END_DATE}")
    print(f"  out_dir={FLOWS_RAW_DIR}")
    print(f"  workers={MAX_WORKERS_FLOWS}, throttle={KRX_THROTTLE_SEC}s, retry={RETRY}, nodata_retry={NO_DATA_RETRY}, append={FLOW_APPEND_IF_EXISTS}, format={SAVE_FORMAT}")

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

    no_data_tickers: list[str] = []
    error_tickers: list[str] = []

    def heartbeat():
        while not stop["v"]:
            time.sleep(HEARTBEAT_SEC)
            with lock:
                done = ok + skip + nodata + err
                print(f"[heartbeat] done={done}/{total} ok={ok} skip={skip} nodata={nodata} err={err} elapsed={int(time.time()-start_ts)}s")
                _write_progress(ok, skip, nodata, err, total, start_ts)

    threading.Thread(target=heartbeat, daemon=True).start()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_FLOWS) as ex:
        futures = [ex.submit(fetch_one_ticker, t) for t in tickers]
        for i, fut in enumerate(as_completed(futures), 1):
            t, status = fut.result()
            with lock:
                if status == "OK":
                    ok += 1
                elif status == "SKIP":
                    skip += 1
                elif status == "NO_DATA":
                    nodata += 1
                    no_data_tickers.append(t)
                else:
                    err += 1
                    error_tickers.append(t)

                if i <= 20 or i % 200 == 0:
                    print(f"  [{i}/{total}] {t}: {status}")
                if i % 50 == 0:
                    _write_progress(ok, skip, nodata, err, total, start_ts)

    stop["v"] = True
    _write_progress(ok, skip, nodata, err, total, start_ts)

    # retry2 호환 파일을 "해당 출력폴더"에 생성
    _write_list(FLOWS_RAW_DIR / "_no_data_tickers.txt", no_data_tickers)
    _write_list(FLOWS_RAW_DIR / "_error_tickers.txt", error_tickers)

    print(f"[fetch_krx_flows] done | OK={ok} SKIP={skip} NO_DATA={nodata} ERROR={err}")
    print(f"[fetch_krx_flows] wrote: {FLOWS_RAW_DIR / '_no_data_tickers.txt'} ({len(set(no_data_tickers))} tickers)")
    print(f"[fetch_krx_flows] wrote: {FLOWS_RAW_DIR / '_error_tickers.txt'} ({len(set(error_tickers))} tickers)")
    print(f"[fetch_krx_flows] progress: {PROGRESS_JSON}")


if __name__ == "__main__":
    main()
