#!/usr/bin/env python3
"""
KRX 투자자별 매매(상세컬럼) 기간 분할 백필/증분 수집 (안정/모니터링 강화)

추가된 기능(중요)
- END_DATE 지원: 기간을 START_DATE~END_DATE로 강제
- FLOW_APPEND_IF_EXISTS=true: 기존 파일이 있어도 기간 슬라이스를 append(중복 제거)
- Heartbeat 로그: 완료가 없어도 주기적으로 진행상황 출력
- progress json 기록: docs/stocks/progress_flows.json

목표
- 전 투자자 컬럼 "필터링 없이" 모두 저장 (detail=True 전체 컬럼)
- 매수/매도/순매수 (on: '매수','매도','순매수')
- 금액/수량 모두 수집:
  - get_market_trading_value_by_date
  - get_market_trading_volume_by_date

env (추가/변경)
- END_DATE: 기본 오늘(YYYYMMDD)
- FLOW_APPEND_IF_EXISTS: default "true"
- HEARTBEAT_SEC: default "60"
- PROGRESS_JSON: default "docs/stocks/progress_flows.json"
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


# ===== Config =====
INCREMENTAL_MODE = os.getenv("INCREMENTAL_MODE", "false").lower() == "true"
INCREMENTAL_DAYS = int(os.getenv("INCREMENTAL_DAYS", "5"))

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))  # legacy
MAX_WORKERS_FLOWS = int(os.getenv("MAX_WORKERS_FLOWS", "3"))

START_DATE = os.getenv("START_DATE", "20150101")
END_DATE = os.getenv("END_DATE", datetime.now().strftime("%Y%m%d"))

SAVE_FORMAT = os.getenv("KRX_FLOWS_SAVE_FORMAT", "parquet").lower()  # csv/parquet

RETRY = int(os.getenv("KRX_RETRY", "3"))
NO_DATA_RETRY = int(os.getenv("KRX_NO_DATA_RETRY", "1"))
NO_DATA_SLEEP_BASE = float(os.getenv("KRX_NO_DATA_SLEEP_BASE", "3.0"))

KRX_THROTTLE_SEC = float(os.getenv("KRX_THROTTLE_SEC", "0.2").strip() or "0.2")

FLOW_APPEND_IF_EXISTS = os.getenv("FLOW_APPEND_IF_EXISTS", "true").lower() == "true"

HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "60"))
PROGRESS_JSON = Path(os.getenv("PROGRESS_JSON", "docs/stocks/progress_flows.json"))

ONS = ["순매수", "매수", "매도"]
PROJECT_ROOT = Path.cwd()


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


def _date_range():
    # 기간 분할 백필을 위해 END_DATE를 항상 존중
    if INCREMENTAL_MODE:
        # 증분 모드라도 END_DATE를 존중하고, start만 N일 전으로
        end_date = END_DATE
        start_date = (datetime.now() - timedelta(days=INCREMENTAL_DAYS)).strftime("%Y%m%d")
        return start_date, end_date
    return START_DATE, END_DATE


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


def _out_path_base(out_dir: Path, ticker: str) -> Path:
    return out_dir / ticker


def _exists_any(out_base: Path) -> bool:
    return out_base.with_suffix(".csv").exists() or out_base.with_suffix(".parquet").exists()


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


def _fetch_frames_for_ticker(ticker: str, start_date: str, end_date: str) -> list:
    frames = []

    for on in ONS:
        df = _fetch_with_retry(
            stock.get_market_trading_value_by_date,
            start_date, end_date, ticker,
            detail=True, on=on
        )
        df = _standardize_date_index(df)
        if not df.empty:
            suffix = {"순매수": "value_net", "매수": "value_buy", "매도": "value_sell"}[on]
            frames.append(_rename_with_suffix(df, suffix))

    for on in ONS:
        df = _fetch_with_retry(
            stock.get_market_trading_volume_by_date,
            start_date, end_date, ticker,
            detail=True, on=on
        )
        df = _standardize_date_index(df)
        if not df.empty:
            suffix = {"순매수": "vol_net", "매수": "vol_buy", "매도": "vol_sell"}[on]
            frames.append(_rename_with_suffix(df, suffix))

    return frames


def fetch_one_ticker(ticker: str, start_date: str, end_date: str, out_dir: Path) -> tuple[str, str]:
    """
    returns (ticker, status) status in {"OK","SKIP","NO_DATA","ERROR"}
    """
    out_base = _out_path_base(out_dir, ticker)

    try:
        # 기간분할 백필에서는 append가 기본이므로, exists여도 SKIP하지 않음(append=false일 때만 skip)
        if (not INCREMENTAL_MODE) and _exists_any(out_base) and (not FLOW_APPEND_IF_EXISTS):
            return ticker, "SKIP"

        frames = _fetch_frames_for_ticker(ticker, start_date, end_date)

        if not frames:
            for n in range(1, NO_DATA_RETRY + 1):
                time.sleep(NO_DATA_SLEEP_BASE * n + random.random() * 1.5)
                frames = _fetch_frames_for_ticker(ticker, start_date, end_date)
                if frames:
                    break
            if not frames:
                return ticker, "NO_DATA"

        df_all = pd.DataFrame()
        for df in frames:
            df_all = _safe_merge(df_all, df)

        if df_all is None or df_all.empty:
            return ticker, "NO_DATA"

        # append 모드: 기존 파일과 합치고 date 기준 최신 유지
        if FLOW_APPEND_IF_EXISTS and _exists_any(out_base):
            df_old = _load_existing(out_base)
            if not df_old.empty:
                df_all = pd.concat([df_old, df_all], ignore_index=True)
                df_all = df_all.drop_duplicates(subset=["date"], keep="last")

        _save(df_all, out_base)
        return ticker, "OK"

    except Exception:
        return ticker, "ERROR"


def _write_progress(ok, skip, nodata, err, total, start_ts, start_date, end_date):
    PROGRESS_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "range": {"start": start_date, "end": end_date},
        "counts": {"ok": ok, "skip": skip, "no_data": nodata, "error": err, "done": ok+skip+nodata+err, "total": total},
        "elapsed_sec": int(time.time() - start_ts),
        "utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    PROGRESS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    workers = MAX_WORKERS_FLOWS if MAX_WORKERS_FLOWS > 0 else MAX_WORKERS

    print("[fetch_krx_flows] start")
    print(f"  CWD={PROJECT_ROOT}")
    print(f"  range env: START_DATE={START_DATE}, END_DATE={END_DATE}")
    print(f"  workers={workers}, throttle={KRX_THROTTLE_SEC}s, save_format={SAVE_FORMAT}")
    print(f"  append_if_exists={FLOW_APPEND_IF_EXISTS}, retry={RETRY}, no_data_retry={NO_DATA_RETRY}")

    master_path = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"master not found: {master_path}")

    df_master = pd.read_parquet(master_path)
    tickers = df_master["ticker"].astype(str).tolist()

    start_date, end_date = _date_range()
    out_dir = PROJECT_ROOT / "data/stocks/raw/krx_flows"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(tickers)
    start_ts = time.time()

    ok = skip = nodata = err = 0
    lock = threading.Lock()

    # Heartbeat thread: 완료가 없어도 주기적으로 현재 상태 출력
    stop_flag = {"stop": False}

    def heartbeat():
        while not stop_flag["stop"]:
            time.sleep(HEARTBEAT_SEC)
            with lock:
                done = ok + skip + nodata + err
                print(f"[heartbeat] done={done}/{total} ok={ok} skip={skip} nodata={nodata} err={err} elapsed={int(time.time()-start_ts)}s")
                _write_progress(ok, skip, nodata, err, total, start_ts, start_date, end_date)

    th = threading.Thread(target=heartbeat, daemon=True)
    th.start()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_one_ticker, t, start_date, end_date, out_dir) for t in tickers]

        for i, fut in enumerate(as_completed(futures), 1):
            ticker, status = fut.result()
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
                    print(f"  [{i}/{total}] {ticker}: {status}")

                if i % 50 == 0:
                    _write_progress(ok, skip, nodata, err, total, start_ts, start_date, end_date)

    stop_flag["stop"] = True
    _write_progress(ok, skip, nodata, err, total, start_ts, start_date, end_date)

    print(f"[fetch_krx_flows] done | OK={ok} SKIP={skip} NO_DATA={nodata} ERROR={err}")
    print(f"[fetch_krx_flows] progress json: {PROGRESS_JSON}")


if __name__ == "__main__":
    main()
