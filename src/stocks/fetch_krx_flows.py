#!/usr/bin/env python3
"""
KRX 투자자별 매매(전 투자자 컬럼) 백필/증분 수집

목표
- 전 투자자 컬럼 "필터링 없이" 모두 저장 (detail=True 전체 컬럼)
- 매수/매도/순매수 (on: '매수','매도','순매수')
- 금액/수량 모두 수집:
  - get_market_trading_value_by_date
  - get_market_trading_volume_by_date

개선사항(이번 버전)
- MAX_WORKERS_FLOWS 도입: flows 병렬수만 별도 관리(기본 6)
  * prices는 MAX_WORKERS=20 유지, flows는 MAX_WORKERS_FLOWS=6 권장
- 'no data'가 뜬 종목은 그 종목만 추가 재시도
- no data / error 티커를 파일로 남김:
  - data/stocks/raw/krx_flows/_no_data_tickers.txt
  - data/stocks/raw/krx_flows/_error_tickers.txt

환경변수
- INCREMENTAL_MODE: true/false (기본 false)
- INCREMENTAL_DAYS: 증분일수 (기본 5)
- MAX_WORKERS: (호환용) 기본 10
- MAX_WORKERS_FLOWS: flows 전용 worker 수 (기본 6)  <-- NEW
- START_DATE: 백필 시작일 (기본 20150101)
- KRX_FLOWS_SAVE_FORMAT: csv or parquet (기본 csv)
- KRX_RETRY: 예외 발생 재시도 횟수 (기본 3)
- KRX_NO_DATA_RETRY: empty(no data) 재시도 횟수 (기본 2)  <-- NEW
- KRX_NO_DATA_SLEEP_BASE: no data 재시도 시 대기 base 초 (기본 3.0) <-- NEW
- KRX_THROTTLE_SEC: 요청 간 최소 간격(초). 0이면 미사용 (기본 0) <-- NEW
"""

import os
import time
import random
import threading
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from pykrx import stock
from concurrent.futures import ThreadPoolExecutor, as_completed


# ===== Config =====
INCREMENTAL_MODE = os.getenv("INCREMENTAL_MODE", "false").lower() == "true"
INCREMENTAL_DAYS = int(os.getenv("INCREMENTAL_DAYS", "5"))

# 호환용(기존)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
# NEW: flows 전용 worker (우선 적용)
MAX_WORKERS_FLOWS = int(os.getenv("MAX_WORKERS_FLOWS", "6"))

START_DATE = os.getenv("START_DATE", "20150101")
SAVE_FORMAT = os.getenv("KRX_FLOWS_SAVE_FORMAT", "csv").lower()  # csv/parquet
RETRY = int(os.getenv("KRX_RETRY", "3"))

# NEW: no data retry
NO_DATA_RETRY = int(os.getenv("KRX_NO_DATA_RETRY", "2"))
NO_DATA_SLEEP_BASE = float(os.getenv("KRX_NO_DATA_SLEEP_BASE", "3.0"))

# NEW: optional throttle (reduce KRX blocking risk)
KRX_THROTTLE_SEC = float(os.getenv("KRX_THROTTLE_SEC", "0").strip() or "0")

ONS = ["순매수", "매수", "매도"]

PROJECT_ROOT = Path.cwd()


# ===== Rate limiter (optional) =====
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
    end_date = datetime.now().strftime("%Y%m%d")
    if INCREMENTAL_MODE:
        start_date = (datetime.now() - timedelta(days=INCREMENTAL_DAYS)).strftime("%Y%m%d")
    else:
        start_date = START_DATE
    return start_date, end_date


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
            df = fetch_fn(*args, **kwargs)
            return df
        except Exception as e:
            last_err = e
            # backoff
            time.sleep(0.7 * (k + 1) + random.random() * 0.3)
    raise last_err


def _fetch_frames_for_ticker(ticker: str, start_date: str, end_date: str) -> list:
    frames = []

    # 1) 금액(value)
    for on in ONS:
        df = _fetch_with_retry(
            stock.get_market_trading_value_by_date,
            start_date, end_date, ticker,
            detail=True, on=on
        )
        df = _standardize_date_index(df)
        if df.empty:
            continue
        suffix = {"순매수": "value_net", "매수": "value_buy", "매도": "value_sell"}[on]
        df = _rename_with_suffix(df, suffix)
        frames.append(df)

    # 2) 수량(volume)
    for on in ONS:
        df = _fetch_with_retry(
            stock.get_market_trading_volume_by_date,
            start_date, end_date, ticker,
            detail=True, on=on
        )
        df = _standardize_date_index(df)
        if df.empty:
            continue
        suffix = {"순매수": "vol_net", "매수": "vol_buy", "매도": "vol_sell"}[on]
        df = _rename_with_suffix(df, suffix)
        frames.append(df)

    return frames


def fetch_one_ticker(ticker: str, start_date: str, end_date: str, out_dir: Path) -> str:
    out_base = _out_path_base(out_dir, ticker)

    try:
        # 백필 모드: 파일이 이미 있으면 스킵
        if (not INCREMENTAL_MODE) and _exists_any(out_base):
            return f"{ticker}: exists -> skip"

        # --- 1차 수집 ---
        frames = _fetch_frames_for_ticker(ticker, start_date, end_date)

        # --- no data면, 그 종목만 추가 재시도 ---
        if not frames:
            for n in range(1, NO_DATA_RETRY + 1):
                sleep_s = NO_DATA_SLEEP_BASE * n + random.random() * 1.5
                time.sleep(sleep_s)

                frames = _fetch_frames_for_ticker(ticker, start_date, end_date)
                if frames:
                    break

            if not frames:
                return f"{ticker}: no data"

        # merge
        df_all = pd.DataFrame()
        for df in frames:
            df_all = _safe_merge(df_all, df)

        if df_all is None or df_all.empty:
            # 이 경우도 no data로 처리(추가 재시도는 이미 끝난 상태)
            return f"{ticker}: no data"

        # 증분이면 기존 파일과 합치기
        if INCREMENTAL_MODE:
            df_old = _load_existing(out_base)
            if not df_old.empty:
                df_all = pd.concat([df_old, df_all], ignore_index=True)
                df_all = df_all.drop_duplicates(subset=["date"], keep="last")

        _save(df_all, out_base)
        return f"{ticker}: OK rows={len(df_all):,} cols={len(df_all.columns)}"

    except Exception as e:
        return f"{ticker}: ERROR ({e})"


def _write_list(path: Path, items: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    uniq = sorted(set(items))
    path.write_text("\n".join(uniq) + ("\n" if uniq else ""), encoding="utf-8")


def main():
    # flows 전용 workers 적용 (없으면 기존 MAX_WORKERS 사용)
    workers = MAX_WORKERS_FLOWS if MAX_WORKERS_FLOWS > 0 else MAX_WORKERS

    print("[fetch_krx_flows] start")
    print(f"  CWD={PROJECT_ROOT}")
    print(f"  INCREMENTAL_MODE={INCREMENTAL_MODE}, INCREMENTAL_DAYS={INCREMENTAL_DAYS}")
    print(f"  MAX_WORKERS(prices,legacy)={MAX_WORKERS}, MAX_WORKERS_FLOWS={MAX_WORKERS_FLOWS} -> using workers={workers}")
    print(f"  START_DATE={START_DATE}, SAVE_FORMAT={SAVE_FORMAT}, RETRY={RETRY}")
    print(f"  NO_DATA_RETRY={NO_DATA_RETRY}, NO_DATA_SLEEP_BASE={NO_DATA_SLEEP_BASE}, KRX_THROTTLE_SEC={KRX_THROTTLE_SEC}")

    master_path = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"master not found: {master_path}")

    df_master = pd.read_parquet(master_path)
    if "ticker" not in df_master.columns:
        raise RuntimeError("listings.parquet must contain 'ticker' column")

    tickers = df_master["ticker"].astype(str).tolist()
    start_date, end_date = _date_range()
    print(f"  range: {start_date} ~ {end_date} | tickers={len(tickers)}")

    out_dir = PROJECT_ROOT / "data/stocks/raw/krx_flows"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    no_data_tickers = []
    error_tickers = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_one_ticker, t, start_date, end_date, out_dir) for t in tickers]

        for i, fut in enumerate(as_completed(futures), 1):
            msg = fut.result()
            results.append(msg)

            # categorize
            if ": no data" in msg:
                no_data_tickers.append(msg.split(":")[0].strip())
            elif ": ERROR" in msg:
                error_tickers.append(msg.split(":")[0].strip())

            if i <= 20 or i % 200 == 0:
                print(f"  [{i}/{len(futures)}] {msg}")

    ok = sum(1 for r in results if ": OK" in r)
    skip = sum(1 for r in results if "skip" in r)
    nodata = sum(1 for r in results if ": no data" in r)
    err = sum(1 for r in results if ": ERROR" in r)

    # write ticker lists
    _write_list(out_dir / "_no_data_tickers.txt", no_data_tickers)
    _write_list(out_dir / "_error_tickers.txt", error_tickers)

    print(f"[fetch_krx_flows] done | OK={ok} SKIP={skip} NO_DATA={nodata} ERROR={err}")
    print(f"[fetch_krx_flows] wrote: {out_dir / '_no_data_tickers.txt'} ({len(set(no_data_tickers))} tickers)")
    print(f"[fetch_krx_flows] wrote: {out_dir / '_error_tickers.txt'} ({len(set(error_tickers))} tickers)")


if __name__ == "__main__":
    main()
