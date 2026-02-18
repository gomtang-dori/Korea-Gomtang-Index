#!/usr/bin/env python3
"""
KRX flows 2차 재시도 전용 스크립트
- data/stocks/raw/krx_flows/_no_data_tickers.txt 를 읽어서 해당 티커만 재시도
- 성공하면 raw/krx_flows/<ticker>.(parquet|csv) 생성/갱신
- 결과 파일:
  - _no_data_tickers_retry2.txt  (2차에도 no data)
  - _error_tickers_retry2.txt    (2차에서 error)
  - _ok_tickers_retry2.txt       (2차에서 OK)
  - _retry2_summary.txt

환경변수(기본 fetch_krx_flows.py와 최대한 호환)
- INCREMENTAL_MODE: true/false (기본 false)
- INCREMENTAL_DAYS: (기본 5)
- START_DATE: (기본 20150101)
- KRX_FLOWS_SAVE_FORMAT: csv|parquet (기본 parquet 권장)
- KRX_RETRY: 예외 재시도 횟수 (기본 3)

Retry2 전용
- MAX_WORKERS_FLOWS_RETRY: 병렬 worker (기본 3)  <-- 안전하게 낮게
- KRX_NO_DATA_RETRY_RETRY2: no data 재시도 횟수 (기본 3)
- KRX_NO_DATA_SLEEP_BASE_RETRY2: no data 재시도 대기 base (기본 5.0초)
- KRX_THROTTLE_SEC_RETRY2: 요청 간 최소 간격(초) (기본 0.2초)  <-- 차단 방지용 권장
- RETRY2_OVERWRITE_IF_EXISTS: true면 기존 파일이 있어도 다시 덮어씀(기본 false)

옵션
- RETRY2_INPUT_FILE: 기본 _no_data_tickers.txt 대신 다른 입력 파일 사용 가능
"""

import os
import time
import random
import threading
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from pykrx import stock


# ----------------------
# Config
# ----------------------
PROJECT_ROOT = Path.cwd()

INCREMENTAL_MODE = os.getenv("INCREMENTAL_MODE", "false").lower() == "true"
INCREMENTAL_DAYS = int(os.getenv("INCREMENTAL_DAYS", "5"))
START_DATE = os.getenv("START_DATE", "20150101")

SAVE_FORMAT = os.getenv("KRX_FLOWS_SAVE_FORMAT", "parquet").lower()   # retry2는 parquet 권장
RETRY = int(os.getenv("KRX_RETRY", "3"))

MAX_WORKERS_RETRY2 = int(os.getenv("MAX_WORKERS_FLOWS_RETRY", "3"))

NO_DATA_RETRY_RETRY2 = int(os.getenv("KRX_NO_DATA_RETRY_RETRY2", "3"))
NO_DATA_SLEEP_BASE_RETRY2 = float(os.getenv("KRX_NO_DATA_SLEEP_BASE_RETRY2", "5.0"))

KRX_THROTTLE_SEC_RETRY2 = float(os.getenv("KRX_THROTTLE_SEC_RETRY2", "0.2").strip() or "0.2")

RETRY2_OVERWRITE_IF_EXISTS = os.getenv("RETRY2_OVERWRITE_IF_EXISTS", "false").lower() == "true"

RETRY2_INPUT_FILE = os.getenv(
    "RETRY2_INPUT_FILE",
    str(PROJECT_ROOT / "data/stocks/raw/krx_flows/_no_data_tickers.txt")
)

OUT_DIR = PROJECT_ROOT / "data/stocks/raw/krx_flows"

ONS = ["순매수", "매수", "매도"]


# ----------------------
# Throttle (global)
# ----------------------
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


throttle = GlobalThrottle(KRX_THROTTLE_SEC_RETRY2)


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


def _out_base(ticker: str) -> Path:
    return OUT_DIR / ticker


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

    # 금액(value)
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

    # 수량(volume)
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


def fetch_retry2_one(ticker: str, start_date: str, end_date: str) -> str:
    out_base = _out_base(ticker)

    # 이미 파일이 존재하고, overwrite가 false이면 스킵(단, 기존이 빈 파일/깨진 파일일 수 있으면 overwrite true 추천)
    if _exists_any(out_base) and not RETRY2_OVERWRITE_IF_EXISTS:
        # 그래도 실제 row가 0이면 재시도하도록 처리
        df_old = _load_existing(out_base)
        if not df_old.empty and len(df_old) > 10:
            return f"{ticker}: exists -> skip"
        # old가 비정상으로 보이면 계속 진행

    try:
        frames = _fetch_frames_for_ticker(ticker, start_date, end_date)

        if not frames:
            # 2차 no data 재시도 (ticker only)
            for n in range(1, NO_DATA_RETRY_RETRY2 + 1):
                sleep_s = NO_DATA_SLEEP_BASE_RETRY2 * n + random.random() * 2.0
                time.sleep(sleep_s)
                frames = _fetch_frames_for_ticker(ticker, start_date, end_date)
                if frames:
                    break

        if not frames:
            return f"{ticker}: no data"

        df_all = pd.DataFrame()
        for df in frames:
            df_all = _safe_merge(df_all, df)

        if df_all is None or df_all.empty:
            return f"{ticker}: no data"

        # 증분 모드면 기존과 합치기
        if INCREMENTAL_MODE:
            df_old = _load_existing(out_base)
            if not df_old.empty:
                df_all = pd.concat([df_old, df_all], ignore_index=True)
                df_all = df_all.drop_duplicates(subset=["date"], keep="last")

        _save(df_all, out_base)
        return f"{ticker}: OK rows={len(df_all):,} cols={len(df_all.columns)}"

    except Exception as e:
        return f"{ticker}: ERROR ({e})"


def _read_tickers_from_file(path: Path) -> list:
    if not path.exists():
        return []
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines()]
    return [x for x in lines if x]


def _write_list(path: Path, items: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    uniq = sorted(set(items))
    path.write_text("\n".join(uniq) + ("\n" if uniq else ""), encoding="utf-8")


def main():
    print("[fetch_krx_flows_retry_nodata] start")
    print(f"  CWD={PROJECT_ROOT}")
    print(f"  INCREMENTAL_MODE={INCREMENTAL_MODE}, INCREMENTAL_DAYS={INCREMENTAL_DAYS}")
    print(f"  START_DATE={START_DATE}, SAVE_FORMAT={SAVE_FORMAT}, RETRY={RETRY}")
    print(f"  workers={MAX_WORKERS_RETRY2}, throttle={KRX_THROTTLE_SEC_RETRY2}s")
    print(f"  no_data_retry_retry2={NO_DATA_RETRY_RETRY2}, no_data_sleep_base_retry2={NO_DATA_SLEEP_BASE_RETRY2}")
    print(f"  overwrite_if_exists={RETRY2_OVERWRITE_IF_EXISTS}")
    print(f"  input_file={RETRY2_INPUT_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    in_path = Path(RETRY2_INPUT_FILE)
    tickers = _read_tickers_from_file(in_path)
    tickers = sorted(set([str(t).strip() for t in tickers if str(t).strip()]))

    if not tickers:
        print("  input tickers empty -> nothing to do")
        return

    start_date, end_date = _date_range()
    print(f"  range: {start_date} ~ {end_date} | retry tickers={len(tickers)}")

    ok_tickers = []
    no_data_tickers = []
    error_tickers = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_RETRY2) as ex:
        futures = {ex.submit(fetch_retry2_one, t, start_date, end_date): t for t in tickers}

        for i, fut in enumerate(as_completed(futures), 1):
            msg = fut.result()
            t = futures[fut]

            if ": OK" in msg:
                ok_tickers.append(t)
            elif ": no data" in msg:
                no_data_tickers.append(t)
            elif ": ERROR" in msg:
                error_tickers.append(t)

            if i <= 20 or i % 50 == 0:
                print(f"  [{i}/{len(tickers)}] {msg}")

    # outputs
    _write_list(OUT_DIR / "_ok_tickers_retry2.txt", ok_tickers)
    _write_list(OUT_DIR / "_no_data_tickers_retry2.txt", no_data_tickers)
    _write_list(OUT_DIR / "_error_tickers_retry2.txt", error_tickers)

    summary = (
        f"retry2 done\n"
        f"- input_tickers={len(tickers)}\n"
        f"- ok={len(set(ok_tickers))}\n"
        f"- no_data={len(set(no_data_tickers))}\n"
        f"- error={len(set(error_tickers))}\n"
        f"- range={start_date}~{end_date}\n"
        f"- throttle={KRX_THROTTLE_SEC_RETRY2}\n"
        f"- workers={MAX_WORKERS_RETRY2}\n"
        f"- overwrite_if_exists={RETRY2_OVERWRITE_IF_EXISTS}\n"
    )
    (OUT_DIR / "_retry2_summary.txt").write_text(summary, encoding="utf-8")

    print("[fetch_krx_flows_retry_nodata] DONE")
    print(summary.strip())
    print(f"  wrote: {OUT_DIR / '_ok_tickers_retry2.txt'}")
    print(f"  wrote: {OUT_DIR / '_no_data_tickers_retry2.txt'}")
    print(f"  wrote: {OUT_DIR / '_error_tickers_retry2.txt'}")
    print(f"  wrote: {OUT_DIR / '_retry2_summary.txt'}")


if __name__ == "__main__":
    main()
