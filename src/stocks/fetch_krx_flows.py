#!/usr/bin/env python3
"""
KRX 투자자별 매매(전 투자자 컬럼) 백필/증분 수집
- 전 투자자 컬럼을 "필터링 없이" 모두 저장
- 매수/매도/순매수 (on: '매수','매도','순매수')
- 금액/수량 모두 수집:
  - get_market_trading_value_by_date
  - get_market_trading_volume_by_date

환경변수:
- INCREMENTAL_MODE: true/false (기본 false)
- INCREMENTAL_DAYS: 증분일수 (기본 5)
- MAX_WORKERS: 병렬 worker 수 (기본 10)
- START_DATE: 백필 시작일 (기본 20150101)
- KRX_FLOWS_SAVE_FORMAT: csv or parquet (기본 csv)  # csv가 크면 parquet 추천
- KRX_RETRY: 재시도 횟수 (기본 3)
"""

import os
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from pykrx import stock
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== Config =====
INCREMENTAL_MODE = os.getenv("INCREMENTAL_MODE", "false").lower() == "true"
INCREMENTAL_DAYS = int(os.getenv("INCREMENTAL_DAYS", "5"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
START_DATE = os.getenv("START_DATE", "20150101")   # 요청하신 2015~ 기준으로 기본값 변경
SAVE_FORMAT = os.getenv("KRX_FLOWS_SAVE_FORMAT", "csv").lower()  # csv/parquet
RETRY = int(os.getenv("KRX_RETRY", "3"))

ONS = ["순매수", "매수", "매도"]

PROJECT_ROOT = Path.cwd()

def _date_range():
    end_date = datetime.now().strftime("%Y%m%d")
    if INCREMENTAL_MODE:
        start_date = (datetime.now() - timedelta(days=INCREMENTAL_DAYS)).strftime("%Y%m%d")
    else:
        start_date = START_DATE
    return start_date, end_date

def _standardize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    PyKRX 반환 DF는 보통 index가 날짜(YYYY-MM-DD) 형태.
    이를 date 컬럼으로 표준화.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index.name = "date"
    out.reset_index(inplace=True)
    out["date"] = pd.to_datetime(out["date"])
    return out

def _rename_with_suffix(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    모든 투자자 컬럼에 suffix를 붙여 충돌 방지.
    예: '외국인합계' -> '외국인합계_value_net'
    """
    if df.empty:
        return df
    rename = {c: f"{c}_{suffix}" for c in df.columns if c != "date"}
    return df.rename(columns=rename)

def _safe_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left.empty:
        return right
    if right.empty:
        return left
    return left.merge(right, on="date", how="outer")

def _load_existing(out_path: Path) -> pd.DataFrame:
    if not out_path.exists():
        return pd.DataFrame()
    try:
        if out_path.suffix.lower() == ".parquet":
            df_old = pd.read_parquet(out_path)
        else:
            df_old = pd.read_csv(out_path, encoding="utf-8-sig")
        if "date" in df_old.columns:
            df_old["date"] = pd.to_datetime(df_old["date"])
        return df_old
    except Exception:
        return pd.DataFrame()

def _save(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("date")
    if SAVE_FORMAT == "parquet":
        df.to_parquet(out_path.with_suffix(".parquet"), index=False)
        if out_path.with_suffix(".csv").exists():
            # 혼재 방지(선택): 이전 csv가 있다면 유지/삭제는 팀 정책에 맞게
            pass
    else:
        df.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

def _fetch_with_retry(fetch_fn, *args, **kwargs) -> pd.DataFrame:
    last_err = None
    for k in range(RETRY):
        try:
            df = fetch_fn(*args, **kwargs)
            return df
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (k + 1))  # simple backoff
    raise last_err

def fetch_one_ticker(ticker: str, start_date: str, end_date: str, out_dir: Path) -> str:
    """
    ticker 하나에 대해:
    - value: net/buy/sell
    - volume: net/buy/sell
    전부 가져와서 date로 outer-merge 후 저장
    """
    out_path_base = out_dir / ticker  # 확장자는 save에서 결정
    try:
        # 백필 모드에서 파일이 이미 있으면 스킵(원하면 재생성 옵션도 가능)
        if (not INCREMENTAL_MODE) and (out_path_base.with_suffix(".csv").exists() or out_path_base.with_suffix(".parquet").exists()):
            return f"{ticker}: exists -> skip"

        frames = []

        # 1) 금액(value)
        for on in ONS:
            df = _fetch_with_retry(stock.get_market_trading_value_by_date,
                                   start_date, end_date, ticker, detail=True, on=on)
            df = _standardize_date_index(df)
            if df.empty:
                continue
            suffix = {"순매수": "value_net", "매수": "value_buy", "매도": "value_sell"}[on]
            df = _rename_with_suffix(df, suffix)
            frames.append(df)

        # 2) 수량(volume)
        for on in ONS:
            df = _fetch_with_retry(stock.get_market_trading_volume_by_date,
                                   start_date, end_date, ticker, detail=True, on=on)
            df = _standardize_date_index(df)
            if df.empty:
                continue
            suffix = {"순매수": "vol_net", "매수": "vol_buy", "매도": "vol_sell"}[on]
            df = _rename_with_suffix(df, suffix)
            frames.append(df)

        if not frames:
            return f"{ticker}: no data"

        df_all = pd.DataFrame()
        for df in frames:
            df_all = _safe_merge(df_all, df)

        # 증분이면 기존 파일과 합치기
        if INCREMENTAL_MODE:
            df_old = _load_existing(out_path_base.with_suffix(".parquet" if SAVE_FORMAT == "parquet" else ".csv"))
            if not df_old.empty:
                df_all = pd.concat([df_old, df_all], ignore_index=True)
                df_all = df_all.drop_duplicates(subset=["date"], keep="last")

        # 저장
        _save(df_all, out_path_base)
        return f"{ticker}: OK rows={len(df_all):,} cols={len(df_all.columns)}"

    except Exception as e:
        return f"{ticker}: ERROR ({e})"

def main():
    print("[fetch_krx_flows] start")
    print(f"  CWD={PROJECT_ROOT}")
    print(f"  INCREMENTAL_MODE={INCREMENTAL_MODE}, INCREMENTAL_DAYS={INCREMENTAL_DAYS}, MAX_WORKERS={MAX_WORKERS}")
    print(f"  START_DATE={START_DATE}, SAVE_FORMAT={SAVE_FORMAT}, RETRY={RETRY}")

    master_path = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"master not found: {master_path}")

    df_master = pd.read_parquet(master_path)
    start_date, end_date = _date_range()
    print(f"  range: {start_date} ~ {end_date} | tickers={len(df_master)}")

    out_dir = PROJECT_ROOT / "data/stocks/raw/krx_flows"
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = df_master["ticker"].astype(str).tolist()

    # 병렬 실행
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fetch_one_ticker, t, start_date, end_date, out_dir) for t in tickers]
        for i, fut in enumerate(as_completed(futures), 1):
            msg = fut.result()
            results.append(msg)
            if i <= 20 or i % 200 == 0:
                print(f"  [{i}/{len(futures)}] {msg}")

    ok = sum(1 for r in results if "OK" in r)
    skip = sum(1 for r in results if "skip" in r)
    err = sum(1 for r in results if "ERROR" in r)
    print(f"[fetch_krx_flows] done | OK={ok} SKIP={skip} ERROR={err}")

if __name__ == "__main__":
    main()
