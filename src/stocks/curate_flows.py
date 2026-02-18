#!/usr/bin/env python3
"""
curate_flows.py
- raw/krx_flows 에 저장된 전투자자 수급 데이터에서
  대표 4개 투자자(외국인합계/기관합계/연기금/금융투자)만 표준화 + rolling 파생 생성
- 금액(value_*)를 주 시그널로, 수량(vol_*)은 보조로 포함
- (요청) 매수/매도/순매수 포함

입력:
- data/stocks/raw/krx_flows/{ticker}.parquet (권장) or .csv

출력:
- data/stocks/curated/{ticker}/flows_daily.parquet

환경변수:
- FLOW_WINDOWS: "3,5,10,20" (기본)
- FLOW_INCLUDE_VOLUME: "true"/"false" (기본 true)
- FLOW_INCLUDE_BUYSELL: "true"/"false" (기본 true)  # 요청: true
- FLOW_RAW_FORMAT: "auto"/"csv"/"parquet" (기본 auto)
- FLOW_ROLL_ON_BUYSELL: "true"/"false" (기본 false) # 매수/매도에도 rolling 생성 여부
"""

import os
import re
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()

FLOW_WINDOWS = os.getenv("FLOW_WINDOWS", "3,5,10,20")
WINDOWS = [int(x.strip()) for x in FLOW_WINDOWS.split(",") if x.strip()]

INCLUDE_VOLUME = os.getenv("FLOW_INCLUDE_VOLUME", "true").lower() == "true"
INCLUDE_BUYSELL = os.getenv("FLOW_INCLUDE_BUYSELL", "true").lower() == "true"  # ✅ 요청 반영: default true
RAW_FORMAT = os.getenv("FLOW_RAW_FORMAT", "auto").lower()
ROLL_ON_BUYSELL = os.getenv("FLOW_ROLL_ON_BUYSELL", "false").lower() == "true"

RAW_DIR = PROJECT_ROOT / "data/stocks/raw/krx_flows"
MASTER_PATH = PROJECT_ROOT / "data/stocks/master/listings.parquet"

# 대표 투자자 표준 컬럼명
TARGETS = {
    "foreign": "외국인합계",
    "inst": "기관합계",
    "pension": "연기금",
    "fininv": "금융투자",
}

def _read_raw_flow(ticker: str) -> pd.DataFrame:
    p_parq = RAW_DIR / f"{ticker}.parquet"
    p_csv = RAW_DIR / f"{ticker}.csv"

    if RAW_FORMAT == "parquet":
        return pd.read_parquet(p_parq) if p_parq.exists() else pd.DataFrame()
    if RAW_FORMAT == "csv":
        return pd.read_csv(p_csv, encoding="utf-8-sig") if p_csv.exists() else pd.DataFrame()

    # auto
    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv, encoding="utf-8-sig")
    return pd.DataFrame()

def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    if "날짜" in out.columns and "date" not in out.columns:
        out.rename(columns={"날짜": "date"}, inplace=True)

    if "date" not in out.columns:
        if out.index.name in ("date", "날짜"):
            out = out.reset_index()
            out.rename(columns={out.columns[0]: "date"}, inplace=True)
        else:
            out.rename(columns={out.columns[0]: "date"}, inplace=True)

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    return out

def _pick_col(df: pd.DataFrame, base_name: str, suffix: str) -> str | None:
    if df.empty:
        return None
    exact = f"{base_name}_{suffix}"
    if exact in df.columns:
        return exact

    pat = re.compile(rf"^{re.escape(base_name)}.*_{re.escape(suffix)}$")
    for c in df.columns:
        if pat.match(c):
            return c
    return None

def _add_rollings(out: pd.DataFrame, col: str):
    if col not in out.columns:
        return
    for w in WINDOWS:
        out[f"{col}_{w}d_sum"] = out[col].rolling(window=w, min_periods=1).sum()

def curate_one_ticker(ticker: str) -> tuple[bool, str]:
    df_raw = _read_raw_flow(ticker)
    if df_raw.empty:
        return False, "raw 없음"

    df_raw = _ensure_date(df_raw)
    if df_raw.empty or "date" not in df_raw.columns:
        return False, "date 파싱 실패"

    out = pd.DataFrame({"date": df_raw["date"]})

    # (A) 순매수(금액/수량) - 핵심
    for key, base in TARGETS.items():
        c = _pick_col(df_raw, base, "value_net")
        if c: out[f"{key}_value_net"] = df_raw[c]

    if INCLUDE_VOLUME:
        for key, base in TARGETS.items():
            c = _pick_col(df_raw, base, "vol_net")
            if c: out[f"{key}_vol_net"] = df_raw[c]

    # (B) 매수/매도(금액/수량) - 요청 반영
    if INCLUDE_BUYSELL:
        for key, base in TARGETS.items():
            c = _pick_col(df_raw, base, "value_buy")
            if c: out[f"{key}_value_buy"] = df_raw[c]
            c = _pick_col(df_raw, base, "value_sell")
            if c: out[f"{key}_value_sell"] = df_raw[c]

        if INCLUDE_VOLUME:
            for key, base in TARGETS.items():
                c = _pick_col(df_raw, base, "vol_buy")
                if c: out[f"{key}_vol_buy"] = df_raw[c]
                c = _pick_col(df_raw, base, "vol_sell")
                if c: out[f"{key}_vol_sell"] = df_raw[c]

    # (C) rolling 파생: 기본은 순매수(net)만
    for key in TARGETS.keys():
        _add_rollings(out, f"{key}_value_net")
    if INCLUDE_VOLUME:
        for key in TARGETS.keys():
            _add_rollings(out, f"{key}_vol_net")

    # (옵션) 매수/매도에도 rolling 원하면
    if INCLUDE_BUYSELL and ROLL_ON_BUYSELL:
        for key in TARGETS.keys():
            _add_rollings(out, f"{key}_value_buy")
            _add_rollings(out, f"{key}_value_sell")
        if INCLUDE_VOLUME:
            for key in TARGETS.keys():
                _add_rollings(out, f"{key}_vol_buy")
                _add_rollings(out, f"{key}_vol_sell")

    # 저장
    out_dir = PROJECT_ROOT / f"data/stocks/curated/{ticker}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "flows_daily.parquet"
    out.to_parquet(out_path, index=False)
    return True, f"OK rows={len(out):,} cols={len(out.columns)}"

def main():
    print("[curate_flows] start")
    print(f"  CWD={PROJECT_ROOT}")
    print(f"  WINDOWS={WINDOWS}, INCLUDE_VOLUME={INCLUDE_VOLUME}, INCLUDE_BUYSELL={INCLUDE_BUYSELL}, RAW_FORMAT={RAW_FORMAT}, ROLL_ON_BUYSELL={ROLL_ON_BUYSELL}")

    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"master not found: {MASTER_PATH}")

    df_master = pd.read_parquet(MASTER_PATH)
    tickers = df_master["ticker"].astype(str).tolist()

    ok = 0
    fail = 0
    for i, t in enumerate(tickers, 1):
        success, msg = curate_one_ticker(t)
        if success:
            ok += 1
            if i <= 20 or i % 300 == 0:
                print(f"  [{i}/{len(tickers)}] {t}: {msg}")
        else:
            fail += 1
            if i <= 20 or i % 300 == 0:
                print(f"  [{i}/{len(tickers)}] {t}: FAIL ({msg})")

    print(f"[curate_flows] done | OK={ok} FAIL={fail}")

if __name__ == "__main__":
    main()
