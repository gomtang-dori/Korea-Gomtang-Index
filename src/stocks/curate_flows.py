#!/usr/bin/env python3
"""
curate_flows.py
- raw/krx_flows 에 저장된 전투자자 수급 데이터에서
  대표 4개 투자자(외국인/기관합계/연기금/금융투자)만 표준화 + rolling 파생 생성
- 금액(value_net)을 주 시그널로, 수량(vol_net)은 보조로 포함

입력:
- data/stocks/raw/krx_flows/{ticker}.csv or .parquet

출력:
- data/stocks/curated/{ticker}/flows_daily.parquet

환경변수:
- FLOW_WINDOWS: "3,5,10,20" (기본)
- FLOW_INCLUDE_VOLUME: "true"/"false" (기본 true)
- FLOW_INCLUDE_BUYSELL: "true"/"false" (기본 false)  # 매수/매도까지 대표 4개로 포함할지
- FLOW_RAW_FORMAT: "auto"/"csv"/"parquet" (기본 auto)
"""

import os
import re
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()

FLOW_WINDOWS = os.getenv("FLOW_WINDOWS", "3,5,10,20")
WINDOWS = [int(x.strip()) for x in FLOW_WINDOWS.split(",") if x.strip()]

INCLUDE_VOLUME = os.getenv("FLOW_INCLUDE_VOLUME", "true").lower() == "true"
INCLUDE_BUYSELL = os.getenv("FLOW_INCLUDE_BUYSELL", "false").lower() == "true"
RAW_FORMAT = os.getenv("FLOW_RAW_FORMAT", "auto").lower()  # auto/csv/parquet

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
    """
    raw 파일 읽기 (parquet 우선 또는 auto)
    """
    p_parq = RAW_DIR / f"{ticker}.parquet"
    p_csv = RAW_DIR / f"{ticker}.csv"

    if RAW_FORMAT == "parquet":
        if p_parq.exists():
            return pd.read_parquet(p_parq)
        return pd.DataFrame()

    if RAW_FORMAT == "csv":
        if p_csv.exists():
            return pd.read_csv(p_csv, encoding="utf-8-sig")
        return pd.DataFrame()

    # auto: parquet 있으면 parquet, 없으면 csv
    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv, encoding="utf-8-sig")
    return pd.DataFrame()

def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    date 컬럼 표준화
    """
    if df.empty:
        return df
    out = df.copy()

    # 과거 버전 호환: "날짜"가 있으면 date로
    if "날짜" in out.columns and "date" not in out.columns:
        out.rename(columns={"날짜": "date"}, inplace=True)

    if "date" not in out.columns:
        # index가 날짜인 케이스 대응
        if out.index.name in ("date", "날짜"):
            out = out.reset_index()
            out.rename(columns={out.columns[0]: "date"}, inplace=True)
        else:
            # 최후: 첫 컬럼을 date로 가정(권장 X)
            out.rename(columns={out.columns[0]: "date"}, inplace=True)

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    return out

def _pick_col(df: pd.DataFrame, base_name: str, suffix: str) -> str | None:
    """
    base_name(예: '외국인합계') + suffix(예: 'value_net')에 해당하는 raw 컬럼 찾기.
    1) 정확히 '{base_name}_{suffix}' 있으면 선택
    2) 없으면 '{base_name}.*_{suffix}' 정규식으로 첫 매치
    """
    if df.empty:
        return None
    exact = f"{base_name}_{suffix}"
    if exact in df.columns:
        return exact

    # 유사 컬럼 탐색(예: '외국인합계(기타포함)_value_net' 같은 경우)
    pat = re.compile(rf"^{re.escape(base_name)}.*_{re.escape(suffix)}$")
    for c in df.columns:
        if pat.match(c):
            return c
    return None

def _add_rollings(out: pd.DataFrame, col: str):
    """
    rolling sum 파생 컬럼 추가
    """
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

    # 표준화 DF
    out = pd.DataFrame({"date": df_raw["date"]})

    # (1) 순매수(금액) - 핵심
    for key, base in TARGETS.items():
        src = _pick_col(df_raw, base, "value_net")
        dst = f"{key}_value_net"
        if src:
            out[dst] = df_raw[src]

    # (2) 순매수(수량) - 보조
    if INCLUDE_VOLUME:
        for key, base in TARGETS.items():
            src = _pick_col(df_raw, base, "vol_net")
            dst = f"{key}_vol_net"
            if src:
                out[dst] = df_raw[src]

    # (3) 매수/매도(금액/수량) - 옵션
    if INCLUDE_BUYSELL:
        for suffix in ["value_buy", "value_sell"]:
            for key, base in TARGETS.items():
                src = _pick_col(df_raw, base, suffix)
                dst = f"{key}_{suffix}"
                if src:
                    out[dst] = df_raw[src]
        if INCLUDE_VOLUME:
            for suffix in ["vol_buy", "vol_sell"]:
                for key, base in TARGETS.items():
                    src = _pick_col(df_raw, base, suffix)
                    dst = f"{key}_{suffix}"
                    if src:
                        out[dst] = df_raw[src]

    # rolling 파생: 금액 순매수는 반드시(있으면)
    for key in TARGETS.keys():
        _add_rollings(out, f"{key}_value_net")
    # 수량 순매수 rolling도 원하면(기본 true라 같이 생성)
    if INCLUDE_VOLUME:
        for key in TARGETS.keys():
            _add_rollings(out, f"{key}_vol_net")

    # 저장
    out_dir = PROJECT_ROOT / f"data/stocks/curated/{ticker}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "flows_daily.parquet"
    out.to_parquet(out_path, index=False)
    return True, f"OK rows={len(out):,} cols={len(out.columns)}"

def main():
    print("[curate_flows] start")
    print(f"  CWD={PROJECT_ROOT}")
    print(f"  RAW_DIR={RAW_DIR}")
    print(f"  WINDOWS={WINDOWS}, INCLUDE_VOLUME={INCLUDE_VOLUME}, INCLUDE_BUYSELL={INCLUDE_BUYSELL}, RAW_FORMAT={RAW_FORMAT}")

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
