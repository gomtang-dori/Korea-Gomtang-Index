# src/caches/cache_k200_close.py
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests


def _env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        if default is None:
            raise RuntimeError(f"Missing env: {name}")
        return default
    return v


def _as_yyyymmdd(d: pd.Timestamp) -> str:
    return d.strftime("%Y%m%d")


def fetch_index_ohlcv_krx_direct(start_yyyymmdd: str, end_yyyymmdd: str, idx_code: str) -> pd.DataFrame:
    """
    KRX 지수 OHLCV를 기간으로 반환하는 웹 엔드포인트를 직접 호출.
    pykrx 내부의 IndexTicker().get_name() (지수명 매핑) 단계를 우회하기 위함.
    """
    # KRX가 쓰는 지수 조회 엔드포인트(웹) - pykrx가 내부적으로 접근하는 계열을 직접 사용
    url = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://data.krx.co.kr/",
    }

    # NOTE: 이 bld 값은 KOSPI/KOSDAQ/기타 지수별로 다를 수 있습니다.
    # KOSPI200은 통상 '전체지수 > KOSPI200' 경로에서 동일 폼으로 내려옵니다.
    # 만약 아래 bld로 빈 값이 오면, (A) 제가 다음 단계에서 bld를 자동탐색하도록 바꿔드릴게요.
    data = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT00301",  # 지수 시세(일별) 계열에서 흔히 쓰이는 bld
        "locale": "ko_KR",
        "idxIndMidclssCd": "01",      # (대분류) KOSPI 계열로 맞추는 값(환경에 따라 필요/불필요)
        "idxIndCd": idx_code,         # 1028
        "strtDd": start_yyyymmdd,
        "endDd": end_yyyymmdd,
        "share": "1",
        "money": "1",
        "csvxls_isNo": "false",
    }

    r = requests.post(url, data=data, headers=headers, timeout=30)
    r.raise_for_status()
    j = r.json()

    # KRX JSON은 보통 "output" 키로 내려옵니다.
    rows = j.get("output") or j.get("Output") or []
    if not rows:
        raise RuntimeError(f"[cache_k200_close:direct] empty output (bld mismatch?) idx_code={idx_code} range={start_yyyymmdd}~{end_yyyymmdd}")

    df = pd.DataFrame(rows)
    return df


def main():
    out_path = Path(os.environ.get("K200_CACHE_PATH", "data/cache/k200_close.parquet"))
    idx_code = os.environ.get("K200_INDEX_CODE", "1028")  # KOSPI200

    backfill_start = pd.to_datetime(_env("BACKFILL_START"), format="%Y%m%d", errors="raise")
    backfill_end = pd.to_datetime(_env("BACKFILL_END"), format="%Y%m%d", errors="raise")
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))

    start = (backfill_start - pd.Timedelta(days=buffer_days)).normalize()
    end = backfill_end.normalize()

    start_s = _as_yyyymmdd(start)
    end_s = _as_yyyymmdd(end)

    print(f"[cache_k200_close:pykrx-bypass] fetch idx_code={idx_code} range={start_s}~{end_s}")

    # 1회 호출
    df = fetch_index_ohlcv_krx_direct(start_s, end_s, idx_code)

    # 컬럼명은 환경마다 한글/영문/약어로 달라질 수 있어 후보를 넓게 둠
    date_candidates = ["TRD_DD", "BAS_DD", "일자", "날짜", "date"]
    close_candidates = ["CLSPRC_IDX", "종가", "close", "종가_지수", "종가 "]

    date_col = next((c for c in date_candidates if c in df.columns), None)
    close_col = next((c for c in close_candidates if c in df.columns), None)
    if date_col is None or close_col is None:
        raise RuntimeError(f"[cache_k200_close:pykrx-bypass] cannot map columns. cols={list(df.columns)}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "k200_close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    ).dropna(subset=["date", "k200_close"]).sort_values("date").reset_index(drop=True)

    # 기존 파일과 upsert (중복 date는 최신으로 덮기)
    if out_path.exists():
        old = pd.read_parquet(out_path)
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
        old["k200_close"] = pd.to_numeric(old["k200_close"], errors="coerce")
        old = old.dropna(subset=["date", "k200_close"])
        merged = pd.concat([old, out], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    else:
        merged = out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"[cache_k200_close:pykrx-bypass] OK rows={len(merged)} -> {out_path}")


if __name__ == "__main__":
    main()
