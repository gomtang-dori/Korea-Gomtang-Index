# src/caches/cache_k200_close.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from pykrx import stock


def _env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        if default is None:
            raise RuntimeError(f"Missing env: {name}")
        return default
    return v


def _as_yyyymmdd(d: pd.Timestamp) -> str:
    return d.strftime("%Y%m%d")


def main():
    out_path = Path(os.environ.get("K200_CACHE_PATH", "data/cache/k200_close.parquet"))
    index_code = os.environ.get("K200_INDEX_CODE", "1028")  # KOSPI200

    backfill_start = pd.to_datetime(_env("BACKFILL_START"), format="%Y%m%d", errors="raise")
    backfill_end = pd.to_datetime(_env("BACKFILL_END"), format="%Y%m%d", errors="raise")
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))

    start = (backfill_start - pd.Timedelta(days=buffer_days)).normalize()
    end = backfill_end.normalize()

    start_s = _as_yyyymmdd(start)
    end_s = _as_yyyymmdd(end)

    print(f"[cache_k200_close:pykrx] fetch index_code={index_code} range={start_s}~{end_s}")

    # 기간 한 번에 조회 (핵심: 일별 호출 제거)
    df = stock.get_index_ohlcv(start_s, end_s, index_code)
    if df is None or len(df) == 0:
        raise RuntimeError(f"[cache_k200_close:pykrx] empty result index_code={index_code} range={start_s}~{end_s}")

    df = df.reset_index()

    # pykrx 컬럼은 보통: 날짜, 시가, 고가, 저가, 종가, 거래량
    # 실제 컬럼명은 한글일 수 있어서 '종가' 우선 탐색, 없으면 Close/종가 유사 컬럼 탐색
    date_col = df.columns[0]
    close_candidates = ["종가", "Close", "종가가", "종가 "]
    close_col = None
    for c in close_candidates:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        # fallback: 가장 그럴듯한 수치형 컬럼을 찾되, 실패 시 바로 에러로 fail-fast
        raise RuntimeError(f"[cache_k200_close:pykrx] cannot find close column in {list(df.columns)}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "k200_close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    ).dropna(subset=["date", "k200_close"]).sort_values("date").reset_index(drop=True)

    # 기존 파일과 upsert (중복 date는 최신으로 덮어쓰기)
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
    print(f"[cache_k200_close:pykrx] OK rows={len(merged)} -> {out_path}")


if __name__ == "__main__":
    main()
