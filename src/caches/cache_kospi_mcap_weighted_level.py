# src/caches/cache_kospi_mcap_weighted_level.py
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


def main():
    out_path = Path(os.environ.get("KOSPI_LEVEL_CACHE_PATH", "data/cache/kospi_mcap_weighted_level.parquet"))

    backfill_start = pd.to_datetime(_env("BACKFILL_START"), format="%Y%m%d", errors="raise").normalize()
    backfill_end = pd.to_datetime(_env("BACKFILL_END"), format="%Y%m%d", errors="raise").normalize()
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))

    start = (backfill_start - pd.Timedelta(days=buffer_days)).normalize()
    end = backfill_end.normalize()

    # 날짜 목록(캘린더 일자). 실제 데이터 없는 날은 스킵될 것
    days = pd.date_range(start, end, freq="D")

    rows = []
    miss = 0

    for d in days:
        ymd = d.strftime("%Y%m%d")

        # 1) KOSPI 전체 종목 종가
        try:
            ohlcv = stock.get_market_ohlcv(ymd, market="KOSPI")
        except Exception:
            ohlcv = None
        if ohlcv is None or len(ohlcv) == 0:
            miss += 1
            continue

        # 2) KOSPI 전체 종목 시총
        try:
            cap = stock.get_market_cap(ymd, market="KOSPI")
        except Exception:
            cap = None
        if cap is None or len(cap) == 0:
            miss += 1
            continue

        # 인덱스는 티커
        o = ohlcv.copy()
        c = cap.copy()

        # 종가 컬럼(한글) 방어
        close_col = "종가" if "종가" in o.columns else None
        mcap_col = "시가총액" if "시가총액" in c.columns else None
        if close_col is None or mcap_col is None:
            miss += 1
            continue

        df = (
            pd.DataFrame({"close": o[close_col]})
            .join(pd.DataFrame({"mcap": c[mcap_col]}), how="inner")
            .dropna()
        )
        # 숫자화
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["mcap"] = pd.to_numeric(df["mcap"], errors="coerce")
        df = df.dropna()
        df = df[(df["mcap"] > 0) & (df["close"] > 0)]

        if len(df) == 0:
            miss += 1
            continue

        level = float((df["close"] * df["mcap"]).sum() / df["mcap"].sum())

        rows.append({"date": pd.to_datetime(ymd), "kospi_level": level})

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    if len(out) == 0:
        raise RuntimeError("[cache_kospi_level] empty output (no trading days parsed)")

    # upsert
    if out_path.exists():
        old = pd.read_parquet(out_path)
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
        old["kospi_level"] = pd.to_numeric(old["kospi_level"], errors="coerce")
        old = old.dropna()
        merged = pd.concat([old, out], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    else:
        merged = out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"[cache_kospi_level] OK rows={len(merged)} miss_days={miss} -> {out_path}")


if __name__ == "__main__":
    main()
