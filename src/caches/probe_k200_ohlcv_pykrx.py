# src/caches/probe_k200_ohlcv_pykrx.py
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


def _as_yyyymmdd(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d")


def _safe_patch_index_name():
    # pykrx 내부 지수명 매핑 KeyError('지수명') 방어
    try:
        import pykrx.stock.stock_api as stock_api
        _orig = stock_api.get_index_ticker_name

        def _safe_get_index_ticker_name(ticker: str) -> str:
            try:
                return _orig(ticker)
            except Exception:
                return f"INDEX_{ticker}"

        stock_api.get_index_ticker_name = _safe_get_index_ticker_name
        print("[probe_k200_ohlcv_pykrx] patched get_index_ticker_name()")
    except Exception as e:
        print(f"[probe_k200_ohlcv_pykrx] WARN patch failed: {type(e).__name__}: {e}")


def main():
    index_code = _env("K200_INDEX_CODE", "1028")
    days = int(_env("DAYS", "7"))
    out_path = Path(os.environ.get("OUT_PATH", "data/cache/probe_k200_ohlcv_pykrx.parquet"))

    # ✅ 핵심: 러너 시간(utcnow) 쓰지 말고, workflow에서 내려준 날짜를 사용
    backfill_end = pd.to_datetime(_env("BACKFILL_END"), format="%Y%m%d", errors="raise").normalize()
    # 주말/휴일 감안해서 넉넉히 30일 전부터 긁고 마지막 7개만 자름
    start = (backfill_end - pd.Timedelta(days=45)).normalize()

    start_s = _as_yyyymmdd(start)
    end_s = _as_yyyymmdd(backfill_end)

    print(f"[probe_k200_ohlcv_pykrx] fetch index_code={index_code} range={start_s}~{end_s} (tail {days})")

    _safe_patch_index_name()

    df = stock.get_index_ohlcv(start_s, end_s, index_code)
    if df is None or len(df) == 0:
        raise RuntimeError(f"[probe_k200_ohlcv_pykrx] empty result index_code={index_code} range={start_s}~{end_s}")

    df = df.tail(days).copy()
    df = df.reset_index()
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    print(f"[probe_k200_ohlcv_pykrx] cols={list(df.columns)}")
    print(df.to_string(index=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    df.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    print(f"[probe_k200_ohlcv_pykrx] OK rows={len(df)} -> {out_path}")


if __name__ == "__main__":
    main()
