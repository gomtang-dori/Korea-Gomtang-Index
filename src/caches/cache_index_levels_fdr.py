# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

import FinanceDataReader as fdr


def _env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        if default is None:
            raise RuntimeError(f"Missing env: {name}")
        return default
    return v


def _parse_yyyymmdd(s: str) -> pd.Timestamp:
    ts = pd.to_datetime(s, format="%Y%m%d", errors="raise")
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _safe_end(backfill_end: pd.Timestamp) -> pd.Timestamp:
    today_utc = pd.Timestamp.utcnow().tz_localize(None).normalize()
    safe_end = today_utc - pd.Timedelta(days=1)
    return min(backfill_end, safe_end)


def _upsert_timeseries(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        out = new.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in ["kospi_close", "kosdaq_close", "k200_close"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _fetch_close(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = fdr.DataReader(symbol, start, end)
    if df is None or df.empty:
        raise RuntimeError(f"[index_levels_fdr] empty result symbol={symbol} range={start}~{end}")
    if "Close" not in df.columns:
        raise RuntimeError(f"[index_levels_fdr] missing Close column symbol={symbol} cols={list(df.columns)}")

    out = df.reset_index().rename(columns={"Date": "date", "Close": "close"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return out[["date", "close"]]


def main():
    # outputs
    out_path = Path(os.environ.get("INDEX_LEVELS_CACHE_PATH", "data/cache/index_levels.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    k200_out_path = Path(os.environ.get("K200_CACHE_PATH", "data/cache/k200_close.parquet"))
    k200_out_path.parent.mkdir(parents=True, exist_ok=True)

    # symbols
    kospi_symbol = os.environ.get("FDR_KOSPI_SYMBOL", "KS11")
    kosdaq_symbol = os.environ.get("FDR_KOSDAQ_SYMBOL", "KQ11")
    k200_symbol = os.environ.get("FDR_K200_SYMBOL", "KS200")

    # range
    backfill_start = _parse_yyyymmdd(_env("BACKFILL_START"))
    backfill_end = _safe_end(_parse_yyyymmdd(_env("BACKFILL_END")))
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))

    start = (backfill_start - pd.Timedelta(days=buffer_days)).normalize()
    end = backfill_end.normalize()

    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    print(f"[index_levels_fdr] range={start:%Y%m%d}~{end:%Y%m%d} out={out_path}")
    print(f"[index_levels_fdr] symbols kospi={kospi_symbol} kosdaq={kosdaq_symbol} k200={k200_symbol}")
    print(f"[index_levels_fdr] also_write_k200_cache={k200_out_path}")

    kospi = _fetch_close(kospi_symbol, start_s, end_s).rename(columns={"close": "kospi_close"})
    kosdaq = _fetch_close(kosdaq_symbol, start_s, end_s).rename(columns={"close": "kosdaq_close"})
    k200 = _fetch_close(k200_symbol, start_s, end_s).rename(columns={"close": "k200_close"})

    merged = kospi.merge(kosdaq, on="date", how="outer").merge(k200, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)

    # upsert into index_levels.parquet
    old = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(
        columns=["date", "kospi_close", "kosdaq_close", "k200_close"]
    )
    out = _upsert_timeseries(old, merged)

    out.to_parquet(out_path, index=False)
    out.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(f"[index_levels_fdr] OK rows(total)={len(out)} -> {out_path}")
    print(out.tail(3).to_string(index=False))

    # write compatibility k200 cache for f01_momentum.py
    k200_cache = out[["date", "k200_close"]].dropna(subset=["k200_close"]).copy()
    old_k200 = pd.read_parquet(k200_out_path) if k200_out_path.exists() else pd.DataFrame(columns=["date", "k200_close"])
    k200_merged = _upsert_timeseries(old_k200, k200_cache)

    k200_merged.to_parquet(k200_out_path, index=False)
    k200_merged.to_csv(k200_out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(f"[index_levels_fdr] OK k200 rows(total)={len(k200_merged)} -> {k200_out_path}")
    print(k200_merged.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
