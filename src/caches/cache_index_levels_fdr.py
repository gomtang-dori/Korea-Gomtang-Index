# src/caches/cache_index_levels_fdr.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import FinanceDataReader as fdr


def _env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        if default is None:
            raise RuntimeError(f"Missing env: {name}")
        return default
    return str(v)


def _parse_yyyymmdd(s: str) -> pd.Timestamp:
    ts = pd.to_datetime(s, format="%Y%m%d", errors="raise")
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _upsert_ts(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        out = new.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return out


def _fetch_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = fdr.DataReader(symbol, str(start.date()), str(end.date()))
    if df is None or len(df) == 0:
        raise RuntimeError(f"[cache_index_levels_fdr] empty result symbol={symbol} range={start.date()}~{end.date()}")

    if "Close" not in df.columns:
        raise RuntimeError(f"[cache_index_levels_fdr] missing Close symbol={symbol} cols={list(df.columns)}")

    out = df.reset_index().rename(columns={"Date": "date"})
    if "date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["date", "close"])[["date", "close"]].sort_values("date").reset_index(drop=True)
    return out


def main():
    backfill_start = _parse_yyyymmdd(_env("BACKFILL_START"))
    backfill_end = _parse_yyyymmdd(_env("BACKFILL_END"))
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))

    start = (backfill_start - pd.Timedelta(days=buffer_days)).normalize()
    end = backfill_end.normalize()

    kospi_symbol = os.environ.get("KOSPI_SYMBOL", "KS11")
    kosdaq_symbol = os.environ.get("KOSDAQ_SYMBOL", "KQ11")
    k200_symbol = os.environ.get("K200_SYMBOL", "KS200")

    levels_path = Path(os.environ.get("INDEX_LEVELS_PATH", "data/cache/index_levels.parquet"))
    k200_cache_path = Path(os.environ.get("K200_CACHE_PATH", "data/cache/k200_close.parquet"))

    levels_path.parent.mkdir(parents=True, exist_ok=True)
    k200_cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cache_index_levels_fdr] range={start:%Y-%m-%d}~{end:%Y-%m-%d}")
    print(f"[cache_index_levels_fdr] symbols: KOSPI={kospi_symbol} KOSDAQ={kosdaq_symbol} K200={k200_symbol}")
    print(f"[cache_index_levels_fdr] out_levels={levels_path}")
    print(f"[cache_index_levels_fdr] out_k200_cache={k200_cache_path}")

    kospi = _fetch_close(kospi_symbol, start, end).rename(columns={"close": "kospi"})
    kosdaq = _fetch_close(kosdaq_symbol, start, end).rename(columns={"close": "kosdaq"})
    k200 = _fetch_close(k200_symbol, start, end).rename(columns={"close": "k200"})

    merged_new = kospi.merge(kosdaq, on="date", how="outer").merge(k200, on="date", how="outer")
    merged_new = merged_new.sort_values("date").reset_index(drop=True)

    # upsert index_levels.parquet
    if levels_path.exists():
        old = pd.read_parquet(levels_path)
        old["date"] = pd.to_datetime(old.get("date"), errors="coerce")
        for c in ["kospi", "kosdaq", "k200"]:
            if c in old.columns:
                old[c] = pd.to_numeric(old.get(c), errors="coerce")
        old = old.dropna(subset=["date"])
        merged = _upsert_ts(old, merged_new)
    else:
        merged = _upsert_ts(pd.DataFrame(), merged_new)

    merged.to_parquet(levels_path, index=False)
    merged.to_csv(levels_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(f"[cache_index_levels_fdr] OK levels rows={len(merged)} -> {levels_path}")

    # also write k200_close.parquet for F01 compatibility
    k200_out_new = merged_new[["date", "k200"]].dropna(subset=["k200"]).rename(columns={"k200": "k200_close"})
    if k200_cache_path.exists():
        old2 = pd.read_parquet(k200_cache_path)
        old2["date"] = pd.to_datetime(old2.get("date"), errors="coerce")
        if "k200_close" in old2.columns:
            old2["k200_close"] = pd.to_numeric(old2.get("k200_close"), errors="coerce")
        old2 = old2.dropna(subset=["date", "k200_close"])
        k200_merged = _upsert_ts(old2, k200_out_new)
    else:
        k200_merged = _upsert_ts(pd.DataFrame(), k200_out_new)

    k200_merged.to_parquet(k200_cache_path, index=False)
    print(f"[cache_index_levels_fdr] OK k200 rows={len(k200_merged)} -> {k200_cache_path}")


if __name__ == "__main__":
    main()
