# src/caches/cache_k200_close_fdr.py
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

def main():
    out_path = Path(os.environ.get("K200_CACHE_PATH", "data/cache/k200_close.parquet"))
    symbol = os.environ.get("K200_SYMBOL", "KS200")  # FDR: KOSPI200

    backfill_start = pd.to_datetime(_env("BACKFILL_START"), format="%Y%m%d").date()
    backfill_end = pd.to_datetime(_env("BACKFILL_END"), format="%Y%m%d").date()
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))

    start = (pd.Timestamp(backfill_start) - pd.Timedelta(days=buffer_days)).date()
    end = pd.Timestamp(backfill_end).date()

    print(f"[cache_k200_close:fdr] fetch symbol={symbol} range={start}~{end}")

    df = fdr.DataReader(symbol, str(start), str(end))
    if df is None or len(df) == 0:
        raise RuntimeError(f"[cache_k200_close:fdr] empty result symbol={symbol} range={start}~{end}")

    # FDR columns: Open High Low Close Volume ... (보통 영문)
    if "Close" not in df.columns:
        raise RuntimeError(f"[cache_k200_close:fdr] missing Close in cols={list(df.columns)}")

    out = df.reset_index().rename(columns={"Date": "date"})
    if "date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["k200_close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["date", "k200_close"]).sort_values("date").reset_index(drop=True)
    out = out[["date", "k200_close"]]

    # upsert
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
    print(f"[cache_k200_close:fdr] OK rows={len(merged)} -> {out_path}")

if __name__ == "__main__":
    main()
