# src/caches/cache_usdkrw_level_fdr.py
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
    # tz-naive 통일
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def main():
    out_path = Path(os.environ.get("USDKRW_CACHE_PATH", "data/cache/usdkrw_level.parquet"))

    backfill_start = _parse_yyyymmdd(_env("BACKFILL_START"))
    backfill_end = _parse_yyyymmdd(_env("BACKFILL_END"))
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))

    today_utc = pd.Timestamp.utcnow().tz_localize(None).normalize()
    safe_end = today_utc - pd.Timedelta(days=1)
    if backfill_end > safe_end:
        print(f"[cache_usdkrw:fdr] WARN: BACKFILL_END {backfill_end:%Y%m%d} > safe_end {safe_end:%Y%m%d}. clamp.")
        backfill_end = safe_end

    start = (backfill_start - pd.Timedelta(days=buffer_days)).normalize()
    end = backfill_end.normalize()

    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    print(f"[cache_usdkrw:fdr] fetch USD/KRW range={start_s}~{end_s}")

    df = fdr.DataReader("USD/KRW", start_s, end_s)
    if df is None or df.empty:
        raise RuntimeError(f"[cache_usdkrw:fdr] empty result range={start_s}~{end_s}")

    if "Close" not in df.columns:
        raise RuntimeError(f"[cache_usdkrw:fdr] cannot find Close. cols={list(df.columns)}")

    out = (
        df[["Close"]]
        .rename(columns={"Close": "usdkrw"})
        .reset_index()
        .rename(columns={"Date": "date"})
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["usdkrw"] = pd.to_numeric(out["usdkrw"], errors="coerce")
    out = out.dropna(subset=["date", "usdkrw"]).sort_values("date").reset_index(drop=True)

    if out_path.exists():
        old = pd.read_parquet(out_path)
        old["date"] = pd.to_datetime(old.get("date"), errors="coerce")
        old["usdkrw"] = pd.to_numeric(old.get("usdkrw"), errors="coerce")
        old = old.dropna(subset=["date", "usdkrw"])
        merged = pd.concat([old, out], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    else:
        merged = out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"[cache_usdkrw:fdr] OK rows={len(merged)} -> {out_path}")


if __name__ == "__main__":
    main()
