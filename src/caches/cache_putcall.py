# src/caches/cache_putcall.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from lib.krx_putcall import fetch_putcall_ratio_by_date


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def upsert_ts(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        out = new.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return out


def main():
    start_s = os.environ.get("BACKFILL_START", "").strip()
    end_s = os.environ.get("BACKFILL_END", "").strip()
    out_path = Path(os.environ.get("PUTCALL_CACHE_PATH", "data/cache/putcall_ratio.parquet"))
    if not (start_s and end_s):
        raise RuntimeError("cache_putcall requires BACKFILL_START/BACKFILL_END")

    start = pd.to_datetime(start_s, format="%Y%m%d")
    end = pd.to_datetime(end_s, format="%Y%m%d")

    df = fetch_putcall_ratio_by_date(start, end)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    ensure_dir(out_path.parent)
    old = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    out = upsert_ts(old, df)
    out.to_parquet(out_path, index=False)
    print(f"[cache_putcall] OK rows={len(out)} -> {out_path}")


if __name__ == "__main__":
    main()
