# src/caches/cache_k200_close.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from lib.krx_kospi_index import KRXKospiIndexAPI


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_dt(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def upsert_ts(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        out = new.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)
    out = safe_dt(out)
    out = out.dropna(subset=["date"])
    out = out.drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def main():
    start_s = os.environ.get("BACKFILL_START", "").strip()
    end_s = os.environ.get("BACKFILL_END", "").strip()
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))
    out_path = Path(os.environ.get("K200_CACHE_PATH", "data/cache/k200_close.parquet"))

    if not (start_s and end_s):
        raise RuntimeError("cache_k200_close requires BACKFILL_START/BACKFILL_END")

    start = pd.to_datetime(start_s, format="%Y%m%d") - pd.Timedelta(days=buffer_days)
    end = pd.to_datetime(end_s, format="%Y%m%d")

    api = KRXKospiIndexAPI.from_env()
    # range를 지원하면 range로, 아니면 내부에서 일자별 호출
    k200 = api.fetch_k200_close_range(start, end)
    k200 = safe_dt(k200)
    k200["k200_close"] = pd.to_numeric(k200.get("k200_close"), errors="coerce")
    k200 = k200.dropna(subset=["date", "k200_close"]).reset_index(drop=True)

    ensure_dir(out_path.parent)
    old = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    out = upsert_ts(old, k200)
    out.to_parquet(out_path, index=False)
    print(f"[cache_k200_close] OK rows={len(out)} -> {out_path}")


if __name__ == "__main__":
    main()
