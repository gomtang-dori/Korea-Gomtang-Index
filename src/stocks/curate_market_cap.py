#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Curate market cap data

raw:
- data/stocks/raw/market_cap_by_date/YYYYMMDD.parquet

curated (per ticker):
- data/stocks/curated/{ticker}/market_cap_daily.parquet

Schema:
- date, market_cap, shares, value

Env:
- MARKETCAP_RAW_DIR (default data/stocks/raw/market_cap_by_date)
- MARKETCAP_CURATED_ROOT (default data/stocks/curated)
- MARKETCAP_CURATE_OVERWRITE (default false): if true, overwrite per ticker file
"""

import os
from pathlib import Path
import pandas as pd


def env_bool(k: str, default: bool = False) -> bool:
    v = os.getenv(k)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


RAW_DIR = Path(os.getenv("MARKETCAP_RAW_DIR", "data/stocks/raw/market_cap_by_date"))
CURATED_ROOT = Path(os.getenv("MARKETCAP_CURATED_ROOT", "data/stocks/curated"))
OVERWRITE = env_bool("MARKETCAP_CURATE_OVERWRITE", False)


def main():
    print("[curate_market_cap] start")
    print(f"  raw_dir={RAW_DIR}")
    print(f"  curated_root={CURATED_ROOT}")
    print(f"  overwrite={OVERWRITE}")

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"missing raw dir: {RAW_DIR}")

    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no raw parquet files in: {RAW_DIR}")

    frames = []
    for p in files:
        try:
            df = pd.read_parquet(p)
            if df is None or df.empty:
                continue
            frames.append(df)
        except Exception as e:
            print(f"  [warn] read error {p.name}: {e}")

    if not frames:
        raise SystemExit("[curate_market_cap] no data frames loaded")

    all_df = pd.concat(frames, ignore_index=True)
    all_df["ticker"] = all_df["ticker"].astype(str).str.zfill(6)
    all_df["date"] = pd.to_datetime(all_df["date"], errors="coerce")

    for c in ["market_cap", "shares", "value"]:
        if c not in all_df.columns:
            all_df[c] = pd.NA
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

    all_df = all_df.dropna(subset=["date", "ticker"]).drop_duplicates(subset=["date", "ticker"], keep="last")
    all_df = all_df.sort_values(["ticker", "date"])

    ok = 0
    for t, g in all_df.groupby("ticker", sort=True):
        out_dir = CURATED_ROOT / t
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "market_cap_daily.parquet"

        if out_path.exists() and not OVERWRITE:
            # append / upsert
            old = pd.read_parquet(out_path)
            old["date"] = pd.to_datetime(old["date"], errors="coerce")
            merged = pd.concat([old, g[["date", "market_cap", "shares", "value"]]], ignore_index=True)
            merged = merged.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
        else:
            merged = g[["date", "market_cap", "shares", "value"]].drop_duplicates(subset=["date"], keep="last").sort_values("date")

        merged.to_parquet(out_path, index=False)
        ok += 1
        if ok <= 10 or ok % 500 == 0:
            print(f"  [{ok}] {t}: rows={len(merged):,}")

    print("[curate_market_cap] done")
    print(f"  tickers_written={ok:,}")


if __name__ == "__main__":
    main()
