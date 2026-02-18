#!/usr/bin/env python3
"""
최근 N일(기본 60일) curated flows를 전종목 통합 CSV로 export (엑셀/검증용)

입력:
- data/stocks/curated/{ticker}/flows_daily.parquet

출력:
- docs/stocks/flows_recent60d.csv

환경변수:
- RECENT_DAYS: 기본 60
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

PROJECT_ROOT = Path.cwd()
RECENT_DAYS = int(os.getenv("RECENT_DAYS", "60"))

MASTER_PATH = PROJECT_ROOT / "data/stocks/master/listings.parquet"
OUT_DIR = PROJECT_ROOT / "docs/stocks"
OUT_PATH = OUT_DIR / "flows_recent60d.csv"

def main():
    print("[export_flows_recent60d_csv] start")
    print(f"  CWD={PROJECT_ROOT}, RECENT_DAYS={RECENT_DAYS}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_master = pd.read_parquet(MASTER_PATH)
    cutoff = pd.to_datetime(datetime.now() - timedelta(days=RECENT_DAYS))

    frames = []
    for i, r in enumerate(df_master.itertuples(index=False), 1):
        ticker = str(r.ticker)
        name = str(r.name)
        market = str(r.market)

        p = PROJECT_ROOT / f"data/stocks/curated/{ticker}/flows_daily.parquet"
        if not p.exists():
            continue

        df = pd.read_parquet(p)
        if df.empty:
            continue

        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= cutoff].copy()
        if df.empty:
            continue

        df.insert(1, "ticker", ticker)
        df.insert(2, "name", name)
        df.insert(3, "market", market)

        frames.append(df)

        if i <= 10 or i % 500 == 0:
            print(f"  [{i}/{len(df_master)}] read {ticker} -> {len(df)} rows")

    if not frames:
        print("  no data -> skip")
        return

    out = pd.concat(frames, ignore_index=True)
    out.sort_values(["date", "ticker"], inplace=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"[export_flows_recent60d_csv] OK -> {OUT_PATH} ({len(out):,} rows, {size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
