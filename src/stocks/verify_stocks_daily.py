#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()

LISTINGS_PATH = PROJECT_ROOT / "data/stocks/master/listings.parquet"
PRICES_DIR = PROJECT_ROOT / "data/stocks/raw/prices"
CURATED_DIR = PROJECT_ROOT / "data/stocks/curated"
PANEL_PATH = PROJECT_ROOT / "data/stocks/mart/panel_daily_tail.parquet"

OUT_CSV = PROJECT_ROOT / "docs/stocks/stocks_daily_health.csv"
OUT_MD  = PROJECT_ROOT / "docs/stocks/stocks_daily_health.md"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def _last_date_prices(ticker: str):
    p = PRICES_DIR / f"{ticker}.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, encoding="utf-8-sig", usecols=["date"])
        d = pd.to_datetime(df["date"], errors="coerce").dropna()
        return None if d.empty else d.max()
    except Exception:
        return None

def _last_date_parquet(p: Path):
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, columns=["date"])
        d = pd.to_datetime(df["date"], errors="coerce").dropna()
        return None if d.empty else d.max()
    except Exception:
        return None

def main():
    if not LISTINGS_PATH.exists():
        raise FileNotFoundError(f"Missing listings: {LISTINGS_PATH}")

    lst = pd.read_parquet(LISTINGS_PATH)
    lst["ticker"] = lst["ticker"].astype(str)
    tickers = lst["ticker"].tolist()

    rows = []
    for t in tickers:
        px_last = _last_date_prices(t)
        fl_last = _last_date_parquet(CURATED_DIR / t / "flows_daily.parquet")
        fu_last = _last_date_parquet(CURATED_DIR / t / "fundamentals_daily.parquet")
        rows.append({
            "ticker": t,
            "prices_last_date": px_last,
            "flows_last_date": fl_last,
            "fund_last_date": fu_last,
            "has_prices": px_last is not None,
            "has_flows": fl_last is not None,
            "has_fund": fu_last is not None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    n = len(df)
    cov_prices = df["has_prices"].mean() * 100
    cov_flows  = df["has_flows"].mean() * 100
    cov_fund   = df["has_fund"].mean() * 100

    panel_ok = PANEL_PATH.exists()
    md = []
    md.append("# Stocks Daily Health Report")
    md.append("")
    md.append(f"- tickers: **{n:,}**")
    md.append(f"- prices coverage: **{cov_prices:.1f}%**")
    md.append(f"- flows coverage: **{cov_flows:.1f}%**")
    md.append(f"- fundamentals coverage: **{cov_fund:.1f}%**")
    md.append(f"- panel parquet exists: **{panel_ok}** (`{PANEL_PATH}`)")
    md.append("")
    md.append("## Latest dates (median)")
    md.append("")
    md.append(f"- prices_last_date median: {pd.to_datetime(df['prices_last_date']).median()}")
    md.append(f"- flows_last_date median:  {pd.to_datetime(df['flows_last_date']).median()}")
    md.append(f"- fund_last_date median:   {pd.to_datetime(df['fund_last_date']).median()}")
    md.append("")
    md.append("## Worst 20 tickers (missing any of prices/flows/fund)")
    md.append("")
    bad = df[~(df["has_prices"] & df["has_flows"] & df["has_fund"])].head(20)
    md.append(bad.to_markdown(index=False) if not bad.empty else "_None_")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[health] wrote: {OUT_CSV}")
    print(f"[health] wrote: {OUT_MD}")

if __name__ == "__main__":
    main()
