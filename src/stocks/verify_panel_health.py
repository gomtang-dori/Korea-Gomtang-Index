#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path.cwd()

LISTINGS_PATH = PROJECT_ROOT / "data/stocks/master/listings.parquet"
PRICES_DIR = PROJECT_ROOT / "data/stocks/raw/prices"
CURATED_DIR = PROJECT_ROOT / "data/stocks/curated"

PANEL_SCOPE = os.getenv("PANEL_SCOPE", "5y").strip().lower()  # 5y|all
PANEL_PATH = Path(os.getenv("PANEL_PATH", ""))

def _default_panel_path():
    if PANEL_SCOPE == "all":
        return PROJECT_ROOT / "data/stocks/mart/panel_all.parquet"
    return PROJECT_ROOT / "data/stocks/mart/panel_5y.parquet"

def _exists_prices(ticker: str) -> bool:
    return (PRICES_DIR / f"{ticker}.csv").exists()

def _exists_flows(ticker: str) -> bool:
    return (CURATED_DIR / ticker / "flows_daily.parquet").exists()

def _exists_fund(ticker: str) -> bool:
    return (CURATED_DIR / ticker / "fundamentals_daily.parquet").exists()

def main():
    if not LISTINGS_PATH.exists():
        raise FileNotFoundError(f"Missing listings: {LISTINGS_PATH}")

    panel_path = PANEL_PATH if PANEL_PATH else _default_panel_path()

    lst = pd.read_parquet(LISTINGS_PATH)
    lst["ticker"] = lst["ticker"].astype(str)
    tickers = lst["ticker"].tolist()

    rows = []
    for t in tickers:
        rows.append({
            "ticker": t,
            "has_prices": _exists_prices(t),
            "has_flows": _exists_flows(t),
            "has_fund": _exists_fund(t),
        })

    df = pd.DataFrame(rows)
    cov_prices = df["has_prices"].mean() * 100
    cov_flows = df["has_flows"].mean() * 100
    cov_fund = df["has_fund"].mean() * 100

    panel_ok = panel_path.exists()
    panel_rows = None
    panel_cols = None
    has_dart_cols = False
    panel_size_mb = None

    if panel_ok:
        pf = pq.ParquetFile(panel_path.as_posix())
        panel_rows = pf.metadata.num_rows
        panel_cols = pf.schema_arrow.names
        has_dart_cols = all(c in panel_cols for c in ["dart_revenue", "dart_operating_income", "dart_net_income", "dart_equity", "dart_roe"])
        panel_size_mb = panel_path.stat().st_size / 1024 / 1024

    out_dir = PROJECT_ROOT / "docs/stocks"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"panel_health_{PANEL_SCOPE}.csv"
    out_md = out_dir / f"panel_health_{PANEL_SCOPE}.md"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    md = []
    md.append(f"# Panel Health Report ({PANEL_SCOPE})")
    md.append("")
    md.append(f"- tickers: **{len(df):,}**")
    md.append(f"- source coverage: prices **{cov_prices:.1f}%**, flows **{cov_flows:.1f}%**, fundamentals **{cov_fund:.1f}%**")
    md.append(f"- panel exists: **{panel_ok}** (`{panel_path}`)")
    if panel_ok:
        md.append(f"- panel rows: **{panel_rows:,}**")
        md.append(f"- panel size: **{panel_size_mb:.1f} MB**")
        md.append(f"- DART columns present: **{has_dart_cols}**")
    md.append("")
    md.append("## Worst 30 tickers (missing any source)")
    md.append("")
    bad = df[~(df["has_prices"] & df["has_flows"] & df["has_fund"])].head(30)
    md.append(bad.to_markdown(index=False) if not bad.empty else "_None_")

    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[health] wrote: {out_csv}")
    print(f"[health] wrote: {out_md}")

if __name__ == "__main__":
    main()
