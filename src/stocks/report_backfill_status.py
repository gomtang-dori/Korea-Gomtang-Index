#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backfill status report:
- per ticker: existence + rows + last_date for prices/flows/fundamentals
- DART: standard summary availability (latest asof_date)
- panel output existence + row count (optional)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path.cwd()

def _safe_to_markdown(df: pd.DataFrame, index: bool = False, max_rows: int = 30) -> str:
    """
    pandas.DataFrame.to_markdown() requires optional dependency 'tabulate'.
    This helper:
    - uses to_markdown when available
    - falls back to a minimal pipe-table markdown when tabulate is missing
    """
    if df is None or len(df) == 0:
        return "(none)"

    # limit rows so MD doesn't explode on Actions logs
    d = df.head(max_rows).copy()

    try:
        # will raise ImportError if tabulate is missing
        return d.to_markdown(index=index)
    except Exception:
        cols = list(d.columns)
        lines = []
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in d.iterrows():
            lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
        if len(df) > max_rows:
            lines.append(f"\n... (showing first {max_rows} rows of {len(df):,})")
        return "\n".join(lines)

MASTER = PROJECT_ROOT / "data/stocks/master/listings.parquet"
PRICES_DIR = PROJECT_ROOT / "data/stocks/raw/prices"
CURATED_DIR = PROJECT_ROOT / "data/stocks/curated"

DART_FULL_FROM = int(os.getenv("DART_STANDARD_FULL_YEAR_FROM", "2015"))
DART_FULL_TO = int(os.getenv("DART_STANDARD_FULL_YEAR_TO", "2026"))
DART_STANDARD_CSV = PROJECT_ROOT / f"docs/stocks/dart_standard_{DART_FULL_FROM}_{DART_FULL_TO}.csv"

OUT_CSV = Path(os.getenv("BACKFILL_STATUS_CSV", "docs/stocks/backfill_status.csv"))
OUT_MD = Path(os.getenv("BACKFILL_STATUS_MD", "docs/stocks/backfill_status.md"))
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

PANEL_OUT = Path(os.getenv("PANEL_OUT_PARQUET", "data/stocks/mart/panel_daily.parquet"))


def _prices_meta(path: Path):
    if not path.exists():
        return False, 0, None, None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", usecols=["date"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna()
        if df.empty:
            return True, 0, None, None
        return True, len(df), df["date"].min().date().isoformat(), df["date"].max().date().isoformat()
    except Exception:
        return True, 0, None, None


def _parquet_date_meta(path: Path, date_col="date"):
    if not path.exists():
        return False, 0, None, None
    try:
        df = pd.read_parquet(path, columns=[date_col])
        if date_col not in df.columns:
            return True, len(df), None, None
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        if df.empty:
            return True, 0, None, None
        return True, len(df), df[date_col].min().date().isoformat(), df[date_col].max().date().isoformat()
    except Exception:
        return True, 0, None, None


def _dart_meta(ticker: str, dart_std: pd.DataFrame):
    if dart_std is None or dart_std.empty:
        return False, 0, None
    d = dart_std[dart_std["ticker"] == ticker]
    if d.empty:
        return False, 0, None
    # asof_date is created in build_panel, but in standard csv we have year/reprt_code only.
    # Here: use year max as proxy + count.
    years = pd.to_numeric(d["year"], errors="coerce").dropna()
    latest_year = int(years.max()) if len(years) else None
    return True, len(d), str(latest_year) if latest_year else None


def main():
    if not MASTER.exists():
        raise FileNotFoundError(f"missing: {MASTER} (run fetch_listings.py first)")

    listings = pd.read_parquet(MASTER).copy()
    listings["ticker"] = listings["ticker"].astype(str).str.zfill(6)

    dart_std = pd.DataFrame()
    if DART_STANDARD_CSV.exists():
        try:
            dart_std = pd.read_csv(DART_STANDARD_CSV, encoding="utf-8-sig")
            dart_std["ticker"] = dart_std["ticker"].astype(str).str.zfill(6)
        except Exception:
            dart_std = pd.DataFrame()

    rows = []
    for _, r in listings.iterrows():
        ticker = r["ticker"]
        name = r.get("name", "")
        market = r.get("market", "")

        prices_path = PRICES_DIR / f"{ticker}.csv"
        flows_path = CURATED_DIR / ticker / "flows_daily.parquet"
        fund_path = CURATED_DIR / ticker / "fundamentals_daily.parquet"

        has_prices, n_prices, min_p, max_p = _prices_meta(prices_path)
        has_flows, n_flows, min_f, max_f = _parquet_date_meta(flows_path)
        has_fund, n_fund, min_u, max_u = _parquet_date_meta(fund_path)

        has_dart, n_dart, dart_latest_year = _dart_meta(ticker, dart_std)

        rows.append({
            "ticker": ticker,
            "name": name,
            "market": market,

            "has_prices": has_prices,
            "prices_rows": n_prices,
            "prices_min_date": min_p,
            "prices_max_date": max_p,

            "has_flows_curated": has_flows,
            "flows_rows": n_flows,
            "flows_min_date": min_f,
            "flows_max_date": max_f,

            "has_fundamentals_curated": has_fund,
            "fund_rows": n_fund,
            "fund_min_date": min_u,
            "fund_max_date": max_u,

            "has_dart_standard": has_dart,
            "dart_rows": n_dart,
            "dart_latest_year": dart_latest_year,
        })

    df = pd.DataFrame(rows)

    # quick coverage stats
    cov = {
        "tickers_total": len(df),
        "tickers_has_prices": int(df["has_prices"].sum()),
        "tickers_has_flows": int(df["has_flows_curated"].sum()),
        "tickers_has_fund": int(df["has_fundamentals_curated"].sum()),
        "tickers_has_dart_standard": int(df["has_dart_standard"].sum()),
    }

    # panel existence
    panel_exists = PANEL_OUT.exists()
    panel_note = str(PANEL_OUT)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # write md summary
    lines = []
    lines.append("# Backfill Status Report (Stocks)")
    lines.append("")
    lines.append("## Coverage summary")
    lines.append("")
    lines.append(f"- tickers_total: {cov['tickers_total']:,}")
    lines.append(f"- has_prices: {cov['tickers_has_prices']:,}")
    lines.append(f"- has_flows_curated: {cov['tickers_has_flows']:,}")
    lines.append(f"- has_fundamentals_curated: {cov['tickers_has_fund']:,}")
    lines.append(f"- has_dart_standard: {cov['tickers_has_dart_standard']:,}  (from {DART_STANDARD_CSV})")
    lines.append("")
    lines.append("## Panel parquet")
    lines.append(f"- exists: {panel_exists}")
    lines.append(f"- path: `{panel_note}`")
    lines.append("")
    lines.append("## Top missing (prices)")
    lines.append("")
    miss_prices = df[~df["has_prices"]].head(30)[["ticker","name","market"]]
    lines.append(_safe_to_markdown(miss_prices, index=False))
    lines.append("")
    lines.append("## Top missing (flows curated)")
    lines.append("")
    miss_flows = df[~df["has_flows_curated"]].head(30)[["ticker","name","market"]]
    lines.append(_safe_to_markdown(miss_flows, index=False))
    lines.append("")
    lines.append("## Top missing (fundamentals curated)")
    lines.append("")
    miss_fund = df[~df["has_fundamentals_curated"]].head(30)[["ticker","name","market"]]
    lines.append(_safe_to_markdown(miss_fund, index=False))
    lines.append("")
    lines.append("## Top missing (DART standard)")
    lines.append("")
    miss_dart = df[~df["has_dart_standard"]].head(30)[["ticker","name","market"]]
    lines.append(_safe_to_markdown(miss_dart, index=False))
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("[backfill_status] OK")
    print(f"  wrote: {OUT_CSV}")
    print(f"  wrote: {OUT_MD}")
    print("  coverage:", cov)


if __name__ == "__main__":
    main()
