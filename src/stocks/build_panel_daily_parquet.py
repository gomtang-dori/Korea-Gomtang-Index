#!/usr/bin/env python3
import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

PROJECT_ROOT = Path.cwd()

PANEL_TAIL_DAYS = int(os.getenv("PANEL_TAIL_DAYS", "260"))
OUT_PATH = PROJECT_ROOT / "data/stocks/mart/panel_daily_tail.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

LISTINGS_PATH = PROJECT_ROOT / "data/stocks/master/listings.parquet"
PRICES_DIR = PROJECT_ROOT / "data/stocks/raw/prices"
FLOWS_DIR = PROJECT_ROOT / "data/stocks/curated"
FUND_DIR = PROJECT_ROOT / "data/stocks/curated"

# full-range standard 파일이 repo에 있거나(권장), DART workflow/curate에서 생성돼 있어야 합니다.
DART_STD = PROJECT_ROOT / "docs/stocks/dart_standard_2015_2026.csv"

def _read_prices(ticker: str) -> pd.DataFrame:
    p = PRICES_DIR / f"{ticker}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, encoding="utf-8-sig")
    if "date" not in df.columns:
        df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    # tail window
    if len(df) > 0:
        cutoff = df["date"].max() - pd.Timedelta(days=PANEL_TAIL_DAYS)
        df = df[df["date"] >= cutoff]
    return df

def _read_parquet_if_exists(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
    return df

def main():
    if not LISTINGS_PATH.exists():
        raise FileNotFoundError(f"Missing listings: {LISTINGS_PATH}")

    listings = pd.read_parquet(LISTINGS_PATH)
    listings["ticker"] = listings["ticker"].astype(str)

    dart_std = None
    if DART_STD.exists():
        dart_std = pd.read_csv(DART_STD)
        dart_std["ticker"] = dart_std["ticker"].astype(str)
        dart_std["year"] = pd.to_numeric(dart_std["year"], errors="coerce")
        dart_std["reprt_code"] = dart_std["reprt_code"].astype(str)

        # report asof_date (단순화): reprt_code를 분기말로 매핑
        # 11011(사업보고서)=12/31, 11012(1Q)=03/31, 11013(반기)=06/30, 11014(3Q)=09/30
        m = {"11011": (12, 31), "11012": (3, 31), "11013": (6, 30), "11014": (9, 30)}
        def to_asof(r):
            y = r["year"]
            rc = r["reprt_code"]
            if pd.isna(y) or rc not in m:
                return pd.NaT
            mm, dd = m[rc]
            return pd.Timestamp(int(y), mm, dd)
        dart_std["dart_asof_date"] = dart_std.apply(to_asof, axis=1)
        # 표준요약 컬럼 rename
        dart_std = dart_std.rename(columns={
            "operating_income": "dart_operating_income",
            "net_income": "dart_net_income",
            "equity": "dart_equity",
            "revenue": "dart_revenue",
            "roe": "dart_roe",
        })
        dart_std = dart_std[[
            "ticker", "dart_asof_date", "reprt_code",
            "dart_revenue", "dart_operating_income", "dart_net_income", "dart_equity", "dart_roe"
        ]].dropna(subset=["ticker"])

    frames = []
    total = len(listings)
    for i, r in listings.iterrows():
        ticker = r["ticker"]
        name = r.get("name", "")
        market = r.get("market", "")

        px = _read_prices(ticker)
        if px.empty:
            continue

        flows = _read_parquet_if_exists(FLOWS_DIR / ticker / "flows_daily.parquet")
        fund = _read_parquet_if_exists(FUND_DIR / ticker / "fundamentals_daily.parquet")

        df = px.copy()
        df["ticker"] = ticker
        df["name"] = name
        df["market"] = market

        if not flows.empty:
            df = df.merge(flows, on="date", how="left")

        if not fund.empty:
            keep = [c for c in fund.columns if c in ["date", "bps", "per", "pbr", "eps", "div", "dps"]]
            df = df.merge(fund[keep], on="date", how="left")

        # DART asof join (가장 최근 보고서)
        if dart_std is not None:
            d = dart_std[dart_std["ticker"] == ticker].sort_values("dart_asof_date")
            if not d.empty:
                df = df.sort_values("date")
                df = pd.merge_asof(
                    df, d,
                    left_on="date", right_on="dart_asof_date",
                    direction="backward"
                )
        frames.append(df)

        if i < 3 or (i + 1) % 500 == 0:
            print(f"[panel] {i+1}/{total} ticker={ticker} rows={len(df):,}")

    if not frames:
        raise RuntimeError("No panel rows produced (prices missing?)")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "ticker"])

    out.to_parquet(OUT_PATH, index=False)
    print(f"[panel] wrote: {OUT_PATH} rows={len(out):,} cols={len(out.columns)}")

if __name__ == "__main__":
    main()
