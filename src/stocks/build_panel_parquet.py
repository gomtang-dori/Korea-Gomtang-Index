#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build unified stock panel parquet (single file, overwrite)

daily(최근5년): PANEL_SCOPE=5y, 출력 data/stocks/mart/panel_5y.parquet (덮어쓰기)
weekly(전체기간): PANEL_SCOPE=all, 출력 data/stocks/mart/panel_all.parquet (덮어쓰기)
DART 표준요약: docs/stocks/dart_standard_2015_2026.csv 그대로 사용, dart_asof_date를 reprt_code로 분기말(또는 연말) 매핑 후 merge_asof(backward)로 일자에 붙임
(DART는 분기 업데이트지만 “최근 일자에도 직전 값이 들어가는” 효과)

Inputs:
- listings: data/stocks/master/listings.parquet
- prices (raw): data/stocks/raw/prices/{ticker}.csv
- flows (curated): data/stocks/curated/{ticker}/flows_daily.parquet
- fundamentals (curated): data/stocks/curated/{ticker}/fundamentals_daily.parquet
- DART standard summary CSV (fixed): docs/stocks/dart_standard_2015_2026.csv

Outputs:
- PANEL_SCOPE=5y  -> data/stocks/mart/panel_5y.parquet   (overwrite)
- PANEL_SCOPE=all -> data/stocks/mart/panel_all.parquet  (overwrite)

Key behavior:
- DART: merge_asof(direction="backward") so latest report is carried to recent dates.
- 5y scope: per-ticker "data max date" 기준 5년만 포함 (데이터 기준 5년).
- Writes parquet in streaming manner (ParquetWriter) to avoid huge memory usage.
"""

import os
from pathlib import Path
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path.cwd()

LISTINGS_PATH = PROJECT_ROOT / "data/stocks/master/listings.parquet"
PRICES_DIR = PROJECT_ROOT / "data/stocks/raw/prices"
CURATED_DIR = PROJECT_ROOT / "data/stocks/curated"

DART_STD_PATH = Path(os.getenv("DART_STD_PATH", "docs/stocks/dart_standard_2015_2026.csv"))
PANEL_SCOPE = os.getenv("PANEL_SCOPE", "5y").strip().lower()  # "5y" or "all"
OUT_PATH = Path(os.getenv("PANEL_OUT_PATH", ""))  # optional override

PARQUET_COMPRESSION = os.getenv("PANEL_PARQUET_COMPRESSION", "zstd")

def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "date" not in df.columns:
        df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df

def _read_prices(ticker: str) -> pd.DataFrame:
    p = PRICES_DIR / f"{ticker}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, encoding="utf-8-sig")
    df = _ensure_date(df)
    # normalize required cols
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = pd.NA
    return df[["date", "open", "high", "low", "close", "volume"]].copy()

def _read_parquet_if_exists(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df = _ensure_date(df)
    return df

def _load_dart_std() -> pd.DataFrame:
    if not DART_STD_PATH.exists():
        print(f"[panel] WARN: DART standard CSV missing: {DART_STD_PATH} -> continue without DART")
        return pd.DataFrame()

    d = pd.read_csv(DART_STD_PATH)
    # expected columns: ticker, year, reprt_code, revenue, operating_income, net_income, equity, roe
    d["ticker"] = d["ticker"].astype(str)
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["reprt_code"] = d["reprt_code"].astype(str)

    # reprt_code -> asof date mapping
    # 11011: annual -> 12/31
    # 11012: Q1 -> 03/31
    # 11013: half -> 06/30
    # 11014: Q3 -> 09/30
    m = {"11011": (12, 31), "11012": (3, 31), "11013": (6, 30), "11014": (9, 30)}
    def to_asof(r):
        y = r["year"]
        rc = r["reprt_code"]
        if pd.isna(y) or rc not in m:
            return pd.NaT
        mm, dd = m[rc]
        return pd.Timestamp(int(y), mm, dd)

    d["dart_asof_date"] = d.apply(to_asof, axis=1)

    d = d.rename(columns={
        "revenue": "dart_revenue",
        "operating_income": "dart_operating_income",
        "net_income": "dart_net_income",
        "equity": "dart_equity",
        "roe": "dart_roe",
        "reprt_code": "dart_reprt_code",
    })

    keep = ["ticker", "dart_asof_date", "dart_reprt_code",
            "dart_revenue", "dart_operating_income", "dart_net_income", "dart_equity", "dart_roe"]
    d = d[keep].dropna(subset=["ticker", "dart_asof_date"]).sort_values(["ticker", "dart_asof_date"])
    return d

def _apply_scope_5y(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    maxd = df["date"].max()
    # "데이터 기준 5년": 해당 ticker의 max(date) 기준으로 5년 컷
    cutoff = maxd - pd.DateOffset(years=5)
    return df[df["date"] >= cutoff].copy()

def _determine_out_path() -> Path:
    if OUT_PATH:
        return OUT_PATH
    if PANEL_SCOPE == "all":
        return PROJECT_ROOT / "data/stocks/mart/panel_all.parquet"
    return PROJECT_ROOT / "data/stocks/mart/panel_5y.parquet"

def main():
    if not LISTINGS_PATH.exists():
        raise FileNotFoundError(f"Missing listings: {LISTINGS_PATH}")

    out_path = _determine_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    listings = pd.read_parquet(LISTINGS_PATH)
    listings["ticker"] = listings["ticker"].astype(str)

    dart = _load_dart_std()

    writer = None
    total_rows = 0
    written_tickers = 0

    for i, r in listings.iterrows():
        ticker = r["ticker"]
        name = str(r.get("name", ""))
        market = str(r.get("market", ""))

        px = _read_prices(ticker)
        if px.empty:
            continue

        if PANEL_SCOPE == "5y":
            px = _apply_scope_5y(px)

        flows = _read_parquet_if_exists(CURATED_DIR / ticker / "flows_daily.parquet")
        fund = _read_parquet_if_exists(CURATED_DIR / ticker / "fundamentals_daily.parquet")

        df = px.copy()
        df["ticker"] = ticker
        df["name"] = name
        df["market"] = market

        if not flows.empty:
            df = df.merge(flows, on="date", how="left")

        if not fund.empty:
            keep = [c for c in fund.columns if c in ["date", "bps", "per", "pbr", "eps", "div", "dps"]]
            df = df.merge(fund[keep], on="date", how="left")

        # DART: attach latest as-of report to each daily row
        if not dart.empty:
            d = dart[dart["ticker"] == ticker].sort_values("dart_asof_date")
            if not d.empty:
                df = df.sort_values("date")
                df = pd.merge_asof(
                    df,
                    d,
                    left_on="date",
                    right_on="dart_asof_date",
                    direction="backward"
                )

        df = df.sort_values("date")
        # reorder: identifiers first
        front = ["ticker", "name", "market", "date", "open", "high", "low", "close", "volume"]
        cols = front + [c for c in df.columns if c not in front]
        df = df[cols]

        table = pa.Table.from_pandas(df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(out_path.as_posix(), table.schema, compression=PARQUET_COMPRESSION)

        # schema align (in case of missing cols)
        if table.schema != writer.schema:
            # add missing columns
            for field in writer.schema:
                if field.name not in table.column_names:
                    table = table.append_column(field.name, pa.array([None] * table.num_rows, type=field.type))
            # drop extra columns not in writer schema
            keep_names = [f.name for f in writer.schema]
            table = table.select(keep_names)

        writer.write_table(table)
        total_rows += table.num_rows
        written_tickers += 1

        if written_tickers <= 3 or written_tickers % 500 == 0:
            print(f"[panel] {written_tickers} tickers written, last={ticker}, rows_total={total_rows:,}")

    if writer:
        writer.close()

    print(f"[panel] DONE scope={PANEL_SCOPE} -> {out_path} rows={total_rows:,} tickers={written_tickers:,}")

if __name__ == "__main__":
    main()
