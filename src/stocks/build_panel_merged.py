#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build merged 1-file stock panel parquet:
- listings (ticker,name,market)
- prices (daily OHLCV CSV per ticker)
- flows_daily.parquet (curated)
- fundamentals_daily.parquet (curated)
- DART standard summary CSV (year, reprt_code, revenue/op/net/equity/roe) -> asof join to daily

Output:
- data/stocks/mart/panel_daily.parquet  (single file)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path.cwd()

MASTER = PROJECT_ROOT / "data/stocks/master/listings.parquet"
PRICES_DIR = PROJECT_ROOT / "data/stocks/raw/prices"
CURATED_DIR = PROJECT_ROOT / "data/stocks/curated"

OUT_PARQUET = Path(os.getenv("PANEL_OUT_PARQUET", "data/stocks/mart/panel_daily.parquet"))
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

PANEL_START_DATE = (os.getenv("PANEL_START_DATE", "") or "").strip()
PANEL_END_DATE = (os.getenv("PANEL_END_DATE", "") or "").strip()

DART_FULL_FROM = int(os.getenv("DART_STANDARD_FULL_YEAR_FROM", "2015"))
DART_FULL_TO = int(os.getenv("DART_STANDARD_FULL_YEAR_TO", "2026"))
DART_STANDARD_CSV = PROJECT_ROOT / f"docs/stocks/dart_standard_{DART_FULL_FROM}_{DART_FULL_TO}.csv"

COMPRESSION = os.getenv("PANEL_PARQUET_COMPRESSION", "zstd")  # zstd/snappy

import re

def _normalize_date_series(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()

    # case A) YYYY-MMDD (예: 2026-0220) -> YYYY-MM-DD
    m = x.str.match(r"^\d{4}-\d{4}$", na=False)
    if m.any():
        tmp = x[m].str.replace("-", "", regex=False)  # YYYYMMDD
        x.loc[m] = tmp.str.slice(0, 4) + "-" + tmp.str.slice(4, 6) + "-" + tmp.str.slice(6, 8)

    # case B) YYYYMMDD -> YYYY-MM-DD
    m = x.str.match(r"^\d{8}$", na=False)
    if m.any():
        x.loc[m] = x[m].str.slice(0, 4) + "-" + x[m].str.slice(4, 6) + "-" + x[m].str.slice(6, 8)

    return x

def _read_prices_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")

    # date 컬럼명 정규화
    if "date" not in df.columns:
        df.rename(columns={df.columns[0]: "date"}, inplace=True)

    # ✅ robust date parsing
    df["date"] = _normalize_date_series(df["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 가격 컬럼 표준화 + 숫자 변환
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _read_parquet_if_exists(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _dart_report_end_date(year: int, reprt_code: str) -> pd.Timestamp:
    # Simple mapping (business-year 기준):
    # 11011: annual -> 12/31
    # 11012: Q1 -> 03/31
    # 11013: half -> 06/30
    # 11014: Q3 -> 09/30
    rc = str(reprt_code)
    if rc == "11011":
        mmdd = (12, 31)
    elif rc == "11012":
        mmdd = (3, 31)
    elif rc == "11013":
        mmdd = (6, 30)
    elif rc == "11014":
        mmdd = (9, 30)
    else:
        mmdd = (12, 31)
    return pd.Timestamp(year=int(year), month=mmdd[0], day=mmdd[1])


def _load_dart_standard() -> pd.DataFrame:
    if not DART_STANDARD_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(DART_STANDARD_CSV, encoding="utf-8-sig")
    need = {"ticker", "year", "reprt_code", "revenue", "operating_income", "net_income", "equity", "roe"}
    miss = need - set(df.columns)
    if miss:
        print(f"[panel][WARN] DART standard missing cols: {miss} in {DART_STANDARD_CSV}")
        return pd.DataFrame()

    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["reprt_code"] = df["reprt_code"].astype(str)

    for c in ["revenue", "operating_income", "net_income", "equity", "roe"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["ticker", "year", "reprt_code"]).copy()
    df["asof_date"] = [
        _dart_report_end_date(int(y), rc) if pd.notna(y) else pd.NaT
        for y, rc in zip(df["year"], df["reprt_code"])
    ]
    df = df.dropna(subset=["asof_date"]).sort_values(["ticker", "asof_date", "reprt_code"]).reset_index(drop=True)
    return df


def _clamp_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    out = df
    if PANEL_START_DATE:
        out = out[out["date"] >= pd.to_datetime(PANEL_START_DATE)]
    if PANEL_END_DATE:
        out = out[out["date"] <= pd.to_datetime(PANEL_END_DATE)]
    return out


def main():
    if not MASTER.exists():
        raise FileNotFoundError(f"missing: {MASTER} (run fetch_listings.py first)")

    listings = pd.read_parquet(MASTER).copy()
    listings["ticker"] = listings["ticker"].astype(str).str.zfill(6)
    listings["name"] = listings.get("name", "").astype(str)
    listings["market"] = listings.get("market", "").astype(str)

    dart_std = _load_dart_standard()
    has_dart = not dart_std.empty
    if has_dart:
        print(f"[panel] DART standard loaded: rows={len(dart_std):,} file={DART_STANDARD_CSV}")
    else:
        print(f"[panel][WARN] DART standard not found/empty -> will build panel without DART columns: {DART_STANDARD_CSV}")

    writer = None
    total_rows = 0
    tickers_written = 0
    tickers_skipped = 0

    for i, r in listings.iterrows():
        ticker = r["ticker"]
        name = r.get("name", "")
        market = r.get("market", "")

        prices_path = PRICES_DIR / f"{ticker}.csv"
        flows_path = CURATED_DIR / ticker / "flows_daily.parquet"
        fund_path = CURATED_DIR / ticker / "fundamentals_daily.parquet"

        if not prices_path.exists():
            tickers_skipped += 1
            continue

        try:
            df_p = _read_prices_csv(prices_path)
            df_p = _clamp_dates(df_p)
            if df_p.empty:
                tickers_skipped += 1
                continue

            df = df_p.copy()

            # Merge flows (curated)
            df_f = _read_parquet_if_exists(flows_path)
            if not df_f.empty:
                df = df.merge(df_f, on="date", how="left")

            # Merge fundamentals (curated)
            df_u = _read_parquet_if_exists(fund_path)
            if not df_u.empty:
                keep = [c for c in df_u.columns if c in ["date", "bps", "per", "pbr", "eps", "div", "dps"]]
                if "date" not in keep:
                    keep = ["date"]
                df = df.merge(df_u[keep], on="date", how="left")

            # Merge DART standard by asof join (last report <= date)
            if has_dart:
                ds = dart_std[dart_std["ticker"] == ticker].copy()
                if not ds.empty:
                    ds = ds.sort_values("asof_date")
                    left = df.sort_values("date").copy()

                    # merge_asof expects both keys sorted
                    merged = pd.merge_asof(
                        left,
                        ds[["asof_date", "reprt_code", "revenue", "operating_income", "net_income", "equity", "roe"]].sort_values("asof_date"),
                        left_on="date",
                        right_on="asof_date",
                        direction="backward",
                        allow_exact_matches=True,
                    )
                    merged.rename(columns={
                        "asof_date": "dart_asof_date",
                        "reprt_code": "dart_reprt_code",
                        "revenue": "dart_revenue",
                        "operating_income": "dart_operating_income",
                        "net_income": "dart_net_income",
                        "equity": "dart_equity",
                        "roe": "dart_roe",
                    }, inplace=True)
                    df = merged

            # Add master columns
            df.insert(0, "ticker", ticker)
            df.insert(1, "name", name)
            df.insert(2, "market", market)

            # Ensure dtypes
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

            # Write batch
            table = pa.Table.from_pandas(df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(str(OUT_PARQUET), table.schema, compression=COMPRESSION)

            writer.write_table(table)
            total_rows += len(df)
            tickers_written += 1

            if tickers_written <= 5 or tickers_written % 300 == 0:
                print(f"[panel] [{tickers_written:,}] {ticker} OK rows={len(df):,}")

        except Exception as e:
            tickers_skipped += 1
            if tickers_skipped <= 20:
                print(f"[panel] {ticker} SKIP/ERR: {e}")

    if writer is not None:
        writer.close()

    print("[panel] DONE")
    print(f"  out={OUT_PARQUET}")
    print(f"  tickers_written={tickers_written:,} tickers_skipped={tickers_skipped:,}")
    print(f"  total_rows={total_rows:,}")
    print(f"  dart_standard_used={'yes' if has_dart else 'no'}")


if __name__ == "__main__":
    main()
