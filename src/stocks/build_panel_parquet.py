#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build unified stock panel parquet (single file, overwrite)

Inputs:
- listings: data/stocks/master/listings.parquet
- prices (raw): data/stocks/raw/prices/{ticker}.csv
- flows (curated): data/stocks/curated/{ticker}/flows_daily.parquet
- fundamentals (curated): data/stocks/curated/{ticker}/fundamentals_daily.parquet
- market cap (curated): data/stocks/curated/{ticker}/market_cap_daily.parquet
- WICS map: data/stocks/master/wics_map.parquet
- DART standard summary CSV (optional): docs/stocks/dart_standard_2015_2026.csv

Env:
- PANEL_SCOPE: "5y" or "all" (default 5y)
- PANEL_OUT_PATH: output parquet path (override)
- PANEL_PARQUET_COMPRESSION: zstd/snappy (default zstd)
- DART_STD_PATH: path to dart standard csv

Behavior:
- 5y scope: per-ticker max(date) 기준 최근 5년만 포함
- DART: merge_asof(backward)로 최신 리포트를 일자에 carry
- WICS: ticker 기준 상수 컬럼(wics_major/wics_mid)로 부착
- MarketCap: date 기준 market_cap/shares/value merge
"""

import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path.cwd()

LISTINGS_PATH = PROJECT_ROOT / "data/stocks/master/listings.parquet"
PRICES_DIR = PROJECT_ROOT / "data/stocks/raw/prices"
CURATED_DIR = PROJECT_ROOT / "data/stocks/curated"

WICS_MAP_PATH = PROJECT_ROOT / "data/stocks/master/wics_map.parquet"

DART_STD_PATH = Path(os.getenv("DART_STD_PATH", "docs/stocks/dart_standard_2015_2026.csv"))
PANEL_SCOPE = (os.getenv("PANEL_SCOPE", "5y") or "5y").strip().lower()  # 5y|all
OUT_PATH = Path(os.getenv("PANEL_OUT_PATH", "") or "")

PARQUET_COMPRESSION = (os.getenv("PANEL_PARQUET_COMPRESSION", "zstd") or "zstd").strip()


def _normalize_date_series(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()

    # YYYY-MMDD (e.g., 2026-0220) -> YYYY-MM-DD
    m = x.str.match(r"^\d{4}-\d{4}$", na=False)
    if m.any():
        tmp = x[m].str.replace("-", "", regex=False)  # YYYYMMDD
        x.loc[m] = tmp.str.slice(0, 4) + "-" + tmp.str.slice(4, 6) + "-" + tmp.str.slice(6, 8)

    # YYYYMMDD -> YYYY-MM-DD
    m = x.str.match(r"^\d{8}$", na=False)
    if m.any():
        x.loc[m] = x[m].str.slice(0, 4) + "-" + x[m].str.slice(4, 6) + "-" + x[m].str.slice(6, 8)

    return x


def _read_prices(ticker: str) -> pd.DataFrame:
    p = PRICES_DIR / f"{ticker}.csv"
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p, encoding="utf-8-sig")
    if df.empty:
        return pd.DataFrame()

    if "date" not in df.columns:
        df.rename(columns={df.columns[0]: "date"}, inplace=True)

    df["date"] = _normalize_date_series(df["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume"]].copy()


def _read_parquet_if_exists(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if df is None or df.empty:
        return pd.DataFrame()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _load_wics_map() -> pd.DataFrame:
    if not WICS_MAP_PATH.exists():
        print(f"[panel][WARN] WICS map missing: {WICS_MAP_PATH}")
        return pd.DataFrame()
    w = pd.read_parquet(WICS_MAP_PATH).copy()
    if w.empty:
        print(f"[panel][WARN] WICS map empty: {WICS_MAP_PATH}")
        return pd.DataFrame()
    w["ticker"] = w["ticker"].astype(str).str.zfill(6)
    # 기대 컬럼명: wics_major, wics_mid
    for c in ["wics_major", "wics_mid"]:
        if c not in w.columns:
            w[c] = pd.NA
    w = w[["ticker", "wics_major", "wics_mid"]].drop_duplicates(subset=["ticker"], keep="last")
    print(f"[panel] WICS loaded: rows={len(w):,} file={WICS_MAP_PATH}")
    return w


def _load_dart_std() -> pd.DataFrame:
    if not DART_STD_PATH.exists():
        print(f"[panel][WARN] DART standard CSV missing: {DART_STD_PATH} -> continue without DART")
        return pd.DataFrame()

    d = pd.read_csv(DART_STD_PATH, encoding="utf-8-sig")
    need = {"ticker", "year", "reprt_code", "revenue", "operating_income", "net_income", "equity", "roe"}
    if not need.issubset(set(d.columns)):
        print(f"[panel][WARN] DART standard CSV columns missing -> continue without DART: {DART_STD_PATH}")
        return pd.DataFrame()

    d["ticker"] = d["ticker"].astype(str).str.zfill(6)
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["reprt_code"] = d["reprt_code"].astype(str)

    m = {"11011": (12, 31), "11012": (3, 31), "11013": (6, 30), "11014": (9, 30)}

    def to_asof(r):
        y = r["year"]
        rc = r["reprt_code"]
        if pd.isna(y) or rc not in m:
            return pd.NaT
        mm, dd = m[rc]
        return pd.Timestamp(int(y), mm, dd)

    d["dart_asof_date"] = d.apply(to_asof, axis=1)

    for c in ["revenue", "operating_income", "net_income", "equity", "roe"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.rename(
        columns={
            "revenue": "dart_revenue",
            "operating_income": "dart_operating_income",
            "net_income": "dart_net_income",
            "equity": "dart_equity",
            "roe": "dart_roe",
            "reprt_code": "dart_reprt_code",
        }
    )

    keep = [
        "ticker",
        "dart_asof_date",
        "dart_reprt_code",
        "dart_revenue",
        "dart_operating_income",
        "dart_net_income",
        "dart_equity",
        "dart_roe",
    ]
    d = d[keep].dropna(subset=["ticker", "dart_asof_date"]).sort_values(["ticker", "dart_asof_date"])
    print(f"[panel] DART loaded: rows={len(d):,} file={DART_STD_PATH}")
    return d


def _apply_scope_5y(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    maxd = df["date"].max()
    cutoff = maxd - pd.DateOffset(years=5)
    return df[df["date"] >= cutoff].copy()


def _determine_out_path() -> Path:
    if str(OUT_PATH).strip():
        return OUT_PATH
    if PANEL_SCOPE == "all":
        return PROJECT_ROOT / "data/stocks/mart/panel_all.parquet"
    return PROJECT_ROOT / "data/stocks/mart/panel_5y.parquet"


def main():
    if not LISTINGS_PATH.exists():
        raise FileNotFoundError(f"Missing listings: {LISTINGS_PATH}")

    out_path = _determine_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    listings = pd.read_parquet(LISTINGS_PATH).copy()
    listings["ticker"] = listings["ticker"].astype(str).str.zfill(6)
    listings["name"] = listings.get("name", "").astype(str)
    listings["market"] = listings.get("market", "").astype(str)

    wics = _load_wics_map()
    dart = _load_dart_std()

    writer = None
    total_rows = 0
    written_tickers = 0
    skipped_tickers = 0

    for _, r in listings.iterrows():
        ticker = r["ticker"]
        name = r["name"]
        market = r["market"]

        px = _read_prices(ticker)
        if px.empty:
            skipped_tickers += 1
            continue

        if PANEL_SCOPE == "5y":
            px = _apply_scope_5y(px)

        df = px.copy()
        df.insert(0, "ticker", ticker)
        df.insert(1, "name", name)
        df.insert(2, "market", market)

        # --- WICS (always create cols) ---
        df["wics_major"] = pd.NA
        df["wics_mid"] = pd.NA
        if not wics.empty:
            ww = wics[wics["ticker"] == ticker]
            if not ww.empty:
                df["wics_major"] = ww["wics_major"].iloc[0]
                df["wics_mid"] = ww["wics_mid"].iloc[0]

        # --- flows ---
        flows = _read_parquet_if_exists(CURATED_DIR / ticker / "flows_daily.parquet")
        if not flows.empty:
            df = df.merge(flows, on="date", how="left")

        # --- fundamentals ---
        fund = _read_parquet_if_exists(CURATED_DIR / ticker / "fundamentals_daily.parquet")
        if not fund.empty:
            keep = [c for c in ["date", "bps", "per", "pbr", "eps", "div", "dps"] if c in fund.columns]
            if "date" not in keep:
                keep = ["date"]
            df = df.merge(fund[keep], on="date", how="left")

        # --- market cap (always create cols) ---
        df["market_cap"] = pd.NA
        df["shares"] = pd.NA
        df["value"] = pd.NA
        mc = _read_parquet_if_exists(CURATED_DIR / ticker / "market_cap_daily.parquet")
        if not mc.empty:
            keep_mc = [c for c in ["date", "market_cap", "shares", "value"] if c in mc.columns]
            if "date" in keep_mc and len(keep_mc) > 1:
                mc2 = mc[keep_mc].drop_duplicates(subset=["date"], keep="last").sort_values("date")
                df = df.merge(mc2, on="date", how="left", suffixes=("", "_mc"))
                # merge 결과가 *_mc로 붙는 경우를 방지(안전)
                for c in ["market_cap", "shares", "value"]:
                    if f"{c}_mc" in df.columns and c in df.columns:
                        df[c] = df[c].combine_first(df[f"{c}_mc"])
                        df.drop(columns=[f"{c}_mc"], inplace=True)

        # --- DART (asof join) ---
        if not dart.empty:
            d = dart[dart["ticker"] == ticker].sort_values("dart_asof_date")
            if not d.empty:
                df = df.sort_values("date")
                df = pd.merge_asof(
                    df,
                    d,
                    left_on="date",
                    right_on="dart_asof_date",
                    direction="backward",
                    allow_exact_matches=True,
                )

        df = df.sort_values("date").reset_index(drop=True)

        table = pa.Table.from_pandas(df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(out_path.as_posix(), table.schema, compression=PARQUET_COMPRESSION)

        # schema align (writer schema 기준으로 누락 컬럼 채움)
        if table.schema != writer.schema:
            for field in writer.schema:
                if field.name not in table.column_names:
                    table = table.append_column(field.name, pa.array([None] * table.num_rows, type=field.type))
            keep_names = [f.name for f in writer.schema]
            table = table.select(keep_names)

        writer.write_table(table)
        total_rows += table.num_rows
        written_tickers += 1

        if written_tickers <= 3 or written_tickers % 500 == 0:
            print(f"[panel] {written_tickers} tickers written, last={ticker}, rows_total={total_rows:,}")

    if writer:
        writer.close()

    print(f"[panel] DONE scope={PANEL_SCOPE} -> {out_path} rows={total_rows:,} tickers={written_tickers:,} skipped={skipped_tickers:,}")


if __name__ == "__main__":
    main()
