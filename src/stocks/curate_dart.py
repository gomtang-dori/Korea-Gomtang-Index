#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Curate OpenDART raw JSON cache -> LONG parquet per ticker (+ optional standard CSV)

Input (raw cache):
- data/stocks/raw/dart/<ticker>/<year>_<reprt_code>_ALL.json

Output:
- LONG parquet (per ticker):
  data/stocks/curated/<ticker>/dart_accounts_long.parquet

- OPTIONAL: standard summary CSV (for quick validation / dashboards):
  docs/stocks/dart_standard_<YEAR_FROM>_<YEAR_TO>.csv

Env:
- DART_YEAR_FROM (default: 2015)
- DART_YEAR_TO   (default: 2017)
- DART_REPRT_CODES (default: "11011,11012,11013,11014")
- DART_RAW_DIR (default: "data/stocks/raw/dart")
- DART_CURATED_DIR (default: "data/stocks/curated")
- DART_LISTINGS_PATH (default: "data/stocks/master/listings.parquet")
- DART_LONG_OVERWRITE (default: "false")  # true면 parquet 재생성
- DART_WRITE_STANDARD_CSV (default: "true")
- DART_STANDARD_FS_DIV_PRIORITY (default: "CFS,OFS")  # standard 만들 때 우선순위
"""

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


FILE_RE = re.compile(r"(?P<year>\d{4})_(?P<reprt>11011|11012|11013|11014)_ALL\.json$")


def env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def parse_codes(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def to_number(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s == "-" or s.lower() == "nan":
        return None
    # keep minus sign
    s = s.replace(",", "")
    # sometimes amounts come with spaces
    s = s.replace(" ", "")
    try:
        return float(s)
    except Exception:
        return None


def load_listings_map(listings_path: Path) -> Dict[str, Dict[str, str]]:
    """
    returns: {ticker: {"name":..., "market":..., ...}}
    """
    if not listings_path.exists():
        print(f"[curate_dart] WARN listings not found: {listings_path} (name mapping will be empty)")
        return {}

    df = pd.read_parquet(listings_path)
    cols = {c.lower(): c for c in df.columns}
    # tolerate common variants
    ticker_col = cols.get("ticker") or cols.get("code") or cols.get("stock_code")
    name_col = cols.get("name") or cols.get("corp_name") or cols.get("company")
    market_col = cols.get("market") or cols.get("market_name")

    out: Dict[str, Dict[str, str]] = {}
    for _, r in df.iterrows():
        t = str(r[ticker_col]).strip() if ticker_col else None
        if not t:
            continue
        out[t] = {
            "name": str(r[name_col]).strip() if name_col else "",
            "market": str(r[market_col]).strip() if market_col else "",
        }
    return out


def iter_raw_json_files(raw_ticker_dir: Path, year_from: int, year_to: int, reprt_codes: set) -> List[Tuple[int, str, Path]]:
    jobs: List[Tuple[int, str, Path]] = []
    if not raw_ticker_dir.exists():
        return jobs

    for p in raw_ticker_dir.glob("*.json"):
        m = FILE_RE.search(p.name)
        if not m:
            continue
        y = int(m.group("year"))
        rc = m.group("reprt")
        if y < year_from or y > year_to:
            continue
        if rc not in reprt_codes:
            continue
        jobs.append((y, rc, p))

    jobs.sort(key=lambda x: (x[0], x[1], x[2].name))
    return jobs


def read_one_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_to_long_rows(
    ticker: str,
    name: str,
    year: int,
    reprt_code: str,
    payload: Dict,
    source_file: str,
) -> List[Dict]:
    status = str(payload.get("status", "")).strip()
    message = str(payload.get("message", "")).strip()
    fetched_utc = payload.get("fetched_utc", "")

    # status 000 only contains useful list; 013 is "no data"
    items = payload.get("list") or []
    if status != "000" or not isinstance(items, list) or len(items) == 0:
        return []

    rows: List[Dict] = []
    for it in items:
        # Keep the raw DART fields as much as possible for later mapping
        row = {
            "ticker": ticker,
            "name": name,
            "bsns_year": str(it.get("bsns_year") or year),
            "reprt_code": str(it.get("reprt_code") or reprt_code),

            "rcept_no": it.get("rcept_no"),
            "stock_code": it.get("stock_code"),

            "fs_div": it.get("fs_div"),
            "fs_nm": it.get("fs_nm"),
            "sj_div": it.get("sj_div"),
            "sj_nm": it.get("sj_nm"),

            "account_nm": it.get("account_nm"),
            "ord": it.get("ord"),
            "currency": it.get("currency"),

            "thstrm_nm": it.get("thstrm_nm"),
            "thstrm_dt": it.get("thstrm_dt"),
            "thstrm_amount": to_number(it.get("thstrm_amount")),
            "thstrm_add_amount": to_number(it.get("thstrm_add_amount")),

            "frmtrm_nm": it.get("frmtrm_nm"),
            "frmtrm_dt": it.get("frmtrm_dt"),
            "frmtrm_amount": to_number(it.get("frmtrm_amount")),
            "frmtrm_add_amount": to_number(it.get("frmtrm_add_amount")),

            "bfefrmtrm_nm": it.get("bfefrmtrm_nm"),
            "bfefrmtrm_dt": it.get("bfefrmtrm_dt"),
            "bfefrmtrm_amount": to_number(it.get("bfefrmtrm_amount")),
            "bfefrmtrm_add_amount": to_number(it.get("bfefrmtrm_add_amount")),

            "status": status,
            "message": message,
            "fetched_utc": fetched_utc,
            "source_file": source_file,
        }
        rows.append(row)

    return rows


def pick_metric_amount(df: pd.DataFrame, candidates: List[str], fs_priority: List[str], sj_div: Optional[str] = None) -> Optional[float]:
    """
    df: long ledger for a single (ticker, year, reprt_code)
    """
    if df.empty:
        return None
    x = df.copy()

    if sj_div is not None and "sj_div" in x.columns:
        x = x[x["sj_div"] == sj_div]

    if "account_nm" not in x.columns:
        return None

    x = x[x["account_nm"].isin(candidates)]
    if x.empty:
        return None

    # prefer CFS then OFS (or env-defined)
    if "fs_div" in x.columns and fs_priority:
        x["fs_rank"] = x["fs_div"].apply(lambda v: fs_priority.index(v) if v in fs_priority else 999)
        x = x.sort_values(["fs_rank"], ascending=True)

    # choose the first non-null thstrm_amount
    for v in x["thstrm_amount"].tolist():
        if v is not None and pd.notna(v):
            return float(v)
    return None


def build_standard_summary(
    long_df: pd.DataFrame,
    fs_priority: List[str],
) -> pd.DataFrame:
    """
    Produce a compact table per (ticker, year, reprt_code)
    NOTE: Heuristic mapping; you can refine later.
    """
    if long_df.empty:
        return pd.DataFrame()

    # candidate account names (heuristic; adjust later as needed)
    REV = ["매출액", "수익(매출액)", "영업수익"]
    OP  = ["영업이익"]
    NI  = ["당기순이익", "당기순이익(손실)", "연결당기순이익", "지배기업소유주지분당기순이익"]
    EQ  = ["자본총계", "자본총계(지배기업소유주지분)", "자본총계(지배기업소유주지분)"]

    group_cols = ["ticker", "name", "bsns_year", "reprt_code"]
    out_rows = []

    for (ticker, name, y, rc), g in long_df.groupby(group_cols, dropna=False):
        g2 = g.copy()
        # numeric year
        year_i = int(str(y)[:4]) if str(y).isdigit() else None

        revenue = pick_metric_amount(g2, REV, fs_priority, sj_div="IS")
        op      = pick_metric_amount(g2, OP,  fs_priority, sj_div="IS")
        ni      = pick_metric_amount(g2, NI,  fs_priority, sj_div="IS")
        equity  = pick_metric_amount(g2, EQ,  fs_priority, sj_div="BS")

        roe = None
        if ni is not None and equity is not None and equity != 0:
            roe = float(ni) / float(equity)

        out_rows.append({
            "ticker": ticker,
            "name": name,
            "year": year_i,
            "reprt_code": rc,
            "revenue": revenue,
            "operating_income": op,
            "net_income": ni,
            "equity": equity,
            "roe": roe,
        })

    out = pd.DataFrame(out_rows)
    # stable sort: year then reprt_code
    if not out.empty:
        out = out.sort_values(["ticker", "year", "reprt_code"], ascending=True)
    return out


def main():
    year_from = int(os.getenv("DART_YEAR_FROM", "2015"))
    year_to = int(os.getenv("DART_YEAR_TO", "2017"))
    reprt_codes = set(parse_codes(os.getenv("DART_REPRT_CODES", "11011,11012,11013,11014")))

    raw_dir = Path(os.getenv("DART_RAW_DIR", "data/stocks/raw/dart"))
    curated_dir = Path(os.getenv("DART_CURATED_DIR", "data/stocks/curated"))
    listings_path = Path(os.getenv("DART_LISTINGS_PATH", "data/stocks/master/listings.parquet"))

    overwrite = env_bool("DART_LONG_OVERWRITE", False)
    write_standard = env_bool("DART_WRITE_STANDARD_CSV", True)

    fs_priority = parse_codes(os.getenv("DART_STANDARD_FS_DIV_PRIORITY", "CFS,OFS"))
    if not fs_priority:
        fs_priority = ["CFS", "OFS"]

    print("[curate_dart] config")
    print(f"  year_from={year_from} year_to={year_to}")
    print(f"  reprt_codes={sorted(list(reprt_codes))}")
    print(f"  raw_dir={raw_dir}")
    print(f"  curated_dir={curated_dir}")
    print(f"  listings_path={listings_path}")
    print(f"  overwrite={overwrite}")
    print(f"  write_standard={write_standard}")
    print(f"  fs_priority={fs_priority}")

    listings_map = load_listings_map(listings_path)

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")

    ticker_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])

    ok, skip, err = 0, 0, 0
    all_long_for_standard = []  # collect minimal for standard CSV (can be big; OK per year-slice)

    for td in ticker_dirs:
        ticker = td.name
        name = listings_map.get(ticker, {}).get("name", "")

        out_dir = curated_dir / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        out_parquet = out_dir / "dart_accounts_long.parquet"

        if out_parquet.exists() and not overwrite:
            skip += 1
            continue

        try:
            jobs = iter_raw_json_files(td, year_from, year_to, reprt_codes)
            if not jobs:
                # no raw files in range -> skip
                skip += 1
                continue

            rows = []
            for y, rc, fp in jobs:
                payload = read_one_json(fp)
                rows.extend(json_to_long_rows(
                    ticker=ticker,
                    name=name,
                    year=y,
                    reprt_code=rc,
                    payload=payload,
                    source_file=str(fp),
                ))

            if not rows:
                # all were status!=000 or empty list
                skip += 1
                continue

            df = pd.DataFrame(rows)

            # normalize dtypes
            for c in ["bsns_year", "reprt_code", "fs_div", "sj_div", "account_nm"]:
                if c in df.columns:
                    df[c] = df[c].astype(str)

            # Keep long parquet compact: sort & dedup
            # (Sometimes same account rows can appear; use a conservative key)
            key_cols = ["ticker", "bsns_year", "reprt_code", "fs_div", "sj_div", "account_nm", "ord", "thstrm_dt"]
            existing = [c for c in key_cols if c in df.columns]
            if existing:
                df = df.drop_duplicates(subset=existing, keep="last")

            sort_cols = [c for c in ["ticker", "bsns_year", "reprt_code", "fs_div", "sj_div", "ord", "account_nm"] if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols, ascending=True)

            df.to_parquet(out_parquet, index=False)
            ok += 1

            if write_standard:
                # keep only columns needed for standard computation to reduce memory
                keep_cols = ["ticker", "name", "bsns_year", "reprt_code", "fs_div", "sj_div", "account_nm", "thstrm_amount"]
                keep_cols = [c for c in keep_cols if c in df.columns]
                all_long_for_standard.append(df[keep_cols].copy())

        except Exception as e:
            err += 1
            print(f"[curate_dart] ERROR ticker={ticker} {e}")

    print("[curate_dart] result")
    print(f"  OK={ok} SKIP={skip} ERROR={err}")

    if write_standard:
        docs_dir = Path("docs/stocks")
        docs_dir.mkdir(parents=True, exist_ok=True)
        out_csv = docs_dir / f"dart_standard_{year_from}_{year_to}.csv"

        if len(all_long_for_standard) == 0:
            print(f"[curate_dart] WARN no rows for standard csv -> not writing: {out_csv}")
        else:
            long_df = pd.concat(all_long_for_standard, ignore_index=True)
            std = build_standard_summary(long_df, fs_priority=fs_priority)
            std.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"[curate_dart] wrote standard csv: {out_csv} rows={len(std):,}")


if __name__ == "__main__":
    main()
