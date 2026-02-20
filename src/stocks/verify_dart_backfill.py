#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verify DART raw cache coverage & integrity for a year range.

Raw cache layout (expected):
- data/stocks/raw/dart/<ticker>/<year>_<reprt_code>_ALL.json

This verifier checks:
- Coverage: missing combinations vs expected tickers/years/reprt_codes
- JSON validity
- Minimal schema: status/message/_meta presence
- Meta consistency: _meta.ticker/_meta.bsns_year/_meta.reprt_code match path/filename
- Anomalies: status=000 but list empty, unexpected status codes, tiny files, etc.

Outputs (default docs/stocks):
- dart_verify_details_<YF>_<YT>.csv      (one row per expected combo)
- dart_verify_issues_<YF>_<YT>.csv       (problem rows only)
- dart_verify_summary_<YF>_<YT>.txt      (human-readable summary)

Env:
- DART_VERIFY_YEAR_FROM (default 2019)
- DART_VERIFY_YEAR_TO   (default 2026)
- DART_VERIFY_REPRT_CODES (default "11011,11012,11013,11014")
- DART_RAW_DIR (default "data/stocks/raw/dart")
- DART_LISTINGS_PATH (default "data/stocks/master/listings.parquet")
- DART_CORPCODE_XML (default "data/stocks/raw/dart/_corpcode_cache/CORPCODE.xml")
- DART_VERIFY_OUT_DIR (default "docs/stocks")
- DART_VERIFY_EXPECT_MODE (default "corp_mapped_or_raw")
    * "corp_mapped_or_raw": (best) if corpCode xml exists -> expected tickers = listings ∩ corp_mapped
                           else expected tickers = listings (fallback)
                           if listings missing -> expected tickers = raw ticker dirs
    * "listings": expected tickers = listings only (no corp filter)
    * "raw_only": expected tickers = raw ticker dirs only
- DART_VERIFY_MAX_TICKERS (default 0 = no limit) : for quick local test
"""

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import xml.etree.ElementTree as ET


FILE_RE = re.compile(r"^(?P<year>\d{4})_(?P<reprt>11011|11012|11013|11014)_ALL\.json$")

def parse_codes(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]

def normalize_ticker6(x: str) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    digits = re.sub(r"[^0-9]", "", s)
    if not digits:
        return None
    if len(digits) > 6:
        digits = digits[-6:]
    return digits.zfill(6)

def load_listings(listings_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(listings_path)
    if "ticker" not in df.columns:
        raise ValueError(f"listings.parquet missing 'ticker' column: {listings_path}")
    df["ticker"] = df["ticker"].astype(str)
    df["ticker6"] = df["ticker"].map(normalize_ticker6)
    if "name" not in df.columns:
        df["name"] = ""
    return df

def parse_corp_map(corp_xml: Path) -> Dict[str, str]:
    """
    returns: {ticker6: corp_code}
    """
    root = ET.fromstring(corp_xml.read_bytes())
    m: Dict[str, str] = {}
    for el in root.findall("list"):
        corp_code = (el.findtext("corp_code") or "").strip()
        stock_code = (el.findtext("stock_code") or "").strip()
        if not corp_code or not stock_code:
            continue
        t6 = normalize_ticker6(stock_code)
        if t6:
            m[t6] = corp_code
    return m

def list_raw_tickers(raw_dir: Path) -> List[str]:
    if not raw_dir.exists():
        return []
    out = []
    for p in raw_dir.iterdir():
        if p.is_dir() and p.name.isdigit() and len(p.name) == 6:
            out.append(p.name)
    return sorted(out)

def read_json_safely(p: Path) -> Tuple[Optional[dict], Optional[str]]:
    try:
        txt = p.read_text(encoding="utf-8")
        data = json.loads(txt)
        return data, None
    except Exception as e:
        return None, str(e)

def main():
    yf = int(os.getenv("DART_VERIFY_YEAR_FROM", "2019"))
    yt = int(os.getenv("DART_VERIFY_YEAR_TO", "2026"))
    reprt_codes = parse_codes(os.getenv("DART_VERIFY_REPRT_CODES", "11011,11012,11013,11014"))

    raw_dir = Path(os.getenv("DART_RAW_DIR", "data/stocks/raw/dart"))
    listings_path = Path(os.getenv("DART_LISTINGS_PATH", "data/stocks/master/listings.parquet"))
    corp_xml = Path(os.getenv("DART_CORPCODE_XML", "data/stocks/raw/dart/_corpcode_cache/CORPCODE.xml"))
    out_dir = Path(os.getenv("DART_VERIFY_OUT_DIR", "docs/stocks"))
    expect_mode = os.getenv("DART_VERIFY_EXPECT_MODE", "corp_mapped_or_raw").strip().lower()
    max_tickers = int(os.getenv("DART_VERIFY_MAX_TICKERS", "0"))

    out_dir.mkdir(parents=True, exist_ok=True)

    years = list(range(yf, yt + 1))
    years_cnt = len(years)
    codes_cnt = len(reprt_codes)

    # Determine expected tickers + names
    expected_tickers: List[str] = []
    ticker_name: Dict[str, str] = {}

    raw_tickers = list_raw_tickers(raw_dir)

    listings_df = None
    if listings_path.exists():
        listings_df = load_listings(listings_path)

    corp_map = None
    if corp_xml.exists():
        try:
            corp_map = parse_corp_map(corp_xml)
        except Exception:
            corp_map = None

    if expect_mode == "raw_only":
        expected_tickers = raw_tickers

    elif expect_mode == "listings":
        if listings_df is None:
            raise FileNotFoundError(f"listings missing: {listings_path}")
        expected_tickers = sorted(listings_df["ticker6"].dropna().unique().tolist())
        ticker_name = {r["ticker6"]: str(r.get("name", "") or "") for _, r in listings_df.iterrows() if pd.notna(r.get("ticker6"))}

    else:
        # corp_mapped_or_raw (default)
        if listings_df is None:
            expected_tickers = raw_tickers
        else:
            ticker_name = {r["ticker6"]: str(r.get("name", "") or "") for _, r in listings_df.iterrows() if pd.notna(r.get("ticker6"))}
            if corp_map is not None:
                # only tickers that can be corp_mapped (matches fetch behavior)
                expected_tickers = sorted([t for t in listings_df["ticker6"].dropna().unique().tolist() if t in corp_map])
            else:
                expected_tickers = sorted(listings_df["ticker6"].dropna().unique().tolist())

        # if raw has more (edge), keep union for verification visibility
        # but mark "expected" set as above.
    if max_tickers and len(expected_tickers) > max_tickers:
        expected_tickers = expected_tickers[:max_tickers]

    expected_jobs = len(expected_tickers) * years_cnt * codes_cnt

    print("[dart_verify] config")
    print(f"  years={yf}..{yt} (count={years_cnt})")
    print(f"  reprt_codes={reprt_codes} (count={codes_cnt})")
    print(f"  raw_dir={raw_dir}")
    print(f"  listings_path_exists={listings_path.exists()}")
    print(f"  corp_xml_exists={corp_xml.exists()}")
    print(f"  expect_mode={expect_mode}")
    print(f"  expected_tickers={len(expected_tickers):,}")
    print(f"  expected_jobs={expected_jobs:,}")
    print(f"  raw_ticker_dirs={len(raw_tickers):,}")

    rows = []
    issues = []

    status_counts = {}
    issue_counts = {}

    def add_issue(kind: str, base: dict):
        issues.append({**base, "issue": kind})
        issue_counts[kind] = issue_counts.get(kind, 0) + 1

    for ti, ticker in enumerate(expected_tickers, 1):
        tdir = raw_dir / ticker
        for y in years:
            for rc in reprt_codes:
                fname = f"{y}_{rc}_ALL.json"
                fpath = tdir / fname

                base = {
                    "ticker": ticker,
                    "name": ticker_name.get(ticker, ""),
                    "year": y,
                    "reprt_code": rc,
                    "file_exists": fpath.exists(),
                    "file_size": fpath.stat().st_size if fpath.exists() else 0,
                    "status": "",
                    "message": "",
                    "list_len": None,
                    "meta_ticker": "",
                    "meta_year": None,
                    "meta_reprt": "",
                    "meta_ok": False,
                    "json_ok": False,
                    "path": str(fpath),
                }

                if not fpath.exists():
                    rows.append(base)
                    add_issue("MISSING_FILE", base)
                    continue

                if base["file_size"] < 50:
                    # too small to be a valid JSON from our fetch script
                    rows.append(base)
                    add_issue("TINY_FILE", base)
                    continue

                data, err = read_json_safely(fpath)
                if data is None:
                    rows.append(base)
                    add_issue(f"INVALID_JSON:{err}", base)
                    continue

                base["json_ok"] = True
                base["status"] = str(data.get("status", "")).strip()
                base["message"] = str(data.get("message", "")).strip()

                status_counts[base["status"]] = status_counts.get(base["status"], 0) + 1

                meta = data.get("_meta") or {}
                base["meta_ticker"] = str(meta.get("ticker6") or meta.get("ticker") or "").strip()
                base["meta_year"] = meta.get("bsns_year")
                base["meta_reprt"] = str(meta.get("reprt_code") or "").strip()

                # Validate meta
                meta_ok = True
                if base["meta_reprt"] and base["meta_reprt"] != rc:
                    meta_ok = False
                    add_issue("META_REPRT_MISMATCH", base)
                if base["meta_year"] is not None:
                    try:
                        if int(base["meta_year"]) != int(y):
                            meta_ok = False
                            add_issue("META_YEAR_MISMATCH", base)
                    except Exception:
                        meta_ok = False
                        add_issue("META_YEAR_INVALID", base)

                if base["meta_ticker"]:
                    # allow either 6-digit or raw string, but if numeric, compare 6-digit
                    mt = normalize_ticker6(base["meta_ticker"]) or base["meta_ticker"]
                    if mt.isdigit() and len(mt) == 6 and mt != ticker:
                        meta_ok = False
                        add_issue("META_TICKER_MISMATCH", base)

                base["meta_ok"] = meta_ok

                # list checks
                lst = data.get("list")
                if isinstance(lst, list):
                    base["list_len"] = len(lst)
                else:
                    base["list_len"] = None

                # anomaly: status=000 but empty list
                if base["status"] == "000" and (not isinstance(lst, list) or len(lst) == 0):
                    add_issue("STATUS_000_BUT_EMPTY_LIST", base)

                # anomaly: status not 000/013
                if base["status"] not in ("000", "013"):
                    add_issue(f"UNEXPECTED_STATUS:{base['status']}", base)

                rows.append(base)

        if ti % 200 == 0:
            print(f"  [progress] tickers {ti:,}/{len(expected_tickers):,}")

    details_df = pd.DataFrame(rows)
    issues_df = pd.DataFrame(issues)

    details_path = out_dir / f"dart_verify_details_{yf}_{yt}.csv"
    issues_path = out_dir / f"dart_verify_issues_{yf}_{yt}.csv"
    summary_path = out_dir / f"dart_verify_summary_{yf}_{yt}.txt"

    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")
    if not issues_df.empty:
        issues_df.to_csv(issues_path, index=False, encoding="utf-8-sig")
    else:
        # create empty issues file for artifact consistency
        pd.DataFrame(columns=list(details_df.columns) + ["issue"]).to_csv(issues_path, index=False, encoding="utf-8-sig")

    # Build summary text
    total = len(details_df)
    exists = int(details_df["file_exists"].sum())
    json_ok = int(details_df["json_ok"].sum())
    meta_ok = int(details_df["meta_ok"].sum())
    missing = total - exists

    status_series = details_df["status"].value_counts(dropna=False)

    # Top missing tickers
    miss_by_ticker = details_df[~details_df["file_exists"]].groupby("ticker")["reprt_code"].count().sort_values(ascending=False).head(30)

    # Issue counts
    issue_series = issues_df["issue"].value_counts() if not issues_df.empty else pd.Series(dtype=int)

    lines = []
    lines.append("====================================")
    lines.append(f"DART VERIFY SUMMARY  {yf}..{yt}")
    lines.append("====================================")
    lines.append(f"expected_tickers={len(expected_tickers):,}")
    lines.append(f"years_count={years_cnt}, reprt_codes_count={codes_cnt}")
    lines.append(f"expected_jobs={expected_jobs:,}")
    lines.append("")
    lines.append(f"files_exist={exists:,}/{total:,}  missing={missing:,}")
    lines.append(f"json_ok={json_ok:,}/{total:,}")
    lines.append(f"meta_ok={meta_ok:,}/{total:,}")
    lines.append("")
    lines.append("[status counts]")
    for k, v in status_series.items():
        kk = k if k != "" else "(empty)"
        lines.append(f"  {kk}: {int(v):,}")
    lines.append("")
    lines.append("[issue counts]")
    if issue_series.empty:
        lines.append("  (none)")
    else:
        for k, v in issue_series.items():
            lines.append(f"  {k}: {int(v):,}")
    lines.append("")
    lines.append("[top missing tickers (up to 30)]")
    if miss_by_ticker.empty:
        lines.append("  (none)")
    else:
        for t, c in miss_by_ticker.items():
            nm = ticker_name.get(t, "")
            lines.append(f"  {t} {nm} : missing={int(c):,}")
    lines.append("")
    lines.append(f"details_csv={details_path}")
    lines.append(f"issues_csv={issues_path}")
    lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[dart_verify] wrote: {details_path} rows={len(details_df):,}")
    print(f"[dart_verify] wrote: {issues_path} rows={len(issues_df):,}")
    print(f"[dart_verify] wrote: {summary_path}")

if __name__ == "__main__":
    main()
