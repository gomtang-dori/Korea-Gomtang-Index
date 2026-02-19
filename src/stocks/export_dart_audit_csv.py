#!/usr/bin/env python3
"""
Export DART raw cache audit CSV for verification (reprt_code/year/status)
Input:  data/stocks/raw/dart/<ticker>/<year>_<reprt_code>_ALL.json
Output: docs/stocks/dart_audit_<YEAR_FROM>_<YEAR_TO>.csv
"""

import os, json, re
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")) if os.getenv("PROJECT_ROOT") else Path.cwd()
RAW_DIR = PROJECT_ROOT / "data/stocks/raw/dart"
OUT_DIR = PROJECT_ROOT / "docs/stocks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_FROM = int(os.getenv("DART_YEAR_FROM", "2015"))
YEAR_TO = int(os.getenv("DART_YEAR_TO", "2017"))

pat = re.compile(r"^(?P<year>\d{4})_(?P<reprt>\d{5})_ALL\.json$")

def main():
    rows = []
    if not RAW_DIR.exists():
        print(f"raw dir not found: {RAW_DIR}")
        return

    for ticker_dir in RAW_DIR.iterdir():
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name

        for fp in ticker_dir.glob("*.json"):
            m = pat.match(fp.name)
            if not m:
                continue
            year = int(m.group("year"))
            if year < YEAR_FROM or year > YEAR_TO:
                continue
            reprt_code = m.group("reprt")

            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                meta = (data.get("_meta") or {})
                status = str(data.get("status", "")).strip()
                message = str(data.get("message", "")).strip()
                lst = data.get("list") or []
                n_list = len(lst) if isinstance(lst, list) else 0
                rows.append({
                    "ticker": ticker,
                    "name": meta.get("name", ""),
                    "year": year,
                    "reprt_code": reprt_code,
                    "status": status,
                    "message": message,
                    "n_list": n_list,
                    "fetched_utc": meta.get("fetched_utc", ""),
                    "file": str(fp.relative_to(PROJECT_ROOT)),
                })
            except Exception as e:
                rows.append({
                    "ticker": ticker,
                    "name": "",
                    "year": year,
                    "reprt_code": reprt_code,
                    "status": "PARSE_ERR",
                    "message": str(e),
                    "n_list": 0,
                    "fetched_utc": "",
                    "file": str(fp.relative_to(PROJECT_ROOT)),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("no audit rows")
        return

    df.sort_values(["ticker", "year", "reprt_code"], inplace=True)
    out_path = OUT_DIR / f"dart_audit_{YEAR_FROM}_{YEAR_TO}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[export_dart_audit_csv] wrote: {out_path} rows={len(df):,}")
    print(df.groupby(["reprt_code", "status"]).size().reset_index(name="cnt").head(20))

if __name__ == "__main__":
    main()
