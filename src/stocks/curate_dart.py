#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Curate OpenDART raw JSON cache -> LONG parquet per ticker
(A안) slice 범위만 overwrite(교체) + 전체는 누적

Input (raw cache):
- data/stocks/raw/dart/<ticker>/<year>_<reprt_code>_ALL.json

Output:
- LONG parquet (per ticker, 누적):
  data/stocks/curated/<ticker>/dart_accounts_long.parquet

- OPTIONAL: standard summary CSV (이번 slice 범위만)
  docs/stocks/dart_standard_<YEAR_FROM>_<YEAR_TO>.csv

Env:
- DART_YEAR_FROM (default: 2015)
- DART_YEAR_TO   (default: 2017)
- DART_REPRT_CODES (default: "11011,11012,11013,11014")
- DART_RAW_DIR (default: "data/stocks/raw/dart")
- DART_CURATED_DIR (default: "data/stocks/curated")
- DART_LISTINGS_PATH (default: "data/stocks/master/listings.parquet")

Mode:
- DART_CURATE_MODE (default: "upsert_slice")
  - "upsert_slice": 기존 parquet에서 (year_from~year_to & reprt_codes) 범위만 drop 후 새 slice append
  - "overwrite_all": 기존 parquet 무시하고 "이번 실행에서 생성된 slice만" 저장(주의: 누적 목적이면 비추)

Other:
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
    s = s.replace(",", "").replace(" ", "")
    try:
        return float(s)
    except Exception:
        return None


def safe_to_datetime_utc(series: pd.Series) -> pd.Series:
    # fetched_utc가 없거나 형식이 달라도 최대한 파싱
    # 파싱 실패는 NaT로
    return pd.to_datetime(series, errors="coerce", utc=True)


def load_listings_map(listings_path: Path) -> Dict[str, Dict[str, str]]:
    """
    returns: {ticker: {"name":..., "market":...}}
    """
    if not listings_path.exists():
        print(f"[curate_dart] WARN listings not found: {listings_path} (name mapping will be empty)")
        return {}

    df = pd.read_parquet(listings_path)
    cols = {c.lower(): c for c in df.columns}

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

    items = payload.get("list") or []
    if status != "000" or not isinstance(items, list) or len(items) == 0:
        return []

    rows: List[Dict] = []
    for it in items:
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


def normalize_long_df_types(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # 핵심 키 컬럼을 문자열로 통일
    for c in ["ticker", "name", "bsns_year", "reprt_code", "fs_div", "sj_div", "account_nm", "ord", "thstrm_dt", "currency"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # fetched_utc -> datetime(utc)
    if "fetched_utc" in df.columns:
        df["_fetched_dt"] = safe_to_datetime_utc(df["fetched_utc"])
    else:
        df["_fetched_dt"] = pd.NaT

    # source_file이 있다면 mtime으로도 tie-break
    if "source_file" in df.columns:
        # 파일이 없을 수도 있으니 예외 처리
        mtimes = []
        for p in df["source_file"].tolist():
            try:
                mt = Path(p).stat().st_mtime
            except Exception:
                mt = 0.0
            mtimes.append(mt)
        df["_source_mtime"] = mtimes
    else:
        df["_source_mtime"] = 0.0

    return df


def slice_mask(df: pd.DataFrame, year_from: int, year_to: int, reprt_codes: set) -> pd.Series:
    # bsns_year가 문자열일 수 있으니 앞 4자리 int 변환 시도
    if df.empty:
        return pd.Series([], dtype=bool)

    y = df["bsns_year"].astype(str).str.slice(0, 4)
    y_num = pd.to_numeric(y, errors="coerce")

    rc = df["reprt_code"].astype(str)
    return (y_num >= year_from) & (y_num <= year_to) & (rc.isin(list(reprt_codes)))


def dedup_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    LONG ledger 중복 제거:
    - 키는 "금액"을 포함하지 않음 (정정공시로 amount가 바뀌는 경우 업데이트를 반영하기 위해)
    - 충돌 시 fetched_utc 최신 우선, 그 다음 source_file mtime 우선
    """
    if df.empty:
        return df

    key_cols = [
        "ticker", "bsns_year", "reprt_code",
        "fs_div", "sj_div", "account_nm",
        "ord", "thstrm_dt", "currency",
    ]
    existing = [c for c in key_cols if c in df.columns]

    # 정렬: 최신 fetched_dt가 뒤로 가도록(keep last)
    sort_cols = []
    if "_fetched_dt" in df.columns:
        sort_cols.append("_fetched_dt")
    if "_source_mtime" in df.columns:
        sort_cols.append("_source_mtime")

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=True)

    if existing:
        df = df.drop_duplicates(subset=existing, keep="last")

    # 보기 좋은 정렬(안정적 저장)
    stable_sort = [c for c in ["ticker", "bsns_year", "reprt_code", "fs_div", "sj_div", "ord", "account_nm"] if c in df.columns]
    if stable_sort:
        df = df.sort_values(stable_sort, ascending=True)

    return df


def pick_metric_amount(df: pd.DataFrame, candidates: List[str], fs_priority: List[str], sj_div: Optional[str] = None) -> Optional[float]:
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

    if "fs_div" in x.columns and fs_priority:
        x["fs_rank"] = x["fs_div"].apply(lambda v: fs_priority.index(v) if v in fs_priority else 999)
        x = x.sort_values(["fs_rank"], ascending=True)

    for v in x["thstrm_amount"].tolist():
        if v is not None and pd.notna(v):
            return float(v)
    return None


def build_standard_summary(long_df: pd.DataFrame, fs_priority: List[str]) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    REV = ["매출액", "수익(매출액)", "영업수익"]
    OP  = ["영업이익"]
    NI  = ["당기순이익", "당기순이익(손실)", "연결당기순이익", "지배기업소유주지분당기순이익"]
    EQ  = ["자본총계", "자본총계(지배기업소유주지분)"]

    group_cols = ["ticker", "name", "bsns_year", "reprt_code"]
    out_rows = []

    for (ticker, name, y, rc), g in long_df.groupby(group_cols, dropna=False):
        g2 = g.copy()
        year_i = None
        try:
            year_i = int(str(y)[:4])
        except Exception:
            year_i = None

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

    mode = os.getenv("DART_CURATE_MODE", "upsert_slice").strip().lower()
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
    print(f"  mode={mode}")
    print(f"  write_standard={write_standard}")
    print(f"  fs_priority={fs_priority}")

    if mode not in ("upsert_slice", "overwrite_all"):
        raise ValueError("DART_CURATE_MODE must be one of: upsert_slice, overwrite_all")

    listings_map = load_listings_map(listings_path)

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")

    ticker_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])

    ok, skip, err = 0, 0, 0
    std_parts = []  # 이번 slice 결과로 standard csv 만들기 위한 최소 컬럼 모음

    for td in ticker_dirs:
        ticker = td.name
        name = listings_map.get(ticker, {}).get("name", "")

        out_dir = curated_dir / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        out_parquet = out_dir / "dart_accounts_long.parquet"

        try:
            # 1) 이번 slice에 해당하는 raw json들만 읽어서 "slice df" 생성
            jobs = iter_raw_json_files(td, year_from, year_to, reprt_codes)
            if not jobs:
                # raw 캐시가 없으면 아무 것도 못 만듦 -> 기존 parquet은 건드리지 않음 (skip)
                skip += 1
                continue

            slice_rows = []
            for y, rc, fp in jobs:
                payload = read_one_json(fp)
                slice_rows.extend(json_to_long_rows(
                    ticker=ticker,
                    name=name,
                    year=y,
                    reprt_code=rc,
                    payload=payload,
                    source_file=str(fp),
                ))

            # slice 자체가 status!=000(=no data)뿐이면 -> 기존 parquet은 건드리지 않음
            if not slice_rows:
                skip += 1
                continue

            slice_df = pd.DataFrame(slice_rows)
            slice_df = normalize_long_df_types(slice_df)
            slice_df = dedup_long(slice_df)

            # 2) 모드에 따라 기존 parquet merge
            if mode == "overwrite_all":
                merged = slice_df.copy()

            else:
                # upsert_slice: 기존 parquet에서 slice 범위 drop 후 새 slice append
                if out_parquet.exists():
                    old = pd.read_parquet(out_parquet)
                    old = normalize_long_df_types(old)

                    # old에서 slice 범위 제거
                    if "bsns_year" in old.columns and "reprt_code" in old.columns:
                        m = slice_mask(old, year_from, year_to, reprt_codes)
                        old_kept = old[~m].copy()
                    else:
                        # 키 컬럼이 없으면 안전상 old 유지(사실상 drop 불가) -> 그래도 concat 후 dedup
                        old_kept = old.copy()
                else:
                    old_kept = pd.DataFrame()

                merged = pd.concat([old_kept, slice_df], ignore_index=True)
                merged = normalize_long_df_types(merged)
                merged = dedup_long(merged)

            # 3) 저장 (임시 컬럼 제거)
            drop_cols = [c for c in ["_fetched_dt", "_source_mtime"] if c in merged.columns]
            if drop_cols:
                merged = merged.drop(columns=drop_cols)

            merged.to_parquet(out_parquet, index=False)
            ok += 1

            if write_standard:
                # standard는 "이번 slice 범위만" 만들기 위해 merged에서 slice만 다시 필터링
                temp = pd.read_parquet(out_parquet)
                temp = normalize_long_df_types(temp)

                if "bsns_year" in temp.columns and "reprt_code" in temp.columns:
                    mm = slice_mask(temp, year_from, year_to, reprt_codes)
                    temp_slice = temp[mm].copy()
                else:
                    temp_slice = temp.copy()

                keep_cols = ["ticker", "name", "bsns_year", "reprt_code", "fs_div", "sj_div", "account_nm", "thstrm_amount"]
                keep_cols = [c for c in keep_cols if c in temp_slice.columns]
                if keep_cols and not temp_slice.empty:
                    std_parts.append(temp_slice[keep_cols].copy())

        except Exception as e:
            err += 1
            print(f"[curate_dart] ERROR ticker={ticker} {e}")

    print("[curate_dart] result")
    print(f"  OK={ok} SKIP={skip} ERROR={err}")

    if write_standard:
        docs_dir = Path("docs/stocks")
        docs_dir.mkdir(parents=True, exist_ok=True)
        out_csv = docs_dir / f"dart_standard_{year_from}_{year_to}.csv"

        if len(std_parts) == 0:
            print(f"[curate_dart] WARN no rows for standard csv -> not writing: {out_csv}")
        else:
            long_df = pd.concat(std_parts, ignore_index=True)
            std = build_standard_summary(long_df, fs_priority=fs_priority)
            std.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"[curate_dart] wrote standard csv: {out_csv} rows={len(std):,}")


if __name__ == "__main__":
    main()
