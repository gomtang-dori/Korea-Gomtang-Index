#!/usr/bin/env python3
"""
Curate DART raw cache -> per-ticker parquet + standard summary CSV
Input:  data/stocks/raw/dart/<ticker>/<year>_<reprt_code>_ALL.json
Output:
  - data/stocks/curated/<ticker>/dart_accounts_long.parquet
  - docs/stocks/dart_standard_<YEAR_FROM>_<YEAR_TO>.csv
"""

import os, re, json
from pathlib import Path
from typing import Optional
import pandas as pd

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")) if os.getenv("PROJECT_ROOT") else Path.cwd()
RAW_DIR = PROJECT_ROOT / "data/stocks/raw/dart"
CURATED_ROOT = PROJECT_ROOT / "data/stocks/curated"
DOCS_DIR = PROJECT_ROOT / "docs/stocks"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

YEAR_FROM = int(os.getenv("DART_YEAR_FROM", "2015"))
YEAR_TO = int(os.getenv("DART_YEAR_TO", "2017"))

REPRT_CODES = [x.strip() for x in os.getenv("DART_REPRT_CODES", "11011,11012,11013,11014").split(",") if x.strip()]
pat = re.compile(r"^(?P<year>\d{4})_(?P<reprt>\d{5})_ALL\.json$")

def to_num(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s in ("", "-", "nan", "NaN"):
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except:
        return None

# 표준 계정명 후보(추후 확장 가능)
REV_CANDS = ["매출액", "영업수익", "수익(매출액)", "I. 매출액"]
OP_CANDS  = ["영업이익", "영업이익(손실)"]
NI_CANDS  = ["당기순이익", "당기순이익(손실)", "당기순손익", "연결당기순이익"]
EQ_CANDS  = ["자본총계", "자본총계(결손금)", "자본총액", "자본"]

def pick_account(df, cands, sj_div):
    s = set(df.loc[df["sj_div"] == sj_div, "account_nm"].dropna().astype(str).tolist())
    for c in cands:
        if c in s:
            return c
    return None

def best_value(row):
    # 손익계산서 계정은 누적(thstrm_add_amount)이 있으면 우선
    v_add = row.get("thstrm_add_amount")
    v = row.get("thstrm_amount")
    return v_add if pd.notna(v_add) and v_add is not None else v

def curate_one_ticker(ticker_dir: Path):
    ticker = ticker_dir.name
    long_rows = []

    for fp in ticker_dir.glob("*.json"):
        m = pat.match(fp.name)
        if not m:
            continue
        year = int(m.group("year"))
        reprt = m.group("reprt")
        if year < YEAR_FROM or year > YEAR_TO:
            continue
        if reprt not in REPRT_CODES:
            continue

        data = json.loads(fp.read_text(encoding="utf-8"))
        meta = data.get("_meta") or {}
        status = str(data.get("status", "")).strip()
        message = str(data.get("message", "")).strip()
        lst = data.get("list") or []
        if status != "000" or not isinstance(lst, list):
            continue

        for r in lst:
            long_rows.append({
                "ticker": ticker,
                "name": meta.get("name", ""),
                "corp_code": meta.get("corp_code", ""),
                "bsns_year": int(r.get("bsns_year") or year),
                "reprt_code": str(r.get("reprt_code") or reprt),
                "fs_div": str(r.get("fs_div", "")).strip(),
                "sj_div": str(r.get("sj_div", "")).strip(),
                "account_nm": r.get("account_nm"),
                "thstrm_amount": to_num(r.get("thstrm_amount")),
                "thstrm_add_amount": to_num(r.get("thstrm_add_amount")),
                "frmtrm_amount": to_num(r.get("frmtrm_amount")),
                "frmtrm_add_amount": to_num(r.get("frmtrm_add_amount")),
                "currency": r.get("currency"),
                "rcept_no": r.get("rcept_no"),
            })

    if not long_rows:
        return None, None

    df_long = pd.DataFrame(long_rows)
    df_long.sort_values(["bsns_year", "reprt_code", "fs_div", "sj_div"], inplace=True)

    # per-ticker parquet 저장
    out_dir = CURATED_ROOT / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    df_long.to_parquet(out_dir / "dart_accounts_long.parquet", index=False)

    # 표준 요약 만들기(연도/보고서별, CFS 우선)
    std_rows = []
    for (y, rc), g0 in df_long.groupby(["bsns_year", "reprt_code"]):
        fs_choice = "CFS" if (g0["fs_div"] == "CFS").any() else ("OFS" if (g0["fs_div"] == "OFS").any() else None)
        g = g0[g0["fs_div"] == fs_choice].copy() if fs_choice else g0.copy()

        rev_nm = pick_account(g, REV_CANDS, "IS")
        op_nm  = pick_account(g, OP_CANDS, "IS")
        ni_nm  = pick_account(g, NI_CANDS, "IS")
        eq_nm  = pick_account(g, EQ_CANDS, "BS")

        def get_val(nm, sj):
            if not nm:
                return None
            sub = g[(g["sj_div"] == sj) & (g["account_nm"] == nm)]
            if sub.empty:
                return None
            return best_value(sub.iloc[0])

        revenue = get_val(rev_nm, "IS")
        op_income = get_val(op_nm, "IS")
        net_income = get_val(ni_nm, "IS")
        equity = get_val(eq_nm, "BS")
        roe = (net_income / equity) if (net_income is not None and equity not in (None, 0)) else None

        std_rows.append({
            "ticker": ticker,
            "name": meta.get("name", ""),
            "bsns_year": int(y),
            "reprt_code": str(rc),
            "fs_div": fs_choice,
            "revenue": revenue,
            "op_income": op_income,
            "net_income": net_income,
            "equity": equity,
            "roe": roe,
        })

    df_std = pd.DataFrame(std_rows).sort_values(["ticker", "bsns_year", "reprt_code"])
    return df_long, df_std

def main():
    std_all = []
    tick_dirs = [p for p in RAW_DIR.iterdir() if p.is_dir()]
    print(f"[curate_dart] tickers with raw dirs: {len(tick_dirs)}")

    ok = 0
    for i, td in enumerate(tick_dirs, 1):
        res = curate_one_ticker(td)
        if res == (None, None):
            continue
        _, df_std = res
        std_all.append(df_std)
        ok += 1
        if i <= 20 or i % 300 == 0:
            print(f"  [{i}/{len(tick_dirs)}] {td.name}: OK")

    if std_all:
        df = pd.concat(std_all, ignore_index=True)
        out = DOCS_DIR / f"dart_standard_{YEAR_FROM}_{YEAR_TO}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[curate_dart] wrote standard csv: {out} rows={len(df):,}")

    print(f"[curate_dart] done ok_tickers={ok}")

if __name__ == "__main__":
    main()
