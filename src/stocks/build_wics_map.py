#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()

WICS_XLSX = Path("data/stocks/master/WICS분류.xlsx")
LISTINGS = Path("data/stocks/master/listings.parquet")

OUT_MAP = Path("data/stocks/master/wics_map.parquet")
OUT_MISSING = Path("docs/stocks/wics_missing_names.csv")
OUT_DUP = Path("docs/stocks/wics_duplicate_names.csv")

def norm_name(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    # 공백/특수문자 제거(한글/영문/숫자만 남김)
    s = re.sub(r"[^0-9A-Za-z가-힣]", "", s)
    return s

def main():
    if not WICS_XLSX.exists():
        raise FileNotFoundError(f"Missing: {WICS_XLSX} (엑셀을 repo에 넣어주세요)")
    if not LISTINGS.exists():
        raise FileNotFoundError(f"Missing: {LISTINGS} (fetch_listings.py 먼저 실행 필요)")

    # 1) WICS 로드
    w = pd.read_excel(WICS_XLSX, sheet_name=0)
    w = w.rename(columns={
        "종목명": "name",
        "섹터명-대분류": "wics_major",
        "섹터명-중분류": "wics_mid",
    })
    w["name"] = w["name"].astype(str).str.strip()
    w["wics_major"] = w["wics_major"].astype(str).str.strip()
    w["wics_mid"] = w["wics_mid"].astype(str).str.strip()
    w["name_key"] = w["name"].map(norm_name)

    # 중복 종목명(키 기준) 체크
    dup = w[w.duplicated("name_key", keep=False)].sort_values("name_key")
    OUT_DUP.parent.mkdir(parents=True, exist_ok=True)
    if not dup.empty:
        dup[["name", "wics_major", "wics_mid"]].to_csv(OUT_DUP, index=False, encoding="utf-8-sig")

    # 2) listings 로드
    lst = pd.read_parquet(LISTINGS).copy()
    lst["ticker"] = lst["ticker"].astype(str).str.zfill(6)
    lst["name"] = lst["name"].astype(str).str.strip()
    lst["name_key"] = lst["name"].map(norm_name)

    # 3) 매칭
    m = lst.merge(
        w[["name_key", "wics_major", "wics_mid"]],
        on="name_key",
        how="left",
    )

    out = m[["ticker", "name", "market", "wics_major", "wics_mid"]].copy()

    # 4) 누락 리포트
    missing = out[out["wics_major"].isna() | (out["wics_major"].astype(str).str.strip() == "")]
    OUT_MISSING.parent.mkdir(parents=True, exist_ok=True)
    if not missing.empty:
        missing[["ticker", "name", "market"]].to_csv(OUT_MISSING, index=False, encoding="utf-8-sig")

    # 5) 저장
    OUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_MAP, index=False)

    # 6) 콘솔 요약
    total = len(out)
    covered = int((out["wics_major"].notna()) & (out["wics_major"].astype(str).str.strip() != "")).sum()
    print("[build_wics_map] DONE")
    print(f"  out={OUT_MAP}")
    print(f"  coverage={covered}/{total} ({covered/total*100:.1f}%)")
    print(f"  missing_report={OUT_MISSING if missing.empty is False else '(none)'}")
    print(f"  duplicate_report={OUT_DUP if dup.empty is False else '(none)'}")

if __name__ == "__main__":
    main()
