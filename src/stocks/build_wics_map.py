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


def _must_have_cols(df: pd.DataFrame, cols: list[str], where: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"[build_wics_map] Missing columns in {where}: {miss} | got={list(df.columns)}")


def main():
    print("[build_wics_map] START")
    print(f"  project_root={PROJECT_ROOT}")
    print(f"  wics_xlsx={WICS_XLSX}")
    print(f"  listings={LISTINGS}")
    print(f"  out_map={OUT_MAP}")

    if not WICS_XLSX.exists():
        raise FileNotFoundError(f"Missing: {WICS_XLSX} (엑셀을 repo에 넣어주세요)")
    if not LISTINGS.exists():
        raise FileNotFoundError(f"Missing: {LISTINGS} (fetch_listings.py 먼저 실행 필요)")

    # -----------------------
    # 1) WICS 로드
    # -----------------------
    w = pd.read_excel(WICS_XLSX, sheet_name=0)
    _must_have_cols(w, ["종목명", "섹터명-대분류", "섹터명-중분류"], where=str(WICS_XLSX))

    w = w.rename(
        columns={
            "종목명": "name",
            "섹터명-대분류": "wics_major",
            "섹터명-중분류": "wics_mid",
        }
    )

    w["name"] = w["name"].astype(str).str.strip()
    w["wics_major"] = w["wics_major"].astype(str).str.strip()
    w["wics_mid"] = w["wics_mid"].astype(str).str.strip()
    w["name_key"] = w["name"].map(norm_name)

    w = w.dropna(subset=["name_key"])
    w = w[w["name_key"].astype(str).str.strip() != ""].copy()

    # 중복 종목명(키 기준) 체크
    dup = w[w.duplicated("name_key", keep=False)].sort_values("name_key")
    OUT_DUP.parent.mkdir(parents=True, exist_ok=True)
    if not dup.empty:
        dup[["name", "wics_major", "wics_mid"]].to_csv(OUT_DUP, index=False, encoding="utf-8-sig")

    # WICS에서 동일 name_key가 여러 개인 경우, 마지막 값을 사용(단, dup 리포트로 확인 가능)
    w_small = w[["name_key", "wics_major", "wics_mid"]].drop_duplicates(subset=["name_key"], keep="last")

    # -----------------------
    # 2) listings 로드
    # -----------------------
    lst = pd.read_parquet(LISTINGS).copy()
    _must_have_cols(lst, ["ticker", "name", "market"], where=str(LISTINGS))

    lst["ticker"] = lst["ticker"].astype(str).str.zfill(6)
    lst["name"] = lst["name"].astype(str).str.strip()
    lst["name_key"] = lst["name"].map(norm_name)

    # -----------------------
    # 3) 매칭
    # -----------------------
    m = lst.merge(w_small, on="name_key", how="left")

    out = m[["ticker", "name", "market", "wics_major", "wics_mid"]].copy()

    # -----------------------
    # 4) 누락 리포트
    # -----------------------
    missing_mask = out["wics_major"].isna() | (out["wics_major"].astype(str).str.strip() == "")
    missing = out[missing_mask].copy()

    OUT_MISSING.parent.mkdir(parents=True, exist_ok=True)
    if not missing.empty:
        missing[["ticker", "name", "market"]].to_csv(OUT_MISSING, index=False, encoding="utf-8-sig")

    # -----------------------
    # 5) 저장
    # -----------------------
    OUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_MAP, index=False)

    # -----------------------
    # 6) 콘솔 요약(강화)
    # -----------------------
    total = len(out)
    mask = out["wics_major"].notna() & (out["wics_major"].astype(str).str.strip() != "")
    covered = int(mask.sum())  # ✅ 괄호/우선순위 버그 방지
    coverage_pct = (covered / total * 100.0) if total else 0.0

    print("[build_wics_map] DONE")
    print(f"  wics_rows={len(w):,} (unique_name_key={w['name_key'].nunique():,})")
    print(f"  listings_rows={len(lst):,} (unique_ticker={lst['ticker'].nunique():,})")
    print(f"  out_rows={total:,}")
    print(f"  coverage={covered:,}/{total:,} ({coverage_pct:.1f}%)")

    print(f"  out_map={OUT_MAP}")
    print(f"  missing_rows={len(missing):,}  missing_report={(OUT_MISSING if not missing.empty else '(none)')}")
    print(f"  dup_name_key_rows={len(dup):,} dup_report={(OUT_DUP if not dup.empty else '(none)')}")

    if not missing.empty:
        print("  [missing sample top10]")
        print(missing[["ticker", "name", "market"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
