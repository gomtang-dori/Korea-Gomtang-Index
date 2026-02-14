from __future__ import annotations

import os
import re
from pathlib import Path
import pandas as pd

from usdkrw_fetch import fetch_ecos_statisticsearch


def _env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        if default is None:
            raise RuntimeError(f"Missing env: {name}")
        return default
    return v


def main():
    ecos_key = (os.environ.get("ECOS_KEY") or "").strip()
    if not ecos_key:
        raise RuntimeError("Missing ECOS_KEY")

    stat_code = (os.environ.get("F07_STAT_CODE") or "817Y002").strip()
    cycle = (os.environ.get("F07_CYCLE") or "D").strip()

    # probe는 최근 1~2년만 보면 ITEM 목록/이름이 충분히 잡힙니다.
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = end - pd.Timedelta(days=365)

    start_yyyymmdd = start.strftime("%Y%m%d")
    end_yyyymmdd = end.strftime("%Y%m%d")

    print(f"[probe_f07] fetch raw stat={stat_code} cycle={cycle} range={start_yyyymmdd}~{end_yyyymmdd}")

    raw_df = fetch_ecos_statisticsearch(
        ecos_key=ecos_key,
        stat_code=stat_code,
        cycle=cycle,
        start_yyyymmdd=start_yyyymmdd,
        end_yyyymmdd=end_yyyymmdd,
        item_code1="",          # ✅ item_code 모르는 상태
        item_code2="",
        item_code3="",
        raw=True,               # ✅ 핵심
    )

    if raw_df is None or raw_df.empty:
        raise RuntimeError("[probe_f07] empty raw_df")

    # 컬럼 존재 확인
    required_cols = ["ITEM_NAME1", "ITEM_CODE1"]
    for c in required_cols:
        if c not in raw_df.columns:
            raise RuntimeError(f"[probe_f07] missing col {c}. cols={list(raw_df.columns)}")

    # ITEM_NAME1 후보 중 "국고채" & "3년" 매칭
    name = raw_df["ITEM_NAME1"].astype(str)
    patterns = [
        re.compile(r"국고채.*3\s*년"),
        re.compile(r"Treasury\s*Bond.*3", re.IGNORECASE),
        re.compile(r"TB.*3", re.IGNORECASE),
    ]

    hit = pd.DataFrame()
    for p in patterns:
        hit = raw_df[name.str.contains(p, na=False)].copy()
        if not hit.empty:
            break

    if hit.empty:
        # 디버깅을 위해 ITEM_NAME1 상위 빈도 출력
        top = raw_df["ITEM_NAME1"].astype(str).value_counts().head(30)
        raise RuntimeError(
            "[probe_f07] cannot find '국고채 3년' in ITEM_NAME1.\n"
            f"Top ITEM_NAME1:\n{top.to_string()}"
        )

    # 대표 코드/명 선정
    item_code1 = hit["ITEM_CODE1"].astype(str).value_counts().index[0]
    item_name1 = hit["ITEM_NAME1"].astype(str).value_counts().index[0]

    print(f"[probe_f07] PICK ITEM_CODE1={item_code1} ITEM_NAME1={item_name1}")

    out_dir = Path("data/cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 전체 raw 저장(문제 생길 때 역추적)
    raw_path = out_dir / "probe_f07_817Y002_raw.parquet"
    raw_df.to_parquet(raw_path, index=False)

    picked = pd.DataFrame([{
        "stat_code": stat_code,
        "cycle": cycle,
        "item_code1": item_code1,
        "item_name1": item_name1,
        "start_yyyymmdd": start_yyyymmdd,
        "end_yyyymmdd": end_yyyymmdd,
    }])
    picked_path = out_dir / "probe_f07_bond3y_itemcode.parquet"
    picked_csv = out_dir / "probe_f07_bond3y_itemcode.csv"
    picked.to_parquet(picked_path, index=False)
    picked.to_csv(picked_csv, index=False, encoding="utf-8-sig")

    print(f"[probe_f07] saved raw -> {raw_path}")
    print(f"[probe_f07] saved pick -> {picked_path} / {picked_csv}")
