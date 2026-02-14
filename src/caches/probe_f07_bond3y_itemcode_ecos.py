from __future__ import annotations

import os
import re
from pathlib import Path
import pandas as pd

# 프로젝트에 이미 있는 ECOS 호출 함수를 그대로 재사용한다고 하셨으니 import만 맞춰주세요.
# 아래 import 경로는 예시입니다. 실제 위치에 맞게 한 줄만 수정하면 됩니다.
from usdkrw_fetch import fetch_ecos_statisticsearch  # TODO: 실제 모듈 경로로 조정


def _env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        if default is None:
            raise RuntimeError(f"Missing env: {name}")
        return default
    return v


def _parse_yyyymmdd(s: str) -> str:
    ts = pd.to_datetime(s, format="%Y%m%d", errors="raise")
    return ts.strftime("%Y%m%d")


def _pick_kTB3y_item_code(df: pd.DataFrame) -> tuple[str, str]:
    """
    ECOS statisticSearch 결과 df에서 '국고채(3년)'에 해당하는 item_code1을 찾아 반환.
    반환: (item_code1, item_name_ko)
    """
    # 가능한 컬럼명들(ECOS 응답 포맷이 프로젝트 구현에 따라 다를 수 있어 방어적으로 처리)
    # fetch_ecos_statisticsearch()가 어떤 칼럼명을 쓰는지 모르므로, name/code 후보를 폭넓게 본다.
    col_candidates = {
        "item_name": ["item_name", "item_name1", "item_nm", "ITEM_NAME1", "itmNm1", "stat_name"],
        "item_code": ["item_code1", "item_cd1", "ITEM_CODE1", "itmCd1", "item_code"],
    }

    def find_col(cands: list[str]) -> str | None:
        for c in cands:
            if c in df.columns:
                return c
        return None

    name_col = find_col(col_candidates["item_name"])
    code_col = find_col(col_candidates["item_code"])

    if name_col is None or code_col is None:
        raise RuntimeError(
            f"[probe_f07] Cannot find item_name/code columns. cols={list(df.columns)}"
        )

    # '국고채' & '3' 키워드 기반 (예: '국고채(3년)', '국고채 3년', 'Treasury Bond (3-year)' 등)
    name_series = df[name_col].astype(str)

    # 우선순위 매칭: (국고채 AND 3년) -> (Treasury Bond AND 3)
    patterns = [
        re.compile(r"국고채.*3\s*년"),
        re.compile(r"Treasury\s*Bond.*3", re.IGNORECASE),
        re.compile(r"TB.*3", re.IGNORECASE),
    ]

    for p in patterns:
        hit = df[name_series.str.contains(p, na=False)].copy()
        if not hit.empty:
            # item_code1이 여러 개면 가장 많이 등장하는(대표) 값 선택
            item_code = hit[code_col].astype(str).value_counts().index[0]
            item_name = hit[name_col].astype(str).value_counts().index[0]
            return item_code, item_name

    raise RuntimeError(
        "[probe_f07] Could not match '국고채 3년' item in returned items. "
        "Please inspect raw output CSV/parquet."
    )


def main():
    ecos_key = _env("ECOS_API_KEY")
    stat_code = os.environ.get("F07_STAT_CODE", "817Y002")  # 시장금리(일별) [Source] 참고
    cycle = os.environ.get("F07_CYCLE", "D")

    # probe는 너무 긴 범위 필요 없으니 최근 60일 정도로 잡아 item 목록/필드 확인만
    end_yyyymmdd = _parse_yyyymmdd(_env("BACKFILL_END"))
    begin_dt = (pd.to_datetime(end_yyyymmdd, format="%Y%m%d") - pd.Timedelta(days=90))
    begin_yyyymmdd = begin_dt.strftime("%Y%m%d")

    print(f"[probe_f07] stat_code={stat_code} cycle={cycle} range={begin_yyyymmdd}~{end_yyyymmdd}")

    # item_code1을 모르는 상태이므로 item_code1 인자 없이 호출 (프로젝트 함수가 허용해야 함)
    # 만약 필수라면: fetch_ecos_statisticsearch를 item_code1=None 허용하도록 1줄 보완 필요.
    df = fetch_ecos_statisticsearch(
        ecos_key=ecos_key,
        stat_code=stat_code,
        cycle=cycle,
        start_yyyymmdd=begin_yyyymmdd,
        end_yyyymmdd=end_yyyymmdd,
        item_code1=None,
    )

    if df is None or df.empty:
        raise RuntimeError("[probe_f07] empty response")

    out_dir = Path("data/cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 원본 저장
    raw_parquet = out_dir / "probe_f07_ecos_817Y002_raw.parquet"
    raw_csv = out_dir / "probe_f07_ecos_817Y002_raw.csv"
    df.to_parquet(raw_parquet, index=False)
    df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    print(f"[probe_f07] saved raw -> {raw_parquet} / {raw_csv}")
    print(f"[probe_f07] cols={list(df.columns)} rows={len(df)}")

    item_code, item_name = _pick_kTB3y_item_code(df)
    print(f"[probe_f07] PICK item_code1={item_code} item_name={item_name}")

    # 선택 결과를 작은 파일로도 저장 (자동화에 유용)
    picked = pd.DataFrame([{"stat_code": stat_code, "cycle": cycle, "item_code1": item_code, "item_name": item_name}])
    picked_path = out_dir / "probe_f07_bond3y_itemcode.parquet"
    picked.to_parquet(picked_path, index=False)
    print(f"[probe_f07] saved pick -> {picked_path}")


if __name__ == "__main__":
    main()
