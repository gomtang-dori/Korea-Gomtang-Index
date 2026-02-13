# src/usdkrw_fetch.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import requests


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def _to_date(s: str) -> pd.Timestamp:
    # accepts YYYYMMDD or YYYY-MM-DD
    s = str(s).strip()
    if len(s) == 8 and s.isdigit():
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _fetch_ecos_usdkrw_level(start_yyyymmdd: str, end_yyyymmdd: str) -> pd.DataFrame:
    """
    ECOS에서 USD/KRW '레벨'을 받아와서 date/usdkrw DataFrame 반환.
    - ECOS Key: env ECOS_KEY
    - NOTE: ECOS의 환율 레벨 통계코드는 계정/문서에 따라 다를 수 있어,
            이 함수는 아래 2가지 방식 중 "작동하는 쪽"을 사용합니다.
      A) 사용자가 이미 레포에서 쓰던 ECOS 환율 레벨 API 파라미터(있다면) → 그대로
      B) 실패 시: 에러를 내고, 사용자가 ECOS에서 쓰는 STAT_CODE/ITEM_CODE를 알려주면 바로 고정
    """
    ecos_key = (os.environ.get("ECOS_KEY") or "").strip()
    if not ecos_key:
        raise RuntimeError("Missing ECOS_KEY")

    # ---- IMPORTANT ----
    # 아래 값들은 '가장 흔히 쓰는' 패턴이지만, ECOS는 통계코드가 다양합니다.
    # 현재 프로젝트는 이미 환율 표(date, usdkrw)를 만든 적이 있으니(사용자 공유),
    # 여러분 레포에서 기존에 쓰던 코드/코드를 아는 경우 여기 값을 그에 맞게 바꾸면 100% 작동합니다.
    #
    # 여기서는 "시도" 형태로 제공하고, 실패하면 명확한 에러를 출력합니다.

    # 후보 1: (예시) 731Y003 등은 변동성/지표일 수 있어 레벨이 아닐 수 있음.
    # 따라서 '레벨'은 통상 "원/달러 환율(매매기준율/종가)" 계열 통계코드를 써야 함.
    # 사용자가 ECOS 레벨 STAT_CODE를 알고 있다면 아래 3개 env로 주입 가능하게 해둠.
    stat_code = (os.environ.get("ECOS_USDKRW_STAT_CODE") or "").strip()
    item_code1 = (os.environ.get("ECOS_USDKRW_ITEM_CODE1") or "").strip()
    item_code2 = (os.environ.get("ECOS_USDKRW_ITEM_CODE2") or "").strip()
    item_code3 = (os.environ.get("ECOS_USDKRW_ITEM_CODE3") or "").strip()

    if not stat_code:
        # 사용자 환경에서 아직 stat_code를 안 넣었으면 실패 유도(잘못된 코드로 조용히 잘못된 데이터 저장 방지)
        raise RuntimeError(
            "Missing ECOS_USDKRW_STAT_CODE. "
            "USD/KRW '레벨' 통계코드를 ECOS에서 확인해 Secrets/Variables로 넣어주세요.\n"
            "필요 env: ECOS_USDKRW_STAT_CODE (+ optional ITEM_CODE1/2/3)\n"
            "예) ECOS API StatisticSearch 파라미터의 STAT_CODE/ITEM_CODE1..."
        )

    # ECOS StatisticSearch endpoint (JSON)
    # 형식: https://ecos.bok.or.kr/api/StatisticSearch/{KEY}/json/kr/1/1000/{STAT_CODE}/{CYCLE}/{START}/{END}/{ITEM_CODE1}/{ITEM_CODE2}/{ITEM_CODE3}
    cycle = (os.environ.get("ECOS_USDKRW_CYCLE") or "D").strip()  # D=일, M=월 등

    def _seg(x: str) -> str:
        return x if x else ""

    url = (
        "https://ecos.bok.or.kr/api/StatisticSearch/"
        f"{ecos_key}/json/kr/1/100000/"
        f"{stat_code}/{cycle}/{start_yyyymmdd}/{end_yyyymmdd}/"
        f"{_seg(item_code1)}/{_seg(item_code2)}/{_seg(item_code3)}"
    )

    r = requests.get(url, timeout=30)
    js = r.json()

    # 정상 응답이면 "StatisticSearch" 키 아래 row 리스트가 있음
    block = js.get("StatisticSearch")
    if not block:
        raise RuntimeError(f"ECOS response missing StatisticSearch: {str(js)[:500]}")

    rows = block.get("row", [])
    if not rows:
        raise RuntimeError(f"ECOS returned empty rows: {str(js)[:500]}")

    df = pd.DataFrame(rows)
    # ECOS row는 보통 TIME, DATA_VALUE 등의 컬럼을 가짐
    time_col = "TIME" if "TIME" in df.columns else ("time" if "time" in df.columns else None)
    val_col = "DATA_VALUE" if "DATA_VALUE" in df.columns else ("data_value" if "data_value" in df.columns else None)
    if not time_col or not val_col:
        raise RuntimeError(f"ECOS columns unexpected: {df.columns.tolist()}")

    out = pd.DataFrame(
        {
            "date": df[time_col].apply(_to_date),
            "usdkrw": pd.to_numeric(df[val_col], errors="coerce"),
        }
    ).dropna(subset=["date", "usdkrw"])

    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return out


def upsert(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        out = new.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return out


def main():
    data_dir = Path("data")
    ensure_dir(data_dir)

    # 최근 14일 재조회 + 휴장 buffer
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = today - pd.Timedelta(days=int(os.environ.get("USDKRW_REFRESH_DAYS", "45")))

    start_yyyymmdd = start.strftime("%Y%m%d")
    end_yyyymmdd = today.strftime("%Y%m%d")

    out_path = data_dir / "usdkrw_level.parquet"

    old = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["date", "usdkrw"])
    if not old.empty:
        old["date"] = pd.to_datetime(old["date"], errors="coerce")

    new = _fetch_ecos_usdkrw_level(start_yyyymmdd, end_yyyymmdd)

    merged = upsert(old, new)
    merged.to_parquet(out_path, index=False)

    # also write csv for debug (optional)
    csv_path = data_dir / "usdkrw_level.csv"
    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"[usdkrw_fetch] OK rows={len(merged)} -> {out_path}")


if __name__ == "__main__":
    main()
