# src/usdkrw_fetch.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import requests


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def _to_date(s: str) -> pd.Timestamp:
    s = str(s).strip()
    if len(s) == 8 and s.isdigit():
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def fetch_ecos_statisticsearch(
    ecos_key: str,
    stat_code: str,
    cycle: str,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    item_code1: str = "",
    item_code2: str = "",
    item_code3: str = "",
) -> pd.DataFrame:
    """
    ECOS StatisticSearch JSON 호출
    반환 DataFrame: date(TIME), value(DATA_VALUE)
    """
    def seg(x: str) -> str:
        return x if x else ""

    url = (
        "https://ecos.bok.or.kr/api/StatisticSearch/"
        f"{ecos_key}/json/kr/1/100000/"
        f"{stat_code}/{cycle}/{start_yyyymmdd}/{end_yyyymmdd}/"
        f"{seg(item_code1)}/{seg(item_code2)}/{seg(item_code3)}"
    )

    r = requests.get(url, timeout=30)
    js = r.json()

    block = js.get("StatisticSearch")
    if not block:
        raise RuntimeError(f"ECOS response missing StatisticSearch: {str(js)[:500]}")

    rows = block.get("row", [])
    if not rows:
        raise RuntimeError(f"ECOS returned empty rows: {str(js)[:500]}")

    df = pd.DataFrame(rows)
    if "TIME" not in df.columns or "DATA_VALUE" not in df.columns:
        raise RuntimeError(f"ECOS columns unexpected: {df.columns.tolist()}")

    out = pd.DataFrame(
        {
            "date": df["TIME"].apply(_to_date),
            "usdkrw": pd.to_numeric(df["DATA_VALUE"], errors="coerce"),
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
    ecos_key = (os.environ.get("ECOS_KEY") or "").strip()
    if not ecos_key:
        raise RuntimeError("Missing ECOS_KEY (GitHub Secrets)")

    # These come from GitHub Actions Variables
    stat_code = (os.environ.get("ECOS_USDKRW_STAT_CODE") or "").strip()
    cycle = (os.environ.get("ECOS_USDKRW_CYCLE") or "").strip() or "D"
    item_code1 = (os.environ.get("ECOS_USDKRW_ITEM_CODE1") or "").strip()

    if not stat_code or not item_code1:
        raise RuntimeError(
            "Missing ECOS_USDKRW_STAT_CODE or ECOS_USDKRW_ITEM_CODE1. "
            "Set them in GitHub Actions Variables."
        )

    # 최근 재조회 기간(기본 45일: 휴장/누락 버퍼)
    refresh_days = int(os.environ.get("USDKRW_REFRESH_DAYS", "45"))
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = today - pd.Timedelta(days=refresh_days)

    start_yyyymmdd = start.strftime("%Y%m%d")
    end_yyyymmdd = today.strftime("%Y%m%d")

    out_path = Path("data/usdkrw_level.parquet")
    ensure_dir(out_path.parent)

    old = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["date", "usdkrw"])
    if not old.empty:
        old["date"] = pd.to_datetime(old["date"], errors="coerce")

    new = fetch_ecos_statisticsearch(
        ecos_key=ecos_key,
        stat_code=stat_code,
        cycle=cycle,
        start_yyyymmdd=start_yyyymmdd,
        end_yyyymmdd=end_yyyymmdd,
        item_code1=item_code1,
    )

    merged = upsert(old, new)
    merged.to_parquet(out_path, index=False)

    # optional debug csv
    merged.to_csv("data/usdkrw_level.csv", index=False, encoding="utf-8-sig")

    print(f"[usdkrw_fetch] OK stat={stat_code} cycle={cycle} item1={item_code1} rows={len(merged)}")


if __name__ == "__main__":
    main()
