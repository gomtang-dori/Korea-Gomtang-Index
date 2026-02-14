from __future__ import annotations

import os
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


def _parse_yyyymmdd(s: str) -> pd.Timestamp:
    ts = pd.to_datetime(s, format="%Y%m%d", errors="raise")
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def upsert(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        out = new.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return out


def _fetch_series(ecos_key: str, stat_code: str, cycle: str,
                  start_yyyymmdd: str, end_yyyymmdd: str, item_code1: str) -> pd.DataFrame:
    df = fetch_ecos_statisticsearch(
        ecos_key=ecos_key,
        stat_code=stat_code,
        cycle=cycle,
        start_yyyymmdd=start_yyyymmdd,
        end_yyyymmdd=end_yyyymmdd,
        item_code1=item_code1,
    )
    if df is None or df.empty:
        raise RuntimeError(f"[cache_rates_3y] empty series item_code1={item_code1}")

    # fetch_ecos_statisticsearch의 value 컬럼명이 usdkrw로 고정 → 캐시에서만 rename (요청사항)
    df = df.rename(columns={"usdkrw": "value"})[["date", "value"]].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return df


def main():
    ecos_key = (os.environ.get("ECOS_KEY") or "").strip()
    if not ecos_key:
        raise RuntimeError("Missing ECOS_KEY (GitHub Secrets)")

    stat_code = (os.environ.get("RATES_STAT_CODE") or "817Y002").strip()
    cycle = (os.environ.get("RATES_CYCLE") or "D").strip()

    # 기본값을 코드로 박아두되, env로도 override 가능
    code_k_tb3y = (os.environ.get("ECOS_KTB3Y_ITEM_CODE1") or "010200000").strip()
    code_corp_aa = (os.environ.get("ECOS_CORP3Y_AA_ITEM_CODE1") or "010300000").strip()
    code_corp_bbb = (os.environ.get("ECOS_CORP3Y_BBB_ITEM_CODE1") or "010320000").strip()

    out_path = Path(os.environ.get("RATES_3Y_BUNDLE_PATH", "data/cache/rates_3y_bundle.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backfill_start = _parse_yyyymmdd(_env("BACKFILL_START"))
    backfill_end = _parse_yyyymmdd(_env("BACKFILL_END"))
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))

    start = (backfill_start - pd.Timedelta(days=buffer_days)).normalize()
    end = backfill_end.normalize()
    start_yyyymmdd = start.strftime("%Y%m%d")
    end_yyyymmdd = end.strftime("%Y%m%d")

    print(f"[cache_rates_3y] stat={stat_code} cycle={cycle} range={start_yyyymmdd}~{end_yyyymmdd}")
    print(f"[cache_rates_3y] codes: KTB3Y={code_k_tb3y} CORP_AA={code_corp_aa} CORP_BBB={code_corp_bbb}")

    tb3y = _fetch_series(ecos_key, stat_code, cycle, start_yyyymmdd, end_yyyymmdd, code_k_tb3y).rename(columns={"value": "ktb3y"})
    aa3y = _fetch_series(ecos_key, stat_code, cycle, start_yyyymmdd, end_yyyymmdd, code_corp_aa).rename(columns={"value": "corp_aa_3y"})
    bbb3y = _fetch_series(ecos_key, stat_code, cycle, start_yyyymmdd, end_yyyymmdd, code_corp_bbb).rename(columns={"value": "corp_bbb_3y"})

    merged_new = tb3y.merge(aa3y, on="date", how="outer").merge(bbb3y, on="date", how="outer")
    merged_new = merged_new.sort_values("date").reset_index(drop=True)

    old = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["date", "ktb3y", "corp_aa_3y", "corp_bbb_3y"])
    if not old.empty:
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
        for c in ["ktb3y", "corp_aa_3y", "corp_bbb_3y"]:
            if c in old.columns:
                old[c] = pd.to_numeric(old[c], errors="coerce")

    merged = upsert(old, merged_new)
    merged.to_parquet(out_path, index=False)
    merged.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    print(f"[cache_rates_3y] OK rows={len(merged)} -> {out_path}")
    print(merged.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
