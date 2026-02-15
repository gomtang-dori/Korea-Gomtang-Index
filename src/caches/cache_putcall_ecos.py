# src/caches/cache_putcall_ecos.py
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


def _fetch_series(
    ecos_key: str,
    stat_code: str,
    cycle: str,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    item_code1: str,
) -> pd.DataFrame:
    df = fetch_ecos_statisticsearch(
        ecos_key=ecos_key,
        stat_code=stat_code,
        cycle=cycle,
        start_yyyymmdd=start_yyyymmdd,
        end_yyyymmdd=end_yyyymmdd,
        item_code1=item_code1,
    )
    if df is None or df.empty:
        raise RuntimeError(f"[cache_putcall_ecos] empty series item_code1={item_code1}")

    # fetch_ecos_statisticsearch의 value 컬럼명이 usdkrw로 고정 → rename
    df = df.rename(columns={"usdkrw": "value"})[["date", "value"]].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return df


def main():
    ecos_key = (os.environ.get("ECOS_KEY") or "").strip()
    if not ecos_key:
        raise RuntimeError("Missing ECOS_KEY (GitHub Secrets)")

    # ECOS: 주가지수옵션거래 (캡처 기준 901Y058)
    stat_code = (os.environ.get("PUTCALL_ECOS_STAT_CODE") or "901Y058").strip()
    cycle = (os.environ.get("PUTCALL_ECOS_CYCLE") or "D").strip()

    # 거래량(계약): CALL=S26BA, PUT=S26CA
    call_vol_code = (os.environ.get("PUTCALL_ECOS_CALL_VOL_CODE1") or "S26BA").strip()
    put_vol_code = (os.environ.get("PUTCALL_ECOS_PUT_VOL_CODE1") or "S26CA").strip()

    # 거래대금(백만원): CALL=S26BC, PUT=S26CC
    call_trdval_code = (os.environ.get("PUTCALL_ECOS_CALL_TRDVAL_CODE1") or "S26BC").strip()
    put_trdval_code = (os.environ.get("PUTCALL_ECOS_PUT_TRDVAL_CODE1") or "S26CC").strip()

    out_path = Path(os.environ.get("PUTCALL_CACHE_PATH", "data/cache/putcall_ratio.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backfill_start = _parse_yyyymmdd(_env("BACKFILL_START"))
    backfill_end = _parse_yyyymmdd(_env("BACKFILL_END"))
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))

    start = (backfill_start - pd.Timedelta(days=buffer_days)).normalize()
    end = backfill_end.normalize()

    start_yyyymmdd = start.strftime("%Y%m%d")
    end_yyyymmdd = end.strftime("%Y%m%d")

    print(f"[cache_putcall_ecos] stat={stat_code} cycle={cycle} range={start_yyyymmdd}~{end_yyyymmdd}")
    print(
        "[cache_putcall_ecos] codes: "
        f"CALL_VOL={call_vol_code} PUT_VOL={put_vol_code} "
        f"CALL_TRDVAL={call_trdval_code} PUT_TRDVAL={put_trdval_code}"
    )

    call_vol = _fetch_series(ecos_key, stat_code, cycle, start_yyyymmdd, end_yyyymmdd, call_vol_code).rename(
        columns={"value": "f04_call_vol"}
    )
    put_vol = _fetch_series(ecos_key, stat_code, cycle, start_yyyymmdd, end_yyyymmdd, put_vol_code).rename(
        columns={"value": "f04_put_vol"}
    )
    call_val = _fetch_series(ecos_key, stat_code, cycle, start_yyyymmdd, end_yyyymmdd, call_trdval_code).rename(
        columns={"value": "f04_call_trdval"}
    )
    put_val = _fetch_series(ecos_key, stat_code, cycle, start_yyyymmdd, end_yyyymmdd, put_trdval_code).rename(
        columns={"value": "f04_put_trdval"}
    )

    merged_new = (
        call_vol.merge(put_vol, on="date", how="outer")
        .merge(call_val, on="date", how="outer")
        .merge(put_val, on="date", how="outer")
        .sort_values("date")
        .reset_index(drop=True)
    )

    # 파생비율(디버그/리포트/팩터 계산용)
    merged_new["f04_call_vol"] = pd.to_numeric(merged_new["f04_call_vol"], errors="coerce")
    merged_new["f04_put_vol"] = pd.to_numeric(merged_new["f04_put_vol"], errors="coerce")
    merged_new["f04_call_trdval"] = pd.to_numeric(merged_new["f04_call_trdval"], errors="coerce")
    merged_new["f04_put_trdval"] = pd.to_numeric(merged_new["f04_put_trdval"], errors="coerce")

    merged_new["pcr_vol"] = merged_new["f04_put_vol"] / merged_new["f04_call_vol"].replace(0, pd.NA)
    merged_new["pcr_val"] = merged_new["f04_put_trdval"] / merged_new["f04_call_trdval"].replace(0, pd.NA)

    # 대표 ratio 컬럼(기존 f04_putcall.py 호환): 거래대금 비율
    merged_new["putcall_ratio"] = merged_new["pcr_val"]

    old = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(
        columns=[
            "date",
            "f04_call_vol", "f04_put_vol",
            "f04_call_trdval", "f04_put_trdval",
            "pcr_vol", "pcr_val", "putcall_ratio",
        ]
    )
    if not old.empty:
        old["date"] = pd.to_datetime(old.get("date"), errors="coerce")
        for c in [
            "f04_call_vol", "f04_put_vol",
            "f04_call_trdval", "f04_put_trdval",
            "pcr_vol", "pcr_val", "putcall_ratio",
        ]:
            if c in old.columns:
                old[c] = pd.to_numeric(old.get(c), errors="coerce")

    merged = upsert(old, merged_new)
    merged.to_parquet(out_path, index=False)
    merged.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    print(f"[cache_putcall_ecos] OK rows={len(merged)} -> {out_path}")
    print(merged.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
