# src/caches/cache_vkospi_backfill_krx.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from lib.krx_dvrprod_index import KRXDrvProdIndexAPI


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


def _safe_end(backfill_end: pd.Timestamp) -> pd.Timestamp:
    today_utc = pd.Timestamp.utcnow().tz_localize(None).normalize()
    safe_end = today_utc - pd.Timedelta(days=1)
    return min(backfill_end, safe_end)


def _upsert_timeseries(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        out = new.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["vkospi"] = pd.to_numeric(out["vkospi"], errors="coerce")
    out = out.dropna(subset=["date", "vkospi"])
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return out


def main():
    out_path = Path(os.environ.get("VKOSPI_OUT_PATH", "data/vkospi_level.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backfill_start = _parse_yyyymmdd(_env("BACKFILL_START"))
    backfill_end = _safe_end(_parse_yyyymmdd(_env("BACKFILL_END")))
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))

    start = (backfill_start - pd.Timedelta(days=buffer_days)).normalize()
    end = backfill_end.normalize()

    idx_nm = os.environ.get("VKOSPI_INDEX_NAME", "코스피 200 변동성지수").strip() or "코스피 200 변동성지수"

    print(f"[cache_vkospi_backfill] range={start:%Y%m%d}~{end:%Y%m%d} out={out_path}")
    print(f"[cache_vkospi_backfill] idx_nm={idx_nm}")

    api = KRXDrvProdIndexAPI.from_env()
    new = api.fetch_vkospi_range(start, end, idx_nm=idx_nm)

    if new is None or new.empty:
        raise RuntimeError("[cache_vkospi_backfill] empty result (check KRX env/secrets or idx_nm)")

    old = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["date", "vkospi"])
    merged = _upsert_timeseries(old, new)

    merged.to_parquet(out_path, index=False)
    merged.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    print(f"[cache_vkospi_backfill] OK rows(new)={len(new)} rows(total)={len(merged)} -> {out_path}")
    print(merged.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
