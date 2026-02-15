# src/caches/cache_usdkrw_backfill_ecos.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from usdkrw_fetch import fetch_ecos_statisticsearch, upsert


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


def main():
    ecos_key = (os.environ.get("ECOS_KEY") or "").strip()
    if not ecos_key:
        raise RuntimeError("Missing ECOS_KEY (GitHub Secrets)")

    stat_code = (os.environ.get("ECOS_USDKRW_STAT_CODE") or "").strip()
    item_code1 = (os.environ.get("ECOS_USDKRW_ITEM_CODE1") or "").strip()
    cycle = (os.environ.get("ECOS_USDKRW_CYCLE") or "D").strip()

    if not stat_code or not item_code1:
        raise RuntimeError("Missing ECOS_USDKRW_STAT_CODE or ECOS_USDKRW_ITEM_CODE1")

    backfill_start = _parse_yyyymmdd(_env("BACKFILL_START"))
    backfill_end = _safe_end(_parse_yyyymmdd(_env("BACKFILL_END")))
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))

    start = (backfill_start - pd.Timedelta(days=buffer_days)).normalize()
    end = backfill_end.normalize()

    out_path = Path(os.environ.get("USDKRW_OUT_PATH", "data/usdkrw_level.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cache_usdkrw_backfill] stat={stat_code} cycle={cycle} item1={item_code1}")
    print(f"[cache_usdkrw_backfill] range={start:%Y%m%d}~{end:%Y%m%d} out={out_path}")

    new = fetch_ecos_statisticsearch(
        ecos_key=ecos_key,
        stat_code=stat_code,
        cycle=cycle,
        start_yyyymmdd=start.strftime("%Y%m%d"),
        end_yyyymmdd=end.strftime("%Y%m%d"),
        item_code1=item_code1,
    )

    if new is None or new.empty:
        raise RuntimeError("[cache_usdkrw_backfill] empty result from ECOS")

    # upsert
    old = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["date", "usdkrw"])
    if not old.empty:
        old["date"] = pd.to_datetime(old.get("date"), errors="coerce")
        old["usdkrw"] = pd.to_numeric(old.get("usdkrw"), errors="coerce")
        old = old.dropna(subset=["date", "usdkrw"])

    merged = upsert(old, new)
    merged.to_parquet(out_path, index=False)
    merged.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    print(f"[cache_usdkrw_backfill] OK rows={len(merged)} -> {out_path}")
    print(merged.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
