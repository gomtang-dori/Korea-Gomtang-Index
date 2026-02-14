# src/caches/cache_f08_foreigner_flow_pykrx.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
from pykrx import stock


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


def _clamp_end(backfill_end: pd.Timestamp) -> pd.Timestamp:
    today_utc = pd.Timestamp.utcnow().tz_localize(None).normalize()
    safe_end = today_utc - pd.Timedelta(days=1)
    if backfill_end > safe_end:
        print(f"[cache_f08] WARN: BACKFILL_END {backfill_end:%Y%m%d} > safe_end {safe_end:%Y%m%d}. clamp.")
        return safe_end
    return backfill_end


def _fetch_foreigner_net_buy(start: str, end: str, market: str) -> pd.DataFrame:
    df = stock.get_market_trading_value_by_date(start, end, market)
    if df is None or df.empty:
        raise RuntimeError(f"[cache_f08] empty df market={market} range={start}~{end}")

    # probe에서 확인: cols에 '외국인합계' 존재
    if "외국인합계" not in df.columns:
        raise RuntimeError(f"[cache_f08] missing '외국인합계' in market={market} cols={list(df.columns)}")

    out = df[["외국인합계"]].copy()
    out.index.name = "date"
    out = out.reset_index()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["foreign_net_buy"] = pd.to_numeric(out["외국인합계"], errors="coerce")
    out = out.dropna(subset=["date", "foreign_net_buy"])[["date", "foreign_net_buy"]]
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return out


def main():
    out_path = Path(os.environ.get("F08_CACHE_PATH", "data/cache/f08_foreigner_flow.parquet"))

    backfill_start = _parse_yyyymmdd(_env("BACKFILL_START"))
    backfill_end = _clamp_end(_parse_yyyymmdd(_env("BACKFILL_END")))
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))

    start = (backfill_start - pd.Timedelta(days=buffer_days)).normalize()
    end = backfill_end.normalize()

    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")
    print(f"[cache_f08] fetch range={start_s}~{end_s} markets=KOSPI+KOSDAQ -> {out_path}")

    kospi = _fetch_foreigner_net_buy(start_s, end_s, "KOSPI")
    kosdaq = _fetch_foreigner_net_buy(start_s, end_s, "KOSDAQ")

    merged = pd.merge(kospi, kosdaq, on="date", how="outer", suffixes=("_kospi", "_kosdaq")).sort_values("date")
    merged["foreign_net_buy_kospi"] = pd.to_numeric(merged.get("foreign_net_buy_kospi"), errors="coerce")
    merged["foreign_net_buy_kosdaq"] = pd.to_numeric(merged.get("foreign_net_buy_kosdaq"), errors="coerce")

    merged["f08_foreigner_net_buy"] = (
        merged["foreign_net_buy_kospi"].fillna(0) + merged["foreign_net_buy_kosdaq"].fillna(0)
    )

    out = merged[["date", "f08_foreigner_net_buy"]].dropna(subset=["date"]).sort_values("date")
    out = out.drop_duplicates("date", keep="last").reset_index(drop=True)

    # upsert
    if out_path.exists():
        old = pd.read_parquet(out_path)
        old["date"] = pd.to_datetime(old.get("date"), errors="coerce")
        old["f08_foreigner_net_buy"] = pd.to_numeric(old.get("f08_foreigner_net_buy"), errors="coerce")
        old = old.dropna(subset=["date", "f08_foreigner_net_buy"])
        final = pd.concat([old, out], ignore_index=True)
        final = final.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    else:
        final = out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(out_path, index=False)
    print(f"[cache_f08] OK rows={len(final)} -> {out_path}")
    print(f"[cache_f08] sample tail:\n{final.tail(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
