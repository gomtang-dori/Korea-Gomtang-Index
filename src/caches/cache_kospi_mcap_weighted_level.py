# src/caches/cache_kospi_mcap_weighted_level.py
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


def main():
    out_path = Path(os.environ.get("KOSPI_LEVEL_CACHE_PATH", "data/cache/kospi_mcap_weighted_level.parquet"))

    backfill_start = pd.to_datetime(_env("BACKFILL_START"), format="%Y%m%d", errors="raise").normalize()
    backfill_end = pd.to_datetime(_env("BACKFILL_END"), format="%Y%m%d", errors="raise").normalize()
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))

    start = (backfill_start - pd.Timedelta(days=buffer_days)).normalize()
    end = backfill_end.normalize()

    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")

    print(f"[cache_kospi_level] range={start_s}~{end_s} buffer_days={buffer_days}")

    # 거래일 캘린더 (삼성전자)
    cal = stock.get_market_ohlcv_by_date(start_s, end_s, "005930")
    if cal is None or len(cal) == 0:
        raise RuntimeError(f"[cache_kospi_level] trading calendar empty for {start_s}~{end_s}")

    days = [d.strftime("%Y%m%d") for d in cal.index]
    print(f"[cache_kospi_level] trading_days={len(days)}")

    rows = []
    fail_cap = 0
    fail_cols = 0
    last_err = None
    progress_every = int(os.environ.get("PROGRESS_EVERY_N_DAYS", "20"))

    for i, ymd in enumerate(days, start=1):
        try:
            cap = stock.get_market_cap(ymd, market="KOSPI")
            if cap is None or len(cap) == 0:
                fail_cap += 1
                continue

            # cap DataFrame에 종가/시총이 같이 있는 경우가 많음
            close_col = "종가" if "종가" in cap.columns else None
            mcap_col = "시가총액" if "시가총액" in cap.columns else None
            if close_col is None or mcap_col is None:
                fail_cols += 1
                if last_err is None:
                    last_err = f"cap.columns={list(cap.columns)}"
                continue

            df = cap[[close_col, mcap_col]].copy()
            df.columns = ["close", "mcap"]

            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["mcap"] = pd.to_numeric(df["mcap"], errors="coerce")
            df = df.dropna()
            df = df[(df["mcap"] > 0) & (df["close"] > 0)]

            if len(df) == 0:
                fail_cols += 1
                continue

            level = float((df["close"] * df["mcap"]).sum() / df["mcap"].sum())
            rows.append({"date": pd.to_datetime(ymd), "kospi_level": level})

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            fail_cap += 1
            continue

        if i % progress_every == 0:
            print(f"[cache_kospi_level] progress {i}/{len(days)} rows={len(rows)} fail_cap={fail_cap} fail_cols={fail_cols} last_err={last_err}")

    if len(rows) == 0:
        raise RuntimeError(
            "[cache_kospi_level] NO DATA. "
            f"fail_cap={fail_cap} fail_cols={fail_cols} last_err={last_err} "
            f"range={start_s}~{end_s}"
        )

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # upsert
    if out_path.exists():
        old = pd.read_parquet(out_path)
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
        old["kospi_level"] = pd.to_numeric(old["kospi_level"], errors="coerce")
        old = old.dropna(subset=["date", "kospi_level"])
        merged = pd.concat([old, out], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    else:
        merged = out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(
        f"[cache_kospi_level] OK rows={len(merged)} (new_rows={len(out)}) "
        f"fail_cap={fail_cap} fail_cols={fail_cols} -> {out_path}"
    )


if __name__ == "__main__":
    main()
