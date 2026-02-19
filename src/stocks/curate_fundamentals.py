#!/usr/bin/env python3
"""
Curate PyKRX fundamentals
raw:     data/stocks/raw/fundamentals/{ticker}.parquet
curated: data/stocks/curated/{ticker}/fundamentals_daily.parquet
"""

import os
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")) if os.getenv("PROJECT_ROOT") else Path.cwd()
RAW_DIR = PROJECT_ROOT / "data/stocks/raw/fundamentals"
MASTER = PROJECT_ROOT / "data/stocks/master/listings.parquet"
CURATED_ROOT = PROJECT_ROOT / "data/stocks/curated"

OVERWRITE = os.getenv("FUND_CURATE_OVERWRITE", "true").lower() == "true"


def _read_raw(ticker: str) -> pd.DataFrame:
    p = RAW_DIR / f"{ticker}.parquet"
    if not p.exists():
        p = RAW_DIR / f"{ticker}.csv"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p, encoding="utf-8-sig")
    else:
        df = pd.read_parquet(p)

    if df.empty:
        return df

    # normalize
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _curate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # 가능한 컬럼들만 표준화해서 뽑기
    cols = ["date"]
    for c in ["bps", "per", "pbr", "eps", "div", "dps"]:
        if c in df.columns:
            cols.append(c)

    out = df[cols].copy()
    out = out.dropna(subset=["date"]).sort_values("date")

    # 수치형 변환(문자/공백 대비)
    for c in ["bps", "per", "pbr", "eps", "div", "dps"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def main():
    print("[curate_fundamentals] start")
    if not MASTER.exists():
        raise FileNotFoundError(f"missing: {MASTER}")

    tickers = pd.read_parquet(MASTER)["ticker"].astype(str).tolist()
    ok = skip = err = 0

    for i, t in enumerate(tickers, 1):
        try:
            df_raw = _read_raw(t)
            if df_raw.empty:
                skip += 1
                continue

            out_dir = CURATED_ROOT / t
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "fundamentals_daily.parquet"
            if out_path.exists() and not OVERWRITE:
                skip += 1
                continue

            df = _curate(df_raw)
            if df.empty:
                skip += 1
                continue

            df.to_parquet(out_path, index=False)
            ok += 1
            if i <= 20 or i % 200 == 0:
                print(f"  [{i}/{len(tickers)}] {t}: OK rows={len(df):,} cols={len(df.columns)}")

        except Exception as e:
            err += 1
            print(f"  [{i}/{len(tickers)}] {t}: ERROR ({e})")

    print(f"[curate_fundamentals] done | OK={ok} SKIP={skip} ERROR={err}")


if __name__ == "__main__":
    main()
