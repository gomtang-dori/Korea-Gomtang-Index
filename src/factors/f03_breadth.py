# src/factors/f03_breadth.py
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def rolling_percentile_last(s: pd.Series, window: int, min_obs: int) -> pd.Series:
    def _pct(x):
        x = pd.Series(x).dropna()
        if len(x) < min_obs:
            return np.nan
        return float(x.rank(pct=True).iloc[-1] * 100.0)
    return s.rolling(window=window, min_periods=min_obs).apply(_pct, raw=False)


def main():
    db_path = Path(os.environ.get("ADVDEC_DB_PATH", "data/cache/adv_dec_daily.sqlite"))
    out_path = Path(os.environ.get("F03_PATH", "data/factors/f03.parquet"))

    mode = os.environ.get("F03_MODE", "").strip().lower()  # "daily" or ""
    if mode == "daily":
        window = int(os.environ.get("ROLLING_DAYS", "60"))
        min_obs = int(os.environ.get("MIN_OBS", "20"))
    else:
        window = int(os.environ.get("ROLLING_DAYS", str(252 * 5)))
        min_obs = int(os.environ.get("MIN_OBS", "252"))

    if not db_path.exists():
        raise RuntimeError(f"Missing {db_path}. Run cache_adv_dec_daily_krx_sqlite first.")

    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(
            """
            SELECT date, adv, dec
            FROM adv_dec_daily
            ORDER BY date
            """,
            conn,
        )

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # ADR (20 trading days): rolling sum adv / rolling sum dec
    df["adv20"] = df["adv"].rolling(20, min_periods=20).sum()
    df["dec20"] = df["dec"].rolling(20, min_periods=20).sum()

    df["f03_raw"] = df["adv20"] / df["dec20"].replace(0, np.nan)
    df.loc[df["dec20"].fillna(0) == 0, "f03_raw"] = df.loc[df["dec20"].fillna(0) == 0, "adv20"] / 1.0  # max(1, sum(dec,20))

    df["f03_score"] = rolling_percentile_last(df["f03_raw"], window=window, min_obs=min_obs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["date", "f03_raw", "f03_score"]].to_parquet(out_path, index=False)

    n_score = int(df["f03_score"].notna().sum())
    print(f"[f03] OK rows={len(df)} scored={n_score} window={window} min_obs={min_obs} -> {out_path}")


if __name__ == "__main__":
    main()
