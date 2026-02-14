# src/factors/f02_strength.py
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def rolling_percentile_last(s: pd.Series, window: int, min_obs: int) -> pd.Series:
    """
    5년(기본) 롤링 윈도우에서 "마지막 값의 퍼센타일(0~100)"을 반환.
    기존 레포의 percentile 스타일과 동일한 패턴 유지.
    """
    def _pct(x):
        x = pd.Series(x).dropna()
        if len(x) < min_obs:
            return np.nan
        return float(x.rank(pct=True).iloc[-1] * 100.0)

    return s.rolling(window=window, min_periods=min_obs).apply(_pct, raw=False)


def main():
    db_path = Path(os.environ.get("ADVDEC_DB_PATH", "data/cache/adv_dec_daily.sqlite"))
    out_path = Path(os.environ.get("F02_PATH", "data/factors/f02.parquet"))

    mode = os.environ.get("F02_MODE", "").strip().lower()  # "daily" or ""
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
            SELECT date, adv, dec, unch, total, unknown, row_count
            FROM adv_dec_daily
            ORDER BY date
            """,
            conn,
        )

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["adv"] = pd.to_numeric(df["adv"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df = df.dropna(subset=["date", "adv", "dec"]).sort_values("date").reset_index(drop=True)

    # F02_raw = (adv - dec) / (adv + dec), 보합 분모 제외 (확정)
    denom = (df["adv"] + df["dec"]).replace(0, np.nan)
    df["f02_raw"] = (df["adv"] - df["dec"]) / denom

    df["f02_score"] = rolling_percentile_last(df["f02_raw"], window=window, min_obs=min_obs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["date", "f02_raw", "f02_score"]].to_parquet(out_path, index=False)

    n_score = int(df["f02_score"].notna().sum())
    print(f"[f02] OK rows={len(df)} scored={n_score} window={window} min_obs={min_obs} -> {out_path}")


if __name__ == "__main__":
    main()
