from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

from utils.rolling_score import to_score_from_raw


def main():
    rolling_days = int(os.environ.get("ROLLING_DAYS", "1260"))
    min_obs = int(os.environ.get("MIN_OBS", "252"))
    horizon = int(os.environ.get("F07_HORIZON_DAYS", "20"))

    k200_path = Path(os.environ.get("K200_CACHE_PATH", "data/cache/k200_close.parquet"))
    rates_path = Path(os.environ.get("RATES_3Y_BUNDLE_PATH", "data/cache/rates_3y_bundle.parquet"))
    out_path = Path(os.environ.get("F07_OUT_PATH", "data/factors/f07.parquet"))

    if not k200_path.exists():
        raise RuntimeError(f"[f07] Missing {k200_path}")
    if not rates_path.exists():
        raise RuntimeError(f"[f07] Missing {rates_path}")

    k = pd.read_parquet(k200_path)
    r = pd.read_parquet(rates_path)

    k["date"] = pd.to_datetime(k["date"], errors="coerce") if "date" in k.columns else pd.to_datetime(k.iloc[:, 0], errors="coerce")
    k_close_col = None
    for c in ["close", "Close", "k200_close", "value"]:
        if c in k.columns:
            k_close_col = c
            break
    if k_close_col is None:
        cand = [c for c in k.columns if c != "date"]
        k_close_col = cand[0]
    k["k200_close"] = pd.to_numeric(k[k_close_col], errors="coerce")
    k = k[["date", "k200_close"]].dropna().sort_values("date")

    r["date"] = pd.to_datetime(r["date"], errors="coerce")
    r["ktb3y"] = pd.to_numeric(r["ktb3y"], errors="coerce")
    r = r[["date", "ktb3y"]].dropna().sort_values("date")

    df = pd.merge(k, r, on="date", how="inner").sort_values("date").reset_index(drop=True)

    df["eq_ret_20d"] = np.log(df["k200_close"]).diff(horizon)
    df["bond_chg_20d"] = df["ktb3y"].diff(horizon)

    df["f07_raw"] = df["bond_chg_20d"] - df["eq_ret_20d"]

    # raw↑(risk-off) 공포형 → invert=True
    df["f07_score"] = to_score_from_raw(
        df["f07_raw"],
        window=rolling_days,
        min_obs=min_obs,
        winsor_p=0.01,
        invert=True,
    )

    out = df[["date", "f07_raw", "f07_score"]].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"[f07] OK rows={len(out)} horizon={horizon} -> {out_path}")
    print(out.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
