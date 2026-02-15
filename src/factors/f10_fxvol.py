# src/factors/f10_fxvol.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

from utils.rolling_score import to_score_from_raw


def main():
    cache_path = Path(os.environ.get("USDKRW_CACHE_PATH", "data/usdkrw_level.parquet"))
    out_path = Path(os.environ.get("F10_OUT_PATH", "data/factors/f10.parquet"))

    vol_window = int(os.environ.get("VOL_WINDOW_DAYS", "20"))
    rolling_days = int(os.environ.get("ROLLING_DAYS", "1260"))
    min_obs = int(os.environ.get("MIN_OBS", "252"))

    if not cache_path.exists():
        raise RuntimeError(
            f"Missing {cache_path}. Run usdkrw_fetch.py (ECOS) first or set USDKRW_CACHE_PATH correctly."
        )

    df = pd.read_parquet(cache_path)
    if "date" not in df.columns:
        raise RuntimeError(f"[f10] Missing 'date' col in {cache_path}. cols={list(df.columns)}")

    price_col = "usdkrw" if "usdkrw" in df.columns else ("Close" if "Close" in df.columns else None)
    if price_col is None:
        raise RuntimeError(f"[f10] Missing usdkrw price col in {cache_path}. cols={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["date", price_col]).sort_values("date").reset_index(drop=True)

    ret = np.log(df[price_col]).diff()
    fxvol_raw = ret.rolling(vol_window, min_periods=max(5, vol_window // 2)).std()

    # 변동성↑ 공포형 → invert=True
    f10_score = to_score_from_raw(
        fxvol_raw,
        window=rolling_days,
        min_obs=min_obs,
        winsor_p=0.01,
        invert=True,
    )

    out = pd.DataFrame({
        "date": df["date"],
        "f10_raw_fxvol": fxvol_raw,
        "f10_score": f10_score,
    }).dropna(subset=["f10_score"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[f10] OK rows={len(out)} vol_window={vol_window} -> {out_path}")


if __name__ == "__main__":
    main()
