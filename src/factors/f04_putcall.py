# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

from utils.rolling_score import to_score_from_raw


def main():
    cache_path = Path(os.environ.get("PUTCALL_CACHE_PATH", "data/cache/putcall_ratio.parquet"))
    out_path = Path(os.environ.get("F04_PATH", "data/factors/f04.parquet"))

    window = int(os.environ.get("ROLLING_DAYS", str(252 * 5)))
    min_obs = int(os.environ.get("MIN_OBS", "252"))

    if not cache_path.exists():
        raise RuntimeError(f"Missing {cache_path}. Run cache_putcall first.")

    df = pd.read_parquet(cache_path)
    if "date" not in df.columns:
        raise RuntimeError("putcall cache must include 'date' column")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if "f04_put_trdval" in df.columns and "f04_call_trdval" in df.columns:
        df["putcall_ratio"] = df["f04_put_trdval"] / df["f04_call_trdval"]
        print("[f04] Raw trdval detected. Calculated putcall_ratio.")

    candidates = ["putcall_ratio", "pcr", "put_call", "ratio", "putcall"]
    ratio_col = next((c for c in candidates if c in df.columns), None)

    if ratio_col is None:
        raise RuntimeError(f"putcall cache columns={list(df.columns)}; expected one of {candidates}")

    df[ratio_col] = pd.to_numeric(df[ratio_col], errors="coerce")
    df = df.dropna(subset=[ratio_col])

    df["f04_raw"] = df[ratio_col]

    # PCR 높을수록 공포형 → invert=True
    df["f04_score"] = to_score_from_raw(
        df["f04_raw"],
        window=window,
        min_obs=min_obs,
        winsor_p=0.01,
        invert=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["date", "f04_raw", "f04_score"]].to_parquet(out_path, index=False)
    print(f"[f04] OK rows={len(df)} ratio_col={ratio_col} -> {out_path}")


if __name__ == "__main__":
    main()
