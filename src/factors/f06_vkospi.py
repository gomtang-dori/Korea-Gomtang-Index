# src/factors/f06_vkospi.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

from utils.rolling_score import to_score_from_raw


def main():
    p1 = Path(os.environ.get("VKOSPI_CACHE_PATH", "data/cache/vkospi_level.parquet"))
    p2 = Path("data/vkospi_level.parquet")

    src = p1 if p1.exists() else p2
    out_path = Path(os.environ.get("F06_PATH", "data/factors/f06.parquet"))

    window = int(os.environ.get("ROLLING_DAYS", str(252 * 5)))
    min_obs = int(os.environ.get("MIN_OBS", "252"))

    df = pd.read_parquet(src)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["vkospi"] = pd.to_numeric(df["vkospi"], errors="coerce")
    df = df.dropna(subset=["date", "vkospi"]).sort_values("date").reset_index(drop=True)

    df["f06_raw"] = df["vkospi"]

    # VKOSPI↑ 공포형 → invert=True
    df["f06_score"] = to_score_from_raw(
        df["f06_raw"],
        window=window,
        min_obs=min_obs,
        winsor_p=0.01,
        invert=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["date", "f06_raw", "f06_score"]].to_parquet(out_path, index=False)
    print(f"[f06] OK rows={len(df)} -> {out_path}")


if __name__ == "__main__":
    main()
