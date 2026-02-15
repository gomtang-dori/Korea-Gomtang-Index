# src/factors/f08_foreigner_flow.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

from utils.rolling_score import to_score_from_raw


def main():
    cache_path = Path(os.environ.get("F08_CACHE_PATH", "data/cache/f08_foreigner_flow.parquet"))
    out_path = Path(os.environ.get("F08_OUT_PATH", "data/factors/f08.parquet"))

    rolling_days = int(os.environ.get("ROLLING_DAYS", "1260"))
    min_obs = int(os.environ.get("MIN_OBS", "252"))

    if not cache_path.exists():
        raise RuntimeError(f"[f08] Missing {cache_path}. Run cache_f08_foreigner_flow_pykrx first.")

    df = pd.read_parquet(cache_path)
    need = {"date", "f08_foreigner_net_buy"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"[f08] missing cols {need}. got={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["f08_foreigner_net_buy"] = pd.to_numeric(df["f08_foreigner_net_buy"], errors="coerce")
    df = df.dropna(subset=["date", "f08_foreigner_net_buy"]).sort_values("date").reset_index(drop=True)

    df["f08_raw"] = df["f08_foreigner_net_buy"]

    # 순매수↑ 탐욕형 → invert=False
    df["f08_score"] = to_score_from_raw(
        df["f08_raw"],
        window=rolling_days,
        min_obs=min_obs,
        winsor_p=0.01,
        invert=False,
    )

    out = df[["date", "f08_raw", "f08_score"]].dropna(subset=["f08_score"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[f08] OK rows={len(out)} -> {out_path}")
    print(f"[f08] tail:\n{out.tail(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
