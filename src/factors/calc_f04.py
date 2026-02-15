# src/factors/calc_f04.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

from utils.rolling_score import to_score_from_raw


def main():
    cache_path = Path(os.environ.get("PUTCALL_CACHE_PATH", "data/cache/putcall_ratio.parquet"))
    out_path = Path(os.environ.get("F04_PATH", "data/factors/f04.parquet"))

    window = int(os.environ.get("ROLLING_DAYS", "1260"))
    min_obs = int(os.environ.get("MIN_OBS", "252"))
    winsor_p = float(os.environ.get("WINSOR_P", "0.01"))

    if not cache_path.exists():
        raise RuntimeError(f"[calc_f04] Missing {cache_path}. Run cache_putcall_ecos first.")

    df = pd.read_parquet(cache_path)
    if "date" not in df.columns:
        raise RuntimeError(f"[calc_f04] cache must include 'date' column. cols={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 캐시에서 비율 컬럼 우선 사용, 없으면 원천으로 계산
    if "pcr_vol" in df.columns:
        pcr_vol = pd.to_numeric(df["pcr_vol"], errors="coerce")
    else:
        need = {"f04_put_vol", "f04_call_vol"}
        if not need.issubset(df.columns):
            raise RuntimeError(f"[calc_f04] missing cols for pcr_vol. need={need} cols={list(df.columns)}")
        pcr_vol = pd.to_numeric(df["f04_put_vol"], errors="coerce") / pd.to_numeric(df["f04_call_vol"], errors="coerce").replace(0, np.nan)

    if "pcr_val" in df.columns:
        pcr_val = pd.to_numeric(df["pcr_val"], errors="coerce")
    else:
        need = {"f04_put_trdval", "f04_call_trdval"}
        if not need.issubset(df.columns):
            raise RuntimeError(f"[calc_f04] missing cols for pcr_val. need={need} cols={list(df.columns)}")
        pcr_val = pd.to_numeric(df["f04_put_trdval"], errors="coerce") / pd.to_numeric(df["f04_call_trdval"], errors="coerce").replace(0, np.nan)

    # 대표 raw = 거래대금 비율
    df["f04_raw"] = pcr_val
    df["f04_raw_vol"] = pcr_vol
    df["f04_raw_val"] = pcr_val

    # 공포형(PCR↑ 공포) → invert=True
    df["f04_score_vol"] = to_score_from_raw(df["f04_raw_vol"], window=window, min_obs=min_obs, winsor_p=winsor_p, invert=True)
    df["f04_score_val"] = to_score_from_raw(df["f04_raw_val"], window=window, min_obs=min_obs, winsor_p=winsor_p, invert=True)

    # 하이브리드(0.5/0.5)
    df["f04_score"] = 0.5 * df["f04_score_vol"] + 0.5 * df["f04_score_val"]

    out = df[[
        "date",
        "f04_raw",
        "f04_score",
        "f04_raw_vol",
        "f04_raw_val",
        "f04_score_vol",
        "f04_score_val",
    ]].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[calc_f04] OK rows={len(out)} -> {out_path}")
    print(out.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
