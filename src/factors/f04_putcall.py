# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


def rolling_percentile(s: pd.Series, window: int, min_obs: int) -> pd.Series:
    def _pct(x):
        x = pd.Series(x).dropna()
        if len(x) < min_obs:
            return np.nan
        return float(x.rank(pct=True).iloc[-1] * 100.0)
    return s.rolling(window=window, min_periods=min_obs).apply(_pct, raw=False)


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

    # 캐시 컬럼명은 레포마다 다를 수 있어 자동 탐지
    # 후보: putcall_ratio / pcr / put_call / ratio
    candidates = ["putcall_ratio", "pcr", "put_call", "ratio", "putcall"]
    ratio_col = next((c for c in candidates if c in df.columns), None)
    if ratio_col is None:
        raise RuntimeError(f"putcall cache columns={list(df.columns)}; expected one of {candidates}")

    df[ratio_col] = pd.to_numeric(df[ratio_col], errors="coerce")
    df = df.dropna(subset=[ratio_col])

    # F04 raw: PCR 자체(높을수록 공포)로 두는 경우가 일반적
    df["f04_raw"] = df[ratio_col]

    # score: 퍼센타일(높을수록 공포) → assemble에서 flip을 하든, 그대로 fear로 쓰든 선택
    df["f04_score"] = rolling_percentile(df["f04_raw"], window=window, min_obs=min_obs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["date", "f04_raw", "f04_score"]].to_parquet(out_path, index=False)
    print(f"[f04] OK rows={len(df)} ratio_col={ratio_col} -> {out_path}")


if __name__ == "__main__":
    main()
