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
    # usdkrw_fetch.py가 저장하는 경로와 맞춰주세요(환경변수로 오버라이드 가능)
    cache_path = Path(os.environ.get("USDKRW_CACHE_PATH", "data/cache/usdkrw_level.parquet"))
    out_path = Path(os.environ.get("F10_PATH", "data/factors/f10.parquet"))

    # Daily 짧은 검증이면 window/min_obs를 낮춰도 됨 (env로 조절)
    vol_window = int(os.environ.get("VOL_WINDOW_DAYS", "20"))  # FX vol rolling window
    pct_window = int(os.environ.get("ROLLING_DAYS", str(252 * 5)))
    min_obs = int(os.environ.get("MIN_OBS", "252"))

    if not cache_path.exists():
        raise RuntimeError(f"Missing {cache_path}. Run usdkrw_fetch first.")

    df = pd.read_parquet(cache_path)
    if "date" not in df.columns:
        raise RuntimeError("usdkrw cache must include 'date' column")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 값 컬럼 자동 탐지 (usdkrw / value / close / rate 등)
    candidates = ["usdkrw", "value", "close", "rate", "fx", "level"]
    val_col = next((c for c in candidates if c in df.columns), None)
    if val_col is None:
        raise RuntimeError(f"usdkrw cache columns={list(df.columns)}; expected one of {candidates}")

    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])

    # FX vol raw: log return의 rolling std (annualize는 굳이 안 해도 percentile이면 충분)
    r = np.log(df[val_col]).diff()
    df["fx_vol"] = r.rolling(vol_window, min_periods=max(5, vol_window // 3)).std()

    df["f10_raw"] = df["fx_vol"]
    df["f10_score"] = rolling_percentile(df["f10_raw"], window=pct_window, min_obs=min_obs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["date", "f10_raw", "f10_score"]].to_parquet(out_path, index=False)
    print(f"[f10] OK rows={len(df)} val_col={val_col} vol_window={vol_window} -> {out_path}")


if __name__ == "__main__":
    main()
