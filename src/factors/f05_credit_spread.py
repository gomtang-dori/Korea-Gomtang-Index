from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


def _rolling_percentile(series: pd.Series, window: int, min_obs: int) -> pd.Series:
    def pct_rank_last(x: np.ndarray) -> float:
        x = pd.Series(x).dropna().to_numpy()
        if len(x) == 0:
            return np.nan
        last = x[-1]
        return float((x <= last).mean() * 100.0)

    return series.rolling(window=window, min_periods=min_obs).apply(pct_rank_last, raw=True)


def main():
    rolling_days = int(os.environ.get("ROLLING_DAYS", "1260"))
    min_obs = int(os.environ.get("MIN_OBS", "252"))

    rates_path = Path(os.environ.get("RATES_3Y_BUNDLE_PATH", "data/cache/rates_3y_bundle.parquet"))
    out_path = Path(os.environ.get("F05_OUT_PATH", "data/factors/f05.parquet"))

    if not rates_path.exists():
        raise RuntimeError(f"[f05] Missing {rates_path}")

    r = pd.read_parquet(rates_path)
    r["date"] = pd.to_datetime(r["date"], errors="coerce")
    r["corp_aa_3y"] = pd.to_numeric(r["corp_aa_3y"], errors="coerce")
    r["corp_bbb_3y"] = pd.to_numeric(r["corp_bbb_3y"], errors="coerce")
    r = r.dropna(subset=["date", "corp_aa_3y", "corp_bbb_3y"]).sort_values("date").reset_index(drop=True)

    # 스프레드(위험프리미엄): 커질수록 Fear
    r["f05_raw"] = r["corp_bbb_3y"] - r["corp_aa_3y"]

    pct = _rolling_percentile(r["f05_raw"], window=rolling_days, min_obs=min_obs)

    # 스프레드가 좁을수록 Greed → Fear-type 반전
    r["f05_score"] = 100.0 - pct

    out = r[["date", "f05_raw", "f05_score"]].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"[f05] OK rows={len(out)} -> {out_path}")
    print(out.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
