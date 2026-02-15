from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

from utils.rolling_score import to_score_from_raw


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

    r["f05_raw"] = r["corp_bbb_3y"] - r["corp_aa_3y"]

    # 스프레드↑ 공포형 → invert=True
    r["f05_score"] = to_score_from_raw(
        r["f05_raw"],
        window=rolling_days,
        min_obs=min_obs,
        winsor_p=0.01,
        invert=True,
    )

    out = r[["date", "f05_raw", "f05_score"]].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"[f05] OK rows={len(out)} -> {out_path}")
    print(out.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
