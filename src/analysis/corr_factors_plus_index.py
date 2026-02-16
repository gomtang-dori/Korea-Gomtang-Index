# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np


FACTOR_SCORE_COLS = {
    "f01": "f01_score",
    "f02": "f02_score",
    "f03": "f03_score",
    "f04": "f04_score",
    "f05": "f05_score",
    "f06": "f06_score",
    "f07": "f07_score",
    "f08": "f08_score",
    "f09": "f09_score",
    "f10": "f10_score",
}


def _read_factor_score(factors_dir: Path, tag: str, score_col: str) -> pd.DataFrame | None:
    p = factors_dir / f"{tag}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "date" not in df.columns or score_col not in df.columns:
        return None
    df = df[["date", score_col]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=["date", score_col]).drop_duplicates("date", keep="last").sort_values("date")
    return df.rename(columns={score_col: tag}).reset_index(drop=True)


def _read_index_levels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Missing index_levels parquet: {path} (run cache_index_levels_fdr.py)")
    df = pd.read_parquet(path).copy()
    need = {"date", "kospi_close", "kosdaq_close", "k200_close"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"index_levels missing cols={need}. got={list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["kospi_close", "kosdaq_close", "k200_close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["kospi_close", "kosdaq_close", "k200_close"]:
        out[f"{c}_ret1d"] = np.log(out[c]).diff()
    return out


def main():
    factors_dir = Path(os.environ.get("FACTORS_DIR", "data/factors"))
    index_levels_path = Path(os.environ.get("INDEX_LEVELS_PATH", "data/cache/index_levels.parquet"))
    out_path = Path(os.environ.get("OUT_PATH", "data/analysis/corr.csv"))
    method = (os.environ.get("CORR_METHOD", "pearson") or "pearson").strip().lower()
    use_returns = (os.environ.get("USE_RETURNS", "0").strip() == "1")
    exclude = set(x.strip().lower() for x in os.environ.get("EXCLUDE_FACTORS", "").split(",") if x.strip())
    start = os.environ.get("START_DATE", "").strip()
    end = os.environ.get("END_DATE", "").strip()

    # 1) factors scores (f01..f10)
    frames = []
    for tag, col in FACTOR_SCORE_COLS.items():
        if tag in exclude:
            continue
        f = _read_factor_score(factors_dir, tag, col)
        if f is not None:
            frames.append(f)

    if not frames:
        raise RuntimeError("No factor scores loaded. Check data/factors/*.parquet")

    base = frames[0]
    for f in frames[1:]:
        base = base.merge(f, on="date", how="inner")

    # 2) index levels
    idx = _read_index_levels(index_levels_path)
    base = base.merge(idx, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # 3) optional: add returns
    if use_returns:
        base = _add_returns(base)

    # 4) date filter (optional)
    if start:
        base = base[base["date"] >= pd.to_datetime(start)]
    if end:
        base = base[base["date"] <= pd.to_datetime(end)]
    base = base.reset_index(drop=True)

    # 5) compute corr
    cols = [c for c in base.columns if c != "date"]
    corr = base[cols].corr(method=method)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_path, encoding="utf-8-sig")
    print(f"[corr] OK method={method} use_returns={use_returns} rows={len(base)} cols={len(cols)} -> {out_path}")
    print(corr.round(3).to_string())


if __name__ == "__main__":
    main()
