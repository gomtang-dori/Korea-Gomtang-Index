# src/assemble/assemble_index.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


W = {
    "f01_score": 0.10,
    "f02_score": 0.10,
    "f03_score": 0.10,
    "f04_score": 0.10,
    "f05_score": 0.05,
    "f06_score": 0.125,
    "f07_score": 0.10,
    "f08_score": 0.10,
    "f10_score": 0.10,
}


def renormalize_weights(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    score_cols = list(weights.keys())
    w = pd.Series(weights, dtype=float)
    avail = df[score_cols].notna().astype(float)
    w_mat = avail.mul(w, axis=1)
    w_sum = w_mat.sum(axis=1).replace(0, np.nan)
    return w_mat.div(w_sum, axis=0)


def read_factor(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date"])
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    keep = ["date"] + [c for c in cols if c in df.columns]
    return df[keep].dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def main():
    factors_dir = Path(os.environ.get("FACTORS_DIR", "data/factors"))
    out_path = Path(os.environ.get("INDEX_PATH", "data/index_daily.parquet"))

    f01 = read_factor(factors_dir / "f01.parquet", ["f01_raw", "f01_score"])
    f02 = read_factor(factors_dir / "f02.parquet", ["f02_raw", "f02_score"])
    f03 = read_factor(factors_dir / "f03.parquet", ["f03_raw", "f03_score"])
    f04 = read_factor(factors_dir / "f04.parquet", ["f04_raw", "f04_score"])
    f05 = read_factor(factors_dir / "f05.parquet", ["f05_raw", "f05_score"])
    f06 = read_factor(factors_dir / "f06.parquet", ["f06_raw", "f06_score"])
    f07 = read_factor(factors_dir / "f07.parquet", ["f07_raw", "f07_score"])
    f08 = read_factor(factors_dir / "f08.parquet", ["f08_raw", "f08_score"])
    f10 = read_factor(factors_dir / "f10.parquet", ["f10_raw", "f10_score"])

    dfs = [f01, f02, f03, f04, f05, f06, f07, f08, f10]
    base = None
    for d in dfs:
        if base is None:
            base = d.copy()
        else:
            base = base.merge(d, on="date", how="outer")
    if base is None:
        raise RuntimeError("No factors found")

    base = base.sort_values("date").reset_index(drop=True)

    score_cols = list(W.keys())
    for c in score_cols:
        if c not in base.columns:
            base[c] = np.nan

    w_norm = renormalize_weights(base, W)
    base["index_score_total"] = (base[score_cols] * w_norm).sum(axis=1)
    base["bucket_5pt"] = (np.floor(base["index_score_total"] / 5.0) * 5.0).clip(0, 100)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(out_path, index=False)
    print(f"[assemble] OK rows={len(base)} -> {out_path}")


if __name__ == "__main__":
    main()
