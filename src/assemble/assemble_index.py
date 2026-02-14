from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


# 기존 비율 유지 (f02는 '존재하면' 포함, 없으면 자동 제외/재정규화)
W = {
    "f01_score": 0.10,
    "f02_score": 0.125,  # Strength (temporarily optional)
    "f03_score": 0.10,
    "f04_score": 0.10,
    "f05_score": 0.05,
    "f06_score": 0.125,
    "f07_score": 0.10,
    "f08_score": 0.10,
    "f09_score": 0.10,
    "f10_score": 0.10,
}


def read_factor_optional(path: Path, cols: list[str]) -> pd.DataFrame | None:
    """
    - 파일이 없으면 None
    - date는 반드시 필요
    - cols 중 유효한 컬럼이 하나도 없으면 None (조인 대상에서 제외)
    """
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    if "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    keep = ["date"] + [c for c in cols if c in df.columns]
    df = (
        df[keep]
        .dropna(subset=["date"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

    # date만 있고 실제 factor 컬럼이 없다면 무의미 → 제외
    if len(df.columns) <= 1:
        return None
    return df


def renormalize_weights(base: pd.DataFrame, weights: dict, score_cols: list[str]) -> pd.Series:
    """
    행 단위로, 존재하는 score_cols만으로 가중치를 재정규화.
    - 없는 항은 0처리하지 않고 제외(=해당 col이 NaN이면 그 날은 가중치에서 빠짐)
    """
    w = pd.Series({k: float(weights[k]) for k in score_cols}, dtype=float)  # only loaded cols
    avail = base[score_cols].notna().astype(float)  # 1 if present else 0
    w_mat = avail.mul(w, axis=1)
    w_sum = w_mat.sum(axis=1).replace(0, np.nan)
    w_norm = w_mat.div(w_sum, axis=0)
    return w_norm


def main():
    factors_dir = Path(os.environ.get("FACTORS_DIR", "data/factors"))
    out_path = Path(os.environ.get("INDEX_PATH", "data/index_daily.parquet"))

    # registry: 남겨두되(optional)
    specs = [
        ("f01", factors_dir / "f01.parquet", ["f01_raw", "f01_score"]),
        ("f02", factors_dir / "f02.parquet", ["f02_raw", "f02_score"]),  # optional
        ("f03", factors_dir / "f03.parquet", ["f03_raw", "f03_score"]),
        ("f04", factors_dir / "f04.parquet", ["f04_raw", "f04_score"]),
        ("f05", factors_dir / "f05.parquet", ["f05_raw", "f05_score"]),
        ("f06", factors_dir / "f06.parquet", ["f06_raw", "f06_score"]),
        ("f07", factors_dir / "f07.parquet", ["f07_raw", "f07_score"]),
        ("f08", factors_dir / "f08.parquet", ["f08_raw", "f08_score"]),
        ("f10", factors_dir / "f10.parquet", ["f10_raw", "f10_score"]),
    ]

    loaded = []
    loaded_score_cols = []

    for tag, p, cols in specs:
        df = read_factor_optional(p, cols)
        if df is None:
            print(f"[assemble] skip {tag}: missing/empty {p}")
            continue
        loaded.append(df)

        # score col 있으면 목록에 추가
        sc = [c for c in cols if c.endswith("_score") and c in df.columns]
        loaded_score_cols.extend(sc)

    loaded_score_cols = sorted(set(loaded_score_cols))

    if not loaded:
        raise RuntimeError("No factor parquet found under data/factors")

    base = loaded[0].copy()
    for d in loaded[1:]:
        base = base.merge(d, on="date", how="outer")

    base = base.sort_values("date").reset_index(drop=True)

    # f02 제외 정책: 파일 없으면 컬럼도 만들지 않음 (report에서도 자동 스킵되게)
    # 단, index_score_total 계산을 위해 실제 로드된 score만 사용
    if not loaded_score_cols:
        raise RuntimeError("No *_score columns loaded")

    w_norm = renormalize_weights(base, W, loaded_score_cols)
    base["index_score_total"] = (base[loaded_score_cols] * w_norm).sum(axis=1)

    base["bucket_5pt"] = (np.floor(base["index_score_total"] / 5.0) * 5.0).clip(0, 100)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(out_path, index=False)
    print(f"[assemble] OK rows={len(base)} score_cols={loaded_score_cols} -> {out_path}")


if __name__ == "__main__":
    main()
