from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


# 기존 비율 유지 (f02는 '존재하면' 포함, 없으면 자동 제외/재정규화)
W = {
    "f01_score": 0.10,
    "f02_score": 0.075,  # Strength (temporarily optional)
    "f03_score": 0.125,
    "f04_score": 0.10,
    "f05_score": 0.05,
    "f06_score": 0.125,
    "f07_score": 0.10,
    "f08_score": 0.10,
    "f09_score": 0.125,
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

    if len(df.columns) <= 1:
        return None
    return df


def renormalize_weights_dynamic_f08(
    base: pd.DataFrame,
    weights: dict[str, float],
    score_cols: list[str],
    *,
    f08_col: str = "f08_score",
    greed_threshold: float = 90.0,
    f08_multiplier_when_greedy: float = 0.5,
) -> pd.DataFrame:
    """
    행 단위로, 존재하는 score_cols만으로 가중치를 재정규화 + F08 동적가중.
    - f08_score >= 90인 날은 f08 가중치를 0.5배로 감쇄(0.1 -> 0.05)
    - f08_score < 90 이거나 f08 미존재/NaN이면 기본 가중치
    """
    w = pd.Series({k: float(weights[k]) for k in score_cols}, dtype=float)

    # NaN이면 해당 날짜 weight=0 (이후 재정규화)
    avail = base[score_cols].notna().astype(float)
    w_mat = avail.mul(w, axis=1)

    if f08_col in w_mat.columns:
        f08v = pd.to_numeric(base[f08_col], errors="coerce")
        mask = f08v >= greed_threshold
        w_mat.loc[mask, f08_col] = w_mat.loc[mask, f08_col] * f08_multiplier_when_greedy

    w_sum = w_mat.sum(axis=1).replace(0, np.nan)
    w_norm = w_mat.div(w_sum, axis=0)
    return w_norm


def main():
    factors_dir = Path(os.environ.get("FACTORS_DIR", "data/factors"))
    out_path = Path(os.environ.get("INDEX_PATH", "data/index_daily.parquet"))

    specs = [
        ("f01", factors_dir / "f01.parquet", ["f01_raw", "f01_score"]),
        ("f02", factors_dir / "f02.parquet", ["f02_raw", "f02_score"]),  # optional
        ("f03", factors_dir / "f03.parquet", ["f03_raw", "f03_score"]),
        ("f04", factors_dir / "f04.parquet", ["f04_raw", "f04_score"]),
        ("f05", factors_dir / "f05.parquet", ["f05_raw", "f05_score"]),
        ("f06", factors_dir / "f06.parquet", ["f06_raw", "f06_score"]),
        ("f07", factors_dir / "f07.parquet", ["f07_raw", "f07_score"]),
        ("f08", factors_dir / "f08.parquet", ["f08_raw", "f08_score"]),
        ("f09", factors_dir / "f09.parquet", ["f09_raw", "f09_score"]),
        ("f10", factors_dir / "f10.parquet", ["f10_raw", "f10_score"]),
    ]

    loaded: list[pd.DataFrame] = []
    loaded_score_cols: list[str] = []

    for tag, p, cols in specs:
        df = read_factor_optional(p, cols)
        if df is None:
            print(f"[assemble] skip {tag}: missing/empty {p}")
            continue
        loaded.append(df)

        sc = [c for c in cols if c.endswith("_score") and c in df.columns]
        loaded_score_cols.extend(sc)

    loaded_score_cols = sorted(set(loaded_score_cols))

    if not loaded:
        raise RuntimeError("No factor parquet found under data/factors")
    if not loaded_score_cols:
        raise RuntimeError("No *_score columns loaded")

    base = loaded[0].copy()
    for d in loaded[1:]:
        base = base.merge(d, on="date", how="outer")

    base = base.sort_values("date").reset_index(drop=True)

    # ✅ F08 동적가중: f08_score >= 90일 때만 0.5 감쇄(0.1->0.05)
    w_norm = renormalize_weights_dynamic_f08(
        base,
        W,
        loaded_score_cols,
        f08_col="f08_score",
        greed_threshold=90.0,
        f08_multiplier_when_greedy=0.5,
    )

    base["index_score_total"] = (base[loaded_score_cols] * w_norm).sum(axis=1)
    base["bucket_5pt"] = (np.floor(base["index_score_total"] / 5.0) * 5.0).clip(0, 100)

    # (선택) 디버그용: 실제 적용된 f08 가중치
    if "f08_score" in w_norm.columns:
        base["w_f08_applied"] = w_norm["f08_score"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(out_path, index=False)
    print(f"[assemble] OK rows={len(base)} score_cols={loaded_score_cols} -> {out_path}")


if __name__ == "__main__":
    main()
