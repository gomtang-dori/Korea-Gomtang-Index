from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


# 기존 비율 유지 (존재하는 팩터만 자동 포함/재정규화)
W = {
    "f01_score": 0.10,
    "f02_score": 0.075,  # optional
    "f03_score": 0.125,
    "f04_score": 0.10,
    "f05_score": 0.05,
    "f06_score": 0.125,
    "f07_score": 0.10,
    "f08_score": 0.10,
    "f09_score": 0.125,
    "f10_score": 0.10,
}


def _parse_exclude_factors() -> set[str]:
    # EXCLUDE_FACTORS="f09,f02" 같은 형태
    raw = (os.environ.get("EXCLUDE_FACTORS", "") or "").strip()
    if not raw:
        return set()
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return set(items)


def read_factor_optional(path: Path, cols: list[str]) -> pd.DataFrame | None:
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
    w = pd.Series({k: float(weights[k]) for k in score_cols}, dtype=float)

    # NaN이면 해당 날짜 weight=0 (이후 재정규화)
    avail = base[score_cols].notna().astype(float)
    w_mat = avail.mul(w, axis=1)

    # F08 동적 가중 (기존 로직 유지)
    if f08_col in w_mat.columns:
        f08v = pd.to_numeric(base[f08_col], errors="coerce")
        mask = f08v >= greed_threshold
        w_mat.loc[mask, f08_col] = w_mat.loc[mask, f08_col] * f08_multiplier_when_greedy

    w_sum = w_mat.sum(axis=1).replace(0, np.nan)
    w_norm = w_mat.div(w_sum, axis=0)
    return w_norm


def apply_ema_per_score_col(base: pd.DataFrame, score_cols: list[str], span: int) -> pd.DataFrame:
    """
    각 score 컬럼별로 EMA 적용.
    - date 정렬된 상태여야 함
    - NaN 구간은 pandas ewm 특성상 자연스럽게 처리됨(연속 NaN은 결과 NaN 유지)
    """
    if span <= 0:
        return base

    out = base.copy()
    for c in score_cols:
        s = pd.to_numeric(out[c], errors="coerce")
        out[c] = s.ewm(span=span, adjust=False, min_periods=1).mean()
    return out


def main():
    factors_dir = Path(os.environ.get("FACTORS_DIR", "data/factors"))
    out_path = Path(os.environ.get("INDEX_PATH", "data/index_daily.parquet"))

    # ✅ 정책 파라미터
    factor_ema_span = int(os.environ.get("FACTOR_EMA_SPAN", "0"))  # 8Y=10, 1Y=20
    index_ema_span = int(os.environ.get("INDEX_EMA_SPAN", "5"))    # 둘 다 5 권장
    exclude = _parse_exclude_factors()

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
        if tag in exclude:
            print(f"[assemble] exclude {tag} by EXCLUDE_FACTORS")
            continue

        df = read_factor_optional(p, cols)
        if df is None:
            print(f"[assemble] skip {tag}: missing/empty {p}")
            continue

        loaded.append(df)
        sc = [c for c in cols if c.endswith("_score") and c in df.columns]
        loaded_score_cols.extend(sc)

    loaded_score_cols = sorted(set(loaded_score_cols))

    if not loaded:
        raise RuntimeError("No factor parquet found under data/factors (after exclusion).")
    if not loaded_score_cols:
        raise RuntimeError("No *_score columns loaded (after exclusion).")

    base = loaded[0].copy()
    for d in loaded[1:]:
        base = base.merge(d, on="date", how="outer")

    base = base.sort_values("date").reset_index(drop=True)

    # ✅ 팩터 score EMA 적용 (정책: 8Y=10, 1Y=20)
    if factor_ema_span > 0:
        base = apply_ema_per_score_col(base, loaded_score_cols, factor_ema_span)
        print(f"[assemble] applied factor EMA span={factor_ema_span}")

    # 가중치 재정규화(+F08 동적 가중)
    w_norm = renormalize_weights_dynamic_f08(
        base,
        W,
        loaded_score_cols,
        f08_col="f08_score",
        greed_threshold=90.0,
        f08_multiplier_when_greedy=0.5,
    )

    # ✅ 원본 합산(팩터 EMA 반영된 score 기준)
    base["index_score_total_raw"] = (base[loaded_score_cols] * w_norm).sum(axis=1)

    # ✅ 최종지수 EMA5 적용
    if index_ema_span > 0:
        s = pd.to_numeric(base["index_score_total_raw"], errors="coerce")
        base["index_score_total"] = s.ewm(span=index_ema_span, adjust=False, min_periods=1).mean()
        print(f"[assemble] applied index EMA span={index_ema_span}")
    else:
        base["index_score_total"] = base["index_score_total_raw"]

    # bucket은 최종(스무딩) 지수 기준으로 생성
    base["bucket_5pt"] = (np.floor(base["index_score_total"] / 5.0) * 5.0).clip(0, 100)

    # (선택) 디버그: 실제 적용된 f08 가중치
    if "f08_score" in w_norm.columns:
        base["w_f08_applied"] = w_norm["f08_score"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(out_path, index=False)
    print(f"[assemble] OK rows={len(base)} score_cols={loaded_score_cols} -> {out_path}")


if __name__ == "__main__":
    main()
