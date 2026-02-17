from __future__ import annotations

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd


# 기본 고정 비율 (weights csv 없을 때 fallback)
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
    raw = (os.environ.get("EXCLUDE_FACTORS", "") or "").strip()
    if not raw:
        return set()
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return set(items)


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name, default) or "").strip()


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


def apply_ema_per_score_col(base: pd.DataFrame, score_cols: list[str], span: int) -> pd.DataFrame:
    if span <= 0:
        return base

    out = base.copy()
    for c in score_cols:
        s = pd.to_numeric(out[c], errors="coerce")
        out[c] = s.ewm(span=span, adjust=False, min_periods=1).mean()
    return out


def _load_weights_csv(path: Path) -> dict[str, float]:
    """
    weights csv 형식:
      - factor: f06_c / f06 / f06_score 같은 이름도 허용(최종적으로 score_col로 정규화)
      - w_final: 최종 weight
    """
    df = pd.read_csv(path)
    need = {"factor", "w_final"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"[assemble] WEIGHTS_CSV missing columns {need}. got={set(df.columns)}")

    df["factor"] = df["factor"].astype(str)
    df["w_final"] = pd.to_numeric(df["w_final"], errors="coerce").fillna(0.0)

    w: dict[str, float] = {}
    for _, r in df.iterrows():
        f = r["factor"].strip()
        v = float(r["w_final"])
        if v <= 0:
            continue

        # normalize factor naming to score column key
        # allow: f06_score, f06, f06_c
        if f.endswith("_score"):
            key = f
        elif f.startswith("f") and len(f) in (3, 4) and f[1:3].isdigit():
            # f06 -> f06_score
            key = f"{f}_score"
        elif f.startswith("f") and f.endswith("_c") and f[1:3].isdigit():
            # f06_c -> treat as contrarian score column name (special feature col)
            key = f  # keep as feature name (not *_score)
        else:
            key = f  # as-is

        w[key] = v

    s = sum(w.values())
    if s <= 0:
        raise RuntimeError("[assemble] WEIGHTS_CSV loaded but all weights are zero.")
    # total 1로 정규화(안전)
    w = {k: float(v) / s for k, v in w.items()}
    return w


def _maybe_add_contrarian_features(base: pd.DataFrame, score_cols: list[str], suffix: str = "_c") -> tuple[pd.DataFrame, list[str]]:
    """
    assemble 내부에서 contrarian feature 생성:
      f06_score -> f06_c = 100 - f06_score
    생성된 feature 컬럼명 목록을 반환
    """
    out = base.copy()
    added: list[str] = []

    for c in score_cols:
        if not c.endswith("_score"):
            continue
        base_name = c.replace("_score", "")  # f06
        c_name = f"{base_name}{suffix}"      # f06_c
        if c_name in out.columns:
            continue
        out[c_name] = 100.0 - pd.to_numeric(out[c], errors="coerce")
        added.append(c_name)

    return out, added


def renormalize_weights_dynamic_f08(
    base: pd.DataFrame,
    weights: dict[str, float],
    feature_cols: list[str],
    *,
    f08_col: str = "f08_score",
    greed_threshold: float = 90.0,
    f08_multiplier_when_greedy: float = 0.5,
) -> pd.DataFrame:
    """
    feature_cols 기준으로 가중치 행렬 생성 후,
    - 결측이면 weight=0 처리
    - (기존 로직 유지) f08_score가 90 이상이면 f08 weight 0.5배
    - 날짜별 합이 1이 되도록 재정규화
    """
    # weights에 없는 feature는 0으로
    w = pd.Series({k: float(weights.get(k, 0.0)) for k in feature_cols}, dtype=float)

    avail = base[feature_cols].notna().astype(float)
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

    factor_ema_span = int(os.environ.get("FACTOR_EMA_SPAN", "0"))
    index_ema_span = int(os.environ.get("INDEX_EMA_SPAN", "5"))
    exclude = _parse_exclude_factors()

    # new: learned weights
    weights_csv_path = _env("WEIGHTS_CSV_PATH", "")
    weights_used_out = _env("WEIGHTS_USED_OUT", "")  # optional: save actual weights used (json/csv)
    contrarian_suffix = _env("CONTRARIAN_SUFFIX", "_c")

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

    # factor EMA
    if factor_ema_span > 0:
        base = apply_ema_per_score_col(base, loaded_score_cols, factor_ema_span)
        print(f"[assemble] applied factor EMA span={factor_ema_span}")

    # weights: learned csv > fallback W
    if weights_csv_path:
        w_path = Path(weights_csv_path)
        if not w_path.exists():
            raise RuntimeError(f"[assemble] WEIGHTS_CSV_PATH set but missing: {w_path}")
        weights = _load_weights_csv(w_path)
        weights_mode = f"CSV:{weights_csv_path}"
    else:
        weights = dict(W)
        weights_mode = "DEFAULT_W"

    # contrarian features: if weights contain f06_c 같은 키가 있으면 생성
    # (안전하게 모든 score col에서 contrarian 컬럼을 만들어도 비용은 매우 작음)
    base, contr_added = _maybe_add_contrarian_features(base, loaded_score_cols, suffix=contrarian_suffix)

    # feature cols to use in sum = (loaded_score_cols + contrarian cols that are in weights)
    feature_cols = list(loaded_score_cols)
    for k in list(weights.keys()):
        if k.endswith("_c") and k in base.columns and k not in feature_cols:
            feature_cols.append(k)

    # 가중치 재정규화(+f08 동적)
    w_norm = renormalize_weights_dynamic_f08(
        base,
        weights,
        feature_cols,
        f08_col="f08_score",
        greed_threshold=90.0,
        f08_multiplier_when_greedy=0.5,
    )

    base["index_score_total_raw"] = (base[feature_cols] * w_norm).sum(axis=1)

    if index_ema_span > 0:
        s = pd.to_numeric(base["index_score_total_raw"], errors="coerce")
        base["index_score_total"] = s.ewm(span=index_ema_span, adjust=False, min_periods=1).mean()
        print(f"[assemble] applied index EMA span={index_ema_span}")
    else:
        base["index_score_total"] = base["index_score_total_raw"]

    base["bucket_5pt"] = (np.floor(base["index_score_total"] / 5.0) * 5.0).clip(0, 100)

    if "f08_score" in w_norm.columns:
        base["w_f08_applied"] = w_norm["f08_score"]

    # save weights used (static weights after normalization; not per-date f08 dynamic)
    # 여기서는 "학습된 static weights"를 그대로 저장(검증용)
    if weights_used_out:
        outp = Path(weights_used_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights_mode": weights_mode,
            "feature_cols_used": feature_cols,
            "weights": weights,
            "note": "weights are static. per-date dynamic adjustment is applied only to f08 when f08_score>=90 (multiplier=0.5) then renormalized.",
        }
        if outp.suffix.lower() in (".json",):
            outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            # csv
            wdf = pd.DataFrame([{"feature": k, "weight": v} for k, v in weights.items()]).sort_values("feature")
            wdf.to_csv(outp, index=False, encoding="utf-8-sig")
        print(f"[assemble] weights_used saved -> {outp} mode={weights_mode}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(out_path, index=False)
    print(f"[assemble] OK rows={len(base)} score_cols={loaded_score_cols} + contr={len(contr_added)} weights={weights_mode} -> {out_path}")


if __name__ == "__main__":
    main()
