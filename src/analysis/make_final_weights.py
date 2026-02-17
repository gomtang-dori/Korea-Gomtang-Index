# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

from config.weight_groups import (
    base_factor_of,
    group_of_base_factor,
    get_config_1y,
    get_config_8y,
)


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name, default) or "").strip()


def _read_weights_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Missing weights csv: {path}")
    df = pd.read_csv(path)
    need = {"factor", "w_raw_avg"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"weights csv must contain {need}. got={set(df.columns)}")
    df["factor"] = df["factor"].astype(str)
    df["base_factor"] = df["factor"].map(base_factor_of)
    df["w_raw_avg"] = pd.to_numeric(df["w_raw_avg"], errors="coerce").fillna(0.0)
    return df


def _dedupe_by_base_factor_keep_best(df: pd.DataFrame) -> pd.DataFrame:
    """
    base_factor 당 1개만 유지: w_raw_avg가 가장 큰 factor(원본/contrarian 중)만 채택
    """
    df = df.sort_values(["base_factor", "w_raw_avg"], ascending=[True, False])
    out = df.drop_duplicates(subset=["base_factor"], keep="first").reset_index(drop=True)
    return out


def _apply_group_sum(df: pd.DataFrame, group_sums: dict[str, float]) -> pd.DataFrame:
    """
    df에는 factor/base_factor/w_raw_avg가 있어야 함.
    그룹별 합을 group_sums로 맞추도록 w를 재스케일링.
    """
    out = df.copy()
    out["group"] = out["base_factor"].map(group_of_base_factor)

    w = out["w_raw_avg"].clip(lower=0.0).to_numpy()
    # 그룹별 raw 합
    gsum_raw = out.groupby("group")["w_raw_avg"].sum().to_dict()

    w_new = []
    for _, r in out.iterrows():
        g = r["group"]
        target = float(group_sums.get(g, 0.0))
        denom = float(gsum_raw.get(g, 0.0))
        if target <= 0.0 or denom <= 0.0:
            w_new.append(0.0)
        else:
            w_new.append(float(r["w_raw_avg"]) * target / denom)

    out["w_grouped"] = w_new
    return out


def _apply_cap_within_group(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """
    그룹별로 cap을 적용하고, 초과분은 같은 그룹 내 (cap 미만) 팩터에 비례 재분배.
    """
    out = df.copy()
    out["w_final"] = out["w_grouped"].clip(lower=0.0)

    for g, idxs in out.groupby("group").groups.items():
        idxs = list(idxs)

        # 반복적으로 cap 적용/재분배(안전하게 여러 번)
        for _ in range(10):
            w = out.loc[idxs, "w_final"].to_numpy()
            over = w > cap + 1e-12
            if not over.any():
                break

            excess = float((w[over] - cap).sum())
            w[over] = cap

            under = w < cap - 1e-12
            if excess <= 0 or not under.any():
                # 재분배 불가면 남겨둠(이 경우 그룹합이 깨질 수 있으나 현실적으로 거의 없음)
                out.loc[idxs, "w_final"] = w
                break

            # under 쪽에 현재 비중대로 재분배
            base = w[under].sum()
            if base <= 0:
                # 균등 재분배
                w[under] = w[under] + excess / under.sum()
            else:
                w[under] = w[under] + excess * (w[under] / base)

            out.loc[idxs, "w_final"] = w

    return out


def _renormalize_total(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    s = float(out["w_final"].sum())
    if s <= 0:
        raise RuntimeError("All final weights are zero.")
    out["w_final"] = out["w_final"] / s
    return out


def main():
    """
    Inputs:
      IN_WEIGHTS_CSV : data/analysis/predict_h5_10_20_1Y_kospi_weights.csv 같은 파일
      POLICY         : "1Y" or "8Y"
      OUT_CSV        : data/analysis/final_weights_1Y_kospi.csv 등

    Output CSV columns:
      factor, base_factor, group, w_final
    """
    in_csv = Path(_env("IN_WEIGHTS_CSV"))
    if str(in_csv).strip() == "":
        raise RuntimeError("Missing env IN_WEIGHTS_CSV")

    policy = _env("POLICY", "1Y").upper()
    cap = float(_env("CAP", "0.25"))

    out_csv = Path(_env("OUT_CSV", f"data/analysis/final_weights_{policy}.csv"))

    if policy == "1Y":
        cfg = get_config_1y(cap=cap)
    elif policy == "8Y":
        cfg = get_config_8y(cap=cap)
    else:
        raise RuntimeError("POLICY must be '1Y' or '8Y'")

    df = _read_weights_csv(in_csv)

    # 정책상 제외 base_factor 제거(8Y는 f09 등)
    df = df[df["base_factor"].isin(cfg.allowed_base_factors)].reset_index(drop=True)

    # base_factor 중복 제거(원본/contrarian 중 best만)
    df = _dedupe_by_base_factor_keep_best(df)

    # 그룹합 고정 재스케일
    df = _apply_group_sum(df, cfg.group_sums)

    # cap 적용 (그룹 내부 재분배)
    df = _apply_cap_within_group(df, cap=cfg.cap)

    # 전체 합=1 재정규화(미세 오차 제거)
    df["group"] = df["base_factor"].map(group_of_base_factor)
    df = _renormalize_total(df)

    out = df[["factor", "base_factor", "group", "w_final"]].copy()
    out = out.sort_values("w_final", ascending=False).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[final-weights] OK -> {out_csv} rows={len(out)} sum={out['w_final'].sum():.6f}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
