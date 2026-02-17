#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_xgb_prob_up10.py에서 만든 prob_up_10d 시계열을
assemble 결과(index parquet/csv)에 date 기준으로 merge하여 컬럼 추가.

Env:
  INDEX_PATH     (필수) 예: data/index_daily_1Y.parquet
  CSV_PATH       (선택) 예: data/index_daily_1Y.csv
  PROB_PATH      (필수) 예: data/analysis/prob_up_10d_1Y.parquet
  OUT_INDEX_PATH (선택) 기본: INDEX_PATH overwrite
  OUT_CSV_PATH   (선택) 기본: CSV_PATH overwrite
"""

from __future__ import annotations

import os
import pandas as pd


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip()
    return default if v == "" else v


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("입력 데이터에 'date' 컬럼이 필요합니다.")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    return out


def main() -> None:
    index_path = _env("INDEX_PATH")
    prob_path = _env("PROB_PATH")
    csv_path = _env("CSV_PATH", "")

    out_index_path = _env("OUT_INDEX_PATH", index_path)
    out_csv_path = _env("OUT_CSV_PATH", csv_path)

    if not index_path:
        raise ValueError("INDEX_PATH는 필수입니다.")
    if not prob_path:
        raise ValueError("PROB_PATH는 필수입니다.")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"INDEX_PATH 파일이 없습니다: {index_path}")
    if not os.path.exists(prob_path):
        raise FileNotFoundError(f"PROB_PATH 파일이 없습니다: {prob_path}")

    idx = _ensure_date(pd.read_parquet(index_path))
    prob = _ensure_date(pd.read_parquet(prob_path))

    if "prob_up_10d" not in prob.columns:
        raise ValueError(f"{prob_path} 에 'prob_up_10d' 컬럼이 없습니다. cols={list(prob.columns)}")

    prob = prob[["date", "prob_up_10d"]].copy()

    out = idx.merge(prob, on="date", how="left").sort_values("date").reset_index(drop=True)
    out.to_parquet(out_index_path, index=False)
    print(f"[merge-prob] OK -> {out_index_path} rows={len(out)}")

    if out_csv_path.strip():
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        out.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
        print(f"[merge-prob] CSV OK -> {out_csv_path}")


if __name__ == "__main__":
    main()
