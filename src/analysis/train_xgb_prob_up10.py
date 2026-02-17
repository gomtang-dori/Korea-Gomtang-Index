#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XGBoost로 prob_up_10d = P(KOSPI 10영업일 후 수익률 > 0) 예측 확률 컬럼 생성.

핵심:
- 기존 Korea-Gomtang Index(선형 가중치 지수)는 그대로 유지
- 별도 확률 시계열(prob_up_10d)을 생성해 index parquet에 merge할 수 있게 저장
- 간단한 walk-forward(확장 윈도우) 검증 지표(AUC/Brier)도 CSV로 저장

입력:
- 팩터 스코어: data/factors/f01.parquet ... f10.parquet (컬럼: date, f01_score 등)
- 지수 레벨: data/cache/index_levels.parquet (컬럼: date, kospi_close)

출력:
- 확률 시계열 parquet: data/analysis/prob_up_10d_{TAG}.parquet
- fold metrics CSV: data/analysis/ml_prob_up10_{TAG}_metrics.csv

환경변수(Workflow에서 env로 주입 권장):
  PYTHONPATH           : src
  FACTORS_DIR          : data/factors
  INDEX_LEVELS_PATH    : data/cache/index_levels.parquet
  TARGET_COL           : kospi_close
  HORIZON_DAYS         : 10
  ADD_CONTRARIAN       : 1  (f06_c = 100 - f06_score 등 파생피처 생성)
  CONTRARIAN_SUFFIX    : _c
  EXCLUDE_FACTORS      : "f09" 같은 형태, 콤마구분 가능
  TAG                  : "1Y" / "8Y"
  OUT_PRED_PARQUET     : data/analysis/prob_up_10d_1Y.parquet 등
  OUT_METRICS_CSV       : data/analysis/ml_prob_up10_1Y_metrics.csv 등
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception as e:
    raise RuntimeError(
        "xgboost가 필요합니다. requirements.txt에 `xgboost>=2.0.0` 추가 후 다시 실행하세요."
    ) from e


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip()
    return default if v == "" else v


def _parse_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("입력 데이터에 'date' 컬럼이 필요합니다.")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _fwd_log_return(px: pd.Series, horizon: int) -> pd.Series:
    # log(P[t+h]/P[t])
    return np.log(px.shift(-horizon) / px)


def _load_factor_scores(factors_dir: str, exclude_factors: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for i in range(1, 11):
        f = f"f{i:02d}"
        if f in exclude_factors:
            continue

        path = os.path.join(factors_dir, f"{f}.parquet")
        if not os.path.exists(path):
            # optional factor는 없을 수 있으므로 skip
            continue

        df = pd.read_parquet(path)
        df = _ensure_date(df)

        score_col = f"{f}_score"
        if score_col not in df.columns:
            if "score" in df.columns:
                df = df.rename(columns={"score": score_col})
            else:
                raise ValueError(f"{path} 에서 예상 컬럼({score_col})을 찾지 못했습니다. cols={list(df.columns)}")

        frames.append(df[["date", score_col]].copy())

    if not frames:
        raise RuntimeError(f"{factors_dir} 에서 로드된 팩터 parquet이 없습니다.")

    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="date", how="outer")

    out = out.sort_values("date").reset_index(drop=True)
    return out


def _add_contrarian_features(df: pd.DataFrame, suffix: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    f06_score -> f06_c 형태로 파생 feature 생성 (값은 100 - score)
    """
    score_cols = [c for c in df.columns if c.endswith("_score")]
    out = df.copy()
    added: List[str] = []

    for c in score_cols:
        base = c.replace("_score", "")  # f06
        c_name = f"{base}{suffix}"      # f06_c
        out[c_name] = 100.0 - out[c]
        added.append(c_name)

    return out, added


def _auc_fast(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    sklearn 없이 AUC 계산(랭크 기반)
    """
    y_true = y_true.astype(int)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(y_score))
    sum_ranks_pos = ranks[pos].sum()

    auc = (sum_ranks_pos - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    return float(auc)


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_rows: int = 600  # 최소 학습 길이(운영 안정)
    step: int = 20             # 20영업일 단위로 앞으로 전진
    test_size: int = 20        # 각 fold 테스트 20일


def _walk_forward_oos_prob(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    model_params: Dict,
    cfg: WalkForwardConfig
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    확장 윈도우 walk-forward:
      train[0:train_end] -> predict next test_size
      train_end += step
    """
    n = len(X)
    oos = pd.Series(index=X.index, dtype=float)
    rows = []

    train_end = cfg.min_train_rows
    while train_end + cfg.test_size <= n:
        test_start = train_end
        test_end = train_end + cfg.test_size

        X_tr = X.iloc[:train_end]
        y_tr = y.iloc[:train_end]
        X_te = X.iloc[test_start:test_end]
        y_te = y.iloc[test_start:test_end]

        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dte = xgb.DMatrix(X_te, label=y_te)

        booster = xgb.train(
            params=model_params,
            dtrain=dtr,
            num_boost_round=2000,
            evals=[(dtr, "train"), (dte, "test")],
            verbose_eval=False,
            early_stopping_rounds=50,
        )

        prob = booster.predict(dte)
        oos.iloc[test_start:test_end] = prob

        auc = _auc_fast(y_te.to_numpy(), prob)
        brier = float(np.mean((prob - y_te.to_numpy()) ** 2))
        rows.append({
            "fold_train_end": int(train_end),
            "test_start_date": str(dates.iloc[test_start].date()),
            "test_end_date": str(dates.iloc[test_end - 1].date()),
            "auc": auc,
            "brier": brier,
            "best_iteration": int(getattr(booster, "best_iteration", -1)),
        })

        train_end += cfg.step

    return oos, pd.DataFrame(rows)


def main() -> None:
    factors_dir = _env("FACTORS_DIR", "data/factors")
    index_levels_path = _env("INDEX_LEVELS_PATH", "data/cache/index_levels.parquet")
    target_col = _env("TARGET_COL", "kospi_close")
    horizon = int(_env("HORIZON_DAYS", "10"))

    add_contrarian = _env("ADD_CONTRARIAN", "1") == "1"
    contrarian_suffix = _env("CONTRARIAN_SUFFIX", "_c")

    exclude = _parse_list(_env("EXCLUDE_FACTORS", ""))
    tag = _env("TAG", "").strip()

    out_pred_parquet = _env(
        "OUT_PRED_PARQUET",
        f"data/analysis/prob_up_10d_{tag}.parquet" if tag else "data/analysis/prob_up_10d.parquet"
    )
    out_metrics_csv = _env(
        "OUT_METRICS_CSV",
        f"data/analysis/ml_prob_up10_{tag}_metrics.csv" if tag else "data/analysis/ml_prob_up10_metrics.csv"
    )

    os.makedirs(os.path.dirname(out_pred_parquet), exist_ok=True)
    os.makedirs(os.path.dirname(out_metrics_csv), exist_ok=True)

    # 1) Load factor scores
    X_df = _load_factor_scores(factors_dir, exclude)
    X_df = _ensure_date(X_df)

    # 2) Load index levels
    idx = pd.read_parquet(index_levels_path)
    idx = _ensure_date(idx)
    if target_col not in idx.columns:
        raise ValueError(f"{index_levels_path} 에 '{target_col}' 컬럼이 없습니다. cols={list(idx.columns)}")

    y_df = idx[["date", target_col]].copy()

    # 3) Merge
    df = X_df.merge(y_df, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # 4) Features (+ contrarian)
    feature_cols = [c for c in df.columns if c.endswith("_score")]
    if add_contrarian:
        df, added = _add_contrarian_features(df, contrarian_suffix)
        feature_cols = feature_cols + added

    # 5) Label
    df["ret10_log"] = _fwd_log_return(df[target_col].astype(float), horizon)
    df["y_up10"] = (df["ret10_log"] > 0).astype(int)

    # Drop tail without forward return, and rows with missing features
    df = df.dropna(subset=["ret10_log"]).reset_index(drop=True)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    if len(df) < 800:
        print(f"[ml] WARNING: usable rows is small: {len(df)}")

    X = df[feature_cols].astype(float)
    y = df["y_up10"].astype(int)
    dates = df["date"]

    # 6) Conservative XGBoost params
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": 42,
        "nthread": 0,
    }

    # 7) Walk-forward OOS metrics (진짜 운영 성능 감시용)
    oos_prob, metrics = _walk_forward_oos_prob(
        X=X, y=y, dates=dates, model_params=params, cfg=WalkForwardConfig()
    )
    if len(metrics) > 0:
        metrics.to_csv(out_metrics_csv, index=False, encoding="utf-8-sig")
        print(f"[ml] metrics OK -> {out_metrics_csv} folds={len(metrics)}")
        print(metrics.tail(3).to_string(index=False))
    else:
        print("[ml] metrics skipped (not enough rows for walk-forward config)")

    # 8) Train final model on full data and predict prob for all rows
    dtr_full = xgb.DMatrix(X, label=y)
    booster = xgb.train(
        params=params,
        dtrain=dtr_full,
        num_boost_round=500,
        verbose_eval=False,
    )
    prob_all = booster.predict(xgb.DMatrix(X))

    out = pd.DataFrame({
        "date": dates,
        "prob_up_10d": prob_all,
        "y_up_10d": y,
        "kospi_ret10_log": df["ret10_log"].astype(float),
    }).sort_values("date").reset_index(drop=True)

    out.to_parquet(out_pred_parquet, index=False)
    print(f"[ml] prob series OK -> {out_pred_parquet} rows={len(out)}")


if __name__ == "__main__":
    main()
