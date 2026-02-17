#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XGBoost로 prob_up_10d = P(KOSPI 10영업일 후 수익률 > 0) 예측 확률 컬럼 생성.

[Feature]
기본: f01_score~f10_score (+contrarian fxx_c = 100 - fxx_score)
A) KOSPI 20D realized volatility: kospi_rv20
B) KOSPI 125MA momentum: kospi_mom125 = kospi_close / MA125 - 1
C) VKOSPI 20D change: vkospi_chg20 = vkospi.diff(20)
E) Foreigner flow 20D sum z-score: f08_flow20_z
D) USDKRW 20D log return: usdkrw_ret20   (옵션: 방향성만)

입력:
- 팩터 스코어: data/factors/f01.parquet ... f10.parquet (date, f01_score 등)
- 지수 레벨: data/cache/index_levels.parquet (date, kospi_close)
- VKOSPI 레벨: data/cache/vkospi_level.parquet 또는 data/vkospi_level.parquet
- 외국인 flow: data/cache/f08_foreigner_flow.parquet
- USDKRW 레벨: data/usdkrw_level.parquet (기본), 또는 env USDKRW_PATH

출력:
- 확률 시계열 parquet: data/analysis/prob_up_10d_{TAG}.parquet
- fold metrics CSV: data/analysis/ml_prob_up10_{TAG}_metrics.csv

환경변수:
  FACTORS_DIR          : data/factors
  INDEX_LEVELS_PATH    : data/cache/index_levels.parquet
  VKOSPI_LEVELS_PATH   : data/cache/vkospi_level.parquet
  F08_FLOW_PATH        : data/cache/f08_foreigner_flow.parquet
  USDKRW_PATH          : data/usdkrw_level.parquet
  TARGET_COL           : kospi_close
  HORIZON_DAYS         : 10
  ADD_CONTRARIAN       : 1
  CONTRARIAN_SUFFIX    : _c
  EXCLUDE_FACTORS      : "f09" 같은 형태
  TAG                  : "1Y" / "8Y"
  OUT_PRED_PARQUET
  OUT_METRICS_CSV
  ADD_USDKRW_RET20     : "1"이면 D 포함, "0"이면 D 제외

Walk-forward(운영 안정):
  min_train_rows=600, step=60, test_size=60
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path

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
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def _fwd_log_return(px: pd.Series, horizon: int) -> pd.Series:
    return np.log(px.shift(-horizon) / px)


def _load_factor_scores(factors_dir: str, exclude_factors: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for i in range(1, 11):
        f = f"f{i:02d}"
        if f in exclude_factors:
            continue

        path = os.path.join(factors_dir, f"{f}.parquet")
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        df = _ensure_date(df)

        score_col = f"{f}_score"
        if score_col not in df.columns:
            if "score" in df.columns:
                df = df.rename(columns={"score": score_col})
            else:
                raise ValueError(
                    f"{path} 에서 예상 컬럼({score_col})을 찾지 못했습니다. cols={list(df.columns)}"
                )

        frames.append(df[["date", score_col]].copy())

    if not frames:
        raise RuntimeError(f"{factors_dir} 에서 로드된 팩터 parquet이 없습니다.")

    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="date", how="outer")

    out = out.sort_values("date").reset_index(drop=True)
    return out


def _add_contrarian_features(df: pd.DataFrame, suffix: str) -> Tuple[pd.DataFrame, List[str]]:
    score_cols = [c for c in df.columns if c.endswith("_score")]
    out = df.copy()
    added: List[str] = []
    for c in score_cols:
        base = c.replace("_score", "")   # f06
        c_name = f"{base}{suffix}"       # f06_c
        out[c_name] = 100.0 - pd.to_numeric(out[c], errors="coerce")
        added.append(c_name)
    return out, added


def _auc_fast(y_true: np.ndarray, y_score: np.ndarray) -> float:
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
    min_train_rows: int = 600
    step: int = 60
    test_size: int = 60


def _walk_forward_oos_prob(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    model_params: Dict,
    cfg: WalkForwardConfig
) -> Tuple[pd.Series, pd.DataFrame]:
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
            num_boost_round=3000,
            evals=[(dtr, "train"), (dte, "test")],
            verbose_eval=False,
            early_stopping_rounds=80,
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
            "train_rows": int(len(X_tr)),
            "test_rows": int(len(X_te)),
        })

        train_end += cfg.step

    return oos, pd.DataFrame(rows)


def _add_feature_kospi_rv20(df: pd.DataFrame, price_col: str) -> pd.Series:
    ret1 = np.log(df[price_col]).diff()
    return ret1.rolling(20, min_periods=10).std()


def _add_feature_kospi_mom125(df: pd.DataFrame, price_col: str) -> pd.Series:
    ma = df[price_col].rolling(125, min_periods=125).mean()
    return df[price_col] / ma - 1.0


def _read_vkospi_series(vkospi_path_env: str) -> pd.DataFrame:
    p1 = Path(vkospi_path_env)
    p2 = Path("data/vkospi_level.parquet")

    src = p1 if p1.exists() else p2
    if not src.exists():
        raise FileNotFoundError(
            f"VKOSPI 레벨 파일이 없습니다. VKOSPI_LEVELS_PATH={p1} / fallback={p2}"
        )

    v = pd.read_parquet(src)
    v = _ensure_date(v)

    if "vkospi" not in v.columns:
        cand = [c for c in v.columns if c != "date"]
        if not cand:
            raise RuntimeError(f"VKOSPI 파일에 값 컬럼이 없습니다. cols={list(v.columns)}")
        v = v.rename(columns={cand[0]: "vkospi"})

    v["vkospi"] = pd.to_numeric(v["vkospi"], errors="coerce")
    v = v.dropna(subset=["vkospi"]).reset_index(drop=True)
    return v[["date", "vkospi"]].copy()


def _read_f08_flow_series(f08_path: str) -> pd.DataFrame:
    p = Path(f08_path)
    if not p.exists():
        raise FileNotFoundError(f"F08 flow cache missing: {p}")

    df = pd.read_parquet(p)
    df = _ensure_date(df)

    if "f08_foreigner_net_buy" not in df.columns:
        raise RuntimeError(
            f"F08 flow cache missing col f08_foreigner_net_buy. cols={list(df.columns)}"
        )

    df["f08_foreigner_net_buy"] = pd.to_numeric(df["f08_foreigner_net_buy"], errors="coerce")
    df = df.dropna(subset=["f08_foreigner_net_buy"]).reset_index(drop=True)
    return df[["date", "f08_foreigner_net_buy"]].copy()


def _flow20_z(net_buy: pd.Series, sum_win: int = 20, z_win: int = 252, clip: float = 5.0) -> pd.Series:
    flow20 = net_buy.rolling(sum_win, min_periods=max(5, sum_win // 2)).sum()
    mu = flow20.rolling(z_win, min_periods=max(60, z_win // 4)).mean()
    sd = flow20.rolling(z_win, min_periods=max(60, z_win // 4)).std().replace(0, np.nan)
    z = (flow20 - mu) / sd
    if clip is not None and clip > 0:
        z = z.clip(-clip, clip)
    return z


def _read_usdkrw_levels(usdk_path: str) -> pd.DataFrame:
    """
    data/usdkrw_level.parquet을 읽어서 (date, usdkrw) 형태로 정규화.
    f10_fxvol.py가 같은 파일을 사용하므로 이 경로/컬럼 후보를 최대한 안전하게 처리.
    """
    p = Path(usdk_path)
    if not p.exists():
        raise FileNotFoundError(f"USDKRW level missing: {p}")

    df = pd.read_parquet(p)
    df = _ensure_date(df)

    # 컬럼 후보: usdkrw / Close / close / value
    price_col = None
    for c in ["usdkrw", "Close", "close", "value", "USDKRW"]:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        cand = [c for c in df.columns if c != "date"]
        if not cand:
            raise RuntimeError(f"USDKRW 파일에 값 컬럼이 없습니다. cols={list(df.columns)}")
        price_col = cand[0]

    df["usdkrw"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["usdkrw"]).reset_index(drop=True)
    return df[["date", "usdkrw"]].copy()


def main() -> None:
    factors_dir = _env("FACTORS_DIR", "data/factors")
    index_levels_path = _env("INDEX_LEVELS_PATH", "data/cache/index_levels.parquet")
    vkospi_levels_path = _env("VKOSPI_LEVELS_PATH", "data/cache/vkospi_level.parquet")
    f08_flow_path = _env("F08_FLOW_PATH", "data/cache/f08_foreigner_flow.parquet")

    target_col = _env("TARGET_COL", "kospi_close")
    horizon = int(_env("HORIZON_DAYS", "10"))

    add_contrarian = _env("ADD_CONTRARIAN", "1") == "1"
    contrarian_suffix = _env("CONTRARIAN_SUFFIX", "_c")

    # ✅ D 토글 (최소 변경)
    add_usdkrw_ret20 = _env("ADD_USDKRW_RET20", "0") == "1"
    usdkrw_path = _env("USDKRW_PATH", "data/usdkrw_level.parquet")

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
    idx[target_col] = pd.to_numeric(idx[target_col], errors="coerce")
    idx = idx.dropna(subset=[target_col]).reset_index(drop=True)

    # 3) Merge base frame
    df = X_df.merge(idx[["date", target_col]], on="date", how="inner").sort_values("date").reset_index(drop=True)

    # 4) A,B,C
    df["kospi_rv20"] = _add_feature_kospi_rv20(df, target_col)
    df["kospi_mom125"] = _add_feature_kospi_mom125(df, target_col)

    vkospi = _read_vkospi_series(vkospi_levels_path)
    df = df.merge(vkospi, on="date", how="left")
    df["vkospi_chg20"] = pd.to_numeric(df["vkospi"], errors="coerce").diff(20)

    # 5) E (f08 flow 20D sum z-score)
    f08 = _read_f08_flow_series(f08_flow_path)
    df = df.merge(f08, on="date", how="left")
    df["f08_flow20_z"] = _flow20_z(df["f08_foreigner_net_buy"])

    # ✅ engineered 기본 (ABCE)
    engineered = ["kospi_rv20", "kospi_mom125", "vkospi_chg20", "f08_flow20_z"]

    # ✅ 6) D (USDKRW 20D log return) - flag=1일 때만 포함
    if add_usdkrw_ret20:
        usd = _read_usdkrw_levels(usdkrw_path)
        df = df.merge(usd, on="date", how="left")
        df["usdkrw_ret20"] = np.log(pd.to_numeric(df["usdkrw"], errors="coerce")).diff(20)
        engineered.append("usdkrw_ret20")

    # 7) Features (+ contrarian)
    feature_cols = [c for c in df.columns if c.endswith("_score")]
    if add_contrarian:
        df, added = _add_contrarian_features(df, contrarian_suffix)
        feature_cols = feature_cols + added

    feature_cols = feature_cols + engineered

    # 8) Label
    df["ret10_log"] = _fwd_log_return(df[target_col].astype(float), horizon)
    df["y_up10"] = (df["ret10_log"] > 0).astype(int)
    df = df.dropna(subset=["ret10_log"]).reset_index(drop=True)

    before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    after = len(df)

    print(f"[ml] ADD_USDKRW_RET20={int(add_usdkrw_ret20)}")
    print(f"[ml] rows before dropna(features)={before} after={after} dropped={before-after}")
    print(f"[ml] feature_cols={len(feature_cols)} (scores+contrarian+engineered)")
    print(f"[ml] engineered={engineered}")

    X = df[feature_cols].astype(float)
    y = df["y_up10"].astype(int)
    dates = df["date"]

    # 9) XGBoost params
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

    # 10) Walk-forward OOS metrics
    _, metrics = _walk_forward_oos_prob(
        X=X, y=y, dates=dates, model_params=params, cfg=WalkForwardConfig()
    )

    if len(metrics) > 0:
        metrics["tag"] = tag
        metrics["horizon_days"] = horizon
        metrics["target_col"] = target_col
        metrics["n_rows_used"] = len(df)
        metrics["n_features"] = X.shape[1]
        metrics["add_usdkrw_ret20"] = int(add_usdkrw_ret20)  # ✅ 비교 편의
        metrics.to_csv(out_metrics_csv, index=False, encoding="utf-8-sig")
        print(f"[ml] metrics OK -> {out_metrics_csv} folds={len(metrics)}")
        print(metrics.tail(3).to_string(index=False))
    else:
        print("[ml] metrics skipped (not enough rows for walk-forward config)")

    # 11) Train final model & predict prob
    dtr_full = xgb.DMatrix(X, label=y)
    booster = xgb.train(
        params=params,
        dtrain=dtr_full,
        num_boost_round=700,
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
