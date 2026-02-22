#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_rules_and_model_2022_2024.py
- Train:
  - global model (LogisticRegression) for y_up20_20d
  - global rules (top 10)
  - sector rules by wics_major (top 5 per sector) with shrink to global thresholds
- Save to: data/stocks/ml_models/up20_20d/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib


FEATURE_CANDIDATES_PREFIX = (
    "ret_", "logret_", "ma_gap_", "vol_",
    "turnover_", "amihud_", "log_value", "log_volume",
    "foreign_net_to_value_", "inst_net_to_value_", "pension_net_to_value_", "fininv_net_to_value_",
    "inv_per", "inv_pbr", "roe_proxy", "peg_approx_"
)


def _is_feature_col(c: str) -> bool:
    if c in ["log_value", "log_volume", "inv_per", "inv_pbr", "roe_proxy"]:
        return True
    return c.startswith(FEATURE_CANDIDATES_PREFIX)


def _lift(precision: float, base_rate: float) -> float:
    if base_rate <= 0 or np.isnan(base_rate):
        return np.nan
    return precision / base_rate


def _scan_thresholds(
    x: pd.Series,
    y: pd.Series,
    base_rate: float,
    quantiles=(0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    min_support=0.01,
    min_pos=50,
) -> Dict[str, Any]:
    """
    For a single feature:
    - scan thresholds for both directions: x >= t and x <= t
    - choose best rule by score = log(lift) * sqrt(support)
    """
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return {"ok": False, "reason": "no_data"}

    n = len(df)
    base = df["y"].mean() if base_rate is None else base_rate

    best = None

    for q in quantiles:
        t = df["x"].quantile(q)

        # direction: >=
        m = df["x"] >= t
        sup = m.mean()
        if sup >= min_support:
            yy = df.loc[m, "y"]
            pos = yy.sum()
            if pos >= min_pos:
                prec = yy.mean()
                lf = _lift(prec, base)
                if np.isfinite(lf) and lf > 1.0:
                    score = np.log(lf) * np.sqrt(sup)
                    cand = dict(direction="ge", threshold=float(t), support=float(sup), precision=float(prec), lift=float(lf), score=float(score))
                    if (best is None) or (cand["score"] > best["score"]):
                        best = cand

        # direction: <=
        m = df["x"] <= t
        sup = m.mean()
        if sup >= min_support:
            yy = df.loc[m, "y"]
            pos = yy.sum()
            if pos >= min_pos:
                prec = yy.mean()
                lf = _lift(prec, base)
                if np.isfinite(lf) and lf > 1.0:
                    score = np.log(lf) * np.sqrt(sup)
                    cand = dict(direction="le", threshold=float(t), support=float(sup), precision=float(prec), lift=float(lf), score=float(score))
                    if (best is None) or (cand["score"] > best["score"]):
                        best = cand

    if best is None:
        return {"ok": False, "reason": "no_good_rule"}

    best["ok"] = True
    best["base_rate"] = float(base)
    best["n"] = int(n)
    return best


def _shrink_threshold(t_sector: float, t_global: float, n_pos_sector: int, K: int) -> float:
    alpha = n_pos_sector / (n_pos_sector + K) if (n_pos_sector + K) > 0 else 0.0
    return float(alpha * t_sector + (1.0 - alpha) * t_global)


def _rule_weight(lift: float) -> float:
    # stable, monotonic weight
    if lift is None or not np.isfinite(lift) or lift <= 1.0:
        return 0.0
    return float(np.clip(np.log(lift), 0.0, 2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_path", default="data/stocks/features/features_daily.parquet")
    ap.add_argument("--labels_path", default="data/stocks/features/labels_daily.parquet")
    ap.add_argument("--out_dir", default="data/stocks/ml_models/up20_20d")

    ap.add_argument("--train_start", default="2022-01-01")
    ap.add_argument("--train_end", default="2024-12-31")

    ap.add_argument("--target", default="y_up20_20d")
    ap.add_argument("--sector_col", default="wics_major")

    ap.add_argument("--top_global_rules", type=int, default=10)
    ap.add_argument("--top_sector_rules", type=int, default=5)
    ap.add_argument("--shrink_K", type=int, default=500)

    ap.add_argument("--min_support", type=float, default=0.01)
    ap.add_argument("--min_pos", type=int, default=50)

    args = ap.parse_args()

    features = pd.read_parquet(args.features_path)
    labels = pd.read_parquet(args.labels_path)

    features["date"] = pd.to_datetime(features["date"])
    labels["date"] = pd.to_datetime(labels["date"])

    df = features.merge(labels, on=["date", "ticker", "market", args.sector_col, "size_q"], how="inner")

    # train window
    train_start = pd.to_datetime(args.train_start)
    train_end = pd.to_datetime(args.train_end)
    df = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()

    if args.target not in df.columns:
        raise ValueError(f"missing target: {args.target}")

    y = df[args.target].astype(float)
    base_rate = float(y.mean())

    # feature columns
    feat_cols = [c for c in df.columns if _is_feature_col(c)]
    feat_cols = sorted(set(feat_cols))
    if not feat_cols:
        raise ValueError("no feature columns found. check build_ml_features output.")

    X = df[feat_cols].copy()

    # simple impute
    med = X.median(numeric_only=True)
    X = X.fillna(med)

    # time split: train(2022-2023), valid(2024)
    valid_start = pd.to_datetime("2024-01-01")
    is_valid = df["date"] >= valid_start

    X_train, y_train = X[~is_valid], y[~is_valid]
    X_valid, y_valid = X[is_valid], y[is_valid]

    pipe = Pipeline([
        ("scaler", RobustScaler(with_centering=True, with_scaling=True, quantile_range=(10.0, 90.0))),
        ("clf", LogisticRegression(
            solver="saga",
            penalty="l1",
            C=0.2,
            max_iter=3000,
            n_jobs=1,
            class_weight="balanced",
        )),
    ])

    pipe.fit(X_train, y_train)

    prob_valid = pipe.predict_proba(X_valid)[:, 1] if len(X_valid) else np.array([])
    auc = roc_auc_score(y_valid, prob_valid) if len(X_valid) and y_valid.nunique() > 1 else np.nan

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, out_dir / "model_joblib.pkl")

    # ---------------- Rules: Global ----------------
    rules_global = []
    for c in feat_cols:
        r = _scan_thresholds(
            df[c], y, base_rate=base_rate,
            min_support=args.min_support,
            min_pos=args.min_pos,
        )
        if r.get("ok"):
            r["feature"] = c
            r["weight"] = _rule_weight(r["lift"])
            rules_global.append(r)

    rules_global = sorted(rules_global, key=lambda z: z["score"], reverse=True)[: args.top_global_rules]

    # Map global threshold per feature for shrink reference
    global_by_feat = {r["feature"]: r for r in rules_global}

    # ---------------- Rules: Sector (raw + shrunk) ----------------
    sector_rules: Dict[str, List[Dict[str, Any]]] = {}
    sector_meta: Dict[str, Dict[str, Any]] = {}

    for sec, dsec in df.groupby(args.sector_col):
        yy = dsec[args.target].astype(float)
        n_pos = int(yy.sum())
        n_total = int(len(dsec))
        base_sec = float(yy.mean()) if n_total else np.nan

        sector_meta[str(sec)] = {"n_total": n_total, "n_pos": n_pos, "base_rate": base_sec}

        # if too small, skip (shrink will effectively use global only)
        rules_sec = []
        for c in feat_cols:
            r = _scan_thresholds(
                dsec[c], yy, base_rate=base_sec,
                min_support=args.min_support,
                min_pos=max(20, args.min_pos // 3),  # allow smaller within-sector
            )
            if r.get("ok"):
                r["feature"] = c
                r["weight"] = _rule_weight(r["lift"])
                rules_sec.append(r)

        rules_sec = sorted(rules_sec, key=lambda z: z["score"], reverse=True)[: args.top_sector_rules]

        # apply shrink for thresholds where we have a global rule for same feature+direction
        shrunk = []
        for r in rules_sec:
            feat = r["feature"]
            if feat in global_by_feat:
                rg = global_by_feat[feat]
                # shrink only if direction is same; else keep sector threshold
                if rg["direction"] == r["direction"]:
                    t_shr = _shrink_threshold(r["threshold"], rg["threshold"], n_pos, args.shrink_K)
                else:
                    t_shr = r["threshold"]
            else:
                t_shr = r["threshold"]

            rr = dict(r)
            rr["threshold_shrunk"] = float(t_shr)
            rr["alpha"] = float(n_pos / (n_pos + args.shrink_K)) if (n_pos + args.shrink_K) > 0 else 0.0
            shrunk.append(rr)

        sector_rules[str(sec)] = shrunk

    # ---------------- Base-rate table: sector x size_q ----------------
    base_tbl = (
        df.assign(is_up20=y.astype(int))
          .groupby([args.sector_col, "size_q"])["is_up20"]
          .agg(P_up20="mean", n="count")
          .reset_index()
          .rename(columns={args.sector_col: "wics_major"})
    )
    base_tbl.to_csv(out_dir / "base_rate_sector_size_train_2022_2024.csv", index=False, encoding="utf-8-sig")

    # ---------------- Save configs ----------------
    config = {
        "target": args.target,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "windows": "1,5,10,20,40",
        "peg_windows": "20,40",
        "universe_filter": {"min_mktcap": 300_000_000_000, "min_value20": 1_000_000_000},
        "global_rules_n": len(rules_global),
        "sector_rules_n_each": args.top_sector_rules,
        "shrink_K": args.shrink_K,
        "model": {"type": "LogisticRegression(L1)+RobustScaler"},
        "valid_auc_2024": float(auc) if np.isfinite(auc) else None,
        "base_rate_train": base_rate,
        "feature_cols": feat_cols,
        "ensemble_weights": {"model_prob": 0.5, "rule_global": 0.3, "rule_sector": 0.2},
    }

    (out_dir / "rules_global.json").write_text(json.dumps(rules_global, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "rules_sector.json").write_text(json.dumps({"meta": sector_meta, "rules": sector_rules}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # Lightweight training log
    msg = {
        "rows_train_total": int(len(df)),
        "base_rate": base_rate,
        "valid_auc_2024": float(auc) if np.isfinite(auc) else None,
        "global_rules": [{"feature": r["feature"], "direction": r["direction"], "threshold": r["threshold"], "lift": r["lift"]} for r in rules_global[:5]],
    }
    (out_dir / "train_log.json").write_text(json.dumps(msg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[train] OK")
    print(f"  out_dir={out_dir}")
    print(f"  rows={len(df):,}, base_rate={base_rate:.4f}, valid_auc_2024={auc}")
    print(f"  global_rules={len(rules_global)}, sector_rules_sectors={len(sector_rules)}")


if __name__ == "__main__":
    main()
