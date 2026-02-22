#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
render_daily_top5_report.py
- Build a daily HTML report for latest date (or specified date)
- Picks KOSPI TOP5 + KOSDAQ TOP5 using ensemble score (model + global rules + sector rules)
- Output:
  - docs/stocks/reports/daily_top5_top5.html
  - docs/stocks/reports/daily_top5_top5.csv
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


def zscore_by_group(df: pd.DataFrame, col: str, group_cols: list) -> pd.Series:
    def _z(x):
        mu = x.mean()
        sd = x.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return x * 0.0
        return (x - mu) / sd
    return df.groupby(group_cols)[col].transform(_z)


def apply_rules_score(df: pd.DataFrame, rules: list, use_shrunk: bool = False) -> pd.Series:
    score = np.zeros(len(df), dtype=float)
    for r in rules:
        feat = r["feature"]
        if feat not in df.columns:
            continue
        thr = r.get("threshold_shrunk") if use_shrunk else r.get("threshold")
        if thr is None:
            continue
        direction = r.get("direction")
        w = float(r.get("weight", 0.0))
        if w <= 0:
            continue
        x = df[feat]
        if direction == "ge":
            m = x >= thr
        else:
            m = x <= thr
        score += w * m.fillna(False).astype(float).values
    return pd.Series(score, index=df.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_path", default="data/stocks/features/features_daily.parquet")
    ap.add_argument("--model_dir", default="data/stocks/ml_models/up20_20d")
    ap.add_argument("--asof_date", default="", help="YYYY-MM-DD. empty => latest date in features")
    ap.add_argument("--top_k", type=int, default=5)

    ap.add_argument("--out_html", default="docs/stocks/reports/daily_top5_top5.html")
    ap.add_argument("--out_csv", default="docs/stocks/reports/daily_top5_top5.csv")
    args = ap.parse_args()

    feats = pd.read_parquet(args.features_path)
    feats["date"] = pd.to_datetime(feats["date"])

    if args.asof_date:
        asof = pd.to_datetime(args.asof_date)
    else:
        asof = feats["date"].max()

    df = feats[feats["date"] == asof].copy()

    model_dir = Path(args.model_dir)
    pipe = joblib.load(model_dir / "model_joblib.pkl")
    rules_global = json.loads((model_dir / "rules_global.json").read_text(encoding="utf-8"))
    rules_sector_blob = json.loads((model_dir / "rules_sector.json").read_text(encoding="utf-8"))
    rules_sector = rules_sector_blob["rules"]

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    feat_cols = config["feature_cols"]
    w_model = config["ensemble_weights"]["model_prob"]
    w_g = config["ensemble_weights"]["rule_global"]
    w_s = config["ensemble_weights"]["rule_sector"]

    X = df[feat_cols].copy()
    med = X.median(numeric_only=True)
    X = X.fillna(med)

    df["model_prob"] = pipe.predict_proba(X)[:, 1]
    df["rule_global"] = apply_rules_score(df, rules_global, use_shrunk=False)

    sec_scores = np.zeros(len(df), dtype=float)
    for sec, rules in rules_sector.items():
        m = (df["wics_major"] == sec)
        if not m.any():
            continue
        sec_scores[m.values] = apply_rules_score(df.loc[m], rules, use_shrunk=True).values
    df["rule_sector"] = sec_scores

    df["z_model"] = zscore_by_group(df, "model_prob", ["date", "market"])
    df["z_g"] = zscore_by_group(df, "rule_global", ["date", "market"])
    df["z_s"] = zscore_by_group(df, "rule_sector", ["date", "market"])
    df["score"] = w_model * df["z_model"] + w_g * df["z_g"] + w_s * df["z_s"]

    picks = (
        df.sort_values(["market", "score"], ascending=[True, False])
          .groupby("market", as_index=False)
          .head(args.top_k)
          .copy()
    )

    # Split for display
    kospi = picks[picks["market"].str.upper().str.contains("KOSPI")].copy()
    kosdaq = picks[picks["market"].str.upper().str.contains("KOSDAQ")].copy()

    # If market labels differ, fallback by market values
    if kospi.empty:
        kospi = picks[picks["market"].str.contains("KOSPI", na=False)].copy()
    if kosdaq.empty:
        kosdaq = picks[picks["market"].str.contains("KOSDAQ", na=False)].copy()

    out_html = Path(args.out_html)
    out_csv = Path(args.out_csv)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    picks_out = picks[[
        "date","market","ticker","name","wics_major","size_q",
        "market_cap","value","value_20d_mean",
        "score","model_prob","rule_global","rule_sector",
        "ret_5d","ret_10d","ret_20d","ret_40d",
        "ma_gap_20d","vol_20d","turnover_20d_mean","amihud_20d",
        "inv_per","inv_pbr","roe_proxy","peg_approx_20d","peg_approx_40d"
    ]].copy()

    picks_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    def _fmt_money(x):
        try:
            if pd.isna(x): return ""
            return f"{x/1e8:,.0f}억"
        except Exception:
            return ""

    for c in ["market_cap","value","value_20d_mean"]:
        if c in picks_out.columns:
            picks_out[c] = picks_out[c].apply(_fmt_money)

    html = f"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>Daily TOP5/TOP5 Report - {asof.date()}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, "Apple SD Gothic Neo", "Noto Sans KR"; margin: 22px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .card {{ border: 1px solid #e3e3e3; border-radius: 10px; padding: 12px 14px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #ededed; padding: 6px 8px; }}
    th {{ background: #fafafa; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    td.left, th.left {{ text-align: left; }}
    .small {{ color: #666; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Daily TOP5 / TOP5 (as-of {asof.date()})</h1>
  <div class="small">
    Ensemble score = 0.5·z(model_prob) + 0.3·z(global_rules) + 0.2·z(sector_rules_shrunk) / Entry=Close / Hold=20d / Cost=0.3% RT
  </div>

  <div class="grid">
    <div class="card">
      <h2>KOSPI TOP{args.top_k}</h2>
      {picks_out[picks_out["market"].str.contains("KOSPI", na=False)].to_html(index=False, escape=False)}
    </div>
    <div class="card">
      <h2>KOSDAQ TOP{args.top_k}</h2>
      {picks_out[picks_out["market"].str.contains("KOSDAQ", na=False)].to_html(index=False, escape=False)}
    </div>
  </div>

  <div class="card" style="margin-top:14px;">
    <h2>Raw Picks (CSV)</h2>
    <div class="small">{out_csv.as_posix()}</div>
  </div>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")
    print(f"[report] OK -> {out_html}")
    print(f"[csv]    OK -> {out_csv}")


if __name__ == "__main__":
    main()
