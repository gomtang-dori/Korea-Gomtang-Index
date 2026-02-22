#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
backtest_2025_2026_top5_top5.py
- Uses trained model + rules (global + sector shrink)
- Scores each date; picks TOP5 KOSPI + TOP5 KOSDAQ
- Entry at close; hold 20 trading days (uses fwd_ret_20d label)
- Round-trip cost 0.3% (net = (1+ret)*(1-0.003)-1)
- Outputs:
  - docs/stocks/ml/backtest_2025_2026_trades.csv
  - docs/stocks/ml/backtest_2025_2026_summary.html
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
    ap.add_argument("--labels_path", default="data/stocks/features/labels_daily.parquet")
    ap.add_argument("--model_dir", default="data/stocks/ml_models/up20_20d")

    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-12-31")

    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--holding_days", type=int, default=20)
    ap.add_argument("--roundtrip_cost", type=float, default=0.003)

    ap.add_argument("--out_dir", default="docs/stocks/ml")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features_path)
    labs = pd.read_parquet(args.labels_path)
    feats["date"] = pd.to_datetime(feats["date"])
    labs["date"] = pd.to_datetime(labs["date"])

    df = feats.merge(labs, on=["date", "ticker", "market", "wics_major", "size_q"], how="inner")
    df = df[(df["date"] >= pd.to_datetime(args.start)) & (df["date"] <= pd.to_datetime(args.end))].copy()

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

    # Impute for model
    X = df[feat_cols].copy()
    med = X.median(numeric_only=True)
    X = X.fillna(med)

    df["model_prob"] = pipe.predict_proba(X)[:, 1]
    df["rule_global"] = apply_rules_score(df, rules_global, use_shrunk=False)

    # sector rule score with shrunk thresholds
    sec_scores = np.zeros(len(df), dtype=float)
    for sec, rules in rules_sector.items():
        m = (df["wics_major"] == sec)
        if not m.any():
            continue
        sec_scores[m.values] = apply_rules_score(df.loc[m], rules, use_shrunk=True).values
    df["rule_sector"] = sec_scores

    # z-scores within date+market (separate KOSPI/KOSDAQ ranking)
    df["z_model"] = zscore_by_group(df, "model_prob", ["date", "market"])
    df["z_g"] = zscore_by_group(df, "rule_global", ["date", "market"])
    df["z_s"] = zscore_by_group(df, "rule_sector", ["date", "market"])

    df["score"] = w_model * df["z_model"] + w_g * df["z_g"] + w_s * df["z_s"]

    # pick topK each market each date
    picks = (
        df.sort_values(["date", "market", "score"], ascending=[True, True, False])
          .groupby(["date", "market"], as_index=False)
          .head(args.top_k)
          .copy()
    )

    # realized return: use fwd_ret_20d (holding_days fixed to 20 now)
    ret_col = f"fwd_ret_{args.holding_days}d"
    close_fwd_col = f"close_fwd_{args.holding_days}d"
    if ret_col not in picks.columns:
        raise ValueError(f"missing label column {ret_col} in labels parquet")

    picks["ret_gross"] = picks[ret_col]
    picks["ret_net"] = (1.0 + picks["ret_gross"]) * (1.0 - args.roundtrip_cost) - 1.0
    picks["hit_up20"] = (picks["ret_gross"] >= 0.20).astype(int)
    picks["hit_pos"] = (picks["ret_gross"] > 0).astype(int)
    picks["exit_close"] = picks[close_fwd_col]

    # daily basket (entry-day) performance
    daily = (
        picks.groupby(["date"], as_index=False)
             .agg(
                 n=("ticker", "count"),
                 ret_net_mean=("ret_net", "mean"),
                 ret_gross_mean=("ret_gross", "mean"),
                 hit_up20=("hit_up20", "mean"),
                 hit_pos=("hit_pos", "mean"),
             )
             .sort_values("date")
    )
    daily["cum_ret_net"] = (1.0 + daily["ret_net_mean"].fillna(0.0)).cumprod() - 1.0

    # summary
    summary = {
        "period": f"{args.start}~{args.end}",
        "top_k_each_market": args.top_k,
        "holding_days": args.holding_days,
        "roundtrip_cost": args.roundtrip_cost,
        "trades": int(len(picks)),
        "avg_ret_net": float(picks["ret_net"].mean(skipna=True)),
        "med_ret_net": float(picks["ret_net"].median(skipna=True)),
        "hit_up20_rate": float(picks["hit_up20"].mean(skipna=True)),
        "hit_pos_rate": float(picks["hit_pos"].mean(skipna=True)),
        "cum_ret_net_entry_day": float(daily["cum_ret_net"].iloc[-1]) if len(daily) else None,
    }

    trades_path = out_dir / "backtest_2025_2026_trades.csv"
    daily_path = out_dir / "backtest_2025_2026_daily.csv"
    html_path = out_dir / "backtest_2025_2026_summary.html"
    json_path = out_dir / "backtest_2025_2026_summary.json"

    picks.to_csv(trades_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # HTML summary
    html = f"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>Backtest 2025-2026 (TOP{args.top_k}/TOP{args.top_k}, 20d hold)</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, "Apple SD Gothic Neo", "Noto Sans KR"; margin: 24px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 14px 16px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #e5e5e5; padding: 6px 8px; text-align: right; }}
    th {{ background: #fafafa; }}
    td.left, th.left {{ text-align: left; }}
    .small {{ color:#666; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>백테스트 요약 (2025~2026)</h1>
  <div class="card">
    <div class="small">매일 리밸런싱 / 종가 진입 / 20영업일 보유 / 왕복 비용 0.3% / KOSPI TOP{args.top_k} + KOSDAQ TOP{args.top_k}</div>
    <pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
  </div>

  <div class="card">
    <h2>일별(Entry-day) 누적 성과</h2>
    <div class="small">주의: 아래 누적은 “매일 새로 진입한 바스켓”의 평균 성과를 누적 곱으로 표현한 지표입니다(신호 품질 확인용).</div>
    {daily.tail(200).to_html(index=False, escape=False)}
  </div>

  <div class="card">
    <h2>트레이드 샘플(최근 200개)</h2>
    {picks.tail(200)[["date","market","ticker","wics_major","size_q","score","model_prob","rule_global","rule_sector","ret_gross","ret_net","hit_up20"]].to_html(index=False, escape=False)}
  </div>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")

    print("[backtest] OK")
    print(f"  trades={len(picks):,} -> {trades_path}")
    print(f"  summary -> {html_path}")


if __name__ == "__main__":
    main()
