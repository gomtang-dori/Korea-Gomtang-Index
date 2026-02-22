#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_ml_features.py
- Input: data/stocks/mart/panel_daily.parquet
- Output:
  - data/stocks/features/features_daily.parquet
  - data/stocks/features/labels_daily.parquet

Windows: 1,5,10,20,40
PEG: approx PEG only for w=20,40
Universe filter: market_cap >= 3000억, value_20d_mean >= 10억
Label winsorize by date: default 1%~99% (optional)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS_DEFAULT = [1, 5, 10, 20, 40]
PEG_WINDOWS_DEFAULT = [20, 40]


def _winsorize_by_date(s: pd.Series, ql: float, qh: float) -> pd.Series:
  # NOTE: ql/qh are within [0,1]
    if s.dropna().empty:
        return s
    lo = s.quantile(ql)
    hi = s.quantile(qh)
    return s.clip(lo, hi)

def _min_periods(window: int, target: int) -> int:
    """
    Pandas rolling constraint: min_periods must be <= window.
    Also min_periods must be >= 1.
    """
    return max(1, min(int(window), int(target)))

def _safe_log1p(x: pd.Series) -> pd.Series:
    return np.log1p(x.clip(lower=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_path", default="data/stocks/mart/panel_daily.parquet")
    ap.add_argument("--out_features", default="data/stocks/features/features_daily.parquet")
    ap.add_argument("--out_labels", default="data/stocks/features/labels_daily.parquet")
    ap.add_argument("--latest_only", action="store_true", help="Only output features for latest date (labels not required).")
    ap.add_argument("--windows", default="1,5,10,20,40")
    ap.add_argument("--peg_windows", default="20,40")

    ap.add_argument("--min_mktcap", type=float, default=300_000_000_000.0)  # 3000억
    ap.add_argument("--min_value20", type=float, default=1_000_000_000.0)    # 10억

    ap.add_argument("--winsorize_labels", action="store_true", default=True)
    ap.add_argument("--winsor_q_low", type=float, default=0.01)
    ap.add_argument("--winsor_q_high", type=float, default=0.99)

    args = ap.parse_args()

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    peg_windows = [int(x.strip()) for x in args.peg_windows.split(",") if x.strip()]

    panel_path = Path(args.panel_path)
    if not panel_path.exists():
        raise FileNotFoundError(f"missing panel parquet: {panel_path}")

    df = pd.read_parquet(panel_path)
    if "date" not in df.columns or "ticker" not in df.columns or "close" not in df.columns:
        raise ValueError("panel must contain at least: date, ticker, close")

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Ensure required columns for filters exist
    if "market_cap" not in df.columns:
        raise ValueError("panel missing market_cap (needed for X=3000억 filter)")
    if "value" not in df.columns:
        raise ValueError("panel missing value (needed for Y=10억 filter and liquidity features)")
    if "shares" not in df.columns and "volume" in df.columns:
        # turnover will be NA; that's acceptable
        pass

    # Fill wics major
    if "wics_major" not in df.columns:
        df["wics_major"] = "UNKNOWN"
    else:
        df["wics_major"] = df["wics_major"].fillna("UNKNOWN").astype(str)

    # Market label
    if "market" not in df.columns:
        df["market"] = "UNKNOWN"
    else:
        df["market"] = df["market"].fillna("UNKNOWN").astype(str)

    g = df.groupby("ticker", group_keys=False)

    # ------------- LABELS (future) -------------
    horizons = [20, 40]
    for h in horizons:
        df[f"close_fwd_{h}d"] = g["close"].shift(-h)
        df[f"fwd_ret_{h}d"] = df[f"close_fwd_{h}d"] / df["close"] - 1.0
        df[f"y_up20_{h}d"] = (df[f"fwd_ret_{h}d"] >= 0.20).astype("float")

    # DN10 / DN group labeling (for analysis convenience)
    def _grp(ret: pd.Series) -> pd.Series:
        # IMPORTANT: DN10 must be checked before DN
        out = pd.Series(["OTHER"] * len(ret), index=ret.index)
        out[ret >= 0.20] = "UP20"
        out[(ret >= 0.0) & (ret < 0.20)] = "UP0_20"
        out[ret <= -0.10] = "DN10"
        out[(ret < 0.0) & (ret > -0.10)] = "DN"
        return out

    df["grp20"] = _grp(df["fwd_ret_20d"])
    df["grp40"] = _grp(df["fwd_ret_40d"])

    # Optional winsorize labels by date
    if args.winsorize_labels and not args.latest_only:
        for h in horizons:
            col = f"fwd_ret_{h}d"
            df[col] = df.groupby("date")[col].transform(
                lambda s: _winsorize_by_date(s, args.winsor_q_low, args.winsor_q_high)
            )

    # ------------- FEATURES (past/current only) -------------
    # Returns
    df["ret_1d"] = g["close"].pct_change(1)
    for w in windows:
        df[f"ret_{w}d"] = g["close"].pct_change(w)
        df[f"logret_{w}d"] = np.log(df["close"]) - np.log(g["close"].shift(w))

        # moving average / gap
        mp = _min_periods(w, max(2, w // 3))
        ma = g["close"].rolling(w, min_periods=mp).mean().reset_index(level=0, drop=True)      
        df[f"ma_{w}d"] = ma
        df[f"ma_gap_{w}d"] = df["close"] / ma - 1.0

    # Volatility based on ret_1d
    for w in windows:
        mp = _min_periods(w, max(5, w // 2))
        df[f"vol_{w}d"] = g["ret_1d"].rolling(w, min_periods=mp).std().reset_index(level=0, drop=True)
      
    # Liquidity features
    df["log_value"] = _safe_log1p(df["value"])
    if "volume" in df.columns:
        df["log_volume"] = _safe_log1p(df["volume"])
    else:
        df["log_volume"] = np.nan

    # value rolling sums/means for normalization & filter
    for w in windows:
        mp = _min_periods(w, max(5, w // 2))
        df[f"value_{w}d_sum"] = g["value"].rolling(w, min_periods=mp).sum().reset_index(level=0, drop=True)
        df[f"value_{w}d_mean"] = g["value"].rolling(w, min_periods=mp).mean().reset_index(level=0, drop=True)
  
    # turnover
    if "shares" in df.columns and "volume" in df.columns:
        df["turnover_1d"] = df["volume"] / df["shares"]
        for w in windows:
            mp = _min_periods(w, max(5, w // 2))
            df[f"turnover_{w}d_mean"] = g["turnover_1d"].rolling(w, min_periods=mp).mean().reset_index(level=0, drop=True)
  
    else:
        df["turnover_1d"] = np.nan
        for w in windows:
            df[f"turnover_{w}d_mean"] = np.nan

    # Amihud
    df["amihud_1d"] = (df["ret_1d"].abs() / df["value"]).replace([np.inf, -np.inf], np.nan)
    for w in windows:
        mp = _min_periods(w, max(5, w // 2))
        df[f"amihud_{w}d"] = g["amihud_1d"].rolling(w, min_periods=mp).mean().reset_index(level=0, drop=True)
      
    # Flows: compute rolling sums & normalized flows (independent of curate windows)
    flow_investors = ["foreign", "inst", "pension", "fininv"]
    for inv in flow_investors:
        base = f"{inv}_value_net"
        if base in df.columns:
            for w in windows:
                mp = _min_periods(w, max(5, w // 2))
                df[f"{inv}_value_net_{w}d_sum"] = g[base].rolling(w, min_periods=mp).sum().reset_index(level=0, drop=True)              
              
                # normalized by value sum
                df[f"{inv}_net_to_value_{w}d"] = df[f"{inv}_value_net_{w}d_sum"] / df[f"value_{w}d_sum"]
        else:
            for w in windows:
                df[f"{inv}_value_net_{w}d_sum"] = np.nan
                df[f"{inv}_net_to_value_{w}d"] = np.nan

    # Value features
    if "per" in df.columns:
        df["inv_per"] = 1.0 / df["per"].replace(0, np.nan)
    else:
        df["inv_per"] = np.nan

    if "pbr" in df.columns:
        df["inv_pbr"] = 1.0 / df["pbr"].replace(0, np.nan)
    else:
        df["inv_pbr"] = np.nan

    # ROE proxy from EPS/BPS
    if "eps" in df.columns and "bps" in df.columns:
        df["roe_proxy"] = (df["eps"] / df["bps"]).where((df["eps"] > 0) & (df["bps"] > 0))
    else:
        df["roe_proxy"] = np.nan

    # Approx PEG only for 20/40 (requires per, eps)
    if "per" in df.columns and "eps" in df.columns:
        for w in peg_windows:
            eps_prev = g["eps"].shift(w)
            eps_g = (df["eps"] / eps_prev) - 1.0
            eps_g = eps_g.where((df["eps"] > 0) & (eps_prev > 0) & (eps_g > 0))
            eps_g = eps_g.clip(lower=0.01)  # floor to avoid blow-ups
            df[f"eps_g_{w}d"] = eps_g
            df[f"peg_approx_{w}d"] = df["per"] / (100.0 * eps_g)
    else:
        for w in peg_windows:
            df[f"eps_g_{w}d"] = np.nan
            df[f"peg_approx_{w}d"] = np.nan

    # ---------------- Universe filter ----------------
    # Use value_20d_mean for Y filter (must exist)
    df["value_20d_mean"] = df["value_20d_mean"] if "value_20d_mean" in df.columns else df["value_20d_mean"]
    uni = df.copy()
    uni = uni[
        (uni["market_cap"].notna()) & (uni["market_cap"] >= args.min_mktcap) &
        (uni["value_20d_mean"].notna()) & (uni["value_20d_mean"] >= args.min_value20)
    ]

    # Size bucket (quintile within date, within filtered universe)
    uni["size_q"] = uni.groupby("date")["market_cap"].transform(
        lambda s: pd.qcut(s, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
    )

    # Latest-only mode: output only last date's features (labels not needed)
    out_features = Path(args.out_features)
    out_labels = Path(args.out_labels)
    out_features.parent.mkdir(parents=True, exist_ok=True)
    out_labels.parent.mkdir(parents=True, exist_ok=True)

    if args.latest_only:
        latest_date = uni["date"].max()
        u2 = uni[uni["date"] == latest_date].copy()
        # features: keep only non-future columns
        label_cols = [c for c in u2.columns if c.startswith("close_fwd_") or c.startswith("fwd_ret_") or c.startswith("y_up20_") or c.startswith("grp")]
        feat = u2.drop(columns=label_cols, errors="ignore")
        feat.to_parquet(out_features, index=False)
        # Write an empty labels file (optional)
        pd.DataFrame({"date": [], "ticker": []}).to_parquet(out_labels, index=False)
        print(f"[features] latest_only OK: {out_features} rows={len(feat):,} latest_date={latest_date.date()}")
        return

    # Full mode: split features / labels
    label_cols = [c for c in uni.columns if c.startswith("close_fwd_") or c.startswith("fwd_ret_") or c.startswith("y_up20_") or c.startswith("grp")]
    labels = uni[["date", "ticker", "market", "wics_major", "size_q"] + label_cols].copy()

    # Features exclude future columns
    features = uni.drop(columns=label_cols, errors="ignore")

    features.to_parquet(out_features, index=False)
    labels.to_parquet(out_labels, index=False)

    print(f"[features] wrote: {out_features} rows={len(features):,} cols={len(features.columns)}")
    print(f"[labels]   wrote: {out_labels} rows={len(labels):,} cols={len(labels.columns)}")


if __name__ == "__main__":
    main()
