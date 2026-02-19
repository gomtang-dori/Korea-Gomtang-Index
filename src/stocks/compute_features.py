#!/usr/bin/env python3
"""
curated/raw -> analysis/features.parquet
- prices + flows + (optional) fundamentals + (optional) dart financials merge
- NOTE: 지금 단계는 "표시용 데이터 포함"이 목적(시그널은 추후 고도화)
"""

import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")) if os.getenv("PROJECT_ROOT") else Path.cwd()
print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}")

def _read_prices_csv(prices_path: Path) -> pd.DataFrame:
    df = pd.read_csv(prices_path, encoding="utf-8-sig")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        # 방어: 첫 컬럼이 날짜인 경우
        df.rename(columns={df.columns[0]: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
    # 필수 컬럼 방어
    for c in ["close", "open", "high", "low", "volume"]:
        if c not in df.columns:
            # 없는 경우 NaN 컬럼 생성(파이프라인이 멈추지 않게)
            df[c] = pd.NA
    return df.sort_values("date")

def _read_parquet_if_exists(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date") if "date" in df.columns else df

def compute_features():
    print("[compute_features] 시작...")

    master_path = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    if not master_path.exists():
        print(f"⚠️ master not found: {master_path}")
        return

    df_master = pd.read_parquet(master_path)
    if "ticker" not in df_master.columns:
        print("⚠️ listings.parquet missing 'ticker'")
        return

    total = len(df_master)
    success = 0
    fail = 0

    for idx, row in df_master.iterrows():
        ticker = str(row["ticker"])

        flows_path = PROJECT_ROOT / f"data/stocks/curated/{ticker}/flows_daily.parquet"
        prices_path = PROJECT_ROOT / f"data/stocks/raw/prices/{ticker}.csv"
        financials_path = PROJECT_ROOT / f"data/stocks/curated/{ticker}/financials_quarterly.parquet"
        fundamentals_path = PROJECT_ROOT / f"data/stocks/curated/{ticker}/fundamentals_daily.parquet"

        if (not prices_path.exists()) or (not flows_path.exists()):
            continue

        try:
            df_prices = _read_prices_csv(prices_path)
            df_flows = _read_parquet_if_exists(flows_path)

            # Merge: prices <- flows
            df = df_prices.merge(df_flows, on="date", how="left")

            # Merge: fundamentals (표시용)
            df_fund = _read_parquet_if_exists(fundamentals_path)
            if not df_fund.empty:
                # date 컬럼만 기준으로 join (fund: bps/per/pbr/eps/div/dps)
                keep_cols = [c for c in df_fund.columns if c in ["date", "bps", "per", "pbr", "eps", "div", "dps"]]
                df = df.merge(df_fund[keep_cols], on="date", how="left")

            # Returns
            df["ret_1d"] = df["close"].pct_change(1)
            df["ret_5d"] = df["close"].pct_change(5)
            df["ret_20d"] = df["close"].pct_change(20)
            df["ret_60d"] = df["close"].pct_change(60)

            # ---- signals (기존 유지: 추후 고도화 예정) ----
            signal_fund = 0
            if financials_path.exists():
                df_fin = pd.read_parquet(financials_path)
                if not df_fin.empty and "roe" in df_fin.columns:
                    latest_roe = df_fin["roe"].iloc[-1]
                    if pd.notna(latest_roe):
                        if latest_roe >= 0.15:
                            signal_fund = 2
                        elif latest_roe >= 0.10:
                            signal_fund = 1

            # flow signal은 "표시용 데이터가 존재한다" 정도로만 유지
            # curate_flows.py가 생성하는 rolling 컬럼명: foreign_value_net_20d_sum, inst_value_net_20d_sum 등
            signal_flow = 0
            if "foreign_value_net_20d_sum" in df.columns:
                v = df["foreign_value_net_20d_sum"].iloc[-1]
                if pd.notna(v) and v > 0:
                    signal_flow += 1
            if "inst_value_net_20d_sum" in df.columns:
                v = df["inst_value_net_20d_sum"].iloc[-1]
                if pd.notna(v) and v > 0:
                    signal_flow += 1

            df["signal_fundamentals"] = signal_fund
            df["signal_flows"] = signal_flow
            df["signal"] = signal_fund + signal_flow

            out_dir = PROJECT_ROOT / f"data/stocks/analysis/{ticker}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "features.parquet"
            df.to_parquet(out_path, index=False)

            success += 1
            if success <= 3:
                print(f"  [{idx+1}/{total}] {ticker} OK (has_fund={'yes' if not df_fund.empty else 'no'})")
            elif success == 4:
                print("  ... (로그 생략, 계속 진행)")

        except Exception as e:
            fail += 1
            if fail <= 20:
                print(f"  [{idx+1}/{total}] {ticker} FAIL: {e}")

    print(f"[compute_features] 완료 (성공: {success}/{total}, 실패로그표시: {min(fail,20)})")

if __name__ == "__main__":
    compute_features()
