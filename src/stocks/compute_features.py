#!/usr/bin/env python3
"""
curated → analysis/features.parquet
"""
import os
from pathlib import Path
import pandas as pd

# ✅ 프로젝트 루트 기준 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent  # src/stocks → src → root

def compute_features():
    print("[compute_features] 시작...")
    
    master_path = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    if not master_path.exists():
        print("⚠️  마스터 파일 없음")
        return
    df_master = pd.read_parquet(master_path)
    
    for idx, row in df_master.iterrows():
        ticker = row["ticker"]
        
        flows_path = PROJECT_ROOT / f"data/stocks/curated/{ticker}/flows_daily.parquet"
        prices_path = PROJECT_ROOT / f"data/stocks/raw/prices/{ticker}.csv"
        financials_path = PROJECT_ROOT / f"data/stocks/curated/{ticker}/financials_quarterly.parquet"
        
        if not flows_path.exists() or not prices_path.exists():
            print(f"  [{idx+1}/{len(df_master)}] {ticker} 필수 파일 없음, 스킵")
            continue
        
        try:
            df_flows = pd.read_parquet(flows_path)
            df_prices = pd.read_csv(prices_path, encoding="utf-8-sig")
            
            df_prices["date"] = pd.to_datetime(df_prices["date"])
            df_flows["date"] = pd.to_datetime(df_flows["date"])
            
            df = df_prices.merge(df_flows, on="date", how="left")
            
            df["ret_1d"] = df["close"].pct_change(1)
            df["ret_5d"] = df["close"].pct_change(5)
            df["ret_20d"] = df["close"].pct_change(20)
            df["ret_60d"] = df["close"].pct_change(60)
            
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
            
            signal_flow = 0
            if "foreign_net_20d" in df.columns:
                latest_foreign = df["foreign_net_20d"].iloc[-1]
                if pd.notna(latest_foreign) and latest_foreign > 0:
                    signal_flow += 1
            
            if "inst_total_net_20d" in df.columns:
                latest_inst = df["inst_total_net_20d"].iloc[-1]
                if pd.notna(latest_inst) and latest_inst > 0:
                    signal_flow += 1
            
            df["signal_fundamentals"] = signal_fund
            df["signal_flows"] = signal_flow
            df["signal"] = signal_fund + signal_flow
            
            out_dir = PROJECT_ROOT / f"data/stocks/analysis/{ticker}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "features.parquet"
            
            df.to_parquet(out_path, index=False)
            
            latest_signal = df["signal"].iloc[-1]
            print(f"  [{idx+1}/{len(df_master)}] {ticker} features OK (signal={latest_signal:.0f})")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(df_master)}] {ticker} 오류: {e}")
    
    print("[compute_features] 완료")

if __name__ == "__main__":
    compute_features()
