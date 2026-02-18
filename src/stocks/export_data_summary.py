#!/usr/bin/env python3
"""
수집된 데이터 요약 CSV 생성
"""
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent

def export_master_summary():
    print("[1/5] master_summary.csv 생성 중...")
    
    master_path = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    if not master_path.exists():
        print("  ⚠️  마스터 파일 없음")
        return
    
    df_master = pd.read_parquet(master_path)
    
    status_rows = []
    for _, row in df_master.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        market = row["market"]
        
        has_price = (PROJECT_ROOT / f"data/stocks/raw/prices/{ticker}.csv").exists()
        has_flow = (PROJECT_ROOT / f"data/stocks/raw/krx_flows/{ticker}.csv").exists()
        has_dart = (PROJECT_ROOT / f"data/stocks/raw/dart/{ticker}_quarterly.json").exists()
        has_financials = (PROJECT_ROOT / f"data/stocks/curated/{ticker}/financials_quarterly.parquet").exists()
        has_features = (PROJECT_ROOT / f"data/stocks/analysis/{ticker}/features.parquet").exists()
        
        price_count = 0
        flow_count = 0
        
        if has_price:
            df_p = pd.read_csv(PROJECT_ROOT / f"data/stocks/raw/prices/{ticker}.csv", encoding="utf-8-sig")
            price_count = len(df_p)
        
        if has_flow:
            df_f = pd.read_csv(PROJECT_ROOT / f"data/stocks/raw/krx_flows/{ticker}.csv", encoding="utf-8-sig")
            flow_count = len(df_f)
        
        status_rows.append({
            "ticker": ticker,
            "name": name,
            "market": market,
            "has_price": has_price,
            "price_rows": price_count,
            "has_flow": has_flow,
            "flow_rows": flow_count,
            "has_dart": has_dart,
            "has_financials": has_financials,
            "has_features": has_features,
            "data_complete": all([has_price, has_flow, has_financials, has_features])
        })
    
    df_status = pd.DataFrame(status_rows)
    out_path = PROJECT_ROOT / "docs/stocks/master_summary.csv"
    df_status.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ OK → {out_path} ({len(df_status)} 종목)")

def export_signals_summary():
    print("[2/5] signals_summary.csv 생성 중...")
    
    master_path = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    if not master_path.exists():
        return
    df_master = pd.read_parquet(master_path)
    
    all_signals = []
    
    for _, row in df_master.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        market = row["market"]
        
        feat_path = PROJECT_ROOT / f"data/stocks/analysis/{ticker}/features.parquet"
        if not feat_path.exists():
            continue
        
        df_feat = pd.read_parquet(feat_path)
        if df_feat.empty:
            continue
        
        latest = df_feat.iloc[-1]
        
        signal_fund = latest.get("signal_fundamentals", 0)
        signal_flow = latest.get("signal_flows", 0)
        total_signal = latest.get("signal", 0)
        
        if total_signal >= 3:
            opinion = "BUY"
            position = "+10~+20%"
        elif total_signal >= 1:
            opinion = "HOLD"
            position = "0%"
        else:
            opinion = "SELL"
            position = "-10~-30%"
        
        all_signals.append({
            "ticker": ticker,
            "name": name,
            "market": market,
            "date": latest.get("date", ""),
            "close": latest.get("close", 0),
            "ret_1d": latest.get("ret_1d", 0),
            "ret_5d": latest.get("ret_5d", 0),
            "ret_20d": latest.get("ret_20d", 0),
            "signal_fundamentals": signal_fund,
            "signal_flows": signal_flow,
            "signal_total": total_signal,
            "opinion": opinion,
            "position_guide": position
        })
    
    if all_signals:
        df_all = pd.DataFrame(all_signals)
        df_all.sort_values("signal_total", ascending=False, inplace=True)
        out_path = PROJECT_ROOT / "docs/stocks/signals_summary.csv"
        df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ OK → {out_path} ({len(df_all)} 종목)")
    else:
        print("  ⚠️  투자 의견 데이터 없음")

def main():
    print("[export_data_summary] 시작...\n")
    
    (PROJECT_ROOT / "docs/stocks").mkdir(parents=True, exist_ok=True)
    
    export_master_summary()
    export_signals_summary()
    
    print("\n[export_data_summary] 완료 ✅")

if __name__ == "__main__":
    main()
