#!/usr/bin/env python3
"""
CSV 요약 생성
"""
import os
from pathlib import Path
import pandas as pd

if os.getenv("PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
else:
    PROJECT_ROOT = Path(__file__).parent.parent.parent

print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}")

def export_master_summary():
    print("[1/2] master_summary.csv 생성 중...")
    
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
        has_features = (PROJECT_ROOT / f"data/stocks/analysis/{ticker}/features.parquet").exists()
        
        status_rows.append({
            "ticker": ticker,
            "name": name,
            "market": market,
            "has_price": has_price,
            "has_flow": has_flow,
            "has_features": has_features,
            "data_complete": all([has_price, has_flow, has_features])
        })
    
    df_status = pd.DataFrame(status_rows)
    out_path = PROJECT_ROOT / "docs/stocks/master_summary.csv"
    df_status.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ OK → {out_path}")

def export_signals_summary():
    print("[2/2] signals_summary.csv 생성 중...")
    
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
        
        total_signal = latest.get("signal", 0)
        
        if total_signal >= 3:
            opinion = "BUY"
        elif total_signal >= 1:
            opinion = "HOLD"
        else:
            opinion = "SELL"
        
        all_signals.append({
            "ticker": ticker,
            "name": name,
            "market": market,
            "close": latest.get("close", 0),
            "ret_1d": latest.get("ret_1d", 0),
            "signal": total_signal,
            "opinion": opinion
        })
    
    if all_signals:
        df_all = pd.DataFrame(all_signals)
        df_all.sort_values("signal", ascending=False, inplace=True)
        out_path = PROJECT_ROOT / "docs/stocks/signals_summary.csv"
        df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ OK → {out_path}")

def main():
    print("[export_data_summary] 시작...\n")
    
    (PROJECT_ROOT / "docs/stocks").mkdir(parents=True, exist_ok=True)
    
    export_master_summary()
    export_signals_summary()
    
    print("\n[export_data_summary] 완료 ✅")

if __name__ == "__main__":
    main()
