#!/usr/bin/env python3
"""
curated 데이터 → analysis/features.parquet
- 펀더멘털 점수 (ROE ≥15% = +2, 10~15% = +1)
- 수급 점수 (외국인 20일 순매수 > 시총 1% = +2)
- 종합 Signal
"""
import os
from pathlib import Path
import pandas as pd

def compute_features():
    print("[compute_features] 시작...")
    
    master_path = Path("data/stocks/master/listings.parquet")
    if not master_path.exists():
        print("⚠️  마스터 파일 없음")
        return
    df_master = pd.read_parquet(master_path)
    
    for idx, row in df_master.iterrows():
        ticker = row["ticker"]
        
        # curated 파일 로드
        flows_path = Path(f"data/stocks/curated/{ticker}/flows_daily.parquet")
        prices_path = Path(f"data/stocks/curated/{ticker}/prices.parquet")
        financials_q = Path(f"data/stocks/curated/{ticker}/financials_quarterly.parquet")
        
        if not flows_path.exists() or not prices_path.exists():
            continue
        
        df_flows = pd.read_parquet(flows_path)
        df_prices = pd.read_parquet(prices_path)
        
        # 예시 Signal 계산
        # df_features = df_prices.merge(df_flows, on='date', how='left')
        # df_features['signal_fundamentals'] = 0  # ROE 기반 (재무 로드 후)
        # df_features['signal_flows'] = 0  # 외국인 순매수 기반
        
        out_dir = Path(f"data/stocks/analysis/{ticker}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "features.parquet"
        
        # df_features.to_parquet(out_path, index=False)
        print(f"  [{idx+1}/{len(df_master)}] {ticker} features 계산 (스킵)")
    
    print("[compute_features] 완료")

if __name__ == "__main__":
    compute_features()
