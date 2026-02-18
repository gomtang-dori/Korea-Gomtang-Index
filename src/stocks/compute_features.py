#!/usr/bin/env python3
"""
curated → analysis/features.parquet
- 펀더멘털 점수 (ROE 기반)
- 수급 점수 (외국인·기관 순매수)
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
        
        # 파일 경로
        flows_path = Path(f"data/stocks/curated/{ticker}/flows_daily.parquet")
        prices_path = Path(f"data/stocks/raw/prices/{ticker}.csv")
        financials_path = Path(f"data/stocks/curated/{ticker}/financials_quarterly.parquet")
        
        if not flows_path.exists() or not prices_path.exists():
            continue
        
        try:
            # 데이터 로드
            df_flows = pd.read_parquet(flows_path)
            df_prices = pd.read_csv(prices_path, encoding="utf-8-sig")
            
            # 날짜 정렬
            df_prices["date"] = pd.to_datetime(df_prices["date"])
            df_flows["date"] = pd.to_datetime(df_flows["날짜"])
            
            # Merge
            df = df_prices.merge(df_flows, on="date", how="left")
            
            # ✅ 수익률 계산
            df["ret_1d"] = df["close"].pct_change(1)
            df["ret_5d"] = df["close"].pct_change(5)
            df["ret_20d"] = df["close"].pct_change(20)
            
            # ✅ 펀더멘털 점수 (ROE 기반)
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
            
            # ✅ 수급 점수 (외국인·기관 20일 순매수)
            signal_flow = 0
            
            # PyKRX 컬럼명 확인 (예: '외국인합계', '기관합계')
            foreign_col = None
            inst_col = None
            
            for col in df_flows.columns:
                if "외국인" in col and "합계" in col:
                    foreign_col = col
                if "기관" in col and "합계" in col:
                    inst_col = col
            
            if foreign_col:
                df[f"{foreign_col}_20d"] = df[foreign_col].rolling(20).sum()
                latest_foreign = df[f"{foreign_col}_20d"].iloc[-1]
                if pd.notna(latest_foreign) and latest_foreign > 0:
                    signal_flow += 1
            
            if inst_col:
                df[f"{inst_col}_20d"] = df[inst_col].rolling(20).sum()
                latest_inst = df[f"{inst_col}_20d"].iloc[-1]
                if pd.notna(latest_inst) and latest_inst > 0:
                    signal_flow += 1
            
            # ✅ 종합 Signal
            df["signal_fundamentals"] = signal_fund
            df["signal_flows"] = signal_flow
            df["signal"] = signal_fund + signal_flow
            
            # 저장
            out_dir = Path(f"data/stocks/analysis/{ticker}")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "features.parquet"
            
            df.to_parquet(out_path, index=False)
            print(f"  [{idx+1}/{len(df_master)}] {ticker} features OK (signal={df['signal'].iloc[-1]})")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(df_master)}] {ticker} 오류: {e}")
    
    print("[compute_features] 완료")

if __name__ == "__main__":
    compute_features()
