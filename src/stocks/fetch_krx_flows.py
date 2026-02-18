#!/usr/bin/env python3
"""
PyKRX로 전종목 투자자별 매수·매도·순매수 수집
- 백필: 2020-01-01 ~ 현재
- 출력: data/stocks/raw/krx_flows/{ticker}.csv
"""
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from pykrx import stock

START_DATE = "20200101"  # 백필 시작일

def fetch_krx_flows():
    print("[fetch_krx_flows] 시작...")
    
    # 마스터 로드
    master_path = Path("data/stocks/master/listings.parquet")
    if not master_path.exists():
        print("⚠️  마스터 파일 없음")
        return
    df_master = pd.read_parquet(master_path)
    
    out_dir = Path("data/stocks/raw/krx_flows")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    end_date = datetime.now().strftime("%Y%m%d")
    
    for idx, row in df_master.iterrows():
        ticker = row["ticker"]
        out_path = out_dir / f"{ticker}.csv"
        
        # 이미 있으면 증분만 (간단 구현은 전체 재수집)
        if out_path.exists():
            print(f"  [{idx+1}/{len(df_master)}] {ticker} 이미 존재, 스킵")
            continue
        
        try:
            # 투자자별 매매 (연기금·금융투자·기관·외국인)
            df_trade = stock.get_market_trading_by_date(
                START_DATE, end_date, ticker, detail=True
            )
            if df_trade.empty:
                print(f"  [{idx+1}/{len(df_master)}] {ticker} 데이터 없음")
                continue
            
            df_trade.reset_index(inplace=True)
            df_trade.rename(columns={"날짜": "date"}, inplace=True)
            df_trade.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  [{idx+1}/{len(df_master)}] {ticker} OK → {len(df_trade)} rows")
        except Exception as e:
            print(f"  [{idx+1}/{len(df_master)}] {ticker} 오류: {e}")
    
    print("[fetch_krx_flows] 완료")

if __name__ == "__main__":
    fetch_krx_flows()
