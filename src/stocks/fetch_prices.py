#!/usr/bin/env python3
"""
전종목 일별 OHLCV + 시가총액 수집
- PyKRX get_market_ohlcv_by_date() 우선
- FinanceDataReader 보조 (필요 시)
- 출력: data/stocks/raw/prices/{ticker}.csv
"""
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from pykrx import stock
# import FinanceDataReader as fdr  # 필요 시

START_DATE = "20200101"

def fetch_prices():
    print("[fetch_prices] 시작...")
    
    master_path = Path("data/stocks/master/listings.parquet")
    if not master_path.exists():
        print("⚠️  마스터 파일 없음")
        return
    df_master = pd.read_parquet(master_path)
    
    out_dir = Path("data/stocks/raw/prices")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    end_date = datetime.now().strftime("%Y%m%d")
    
    for idx, row in df_master.iterrows():
        ticker = row["ticker"]
        out_path = out_dir / f"{ticker}.csv"
        
        if out_path.exists():
            print(f"  [{idx+1}/{len(df_master)}] {ticker} 이미 존재, 스킵")
            continue
        
        try:
            df_ohlcv = stock.get_market_ohlcv_by_date(START_DATE, end_date, ticker)
            if df_ohlcv.empty:
                print(f"  [{idx+1}/{len(df_master)}] {ticker} 데이터 없음")
                continue
            
            df_ohlcv.reset_index(inplace=True)
            df_ohlcv.rename(columns={"날짜": "date", "종가": "close", "시가": "open",
                                      "고가": "high", "저가": "low", "거래량": "volume"}, inplace=True)
            
            # 시가총액 (별도 조회 필요 시)
            # df_cap = stock.get_market_cap_by_date(START_DATE, end_date, ticker)
            # df_ohlcv = df_ohlcv.merge(df_cap[['시가총액']], left_on='date', right_index=True, how='left')
            
            df_ohlcv.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  [{idx+1}/{len(df_master)}] {ticker} OK → {len(df_ohlcv)} rows")
        except Exception as e:
            print(f"  [{idx+1}/{len(df_master)}] {ticker} 오류: {e}")
    
    print("[fetch_prices] 완료")

if __name__ == "__main__":
    fetch_prices()
