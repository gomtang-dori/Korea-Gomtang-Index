#!/usr/bin/env python3
"""
가격 데이터 수집 (병렬처리 + 증분 업데이트)
"""
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from pykrx import stock
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import INCREMENTAL_MODE, INCREMENTAL_DAYS, MAX_WORKERS, START_DATE

def fetch_one_ticker_price(ticker, start_date, end_date, out_path):
    """단일 종목 가격 수집"""
    try:
        df_ohlcv = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
        
        if df_ohlcv.empty:
            return f"{ticker}: 데이터 없음"
        
        df_ohlcv.reset_index(inplace=True)
        df_ohlcv.rename(columns={
            "날짜": "date", "종가": "close", "시가": "open",
            "고가": "high", "저가": "low", "거래량": "volume"
        }, inplace=True)
        
        # ✅ 증분 모드: 기존 데이터와 병합
        if INCREMENTAL_MODE and out_path.exists():
            df_old = pd.read_csv(out_path, encoding="utf-8-sig")
            df_ohlcv = pd.concat([df_old, df_ohlcv]).drop_duplicates(subset=["date"], keep="last")
        
        df_ohlcv.to_csv(out_path, index=False, encoding="utf-8-sig")
        return f"{ticker}: OK ({len(df_ohlcv)} rows)"
        
    except Exception as e:
        return f"{ticker}: 오류 ({e})"

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
    if INCREMENTAL_MODE:
        start_date = (datetime.now() - timedelta(days=INCREMENTAL_DAYS)).strftime("%Y%m%d")
        print(f"  [증분 모드] {start_date} ~ {end_date}")
    else:
        start_date = START_DATE
        print(f"  [백필 모드] {start_date} ~ {end_date}")
    
    # ✅ 병렬처리
    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx, row in df_master.iterrows():
            ticker = row["ticker"]
            out_path = out_dir / f"{ticker}.csv"
            
            if not INCREMENTAL_MODE and out_path.exists():
                print(f"  [{idx+1}/{len(df_master)}] {ticker} 이미 존재, 스킵")
                continue
            
            future = executor.submit(fetch_one_ticker_price, ticker, start_date, end_date, out_path)
            tasks.append(future)
        
        for i, future in enumerate(as_completed(tasks)):
            result = future.result()
            print(f"  [{i+1}/{len(tasks)}] {result}")
    
    print("[fetch_prices] 완료")

if __name__ == "__main__":
    fetch_prices()
