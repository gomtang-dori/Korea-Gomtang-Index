#!/usr/bin/env python3
"""
가격 데이터 수집 (병렬처리 + 증분)
"""
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from pykrx import stock
from concurrent.futures import ThreadPoolExecutor, as_completed

# ✅ 설정
INCREMENTAL_MODE = os.getenv("INCREMENTAL_MODE", "false").lower() == "true"
INCREMENTAL_DAYS = int(os.getenv("INCREMENTAL_DAYS", "5"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
START_DATE = "20200101"

def fetch_one_ticker_price(ticker, start_date, end_date, out_path):
    try:
        df_ohlcv = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
        
        if df_ohlcv.empty:
            return f"{ticker}: 데이터 없음"
        
        df_ohlcv.reset_index(inplace=True)
        df_ohlcv.rename(columns={
            "날짜": "date", "종가": "close", "시가": "open",
            "고가": "high", "저가": "low", "거래량": "volume"
        }, inplace=True)
        
        if INCREMENTAL_MODE and out_path.exists():
            df_old = pd.read_csv(out_path, encoding="utf-8-sig")
            df_ohlcv = pd.concat([df_old, df_ohlcv]).drop_duplicates(subset=["date"], keep="last")
        
        df_ohlcv.to_csv(out_path, index=False, encoding="utf-8-sig")
        return f"{ticker}: OK ({len(df_ohlcv)} rows)"
        
    except Exception as e:
        return f"{ticker}: 오류 ({e})"

def fetch_prices():
    print("[fetch_prices] 시작...")
    print(f"  INCREMENTAL_MODE={INCREMENTAL_MODE}, MAX_WORKERS={MAX_WORKERS}")
    
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
    
    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx, row in df_master.iterrows():
            ticker = row["ticker"]
            out_path = out_dir / f"{ticker}.csv"
            
            if not INCREMENTAL_MODE and out_path.exists():
                continue
            
            future = executor.submit(fetch_one_ticker_price, ticker, start_date, end_date, out_path)
            tasks.append((ticker, future))
        
        for i, (ticker, future) in enumerate(tasks):
            result = future.result()
            print(f"  [{i+1}/{len(tasks)}] {result}")
    
    print("[fetch_prices] 완료")

if __name__ == "__main__":
    fetch_prices()
