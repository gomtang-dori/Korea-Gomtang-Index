#!/usr/bin/env python3
"""
PyKRX 투자자별 매매 수집
"""
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from pykrx import stock
from concurrent.futures import ThreadPoolExecutor, as_completed

INCREMENTAL_MODE = os.getenv("INCREMENTAL_MODE", "false").lower() == "true"
INCREMENTAL_DAYS = int(os.getenv("INCREMENTAL_DAYS", "5"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
START_DATE = "20200101"

def fetch_one_ticker(ticker, start_date, end_date, out_path):
    try:
        df_value = stock.get_market_trading_value_by_date(
            start_date, end_date, ticker, detail=True
        )
        
        if df_value.empty:
            return f"{ticker}: 데이터 없음"
        
        df_value.reset_index(inplace=True)
        
        if INCREMENTAL_MODE and out_path.exists():
            df_old = pd.read_csv(out_path, encoding="utf-8-sig")
            df_value = pd.concat([df_old, df_value]).drop_duplicates(subset=["날짜"], keep="last")
        
        df_value.to_csv(out_path, index=False, encoding="utf-8-sig")
        return f"{ticker}: OK ({len(df_value)} rows)"
        
    except Exception as e:
        return f"{ticker}: 오류 ({e})"

def fetch_krx_flows():
    print("[fetch_krx_flows] 시작...")
    print(f"  INCREMENTAL_MODE={INCREMENTAL_MODE}, MAX_WORKERS={MAX_WORKERS}")
    
    master_path = Path("data/stocks/master/listings.parquet")
    if not master_path.exists():
        print("⚠️  마스터 파일 없음")
        return
    df_master = pd.read_parquet(master_path)
    
    out_dir = Path("data/stocks/raw/krx_flows")
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
            
            future = executor.submit(fetch_one_ticker, ticker, start_date, end_date, out_path)
            tasks.append((ticker, future))
        
        for i, (ticker, future) in enumerate(tasks):
            result = future.result()
            print(f"  [{i+1}/{len(tasks)}] {result}")
    
    print("[fetch_krx_flows] 완료")

if __name__ == "__main__":
    fetch_krx_flows()
