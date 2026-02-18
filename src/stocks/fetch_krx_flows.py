#!/usr/bin/env python3
"""
PyKRX 투자자별 매매 수집 (병렬처리 + 증분 업데이트)
- INCREMENTAL_MODE=true → 최근 N일만
- MAX_WORKERS로 병렬처리
"""
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from pykrx import stock
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import INCREMENTAL_MODE, INCREMENTAL_DAYS, MAX_WORKERS, START_DATE

def fetch_one_ticker(ticker, start_date, end_date, out_path):
    """단일 종목 수집"""
    try:
        df_value = stock.get_market_trading_value_by_date(
            start_date, end_date, ticker, detail=True
        )
        
        if df_value.empty:
            return f"{ticker}: 데이터 없음"
        
        df_value.reset_index(inplace=True)
        
        # ✅ 증분 모드: 기존 데이터와 병합
        if INCREMENTAL_MODE and out_path.exists():
            df_old = pd.read_csv(out_path, encoding="utf-8-sig")
            df_value = pd.concat([df_old, df_value]).drop_duplicates(subset=["날짜"], keep="last")
        
        df_value.to_csv(out_path, index=False, encoding="utf-8-sig")
        return f"{ticker}: OK ({len(df_value)} rows)"
        
    except Exception as e:
        return f"{ticker}: 오류 ({e})"

def fetch_krx_flows():
    print("[fetch_krx_flows] 시작...")
    
    master_path = Path("data/stocks/master/listings.parquet")
    if not master_path.exists():
        print("⚠️  마스터 파일 없음")
        return
    df_master = pd.read_parquet(master_path)
    
    out_dir = Path("data/stocks/raw/krx_flows")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ✅ 날짜 범위
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
            
            # 증분 모드가 아니고 파일 존재 시 스킵
            if not INCREMENTAL_MODE and out_path.exists():
                print(f"  [{idx+1}/{len(df_master)}] {ticker} 이미 존재, 스킵")
                continue
            
            future = executor.submit(fetch_one_ticker, ticker, start_date, end_date, out_path)
            tasks.append(future)
        
        # 결과 수집
        for i, future in enumerate(as_completed(tasks)):
            result = future.result()
            print(f"  [{i+1}/{len(tasks)}] {result}")
    
    print("[fetch_krx_flows] 완료")

if __name__ == "__main__":
    fetch_krx_flows()
