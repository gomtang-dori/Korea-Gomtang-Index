#!/usr/bin/env python3
"""
PyKRX로 전종목 투자자별 매수·매도·순매수 수집
- 수정: get_market_trading_value_by_date + get_market_trading_volume_by_date 사용
- 출력: data/stocks/raw/krx_flows/{ticker}.csv
"""
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from pykrx import stock

START_DATE = "20200101"

def fetch_krx_flows():
    print("[fetch_krx_flows] 시작...")
    
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
        
        if out_path.exists():
            print(f"  [{idx+1}/{len(df_master)}] {ticker} 이미 존재, 스킵")
            continue
        
        try:
            # ✅ 수정: 거래대금 기준 투자자별 매매
            df_value = stock.get_market_trading_value_by_date(
                START_DATE, end_date, ticker, detail=True
            )
            
            if df_value.empty:
                print(f"  [{idx+1}/{len(df_master)}] {ticker} 데이터 없음")
                continue
            
            # ✅ 거래량 기반도 추가 가능 (선택)
            # df_volume = stock.get_market_trading_volume_by_date(
            #     START_DATE, end_date, ticker, detail=True
            # )
            
            df_value.reset_index(inplace=True)
            
            # ✅ 컬럼명 표준화 (PyKRX 출력: 날짜, 기관합계, 기타법인, 개인, 외국인, 기타외국인 등)
            # 필요한 투자자만 추출: 연기금·금융투자·기관합계·외국인
            # 실제 컬럼명은 PyKRX 버전에 따라 다를 수 있음 (확인 필요)
            
            df_value.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  [{idx+1}/{len(df_master)}] {ticker} OK → {len(df_value)} rows")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(df_master)}] {ticker} 오류: {e}")
    
    print("[fetch_krx_flows] 완료")

if __name__ == "__main__":
    fetch_krx_flows()
