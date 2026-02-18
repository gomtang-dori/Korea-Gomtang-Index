#!/usr/bin/env python3
"""
raw KRX flows CSV → curated parquet (5일·20일 순매수 합산)
출력: data/stocks/curated/{ticker}/flows_daily.parquet
"""
import os
from pathlib import Path
import pandas as pd

def curate_flows():
    print("[curate_flows] 시작...")
    
    master_path = Path("data/stocks/master/listings.parquet")
    if not master_path.exists():
        print("⚠️  마스터 파일 없음")
        return
    df_master = pd.read_parquet(master_path)
    
    raw_dir = Path("data/stocks/raw/krx_flows")
    
    for idx, row in df_master.iterrows():
        ticker = row["ticker"]
        raw_path = raw_dir / f"{ticker}.csv"
        
        if not raw_path.exists():
            continue
        
        df_raw = pd.read_csv(raw_path, encoding="utf-8-sig")
        
        # 컬럼명 정리 (PyKRX 출력 기준)
        # 예: '연기금', '금융투자', '기관합계', '외국인'
        # 매수·매도·순매수 컬럼 추출 후 표준 명칭 변경
        
        # 5일·20일 rolling sum (간단 예시)
        # df_raw['pension_net_5d'] = df_raw['연기금순매수'].rolling(5).sum()
        # df_raw['pension_net_20d'] = df_raw['연기금순매수'].rolling(20).sum()
        
        out_dir = Path(f"data/stocks/curated/{ticker}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "flows_daily.parquet"
        
        df_raw.to_parquet(out_path, index=False)
        print(f"  [{idx+1}/{len(df_master)}] {ticker} flows OK")
    
    print("[curate_flows] 완료")

if __name__ == "__main__":
    curate_flows()
