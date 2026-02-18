#!/usr/bin/env python3
"""
raw DART JSON → curated parquet (YoY/QoQ/ROE/EPS 계산)
출력: data/stocks/curated/{ticker}/financials_quarterly.parquet, financials_yearly.parquet
"""
import os
from pathlib import Path
import pandas as pd

def curate_financials():
    print("[curate_financials] 시작...")
    
    master_path = Path("data/stocks/master/listings.parquet")
    if not master_path.exists():
        print("⚠️  마스터 파일 없음")
        return
    df_master = pd.read_parquet(master_path)
    
    raw_dir = Path("data/stocks/raw/dart")
    
    for idx, row in df_master.iterrows():
        ticker = row["ticker"]
        # raw JSON 파일 찾기 (생략, DART 구현 후 작성)
        # 예: raw_path = raw_dir / f"{ticker}_quarter.json"
        # 여기서는 스킵
        pass
    
    print("[curate_financials] 완료 (DART 구현 후 작성 필요)")

if __name__ == "__main__":
    curate_financials()
