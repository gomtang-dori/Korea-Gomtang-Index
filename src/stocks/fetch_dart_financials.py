#!/usr/bin/env python3
"""
OpenDART API로 전종목 분기·연간 재무제표 수집
- 환경변수: DART_API_KEY
- 출력: data/stocks/raw/dart/{ticker}_{quarter/annual}.json
"""
import os
import time
import json
from pathlib import Path
import pandas as pd
import requests

DART_API_KEY = os.getenv("DART_API_KEY", "")
DART_BASE = "https://opendart.fss.or.kr/api"

def fetch_dart_financials():
    print("[fetch_dart_financials] 시작...")
    if not DART_API_KEY:
        print("⚠️  DART_API_KEY 없음, 스킵")
        return
    
    # 마스터 로드
    master_path = Path("data/stocks/master/listings.parquet")
    if not master_path.exists():
        print("⚠️  마스터 파일 없음, fetch_listings.py 먼저 실행")
        return
    df_master = pd.read_parquet(master_path)
    
    out_dir = Path("data/stocks/raw/dart")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # DART는 corp_code 필요 → 종목 코드 매핑 조회 (간단 예시: corpCode.xml 파싱 생략, 직접 API 호출)
    # 실제로는 https://opendart.fss.or.kr/api/corpCode.xml 다운로드 후 매핑 권장
    # 여기서는 간단히 ticker → corp_code 가정 (실무에서는 corpCode.xml 파싱 필수)
    
    for idx, row in df_master.iterrows():
        ticker = row["ticker"]
        # 실제 corp_code 매핑 로직 필요 (생략)
        # 예시: corp_code = ticker_to_corp_code_map.get(ticker)
        # 임시로 스킵
        print(f"  [{idx+1}/{len(df_master)}] {ticker} (corp_code 매핑 필요, 스킵)")
        # time.sleep(0.1)  # API rate limit
    
    print("[fetch_dart_financials] 완료 (실제 구현 시 corpCode.xml 파싱 필요)")

if __name__ == "__main__":
    fetch_dart_financials()
