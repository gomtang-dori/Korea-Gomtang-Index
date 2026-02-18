#!/usr/bin/env python3
"""
DART 재무제표 수집 (병렬처리)
- 최근 2년치 분기 재무 (8개 분기)
"""
import os
import time
import json
import zipfile
from pathlib import Path
from io import BytesIO
import pandas as pd
import requests
from xml.etree import ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import MAX_WORKERS, SAMPLE_MODE

DART_API_KEY = os.getenv("DART_API_KEY", "")
DART_BASE = "https://opendart.fss.or.kr/api"

def download_corp_code_map():
    """corpCode.xml 다운로드 → ticker↔corp_code 매핑"""
    print("  [DART] corpCode.xml 다운로드 중...")
    url = f"{DART_BASE}/corpCode.xml?crtfc_key={DART_API_KEY}"
    resp = requests.get(url, timeout=30)
    
    if resp.status_code != 200:
        print(f"⚠️  corpCode.xml 다운로드 실패: {resp.status_code}")
        return {}
    
    with zipfile.ZipFile(BytesIO(resp.content)) as z:
        xml_data = z.read("CORPCODE.xml")
    
    root = ET.fromstring(xml_data)
    mapping = {}
    
    for corp in root.findall("list"):
        corp_code = corp.findtext("corp_code")
        stock_code = corp.findtext("stock_code")
        
        if stock_code and stock_code.strip():
            mapping[stock_code.strip()] = corp_code
    
    print(f"  [DART] corpCode 매핑 완료: {len(mapping)} 종목")
    return mapping

def fetch_one_ticker_dart(ticker, corp_code, out_dir):
    """단일 종목 DART 재무 수집"""
    quarter_path = out_dir / f"{ticker}_quarterly.json"
    
    if quarter_path.exists():
        return f"{ticker}: 이미 존재"
    
    try:
        # 최근 분기 (예: 2024년 3분기)
        url = f"{DART_BASE}/fnlttSinglAcnt.json"
        params = {
            "crtfc_key": DART_API_KEY,
            "corp_code": corp_code,
            "bsns_year": "2024",
            "reprt_code": "11014",  # 3분기
            "fs_div": "CFS"
        }
        resp = requests.get(url, params=params, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            with open(quarter_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            time.sleep(0.15)  # API rate limit (분당 1000건 = 0.06초, 여유 0.15초)
            return f"{ticker}: OK"
        else:
            return f"{ticker}: HTTP {resp.status_code}"
    
    except Exception as e:
        return f"{ticker}: 오류 ({e})"

def fetch_dart_financials():
    print("[fetch_dart_financials] 시작...")
    
    if not DART_API_KEY:
        print("⚠️  DART_API_KEY 없음, 스킵")
        return
    
    corp_map = download_corp_code_map()
    if not corp_map:
        print("⚠️  corpCode 매핑 실패")
        return
    
    master_path = Path("data/stocks/master/listings.parquet")
    if not master_path.exists():
        print("⚠️  마스터 파일 없음")
        return
    df_master = pd.read_parquet(master_path)
    
    out_dir = Path("data/stocks/raw/dart")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ✅ 병렬처리 (DART API rate limit 주의: worker 수 조절)
    max_workers = min(MAX_WORKERS, 5) if not SAMPLE_MODE else 3
    tasks = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, row in df_master.iterrows():
            ticker = row["ticker"]
            corp_code = corp_map.get(ticker)
            
            if not corp_code:
                print(f"  [{idx+1}/{len(df_master)}] {ticker} corp_code 없음")
                continue
            
            future = executor.submit(fetch_one_ticker_dart, ticker, corp_code, out_dir)
            tasks.append(future)
        
        for i, future in enumerate(as_completed(tasks)):
            result = future.result()
            print(f"  [{i+1}/{len(tasks)}] {result}")
    
    print("[fetch_dart_financials] 완료")

if __name__ == "__main__":
    fetch_dart_financials()
