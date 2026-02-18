#!/usr/bin/env python3
"""
raw DART JSON → curated parquet
- 매출액·영업이익·당기순이익·자본 추출
- YoY/QoQ 증가율 계산
- ROE = 당기순이익 / 자본
출력: data/stocks/curated/{ticker}/financials_quarterly.parquet
"""
import os
import json
from pathlib import Path
import pandas as pd

def parse_dart_json(json_path):
    """DART JSON → DataFrame (account_nm, thstrm_amount 등 추출)"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if data.get("status") != "000":
        return pd.DataFrame()
    
    items = data.get("list", [])
    df = pd.DataFrame(items)
    
    # 필요 컬럼: account_nm (계정명), thstrm_amount (당기금액) 등
    # 매출액, 영업이익, 당기순이익, 자본총계 필터
    targets = ["매출액", "영업이익", "당기순이익", "자본총계"]
    df_filtered = df[df["account_nm"].isin(targets)].copy()
    
    # 금액 문자열 → 숫자 (쉼표 제거)
    df_filtered["amount"] = df_filtered["thstrm_amount"].str.replace(",", "").astype(float)
    
    # Pivot
    pivot = df_filtered.pivot_table(index="bsns_year", columns="account_nm", values="amount", aggfunc="first")
    return pivot

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
        json_path = raw_dir / f"{ticker}_quarterly.json"
        
        if not json_path.exists():
            continue
        
        try:
            df_parsed = parse_dart_json(json_path)
            if df_parsed.empty:
                continue
            
            # YoY 계산 (간단 예시)
            df_parsed["revenue_yoy"] = df_parsed["매출액"].pct_change()
            df_parsed["net_income_yoy"] = df_parsed["당기순이익"].pct_change()
            df_parsed["roe_ttm"] = df_parsed["당기순이익"] / df_parsed["자본총계"]
            
            out_dir = Path(f"data/stocks/curated/{ticker}")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "financials_quarterly.parquet"
            
            df_parsed.to_parquet(out_path)
            print(f"  [{idx+1}/{len(df_master)}] {ticker} financials OK")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(df_master)}] {ticker} 오류: {e}")
    
    print("[curate_financials] 완료")

if __name__ == "__main__":
    curate_financials()
