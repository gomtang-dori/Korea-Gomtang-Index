#!/usr/bin/env python3
"""
raw DART JSON → curated parquet
- 계정명 유연 매칭 (당기순이익 vs 당기순손익 등)
"""
import os
import json
from pathlib import Path
import pandas as pd

def parse_dart_json(json_path):
    """DART JSON → DataFrame"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if data.get("status") != "000":
            return pd.DataFrame()
        
        items = data.get("list", [])
        if not items:
            return pd.DataFrame()
        
        df = pd.DataFrame(items)
        
        # ✅ 계정명 유연 매칭 (당기순이익, 당기순손익, 분기순이익 등)
        revenue_keywords = ["매출액", "영업수익"]
        op_income_keywords = ["영업이익", "영업손익"]
        net_income_keywords = ["당기순이익", "당기순손익", "분기순이익"]
        equity_keywords = ["자본총계", "자본"]
        
        def find_account(keywords):
            for kw in keywords:
                found = df[df["account_nm"].str.contains(kw, na=False)]
                if not found.empty:
                    return found.iloc[0]["account_nm"]
            return None
        
        revenue_name = find_account(revenue_keywords)
        op_income_name = find_account(op_income_keywords)
        net_income_name = find_account(net_income_keywords)
        equity_name = find_account(equity_keywords)
        
        if not all([revenue_name, op_income_name, net_income_name, equity_name]):
            return pd.DataFrame()
        
        # 필터링
        df_filtered = df[df["account_nm"].isin([revenue_name, op_income_name, net_income_name, equity_name])].copy()
        
        # 금액 변환
        df_filtered["amount"] = df_filtered["thstrm_amount"].str.replace(",", "").astype(float)
        
        # Pivot
        pivot = df_filtered.pivot_table(
            index=["bsns_year", "reprt_code"], 
            columns="account_nm", 
            values="amount", 
            aggfunc="first"
        ).reset_index()
        
        # 표준 컬럼명
        rename_map = {
            revenue_name: "revenue",
            op_income_name: "op_income",
            net_income_name: "net_income",
            equity_name: "equity"
        }
        pivot.rename(columns=rename_map, inplace=True)
        
        return pivot
        
    except Exception as e:
        print(f"    파싱 오류: {e}")
        return pd.DataFrame()

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
                print(f"  [{idx+1}/{len(df_master)}] {ticker} 파싱 실패 (데이터 없음)")
                continue
            
            # YoY 계산 (간단: 작년 같은 분기 대비)
            df_parsed = df_parsed.sort_values(["bsns_year", "reprt_code"])
            df_parsed["revenue_yoy"] = df_parsed["revenue"].pct_change(periods=4)
            df_parsed["net_income_yoy"] = df_parsed["net_income"].pct_change(periods=4)
            
            # ROE = 당기순이익 / 자본
            df_parsed["roe"] = df_parsed["net_income"] / df_parsed["equity"]
            
            out_dir = Path(f"data/stocks/curated/{ticker}")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "financials_quarterly.parquet"
            
            df_parsed.to_parquet(out_path, index=False)
            print(f"  [{idx+1}/{len(df_master)}] {ticker} financials OK ({len(df_parsed)} rows)")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(df_master)}] {ticker} 오류: {e}")
    
    print("[curate_financials] 완료")

if __name__ == "__main__":
    curate_financials()
