#!/usr/bin/env python3
"""
Raw 데이터를 CSV로 통합 추출
출력: docs/stocks/raw_*.csv
"""
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

PROJECT_ROOT = Path.cwd()
print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}")

def export_prices_csv():
    """전종목 가격 데이터 통합 CSV"""
    print("[1/3] prices_raw.csv 생성 중...")
    
    master_path = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    if not master_path.exists():
        print("  ⚠️  마스터 없음")
        return
    
    df_master = pd.read_parquet(master_path)
    prices_dir = PROJECT_ROOT / "data/stocks/raw/prices"
    
    if not prices_dir.exists():
        print("  ⚠️  가격 데이터 없음")
        return
    
    all_prices = []
    
    for idx, row in df_master.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        price_path = prices_dir / f"{ticker}.csv"
        
        if not price_path.exists():
            continue
        
        try:
            df_price = pd.read_csv(price_path, encoding="utf-8-sig")
            df_price["ticker"] = ticker
            df_price["name"] = name
            all_prices.append(df_price[["date", "ticker", "name", "open", "high", "low", "close", "volume"]])
            
            if len(all_prices) % 500 == 0:
                print(f"  진행: {len(all_prices)}/{len(df_master)}")
        except Exception as e:
            print(f"  ⚠️ {ticker} 오류: {e}")
    
    if all_prices:
        df_all = pd.concat(all_prices, ignore_index=True)
        df_all.sort_values(["date", "ticker"], inplace=True)
        
        out_path = PROJECT_ROOT / "docs/stocks/prices_raw.csv"
        df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
        
        rows = len(df_all)
        size_mb = out_path.stat().st_size / (1024*1024)
        print(f"  ✅ OK → {out_path}")
        print(f"     {rows:,} rows, {size_mb:.1f} MB")
    else:
        print("  ⚠️  데이터 없음")

def export_flows_csv():
    """전종목 투자자 매매 통합 CSV"""
    print("[2/3] flows_raw.csv 생성 중...")
    
    master_path = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    if not master_path.exists():
        return
    
    df_master = pd.read_parquet(master_path)
    flows_dir = PROJECT_ROOT / "data/stocks/raw/krx_flows"
    
    if not flows_dir.exists():
        print("  ⚠️  수급 데이터 없음")
        return
    
    all_flows = []
    
    for idx, row in df_master.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        flow_path = flows_dir / f"{ticker}.csv"
        
        if not flow_path.exists():
            continue
        
        try:
            df_flow = pd.read_csv(flow_path, encoding="utf-8-sig")
            df_flow["ticker"] = ticker
            df_flow["name"] = name
            
            # 날짜 컬럼 표준화
            if "날짜" in df_flow.columns:
                df_flow.rename(columns={"날짜": "date"}, inplace=True)
            
            # 주요 컬럼만 추출 (파일 크기 축소)
            keep_cols = ["date", "ticker", "name"]
            for col in df_flow.columns:
                if "기관" in col and "합계" in col:
                    keep_cols.append(col)
                if "외국인" in col and "합계" in col:
                    keep_cols.append(col)
                if "금융투자" in col:
                    keep_cols.append(col)
            
            df_filtered = df_flow[[c for c in keep_cols if c in df_flow.columns]].copy()
            all_flows.append(df_filtered)
            
            if len(all_flows) % 500 == 0:
                print(f"  진행: {len(all_flows)}/{len(df_master)}")
        except Exception as e:
            print(f"  ⚠️ {ticker} 오류: {e}")
    
    if all_flows:
        df_all = pd.concat(all_flows, ignore_index=True)
        df_all.sort_values(["date", "ticker"], inplace=True)
        
        out_path = PROJECT_ROOT / "docs/stocks/flows_raw.csv"
        df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
        
        rows = len(df_all)
        size_mb = out_path.stat().st_size / (1024*1024)
        print(f"  ✅ OK → {out_path}")
        print(f"     {rows:,} rows, {size_mb:.1f} MB")
    else:
        print("  ⚠️  데이터 없음")

def export_financials_csv():
    """전종목 재무제표 통합 CSV"""
    print("[3/3] financials_raw.csv 생성 중...")
    
    master_path = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    if not master_path.exists():
        return
    
    df_master = pd.read_parquet(master_path)
    
    all_fin = []
    
    for idx, row in df_master.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        
        fin_path = PROJECT_ROOT / f"data/stocks/curated/{ticker}/financials_quarterly.parquet"
        
        if not fin_path.exists():
            continue
        
        try:
            df_fin = pd.read_parquet(fin_path)
            df_fin["ticker"] = ticker
            df_fin["name"] = name
            all_fin.append(df_fin)
        except Exception as e:
            print(f"  ⚠️ {ticker} 오류: {e}")
    
    if all_fin:
        df_all = pd.concat(all_fin, ignore_index=True)
        
        out_path = PROJECT_ROOT / "docs/stocks/financials_raw.csv"
        df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
        
        rows = len(df_all)
        size_mb = out_path.stat().st_size / (1024*1024)
        print(f"  ✅ OK → {out_path}")
        print(f"     {rows:,} rows, {size_mb:.1f} MB")
    else:
        print("  ⚠️  데이터 없음")

def main():
    print("[export_raw_to_csv] 시작...\n")
    
    (PROJECT_ROOT / "docs/stocks").mkdir(parents=True, exist_ok=True)
    
    export_prices_csv()
    export_flows_csv()
    export_financials_csv()
    
    print("\n[export_raw_to_csv] ✅ 완료")
    print("\n생성된 파일:")
    for csv_file in ["prices_raw.csv", "flows_raw.csv", "financials_raw.csv"]:
        path = PROJECT_ROOT / f"docs/stocks/{csv_file}"
        if path.exists():
            size_mb = path.stat().st_size / (1024*1024)
            print(f"  {csv_file}: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
