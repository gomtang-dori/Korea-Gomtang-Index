#!/usr/bin/env python3
"""
raw KRX flows CSV → curated parquet
- 컬럼명 표준화: date, inst_total_net, foreign_net 등
- 5일·20일 rolling sum 추가
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
        
        try:
            df_raw = pd.read_csv(raw_path, encoding="utf-8-sig")
            
            # ✅ 1. 날짜 컬럼 표준화
            if "날짜" in df_raw.columns:
                df_raw.rename(columns={"날짜": "date"}, inplace=True)
            
            # ✅ 2. 투자자별 컬럼 표준화 (PyKRX detail=True 출력 기준)
            # 실제 컬럼명 예시: '기관합계', '외국인합계', '금융투자', '연기금' 등
            rename_map = {}
            
            for col in df_raw.columns:
                if "기관" in col and "합계" in col:
                    rename_map[col] = "inst_total_net"
                elif "외국인" in col and "합계" in col:
                    rename_map[col] = "foreign_net"
                elif "금융투자" in col:
                    rename_map[col] = "fin_invest_net"
                elif "연기금" in col:
                    rename_map[col] = "pension_net"
            
            df_raw.rename(columns=rename_map, inplace=True)
            
            # ✅ 3. Rolling sum (5일, 20일)
            for col in ["inst_total_net", "foreign_net", "fin_invest_net", "pension_net"]:
                if col in df_raw.columns:
                    df_raw[f"{col}_5d"] = df_raw[col].rolling(5, min_periods=1).sum()
                    df_raw[f"{col}_20d"] = df_raw[col].rolling(20, min_periods=1).sum()
            
            # ✅ 4. 날짜를 datetime으로 변환
            df_raw["date"] = pd.to_datetime(df_raw["date"])
            
            # ✅ 5. 저장
            out_dir = Path(f"data/stocks/curated/{ticker}")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "flows_daily.parquet"
            
            df_raw.to_parquet(out_path, index=False)
            print(f"  [{idx+1}/{len(df_master)}] {ticker} flows OK")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(df_master)}] {ticker} 오류: {e}")
    
    print("[curate_flows] 완료")

if __name__ == "__main__":
    curate_flows()
