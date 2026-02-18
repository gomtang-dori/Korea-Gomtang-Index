#!/usr/bin/env python3
"""
raw KRX flows CSV → curated parquet
- PyKRX 컬럼명: 날짜, 기관합계, 기타법인, 개인, 외국인, 기타외국인 등
- 표준 컬럼: pension_net, fin_invest_net, inst_total_net, foreign_net
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
            
            # ✅ PyKRX 컬럼명 확인 후 매핑 (예시, 실제는 print(df_raw.columns) 확인)
            # 컬럼: 날짜, 기관합계, 기타법인, 개인, 외국인합계, 등
            # 순매수 = 매수 - 매도 (이미 계산된 컬럼 사용 또는 직접 계산)
            
            # 예시 매핑 (실제 컬럼명에 맞게 수정 필요)
            rename_map = {
                "날짜": "date",
                "기관합계": "inst_total_net",
                "외국인합계": "foreign_net",
                # "금융투자": "fin_invest_net",  # detail=True 시 존재
                # "연기금": "pension_net"
            }
            
            df_raw.rename(columns=rename_map, inplace=True)
            
            # 5일·20일 rolling sum
            for col in ["inst_total_net", "foreign_net"]:
                if col in df_raw.columns:
                    df_raw[f"{col}_5d"] = df_raw[col].rolling(5).sum()
                    df_raw[f"{col}_20d"] = df_raw[col].rolling(20).sum()
            
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
