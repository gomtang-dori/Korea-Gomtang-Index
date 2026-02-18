#!/usr/bin/env python3
"""
전종목 마스터 목록 생성 (KOSPI + KOSDAQ)
- PyKRX get_market_ticker_list() 사용
- 출력: data/stocks/master/listings.parquet
"""
import os
from pathlib import Path
import pandas as pd
from pykrx import stock

def fetch_listings():
    print("[fetch_listings] 시작...")
    
    # 출력 경로
    out_dir = Path("data/stocks/master")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "listings.parquet"
    
    # KOSPI + KOSDAQ 전종목
    kospi = stock.get_market_ticker_list(market="KOSPI")
    kosdaq = stock.get_market_ticker_list(market="KOSDAQ")
    
    rows = []
    for ticker in kospi:
        name = stock.get_market_ticker_name(ticker)
        rows.append({"ticker": ticker, "name": name, "market": "KOSPI"})
    for ticker in kosdaq:
        name = stock.get_market_ticker_name(ticker)
        rows.append({"ticker": ticker, "name": name, "market": "KOSDAQ"})
    
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"[fetch_listings] OK → {out_path} ({len(df)} 종목)")

if __name__ == "__main__":
    fetch_listings()
