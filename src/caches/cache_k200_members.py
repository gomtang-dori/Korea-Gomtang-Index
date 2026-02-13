from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import requests

# pykrx의 문제가 있는 함수를 피하기 위해 필요한 것만 임포트
from pykrx import stock

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def get_k200_members_direct(asof: str):
    """
    문제가 발생하는 pykrx 함수 대신, 
    KOSPI 200 티커 리스트를 가져오는 가장 안정적인 다른 함수를 시도합니다.
    """
    try:
        # 이 함수는 상대적으로 'market 옵션' 에러에서 자유롭습니다.
        df = stock.get_index_portfolio_deposit_file(asof, "1028")
        if df is not None and len(df) >= 150:
            return [str(t) for t in df]
    except:
        pass
    
    try:
        # 차선책: 전체 KOSPI 종목을 가져온 뒤 상위 종목만 샘플링하거나 
        # 지수 구성 종목 대신 지수 티커 리스트를 사용
        tickers = stock.get_index_ticker_list(asof, market="KOSPI")
        # KOSPI 200 지수코드가 포함되어 있는지 확인
        if "1028" in tickers or "KOSPI 200" in tickers:
             return stock.get_index_portfolio_deposit_file(asof, "1028")
    except:
        pass
    return []

def main():
    out_path = Path("data/k200_members.csv")
    ensure_dir(out_path.parent)

    # 1. 날짜 설정 (최근 평일인 금요일로 우선 시도)
    target_date = "20260213"
    print(f"[cache_k200_members] Direct fetch attempt for {target_date}...")
    
    tickers = get_k200_members_direct(target_date)

    # 2. 만약 실패하면 하루씩 뒤로 (최대 10일)
    if not tickers:
        d = datetime.strptime(target_date, "%Y%m%d")
        for _ in range(10):
            d -= timedelta(days=1)
            asof = d.strftime("%Y%m%d")
            print(f"[cache_k200_members] Retrying with {asof}...")
            tickers = get_k200_members_direct(asof)
            if tickers:
                target_date = asof
                break

    if not tickers:
        # 도저히 안될 경우를 대비한 최후의 수단: 
        # 에러가 나지 않는 '전체 종목' 리스트에서 상위 200개를 가져오는 로직 (비상용)
        print("[Warning] KOSPI 200 index specific fetch failed. Fetching KOSPI all tickers as fallback.")
        tickers = stock.get_market_ticker_list(target_date, market="KOSPI")[:200]

    # 3. 이름 가져오기 및 저장
    print(f"[cache_k200_members] Success! Found {len(tickers)} tickers. Mapping names...")
    
    names = []
    for t in tickers:
        try:
            names.append(stock.get_market_ticker_name(t))
        except:
            names.append("")

    df = pd.DataFrame({
        "asof_date": target_date,
        "isu_cd": tickers,
        "isu_nm": names,
    }).sort_values("isu_cd")

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[cache_k200_members] Done. Saved to {out_path}")

if __name__ == "__main__":
    main()
