from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from pykrx import stock

def ensure_dir(p: Path):
    """폴더가 없으면 생성하는 함수"""
    p.mkdir(parents=True, exist_ok=True)

def _try_get_members(asof: str) -> list[str]:
    """
    특정 날짜의 코스피 200 구성 종목 리스트를 가져오기 시도합니다.
    """
    tickers = []
    try:
        # 1차 시도: 지수 구성 종목 함수 (지수코드 1028 = KOSPI 200)
        tickers = stock.get_index_portfolio_deposit_file(asof, "1028")
        if isinstance(tickers, list) and len(tickers) > 0:
            return [str(t) for t in tickers]
    except Exception:
        pass

    try:
        # 2차 시도: 다른 함수로 재시도
        tickers = stock.get_index_ticker_list(asof, "KOSPI 200")
        if isinstance(tickers, list) and len(tickers) > 0:
            return [str(t) for t in tickers]
    except Exception:
        pass

    return []

def pick_recent_business_day_kr(max_lookback_days: int = 30) -> tuple[str, list[str]]:
    """
    오늘부터 과거로 30일간 거슬러 올라가며, 데이터가 있는 가장 최근 영업일을 찾습니다.
    """
    d = datetime.utcnow().date()
    for _ in range(max_lookback_days):
        asof = d.strftime("%Y%m%d")
        tickers = _try_get_members(asof)

        # 종목이 150개 이상 발견되면 정상적인 영업일로 판단
        if len(tickers) >= 150:
            return asof, tickers

        d = d - timedelta(days=1)

    return "", []

def main():
    # 1. 파일 저장 경로 및 폴더 준비
    out_path = Path("data/k200_members.csv")
    ensure_dir(out_path.parent)

    print("[cache_k200_members] Searching for recent business day...")
    
    # 2. 최근 영업일 데이터 수집
    asof, tickers = pick_recent_business_day_kr(max_lookback_days=30)
    
    # 데이터가 비어있는지 확인하는 안전한 방법 (ValueError 방지)
    is_invalid = False
    if not tickers: # 리스트가 비어있음
        is_invalid = True
    elif isinstance(tickers, pd.DataFrame) and tickers.empty: # 데이터프레임이 비어있음
        is_invalid = True
    
    if is_invalid:
        # 마지막 수단: 확실한 평일 날짜(어제 등)로 다시 시도
        test_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y%m%d")
        print(f"[cache_k200_members] Fallback attempt with date: {test_date}")
        tickers = stock.get_index_portfolio_deposit_file(test_date, "1028")
        asof = test_date

    # 3. 최종 데이터 유효성 검사
    # 리스트인지 데이터프레임인지 상관없이 개수를 체크
    final_count = len(tickers) if tickers is not None else 0
    if final_count < 150:
        raise RuntimeError("Failed to fetch KOSPI200 members. Check pykrx availability or network.")

    # 4. 종목명(이름) 가져오기
    print(f"[cache_k200_members] Fetching names for {final_count} tickers...")
    names = {}
    for t in tickers:
        try:
            # 하나씩 이름을 가져오되, 에러 나면 공백 처리
            names[t] = stock.get_market_ticker_name(t)
        except:
            names[t] = ""

    # 5. 결과 정리 및 CSV 저장
    df = pd.DataFrame({
        "asof_date": asof,
        "isu_cd": tickers,
        "isu_nm": [names.get(t, "") for t in tickers],
    }).sort_values("isu_cd")

    # 한글 깨짐 방지를 위해 utf-8-sig 인코딩 사용
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[cache_k200_members] Success! asof={asof}, rows={len(df)} -> {out_path}")

if __name__ == "__main__":
    main()
