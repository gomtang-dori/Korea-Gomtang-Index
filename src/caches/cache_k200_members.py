from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from pykrx import stock

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _try_get_members(asof: str) -> list[str]:
    """
    여러 가지 함수와 파라미터 순서를 바꿔가며 종목 리스트를 가져옵니다.
    """
    # 방법 1: 가장 표준적인 방식 (종목 구성 내역)
    try:
        # 순서 1: (날짜, 지수코드)
        tickers = stock.get_index_portfolio_deposit_file(asof, "1028")
        if tickers is not None and len(tickers) >= 150:
            return [str(t) for t in tickers]
        
        # 순서 2: (지수코드, 날짜) - 버전 차이 대비
        tickers = stock.get_index_portfolio_deposit_file("1028", asof)
        if tickers is not None and len(tickers) >= 150:
            return [str(t) for t in tickers]
    except:
        pass

    # 방법 2: 티커 리스트 함수 사용
    try:
        tickers = stock.get_index_ticker_list(asof, "KOSPI 200")
        if tickers is not None and len(tickers) >= 150:
            return [str(t) for t in tickers]
    except:
        pass

    return []

def pick_recent_business_day_kr(max_lookback_days: int = 30) -> tuple[str, list[str]]:
    # 한국 시간(UTC+9)을 고려하여 오늘 날짜 설정
    d = (datetime.utcnow() + timedelta(hours=9)).date()
    
    for _ in range(max_lookback_days):
        asof = d.strftime("%Y%m%d")
        print(f"[cache_k200_members] Trying {asof}...")
        tickers = _try_get_members(asof)

        if len(tickers) >= 150:
            return asof, tickers

        d = d - timedelta(days=1)

    return "", []

def main():
    out_path = Path("data/k200_members.csv")
    ensure_dir(out_path.parent)

    print("[cache_k200_members] Searching for recent business day...")
    asof, tickers = pick_recent_business_day_kr(max_lookback_days=30)
    
    # 최종 결과 확인 (tickers가 데이터프레임일 경우를 대비해 len() 사용)
    valid_count = 0
    if tickers is not None:
        try:
            valid_count = len(tickers)
        except:
            valid_count = 0

    if valid_count < 150:
        # 수동 강제 날짜 설정 (최근 확실한 평일인 2026-02-13 금요일)
        fallback_date = "20260213"
        print(f"[cache_k200_members] All dynamic attempts failed. Trying hardcoded date: {fallback_date}")
        tickers = _try_get_members(fallback_date)
        asof = fallback_date
        
        if not tickers or len(tickers) < 150:
            raise RuntimeError("Failed to fetch KOSPI200 members. Check pykrx version or KRX server status.")

    print(f"[cache_k200_members] Found {len(tickers)} members for {asof}. Fetching names...")

    # 종목명 가져오기
    names_list = []
    for t in tickers:
        try:
            name = stock.get_market_ticker_name(t)
            names_list.append(name)
        except:
            names_list.append("")

    # 데이터프레임 생성 및 저장
    df = pd.DataFrame({
        "asof_date": asof,
        "isu_cd": tickers,
        "isu_nm": names_list,
    }).sort_values("isu_cd")

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[cache_k200_members] Success! -> {out_path}")

if __name__ == "__main__":
    main()
