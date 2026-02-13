from __future__ import annotations

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from pykrx import stock

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _try_get_members(asof: str) -> list[str]:
    """
    1차: get_index_portfolio_deposit_file (추천)
    2차: get_index_ticker_list (fallback)
    """
    tickers = []
    try:
        # 코스피 200 지수 코드 '1028' 사용
        tickers = stock.get_index_portfolio_deposit_file(asof, "1028")
        tickers = list(tickers) if tickers is not None else []
    except Exception:
        tickers = []

    if tickers:
        return [str(t) for t in tickers]

    # fallback
    try:
        tickers2 = stock.get_index_ticker_list(asof, "1028")
        tickers2 = list(tickers2) if tickers2 is not None else []
        if tickers2:
            return [str(t) for t in tickers2]
    except Exception:
        pass

    return []

def pick_recent_business_day_kr(max_lookback_days: int = 30) -> tuple[str, list[str]]:
    """
    오늘부터 과거로 거슬러 올라가며 종목 리스트가 150개 이상인 날짜를 찾습니다.
    """
    # UTC 기준이나 한국 시간 기준으로 오늘 날짜 설정
    d = datetime.utcnow().date()
    for _ in range(max_lookback_days):
        asof = d.strftime("%Y%m%d")
        tickers = _try_get_members(asof)

        # 150개 이상의 종목이 발견되면 유효한 영업일로 판단
        if len(tickers) >= 150:
            return asof, tickers

        d = d - timedelta(days=1)

    return "", []

def main():
    # 1. 출력 경로 설정
    out_path = Path("data/k200_members.csv")
    ensure_dir(out_path.parent)

    # 2. 최근 유효 영업일 및 종목 리스트 획득
    print("[cache_k200_members] Searching for recent business day...")
    asof, tickers = pick_recent_business_day_kr(max_lookback_days=30)
    
    # 만약 위 함수에서 실패했을 경우를 대비한 최종 시도
    if not tickers:
        test_date = datetime.utcnow().strftime("%Y%m%d")
        print(f"[cache_k200_members] Fallback attempt with date: {test_date}")
        tickers = stock.get_index_portfolio_deposit_file(test_date, "KOSPI 200")
        asof = test_date

    if not tickers or len(tickers) < 150:
        raise RuntimeError("Failed to fetch KOSPI200 members. Check pykrx availability or network.")

    # 3. 종목명 매핑
    print(f"[cache_k200_members] Fetching names for {len(tickers)} tickers...")
    names = {}
    try:
        # 하나씩 가져오되 실패 시 빈 문자열 처리 (속도 및 안정성)
        for t in tickers:
            try:
                names[t] = stock.get_market_ticker_name(t)
            except:
                names[t] = ""
    except Exception as e:
        print(f"[Warning] Error during naming: {e}")
        names = {t: "" for t in tickers}

    # 4. 데이터프레임 생성 및 저장
    df = pd.DataFrame({
        "asof_date": asof,
        "isu_cd": tickers,
        "isu_nm": [names.get(t, "") for t in tickers],
    }).sort_values("isu_cd")

    # 한글 깨짐 방지를 위해 utf-8-sig 사용
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[cache_k200_members] OK! asof={asof}, rows={len(df)} -> {out_path}")

if __name__ == "__main__":
    main()
