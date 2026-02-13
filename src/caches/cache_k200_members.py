from __future__ import annotations
import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from pykrx import stock

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    out_path = Path("data/k200_members.csv")
    ensure_dir(out_path.parent)

    # 1. 날짜 설정 (최근 확실한 평일인 금요일)
    # 현재 실행 시간이 토요일이므로 금요일 데이터를 가져옵니다.
    target_date = "20260213"
    print(f"[cache_k200_members] Fetching K200 members via pykrx for {target_date}...")

    tickers = []
    try:
        # 방법 A: 지수 구성 종목 (가장 정확함)
        # 'KOSPI 200' 명칭을 직접 사용하여 내부 'market' 옵션 충돌 방지
        tickers = stock.get_index_portfolio_deposit_file(target_date, "KOSPI 200")
    except Exception as e:
        print(f"[Log] Method A failed: {e}")

    if tickers is None or len(tickers) < 150:
        try:
            # 방법 B: 지수 티커 리스트
            tickers = stock.get_index_ticker_list(target_date, market="KOSPI")
            # KOSPI 전체 중 상위 200개를 가져옴 (비상용)
            if len(tickers) > 200: tickers = tickers[:200]
        except Exception as e:
            print(f"[Log] Method B failed: {e}")

    # 2. 최종 확인 및 저장
    if tickers is None or len(tickers) < 50:
        # 만약 여기까지 실패했다면, 날짜를 하루 더 뒤로 미뤄서 시도 (목요일)
        target_date = "20260212"
        print(f"[cache_k200_members] Retrying with {target_date}...")
        tickers = stock.get_index_portfolio_deposit_file(target_date, "KOSPI 200")

    if tickers is None or len(tickers) < 50:
        raise RuntimeError("모든 데이터 수집 시도가 실패했습니다. KRX 서버 점검 중일 수 있습니다.")

    # 3. 종목 정보 정리
    print(f"[cache_k200_members] Found {len(tickers)} tickers. Mapping names...")
    
    # 리스트 형태로 변환하여 데이터프레임 생성
    ticker_list = list(tickers)
    names = []
    for t in ticker_list:
        try:
            names.append(stock.get_market_ticker_name(t))
        except:
            names.append("")

    df = pd.DataFrame({
        "asof_date": target_date,
        "isu_cd": ticker_list,
        "isu_nm": names
    }).sort_values("isu_cd")

    # 4. 저장
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[cache_k200_members] Success! {len(df)} rows saved to {out_path}")

if __name__ == "__main__":
    main()
