from __future__ import annotations
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from pykrx import stock

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    out_path = Path("data/k200_members.csv")
    ensure_dir(out_path.parent)

    # 1. 날짜 설정: 확실히 데이터가 존재하는 가장 최근 평일(금요일)
    # 한국 시간 기준 금요일 데이터를 가져오기 위해 날짜를 20260213으로 고정하거나 계산
    today = datetime.now()
    # 안전하게 금요일(2026-02-13) 혹은 그 이전 영업일을 타겟팅
    target_date = "20260213" 
    
    print(f"[cache_k200_members] Target Date: {target_date}")
    
    tickers = []
    # [시도 1] 표준 인덱스 구성 종목 함수
    try:
        tickers = stock.get_index_portfolio_deposit_file(target_date, "1028")
        tickers = list(tickers) if tickers is not None else []
    except:
        tickers = []

    # [시도 2] 실패 시, 대체 인덱스 함수 사용
    if len(tickers) < 150:
        print("[cache_k200_members] Attempt 2: get_index_ticker_list")
        try:
            tickers = stock.get_index_ticker_list(target_date, market="KOSPI")
            # KOSPI 전체 중 상위 200개를 가져오거나 필터링 (비상용)
            if len(tickers) > 200: tickers = tickers[:200]
        except:
            tickers = []

    # [시도 3] 최후의 수단: 시가총액 기반으로 강제 수집 (에러 방지용)
    if len(tickers) < 150:
        print("[cache_k200_members] Attempt 3: Market Cap sorting (Emergency)")
        try:
            df_cap = stock.get_market_cap_by_ticker(target_date, market="KOSPI")
            tickers = df_cap.sort_values("시가총액", ascending=False).index[:200].tolist()
        except:
            pass

    # 최종 확인: 여전히 0개면 에러를 내어 덮어쓰기 방지
    if not tickers or len(tickers) < 50:
        raise RuntimeError("CRITICAL: 모든 수집 수단이 실패했습니다. 0행 파일을 생성하지 않습니다.")

    # 2. 종목명 매핑
    print(f"[cache_k200_members] Successfully found {len(tickers)} tickers. Mapping names...")
    names = []
    for t in tickers:
        try:
            names.append(stock.get_market_ticker_name(t))
        except:
            names.append("")

    # 3. 데이터프레임 저장
    df = pd.DataFrame({
        "asof_date": target_date,
        "isu_cd": tickers,
        "isu_nm": names,
    }).sort_values("isu_cd")

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[cache_k200_members] Success! {len(df)} rows saved to {out_path}")

if __name__ == "__main__":
    main()
