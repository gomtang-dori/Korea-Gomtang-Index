# src/caches/cache_k200_members.py
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# pykrx: 구성종목 리스트만 사용
from pykrx import stock


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pick_recent_business_day_kr() -> str:
    """
    KRX/pykrx는 '가장 최근 영업일' 날짜 문자열이 필요.
    여기서는 단순히 UTC 오늘 기준으로 하루씩 뒤로 가며
    데이터가 나오는 날을 찾는 방식(안정적).
    """
    # 최대 10일 탐색
    d = datetime.utcnow().date()
    for _ in range(10):
        ymd = d.strftime("%Y%m%d")
        try:
            # KOSPI200 포트폴리오 구성(지수코드 1028이 일반적)
            _ = stock.get_index_portfolio_deposit_file("1028", ymd)
            return ymd
        except Exception:
            d = d - timedelta(days=1)
    # fallback
    return (datetime.utcnow().date() - timedelta(days=1)).strftime("%Y%m%d")


def main():
    out_path = Path(os.environ.get("K200_MEMBERS_PATH", "data/k200_members.csv"))
    ensure_dir(out_path.parent)

    asof = pick_recent_business_day_kr()

    # KOSPI200 구성종목 (지수코드 1028)
    tickers = stock.get_index_portfolio_deposit_file("1028", asof)

    # 이름 매핑 (속도 이슈가 있으면 이름은 나중에 붙여도 됨)
    names = {t: stock.get_market_ticker_name(t) for t in tickers}

    df = pd.DataFrame({
        "asof_date": asof,
        "isu_cd": [str(t) for t in tickers],
        "isu_nm": [names.get(t, "") for t in tickers],
    }).sort_values("isu_cd")

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[cache_k200_members] OK asof={asof} rows={len(df)} -> {out_path}")


if __name__ == "__main__":
    main()
