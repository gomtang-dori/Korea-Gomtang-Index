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
    1차: get_index_portfolio_deposit_file
    2차: get_index_ticker_list (환경/버전 차이 대비)
    """
    tickers = []
    try:
        tickers = stock.get_index_portfolio_deposit_file("1028", asof)
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
    '성공'이 아니라 '비어있지 않은 구성종목(>=150개 같은)'이 나오는 날짜를 찾는다.
    """
    d = datetime.utcnow().date()
    for _ in range(max_lookback_days):
        asof = d.strftime("%Y%m%d")
        tickers = _try_get_members(asof)

        # 빈 리스트는 실패로 간주하고 날짜를 뒤로 이동
        if len(tickers) >= 150:  # 보수적으로 기준
            return asof, tickers

        d = d - timedelta(days=1)

    return "", []


def main():
    out_path = Path(os.environ.get("K200_MEMBERS_PATH", "data/k200_members.csv"))
    ensure_dir(out_path.parent)

    asof, tickers = pick_recent_business_day_kr()

    if not tickers:
        raise RuntimeError(
            "Failed to fetch KOSPI200 members from pykrx (empty for last 30 days). "
            "Check pykrx availability / index code / network."
        )

    # 종목명은 optional (속도/실패 방지)
    names = {}
    try:
        names = {t: stock.get_market_ticker_name(t) for t in tickers}
    except Exception:
        names = {t: "" for t in tickers}

    df = pd.DataFrame({
        "asof_date": asof,
        "isu_cd": tickers,
        "isu_nm": [names.get(t, "") for t in tickers],
    }).sort_values("isu_cd")

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[cache_k200_members] OK asof={asof} rows={len(df)} -> {out_path}")


if __name__ == "__main__":
    main()
