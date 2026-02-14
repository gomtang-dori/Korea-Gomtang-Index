# src/caches/probe_f08_foreigner_pykrx.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from pykrx import stock


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


def _safe_end_yyyymmdd() -> str:
    # GitHub Actions에서 미래 날짜 문제 방지: UTC 어제
    end = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)
    return end.strftime("%Y%m%d")


def _start_yyyymmdd_from(end_yyyymmdd: str, lookback_days: int) -> str:
    end = pd.to_datetime(end_yyyymmdd, format="%Y%m%d")
    # 주말/휴일 감안해서 넉넉히
    start = (end - pd.Timedelta(days=lookback_days)).normalize()
    return start.strftime("%Y%m%d")


def _try_fetch(start: str, end: str, market: str):
    """
    pykrx 버전별로 함수/컬럼이 다를 수 있어 몇 가지 후보를 시도.
    probe 목적: "Actions에서 수집이 되나/컬럼이 뭐로 오나" 확인.
    """
    attempts = []

    # 후보 1) 투자자별 '거래대금' 일별 (개인/외국인/기관…)
    # (많이 쓰는 패턴: get_market_trading_value_by_date)
    attempts.append(("get_market_trading_value_by_date", lambda: stock.get_market_trading_value_by_date(start, end, market)))

    # 후보 2) 투자자별 '거래량' 일별
    attempts.append(("get_market_trading_volume_by_date", lambda: stock.get_market_trading_volume_by_date(start, end, market)))

    last_err = None
    for name, fn in attempts:
        try:
            df = fn()
            return name, df, None
        except Exception as e:
            last_err = e
    return None, None, last_err


def _pick_foreigner_net_buy(df: pd.DataFrame) -> tuple[str | None, str]:
    """
    컬럼명이 버전/응답에 따라 다를 수 있어 후보를 넓게 둠.
    - 보통 '외국인' 컬럼이 net buy(순매수)로 오는 경우가 많음(예제/블로그들).
    """
    cols = list(df.columns)

    # 가장 흔한 후보들
    candidates = [
        "외국인",
        "외국인합계",
        "FOREIGNER",
        "Foreign",
        "외국인투자자",
    ]
    for c in candidates:
        if c in cols:
            return c, f"matched direct col '{c}'"

    # 혹시 다중 인덱스/컬럼 구조면 문자열 포함으로 탐색
    for c in cols:
        if isinstance(c, str) and ("외국인" in c or "Foreign" in c):
            return c, f"matched fuzzy col '{c}'"

    return None, f"cannot find foreigner column. cols={cols}"


def main():
    market = os.environ.get("F08_MARKET", "KOSPI")
    out_path = Path(os.environ.get("F08_PROBE_OUT", "data/cache/probe_f08_foreigner_pykrx.parquet"))
    days = _env_int("F08_DAYS", 10)

    end = os.environ.get("BACKFILL_END", "").strip() or _safe_end_yyyymmdd()
    start = os.environ.get("BACKFILL_START", "").strip()
    if not start:
        # 영업일 10개 확보하려면 캘린더상 30~45일 정도가 안전
        start = _start_yyyymmdd_from(end, lookback_days=max(45, days * 5))

    print(f"[probe_f08:pykrx] market={market} range={start}~{end} target_days={days}")

    method, df, err = _try_fetch(start, end, market)
    if df is None or df is False:
        raise RuntimeError(f"[probe_f08:pykrx] fetch failed. last_err={type(err).__name__}: {err}")

    if df is None or df.empty:
        raise RuntimeError(f"[probe_f08:pykrx] empty result. method={method} range={start}~{end}")

    print(f"[probe_f08:pykrx] method={method} rows={len(df)} cols={list(df.columns)}")
    df = df.copy()
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 최근 N개만
    tail = df.tail(days).reset_index(drop=True)

    foreign_col, msg = _pick_foreigner_net_buy(tail.drop(columns=["date"]))
    print(f"[probe_f08:pykrx] foreign_col_pick: {msg}")

    # 결과 저장(전체 컬럼 보존 + foreign_col 있으면 표준 컬럼도 추가)
    if foreign_col is not None:
        tail["foreigner_net_buy"] = pd.to_numeric(tail[foreign_col], errors="coerce")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tail.to_parquet(out_path, index=False)
    csv_path = out_path.with_suffix(".csv")
    tail.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"[probe_f08:pykrx] saved parquet={out_path} csv={csv_path}")
    print("[probe_f08:pykrx] head:")
    print(tail.head(3).to_string(index=False))
    print("[probe_f08:pykrx] tail:")
    print(tail.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
