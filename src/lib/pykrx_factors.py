# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pykrx import stock

def _to_yyyymmdd(dt):
    return dt.strftime("%Y%m%d")

def _as_date_index(df):
    # pykrx index는 datetime/str 혼합일 수 있어 통일
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def fetch_kospi200_ohlcv(start_date: str, end_date: str) -> pd.DataFrame:
    """
    KOSPI200 프록시: 069500(KODEX 200) OHLCV에서 종가를 k200_close로 표준화.
    반환: date, k200_close
    - 어떤 예외가 나도 UnboundLocalError 없이 '빈 DF'를 반환
    """
    import pandas as pd
    from pykrx import stock

    ticker = "069500"

    # 항상 out을 먼저 안전하게 초기화
    out = pd.DataFrame(columns=["date", "k200_close"])

    try:
        df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
        if df is None or df.empty:
            return out

        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # pykrx OHLCV의 종가 컬럼명은 보통 "종가"
        if "종가" in df.columns:
            close = pd.to_numeric(df["종가"], errors="coerce")
        else:
            # 혹시 컬럼명이 다른 경우를 대비
            # (가능한 후보들)
            for cand in ["Close", "close", "종가 "]:
                if cand in df.columns:
                    close = pd.to_numeric(df[cand], errors="coerce")
                    break
            else:
                return out

        out = pd.DataFrame({"date": df.index, "k200_close": close}).dropna()
        out = out.sort_values("date").reset_index(drop=True)
        return out

    except Exception:
        # 어떤 실패든 빈 DF 반환(파이프라인 중단 방지)
        return out


def factor1_momentum(k200: pd.DataFrame) -> pd.DataFrame:
    # ① 모멘텀: 252일 수익률(1Y) 퍼센타일용 raw
    s = k200.set_index("date")["k200_close"].astype(float)
    raw = s.pct_change(252)
    return raw.reset_index().rename(columns={"k200_close":"f01_raw", 0:"f01_raw"})

def factor2_strength(start_date: str, end_date: str) -> pd.DataFrame:
    """
    ② Strength(52주 신고/신저가 비율):
    - 유니버스: KOSPI+KOSDAQ (원하면 KOSPI만도 가능)
    - 계산: 각 일자별 (신고가 비율 - 신저가 비율) 같은 형태로 raw 생성
    구현을 단순화: '당일 종가가 252일 최고면 +1, 최저면 -1'을 종목별로 계산 후 평균
    """
    # 티커 목록
    tickers = stock.get_market_ticker_list(end_date, market="ALL")
    # 일자별 종가 패널을 직접 다 모으면 매우 무거워서,
    # 현실적인 MVP: 최근 400영업일만 가져와서 rolling 252 계산
    # (start_date는 호출자가 넉넉히 잡도록)
    panel = []
    for t in tickers[:300]:  # MVP: 300개로 제한(속도). 추후 전체로 확대 가능
        try:
            df = stock.get_market_ohlcv_by_date(start_date, end_date, t)
            if df.empty: 
                continue
            df = _as_date_index(df)
            close = pd.to_numeric(df["종가"], errors="coerce")
            hi = close.rolling(252).max()
            lo = close.rolling(252).min()
            sig = (close == hi).astype(float) - (close == lo).astype(float)
            panel.append(sig.rename(t))
        except Exception:
            continue
    if not panel:
        return pd.DataFrame(columns=["date","f02_raw"])
    mat = pd.concat(panel, axis=1)
    raw = mat.mean(axis=1)
    return raw.reset_index().rename(columns={0:"f02_raw", "index":"date"})

def factor3_breadth(start_date: str, end_date: str) -> pd.DataFrame:
    """
    ③ Breadth: 상승 종목 비율(advancers/decliners) 기반 raw
    MVP: KOSPI+KOSDAQ 일부 티커로 당일 수익률>0 비율을 계산
    """
    tickers = stock.get_market_ticker_list(end_date, market="ALL")
    panel = []
    for t in tickers[:500]:
        try:
            df = stock.get_market_ohlcv_by_date(start_date, end_date, t)
            if df.empty:
                continue
            df = _as_date_index(df)
            ret = pd.to_numeric(df["종가"], errors="coerce").pct_change(1)
            panel.append((ret > 0).astype(float).rename(t))
        except Exception:
            continue
    if not panel:
        return pd.DataFrame(columns=["date","f03_raw"])
    mat = pd.concat(panel, axis=1)
    raw = mat.mean(axis=1)  # 상승 비율
    return raw.reset_index().rename(columns={0:"f03_raw", "index":"date"})

def factor6_volatility(k200: pd.DataFrame) -> pd.DataFrame:
    """
    ⑥ 변동성(VKOSPI 대체): KODEX200(프록시)의 20일 실현변동성(연율화)
    """
    s = k200.set_index("date")["k200_close"].astype(float)
    r = np.log(s / s.shift(1))
    vol = r.rolling(20).std() * np.sqrt(252)
    return vol.reset_index().rename(columns={0:"f06_raw"})

def factor7_safe_haven(k200: pd.DataFrame, usdkrw: pd.DataFrame) -> pd.DataFrame:
    """
    ⑦ Safe haven(간편 MVP):
    - 위험자산(K200) vs 안전선호(환율변동성/환율상승) 조합의 상대 강도
    - raw = (K200 60일 수익률) - (USD/KRW 60일 변화율)
    """
    k = k200.set_index("date")["k200_close"].astype(float)
    u = usdkrw.set_index("date")["usdkrw"].astype(float)
    kret = k.pct_change(60)
    uret = u.pct_change(60)
    raw = kret - uret
    raw.name = "f07_raw"
    return raw.reset_index()

# --- compatibility alias (daily_update/backfill uses factor7_safehaven name) ---
def factor7_safehaven(k200: pd.DataFrame, usdkrw: pd.DataFrame) -> pd.DataFrame:
    return factor7_safe_haven(k200, usdkrw)


def factor8_foreign_netbuy(start_date: str, end_date: str) -> pd.DataFrame:
    """
    ⑧ 외국인 순매수(거래대금) - pykrx 버전 차이 안전 처리
    - market 키워드 미사용 (positional arg)
    - '외국인' 컬럼 우선 탐색, 없으면 유사 키워드 탐색
    - 실패 시 NaN 대신 0으로 반환(파이프라인 중단 방지)
    """
    def _pick_foreign_col(df: pd.DataFrame):
        # 1) 가장 흔한 컬럼명
        if "외국인" in df.columns:
            return "외국인"
        # 2) 혹시 공백/변형이 있는 경우
        for c in df.columns:
            if isinstance(c, str) and ("외국인" in c):
                return c
        return None

    try:
        # 키워드 market= 사용 금지 (TypeError 방지)
        kospi = stock.get_market_trading_value_by_investor(start_date, end_date, "KOSPI")
        kosdaq = stock.get_market_trading_value_by_investor(start_date, end_date, "KOSDAQ")

        kospi = _as_date_index(kospi)
        kosdaq = _as_date_index(kosdaq)

        c1 = _pick_foreign_col(kospi)
        c2 = _pick_foreign_col(kosdaq)
        if (c1 is None) or (c2 is None):
            # 컬럼이 예상과 다를 때: 파이프라인이 죽지 않게 0으로 리턴
            idx = kospi.index.union(kosdaq.index)
            out = pd.DataFrame({"date": idx, "f08_raw": 0.0})
            return out.reset_index(drop=True)

        raw = pd.to_numeric(kospi[c1], errors="coerce").fillna(0) + pd.to_numeric(kosdaq[c2], errors="coerce").fillna(0)
        return raw.reset_index().rename(columns={0: "f08_raw", c1: "f08_raw", "index": "date"})

    except Exception:
        # 어떤 이유로든 실패하면 0으로 반환 (중단 방지)
        return pd.DataFrame(columns=["date", "f08_raw"])

