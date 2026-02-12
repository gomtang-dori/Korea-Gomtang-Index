# -*- coding: utf-8 -*-
"""
backfill_max.py
- 목적: 가능한 최대 기간으로 ⑤/⑨/⑩ + (가능하면) KOSPI200을 백필하고,
        5년 롤링 퍼센타일(0~100) 점수, 히트맵용 forward 통계까지 생성.

실행:
  python src/backfill_max.py

필수 환경변수:
  SERVICE_KEY (data.go.kr)
  ECOS_KEY (ECOS)

산출물:
  data/f05.parquet
  data/f09.parquet
  data/f10.parquet
  data/k200.parquet (성공 시)
  data/index_daily.parquet  (지수/팩터/추세/forward통계 포함)
  docs/index.html (daily_update에서 생성 권장. backfill은 데이터만 생성)
"""
import os
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
from pathlib import Path

DATA_DIR = Path("data")
DOCS_DIR = Path("docs")
ASSET_DIR = DOCS_DIR / "assets"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
ASSET_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 설정 ----------
@dataclass(frozen=True)
class CFG:
    # Daily update에서도 동일하게 씁니다
    REFETCH_DAYS: int = 14
    ROLLING_DAYS: int = 252 * 5
    MIN_OBS: int = 252

    # Factor 5: AA- 3Y - A- 3Y (ctg=2년~3년미만)
    F05_CTG_3Y: str = "2년~3년미만"
    F05_GRADE_HI: str = "AA-"
    F05_GRADE_LO: str = "A-"

    # ECOS USDKRW
    ECOS_STAT_CODE_USDKRW: str = "731Y003"
    ECOS_CYCLE: str = "D"
    ECOS_ITEM_USDKRW: str = "0000003"

    # 가중치 (⑤ 낮춤)
    W_F05: float = 0.05
    W_F09: float = 0.125
    W_F10: float = 0.10

cfg = CFG()

# ---------- 공통 유틸 ----------
def yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")

def save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

def upsert(old: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if old.empty:
        out = new.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)
        out = out.drop_duplicates(subset=keys, keep="last")
    return out.sort_values(keys).reset_index(drop=True)

def rolling_percentile(series: pd.Series, window: int, min_obs: int) -> pd.Series:
    x = series.astype(float)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        w = x.iloc[start:i+1].dropna().values
        if len(w) < min_obs or np.isnan(x.iloc[i]):
            continue
        out[i] = 100.0 * (w <= x.iloc[i]).mean()
    return pd.Series(out, index=series.index)

def pct_change_n(s: pd.Series, n: int) -> pd.Series:
    return s.astype(float).pct_change(n)

def forward_return(s: pd.Series, n: int) -> pd.Series:
    s = s.astype(float)
    return s.shift(-n) / s - 1.0

def forward_win(s: pd.Series, n: int) -> pd.Series:
    r = forward_return(s, n)
    return (r > 0).astype(float)

# ---------- Fetch: Factor 5 ----------
def fetch_f05(begin: str, end: str) -> pd.DataFrame:
    # data.go.kr 소매채권수익률요약
    base = "https://apis.data.go.kr/1160100/service/GetBondInfoService/getBondSecurityBenefitRate"
    service_key = os.environ.get("SERVICE_KEY", "").strip()
    if not service_key:
        raise RuntimeError("SERVICE_KEY 환경변수가 비어 있습니다.")
    params = {
        "serviceKey": service_key,
        "resultType": "json",
        "numOfRows": 3000,
        "pageNo": 1,
        "beginBasDt": begin,
        "endBasDt": end,
    }
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    body = r.json()["response"]["body"]
    items = body.get("items", [])
    if isinstance(items, dict):
        items = items.get("item", [])
    df = pd.DataFrame(items)
    if df.empty:
        return df
    df = df.rename(columns={"basDt":"date","crdtSc":"grade","ctg":"bucket","bnfRt":"yield"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    return df[["date","grade","bucket","yield"]].dropna(subset=["date"])

# ---------- Fetch: Factor 10 ----------
def fetch_f10(begin: str, end: str) -> pd.DataFrame:
    ecos_key = os.environ.get("ECOS_KEY", "").strip()
    if not ecos_key:
        raise RuntimeError("ECOS_KEY 환경변수가 비어 있습니다.")
    stat_code = cfg.ECOS_STAT_CODE_USDKRW
    cycle = cfg.ECOS_CYCLE
    item = cfg.ECOS_ITEM_USDKRW
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{ecos_key}/json/kr/1/100000/{stat_code}/{cycle}/{begin}/{end}/{item}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    rows = r.json().get("StatisticSearch", {}).get("row", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["TIME"], format="%Y%m%d", errors="coerce")
    df["usdkrw"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    return df[["date","usdkrw"]].dropna(subset=["date"])

# ---------- Fetch: Factor 9 (TODO: 오퍼레이션/컬럼명 확정 필요) ----------
def fetch_f09(begin: str, end: str) -> pd.DataFrame:
    """
    ⑨는 data.go.kr GetKofiaStatisticsInfoService의 실제 operationId/필드명이
    사용자 계정에서 확인되어야 합니다.
    - 아래 URL 2개를 본인 문서의 operation endpoint로 교체하세요.
    - 아래 value 컬럼 2개도 응답에 맞게 교체하세요.
    """
    service_key = os.environ.get("SERVICE_KEY", "").strip()
    if not service_key:
        raise RuntimeError("SERVICE_KEY 환경변수가 비어 있습니다.")

    # TODO(1): endpoint 교체
    NUM_URL = "https://apis.data.go.kr/1160100/service/GetKofiaStatisticsInfoService/getGrantingOfCreditBalanceInfo"
    DEN_URL = "https://apis.data.go.kr/1160100/service/GetKofiaStatisticsInfoService/getSecuritiesMarketTotalCapitalInfo"

    # TODO(2): value 컬럼명 교체
    NUM_VAL_COL = "crdtrRlngWth"  # 분자(신용공여잔고) 예시
    DEN_VAL_COL = "intDpsAmnt"    # 분모(예탁금) 예시

    def _fetch(url: str) -> pd.DataFrame:
        params = {
            "serviceKey": service_key,
            "resultType": "json",
            "numOfRows": 3000,
            "pageNo": 1,
            "beginBasDt": begin,
            "endBasDt": end,
        }
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        body = r.json()["response"]["body"]
        items = body.get("items", [])
        if isinstance(items, dict):
            items = items.get("item", [])
        return pd.DataFrame(items)

    num = _fetch(NUM_URL)
    den = _fetch(DEN_URL)
    if num.empty or den.empty:
        return pd.DataFrame()

    for df in (num, den):
        if "basDt" in df.columns:
            df["date"] = pd.to_datetime(df["basDt"], format="%Y%m%d", errors="coerce")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            raise RuntimeError("⑨ 응답에서 날짜 컬럼(basDt/date)을 찾지 못했습니다. 응답 컬럼명을 확인하세요.")

    if NUM_VAL_COL not in num.columns or DEN_VAL_COL not in den.columns:
        raise RuntimeError(f"⑨ 컬럼명이 불일치합니다. 분자:{NUM_VAL_COL} 분모:{DEN_VAL_COL} (응답 컬럼 확인 필요)")

    num = num[["date", NUM_VAL_COL]].rename(columns={NUM_VAL_COL:"credit_balance"})
    den = den[["date", DEN_VAL_COL]].rename(columns={DEN_VAL_COL:"deposit"})
    out = pd.merge(num, den, on="date", how="inner")
    out["credit_balance"] = pd.to_numeric(out["credit_balance"], errors="coerce")
    out["deposit"] = pd.to_numeric(out["deposit"], errors="coerce")
    out["f09_raw"] = out["credit_balance"] / out["deposit"]
    return out.dropna(subset=["date"])[["date","credit_balance","deposit","f09_raw"]]

# ---------- KOSPI200 Fetch (KRX OpenAPI는 환경별로 서비스 선택 필요 → placeholder) ----------
def fetch_k200_placeholder() -> pd.DataFrame:
    """
    사용자가 원한 'KRX Open API로 KOSPI200'은
    어떤 KRX 서비스/필드로 가져올지 최종 확정이 필요합니다.
    - 우선 placeholder: data/k200_manual.csv가 있으면 읽어옵니다.
    컬럼: date(YYYY-MM-DD), k200_close
    """
    p = DATA_DIR / "k200_manual.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["k200_close"] = pd.to_numeric(df["k200_close"], errors="coerce")
    return df.dropna(subset=["date"])[["date","k200_close"]].sort_values("date").reset_index(drop=True)

# ---------- 팩터 raw → 점수/지수 ----------
def build_factor5_spread(f05: pd.DataFrame) -> pd.DataFrame:
    # 필터: ctg=2년~3년미만, grade=AA-, A-
    df = f05.copy()
    df = df[df["bucket"].astype(str) == cfg.F05_CTG_3Y]
    hi = df[df["grade"].astype(str) == cfg.F05_GRADE_HI][["date","yield"]].rename(columns={"yield":"y_hi"})
    lo = df[df["grade"].astype(str) == cfg.F05_GRADE_LO][["date","yield"]].rename(columns={"yield":"y_lo"})
    m = pd.merge(hi, lo, on="date", how="inner")
    m["f05_raw"] = m["y_hi"] - m["y_lo"]
    return m[["date","f05_raw"]].sort_values("date").reset_index(drop=True)

def build_factor10_vol(f10: pd.DataFrame) -> pd.DataFrame:
    df = f10.sort_values("date").reset_index(drop=True).copy()
    df["ret"] = np.log(df["usdkrw"] / df["usdkrw"].shift(1))
    df["f10_raw"] = df["ret"].rolling(20).std() * math.sqrt(252)
    return df[["date","f10_raw"]]

def compute_scores(index_df: pd.DataFrame) -> pd.DataFrame:
    df = index_df.sort_values("date").reset_index(drop=True).copy()

    # 퍼센타일 점수(0~100)
    # ⑤: 스프레드 ↑ = 공포 ↑  => pct 그대로
    df["f05_score"] = rolling_percentile(df["f05_raw"], cfg.ROLLING_DAYS, cfg.MIN_OBS)

    # ⑨: ratio ↑ = (대체로 risk-on)로 설계할 수도 있고, 반대로도 가능.
    # 여기서는 "ratio ↑ = 탐욕 ↑"라고 가정하고, 공포 점수로 만들려면 100-pct가 필요.
    # 사용자의 기존 합의가 '⑨ 쪽으로 가중치 상향'이었으므로, 일단 '탐욕' 방향 유지:
    #  -> 최종 지수(0~100)에서 높은 점수=탐욕이면 그대로, 높은 점수=공포면 100-pct.
    # 아래는 "탐욕 점수"로 산출:
if "f09_raw" in df.columns:
    df["f09_score"] = rolling_percentile(df["f09_raw"], cfg.ROLLING_DAYS, cfg.MIN_OBS)
    df["f09_score_greed"] = df["f09_score"]
else:
    df["f09_score"] = np.nan
    df["f09_score_greed"] = np.nan


    # ⑩: 변동성 ↑ = 공포 ↑ => pct 그대로
    df["f10_score"] = rolling_percentile(df["f10_raw"], cfg.ROLLING_DAYS, cfg.MIN_OBS)

    # 부분지수(⑤/⑨/⑩) — 나머지 팩터 합치기 전까지 임시 total로 사용 가능
    # 여기서는 "탐욕 지수"로 해석: ⑤,⑩은 공포성(↑공포), ⑨는 탐욕성(↑탐욕).
    # 사용자 최종 지수 방향(점수 높을수록 탐욕)으로 통일하려면:
    # - 공포성 팩터는 (100 - pct)로 뒤집어 탐욕 점수로 변환
    df["f05_score_greed"] = 100 - df["f05_score"]
    df["f10_score_greed"] = 100 - df["f10_score"]
    df["f09_score_greed"] = df["f09_score"]

parts = []
weights = []

if "f05_score_greed" in df.columns:
    parts.append(df["f05_score_greed"]); weights.append(cfg.W_F05)
if "f09_score_greed" in df.columns and df["f09_score_greed"].notna().any():
    parts.append(df["f09_score_greed"]); weights.append(cfg.W_F09)
if "f10_score_greed" in df.columns:
    parts.append(df["f10_score_greed"]); weights.append(cfg.W_F10)

wsum = sum(weights)
df["index_score_total"] = np.nan
if wsum > 0:
    df["index_score_total"] = sum(w*p for w,p in zip(weights, parts)) / wsum


    # 5점 단위 구간
    df["bucket_5pt"] = (df["index_score_total"] // 5 * 5).clip(0, 95).astype("Int64")

    return df

def attach_k200_features(df: pd.DataFrame, k200: pd.DataFrame) -> pd.DataFrame:
    if k200.empty:
        # 리포트에서 “K200 관련 섹션을 비활성화”하게 표시할 수 있도록 NaN 유지
        df["k200_close"] = np.nan
        df["k200_ret_3d"] = np.nan
        df["k200_ret_5d"] = np.nan
        df["k200_ret_7d"] = np.nan
        df["k200_fwd_10d_return"] = np.nan
        df["k200_fwd_10d_win"] = np.nan
        return df

    out = pd.merge(df, k200, on="date", how="left")
    out = out.sort_values("date").reset_index(drop=True)
    out["k200_ret_3d"] = pct_change_n(out["k200_close"], 3)
    out["k200_ret_5d"] = pct_change_n(out["k200_close"], 5)
    out["k200_ret_7d"] = pct_change_n(out["k200_close"], 7)
    out["k200_fwd_10d_return"] = forward_return(out["k200_close"], 10)
    out["k200_fwd_10d_win"] = forward_win(out["k200_close"], 10)
    return out

def main():
    print("[backfill_max] 시작")

    # 가능한 최대: 2000-01-01부터
    start = date(2000, 1, 1)
    end = datetime.utcnow().date()

    # 기간 chunk: 180일 단위
    step = timedelta(days=180)
    cur = start

    f05_all = load_parquet(DATA_DIR / "f05.parquet")
    f09_all = load_parquet(DATA_DIR / "f09.parquet")
    f10_all = load_parquet(DATA_DIR / "f10.parquet")

    while cur <= end:
        chunk_end = min(end, cur + step)
        b = yyyymmdd(cur)
        e = yyyymmdd(chunk_end)
        print(f"  - chunk {b} ~ {e}")

        # ⑤/⑩은 확정
        f05_new = fetch_f05(b, e)
        f10_new = fetch_f10(b, e)

        if not f05_new.empty:
            f05_all = upsert(f05_all, f05_new, ["date","grade","bucket"])
        if not f10_new.empty:
            f10_all = upsert(f10_all, f10_new, ["date"])

        # ⑨은 컬럼 확정 전이면 실패할 수 있으니 예외를 잡고 계속 진행
        try:
            f09_new = fetch_f09(b, e)
            if not f09_new.empty:
                f09_all = upsert(f09_all, f09_new, ["date"])
        except Exception as ex:
            print(f"    [warn] f09 chunk 실패(일단 스킵): {ex}")

        cur = chunk_end + timedelta(days=1)

    # 저장
    if not f05_all.empty: save_parquet(f05_all, DATA_DIR / "f05.parquet")
    if not f09_all.empty: save_parquet(f09_all, DATA_DIR / "f09.parquet")
    if not f10_all.empty: save_parquet(f10_all, DATA_DIR / "f10.parquet")

    # K200은 placeholder: 수동 파일 있으면 읽음
    k200 = fetch_k200_placeholder()
    if not k200.empty:
        save_parquet(k200, DATA_DIR / "k200.parquet")

    # index 계산(⑤/⑨/⑩ 기반)
    f05_spread = build_factor5_spread(f05_all) if not f05_all.empty else pd.DataFrame()
    f10_vol = build_factor10_vol(f10_all) if not f10_all.empty else pd.DataFrame()

    # ⑨ raw는 date,f09_raw
    f09_raw = f09_all[["date","f09_raw"]] if (not f09_all.empty and "f09_raw" in f09_all.columns) else pd.DataFrame()

    # date 기준 결합(outer)
    df = None
    for part in [f05_spread, f09_raw, f10_vol]:
        if part.empty:
            continue
        df = part if df is None else pd.merge(df, part, on="date", how="outer")

    if df is None or df.empty:
        raise RuntimeError("백필 결과로 결합 가능한 데이터가 없습니다(⑤/⑩ 호출/필터 확인).")

    df = df.sort_values("date").reset_index(drop=True)

    # 점수/지수
    df = compute_scores(df)

    # K200 feature 붙이기(가능하면)
    df = attach_k200_features(df, k200)

    save_parquet(df, DATA_DIR / "index_daily.parquet")
    print("[backfill_max] 완료: data/index_daily.parquet 생성")

if __name__ == "__main__":
    main()
