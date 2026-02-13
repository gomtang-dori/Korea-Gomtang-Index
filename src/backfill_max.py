# -*- coding: utf-8 -*-
"""
backfill_max.py (v2 안정판)
- ①/②/③/⑥/⑦/⑧: pykrx (src/lib/pykrx_factors.py 사용)
- ⑤: data.go.kr (소매채권수익률요약) AA-3Y - A-3Y 스프레드
- ⑩: ECOS USD/KRW 기반 변동성
- ⑨: 이번 단계에서 제외(없어도 동작) + 가중치 자동 재정규화

산출:
- data/index_daily.parquet
- data/f05.parquet, data/f10.parquet
- (리포트는 daily_update가 생성)
"""
import os
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from lib.pykrx_factors import (
    fetch_kospi200_ohlcv,
    factor1_momentum,
    factor2_strength,
    factor3_breadth,
    factor6_volatility,
    factor7_safe_haven,
    factor8_foreign_netbuy,
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- config ----------------
@dataclass(frozen=True)
class CFG:
    START_DATE: str = "20000101"        # 가능한 최대 시작(실제 데이터는 pykrx가 가능한 범위에서 채움)
    ROLLING_DAYS: int = 252 * 5
    MIN_OBS: int = 252

    # ⑤
    F05_CTG_3Y: str = "2년~3년미만"
    F05_GRADE_HI: str = "AA-"
    F05_GRADE_LO: str = "A-"

    # ⑩
    ECOS_STAT_CODE_USDKRW: str = "731Y003"
    ECOS_CYCLE: str = "D"
    ECOS_ITEM_USDKRW: str = "0000003"

    # weights (⑨ 제외, ④는 다음 단계(KRX)에서 붙일 예정이지만, 지금은 미포함)
    W = {
        "f01": 0.10,
        "f02": 0.10,
        "f03": 0.10,
        "f05": 0.05,
        "f06": 0.125,
        "f07": 0.10,
        "f08": 0.10,
        "f10": 0.10,
    }

cfg = CFG()

# ---------------- utils ----------------
def save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

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

def forward_return(s: pd.Series, n: int) -> pd.Series:
    s = s.astype(float)
    return s.shift(-n) / s - 1.0

def forward_win(s: pd.Series, n: int) -> pd.Series:
    return (forward_return(s, n) > 0).astype(float)

# ---------------- fetch ⑤ ----------------
def fetch_f05(begin: str, end: str) -> pd.DataFrame:
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
    df = df.rename(columns={"basDt": "date", "crdtSc": "grade", "ctg": "bucket", "bnfRt": "yield"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    return df[["date", "grade", "bucket", "yield"]].dropna(subset=["date"])

def build_f05_raw(f05: pd.DataFrame) -> pd.DataFrame:
    df = f05.copy()
    df = df[df["bucket"].astype(str) == cfg.F05_CTG_3Y]
    hi = df[df["grade"].astype(str) == cfg.F05_GRADE_HI][["date", "yield"]].rename(columns={"yield": "y_hi"})
    lo = df[df["grade"].astype(str) == cfg.F05_GRADE_LO][["date", "yield"]].rename(columns={"yield": "y_lo"})
    m = pd.merge(hi, lo, on="date", how="inner")
    m["f05_raw"] = m["y_hi"] - m["y_lo"]
    return m[["date", "f05_raw"]]

# ---------------- fetch ⑩ ----------------
def fetch_f10(begin: str, end: str) -> pd.DataFrame:
    ecos_key = os.environ.get("ECOS_KEY", "").strip()
    if not ecos_key:
        raise RuntimeError("ECOS_KEY 환경변수가 비어 있습니다.")
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{ecos_key}/json/kr/1/100000/{cfg.ECOS_STAT_CODE_USDKRW}/{cfg.ECOS_CYCLE}/{begin}/{end}/{cfg.ECOS_ITEM_USDKRW}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    rows = r.json().get("StatisticSearch", {}).get("row", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["TIME"], format="%Y%m%d", errors="coerce")
    df["usdkrw"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    return df[["date", "usdkrw"]].dropna(subset=["date"])

def build_f10_raw(f10: pd.DataFrame) -> pd.DataFrame:
    df = f10.sort_values("date").reset_index(drop=True).copy()
    df["ret"] = np.log(df["usdkrw"] / df["usdkrw"].shift(1))
    df["f10_raw"] = df["ret"].rolling(20).std() * math.sqrt(252)
    return df[["date", "f10_raw"]]

# ---------------- main ----------------
def main():
    end = datetime.utcnow().date()
    start = datetime.strptime(cfg.START_DATE, "%Y%m%d").date()
    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")

    # pykrx long window: 12년 정도면 히트맵(10년) 충분
    start_long = (end - timedelta(days=365 * 12)).strftime("%Y%m%d")

    # 1) K200 proxy (KODEX200)
    k200 = fetch_kospi200_ohlcv(start_long, end_s)  # date, k200_close

    # 2) ⑤/⑩
    f05 = fetch_f05(start_s, end_s)
    f10 = fetch_f10(start_s, end_s)
    if not f05.empty:
        save_parquet(f05, DATA_DIR / "f05.parquet")
    if not f10.empty:
        save_parquet(f10, DATA_DIR / "f10.parquet")

    f05_raw = build_f05_raw(f05) if not f05.empty else pd.DataFrame()
    f10_raw = build_f10_raw(f10) if not f10.empty else pd.DataFrame()

    # 3) pykrx factors
    f01 = factor1_momentum(k200).rename(columns={0: "f01_raw"})
    f02 = factor2_strength(start_long, end_s)
    f03 = factor3_breadth(start_long, end_s)
    f06 = factor6_volatility(k200).rename(columns={0: "f06_raw"})
    f07 = factor7_safe_haven(k200, f10).rename(columns={0: "f07_raw"})
    f08 = factor8_foreign_netbuy(start_long, end_s)

    # 4) merge
    base = k200.copy()
    for dfx in [f01, f02, f03, f05_raw, f06, f07, f08, f10_raw]:
        if dfx is None or dfx.empty:
            continue
        base = pd.merge(base, dfx, on="date", how="outer")
    base = base.sort_values("date").reset_index(drop=True)

    if "k200_close" not in base.columns or base["k200_close"].dropna().empty:
    # K200이 없으면 이후 K200 파생/히트맵 섹션 스킵 가능하도록 NaN 컬럼만 만들어 둠
    base["k200_close"] = np.nan

    # 5) K200 derived (trend & forward)
if base["k200_close"].notna().sum() > 10:
    base["k200_ret_3d"] = base["k200_close"].pct_change(3)
    base["k200_ret_5d"] = base["k200_close"].pct_change(5)
    base["k200_ret_7d"] = base["k200_close"].pct_change(7)
    base["k200_fwd_10d_return"] = forward_return(base["k200_close"], 10)
    base["k200_fwd_10d_win"] = forward_win(base["k200_close"], 10)
else:
    base["k200_ret_3d"] = np.nan
    base["k200_ret_5d"] = np.nan
    base["k200_ret_7d"] = np.nan
    base["k200_fwd_10d_return"] = np.nan
    base["k200_fwd_10d_win"] = np.nan


    # 6) percentile scores
    for key in ["f01", "f02", "f03", "f05", "f06", "f07", "f08", "f10"]:
        raw = f"{key}_raw"
        sc = f"{key}_score"
        if raw in base.columns:
            base[sc] = rolling_percentile(base[raw], cfg.ROLLING_DAYS, cfg.MIN_OBS)

    # 7) final index (greed direction; fear factors inverted)
    fear_keys = {"f05", "f06", "f10"}  # 공포성: 높을수록 공포 -> 탐욕점수는 100-pct
    weights = cfg.W
    base["index_score_total"] = np.nan

    for i in range(len(base)):
        row = base.iloc[i]
        acc = 0.0
        wsum = 0.0
        for k, w in weights.items():
            sc = row.get(f"{k}_score", np.nan)
            if pd.isna(sc):
                continue
            gs = 100 - sc if k in fear_keys else sc
            acc += w * gs
            wsum += w
        if wsum > 0:
            base.at[i, "index_score_total"] = acc / wsum

    base["bucket_5pt"] = (base["index_score_total"] // 5 * 5).clip(0, 95).astype("Int64")
    base["index_chg_3d"] = base["index_score_total"].diff(3)
    base["index_chg_5d"] = base["index_score_total"].diff(5)
    base["index_chg_7d"] = base["index_score_total"].diff(7)

    save_parquet(base, DATA_DIR / "index_daily.parquet")
    print("[backfill_max] OK: data/index_daily.parquet 생성 완료")

if __name__ == "__main__":
    main()
