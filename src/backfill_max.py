# -*- coding: utf-8 -*-
"""
backfill_max.py (v3 안정판)
- 목적: 가능한 긴 기간(실제로는 12년)으로 지수/팩터 원시값과 점수를 1회 생성해 data/index_daily.parquet로 저장
- ①/②/③/⑥/⑦/⑧: pykrx 기반 (src/lib/pykrx_factors.py)
- ⑤: data.go.kr 소매채권수익률요약(AA-3Y - A-3Y)
- ⑩: ECOS USD/KRW 변동성
- ⑨ 제외(합의대로)
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

@dataclass(frozen=True)
class CFG:
    # 히트맵(최근 10년~)을 위해 12년 확보
    YEARS: int = 12

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

    # ⑨ 제외, ④는 추후 KRX 옵션으로 추가
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
fear_keys = {"f05", "f06", "f10"}  # 공포성

def yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")

def save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def rolling_percentile(series: pd.Series, window: int, min_obs: int) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        w = x.iloc[start:i+1].dropna().values
        if len(w) < min_obs or np.isnan(x.iloc[i]):
            continue
        out[i] = 100.0 * (w <= x.iloc[i]).mean()
    return pd.Series(out, index=series.index)

def forward_return(s: pd.Series, n: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.shift(-n) / s - 1.0

def forward_win(s: pd.Series, n: int) -> pd.Series:
    return (forward_return(s, n) > 0).astype(float)

# ---------------- ⑤ data.go.kr ----------------
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
    # ⑤ 필드 구조: basDt/crdtSc/ctg/bnfRt [Source](https://www.genspark.ai/api/files/s/S7VQug0I)
    df = df.rename(columns={"basDt": "date", "crdtSc": "grade", "ctg": "bucket", "bnfRt": "yield"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    return df[["date", "grade", "bucket", "yield"]].dropna(subset=["date"])

def build_f05_raw(f05: pd.DataFrame) -> pd.DataFrame:
    if f05.empty:
        return pd.DataFrame(columns=["date", "f05_raw"])
    df = f05.copy()
    df = df[df["bucket"].astype(str) == cfg.F05_CTG_3Y]
    hi = df[df["grade"].astype(str) == cfg.F05_GRADE_HI][["date", "yield"]].rename(columns={"yield": "y_hi"})
    lo = df[df["grade"].astype(str) == cfg.F05_GRADE_LO][["date", "yield"]].rename(columns={"yield": "y_lo"})
    m = pd.merge(hi, lo, on="date", how="inner")
    if m.empty:
        return pd.DataFrame(columns=["date", "f05_raw"])
    m["f05_raw"] = m["y_hi"] - m["y_lo"]
    return m[["date", "f05_raw"]]

# ---------------- ⑩ ECOS ----------------
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
    if f10.empty:
        return pd.DataFrame(columns=["date", "f10_raw"])
    df = f10.sort_values("date").reset_index(drop=True).copy()
    df["ret"] = np.log(df["usdkrw"] / df["usdkrw"].shift(1))
    df["f10_raw"] = df["ret"].rolling(20).std() * math.sqrt(252)
    return df[["date", "f10_raw"]]

def main():
    today = datetime.utcnow().date()
    end_s = yyyymmdd(today)
    start_long_s = yyyymmdd(today - timedelta(days=365 * cfg.YEARS))

    # 1) K200 (프록시 069500) — fetch_kospi200_ohlcv는 이제 안정판이어야 함
    k200 = fetch_kospi200_ohlcv(start_long_s, end_s)
    k200_ok = (not k200.empty) and ("k200_close" in k200.columns) and (k200["k200_close"].dropna().size > 10)

    # 2) ⑤/⑩ long window
    f05 = fetch_f05(start_long_s, end_s)
    f10 = fetch_f10(start_long_s, end_s)

    # 원시 저장(선택)
    if not f05.empty:
        save_parquet(f05, DATA_DIR / "f05.parquet")
    if not f10.empty:
        save_parquet(f10, DATA_DIR / "f10.parquet")

    f05_raw = build_f05_raw(f05)
    f10_raw = build_f10_raw(f10)

    # 3) pykrx factors (안전 처리)
    f01 = factor1_momentum(k200).rename(columns={0: "f01_raw"}) if k200_ok else pd.DataFrame(columns=["date", "f01_raw"])
    f06 = factor6_volatility(k200).rename(columns={0: "f06_raw"}) if k200_ok else pd.DataFrame(columns=["date", "f06_raw"])

    # Strength/Breadth는 시간이 길거나 실패할 수 있으니 try/except
    try:
        f02 = factor2_strength(start_long_s, end_s)
    except Exception:
        f02 = pd.DataFrame(columns=["date", "f02_raw"])
    try:
        f03 = factor3_breadth(start_long_s, end_s)
    except Exception:
        f03 = pd.DataFrame(columns=["date", "f03_raw"])

    # ⑦ SafeHaven
    try:
        f07 = factor7_safe_haven(k200, f10).rename(columns={0: "f07_raw"}) if (k200_ok and not f10.empty) else pd.DataFrame(columns=["date", "f07_raw"])
    except Exception:
        f07 = pd.DataFrame(columns=["date", "f07_raw"])

    # ⑧ Foreign netbuy
    try:
        f08 = factor8_foreign_netbuy(start_long_s, end_s)
        if "f08_raw" not in f08.columns:
            f08 = pd.DataFrame(columns=["date", "f08_raw"])
    except Exception:
        f08 = pd.DataFrame(columns=["date", "f08_raw"])

    # 4) merge
    base = k200.copy() if k200_ok else (f10[["date"]].drop_duplicates().copy() if not f10.empty else pd.DataFrame(columns=["date"]))
    for dfx in [f01, f02, f03, f05_raw, f06, f07, f08, f10_raw]:
        if dfx is None or dfx.empty:
            continue
        base = pd.merge(base, dfx, on="date", how="outer")
    base = base.sort_values("date").reset_index(drop=True)

    # 5) K200 derived (없으면 NaN)
    if "k200_close" not in base.columns:
        base["k200_close"] = np.nan

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

    # 6) scores
    for key in ["f01", "f02", "f03", "f05", "f06", "f07", "f08", "f10"]:
        raw = f"{key}_raw"
        sc = f"{key}_score"
        if raw in base.columns:
            base[sc] = rolling_percentile(base[raw], cfg.ROLLING_DAYS, cfg.MIN_OBS)

    # 7) index (결측 자동 재정규화)
    base["index_score_total"] = np.nan
    for i in range(len(base)):
        row = base.iloc[i]
        acc = 0.0
        wsum = 0.0
        for k, w in cfg.W.items():
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
