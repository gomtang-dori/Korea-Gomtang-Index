# -*- coding: utf-8 -*-
"""
daily_update.py (v3 안정판)
- ①/②/③/⑥/⑦/⑧: pykrx 기반 (src/lib/pykrx_factors.py 사용)
- ⑤: data.go.kr 소매채권수익률요약(AA-3Y - A-3Y)
- ⑩: ECOS USD/KRW 변동성
- ⑨: 이번 단계에서 제외(없어도 동작)
- 리포트: docs/index.html 생성
"""
import os
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from jinja2 import Template

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
DOCS_DIR = Path("docs")
ASSET_DIR = DOCS_DIR / "assets"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
ASSET_DIR.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class CFG:
    REFETCH_DAYS: int = 14
    ROLLING_DAYS: int = 252 * 5
    MIN_OBS: int = 252

    F05_CTG_3Y: str = "2년~3년미만"
    F05_GRADE_HI: str = "AA-"
    F05_GRADE_LO: str = "A-"

    ECOS_STAT_CODE_USDKRW: str = "731Y003"
    ECOS_CYCLE: str = "D"
    ECOS_ITEM_USDKRW: str = "0000003"

    # ⑨ 제외, ④는 추후 KRX 옵션으로 추가 예정
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
fear_keys = {"f05", "f06", "f10"}  # 공포성: 점수 높을수록 공포 -> 탐욕점수는 100-pct

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

# ---------------- heatmap bins ----------------
X_ORDER = ["7day 하락", "5day 하락", "3day 하락", "보합", "3day 상승", "5day 상승", "7day 상승"]

def classify_x_k200(ret3, ret5, ret7) -> str:
    thr = 0.0
    if pd.notna(ret7) and ret7 < -thr: return "7day 하락"
    if pd.notna(ret5) and ret5 < -thr: return "5day 하락"
    if pd.notna(ret3) and ret3 < -thr: return "3day 하락"
    if pd.notna(ret3) and abs(ret3) <= thr: return "보합"
    if pd.notna(ret3) and ret3 > thr: return "3day 상승"
    if pd.notna(ret5) and ret5 > thr: return "5day 상승"
    if pd.notna(ret7) and ret7 > thr: return "7day 상승"
    return "보합"

def build_heatmaps(df: pd.DataFrame, years=10):
    cutoff = df["date"].max() - pd.Timedelta(days=365 * years)
    sub = df[df["date"] >= cutoff].copy()
    sub["xbin"] = sub.apply(lambda r: classify_x_k200(r["k200_ret_3d"], r["k200_ret_5d"], r["k200_ret_7d"]), axis=1)
    sub["ybin"] = sub["bucket_5pt"]
    hm_ret = sub.groupby(["ybin", "xbin"])["k200_fwd_10d_return"].mean().reset_index()
    hm_win = sub.groupby(["ybin", "xbin"])["k200_fwd_10d_win"].mean().reset_index()
    ret_pv = hm_ret.pivot(index="ybin", columns="xbin", values="k200_fwd_10d_return").reindex(columns=X_ORDER).sort_index(ascending=False)
    win_pv = hm_win.pivot(index="ybin", columns="xbin", values="k200_fwd_10d_win").reindex(columns=X_ORDER).sort_index(ascending=False)
    return ret_pv, win_pv, sub

def plotly_div(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def fmt_num(x): return "-" if pd.isna(x) else f"{x:.2f}"
def fmt_pct(x): return "-" if pd.isna(x) else f"{x*100:.2f}%"

def make_index_fig(df):
    s = df.dropna(subset=["index_score_total"]).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["date"], y=s["index_score_total"], mode="lines", name="Gomtang Index"))
    fig.update_layout(height=360, margin=dict(l=30, r=20, t=20, b=30), yaxis_title="Score (0~100)")
    return plotly_div(fig)

def make_components_fig(df):
    fig = go.Figure()
    for col, name in [
        ("f01_score", "① Momentum"),
        ("f02_score", "② Strength"),
        ("f03_score", "③ Breadth"),
        ("f05_score", "⑤ CreditSpread(공포성)"),
        ("f06_score", "⑥ RealizedVol(공포성)"),
        ("f07_score", "⑦ SafeHaven"),
        ("f08_score", "⑧ ForeignNetBuy"),
        ("f10_score", "⑩ FXVol(공포성)"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines", name=name))
    fig.update_layout(height=360, margin=dict(l=30, r=20, t=20, b=30), yaxis_title="Percentile (0~100)")
    return plotly_div(fig)

def make_heatmap(mat: pd.DataFrame, title: str):
    fig = go.Figure(data=go.Heatmap(z=mat.values, x=mat.columns.astype(str), y=mat.index.astype(str), colorscale="RdYlGn"))
    fig.update_layout(height=420, margin=dict(l=40, r=20, t=30, b=40), title=title)
    return fig

def annotate_today(fig, xbin, ybin):
    if xbin is None or ybin is None:
        return fig
    fig.add_trace(go.Scatter(
        x=[xbin], y=[str(int(ybin))],
        mode="markers+text",
        marker=dict(size=14, color="black"),
        text=["TODAY"], textposition="top center",
        name="Today"
    ))
    return fig

HTML = Template(r"""
<!doctype html><html lang="ko"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>한국 곰탕 지수 리포트</title>
<link rel="stylesheet" href="assets/style.css"/>
</head><body>
<header class="topbar"><div class="wrap">
  <div class="brand"><div class="title">한국 곰탕 지수</div><div class="subtitle">Daily Report (자동 업데이트)</div></div>
  <div class="meta"><div>업데이트(UTC): <b>{{ updated }}</b></div><div>데이터 최신일: <b>{{ latest }}</b></div></div>
</div></header>
<main class="wrap">
<section class="grid cards">
  <div class="card"><div class="k">오늘의 지수</div><div class="v">{{ score }}</div><div class="s">구간: <b>{{ bucket }}</b> (높을수록 탐욕)</div></div>
  <div class="card"><div class="k">지수 추세</div><div class="s">3일: <b>{{ chg3 }}</b> / 5일: <b>{{ chg5 }}</b> / 7일: <b>{{ chg7 }}</b></div></div>
  <div class="card"><div class="k">KOSPI200 수익률(프록시: 069500)</div><div class="s">3일: <b>{{ k3 }}</b> / 5일: <b>{{ k5 }}</b> / 7일: <b>{{ k7 }}</b></div></div>
  <div class="card"><div class="k">상태</div><div class="s"><b>{{ status }}</b></div><div class="mini muted">{{ status2 }}</div></div>
</section>

<section class="card block"><h2>지수 라인차트</h2>{{ fig_index | safe }}</section>
<section class="card block"><h2>구성요소(팩터 점수) 라인차트</h2>{{ fig_comp | safe }}</section>

<section class="grid heatmaps">
  <section class="card block"><h2>히트맵1: 10일 후 KOSPI200 평균 수익률</h2>{{ hm1 | safe }}</section>
  <section class="card block"><h2>히트맵2: 10일 후 KOSPI200 승률</h2>{{ hm2 | safe }}</section>
</section>

<footer class="footer">
  <div class="muted">
    ⑤ data.go.kr 소매채권수익률요약(필드 basDt/crdtSc/ctg/bnfRt) · ⑩ ECOS(USD/KRW) 기반.
    [Source](https://www.genspark.ai/api/files/s/S7VQug0I)
  </div>
</footer>
</main></body></html>
""")

def main():
    today = datetime.utcnow().date()
    end_s = yyyymmdd(today)
    begin_recent_s = yyyymmdd(today - timedelta(days=cfg.REFETCH_DAYS))
    start_long_s = yyyymmdd(today - timedelta(days=365 * 12))

    # 1) K200 (프록시) — 없으면 비활성
    k200 = fetch_kospi200_ohlcv(start_long_s, end_s)
    k200_ok = (not k200.empty) and ("k200_close" in k200.columns) and (k200["k200_close"].dropna().size > 10)

    # 2) ⑤/⑩ long window로 점수 안정화
    f05 = fetch_f05(start_long_s, end_s)
    f10 = fetch_f10(start_long_s, end_s)
    f05_raw = build_f05_raw(f05)
    f10_raw = build_f10_raw(f10)

    # 3) pykrx factors (외국인 순매수는 버전 차이로 실패 가능하므로 안전 처리)
    f01 = factor1_momentum(k200).rename(columns={0: "f01_raw"}) if k200_ok else pd.DataFrame(columns=["date", "f01_raw"])
    f06 = factor6_volatility(k200).rename(columns={0: "f06_raw"}) if k200_ok else pd.DataFrame(columns=["date", "f06_raw"])
    try:
        f08 = factor8_foreign_netbuy(start_long_s, end_s)
        if "f08_raw" not in f08.columns:
            # 혹시 컬럼명이 다르면 안전하게 비움
            f08 = pd.DataFrame(columns=["date", "f08_raw"])
    except Exception:
        f08 = pd.DataFrame(columns=["date", "f08_raw"])

    # Strength/Breadth는 느릴 수 있으니 실패해도 파이프라인이 죽지 않게
    try:
        f02 = factor2_strength(start_long_s, end_s)
    except Exception:
        f02 = pd.DataFrame(columns=["date", "f02_raw"])
    try:
        f03 = factor3_breadth(start_long_s, end_s)
    except Exception:
        f03 = pd.DataFrame(columns=["date", "f03_raw"])

    # SafeHaven은 K200/FX 모두 필요
    try:
        f07 = factor7_safe_haven(k200, f10).rename(columns={0: "f07_raw"}) if k200_ok and (not f10.empty) else pd.DataFrame(columns=["date", "f07_raw"])
    except Exception:
        f07 = pd.DataFrame(columns=["date", "f07_raw"])

    # 4) merge (base = k200, 없으면 date 축을 f10으로)
    if k200_ok:
        base = k200.copy()
    else:
        base = f10[["date"]].drop_duplicates().copy() if not f10.empty else pd.DataFrame(columns=["date"])

    for dfx in [f01, f02, f03, f05_raw, f06, f07, f08, f10_raw]:
        if dfx is None or dfx.empty:
            continue
        base = pd.merge(base, dfx, on="date", how="outer")

    base = base.sort_values("date").reset_index(drop=True)

    # 5) K200 derived (K200 없으면 NaN)
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

    # 6) score
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

    # 8) report (K200 없으면 히트맵 비활성)
    df = base.dropna(subset=["index_score_total"]).copy()
    if df.empty:
        status = "데이터 부족(점수 산출 전)"
        status2 = "backfill을 먼저 실행하거나 기간을 늘려주세요."
        latest = base.iloc[-1] if not base.empty else None
        latest_date = "-" if latest is None else str(latest["date"].date())
        fig_index = "<div class='muted'>데이터 부족</div>"
        fig_comp = "<div class='muted'>데이터 부족</div>"
        hm1 = "<div class='muted'>K200 데이터 부족으로 히트맵 생성 불가</div>"
        hm2 = "<div class='muted'>K200 데이터 부족으로 히트맵 생성 불가</div>"
        score = "-"
        bucket = "-"
        chg3 = chg5 = chg7 = "-"
        k3 = k5 = k7 = "-"
    else:
        latest = df.iloc[-1]
        latest_date = str(latest["date"].date())
        score = fmt_num(latest["index_score_total"])
        b = latest["bucket_5pt"]
        bucket = f"{int(b)}~{int(b)+5}" if pd.notna(b) else "-"
        chg3 = fmt_num(latest["index_chg_3d"])
        chg5 = fmt_num(latest["index_chg_5d"])
        chg7 = fmt_num(latest["index_chg_7d"])
        k3 = fmt_pct(latest["k200_ret_3d"])
        k5 = fmt_pct(latest["k200_ret_5d"])
        k7 = fmt_pct(latest["k200_ret_7d"])
        fig_index = make_index_fig(df)
        fig_comp = make_components_fig(df)

        if base["k200_fwd_10d_return"].notna().sum() > 200:
            ret_pv, win_pv, _sub = build_heatmaps(df, years=10)
            hm1_fig = make_heatmap(ret_pv, "10일 후 평균 수익률")
            hm2_fig = make_heatmap(win_pv, "10일 후 승률")
            xbin = classify_x_k200(latest["k200_ret_3d"], latest["k200_ret_5d"], latest["k200_ret_7d"])
            ybin = int(b) if pd.notna(b) else None
            hm1_fig = annotate_today(hm1_fig, xbin, ybin)
            hm2_fig = annotate_today(hm2_fig, xbin, ybin)
            hm1 = plotly_div(hm1_fig)
            hm2 = plotly_div(hm2_fig)
            status = "정상"
            status2 = "K200 기반 히트맵/장기통계 활성"
        else:
            hm1 = "<div class='muted'>K200(프록시) 데이터가 부족해 히트맵을 생성하지 않습니다.</div>"
            hm2 = "<div class='muted'>K200(프록시) 데이터가 부족해 히트맵을 생성하지 않습니다.</div>"
            status = "부분 정상"
            status2 = "히트맵 비활성(데이터 부족)"

    html = HTML.render(
        updated=datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        latest=latest_date,
        score=score,
        bucket=bucket,
        chg3=chg3, chg5=chg5, chg7=chg7,
        k3=k3, k5=k5, k7=k7,
        status=status, status2=status2,
        fig_index=fig_index, fig_comp=fig_comp,
        hm1=hm1, hm2=hm2,
    )
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    print("[daily_update] OK: docs/index.html 생성 완료")

if __name__ == "__main__":
    main()
