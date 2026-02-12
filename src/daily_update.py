# -*- coding: utf-8 -*-
import os, math
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
    ROLLING_DAYS: int = 252*5
    MIN_OBS: int = 252

    # ⑤
    F05_CTG_3Y: str = "2년~3년미만"
    F05_GRADE_HI: str = "AA-"
    F05_GRADE_LO: str = "A-"

    # ⑩
    ECOS_STAT_CODE_USDKRW: str = "731Y003"
    ECOS_CYCLE: str = "D"
    ECOS_ITEM_USDKRW: str = "0000003"

    # 가중치(⑨ 제외하고 재정규화할 것)
    W = {
        "f01": 0.10,
        "f02": 0.10,
        "f03": 0.10,
        "f04": 0.10,  # Put/Call은 아직 미구현(나중에 KRX)
        "f05": 0.05,
        "f06": 0.125,
        "f07": 0.10,
        "f08": 0.10,
        "f09": 0.0,   # 이번 단계에서 제외
        "f10": 0.10,
    }

cfg = CFG()

def yyyymmdd(d: date) -> str: return d.strftime("%Y%m%d")

def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

def save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def upsert(old: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if old.empty: out = new.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)
        out = out.drop_duplicates(subset=keys, keep="last")
    return out.sort_values(keys).reset_index(drop=True)

def rolling_percentile(series: pd.Series, window: int, min_obs: int) -> pd.Series:
    x = series.astype(float)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        start = max(0, i-window+1)
        w = x.iloc[start:i+1].dropna().values
        if len(w) < min_obs or np.isnan(x.iloc[i]): continue
        out[i] = 100.0 * (w <= x.iloc[i]).mean()
    return pd.Series(out, index=series.index)

# ---------------- ⑤ fetch ----------------
def fetch_f05(begin: str, end: str) -> pd.DataFrame:
    base = "https://apis.data.go.kr/1160100/service/GetBondInfoService/getBondSecurityBenefitRate"
    service_key = os.environ.get("SERVICE_KEY","").strip()
    if not service_key: raise RuntimeError("SERVICE_KEY 비어있음")
    params = dict(serviceKey=service_key, resultType="json", numOfRows=3000, pageNo=1, beginBasDt=begin, endBasDt=end)
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    body = r.json()["response"]["body"]
    items = body.get("items", [])
    if isinstance(items, dict): items = items.get("item", [])
    df = pd.DataFrame(items)
    if df.empty: return df
    df = df.rename(columns={"basDt":"date","crdtSc":"grade","ctg":"bucket","bnfRt":"yield"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    return df[["date","grade","bucket","yield"]].dropna(subset=["date"])

def build_factor5_spread(f05: pd.DataFrame) -> pd.DataFrame:
    df = f05.copy()
    df = df[df["bucket"].astype(str) == cfg.F05_CTG_3Y]
    hi = df[df["grade"].astype(str) == cfg.F05_GRADE_HI][["date","yield"]].rename(columns={"yield":"y_hi"})
    lo = df[df["grade"].astype(str) == cfg.F05_GRADE_LO][["date","yield"]].rename(columns={"yield":"y_lo"})
    m = pd.merge(hi, lo, on="date", how="inner")
    m["f05_raw"] = m["y_hi"] - m["y_lo"]
    return m[["date","f05_raw"]]

# ---------------- ⑩ fetch ----------------
def fetch_f10(begin: str, end: str) -> pd.DataFrame:
    ecos_key = os.environ.get("ECOS_KEY","").strip()
    if not ecos_key: raise RuntimeError("ECOS_KEY 비어있음")
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{ecos_key}/json/kr/1/100000/{cfg.ECOS_STAT_CODE_USDKRW}/{cfg.ECOS_CYCLE}/{begin}/{end}/{cfg.ECOS_ITEM_USDKRW}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    rows = r.json().get("StatisticSearch",{}).get("row",[])
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["date"] = pd.to_datetime(df["TIME"], format="%Y%m%d", errors="coerce")
    df["usdkrw"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    return df[["date","usdkrw"]].dropna(subset=["date"])

def build_factor10_vol(f10: pd.DataFrame) -> pd.DataFrame:
    df = f10.sort_values("date").reset_index(drop=True).copy()
    df["ret"] = np.log(df["usdkrw"] / df["usdkrw"].shift(1))
    df["f10_raw"] = df["ret"].rolling(20).std() * math.sqrt(252)
    return df[["date","f10_raw"]]

# ---------------- Heatmap bins ----------------
X_ORDER = ["7day 하락","5day 하락","3day 하락","보합","3day 상승","5day 상승","7day 상승"]
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

def forward_return(s: pd.Series, n: int) -> pd.Series:
    s = s.astype(float)
    return s.shift(-n)/s - 1.0
def forward_win(s: pd.Series, n: int) -> pd.Series:
    return (forward_return(s,n) > 0).astype(float)

def build_heatmaps(df: pd.DataFrame, years=10):
    cutoff = df["date"].max() - pd.Timedelta(days=365*years)
    sub = df[df["date"] >= cutoff].copy()
    sub["xbin"] = sub.apply(lambda r: classify_x_k200(r["k200_ret_3d"], r["k200_ret_5d"], r["k200_ret_7d"]), axis=1)
    sub["ybin"] = sub["bucket_5pt"]
    hm_ret = sub.groupby(["ybin","xbin"])["k200_fwd_10d_return"].mean().reset_index()
    hm_win = sub.groupby(["ybin","xbin"])["k200_fwd_10d_win"].mean().reset_index()
    ret_pv = hm_ret.pivot(index="ybin", columns="xbin", values="k200_fwd_10d_return").reindex(columns=X_ORDER).sort_index(ascending=False)
    win_pv = hm_win.pivot(index="ybin", columns="xbin", values="k200_fwd_10d_win").reindex(columns=X_ORDER).sort_index(ascending=False)
    return ret_pv, win_pv, sub

def plotly_div(fig): return fig.to_html(full_html=False, include_plotlyjs="cdn")
def make_line(df, ycol, name):
    s = df.dropna(subset=[ycol]).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["date"], y=s[ycol], mode="lines", name=name))
    fig.update_layout(height=360, margin=dict(l=30,r=20,t=20,b=30))
    return plotly_div(fig)

def make_components(df):
    fig = go.Figure()
    for col, name in [
        ("f01_score","① Momentum"),
        ("f02_score","② Strength"),
        ("f03_score","③ Breadth"),
        ("f05_score","⑤ CreditSpread(공포성)"),
        ("f06_score","⑥ Volatility(공포성)"),
        ("f07_score","⑦ SafeHaven"),
        ("f08_score","⑧ ForeignNetBuy"),
        ("f10_score","⑩ FXVol(공포성)"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines", name=name))
    fig.update_layout(height=360, margin=dict(l=30,r=20,t=20,b=30), yaxis_title="Percentile Score (0~100)")
    return plotly_div(fig)

def make_heatmap(mat, title):
    z = mat.values
    y = mat.index.astype(str).tolist()
    x = mat.columns.astype(str).tolist()
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale="RdYlGn"))
    fig.update_layout(height=420, margin=dict(l=40,r=20,t=20,b=40), title=title)
    return fig

def annotate_today(fig, xbin, ybin):
    if xbin is None or ybin is None: return fig
    fig.add_trace(go.Scatter(x=[xbin], y=[str(int(ybin))], mode="markers+text",
                             marker=dict(size=14, color="black"), text=["TODAY"], textposition="top center",
                             name="Today"))
    return fig

def fmt_num(x): return "-" if pd.isna(x) else f"{x:.2f}"
def fmt_pct(x): return "-" if pd.isna(x) else f"{x*100:.2f}%"

HTML_TMPL = Template(r"""
<!doctype html><html lang="ko"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>한국 곰탕 지수 리포트</title><link rel="stylesheet" href="assets/style.css"/>
</head><body>
<header class="topbar"><div class="wrap">
  <div class="brand"><div class="title">한국 곰탕 지수</div><div class="subtitle">Daily Report (자동 업데이트)</div></div>
  <div class="meta"><div>업데이트(UTC): <b>{{ updated_utc }}</b></div><div>데이터 최신일: <b>{{ latest_date }}</b></div></div>
</div></header>

<main class="wrap">
<section class="grid cards">
  <div class="card"><div class="k">오늘의 지수</div><div class="v">{{ today_score }}</div><div class="s">구간: <b>{{ today_bucket }}</b></div></div>
  <div class="card"><div class="k">지수 추세</div><div class="s">3일: <b>{{ idx3 }}</b> / 5일: <b>{{ idx5 }}</b> / 7일: <b>{{ idx7 }}</b></div></div>
  <div class="card"><div class="k">KOSPI200 수익률</div><div class="s">3일: <b>{{ k3 }}</b> / 5일: <b>{{ k5 }}</b> / 7일: <b>{{ k7 }}</b></div></div>
  <div class="card"><div class="k">데이터 기반 원인(전일 대비)</div><div class="s"><b>{{ driver }}</b></div><div class="mini muted">|Δscore| 큰 팩터부터</div></div>
</section>

<section class="card block"><h2>지수 라인차트</h2>{{ fig_index | safe }}</section>
<section class="card block"><h2>구성요소(팩터 점수) 라인차트</h2>{{ fig_comp | safe }}</section>

<section class="grid heatmaps">
  <section class="card block"><h2>히트맵1: 10일 후 KOSPI200 평균 수익률</h2>{{ hm1 | safe }}</section>
  <section class="card block"><h2>히트맵2: 10일 후 KOSPI200 승률</h2>{{ hm2 | safe }}</section>
</section>

<section class="card block">
<h2>오늘 위치한 히트맵 셀의 장기 통계(최근 10년~)</h2>
<div class="grid two">
  <div class="stat"><div class="k">오늘 X 상태</div><div class="v">{{ xbin }}</div></div>
  <div class="stat"><div class="k">오늘 Y 구간</div><div class="v">{{ ybin }}</div></div>
  <div class="stat"><div class="k">표본 수</div><div class="v">{{ n }}</div></div>
  <div class="stat"><div class="k">10일 후 평균 수익률</div><div class="v">{{ retm }}</div></div>
  <div class="stat"><div class="k">10일 후 승률</div><div class="v">{{ win }}</div></div>
</div>
</section>

<footer class="footer"><div class="muted">
⑤ data.go.kr 소매채권수익률요약(필드 basDt/crdtSc/ctg/bnfRt) 및 ⑩ ECOS 환율 기반. [Source](https://www.genspark.ai/api/files/s/S7VQug0I) [Source](https://www.genspark.ai/api/files/s/QCBnq072)
</div></footer>
</main></body></html>
""")

def driver_text(df):
    if len(df) < 2: return "데이터 부족"
    a, b = df.iloc[-2], df.iloc[-1]
    parts=[]
    for k in ["f01_score","f02_score","f03_score","f05_score","f06_score","f07_score","f08_score","f10_score"]:
        if pd.notna(a.get(k)) and pd.notna(b.get(k)):
            parts.append((k, float(b[k]-a[k])))
    parts=sorted(parts, key=lambda x: abs(x[1]), reverse=True)[:4]
    return ", ".join([f"{k} {v:+.1f}" for k,v in parts]) if parts else "데이터 부족"

def main():
    # 기간 설정
    today = datetime.utcnow().date()
    begin_recent = today - timedelta(days=cfg.REFETCH_DAYS)
    begin_s, end_s = yyyymmdd(begin_recent), yyyymmdd(today)

    # ⑤/⑩ 저장소 로드 & 업데이트
    f05_old = load_parquet(DATA_DIR/"f05.parquet")
    f10_old = load_parquet(DATA_DIR/"f10.parquet")

    f05_new = fetch_f05(begin_s, end_s)
    f10_new = fetch_f10(begin_s, end_s)

    f05_all = upsert(f05_old, f05_new, ["date","grade","bucket"]) if not f05_new.empty else f05_old
    f10_all = upsert(f10_old, f10_new, ["date"]) if not f10_new.empty else f10_old

    if not f05_all.empty: save_parquet(f05_all, DATA_DIR/"f05.parquet")
    if not f10_all.empty: save_parquet(f10_all, DATA_DIR/"f10.parquet")

    # pykrx는 히트맵 위해 넉넉히(최근 12년) 가져오기
    start_long = (today - timedelta(days=365*12))
    start_s = yyyymmdd(start_long)

    k200 = fetch_kospi200_ohlcv(start_s, end_s)  # date, k200_close
    usd = f10_all.copy()
    usd = usd.sort_values("date").dropna()

    # 팩터 raw 생성
    f01 = factor1_momentum(k200).rename(columns={0:"f01_raw"})
    f02 = factor2_strength(start_s, end_s)
    f03 = factor3_breadth(start_s, end_s)
    f05 = build_factor5_spread(f05_all)
    f06 = factor6_volatility(k200)
    f07 = factor7_safe_haven(k200, usd)
    f08 = factor8_foreign_netbuy(start_s, end_s)
    f10 = build_factor10_vol(usd)

    # 결합(outer)
    dfs=[]
    for dfx in [k200, f01, f02, f03, f05, f06, f07, f08, f10]:
        if dfx is None or dfx.empty: continue
        if "date" not in dfx.columns: continue
        dfs.append(dfx)

    base = dfs[0]
    for dfx in dfs[1:]:
        base = pd.merge(base, dfx, on="date", how="outer")

    base = base.sort_values("date").reset_index(drop=True)

    # K200 파생치
    base["k200_ret_3d"] = base["k200_close"].pct_change(3)
    base["k200_ret_5d"] = base["k200_close"].pct_change(5)
    base["k200_ret_7d"] = base["k200_close"].pct_change(7)
    base["k200_fwd_10d_return"] = forward_return(base["k200_close"], 10)
    base["k200_fwd_10d_win"] = forward_win(base["k200_close"], 10)

    # 점수(퍼센타일)
    for key in ["f01","f02","f03","f05","f06","f07","f08","f10"]:
        raw = f"{key}_raw"
        sc = f"{key}_score"
        if raw in base.columns:
            base[sc] = rolling_percentile(base[raw], cfg.ROLLING_DAYS, cfg.MIN_OBS)

    # 최종 지수(⑨ 제외 → 가중치 재정규화)
    # 공포성(⑤,⑥,⑩)은 탐욕 점수로 뒤집고, 나머지는 그대로
    def greed_score(row, k):
        v = row.get(f"{k}_score", np.nan)
        if pd.isna(v): return np.nan
        if k in ["f05","f06","f10"]:  # 공포성
            return 100 - v
        return v

    weights = {k:v for k,v in cfg.W.items() if v > 0 and k != "f09" and k != "f04"}
    wsum = sum(weights.values())

    base["index_score_total"] = np.nan
    for i in range(len(base)):
        row = base.iloc[i]
        acc=0.0; ok=0.0
        for k,w in weights.items():
            gs = greed_score(row, k)
            if pd.isna(gs): 
                continue
            acc += w*gs
            ok += w
        if ok > 0:
            base.at[i,"index_score_total"] = acc/ok  # ok로 재정규화(결측 대응)

    base["bucket_5pt"] = (base["index_score_total"] // 5 * 5).clip(0,95).astype("Int64")
    base["index_chg_3d"] = base["index_score_total"].diff(3)
    base["index_chg_5d"] = base["index_score_total"].diff(5)
    base["index_chg_7d"] = base["index_score_total"].diff(7)

    # 저장
    save_parquet(base, DATA_DIR/"index_daily.parquet")

    # 리포트
    df = base.dropna(subset=["index_score_total"]).copy()
    latest = df.iloc[-1]
    latest_date = str(latest["date"].date())
    today_score = latest["index_score_total"]
    today_bucket = latest["bucket_5pt"]

    ret_pv, win_pv, sub = build_heatmaps(df, years=10)
    today_xbin = classify_x_k200(latest["k200_ret_3d"], latest["k200_ret_5d"], latest["k200_ret_7d"])
    today_ybin = int(today_bucket) if pd.notna(today_bucket) else None

    hm1 = make_heatmap(ret_pv, "10일 후 평균 수익률")
    hm2 = make_heatmap(win_pv, "10일 후 승률")
    hm1 = annotate_today(hm1, today_xbin, today_ybin)
    hm2 = annotate_today(hm2, today_xbin, today_ybin)

    # 오늘 셀 통계
    cell = sub[(sub["xbin"]==today_xbin) & (sub["ybin"]==today_ybin)]
    n = int(len(cell))
    retm = cell["k200_fwd_10d_return"].mean()
    win = cell["k200_fwd_10d_win"].mean()

    html = HTML_TMPL.render(
        updated_utc=datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        latest_date=latest_date,
        today_score=fmt_num(today_score),
        today_bucket=f"{int(today_bucket)}~{int(today_bucket)+5}" if pd.notna(today_bucket) else "-",
        idx3=fmt_num(latest["index_chg_3d"]),
        idx5=fmt_num(latest["index_chg_5d"]),
        idx7=fmt_num(latest["index_chg_7d"]),
        k3=fmt_pct(latest["k200_ret_3d"]),
        k5=fmt_pct(latest["k200_ret_5d"]),
        k7=fmt_pct(latest["k200_ret_7d"]),
        driver=driver_text(df),
        fig_index=make_line(df, "index_score_total", "Gomtang Index"),
        fig_comp=make_components(df),
        hm1=plotly_div(hm1),
        hm2=plotly_div(hm2),
        xbin=today_xbin,
        ybin=f"{int(today_ybin)}~{int(today_ybin)+5}" if today_ybin is not None else "-",
        n=n,
        retm=fmt_pct(retm),
        win="-" if pd.isna(win) else f"{win*100:.1f}%",
    )
    (DOCS_DIR/"index.html").write_text(html, encoding="utf-8")

if __name__ == "__main__":
    main()
