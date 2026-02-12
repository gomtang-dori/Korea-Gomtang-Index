# -*- coding: utf-8 -*-
"""
daily_update.py
- 매일 실행: ⑤/⑨/⑩ 최근 14일 재조회 + 머지 + 5년 롤링 퍼센타일 재계산
- 리포트 생성: docs/index.html + docs/assets/style.css (CSS는 별도 파일)

실행:
  python src/daily_update.py
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

DATA_DIR = Path("data")
DOCS_DIR = Path("docs")
ASSET_DIR = DOCS_DIR / "assets"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
ASSET_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 설정 ----------
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

    W_F05: float = 0.05
    W_F09: float = 0.125
    W_F10: float = 0.10

cfg = CFG()

# ---------- 유틸 ----------
def yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")

def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

def save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

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

# ---------- Fetch ⑤ ----------
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
    df = df.rename(columns={"basDt":"date","crdtSc":"grade","ctg":"bucket","bnfRt":"yield"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    return df[["date","grade","bucket","yield"]].dropna(subset=["date"])

# ---------- Fetch ⑩ ----------
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

# ---------- Fetch ⑨ (TODO) ----------
def fetch_f09(begin: str, end: str) -> pd.DataFrame:
    service_key = os.environ.get("SERVICE_KEY", "").strip()
    if not service_key:
        raise RuntimeError("SERVICE_KEY 환경변수가 비어 있습니다.")

    # TODO: endpoint/컬럼명 본인 응답에 맞게 수정
    NUM_URL = "https://apis.data.go.kr/1160100/service/GetKofiaStatisticsInfoService/getGrantingOfCreditBalanceInfo"
    DEN_URL = "https://apis.data.go.kr/1160100/service/GetKofiaStatisticsInfoService/getSecuritiesMarketTotalCapitalInfo"
    NUM_VAL_COL = "crdtrRlngWth"
    DEN_VAL_COL = "intDpsAmnt"

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
            raise RuntimeError("⑨ 응답에서 날짜 컬럼(basDt/date)을 찾지 못했습니다.")

    if NUM_VAL_COL not in num.columns or DEN_VAL_COL not in den.columns:
        raise RuntimeError("⑨ 분자/분모 value 컬럼명을 응답에 맞게 수정해야 합니다.")

    num = num[["date", NUM_VAL_COL]].rename(columns={NUM_VAL_COL:"credit_balance"})
    den = den[["date", DEN_VAL_COL]].rename(columns={DEN_VAL_COL:"deposit"})
    out = pd.merge(num, den, on="date", how="inner")
    out["credit_balance"] = pd.to_numeric(out["credit_balance"], errors="coerce")
    out["deposit"] = pd.to_numeric(out["deposit"], errors="coerce")
    out["f09_raw"] = out["credit_balance"] / out["deposit"]
    return out.dropna(subset=["date"])[["date","credit_balance","deposit","f09_raw"]]

# ---------- KOSPI200: placeholder (수동 CSV 지원) ----------
def load_k200():
    # data/k200_manual.csv 를 지원: date,k200_close
    p = DATA_DIR / "k200_manual.csv"
    if not p.exists():
        p2 = DATA_DIR / "k200.parquet"
        if p2.exists():
            df = pd.read_parquet(p2)
            return df
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["k200_close"] = pd.to_numeric(df["k200_close"], errors="coerce")
    return df.dropna(subset=["date"])[["date","k200_close"]].sort_values("date").reset_index(drop=True)

# ---------- raw -> features ----------
def build_factor5_spread(f05: pd.DataFrame) -> pd.DataFrame:
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

def attach_k200_features(df: pd.DataFrame, k200: pd.DataFrame) -> pd.DataFrame:
    if k200.empty:
        for c in ["k200_close","k200_ret_3d","k200_ret_5d","k200_ret_7d","k200_fwd_10d_return","k200_fwd_10d_win"]:
            df[c] = np.nan
        return df

    out = pd.merge(df, k200, on="date", how="left")
    out = out.sort_values("date").reset_index(drop=True)
    out["k200_ret_3d"] = pct_change_n(out["k200_close"], 3)
    out["k200_ret_5d"] = pct_change_n(out["k200_close"], 5)
    out["k200_ret_7d"] = pct_change_n(out["k200_close"], 7)
    out["k200_fwd_10d_return"] = forward_return(out["k200_close"], 10)
    out["k200_fwd_10d_win"] = forward_win(out["k200_close"], 10)
    return out

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()

    df["f05_score"] = rolling_percentile(df["f05_raw"], cfg.ROLLING_DAYS, cfg.MIN_OBS)
    df["f09_score"] = rolling_percentile(df["f09_raw"], cfg.ROLLING_DAYS, cfg.MIN_OBS)
    df["f10_score"] = rolling_percentile(df["f10_raw"], cfg.ROLLING_DAYS, cfg.MIN_OBS)

    # 최종 점수 방향: "높을수록 탐욕"으로 통일
    df["f05_score_greed"] = 100 - df["f05_score"]  # 스프레드↑=공포↑ => 탐욕점수는 역변환
    df["f10_score_greed"] = 100 - df["f10_score"]  # 변동성↑=공포↑
    df["f09_score_greed"] = df["f09_score"]        # ⑨는 탐욕성으로 가정

    wsum = cfg.W_F05 + cfg.W_F09 + cfg.W_F10
    df["index_score_total"] = (
        cfg.W_F05 * df["f05_score_greed"] +
        cfg.W_F09 * df["f09_score_greed"] +
        cfg.W_F10 * df["f10_score_greed"]
    ) / wsum

    df["bucket_5pt"] = (df["index_score_total"] // 5 * 5).clip(0, 95).astype("Int64")

    # 지수 추세(3/5/7)
    df["index_chg_3d"] = df["index_score_total"].diff(3)
    df["index_chg_5d"] = df["index_score_total"].diff(5)
    df["index_chg_7d"] = df["index_score_total"].diff(7)
    return df

# ---------- 히트맵 계산 ----------
def classify_x_k200(ret3, ret5, ret7) -> str:
    # X축: 7day 하락, 5day 하락, 3day 하락, 보합, 3day 상승, 5day 상승, 7day 상승
    # 규칙(우선순위): 가장 긴 기간부터 판단
    thr = 0.000  # 보합 기준(0). 원하면 0.001 등으로 조정 가능
    if pd.notna(ret7) and ret7 < -thr: return "7day 하락"
    if pd.notna(ret5) and ret5 < -thr: return "5day 하락"
    if pd.notna(ret3) and ret3 < -thr: return "3day 하락"
    if pd.notna(ret3) and abs(ret3) <= thr: return "보합"
    if pd.notna(ret3) and ret3 > thr: return "3day 상승"
    if pd.notna(ret5) and ret5 > thr: return "5day 상승"
    if pd.notna(ret7) and ret7 > thr: return "7day 상승"
    return "보합"

X_ORDER = ["7day 하락","5day 하락","3day 하락","보합","3day 상승","5day 상승","7day 상승"]

def build_heatmaps(df: pd.DataFrame, years=10):
    # 최근 10년~ 데이터로 장기 통계 계산
    cutoff = df["date"].max() - pd.Timedelta(days=365*years)
    sub = df[df["date"] >= cutoff].copy()

    # X축 분류
    sub["xbin"] = sub.apply(lambda r: classify_x_k200(r.get("k200_ret_3d"), r.get("k200_ret_5d"), r.get("k200_ret_7d")), axis=1)

    # y축: 5점 단위
    sub["ybin"] = sub["bucket_5pt"]

    # 색깔 1: 10day 이후 kospi 상승률(여기선 K200 forward 10d return)
    hm_ret = sub.groupby(["ybin","xbin"])["k200_fwd_10d_return"].mean().reset_index()
    # 색깔 2: 10day 이후 kospi 승률
    hm_win = sub.groupby(["ybin","xbin"])["k200_fwd_10d_win"].mean().reset_index()

    # pivot
    ret_pv = hm_ret.pivot(index="ybin", columns="xbin", values="k200_fwd_10d_return").reindex(columns=X_ORDER)
    win_pv = hm_win.pivot(index="ybin", columns="xbin", values="k200_fwd_10d_win").reindex(columns=X_ORDER)

    # y축 정렬(높은 점수 위로 보이게)
    ret_pv = ret_pv.sort_index(ascending=False)
    win_pv = win_pv.sort_index(ascending=False)

    return ret_pv, win_pv, sub

def investment_opinion(today_score, cell_ret_mean, cell_win):
    """
    투자의견(간단 룰):
    - today_score(탐욕) 낮으면(=공포) 분할매수 성향
    - today_score 높으면(=과열) 보수적
    - 같은 구간이라도 해당 셀의 과거 10d 기대수익/승률이 낮으면 보수적으로
    """
    if pd.isna(today_score):
        return "데이터 부족(점수 산출 전) — 백필 완료 후 자동 생성됩니다."

    if today_score < 25:
        base = "공포 구간 — 분할매수 우위(리스크 관리 전제)"
    elif today_score < 45:
        base = "중립-공포 — 신중한 분할 접근"
    elif today_score < 65:
        base = "중립 — 전략 유지"
    elif today_score < 80:
        base = "탐욕 — 추격매수 자제, 리밸런싱 고려"
    else:
        base = "극단적 탐욕 — 방어적 비중 확대 고려"

    # 셀 통계로 미세조정
    adj = ""
    if pd.notna(cell_win) and cell_win < 0.45:
        adj = " (주의: 이 구간의 10영업일 후 승률이 낮은 편)"
    if pd.notna(cell_ret_mean) and cell_ret_mean < 0:
        adj = " (주의: 이 구간의 10영업일 후 기대수익률이 음수)"
    return base + adj

# ---------- 리포트(HTML) ----------
HTML_TMPL = Template(r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>한국 곰탕 지수 리포트</title>
  <link rel="stylesheet" href="assets/style.css"/>
</head>
<body>
  <header class="topbar">
    <div class="wrap">
      <div class="brand">
        <div class="title">한국 곰탕 지수</div>
        <div class="subtitle">Daily Report (GitHub Actions 자동 업데이트)</div>
      </div>
      <div class="meta">
        <div>업데이트(UTC): <b>{{ updated_utc }}</b></div>
        <div>데이터 최신일: <b>{{ latest_date }}</b></div>
      </div>
    </div>
  </header>

  <main class="wrap">
    <section class="grid cards">
      <div class="card">
        <div class="k">오늘의 지수</div>
        <div class="v">{{ today_score }}</div>
        <div class="s">구간: <b>{{ today_bucket }}</b> / (높을수록 탐욕)</div>
      </div>
      <div class="card">
        <div class="k">지수 추세</div>
        <div class="s">3일: <b>{{ idx_chg_3d }}</b> / 5일: <b>{{ idx_chg_5d }}</b> / 7일: <b>{{ idx_chg_7d }}</b></div>
        <div class="mini muted">최근 3/5/7 영업일 변화(점수)</div>
      </div>
      <div class="card">
        <div class="k">KOSPI200 수익률</div>
        <div class="s">3일: <b>{{ k200_ret_3d }}</b> / 5일: <b>{{ k200_ret_5d }}</b> / 7일: <b>{{ k200_ret_7d }}</b></div>
        <div class="mini muted">※ K200 데이터가 없으면 표시되지 않습니다</div>
      </div>
      <div class="card">
        <div class="k">오늘 투자의견</div>
        <div class="s"><b>{{ opinion }}</b></div>
        <div class="mini muted">근거: 점수 구간 + 동일 셀의 과거 10일 후 통계</div>
      </div>
    </section>

    <section class="card block">
      <h2>지수 라인차트</h2>
      {{ fig_index | safe }}
    </section>

    <section class="card block">
      <h2>지수 구성요소(컴포넌트) 라인차트</h2>
      {{ fig_components | safe }}
      <div class="mini muted">
        현재는 ⑤/⑨/⑩ 기반(부분지수)입니다. 나머지 팩터를 합치면 동일 방식으로 확장됩니다.
      </div>
    </section>

    <section class="grid heatmaps">
      <section class="card block">
        <h2>히트맵 1: 10영업일 후 KOSPI200 평균 수익률</h2>
        {{ fig_hm_ret | safe }}
        <div class="mini muted">X축: KOSPI200 3/5/7일 수익률 상태 / Y축: 지수 점수 구간(5점)</div>
      </section>

      <section class="card block">
        <h2>히트맵 2: 10영업일 후 KOSPI200 승률</h2>
        {{ fig_hm_win | safe }}
        <div class="mini muted">승률=10영업일 뒤 상승 확률</div>
      </section>
    </section>

    <section class="card block">
      <h2>오늘 위치한 히트맵 셀의 장기 통계(최근 {{ years }}년~)</h2>
      <div class="grid two">
        <div class="stat">
          <div class="k">오늘 X축 상태</div>
          <div class="v">{{ today_xbin }}</div>
        </div>
        <div class="stat">
          <div class="k">오늘 Y축 구간</div>
          <div class="v">{{ today_bucket }}</div>
        </div>
        <div class="stat">
          <div class="k">해당 셀 표본 수</div>
          <div class="v">{{ cell_n }}</div>
        </div>
        <div class="stat">
          <div class="k">10일 후 평균 수익률</div>
          <div class="v">{{ cell_ret_mean }}</div>
        </div>
        <div class="stat">
          <div class="k">10일 후 승률</div>
          <div class="v">{{ cell_win }}</div>
        </div>
        <div class="stat">
          <div class="k">10일 후 수익률(분포) 분위값</div>
          <div class="v">P25 {{ cell_ret_p25 }} / P50 {{ cell_ret_p50 }} / P75 {{ cell_ret_p75 }}</div>
        </div>
      </div>
    </section>

    <section class="card block">
      <h2>뉴스 기반 오늘 지수의 원인 추정(초기 버전)</h2>
      <p class="muted">
        현재는 “뉴스 본문 수집/요약”을 자동화하지 않고, 데이터 기반으로 원인을 추정합니다.
        (추후 뉴스 수집 소스가 확정되면 기사 요약을 결합할 수 있습니다.)
      </p>
      <ul>
        <li>오늘 지수 변화의 주요 요인(팩터 기여도): <b>{{ driver_text }}</b></li>
        <li>⑤(스프레드), ⑩(환율 변동성)이 상승하면 보통 ‘공포 성향’이 커져 지수를 낮춥니다.</li>
      </ul>
    </section>

    <footer class="footer">
      <div class="muted">
        데이터 소스: ⑤ data.go.kr 소매채권수익률요약(필드 basDt/crdtSc/ctg/bnfRt) · ⑩ ECOS StatisticSearch [Source](https://www.genspark.ai/api/files/s/S7VQug0I) [Source](https://ecos.bok.or.kr/api/)
      </div>
    </footer>
  </main>
</body>
</html>
""")

def plotly_div(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def make_index_fig(df: pd.DataFrame) -> str:
    sub = df.dropna(subset=["index_score_total"]).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["index_score_total"], mode="lines", name="Gomtang Index"))
    fig.update_layout(height=360, margin=dict(l=30,r=20,t=20,b=30), yaxis_title="Score (0~100)")
    return plotly_div(fig)

def make_components_fig(df: pd.DataFrame) -> str:
    sub = df.copy()
    fig = go.Figure()
    for col, name in [
        ("f05_score_greed","⑤ CreditSpread(탐욕점수)"),
        ("f09_score_greed","⑨ Credit/Deposit(탐욕점수)"),
        ("f10_score_greed","⑩ USD/KRW Vol(탐욕점수)"),
    ]:
        if col in sub.columns:
            fig.add_trace(go.Scatter(x=sub["date"], y=sub[col], mode="lines", name=name))
    fig.update_layout(height=360, margin=dict(l=30,r=20,t=20,b=30), yaxis_title="Score (0~100)")
    return plotly_div(fig)

def make_heatmap_fig(mat: pd.DataFrame, title: str, zfmt: str):
    # mat index: ybin (높은점수->위), columns: xbin order
    z = mat.values
    y = mat.index.astype(str).tolist()
    x = mat.columns.astype(str).tolist()

    hover = "%{y}점 / %{x}<br>값: %{z}<extra></extra>"
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale="RdYlGn", hovertemplate=hover))
    fig.update_layout(height=420, margin=dict(l=40,r=20,t=20,b=40))
    return fig

def annotate_today(fig, xbin, ybin):
    if xbin is None or ybin is None:
        return fig
    fig.add_trace(go.Scatter(
        x=[xbin], y=[str(int(ybin))],
        mode="markers+text",
        marker=dict(size=14, color="black"),
        text=["TODAY"],
        textposition="top center",
        name="Today"
    ))
    return fig

def driver_estimation(df: pd.DataFrame) -> str:
    # 전일 대비 변화에서 기여도(간단): 각 팩터 탐욕점수 변화량
    if len(df) < 2:
        return "데이터 부족"
    a = df.iloc[-2]
    b = df.iloc[-1]
    parts = []
    for k, label in [("f05_score_greed","⑤"),("f09_score_greed","⑨"),("f10_score_greed","⑩")]:
        if pd.notna(a.get(k)) and pd.notna(b.get(k)):
            parts.append((label, float(b[k]-a[k])))
    if not parts:
        return "데이터 부족"
    parts = sorted(parts, key=lambda x: abs(x[1]), reverse=True)
    return ", ".join([f"{lab} {chg:+.1f}" for lab, chg in parts[:3]])

def cell_long_stats(sub: pd.DataFrame, today_xbin: str, today_ybin: int):
    cell = sub[(sub["xbin"] == today_xbin) & (sub["ybin"] == today_ybin)].copy()
    if cell.empty:
        return dict(n=0, ret_mean=np.nan, win=np.nan, p25=np.nan, p50=np.nan, p75=np.nan)
    r = cell["k200_fwd_10d_return"].dropna()
    w = cell["k200_fwd_10d_win"].dropna()
    return dict(
        n=len(cell),
        ret_mean=float(r.mean()) if len(r) else np.nan,
        win=float(w.mean()) if len(w) else np.nan,
        p25=float(r.quantile(0.25)) if len(r) else np.nan,
        p50=float(r.quantile(0.50)) if len(r) else np.nan,
        p75=float(r.quantile(0.75)) if len(r) else np.nan,
    )

def fmt_pct(x):
    return "-" if pd.isna(x) else f"{x*100:.2f}%"

def fmt_num(x):
    return "-" if pd.isna(x) else f"{x:.2f}"

def main():
    print("[daily_update] 시작")
    today = datetime.utcnow().date()
    begin = today - timedelta(days=cfg.REFETCH_DAYS)
    begin_s, end_s = yyyymmdd(begin), yyyymmdd(today)

    # load existing
    f05_old = load_parquet(DATA_DIR / "f05.parquet")
    f09_old = load_parquet(DATA_DIR / "f09.parquet")
    f10_old = load_parquet(DATA_DIR / "f10.parquet")

    # fetch recent
    f05_new = fetch_f05(begin_s, end_s)
    f10_new = fetch_f10(begin_s, end_s)

    if not f05_new.empty:
        f05_all = upsert(f05_old, f05_new, ["date","grade","bucket"])
        save_parquet(f05_all, DATA_DIR / "f05.parquet")
    else:
        f05_all = f05_old

    if not f10_new.empty:
        f10_all = upsert(f10_old, f10_new, ["date"])
        save_parquet(f10_all, DATA_DIR / "f10.parquet")
    else:
        f10_all = f10_old

    # f09 recent (fail-safe)
    try:
        f09_new = fetch_f09(begin_s, end_s)
        if not f09_new.empty:
            f09_all = upsert(f09_old, f09_new, ["date"])
            save_parquet(f09_all, DATA_DIR / "f09.parquet")
        else:
            f09_all = f09_old
    except Exception as ex:
        print(f"[warn] f09 업데이트 실패(스킵): {ex}")
        f09_all = f09_old

    # build index parts
    f05_spread = build_factor5_spread(f05_all) if not f05_all.empty else pd.DataFrame()
    f10_vol = build_factor10_vol(f10_all) if not f10_all.empty else pd.DataFrame()
    f09_raw = f09_all[["date","f09_raw"]] if (not f09_all.empty and "f09_raw" in f09_all.columns) else pd.DataFrame()

    df = None
    for part in [f05_spread, f09_raw, f10_vol]:
        if part.empty:
            continue
        df = part if df is None else pd.merge(df, part, on="date", how="outer")
    if df is None or df.empty:
        raise RuntimeError("지수 계산용 데이터가 없습니다(⑤/⑩ 호출/필터 확인).")

    df = df.sort_values("date").reset_index(drop=True)

    # score/index
    df = compute_scores(df)

    # attach K200 features (수동/파케 지원)
    k200 = load_k200()
    df = attach_k200_features(df, k200)

    # save
    save_parquet(df, DATA_DIR / "index_daily.parquet")

    # ---------- 리포트 ----------
    latest_row = df.dropna(subset=["index_score_total"]).iloc[-1] if not df.dropna(subset=["index_score_total"]).empty else df.iloc[-1]
    latest_date = str(latest_row["date"].date())

    today_score = latest_row.get("index_score_total", np.nan)
    today_bucket = latest_row.get("bucket_5pt", pd.NA)
    idx_chg_3d = latest_row.get("index_chg_3d", np.nan)
    idx_chg_5d = latest_row.get("index_chg_5d", np.nan)
    idx_chg_7d = latest_row.get("index_chg_7d", np.nan)

    k200_ret_3d = latest_row.get("k200_ret_3d", np.nan)
    k200_ret_5d = latest_row.get("k200_ret_5d", np.nan)
    k200_ret_7d = latest_row.get("k200_ret_7d", np.nan)

    # 히트맵 만들기(최근 10년)
    years = 10
    if df["k200_close"].notna().sum() > 500:
        ret_pv, win_pv, sub = build_heatmaps(df, years=years)

        hm1 = make_heatmap_fig(ret_pv, "10d 후 수익률", "%")
        hm2 = make_heatmap_fig(win_pv, "10d 후 승률", "%")

        today_xbin = classify_x_k200(k200_ret_3d, k200_ret_5d, k200_ret_7d)
        today_ybin = int(today_bucket) if pd.notna(today_bucket) else None

        hm1 = annotate_today(hm1, today_xbin, today_ybin)
        hm2 = annotate_today(hm2, today_xbin, today_ybin)

        fig_hm_ret = plotly_div(hm1)
        fig_hm_win = plotly_div(hm2)

        stats = cell_long_stats(sub, today_xbin, today_ybin) if (today_xbin and today_ybin is not None) else dict(n=0, ret_mean=np.nan, win=np.nan, p25=np.nan, p50=np.nan, p75=np.nan)
        cell_ret_mean = stats["ret_mean"]
        cell_win = stats["win"]
        opinion = investment_opinion(today_score, cell_ret_mean, cell_win)
        cell_n = stats["n"]
        cell_ret_p25, cell_ret_p50, cell_ret_p75 = stats["p25"], stats["p50"], stats["p75"]
    else:
        # K200 데이터가 없으면 히트맵/셀통계 비활성
        fig_hm_ret = "<div class='muted'>KOSPI200 데이터가 없어 히트맵을 생성할 수 없습니다. data/k200_manual.csv 또는 KRX 연동을 설정하세요.</div>"
        fig_hm_win = "<div class='muted'>KOSPI200 데이터가 없어 히트맵을 생성할 수 없습니다.</div>"
        today_xbin = "N/A"
        cell_n = 0
        cell_ret_mean = np.nan
        cell_win = np.nan
        cell_ret_p25 = cell_ret_p50 = cell_ret_p75 = np.nan
        opinion = investment_opinion(today_score, np.nan, np.nan)

    fig_index = make_index_fig(df)
    fig_components = make_components_fig(df)
    driver_text = driver_estimation(df)

    html = HTML_TMPL.render(
        updated_utc=datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        latest_date=latest_date,
        today_score=fmt_num(today_score),
        today_bucket=f"{int(today_bucket)}~{int(today_bucket)+5}" if pd.notna(today_bucket) else "-",
        idx_chg_3d=fmt_num(idx_chg_3d),
        idx_chg_5d=fmt_num(idx_chg_5d),
        idx_chg_7d=fmt_num(idx_chg_7d),
        k200_ret_3d=fmt_pct(k200_ret_3d),
        k200_ret_5d=fmt_pct(k200_ret_5d),
        k200_ret_7d=fmt_pct(k200_ret_7d),
        opinion=opinion,
        fig_index=fig_index,
        fig_components=fig_components,
        fig_hm_ret=fig_hm_ret,
        fig_hm_win=fig_hm_win,
        years=years,
        today_xbin=today_xbin,
        cell_n=cell_n,
        cell_ret_mean=fmt_pct(cell_ret_mean),
        cell_win="-" if pd.isna(cell_win) else f"{cell_win*100:.1f}%",
        cell_ret_p25=fmt_pct(cell_ret_p25),
        cell_ret_p50=fmt_pct(cell_ret_p50),
        cell_ret_p75=fmt_pct(cell_ret_p75),
        driver_text=driver_text,
    )

    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    print("[daily_update] 완료: docs/index.html 갱신")

if __name__ == "__main__":
    main()
