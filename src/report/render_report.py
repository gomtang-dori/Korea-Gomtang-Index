# -*- coding: utf-8 -*-
"""
Gomtang Index Daily Report (v3: Overview → Factors → Heatmaps → Backtesting → Conclusion)

Confirmed specs:
- Heatmap X-axis state: GOMTANG trend (index_score_total Δ3/Δ5), flat threshold SCORE_FLAT_PTS=1.0 point
- Heatmap X labels: GOMTANG 5D↓, GOMTANG 3D↓, GOMTANG Flat, GOMTANG 3D↑, GOMTANG 5D↑
- Factor bands: percentile 20/40/60/80 (0-20/20-40/40-60/60-80/80-100)
- Greed-direction:
  * Higher = Greed: F01,F02,F03,F07,F08,F09
  * Lower  = Greed: F04,F05,F06,F10  (same percentile, display band flipped)
- Factor charts: name+desc, Today value, Percentile, Band, and SPEC lines at P20/P40/P60/P80
- Backtesting (within report window): bucket table/plot + Top20/Bot20 summary (fixed 20%)
- Mean-return heatmap clipped ±3%
- prob_up_10d: show last non-null value + date
- Opinion: ML 우선 (prob_up_10d >=0.6 BUY, <0.4 SELL, else HOLD). ML 결측 시: heatmap cell win_rate(>=60% BUY, <=40% SELL, else HOLD). (trend filter 미적용)
- Position guide (%): Option2 standard
  * BUY +10~+20, HOLD 0, SELL -10~-30
  * If Confidence Low → halve the guidance (BUY +5~+10, SELL -5~-15)
- Confidence:
  * High: ML exists AND current-cell n>=30
  * Medium: ML exists OR n>=15
  * Low: otherwise

Patch (A안):
- Use include_plotlyjs="cdn" everywhere to avoid "Plotly is not defined" without adding <head> script tag.
"""

import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template


# ---------------------------
# Env / formatting utils
# ---------------------------
def _env(key: str, default: str = "") -> str:
    v = os.getenv(key, default)
    return v if v is not None else default


def _fmt_int(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    try:
        return f"{int(x):,}"
    except Exception:
        return "-"


def _fmt_float(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


def _fmt_pct(x, nd=2, signed=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    try:
        v = float(x) * 100.0
        if signed:
            return f"{v:+.{nd}f}%"
        return f"{v:.{nd}f}%"
    except Exception:
        return "-"


def _last_valid(series: pd.Series):
    """Return (last_value, last_timestamp) for last non-null element, else (nan, NaT)."""
    if series is None or len(series) == 0:
        return (np.nan, pd.NaT)
    s = series.dropna()
    if len(s) == 0:
        return (np.nan, pd.NaT)
    idx = s.index[-1]
    return (s.iloc[-1], idx)


def _ret_over_n(close: pd.Series, n: int):
    if close is None or len(close) < n + 1:
        return np.nan
    try:
        return close.iloc[-1] / close.iloc[-(n + 1)] - 1.0
    except Exception:
        return np.nan


def _delta_over_n(x: pd.Series, n: int):
    if x is None or len(x) < n + 1:
        return np.nan
    try:
        return x.iloc[-1] - x.iloc[-(n + 1)]
    except Exception:
        return np.nan


def _bucket_5pt(score: float) -> str:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "-"
    lo = int(math.floor(score / 5.0) * 5)
    hi = lo + 5
    return f"{lo:02d}–{hi:02d}"


def _regime_label(score: float) -> str:
    # Fear < 40, Neutral 40~60, Greed > 60
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "-"
    if score < 40:
        return "Fear"
    if score <= 60:
        return "Neutral"
    return "Greed"


# ---------------------------
# Heatmap state labels
# ---------------------------
STATE_LABEL = {
    "5D decline": "GOMTANG 5D↓",
    "3D decline": "GOMTANG 3D↓",
    "Flat": "GOMTANG Flat",
    "3D rise": "GOMTANG 3D↑",
    "5D rise": "GOMTANG 5D↑",
}
STATE_ORDER_INTERNAL = ["5D decline", "3D decline", "Flat", "3D rise", "5D rise"]


# ---------------------------
# Factor meta + greed-direction
# ---------------------------
FACTOR_META = {
    "f01": ("Market Momentum", "KOSPI 200 / 125거래일 이동평균 - 1"),
    "f02": ("Stock Price Strength", "상승 종목 수 vs 하락 종목수"),
    "f03": ("BREATH", "ADR (20거래일 상승 종목수 / 20거래일 하락 종목수)"),
    "f04": ("Put/Call Options", "Put_Vol / Call_Vol"),
    "f05": ("Junk Bond Demand", "회사채 BBB- 수익률 − AA- 수익률"),
    "f06": ("Market Volatility", "VKOSPI (KOSPI200 변동성)"),
    "f07": ("Safe Haven Demand", "국고채 3년 수익률 변화 vs KOSPI 수익률 상대강도"),
    "f08": ("외국인 순매수 강도", "외국인 순매수 강도"),
    "f09": ("신용융자/예탁금", "신용융자 잔고 / 투자자예탁금"),
    "f10": ("원/달러 환율 변동성", "StdDev20(USD_KRW_Ret)"),
}
FACTOR_GREED_IS_HIGH = {
    "f01": True,
    "f02": True,
    "f03": True,
    "f07": True,
    "f08": True,
    "f09": True,
    "f04": False,
    "f05": False,
    "f06": False,
    "f10": False,
}


# ---------------------------
# Factor percentile bands (20/40/60/80)
# ---------------------------
BAND_THRESH = [0.20, 0.40, 0.60, 0.80]
BAND_NAMES = ["EXTREME FEAR", "FEAR", "NEUTRAL", "GREED", "EXTREME GREED"]
BAND_ORDER = BAND_NAMES[:]


def pct_to_band(p: float) -> str:
    if p < BAND_THRESH[0]:
        return BAND_NAMES[0]
    if p < BAND_THRESH[1]:
        return BAND_NAMES[1]
    if p < BAND_THRESH[2]:
        return BAND_NAMES[2]
    if p < BAND_THRESH[3]:
        return BAND_NAMES[3]
    return BAND_NAMES[4]


def flip_band(band: str) -> str:
    if band not in BAND_ORDER:
        return band
    return BAND_ORDER[::-1][BAND_ORDER.index(band)]


# ---------------------------
# Gomtang trend state (Δ3/Δ5 in points)
# ---------------------------
def _gomtang_state_5bins(d3: float, d5: float, flat_pts: float) -> str:
    if d5 <= -flat_pts:
        return "5D decline"
    if d5 >= flat_pts:
        return "5D rise"
    if d3 <= -flat_pts:
        return "3D decline"
    if d3 >= flat_pts:
        return "3D rise"
    return "Flat"


# ---------------------------
# Opinion / confidence / position guide
# ---------------------------
def _opinion_from_prob(prob: float):
    if prob is None or (isinstance(prob, float) and np.isnan(prob)):
        return None
    if prob >= 0.6:
        return "BUY"
    if prob < 0.4:
        return "SELL"
    return "HOLD"


def _opinion_from_win_rate(win_rate: float, n: int):
    """
    ML 결측 시: heatmap 현재 셀(win_rate) 기반 Action.
    - win_rate >= 0.60 -> BUY
    - win_rate <= 0.40 -> SELL
    - else -> HOLD
    (n==0 또는 win_rate NaN이면 HOLD)
    """
    if n is None or n <= 0:
        return "HOLD"
    if win_rate is None or (isinstance(win_rate, float) and np.isnan(win_rate)):
        return "HOLD"
    if win_rate >= 0.60:
        return "BUY"
    if win_rate <= 0.40:
        return "SELL"
    return "HOLD"


def _opinion_from_trend(delta3: float, delta5: float):
    if (delta3 is None or np.isnan(delta3)) or (delta5 is None or np.isnan(delta5)):
        return "HOLD"
    if delta3 >= 0 and delta5 >= 0:
        return "BUY"
    if delta3 <= 0 and delta5 <= 0:
        return "SELL"
    return "HOLD"


def _apply_trend_filter(opinion: str, delta3: float, delta5: float):
    if opinion not in ("BUY", "HOLD", "SELL"):
        return opinion
    if (delta3 is None or np.isnan(delta3)) or (delta5 is None or np.isnan(delta5)):
        return opinion

    if delta3 <= 0 and delta5 <= 0:
        if opinion == "BUY":
            return "HOLD"
        if opinion == "HOLD":
            return "SELL"
        return "SELL"

    if delta3 >= 0 and delta5 >= 0:
        if opinion == "SELL":
            return "HOLD"
        if opinion == "HOLD":
            return "BUY"
        return "BUY"

    return opinion


def _confidence_level(has_ml: bool, cell_n: int) -> str:
    if has_ml and cell_n >= 30:
        return "High"
    if has_ml or cell_n >= 15:
        return "Medium"
    return "Low"


def _position_guide(opinion: str, confidence: str) -> str:
    # Option2 standard: BUY +10~+20, HOLD 0, SELL -10~-30
    base = {
        "BUY": (10, 20),
        "HOLD": (0, 0),
        "SELL": (-30, -10),
    }
    if opinion not in base:
        return "-"

    lo, hi = base[opinion]

    # Low confidence -> half-size
    if confidence == "Low":
        lo = int(round(lo / 2))
        hi = int(round(hi / 2))

    if lo == 0 and hi == 0:
        return "0% (유지)"
    return f"{lo}% ~ {hi}%"


# ---------------------------
# Plot helpers
# ---------------------------
def _line_fig(df: pd.DataFrame, xcol: str, ycol: str, title: str):
    if ycol not in df.columns:
        return None
    d = df[[xcol, ycol]].dropna()
    if len(d) == 0:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=d[xcol], y=d[ycol], mode="lines", line=dict(width=2), name=ycol)
    )
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=30, r=20, t=40, b=30),
        template="plotly_white",
        xaxis_title="",
        yaxis_title="",
    )
    return fig


def _forward_return(close: pd.Series, n: int) -> pd.Series:
    return close.shift(-n) / close - 1.0


def _heatmap_fig(
    pivot_val: pd.DataFrame,
    pivot_n: pd.DataFrame,
    title: str,
    colorscale: str,
    zmid=None,
    zmin=None,
    zmax=None,
    is_percent: bool = False,
    highlight_xy=None,  # (x_label, y_label)
    cell_font_size: int = 16,
):
    if pivot_val is None or pivot_n is None:
        return None
    if pivot_val.shape[0] == 0 or pivot_val.shape[1] == 0:
        return None

    y_labels = list(pivot_val.index)
    x_labels = list(pivot_val.columns)

    z = pivot_val.values.astype(float)
    n = pivot_n.reindex(index=y_labels, columns=x_labels).values

    text = np.empty_like(z, dtype=object)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if np.isnan(z[i, j]):
                text[i, j] = ""
                continue

            if is_percent:
                main = f"{z[i, j]*100:,.1f}%"
            else:
                main = f"{z[i, j]*100:+,.2f}%"

            nij = n[i, j]
            nij_s = (
                "-"
                if (nij is None or (isinstance(nij, float) and np.isnan(nij)))
                else str(int(nij))
            )
            text[i, j] = (
                f"<b style='font-size:{cell_font_size}px'>{main}</b>"
                f"<br><span style='font-size:{max(11, cell_font_size-5)}px;color:#333'>n={nij_s}</span>"
            )

    hovertemplate = (
        "Bucket: %{y}<br>"
        "State: %{x}<br>"
        + ("Value: %{z:.1%}<br>" if is_percent else "Value: %{z:.2%}<br>")
        + "n=%{customdata}<extra></extra>"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            zmid=zmid,
            zmin=zmin,
            zmax=zmax,
            text=text,
            texttemplate="%{text}",
            textfont=dict(color="#111", size=cell_font_size),
            customdata=n,
            hovertemplate=hovertemplate,
            showscale=True,
            colorbar=dict(title="", len=0.85),
        )
    )

    fig.update_layout(
        title=title,
        height=560,
        margin=dict(l=45, r=20, t=55, b=40),
        template="plotly_white",
        xaxis_title="Gomtang trend state (Δ3/Δ5)",
        yaxis_title="Gomtang bucket (5pt)",
    )
    fig.update_xaxes(side="top")

    if highlight_xy and highlight_xy[0] in x_labels and highlight_xy[1] in y_labels:
        x0 = x_labels.index(highlight_xy[0]) - 0.5
        x1 = x_labels.index(highlight_xy[0]) + 0.5
        y0 = y_labels.index(highlight_xy[1]) - 0.5
        y1 = y_labels.index(highlight_xy[1]) + 0.5
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            line=dict(color="#111", width=3),
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )

    return fig


def _barline_bucket_fig(tbl: pd.DataFrame, title: str):
    if tbl is None or len(tbl) == 0:
        return None

    x = tbl.index.tolist()
    avg = tbl["avg"].values.astype(float)
    win = tbl["win"].values.astype(float)
    n = tbl["n"].values.astype(float)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=avg,
            name="Mean 10D return",
            marker_color="#3b82f6",
            hovertemplate="Bucket=%{x}<br>Mean=%{y:.2%}<br>n=%{customdata}<extra></extra>",
            customdata=n,
            opacity=0.85,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=win,
            name="Win-rate",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#10b981", width=2),
            hovertemplate="Bucket=%{x}<br>Win=%{y:.1%}<br>n=%{customdata}<extra></extra>",
            customdata=n,
        )
    )

    fig.update_layout(
        title=title,
        height=360,
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=80),
        xaxis=dict(tickangle=-45),
        yaxis=dict(title="Mean return", tickformat=".1%"),
        yaxis2=dict(
            title="Win-rate",
            overlaying="y",
            side="right",
            tickformat=".0%",
            range=[0, 1],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _group_cell_stats(
    df: pd.DataFrame, state_col: str, bucket_col: str, fwd_col: str, state: str, bucket: str
):
    d = df[(df[state_col] == state) & (df[bucket_col] == bucket)][fwd_col].dropna()
    if len(d) == 0:
        return {"n": 0, "win": np.nan, "avg": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan}
    return {
        "n": int(d.shape[0]),
        "win": float((d > 0).mean()),
        "avg": float(d.mean()),
        "median": float(d.median()),
        "q1": float(d.quantile(0.25)),
        "q3": float(d.quantile(0.75)),
    }


def _regime_forward_stats(df: pd.DataFrame, score_col: str, close_col: str, horizon: int, regime: str):
    d = df[[score_col, close_col]].dropna().copy()
    if len(d) == 0:
        return {"n": 0, "win": np.nan, "avg": np.nan}

    d["regime"] = d[score_col].apply(_regime_label)
    d[f"fwd{horizon}"] = _forward_return(d[close_col], horizon)

    x = d[d["regime"] == regime][f"fwd{horizon}"].dropna()
    if len(x) == 0:
        return {"n": 0, "win": np.nan, "avg": np.nan}

    return {"n": int(len(x)), "win": float((x > 0).mean()), "avg": float(x.mean())}


# ---------------------------
# HTML Template (v3 sections)
# ---------------------------
HTML_TMPL = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ title }}</title>
  <style>
    body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,"Apple SD Gothic Neo","Noto Sans KR","Malgun Gothic",sans-serif;
           margin: 18px; color: #111; background: #fff; }
    h1 { font-size: 20px; margin: 0 0 10px 0; }
    .sub { color:#666; font-size: 12px; margin-bottom: 14px; line-height: 1.4; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(220px, 1fr)); gap: 10px; }
    .card { border: 1px solid #e6e6e6; border-radius: 12px; padding: 12px 12px; background: #fff; }
    .k { color:#666; font-size: 12px; margin-bottom: 4px; }
    .v { font-size: 20px; font-weight: 900; }
    .v2 { font-size: 14px; font-weight: 650; margin-top: 4px; }
    .warn { margin-top: 6px; color: #b00020; font-size: 12px; font-weight: 900; }
    .muted { color:#666; font-size: 12px; margin-top: 6px; line-height: 1.4; }
    .row { margin-top: 12px; display:grid; grid-template-columns: 1fr; gap: 12px; }
    .plot { border: 1px solid #eee; border-radius: 12px; padding: 8px; background:#fff; }
    .section { margin-top: 18px; }
    .sec-title { font-size: 16px; font-weight: 950; margin: 6px 0 10px; }
    .small { font-size: 12px; color:#444; }
    .pill { display:inline-block; padding:2px 8px; border:1px solid #eee; border-radius:999px; font-size:12px; margin-right:6px; background:#fafafa; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 8px; }
    th, td { border: 1px solid #eee; padding: 6px 8px; text-align: right; }
    th { background: #fafafa; }
    th:first-child, td:first-child { text-align: left; }
    a { color:#0b57d0; text-decoration:none; }
    a:hover { text-decoration:underline; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="sub">
    기준일: <b>{{ asof_date }}</b> · Lookback: {{ lookback_days }} days<br>
    <span class="pill">Trend: Δ3/Δ5 (Flat=1.0pt)</span>
    <span class="pill">Bands: 20/40/60/80 percentile</span>
    <span class="pill">Top/Bottom: 20%</span>
    · Guide: <a href="{{ guide_url }}" target="_blank">{{ guide_label }}</a>
  </div>

  <!-- A. OVERVIEW -->
  <div class="section">
    <div class="sec-title">A. 개요(Overview) — 오늘 무엇을 의미하나</div>
    <div class="grid">
      <div class="card">
        <div class="k">오늘의 곰탕지수</div>
        <div class="v">{{ gomtang_score }}</div>
        <div class="v2">Bucket: {{ gomtang_bucket }} · Regime: {{ regime }}</div>
        <div class="muted">Δ3: {{ gomtang_d3 }} · Δ5: {{ gomtang_d5 }} · Δ10: {{ gomtang_d10 }}</div>
        <div class="muted">Trend state: <b>{{ gomtang_state_label }}</b></div>
      </div>

      <div class="card">
        <div class="k">KOSPI (raw)</div>
        <div class="v">{{ kospi_close }}</div>
        <div class="muted">3D: {{ kospi_r3 }} · 5D: {{ kospi_r5 }} · 10D: {{ kospi_r10 }}</div>
      </div>

      <div class="card">
        <div class="k">ML: prob_up_10d</div>
        <div class="v">{{ prob_up_10d }}</div>
        <div class="muted">as of: {{ prob_asof }}</div>
        <div class="muted">Rule: BUY≥0.60 / SELL&lt;0.40 (else HOLD)</div>
      </div>

      <div class="card">
        <div class="k">오늘의 Action (BUY/HOLD/SELL + 비중 가이드)</div>
        <div class="v">{{ action }}</div>
        <div class="v2">Position guide: {{ position_guide }}</div>
        <div class="muted">Confidence: <b>{{ confidence }}</b> (ML={{ has_ml }}, cell_n={{ cell_n_plain }})</div>
        <div class="muted">{{ action_reason_1 }}</div>
        <div class="muted">{{ action_reason_2 }}</div>
        <div class="muted">{{ action_reason_3 }}</div>
        {% if confidence_note %}
          <div class="warn">{{ confidence_note }}</div>
        {% endif %}
      </div>
    </div>

    <div class="row">
      <div class="plot">{{ fig_gomtang|safe }}</div>
      <div class="plot">{{ fig_kospi|safe }}</div>
      <div class="plot">{{ fig_prob|safe }}</div>
    </div>
  </div>

  <!-- B. FACTORS -->
  <div class="section">
    <div class="sec-title">B. 팩터(Factors) — 왜 이런 점수가 나왔나</div>
    <div class="card">
      <div class="k">팩터 요약(오늘 밴드 기준)</div>
      <div class="muted">Extreme Greed: {{ factors_extreme_greed }}</div>
      <div class="muted">Extreme Fear: {{ factors_extreme_fear }}</div>
      <div class="muted small">* 밴드: 20/40/60/80 퍼센타일. 일부 팩터는 “낮을수록 탐욕”으로 표시 밴드가 반전됩니다.</div>
    </div>
    {% for fc in factor_cards %}
      <div class="plot">{{ fc|safe }}</div>
    {% endfor %}
  </div>

  <!-- C. HEATMAPS -->
  <div class="section">
    <div class="sec-title">C. 히트맵(Heatmaps) — 현재 조건에서 과거 평균적으로 어땠나</div>
    <div class="row">
      <div class="plot">{{ fig_hm_mean|safe }}</div>
      <div class="plot">{{ fig_hm_win|safe }}</div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="k">현재 셀 통계 ({{ cell_bucket }} × {{ cell_state_label }})</div>
        <div class="v2">n={{ cell_n }} · win={{ cell_win }} · avg={{ cell_avg }}</div>
        <div class="muted">median={{ cell_med }} · Q1/Q3={{ cell_q1 }}/{{ cell_q3 }}</div>
        {% if cell_warn %}
          <div class="warn">{{ cell_warn }}</div>
        {% endif %}
      </div>
      <div class="card">
        <div class="k">Regime 통계 ({{ regime }})</div>
        <div class="v2">20D: win={{ reg20_win }} (n={{ reg20_n }}), avg={{ reg20_avg }}</div>
        <div class="v2">60D: win={{ reg60_win }} (n={{ reg60_n }}), avg={{ reg60_avg }}</div>
        <div class="muted small">* 각 리포트 기간(1Y/8Y) 데이터만 사용</div>
      </div>
    </div>
  </div>

  <!-- D. BACKTESTING -->
  <div class="section">
    <div class="sec-title">D. 백테스트(Backtesting) — 지표가 실제로 성과를 분리했나</div>
    <div class="grid">
      <div class="card">
        <div class="k">Top20/Bot20 요약 (10D fwd KOSPI)</div>
        <div class="v2">Top20: win={{ bt_top_win }} (n={{ bt_top_n }}), avg={{ bt_top_avg }}</div>
        <div class="v2">Bot20: win={{ bt_bot_win }} (n={{ bt_bot_n }}), avg={{ bt_bot_avg }}</div>
        <div class="muted small">{{ bt_note }}</div>
      </div>
      <div class="card">
        <div class="k">해석 가이드</div>
        <div class="muted">1) bucket/Top20가 Bot20보다 유의미하게 좋으면 분리력↑</div>
        <div class="muted">2) 최근(1Y)과 장기(8Y)가 다르면 레짐 변화 가능</div>
        <div class="muted">3) n이 작으면 과신 금지</div>
      </div>
    </div>

    <div class="row">
      <div class="plot">{{ fig_bt_bucket|safe }}</div>
    </div>
    <div class="small">표(버킷별): n / win-rate / mean / median</div>
    {{ bt_table_html|safe }}
  </div>

  <!-- E. CONCLUSION -->
  <div class="section">
    <div class="sec-title">E. 결론(Conclusion) — 오늘 실행안 + 리스크</div>
    <div class="card">
      <div class="k">오늘의 실행안</div>
      <div class="v">{{ action }}</div>
      <div class="v2">Position guide: {{ position_guide }}</div>
      <div class="muted">Confidence: <b>{{ confidence }}</b> (Low면 비중가이드 절반 적용)</div>
      <div class="muted">Why(3줄):</div>
      <div class="muted">- {{ action_reason_1 }}</div>
      <div class="muted">- {{ action_reason_2 }}</div>
      <div class="muted">- {{ action_reason_3 }}</div>
      <div class="muted">Risks:</div>
      <div class="muted">- 표본 부족/레짐 변화/ML 결측(마지막 non-null 사용) 가능</div>
      <div class="muted">Tomorrow checklist:</div>
      <div class="muted">- Trend state 변화(Δ3/Δ5)와 ML(prob) 변화를 함께 확인</div>
      <div class="muted">- Top20/Bot20 분리력이 최근에도 유지되는지 점검</div>
    </div>
  </div>

</body>
</html>
"""


def main():
    index_path = _env("INDEX_PATH", "data/index_daily.parquet")
    report_path = _env("REPORT_PATH", "docs/index.html")
    lookback_days = int(_env("LOOKBACK_DAYS", "252"))
    title = _env("REPORT_TITLE", "한국곰탕지수 일일 리포트")
    factors_dir = _env("FACTORS_DIR", "data/factors")
    index_levels_path = _env("INDEX_LEVELS_PATH", "data/cache/index_levels.parquet")

    # Guide link
    guide_url = _env("GUIDE_URL", "gomtang_index_report_guide.html")
    guide_label = _env("GUIDE_LABEL", "Report Guide (Full)")

    date_col = _env("DATE_COL", "date")
    score_col = _env("SCORE_COL", "index_score_total")
    kospi_col = _env("KOSPI_COL", "kospi_close")
    kosdaq_col = _env("KOSDAQ_COL", "kosdaq_close")
    k200_col = _env("K200_COL", "kospi200_close")

    # Confirmed: flat threshold 1.0
    score_flat_pts = float(_env("SCORE_FLAT_PTS", "1.0"))

    hm_font_size = int(_env("HEATMAP_FONT_SIZE", "16"))

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    # Load index daily
    df = pd.read_parquet(index_path)
    if date_col not in df.columns:
        raise RuntimeError(f"[report] missing '{date_col}' in {index_path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Merge index levels for KOSPI/KOSDAQ/K200 raw lines
    if Path(index_levels_path).exists():
        lv = pd.read_parquet(index_levels_path)
        if date_col in lv.columns:
            lv[date_col] = pd.to_datetime(lv[date_col])
            lv = lv.sort_values(date_col)
            df = df.merge(lv, on=date_col, how="left", suffixes=("", "_lv"))

    base = df.tail(lookback_days).copy()

    if score_col not in base.columns:
        raise RuntimeError(f"[report] missing '{score_col}'")
    if kospi_col not in base.columns:
        raise RuntimeError(f"[report] missing '{kospi_col}' (check INDEX_LEVELS_PATH merge)")

    asof_date = base[date_col].iloc[-1].date().isoformat()

    # --- Overview KPIs ---
    g_score = float(base[score_col].iloc[-1]) if pd.notna(base[score_col].iloc[-1]) else np.nan
    g_bucket = _bucket_5pt(g_score)
    regime = _regime_label(g_score)

    g_d3 = _delta_over_n(base[score_col], 3)
    g_d5 = _delta_over_n(base[score_col], 5)
    g_d10 = _delta_over_n(base[score_col], 10)

    g_state_internal = _gomtang_state_5bins(g_d3, g_d5, score_flat_pts)
    g_state_label = STATE_LABEL.get(g_state_internal, g_state_internal)

    kospi_close = float(base[kospi_col].iloc[-1])
    kospi_r3 = _ret_over_n(base[kospi_col], 3)
    kospi_r5 = _ret_over_n(base[kospi_col], 5)
    kospi_r10 = _ret_over_n(base[kospi_col], 10)

    # ML prob (last non-null)
    prob_col = "prob_up_10d"
    prob_last = np.nan
    prob_asof = "-"
    if prob_col in base.columns:
        s = base.set_index(date_col)[prob_col]
        v, idx = _last_valid(s)
        prob_last = v
        if pd.notna(idx):
            prob_asof = pd.to_datetime(idx).date().isoformat()
    prob_disp = "-" if (prob_last is None or (isinstance(prob_last, float) and np.isnan(prob_last))) else f"{prob_last:.3f}"

    # Work DF for heatmaps/backtests
    work = base[[date_col, score_col, kospi_col]].dropna().copy()
    work["bucket_5pt"] = work[score_col].apply(_bucket_5pt)
    work["g_d3"] = work[score_col].diff(3)
    work["g_d5"] = work[score_col].diff(5)
    work["state_internal"] = [
        _gomtang_state_5bins(d3, d5, score_flat_pts)
        for d3, d5 in zip(work["g_d3"].fillna(0), work["g_d5"].fillna(0))
    ]
    work["market_state"] = work["state_internal"]
    work["fwd10"] = _forward_return(work[kospi_col], 10)

    today_bucket = g_bucket
    today_state_internal = g_state_internal
    today_state_label = STATE_LABEL.get(today_state_internal, today_state_internal)

    # Current cell stats (unclipped)
    cell_stats = _group_cell_stats(
        df=work,
        state_col="market_state",
        bucket_col="bucket_5pt",
        fwd_col="fwd10",
        state=today_state_internal,
        bucket=today_bucket,
    )
    cell_warn = "표본 부족 (n<10) — 해석 주의" if cell_stats["n"] < 10 else ""

    # Confidence and Action
    has_ml = not (prob_last is None or (isinstance(prob_last, float) and np.isnan(prob_last)))
    confidence = _confidence_level(has_ml, int(cell_stats["n"]))
    confidence_note = "Confidence Low → 비중 가이드를 절반으로 축소" if confidence == "Low" else ""
    op0 = _opinion_from_prob(prob_last)
    if has_ml:
        reason_ml = f"ML(prob={prob_last:.3f}, as of {prob_asof})"
    else:
        # ML 결측 시: heatmap 현재 셀(win_rate)로 Action 결정 (trend filter 미적용)
        op0 = _opinion_from_win_rate(float(cell_stats["win"]), int(cell_stats["n"]))
        reason_ml = (
            f"ML(prob 결측 → Heatmap win_rate 규칙) (win={_fmt_pct(cell_stats['win'],1, signed=False)}, n={cell_stats['n']})"
        )

    action = op0
    position_guide = _position_guide(action, confidence)

    # 3 fixed reasons
    action_reason_1 = f"Signal: {reason_ml}"
    action_reason_2 = f"Trend: Δ3={_fmt_float(g_d3,1)}, Δ5={_fmt_float(g_d5,1)} → {g_state_label}"
    action_reason_3 = f"Heatmap cell: n={cell_stats['n']}, win={_fmt_pct(cell_stats['win'],1, signed=False)}, avg={_fmt_pct(cell_stats['avg'],2, signed=True)}"

    # --- Charts (overview) ---
    fig_g = _line_fig(base, date_col, score_col, "GOMTANG INDEX (score 0–100)")
    fig_k = _line_fig(base, date_col, kospi_col, "KOSPI (raw)")

    # prob chart (dropna)
    fig_p = None
    if prob_col in base.columns:
        dprob = base[[date_col, prob_col]].dropna()
        if len(dprob) > 0:
            fig_p = go.Figure()
            fig_p.add_trace(
                go.Scatter(
                    x=dprob[date_col],
                    y=dprob[prob_col],
                    mode="lines",
                    line=dict(width=2),
                    name="prob_up_10d",
                )
            )
            fig_p.update_layout(
                title="XGBoost: prob_up_10d (series, non-null)",
                height=260,
                margin=dict(l=30, r=20, t=40, b=30),
                template="plotly_white",
                yaxis=dict(range=[0, 1]),
            )

    # --- Heatmaps ---
    grp = work.dropna(subset=["fwd10"]).groupby(["bucket_5pt", "market_state"])["fwd10"]
    pivot_mean = grp.mean().unstack("market_state").reindex(columns=STATE_ORDER_INTERNAL)
    pivot_n = grp.count().unstack("market_state").reindex(columns=STATE_ORDER_INTERNAL)

    pivot_mean_clip = pivot_mean.clip(lower=-0.03, upper=0.03)

    # relabel for display
    pivot_mean_clip.columns = [STATE_LABEL.get(c, c) for c in pivot_mean_clip.columns]
    pivot_n.columns = [STATE_LABEL.get(c, c) for c in pivot_n.columns]

    grp_win = (
        work.dropna(subset=["fwd10"])
        .groupby(["bucket_5pt", "market_state"])["fwd10"]
        .apply(lambda x: (x > 0).mean())
    )
    pivot_win = grp_win.unstack("market_state").reindex(columns=STATE_ORDER_INTERNAL)
    pivot_win.columns = [STATE_LABEL.get(c, c) for c in pivot_win.columns]
    pivot_win_n = pivot_n

    fig_hm_mean = _heatmap_fig(
        pivot_val=pivot_mean_clip,
        pivot_n=pivot_n,
        title="Heatmap: 10D Forward KOSPI Mean Return (clipped ±3%)",
        colorscale="RdBu",
        zmid=0,
        zmin=-0.03,
        zmax=0.03,
        is_percent=False,
        highlight_xy=(today_state_label, today_bucket),
        cell_font_size=hm_font_size,
    )

    fig_hm_win = _heatmap_fig(
        pivot_val=pivot_win,
        pivot_n=pivot_win_n,
        title="Heatmap: 10D Forward KOSPI Win-rate",
        colorscale="Greens",
        zmid=None,
        zmin=0.0,
        zmax=1.0,
        is_percent=True,
        highlight_xy=(today_state_label, today_bucket),
        cell_font_size=hm_font_size,
    )

    # Regime stats 20D/60D
    reg20 = _regime_forward_stats(work, score_col, kospi_col, 20, regime)
    reg60 = _regime_forward_stats(work, score_col, kospi_col, 60, regime)

    # --- Backtesting ---
    bt = work.dropna(subset=["fwd10"]).copy()
    bt_tbl = (
        bt.groupby("bucket_5pt")
        .agg(
            n=("fwd10", "count"),
            win=("fwd10", lambda x: (x > 0).mean()),
            avg=("fwd10", "mean"),
            med=("fwd10", "median"),
        )
        .sort_index()
    )

    fig_bt_bucket = _barline_bucket_fig(bt_tbl, "Backtest: bucket별 10D forward KOSPI (mean + win-rate)")

    if len(bt_tbl) > 0:
        bt_tbl_disp = bt_tbl.copy()
        bt_tbl_disp["win"] = bt_tbl_disp["win"].map(lambda x: _fmt_pct(x, 1, signed=False))
        bt_tbl_disp["avg"] = bt_tbl_disp["avg"].map(lambda x: _fmt_pct(x, 2, signed=True))
        bt_tbl_disp["med"] = bt_tbl_disp["med"].map(lambda x: _fmt_pct(x, 2, signed=True))
        bt_table_html = (
            bt_tbl_disp.reset_index()
            .rename(columns={"bucket_5pt": "bucket"})
            .to_html(index=False, escape=False)
        )
    else:
        bt_table_html = "<div class='small'>데이터 없음</div>"

    bt_note = "score(곰탕지수) 상/하위 20% 기준의 10D forward KOSPI 요약 (리포트 기간 내)"
    bt_top_n = bt_bot_n = 0
    bt_top_win = bt_bot_win = np.nan
    bt_top_avg = bt_bot_avg = np.nan

    if len(bt) >= 30:
        q80 = bt[score_col].quantile(0.80)
        q20 = bt[score_col].quantile(0.20)

        top = bt[bt[score_col] >= q80]["fwd10"].dropna()
        bot = bt[bt[score_col] <= q20]["fwd10"].dropna()

        bt_top_n = int(len(top))
        bt_bot_n = int(len(bot))
        bt_top_win = float((top > 0).mean()) if len(top) > 0 else np.nan
        bt_bot_win = float((bot > 0).mean()) if len(bot) > 0 else np.nan
        bt_top_avg = float(top.mean()) if len(top) > 0 else np.nan
        bt_bot_avg = float(bot.mean()) if len(bot) > 0 else np.nan
    else:
        bt_note += " (표본 부족으로 신뢰 낮음)"

    # --- Factors (summary + cards) ---
    factors_extreme_greed = []
    factors_extreme_fear = []
    factor_cards = []

    for i in range(1, 11):
        tag = f"f{i:02d}"
        p = Path(factors_dir) / f"{tag}.parquet"
        if not p.exists():
            continue

        try:
            f = pd.read_parquet(p)
        except Exception:
            continue
        if date_col not in f.columns:
            continue

        f[date_col] = pd.to_datetime(f[date_col])
        f = f.sort_values(date_col)
        f = f[(f[date_col] >= base[date_col].min()) & (f[date_col] <= base[date_col].max())].copy()
        if len(f) == 0:
            continue

        raw_cols = [c for c in f.columns if c.lower().endswith("_raw")]
        score_cols = [c for c in f.columns if c.lower().endswith("_score")]

        ycol = None
        mode = None
        if raw_cols:
            ycol = raw_cols[0]
            mode = "RAW"
        elif score_cols:
            ycol = score_cols[0]
            mode = "SCORE"
        else:
            continue

        name, desc = FACTOR_META.get(tag, (tag.upper(), ""))

        series = f[ycol].dropna()
        today_val = np.nan
        pct = np.nan
        band_disp = "-"
        q20 = q40 = q60 = q80 = None

        if len(series) >= 5:
            today_val = float(series.iloc[-1])
            pct = float(series.rank(pct=True).iloc[-1])
            band = pct_to_band(pct)
            greed_is_high = FACTOR_GREED_IS_HIGH.get(tag, True)
            band_disp = band if greed_is_high else flip_band(band)
            q20, q40, q60, q80 = series.quantile([0.2, 0.4, 0.6, 0.8]).tolist()

            # summary lists
            if band_disp == "EXTREME GREED":
                factors_extreme_greed.append(tag.upper())
            if band_disp == "EXTREME FEAR":
                factors_extreme_fear.append(tag.upper())

        fig = _line_fig(f, date_col, ycol, f"{tag.upper()} · {name} ({mode})")
        if fig is None:
            continue

        if q20 is not None and pd.notna(q20):
            for q, lab in [(q20, "P20"), (q40, "P40"), (q60, "P60"), (q80, "P80")]:
                fig.add_hline(
                    y=float(q),
                    line_width=1,
                    line_dash="dot",
                    line_color="#666",
                    annotation_text=lab,
                    annotation_position="top left",
                )

        if pd.notna(today_val):
            fig.add_trace(
                go.Scatter(
                    x=[f[date_col].iloc[-1]],
                    y=[today_val],
                    mode="markers",
                    marker=dict(size=10, color="#111"),
                    name="Today",
                    hovertemplate="Today=%{y}<extra></extra>",
                )
            )

        pct_disp = "-" if np.isnan(pct) else f"{pct*100:,.1f}%"
        today_disp = "-" if (isinstance(today_val, float) and np.isnan(today_val)) else _fmt_float(today_val, 4)
        direction_note = "높을수록 탐욕" if FACTOR_GREED_IS_HIGH.get(tag, True) else "낮을수록 탐욕(표시 밴드 반전)"

        header_html = (
            f"<div class='card' style='margin:4px 4px 10px 4px'>"
            f"  <div class='k'>{tag.upper()} · {name}</div>"
            f"  <div class='muted'>{desc}</div>"
            f"  <div class='muted'>Today: <b>{today_disp}</b> · Percentile: <b>{pct_disp}</b> · Band: <b>{band_disp}</b></div>"
            f"  <div class='muted small'>Direction: {direction_note} · Bands: 20/40/60/80 · SPEC: P20/P40/P60/P80</div>"
            f"</div>"
        )

        # A안: Plotly.js를 HTML에 내장하지 않고, figure 블록에서 CDN으로 로드
        factor_cards.append(header_html + fig.to_html(include_plotlyjs="cdn", full_html=False))

    factors_extreme_greed_str = ", ".join(factors_extreme_greed) if factors_extreme_greed else "-"
    factors_extreme_fear_str = ", ".join(factors_extreme_fear) if factors_extreme_fear else "-"

    # Convert figures to HTML (A안: include_plotlyjs="cdn")
    def _fig_html(fig):
        if fig is None:
            return "<div style='color:#666;font-size:12px'>데이터 없음</div>"
        return fig.to_html(include_plotlyjs="cdn", full_html=False)

    html = Template(HTML_TMPL).render(
        title=title,
        asof_date=asof_date,
        lookback_days=lookback_days,
        guide_url=guide_url,
        guide_label=guide_label,
        gomtang_score=_fmt_float(g_score, 1),
        gomtang_bucket=g_bucket,
        regime=regime,
        gomtang_d3=_fmt_float(g_d3, 1),
        gomtang_d5=_fmt_float(g_d5, 1),
        gomtang_d10=_fmt_float(g_d10, 1),
        gomtang_state_label=g_state_label,
        kospi_close=_fmt_float(kospi_close, 2),
        kospi_r3=_fmt_pct(kospi_r3, 2, signed=True),
        kospi_r5=_fmt_pct(kospi_r5, 2, signed=True),
        kospi_r10=_fmt_pct(kospi_r10, 2, signed=True),
        prob_up_10d=prob_disp,
        prob_asof=prob_asof,
        action=action,
        position_guide=position_guide,
        confidence=confidence,
        confidence_note=confidence_note,
        has_ml=("Y" if has_ml else "N"),
        cell_n_plain=int(cell_stats["n"]),
        action_reason_1=action_reason_1,
        action_reason_2=action_reason_2,
        action_reason_3=action_reason_3,
        fig_gomtang=_fig_html(fig_g),
        fig_kospi=_fig_html(fig_k),
        fig_prob=_fig_html(fig_p),
        factor_cards=factor_cards,
        factors_extreme_greed=factors_extreme_greed_str,
        factors_extreme_fear=factors_extreme_fear_str,
        fig_hm_mean=_fig_html(fig_hm_mean),
        fig_hm_win=_fig_html(fig_hm_win),
        cell_bucket=today_bucket,
        cell_state_label=today_state_label,
        cell_n=_fmt_int(cell_stats["n"]),
        cell_win=_fmt_pct(cell_stats["win"], 1, signed=False),
        cell_avg=_fmt_pct(cell_stats["avg"], 2, signed=True),
        cell_med=_fmt_pct(cell_stats["median"], 2, signed=True),
        cell_q1=_fmt_pct(cell_stats["q1"], 2, signed=True),
        cell_q3=_fmt_pct(cell_stats["q3"], 2, signed=True),
        cell_warn=cell_warn,
        reg20_n=_fmt_int(reg20["n"]),
        reg20_win=_fmt_pct(reg20["win"], 1, signed=False),
        reg20_avg=_fmt_pct(reg20["avg"], 2, signed=True),
        reg60_n=_fmt_int(reg60["n"]),
        reg60_win=_fmt_pct(reg60["win"], 1, signed=False),
        reg60_avg=_fmt_pct(reg60["avg"], 2, signed=True),
        bt_top_n=_fmt_int(bt_top_n),
        bt_top_win=_fmt_pct(bt_top_win, 1, signed=False),
        bt_top_avg=_fmt_pct(bt_top_avg, 2, signed=True),
        bt_bot_n=_fmt_int(bt_bot_n),
        bt_bot_win=_fmt_pct(bt_bot_win, 1, signed=False),
        bt_bot_avg=_fmt_pct(bt_bot_avg, 2, signed=True),
        bt_note=bt_note,
        fig_bt_bucket=_fig_html(fig_bt_bucket),
        bt_table_html=bt_table_html,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[report] OK -> {report_path}")


if __name__ == "__main__":
    main()
