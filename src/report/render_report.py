# -*- coding: utf-8 -*-
"""
Gomtang Index Daily Report

Key specs (user-confirmed):
- Heatmap X-axis state is based on GOMTANG INDEX trend (index_score_total Δ3/Δ5), not KOSPI
- Flat threshold for trend state: SCORE_FLAT_PTS = 1.0 point
- Heatmap text visibility: larger font, bold main value, n on second line
- Factor charts: rename + add descriptions (F01~F10)
- ML prob_up_10d: display last non-null value + its date (to avoid NaN on latest row)
- Investment opinion: conservative 0.6/0.4; ML missing -> trend-only fallback; apply trend filter
- Highlight today's heatmap cell (today bucket_5pt × today state)
- Mean-return heatmap: clip ±3% (option A)
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
    # user confirmed: Fear < 40, Neutral 40~60, Greed > 60
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "-"
    if score < 40:
        return "Fear"
    if score <= 60:
        return "Neutral"
    return "Greed"


# ---------------------------
# Factor names / descriptions (user-provided)
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


# ---------------------------
# Gomtang trend state (X-axis for heatmaps)
# ---------------------------
def _gomtang_state_5bins(d3: float, d5: float, flat_pts: float) -> str:
    """
    Priority: 5D down/up, then 3D down/up, else flat.
    d3/d5 are point changes in index_score_total, not %.
    """
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
# Plot helpers
# ---------------------------
def _line_fig(df: pd.DataFrame, xcol: str, ycol: str, title: str):
    if ycol not in df.columns:
        return None
    d = df[[xcol, ycol]].dropna()
    if len(d) == 0:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d[xcol], y=d[ycol],
        mode="lines",
        line=dict(width=2),
        name=ycol
    ))
    fig.update_layout(
        title=title,
        height=320,
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
    highlight_xy=None,   # (x_label, y_label)
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

    # More readable text (bold main value + small n)
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
            nij_s = "-" if (nij is None or (isinstance(nij, float) and np.isnan(nij))) else str(int(nij))
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

    fig = go.Figure(data=go.Heatmap(
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
    ))

    fig.update_layout(
        title=title,
        height=620,
        margin=dict(l=45, r=20, t=55, b=40),
        template="plotly_white",
        xaxis_title="Gomtang trend state (Δ3/Δ5)",
        yaxis_title="Gomtang bucket (5pt)",
    )
    fig.update_xaxes(side="top")

    # Highlight today's cell with border
    if highlight_xy and highlight_xy[0] in x_labels and highlight_xy[1] in y_labels:
        x0 = x_labels.index(highlight_xy[0]) - 0.5
        x1 = x_labels.index(highlight_xy[0]) + 0.5
        y0 = y_labels.index(highlight_xy[1]) - 0.5
        y1 = y_labels.index(highlight_xy[1]) + 0.5
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=x0, x1=x1, y0=y0, y1=y1,
            line=dict(color="#111", width=3),
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )

    return fig


# ---------------------------
# Stats helpers
# ---------------------------
def _group_cell_stats(df: pd.DataFrame, state_col: str, bucket_col: str, fwd_col: str, state: str, bucket: str):
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
    if score_col not in df.columns or close_col not in df.columns:
        return {"n": 0, "win": np.nan, "avg": np.nan}
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
# Opinion logic
# ---------------------------
def _opinion_from_prob(prob: float):
    # user confirmed: conservative thresholds 0.6/0.4
    if prob is None or (isinstance(prob, float) and np.isnan(prob)):
        return None
    if prob >= 0.6:
        return "BUY"
    if prob < 0.4:
        return "SELL"
    return "HOLD"


def _opinion_from_trend(delta3: float, delta5: float):
    # trend-only fallback
    if (delta3 is None or np.isnan(delta3)) or (delta5 is None or np.isnan(delta5)):
        return "HOLD"
    if delta3 >= 0 and delta5 >= 0:
        return "BUY"
    if delta3 <= 0 and delta5 <= 0:
        return "SELL"
    return "HOLD"


def _apply_trend_filter(opinion: str, delta3: float, delta5: float):
    # If both down => downgrade; if both up => upgrade
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


# ---------------------------
# HTML Template
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
    .sub { color:#666; font-size: 12px; margin-bottom: 14px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(220px, 1fr)); gap: 10px; }
    .card { border: 1px solid #e6e6e6; border-radius: 12px; padding: 12px 12px; background: #fff; }
    .k { color:#666; font-size: 12px; margin-bottom: 4px; }
    .v { font-size: 20px; font-weight: 800; }
    .v2 { font-size: 14px; font-weight: 650; margin-top: 4px; }
    .warn { margin-top: 6px; color: #b00020; font-size: 12px; font-weight: 800; }
    .muted { color:#666; font-size: 12px; margin-top: 6px; line-height: 1.4; }
    .row { margin-top: 14px; display:grid; grid-template-columns: 1fr; gap: 12px; }
    .plot { border: 1px solid #eee; border-radius: 12px; padding: 8px; background:#fff; }
    .section { margin-top: 18px; }
    .sec-title { font-size: 16px; font-weight: 900; margin: 6px 0 8px; }
    .links a { color:#0b57d0; text-decoration:none; }
    .links a:hover { text-decoration:underline; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="sub">
    기준일: <b>{{ asof_date }}</b> · Lookback: {{ lookback_days }} days · SCORE_FLAT_PTS={{ score_flat_pts }}
  </div>

  <div class="grid">
    <div class="card">
      <div class="k">오늘의 곰탕지수</div>
      <div class="v">{{ gomtang_score }}</div>
      <div class="v2">Bucket: {{ gomtang_bucket }} · Regime: {{ regime }}</div>
      <div class="muted">Δ3: {{ gomtang_d3 }} · Δ5: {{ gomtang_d5 }} · Δ10: {{ gomtang_d10 }}</div>
      <div class="muted">Trend state: <b>{{ gomtang_state }}</b></div>
    </div>

    <div class="card">
      <div class="k">KOSPI (raw)</div>
      <div class="v">{{ kospi_close }}</div>
      <div class="muted">3D: {{ kospi_r3 }} · 5D: {{ kospi_r5 }} · 10D: {{ kospi_r10 }}</div>
    </div>

    <div class="card">
      <div class="k">XGBoost: prob_up_10d</div>
      <div class="v">{{ prob_up_10d }}</div>
      <div class="muted">as of: {{ prob_asof }}</div>
      <div class="muted">Rule: BUY≥0.60 / SELL&lt;0.40 (else HOLD)</div>
    </div>

    <div class="card">
      <div class="k">오늘의 의견 (ML+추세)</div>
      <div class="v">{{ opinion }}</div>
      <div class="muted">{{ opinion_reason }}</div>
    </div>
  </div>

  <div class="row section">
    <div class="plot">{{ fig_gomtang|safe }}</div>
    <div class="plot">{{ fig_kospi|safe }}</div>
    <div class="plot">{{ fig_kosdaq|safe }}</div>
    <div class="plot">{{ fig_k200|safe }}</div>
    <div class="plot">{{ fig_prob|safe }}</div>
  </div>

  <div class="row section">
    <div class="plot">{{ fig_hm_mean|safe }}</div>
    <div class="plot">{{ fig_hm_win|safe }}</div>
  </div>

  <div class="grid section">
    <div class="card">
      <div class="k">현재 셀 통계 ({{ cell_bucket }} × {{ cell_state }})</div>
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
      <div class="muted">* 각 리포트 기간(1Y/8Y) 데이터만 사용</div>
    </div>

    <div class="card">
      <div class="k">참고</div>
      <div class="muted links">
        - Index assemble:
          <a href="https://raw.githubusercontent.com/gomtang-dori/Korea-Gomtang-Index/main/src/assemble/assemble_index.py" target="_blank">assemble_index.py</a><br>
        - Merge prob:
          <a href="https://raw.githubusercontent.com/gomtang-dori/Korea-Gomtang-Index/main/src/analysis/merge_prob_into_index.py" target="_blank">merge_prob_into_index.py</a><br>
        - adv/dec workflow:
          <a href="https://raw.githubusercontent.com/gomtang-dori/Korea-Gomtang-Index/main/.github/workflows/advdec_daily.yml" target="_blank">advdec_daily.yml</a>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="sec-title">팩터 차트 (RAW 우선, 없으면 SCORE)</div>
    {% for fc in factor_cards %}
      <div class="plot">{{ fc|safe }}</div>
    {% endfor %}
  </div>
</body>
</html>
"""


# ---------------------------
# Main
# ---------------------------
def main():
    index_path = _env("INDEX_PATH", "data/index_daily.parquet")
    report_path = _env("REPORT_PATH", "docs/index.html")
    lookback_days = int(_env("LOOKBACK_DAYS", "252"))
    title = _env("REPORT_TITLE", "한국곰탕지수 일일 리포트")
    factors_dir = _env("FACTORS_DIR", "data/factors")
    index_levels_path = _env("INDEX_LEVELS_PATH", "data/cache/index_levels.parquet")

    # Columns
    date_col = _env("DATE_COL", "date")
    score_col = _env("SCORE_COL", "index_score_total")
    kospi_col = _env("KOSPI_COL", "kospi_close")
    kosdaq_col = _env("KOSDAQ_COL", "kosdaq_close")
    k200_col = _env("K200_COL", "kospi200_close")

    # Heatmap x-axis flat threshold in points (user confirmed: 1.0)
    score_flat_pts = float(_env("SCORE_FLAT_PTS", "1.0"))

    # Heatmap font size
    hm_font_size = int(_env("HEATMAP_FONT_SIZE", "16"))

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    # Load index daily
    df = pd.read_parquet(index_path)
    if date_col not in df.columns:
        raise RuntimeError(f"[report] missing '{date_col}' in {index_path}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Load index levels and merge (for KOSPI/KOSDAQ/K200 lines)
    if Path(index_levels_path).exists():
        lv = pd.read_parquet(index_levels_path)
        if date_col in lv.columns:
            lv[date_col] = pd.to_datetime(lv[date_col])
            lv = lv.sort_values(date_col)
            df = df.merge(lv, on=date_col, how="left", suffixes=("", "_lv"))

    # Lookback slice
    base = df.tail(lookback_days).copy()

    if kospi_col not in base.columns:
        raise RuntimeError(f"[report] missing '{kospi_col}'. Check INDEX_LEVELS_PATH merge.")
    if score_col not in base.columns:
        raise RuntimeError(f"[report] missing '{score_col}'. Check index assembly output.")

    asof_date = base[date_col].iloc[-1].date().isoformat()

    # Gomtang KPIs
    g_score = float(base[score_col].iloc[-1]) if pd.notna(base[score_col].iloc[-1]) else np.nan
    g_bucket = _bucket_5pt(g_score)
    regime = _regime_label(g_score)

    g_d3 = _delta_over_n(base[score_col], 3)
    g_d5 = _delta_over_n(base[score_col], 5)
    g_d10 = _delta_over_n(base[score_col], 10)
    g_state = _gomtang_state_5bins(g_d3, g_d5, score_flat_pts)

    # KOSPI KPIs
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

    # Opinion
    op0 = _opinion_from_prob(prob_last)
    if op0 is None:
        op0 = _opinion_from_trend(g_d3, g_d5)
        reason0 = "ML(prob)이 공란 → 곰탕지수 추세(Δ3/Δ5)로 대체 산출"
    else:
        reason0 = f"ML(prob={prob_last:.3f}) 기반"
    op1 = _apply_trend_filter(op0, g_d3, g_d5)
    if op1 != op0:
        reason = reason0 + f" + 추세 필터(Δ3={_fmt_float(g_d3,1)}, Δ5={_fmt_float(g_d5,1)})로 보정 → {op1}"
    else:
        reason = reason0 + f" (Δ3={_fmt_float(g_d3,1)}, Δ5={_fmt_float(g_d5,1)})"

    # Line charts
    fig_g = _line_fig(base, date_col, score_col, "GOMTANG INDEX (score 0–100)")
    fig_k = _line_fig(base, date_col, kospi_col, "KOSPI (raw)")
    fig_kq = _line_fig(base, date_col, kosdaq_col, "KOSDAQ (raw)")
    fig_k200 = _line_fig(base, date_col, k200_col, "KOSPI 200 (raw)")

    # prob chart (dropna)
    fig_p = None
    if prob_col in base.columns:
        dprob = base[[date_col, prob_col]].dropna()
        if len(dprob) > 0:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=dprob[date_col], y=dprob[prob_col],
                mode="lines",
                line=dict(width=2),
                name="prob_up_10d"
            ))
            fig_p.update_layout(
                title="XGBoost: prob_up_10d (series, non-null)",
                height=280,
                margin=dict(l=30, r=20, t=40, b=30),
                template="plotly_white",
                yaxis=dict(range=[0, 1]),
            )

    # Heatmap prep: use GOMTANG trend state on X-axis; forward KOSPI 10D on cell values
    work = base[[date_col, score_col, kospi_col]].dropna().copy()
    work["bucket_5pt"] = work[score_col].apply(_bucket_5pt)
    work["g_d3"] = work[score_col].diff(3)
    work["g_d5"] = work[score_col].diff(5)
    work["market_state"] = [
        _gomtang_state_5bins(d3, d5, score_flat_pts)
        for d3, d5 in zip(work["g_d3"].fillna(0), work["g_d5"].fillna(0))
    ]
    work["fwd10"] = _forward_return(work[kospi_col], 10)

    today_bucket = g_bucket
    today_state = g_state

    grp = work.dropna(subset=["fwd10"]).groupby(["bucket_5pt", "market_state"])["fwd10"]
    pivot_mean = grp.mean().unstack("market_state")
    pivot_n = grp.count().unstack("market_state")

    # clip ±3% (option A)
    pivot_mean_clip = pivot_mean.clip(lower=-0.03, upper=0.03)

    grp_win = work.dropna(subset=["fwd10"]).groupby(["bucket_5pt", "market_state"])["fwd10"].apply(lambda x: (x > 0).mean())
    pivot_win = grp_win.unstack("market_state")
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
        highlight_xy=(today_state, today_bucket),
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
        highlight_xy=(today_state, today_bucket),
        cell_font_size=hm_font_size,
    )

    # Current cell stats (unclipped fwd10)
    cell_stats = _group_cell_stats(
        df=work,
        state_col="market_state",
        bucket_col="bucket_5pt",
        fwd_col="fwd10",
        state=today_state,
        bucket=today_bucket,
    )
    cell_warn = "표본 부족 (n<10) — 해석 주의" if cell_stats["n"] < 10 else ""

    # Regime stats 20D / 60D
    reg20 = _regime_forward_stats(work, score_col, kospi_col, 20, regime)
    reg60 = _regime_forward_stats(work, score_col, kospi_col, 60, regime)

    # Factor cards: rename + description; plot RAW if present else SCORE
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

        fig = _line_fig(f, date_col, ycol, f"{tag.upper()} · {name} ({mode})")
        if fig is None:
            continue

        header_html = (
            f"<div class='card' style='margin:4px 4px 10px 4px'>"
            f"  <div class='k'>{tag.upper()} · {name}</div>"
            f"  <div class='muted'>{desc}</div>"
            f"</div>"
        )

        # Important: to avoid CDN issues, embed plotly.js for each figure (safe mode)
        factor_cards.append(header_html + fig.to_html(include_plotlyjs=True, full_html=False))

    # Embed Plotly safely in each figure block (safe mode)
    def _fig_html(fig):
        if fig is None:
            return "<div style='color:#666;font-size:12px'>데이터 없음</div>"
        return fig.to_html(include_plotlyjs=True, full_html=False)

    html = Template(HTML_TMPL).render(
        title=title,
        asof_date=asof_date,
        lookback_days=lookback_days,
        score_flat_pts=_fmt_float(score_flat_pts, 1),

        gomtang_score=_fmt_float(g_score, 1),
        gomtang_bucket=g_bucket,
        regime=regime,
        gomtang_d3=_fmt_float(g_d3, 1),
        gomtang_d5=_fmt_float(g_d5, 1),
        gomtang_d10=_fmt_float(g_d10, 1),
        gomtang_state=g_state,

        kospi_close=_fmt_float(kospi_close, 2),
        kospi_r3=_fmt_pct(kospi_r3, 2, signed=True),
        kospi_r5=_fmt_pct(kospi_r5, 2, signed=True),
        kospi_r10=_fmt_pct(kospi_r10, 2, signed=True),

        prob_up_10d=prob_disp,
        prob_asof=prob_asof,

        opinion=op1,
        opinion_reason=reason,

        fig_gomtang=_fig_html(fig_g),
        fig_kospi=_fig_html(fig_k),
        fig_kosdaq=_fig_html(fig_kq),
        fig_k200=_fig_html(fig_k200),
        fig_prob=_fig_html(fig_p),

        fig_hm_mean=_fig_html(fig_hm_mean),
        fig_hm_win=_fig_html(fig_hm_win),

        cell_bucket=today_bucket,
        cell_state=today_state,
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

        factor_cards=factor_cards,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[report] OK -> {report_path}")


if __name__ == "__main__":
    main()
