# src/report/render_report.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template


HTML_TMPL = Template(
    """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Gomtang Index</title>
  <link rel="stylesheet" href="assets/style.css"/>
</head>
<body>
<div class="container">
  <h1>Gomtang Index</h1>

  <div class="grid">
    <div class="card">
      <h2>오늘의 지수</h2>
      <div class="big">{{ today_score }}</div>
      <div class="muted">기준일: {{ today_date }}</div>
      <div class="muted">점수구간(5점): {{ today_bucket }}</div>
    </div>
    <div class="card">
      <h2>데이터 소스</h2>
      <div class="muted">
        ③은 KRX OpenAPI 유가증권 일별매매정보(stk_bydd_trd)의 OHLCV(고가/저가/종가/거래량)를 캐시화해 계산합니다. [Source](https://www.genspark.ai/api/files/s/B1PYOvrO)
      </div>
      <div class="muted">
        VKOSPI(코스피 200 변동성지수)는 파생상품지수 시세정보에서 IDX_NM 필터로 수집하며 종가 컬럼은 CLSPRC_IDX입니다. [Source](https://www.genspark.ai/api/files/s/uX7923Iq)
      </div>
    </div>
  </div>

  <div class="card">
    <h2>지수 라인차트</h2>
    {{ fig_index | safe }}
  </div>

  <div class="card">
    <h2>팩터 점수 라인차트</h2>
    {{ fig_factors | safe }}
  </div>

</div>
</body>
</html>
"""
)


def fmt_score(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.1f}"


def main():
    index_path = Path(os.environ.get("INDEX_PATH", "data/index_daily.parquet"))
    out_html = Path(os.environ.get("REPORT_PATH", "docs/index.html"))

    if not index_path.exists():
        raise RuntimeError(f"Missing {index_path}. Run assemble first.")

    df = pd.read_parquet(index_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    last = df.dropna(subset=["index_score_total"]).tail(1)
    if last.empty:
        today_date = today_score = today_bucket = "-"
    else:
        r = last.iloc[0]
        today_date = pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
        today_score = fmt_score(float(r["index_score_total"]))
        b = float(r.get("bucket_5pt", np.nan))
        today_bucket = "-" if np.isnan(b) else f"{b:.0f} ~ {b+5:.0f}"

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["date"], y=df.get("index_score_total"), mode="lines", name="Index Score"))
    fig1.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(range=[0, 100]))
    fig_index_html = fig1.to_html(full_html=False, include_plotlyjs="cdn")

    fig2 = go.Figure()
    for c in ["f01_score","f02_score","f03_score","f04_score","f05_score","f06_score","f07_score","f08_score","f10_score"]:
        if c in df.columns:
            fig2.add_trace(go.Scatter(x=df["date"], y=df[c], mode="lines", name=c))
    fig2.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(range=[0, 100]))
    fig_factors_html = fig2.to_html(full_html=False, include_plotlyjs=False)

    html = HTML_TMPL.render(
        today_date=today_date,
        today_score=today_score,
        today_bucket=today_bucket,
        fig_index=fig_index_html,
        fig_factors=fig_factors_html,
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"[report] OK -> {out_html}")


if __name__ == "__main__":
    main()
