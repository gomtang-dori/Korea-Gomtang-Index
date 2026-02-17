#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip()
    return default if v == "" else v


def _read_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def _tail_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df is None or len(df) == 0 or "date" not in df.columns:
        return df
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    return df[df["date"] >= cutoff].reset_index(drop=True)


def _pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{100.0 * float(x):.1f}%"


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{float(x):.{nd}f}"


HTML = Template(
r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Gomtang Diagnostics</title>
  <style>
    body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans KR",Arial,sans-serif; background:#0b1220; color:#e6edf3; margin:0; }
    a { color:#9ecbff; text-decoration:none; }
    .wrap { max-width:1200px; margin:0 auto; padding:22px; }
    .h1 { font-size:22px; font-weight:800; margin:0 0 10px; }
    .sub { color:#9fb0c0; margin:0 0 20px; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap:14px; }
    .card { background:#0f1a2c; border:1px solid #22314a; border-radius:14px; padding:14px; }
    .card h2 { font-size:15px; margin:0 0 10px; }
    .kpi { display:flex; gap:12px; flex-wrap:wrap; }
    .kpi .box { background:#0b1425; border:1px solid #22314a; border-radius:12px; padding:10px 12px; min-width:180px; }
    .kpi .label { color:#9fb0c0; font-size:12px; }
    .kpi .value { font-size:18px; font-weight:800; margin-top:4px; }
    table { width:100%; border-collapse:collapse; }
    th,td { border-bottom:1px solid #22314a; padding:8px 8px; font-size:13px; text-align:left; }
    th { color:#bcd0e5; font-weight:700; }
    .muted { color:#9fb0c0; font-size:12px; line-height:1.5; }
    .mono { font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace; }
    .full { grid-column: 1 / -1; }
  </style>
</head>
<body>
<div class="wrap">
  <div class="h1">Gomtang Diagnostics (최근 {{ days }}일)</div>
  <div class="sub">
    대상: <b>{{ tag }}</b> /
    Index: <span class="mono">{{ index_path }}</span> /
    ML metrics: <span class="mono">{{ ml_metrics_path }}</span>
  </div>

  <div class="grid">

    <div class="card full">
      <h2>1) 지수 + ML 확률(prob_up_10d) 추이</h2>
      {{ fig_index_prob | safe }}
      <div class="muted">
        prob_up_10d는 <span class="mono">train_xgb_prob_up10.py</span>에서 생성 후
        <span class="mono">merge_prob_into_index.py</span>로 index parquet에 merge됩니다. [Source](https://raw.githubusercontent.com/gomtang-dori/Korea-Gomtang-Index/main/src/analysis/train_xgb_prob_up10.py)
        [Source](https://raw.githubusercontent.com/gomtang-dori/Korea-Gomtang-Index/main/src/analysis/merge_prob_into_index.py)
      </div>
    </div>

    <div class="card">
      <h2>2) ML 출력 검증(KPI)</h2>
      <div class="kpi">
        <div class="box">
          <div class="label">prob_up_10d NaN 비율(최근 {{ days }}일)</div>
          <div class="value">{{ prob_nan_rate }}</div>
        </div>
        <div class="box">
          <div class="label">prob_up_10d 최소/최대</div>
          <div class="value">{{ prob_min }} ~ {{ prob_max }}</div>
        </div>
        <div class="box">
          <div class="label">index_score_total 최소/최대</div>
          <div class="value">{{ idx_min }} ~ {{ idx_max }}</div>
        </div>
      </div>
      <div class="muted" style="margin-top:10px;">
        기대값: prob_up_10d는 0~1 범위, NaN 비율이 과도하게 높으면 date merge 불일치/파일 누락 가능성이 큽니다.
      </div>
    </div>

    <div class="card">
      <h2>3) Walk-forward 성능 요약 (AUC/Brier)</h2>
      <div class="kpi">
        <div class="box">
          <div class="label">AUC 평균/중앙값</div>
          <div class="value">{{ auc_mean }} / {{ auc_median }}</div>
        </div>
        <div class="box">
          <div class="label">Brier 평균/중앙값</div>
          <div class="value">{{ brier_mean }} / {{ brier_median }}</div>
        </div>
      </div>
      <div class="muted" style="margin-top:10px;">
        fold별 AUC가 비어있는 행은 테스트 구간에서 클래스가 한쪽으로만 존재하는 등으로 AUC 계산 불가인 경우가 흔합니다.
        (현재 AUC 계산 함수는 단일 클래스면 NaN 반환) [Source](https://raw.githubusercontent.com/gomtang-dori/Korea-Gomtang-Index/main/src/analysis/train_xgb_prob_up10.py)
      </div>
    </div>

    <div class="card full">
      <h2>4) Walk-forward fold 테이블(최근 일부)</h2>
      {{ fold_table | safe }}
    </div>

    <div class="card full">
      <h2>5) 가중치 크로스체크</h2>
      <div class="muted">
        <b>중요:</b> 현재 <span class="mono">assemble_index.py</span>는 코드 상단의 고정 딕셔너리 <span class="mono">W</span>만 사용합니다.
        <span class="mono">make_final_weights.py</span>가 생성하는 final weights CSV를 읽는 로직은 없습니다. [Source](https://raw.githubusercontent.com/gomtang-dori/Korea-Gomtang-Index/main/src/assemble/assemble_index.py)
        [Source](https://raw.githubusercontent.com/gomtang-dori/Korea-Gomtang-Index/main/src/analysis/make_final_weights.py)
      </div>

      <h3 style="margin:12px 0 6px; font-size:14px;">5-1) assemble_index.py의 고정 W (현재 실제 사용)</h3>
      {{ w_table | safe }}

      <h3 style="margin:12px 0 6px; font-size:14px;">5-2) final_weights CSV (있으면 표시 / assemble 반영 여부는 별도)</h3>
      {{ final_w_table | safe }}
      <div class="muted">
        final_weights CSV를 실제 지수에 반영하려면, assemble에 <span class="mono">WEIGHTS_CSV_PATH</span> 같은 로딩 로직을 추가해야 합니다.
      </div>
    </div>

    <div class="card full">
      <h2>6) 최근 {{ days }}일 샘플(지수/확률 컬럼)</h2>
      {{ sample_table | safe }}
    </div>

  </div>
</div>
</body>
</html>
"""
)


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or len(df) == 0:
        return '<div class="muted">데이터가 없습니다.</div>'
    d = df.head(max_rows).copy()
    return d.to_html(index=False, escape=False)


def main():
    tag = _env("TAG", "1Y")
    days = int(_env("DAYS", "180"))

    index_path = Path(_env("INDEX_PATH", f"data/index_daily_{tag}.parquet"))
    ml_metrics_path = Path(_env("ML_METRICS_PATH", f"data/analysis/ml_prob_up10_{tag}_metrics.csv"))

    final_weights_path = Path(_env("FINAL_WEIGHTS_PATH", f"data/analysis/final_weights_{tag}_kospi.csv"))
    out_html = Path(_env("OUT_HTML", f"docs/diagnostics_{tag}.html"))

    if not index_path.exists():
        raise RuntimeError(f"Missing index parquet: {index_path}")

    df = _ensure_date(pd.read_parquet(index_path))
    df180 = _tail_days(df, days)

    # KPI for prob_up_10d
    prob = df180["prob_up_10d"] if "prob_up_10d" in df180.columns else pd.Series(dtype=float)
    prob_nan_rate = float(prob.isna().mean()) if len(prob) else float("nan")
    prob_min = float(prob.min()) if len(prob.dropna()) else float("nan")
    prob_max = float(prob.max()) if len(prob.dropna()) else float("nan")

    idxs = df180["index_score_total"] if "index_score_total" in df180.columns else pd.Series(dtype=float)
    idx_min = float(idxs.min()) if len(idxs.dropna()) else float("nan")
    idx_max = float(idxs.max()) if len(idxs.dropna()) else float("nan")

    # Chart: index + prob
    fig = go.Figure()
    if "index_score_total" in df180.columns:
        fig.add_trace(go.Scatter(x=df180["date"], y=df180["index_score_total"], mode="lines", name="index_score_total", yaxis="y1"))
    if "prob_up_10d" in df180.columns:
        fig.add_trace(go.Scatter(x=df180["date"], y=df180["prob_up_10d"], mode="lines", name="prob_up_10d", yaxis="y2"))

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h"),
        yaxis=dict(title="Index (0~100)", range=[0, 100]),
        yaxis2=dict(title="prob_up_10d (0~1)", overlaying="y", side="right", range=[0, 1]),
    )
    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # ML metrics fold table
    m = _read_csv_if_exists(ml_metrics_path)
    if m is not None and len(m):
        m["auc"] = pd.to_numeric(m["auc"], errors="coerce")
        m["brier"] = pd.to_numeric(m["brier"], errors="coerce")
        auc_mean = _fmt(float(m["auc"].mean()), 4)
        auc_median = _fmt(float(m["auc"].median()), 4)
        brier_mean = _fmt(float(m["brier"].mean()), 4)
        brier_median = _fmt(float(m["brier"].median()), 4)
        fold_table = _df_to_html_table(m.tail(25))
    else:
        auc_mean = auc_median = brier_mean = brier_median = "-"
        fold_table = '<div class="muted">ml metrics csv가 없습니다.</div>'

    # assemble fixed W table (hard-coded here to mirror assemble_index.py)
    W = {
        "f01_score": 0.10,
        "f02_score": 0.075,
        "f03_score": 0.125,
        "f04_score": 0.10,
        "f05_score": 0.05,
        "f06_score": 0.125,
        "f07_score": 0.10,
        "f08_score": 0.10,
        "f09_score": 0.125,
        "f10_score": 0.10,
    }
    w_df = pd.DataFrame([{"score_col": k, "weight": v} for k, v in W.items()]).sort_values("score_col")
    w_table = _df_to_html_table(w_df, max_rows=30)

    # final weights table (if exists)
    fw = _read_csv_if_exists(final_weights_path)
    if fw is None or len(fw) == 0:
        final_w_table = '<div class="muted">final_weights CSV가 아직 없거나 경로가 다릅니다.</div>'
    else:
        final_w_table = _df_to_html_table(fw.sort_values(fw.columns[-1], ascending=False), max_rows=30)

    # sample table
    keep_cols = [c for c in ["date", "index_score_total", "bucket_5pt", "prob_up_10d", "w_f08_applied"] if c in df180.columns]
    for i in range(1, 11):
        c = f"f{i:02d}_score"
        if c in df180.columns:
            keep_cols.append(c)
    sample_table = _df_to_html_table(df180[keep_cols].tail(30), max_rows=30)

    html = HTML.render(
        tag=tag,
        days=days,
        index_path=str(index_path),
        ml_metrics_path=str(ml_metrics_path),
        fig_index_prob=fig_html,
        prob_nan_rate=_pct(prob_nan_rate),
        prob_min=_fmt(prob_min, 4),
        prob_max=_fmt(prob_max, 4),
        idx_min=_fmt(idx_min, 2),
        idx_max=_fmt(idx_max, 2),
        auc_mean=auc_mean,
        auc_median=auc_median,
        brier_mean=brier_mean,
        brier_median=brier_median,
        fold_table=fold_table,
        w_table=w_table,
        final_w_table=final_w_table,
        sample_table=sample_table,
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"[diagnostics] OK -> {out_html}")


if __name__ == "__main__":
    main()
