# src/report/render_report.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template


# ----------------------------
# HTML Template (single-file)
# ----------------------------
HTML_TMPL = Template(
    """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{{ title }}</title>
  <link rel="stylesheet" href="assets/style.css"/>
  <style>
    .grid-2 { display:grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .grid-3 { display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
    @media (max-width: 1100px) { .grid-3 { grid-template-columns: 1fr; } .grid-2 { grid-template-columns: 1fr; } }
    .kpi { font-variant-numeric: tabular-nums; }
    .kpi .big { font-size: 34px; font-weight: 800; letter-spacing: -0.5px; }
    .kpi .sub { color:#aab; font-size: 12px; margin-top: 2px; }
    .kpi .row { display:flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
    .pill { padding: 4px 9px; border-radius: 999px; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08); font-size: 12px; color:#dde; }
    .small { font-size: 12px; color:#aab; line-height: 1.5; }
    .sec-title { margin: 0 0 8px 0; font-size: 16px; font-weight: 800; }
    .hr { height:1px; background: rgba(255,255,255,0.08); margin: 14px 0; }
    table { border-collapse: collapse; width:100%; }
    th, td { padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.08); font-size: 12px; }
    th { text-align:left; color:#cdd; position: sticky; top: 0; background: rgba(10,14,24,0.96); }
    .mono { font-variant-numeric: tabular-nums; }
  </style>
</head>
<body>
<div class="container">
  <h1>{{ title }}</h1>
  <div class="small">기준: {{ today_date }} · 기간(lookback): {{ lookback_days }} trading days · eps(보합밴드): {{ eps_pct }}%</div>

  <div class="grid-2">
    <div class="card kpi">
      <div class="sec-title">오늘의 곰탕지수</div>
      <div class="big">{{ gomtang_today }}</div>
      <div class="sub">버킷(5점): {{ gomtang_bucket }}</div>
      <div class="row">
        <span class="pill">Δ3D: {{ gomtang_d3 }}</span>
        <span class="pill">Δ5D: {{ gomtang_d5 }}</span>
        <span class="pill">Δ10D: {{ gomtang_d10 }}</span>
      </div>
    </div>

    <div class="card kpi">
      <div class="sec-title">오늘의 KOSPI</div>
      <div class="big">{{ kospi_today }}</div>
      <div class="sub">최근 추세(원시값 기준 수익률)</div>
      <div class="row">
        <span class="pill">3D: {{ kospi_r3 }}</span>
        <span class="pill">5D: {{ kospi_r5 }}</span>
        <span class="pill">10D: {{ kospi_r10 }}</span>
      </div>
      {% if prob_today is not none %}
      <div class="hr"></div>
      <div class="sec-title">XGBoost 확률 (P(KOSPI 10D Up))</div>
      <div class="big">{{ prob_today }}</div>
      <div class="sub">prob_up_10d (0~1)</div>
      {% endif %}
    </div>
  </div>

  <div class="grid-3">
    <div class="card">
      <div class="sec-title">GOMTANG INDEX (Score 0~100)</div>
      {{ fig_gomtang | safe }}
    </div>
    <div class="card">
      <div class="sec-title">KOSPI (원시값)</div>
      {{ fig_kospi | safe }}
    </div>
    <div class="card">
      <div class="sec-title">KOSDAQ (원시값)</div>
      {{ fig_kosdaq | safe }}
    </div>
  </div>

  <div class="grid-2" style="margin-top:14px;">
    <div class="card">
      <div class="sec-title">KOSPI200 (K200, 원시값)</div>
      {{ fig_k200 | safe }}
    </div>
    <div class="card">
      <div class="sec-title">prob_up_10d 라인차트 (0~1)</div>
      {% if fig_prob %}
        {{ fig_prob | safe }}
      {% else %}
        <div class="small">prob_up_10d 컬럼이 없어 표시하지 않습니다.</div>
      {% endif %}
    </div>
  </div>

  <div class="grid-2" style="margin-top:14px;">
    <div class="card">
      <div class="sec-title">히트맵: (시장상태 × 곰탕버킷) → 10D 후 KOSPI 평균 변화율</div>
      <div class="small">X축 상태는 KOSPI 기준: 5D/3D 하락·상승 및 보합(±eps) 우선순위 분류</div>
      {{ fig_hm_mean | safe }}
    </div>
    <div class="card">
      <div class="sec-title">히트맵: (시장상태 × 곰탕버킷) → 10D 후 KOSPI 승률</div>
      {{ fig_hm_win | safe }}
    </div>
  </div>

  <div class="card" style="margin-top:14px;">
    <div class="sec-title">팩터 RAW(없으면 SCORE) — 각각 라인차트</div>
    <div class="small">
      - RAW 컬럼이 있으면 RAW를 우선 사용합니다. (예: f05_raw)<br/>
      - RAW가 없으면 SCORE(0~100)를 사용합니다.<br/>
      - 파일/컬럼이 없으면 자동 스킵합니다. (F02/F03은 adv/dec 캐시 워크플로우에서 생성될 수 있음)
      [Source](https://raw.githubusercontent.com/gomtang-dori/Korea-Gomtang-Index/main/.github/workflows/advdec_daily.yml)
    </div>
    <div class="hr"></div>
    {{ factors_block | safe }}
  </div>

  <div class="card" style="margin-top:14px;">
    <div class="sec-title">데이터/계산 메모</div>
    <div class="small">
      - 지수 계산(가중치/EMA/contrarian 생성/f08 동적 조정)은 assemble_index.py에서 수행됩니다.
      [Source](https://raw.githubusercontent.com/gomtang-dori/Korea-Gomtang-Index/main/src/assemble/assemble_index.py)<br/>
      - prob_up_10d는 train_xgb_prob_up10.py 산출 후 merge_prob_into_index.py에서 index parquet에 merge됩니다.
      [Source](https://raw.githubusercontent.com/gomtang-dori/Korea-Gomtang-Index/main/src/analysis/merge_prob_into_index.py)
    </div>
  </div>

</div>
</body>
</html>
"""
)

# ----------------------------
# Helpers
# ----------------------------
FACTOR_TAGS = [f"f{i:02d}" for i in range(1, 11)]


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip()
    return default if v == "" else v


def _to_dt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def _fmt_num(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and np.isnan(x):
        return "-"
    return f"{float(x):,.{nd}f}"


def _fmt_pct(x: Optional[float], nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{float(x) * 100:.{nd}f}%"


def _safe_last(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() == 0:
        return None
    return float(s.iloc[-1])


def _delta_n(df: pd.DataFrame, col: str, n: int) -> Optional[float]:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if len(s) <= n:
        return None
    a = s.iloc[-1]
    b = s.iloc[-1 - n]
    if not np.isfinite(a) or not np.isfinite(b):
        return None
    return float(a - b)


def _ret_n(df: pd.DataFrame, col: str, n: int) -> Optional[float]:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if len(s) <= n:
        return None
    a = s.iloc[-1]
    b = s.iloc[-1 - n]
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return None
    return float(a / b - 1.0)


def _slice_lookback(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if lookback_days <= 0:
        return df
    return df.tail(int(lookback_days)).reset_index(drop=True)


def _fig_line(df: pd.DataFrame, ycol: str, name: str, *, height: int = 320, y_range: Optional[Tuple[float, float]] = None) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df.get(ycol), mode="lines", name=name))
    layout = dict(height=height, margin=dict(l=20, r=20, t=10, b=20))
    if y_range is not None:
        layout["yaxis"] = dict(range=[y_range[0], y_range[1]])
    fig.update_layout(**layout)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _classify_market_state_kospi(ret3: pd.Series, ret5: pd.Series, eps: float) -> pd.Series:
    """
    우선순위 기반 단일 분류 (겹침 제거)
    1) 5D 하락 (ret5 <= -eps)
    2) 5D 상승 (ret5 >= +eps)
    3) 3D 하락 (ret3 <= -eps)
    4) 3D 상승 (ret3 >= +eps)
    5) 보합
    """
    s = pd.Series(["보합"] * len(ret3), index=ret3.index, dtype=object)

    m1 = ret5 <= -eps
    m2 = ret5 >= +eps
    m3 = ret3 <= -eps
    m4 = ret3 >= +eps

    s.loc[m1] = "5일 하락"
    s.loc[~m1 & m2] = "5일 상승"
    s.loc[~m1 & ~m2 & m3] = "3일 하락"
    s.loc[~m1 & ~m2 & ~m3 & m4] = "3일 상승"
    # 나머지 보합 유지
    return s


def _heatmap_2d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    agg: str,
    *,
    title: str,
    value_format: str,
    colorscale: str,
    zmid: Optional[float] = None,
) -> str:
    """
    agg: "mean" or "winrate"
    value_format: ".2%" or ".2f" etc
    """
    tmp = df[[x_col, y_col, value_col]].copy()
    tmp = tmp.dropna(subset=[x_col, y_col, value_col]).reset_index(drop=True)

    # pivot mean
    pivot_mean = tmp.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="mean")
    pivot_n = tmp.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="count")

    # sort y (bucket)
    try:
        pivot_mean = pivot_mean.sort_index()
        pivot_n = pivot_n.loc[pivot_mean.index]
    except Exception:
        pass

    x_order = ["5일 하락", "3일 하락", "보합", "3일 상승", "5일 상승"]
    cols = [c for c in x_order if c in pivot_mean.columns] + [c for c in pivot_mean.columns if c not in x_order]
    pivot_mean = pivot_mean[cols]
    pivot_n = pivot_n[cols]

    z = pivot_mean.to_numpy()
    n = pivot_n.to_numpy()

    # text: value + n
    text = []
    for i in range(z.shape[0]):
        row = []
        for j in range(z.shape[1]):
            if np.isnan(z[i, j]):
                row.append("")
            else:
                if value_format.endswith("%"):
                    # expects already in fraction (e.g., 0.0123)
                    row.append(f"{z[i,j]*100:.2f}%<br><span style='font-size:11px;color:#666'>n={int(n[i,j])}</span>")
                else:
                    row.append(f"{z[i,j]:.4f}<br><span style='font-size:11px;color:#666'>n={int(n[i,j])}</span>")
        text.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(pivot_mean.columns),
            y=[str(v) for v in pivot_mean.index],
            colorscale=colorscale,
            zmid=zmid,
            text=text,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        title=dict(text=title, x=0.01, xanchor="left"),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _read_factor_for_chart(factors_dir: Path, tag: str) -> Optional[Tuple[pd.DataFrame, str, str]]:
    """
    return: (df, ycol, label)
    - prefer raw: f##_raw
    - fallback score: f##_score
    """
    p = factors_dir / f"{tag}.parquet"
    if not p.exists():
        return None

    df = pd.read_parquet(p)
    if "date" not in df.columns:
        return None
    df = _to_dt(df)

    raw_col = f"{tag}_raw"
    score_col = f"{tag}_score"

    if raw_col in df.columns:
        ycol = raw_col
        label = f"{tag.upper()} RAW"
    elif score_col in df.columns:
        ycol = score_col
        label = f"{tag.upper()} SCORE (0~100)"
    else:
        return None

    out = df[["date", ycol]].copy()
    out[ycol] = pd.to_numeric(out[ycol], errors="coerce")
    out = out.dropna(subset=[ycol]).reset_index(drop=True)
    if len(out) < 30:
        return None
    return out, ycol, label


def _fwd_ret(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    index_path = Path(_env("INDEX_PATH", "data/index_daily.parquet"))
    out_html = Path(_env("REPORT_PATH", "docs/index.html"))

    # lookback settings
    lookback_days = int(_env("LOOKBACK_DAYS", "252"))
    eps = float(_env("EPS_FLAT", "0.002"))  # 0.2%
    eps_pct = eps * 100

    # index levels for KOSPI/KOSDAQ/K200
    index_levels_path = Path(_env("INDEX_LEVELS_PATH", "data/cache/index_levels.parquet"))

    # factors dir
    factors_dir = Path(_env("FACTORS_DIR", "data/factors"))

    title = _env("REPORT_TITLE", out_html.stem.replace("_", " ")).strip() or "Gomtang Index Report"

    if not index_path.exists():
        raise RuntimeError(f"Missing {index_path}. Run assemble first.")
    if not index_levels_path.exists():
        raise RuntimeError(f"Missing {index_levels_path}. Need index levels for charts/heatmaps.")

    # --------
    # Load data
    # --------
    df_idx = pd.read_parquet(index_path)
    df_idx = _to_dt(df_idx)

    df_levels = pd.read_parquet(index_levels_path)
    df_levels = _to_dt(df_levels)

    # 1Y/8Y 기간은 index_daily 날짜 범위를 기준으로 동일하게 맞춤(“각 리포트 기간만 사용”)
    # -> index_daily의 마지막 lookback_days에 맞춰 보고, levels도 그 날짜들로 inner join
    df_idx = _slice_lookback(df_idx, lookback_days)
    # levels는 idx에 맞춰 날짜 필터
    if len(df_idx) > 0:
        start_date = df_idx["date"].min()
        end_date = df_idx["date"].max()
        df_levels = df_levels[(df_levels["date"] >= start_date) & (df_levels["date"] <= end_date)].reset_index(drop=True)

    # join for heatmap computations
    base = df_idx.merge(df_levels, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # Ensure numeric
    for c in ["index_score_total", "bucket_5pt", "prob_up_10d", "kospi_close", "kosdaq_close", "k200_close"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    # today date
    today_date = "-"
    if len(base) > 0:
        today_date = pd.to_datetime(base["date"].iloc[-1]).strftime("%Y-%m-%d")

    # --------
    # KPI cards
    # --------
    g_today = _safe_last(base, "index_score_total")
    g_bucket = None
    if "bucket_5pt" in base.columns and base["bucket_5pt"].notna().sum() > 0:
        g_bucket = float(base["bucket_5pt"].dropna().iloc[-1])
    g_bucket_str = "-" if g_bucket is None or np.isnan(g_bucket) else f"{g_bucket:.0f} ~ {g_bucket+5:.0f}"

    g_d3 = _delta_n(base, "index_score_total", 3)
    g_d5 = _delta_n(base, "index_score_total", 5)
    g_d10 = _delta_n(base, "index_score_total", 10)

    kospi_today = _safe_last(base, "kospi_close")
    kospi_r3 = _ret_n(base, "kospi_close", 3)
    kospi_r5 = _ret_n(base, "kospi_close", 5)
    kospi_r10 = _ret_n(base, "kospi_close", 10)

    prob_today = _safe_last(base, "prob_up_10d") if "prob_up_10d" in base.columns else None

    # --------
    # Figures (each separate)
    # --------
    # include_plotlyjs only once (first figure)
    fig_gomtang = go.Figure()
    fig_gomtang.add_trace(go.Scatter(x=base["date"], y=base.get("index_score_total"), mode="lines", name="GOMTANG"))
    fig_gomtang.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=20), yaxis=dict(range=[0, 100]))
    fig_gomtang_html = fig_gomtang.to_html(full_html=False, include_plotlyjs="cdn")

    # KOSPI
    fig_kospi_html = _fig_line(base, "kospi_close", "KOSPI", height=320)

    # KOSDAQ
    fig_kosdaq_html = _fig_line(base, "kosdaq_close", "KOSDAQ", height=320)

    # K200
    fig_k200_html = _fig_line(base, "k200_close", "KOSPI200 (K200)", height=320)

    # prob_up_10d
    fig_prob_html = ""
    if "prob_up_10d" in base.columns and base["prob_up_10d"].notna().sum() > 10:
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(x=base["date"], y=base["prob_up_10d"], mode="lines", name="prob_up_10d"))
        fig_prob.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=20), yaxis=dict(range=[0, 1]))
        fig_prob_html = fig_prob.to_html(full_html=False, include_plotlyjs=False)
    else:
        fig_prob_html = ""

    # --------
    # Heatmaps (KOSPI-based)
    # --------
    # market states
    if "kospi_close" not in base.columns or base["kospi_close"].notna().sum() < 200:
        # fallback empty heatmaps
        fig_hm_mean_html = go.Figure().to_html(full_html=False, include_plotlyjs=False)
        fig_hm_win_html = go.Figure().to_html(full_html=False, include_plotlyjs=False)
    else:
        close = pd.to_numeric(base["kospi_close"], errors="coerce")
        ret3 = close / close.shift(3) - 1.0
        ret5 = close / close.shift(5) - 1.0
        state = _classify_market_state_kospi(ret3, ret5, eps=eps)
        base["market_state"] = state

        # y bucket
        if "bucket_5pt" not in base.columns:
            # create from index score if missing
            s = pd.to_numeric(base["index_score_total"], errors="coerce")
            base["bucket_5pt"] = (np.floor(s / 5.0) * 5.0).clip(0, 100)

        # 10D forward return
        base["kospi_fwd10"] = _fwd_ret(close, horizon=10)
        base["kospi_win10"] = (base["kospi_fwd10"] > 0).astype(float)

        hm_df = base.dropna(subset=["market_state", "bucket_5pt", "kospi_fwd10"]).copy()
        hm_df["bucket_5pt"] = pd.to_numeric(hm_df["bucket_5pt"], errors="coerce")

        fig_hm_mean_html = _heatmap_2d(
            hm_df,
            x_col="market_state",
            y_col="bucket_5pt",
            value_col="kospi_fwd10",
            agg="mean",
            title="10D 후 KOSPI 평균 변화율 (셀 텍스트: 값 + n)",
            value_format="%",
            colorscale="RdBu",
            zmid=0.0,
        )

        fig_hm_win_html = _heatmap_2d(
            hm_df,
            x_col="market_state",
            y_col="bucket_5pt",
            value_col="kospi_win10",
            agg="winrate",
            title="10D 후 KOSPI 승률 (셀 텍스트: 승률 + n)",
            value_format="%",
            colorscale="Greens",
            zmid=None,
        )

    # --------
    # Factors block: each chart (RAW preferred)
    # --------
    factor_cards_html: List[str] = []
    for tag in FACTOR_TAGS:
        loaded = _read_factor_for_chart(factors_dir, tag)
        if loaded is None:
            continue
        dff, ycol, label = loaded

        # align factor to report range (optional): match base date range
        if len(base) > 0:
            start_date = base["date"].min()
            end_date = base["date"].max()
            dff = dff[(dff["date"] >= start_date) & (dff["date"] <= end_date)].reset_index(drop=True)
            if len(dff) < 30:
                continue

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dff["date"], y=dff[ycol], mode="lines", name=label))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
        fig_html = fig.to_html(full_html=False, include_plotlyjs=False)

        factor_cards_html.append(
            f"""
            <div class="card" style="margin: 12px 0;">
              <div class="sec-title">{label}</div>
              {fig_html}
              <div class="small mono">file: data/factors/{tag}.parquet · col: {ycol}</div>
            </div>
            """
        )

    if not factor_cards_html:
        factors_block = "<div class='small'>표시할 팩터 파일/컬럼이 없습니다.</div>"
    else:
        factors_block = "\n".join(factor_cards_html)

    # --------
    # Render HTML
    # --------
    html = HTML_TMPL.render(
        title=title,
        today_date=today_date,
        lookback_days=lookback_days,
        eps_pct=f"{eps_pct:.2f}",
        gomtang_today=_fmt_num(g_today, 2),
        gomtang_bucket=g_bucket_str,
        gomtang_d3=_fmt_num(g_d3, 2),
        gomtang_d5=_fmt_num(g_d5, 2),
        gomtang_d10=_fmt_num(g_d10, 2),
        kospi_today=_fmt_num(kospi_today, 2),
        kospi_r3=_fmt_pct(kospi_r3, 2),
        kospi_r5=_fmt_pct(kospi_r5, 2),
        kospi_r10=_fmt_pct(kospi_r10, 2),
        prob_today=None if prob_today is None else _fmt_num(prob_today, 4),
        fig_gomtang=fig_gomtang_html,
        fig_kospi=fig_kospi_html,
        fig_kosdaq=fig_kosdaq_html,
        fig_k200=fig_k200_html,
        fig_prob=fig_prob_html,
        fig_hm_mean=fig_hm_mean_html,
        fig_hm_win=fig_hm_win_html,
        factors_block=factors_block,
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"[report] OK -> {out_html}")


if __name__ == "__main__":
    main()
