# src/daily_update.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from jinja2 import Template
import plotly.graph_objects as go

# existing local modules (you already have pykrx_factors.py in repo)
from lib.pykrx_factors import (
    fetch_kospi200_ohlcv,
    factor1_momentum,
    factor2_strength,
    factor3_breadth,
    factor6_volatility,
    factor7_safehaven,
    factor8_foreign_netbuy,
)
from lib.krx_putcall import fetch_putcall_ratio_by_date


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    REFRESH_DAYS: int = 14
    ROLLING_DAYS: int = 252 * 5
    MIN_OBS: int = 252

    DATA_DIR: str = "data"
    DOCS_DIR: str = "docs"
    ASSETS_DIR: str = "docs/assets"

    # Factor weights (f09 excluded by design)
    # We'll renormalize among available factors each day.
    W: dict = None

    def __post_init__(self):
        if self.W is None:
            self.W = {
                "f01_score": 0.10,
                "f02_score": 0.10,
                "f03_score": 0.10,
                "f04_score": 0.10,  # Put/Call (new)
                "f05_score": 0.05,
                "f06_score": 0.125,
                "f07_score": 0.10,
                "f08_score": 0.10,
                "f10_score": 0.10,
            }


cfg = CFG()


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def rolling_percentile(s: pd.Series, window: int, min_obs: int) -> pd.Series:
    # Percentile rank of latest value within rolling window (0-100)
    def _pct(x):
        if len(x) < min_obs:
            return np.nan
        v = x.iloc[-1]
        return float(pd.Series(x).rank(pct=True).iloc[-1] * 100.0)

    return s.rolling(window=window, min_periods=min_obs).apply(_pct, raw=False)


def pct_change_n(s: pd.Series, n: int) -> pd.Series:
    return s.pct_change(n)


def forward_return(level: pd.Series, n: int) -> pd.Series:
    return level.shift(-n) / level - 1.0


def forward_win(level: pd.Series, n: int) -> pd.Series:
    return (forward_return(level, n) > 0).astype(float)


def load_parquet(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def save_parquet(df: pd.DataFrame, path: str | Path):
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False)


def safe_to_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def upsert_timeseries(old: pd.DataFrame, new: pd.DataFrame, key: str = "date") -> pd.DataFrame:
    if old is None or old.empty:
        out = new.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)
    out = safe_to_datetime(out, key)
    out = out.dropna(subset=[key])
    out = out.drop_duplicates(subset=[key], keep="last")
    out = out.sort_values(key).reset_index(drop=True)
    return out


def renormalize_weights(df: pd.DataFrame, weights: dict) -> pd.Series:
    # For each row: use only factors where score is not null
    score_cols = list(weights.keys())
    w = pd.Series(weights, dtype=float)

    avail = df[score_cols].notna().astype(float)
    w_mat = avail.mul(w, axis=1)
    w_sum = w_mat.sum(axis=1).replace(0, np.nan)
    w_norm = w_mat.div(w_sum, axis=0)
    return w_norm


# -----------------------------
# Factor 5 / 10 placeholders
# NOTE: In your repo you already have factor5/factor10 working.
# To keep this replacement self-contained and safe, we load from existing index_daily if present.
# If you already fetch f05/f10 inside daily_update previously, you can re-add your fetch functions here.
# -----------------------------
def load_existing_f05_f10_from_index(index_df: pd.DataFrame) -> pd.DataFrame:
    keep = ["date", "f05_raw", "f10_raw"]
    cols = [c for c in keep if c in index_df.columns]
    if not cols:
        return pd.DataFrame(columns=["date"])
    return index_df[cols].copy()


# -----------------------------
# Report
# -----------------------------
HTML_TMPL = Template(
    """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Gomtang Index Report</title>
  <link rel="stylesheet" href="assets/style.css"/>
</head>
<body>
  <div class="container">
    <h1>Gomtang Index (KOSPI200 기반)</h1>

    <div class="grid">
      <div class="card">
        <h2>오늘의 지수</h2>
        <div class="big">{{ today_score }}</div>
        <div class="muted">기준일: {{ today_date }}</div>
        <div class="muted">점수구간(5점): {{ today_bucket }}</div>
      </div>

      <div class="card">
        <h2>최근 추세</h2>
        <div>3일: {{ idx_3d }}</div>
        <div>5일: {{ idx_5d }}</div>
        <div>7일: {{ idx_7d }}</div>
      </div>

      <div class="card">
        <h2>KOSPI200 수익률</h2>
        <div>3일: {{ k3 }}</div>
        <div>5일: {{ k5 }}</div>
        <div>7일: {{ k7 }}</div>
      </div>

      <div class="card">
        <h2>투자의견(룰 기반)</h2>
        <div class="opinion">{{ opinion }}</div>
        <div class="muted">{{ opinion_reason }}</div>
      </div>
    </div>

    <div class="card">
      <h2>지수 라인차트</h2>
      {{ fig_index | safe }}
    </div>

    <div class="card">
      <h2>지수 구성요소(컴포넌트) 라인차트</h2>
      {{ fig_components | safe }}
    </div>

    <div class="card">
      <h2>팩터④ Put/Call (거래대금 기준) — 최근 14일 재조회 적용</h2>
      <div class="muted">
        KRX 주식옵션(유가/코스닥) 일별매매정보 API의 OutBlock_1에서
        RGHT_TP_NM(CALL/PUT), ACC_TRDVAL(거래대금)을 사용해 Put/Call ratio를 산출합니다.
        [Source] https://www.genspark.ai/api/files/s/6ynCkTUc
      </div>
    </div>

    <div class="footer muted">
      KRX 옵션 API Spec endpoint:
      유가 https://data-dbg.krx.co.kr/svc/apis/drv/eqsop_bydd_trd,
      코스닥 https://data-dbg.krx.co.kr/svc/apis/drv/eqkop_bydd_trd
      (요청 basDd, 응답 OutBlock_1) [Source](https://www.genspark.ai/api/files/s/08Toc4xA)
    </div>
  </div>
</body>
</html>
"""
)


def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x*100:,.2f}%"


def fmt_score(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.1f}"


def make_opinion(today_score: float, k200_ret_5d: float) -> tuple[str, str]:
    # simple rule-based: low score + recent weakness => contrarian buy
    if np.isnan(today_score):
        return ("중립", "지수 데이터가 부족합니다.")
    if today_score <= 20 and (np.isnan(k200_ret_5d) or k200_ret_5d <= 0):
        return ("매수 우위", "지수가 낮은 구간(공포)이며 최근 수익률이 약합니다(역발상).")
    if today_score >= 80 and (np.isnan(k200_ret_5d) or k200_ret_5d >= 0):
        return ("매도/경계", "지수가 높은 구간(탐욕)이며 과열 가능성에 유의합니다.")
    return ("중립", "명확한 극단 구간이 아닙니다.")


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dir(cfg.DATA_DIR)
    ensure_dir(cfg.DOCS_DIR)
    ensure_dir(cfg.ASSETS_DIR)

    index_path = Path(cfg.DATA_DIR) / "index_daily.parquet"
    old = load_parquet(index_path)
    old = safe_to_datetime(old, "date")

    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = today - pd.Timedelta(days=cfg.REFRESH_DAYS + 30)  # extra buffer for holidays

    # ---- KOSPI200 proxy (ETF 069500) ----
    k200 = fetch_kospi200_ohlcv(start.strftime("%Y%m%d"), today.strftime("%Y%m%d"))
    k200 = safe_to_datetime(k200, "date")
    if "k200_close" not in k200.columns:
        # robust fallback
        for cand in ["종가", "close", "Close"]:
            if cand in k200.columns:
                k200 = k200.rename(columns={cand: "k200_close"})
                break
    k200["k200_close"] = pd.to_numeric(k200.get("k200_close"), errors="coerce")
    k200 = k200.dropna(subset=["date", "k200_close"]).sort_values("date").reset_index(drop=True)

    # ---- pykrx factors (1/2/3/6/7/8) ----
    # These functions should return columns: date, f01_raw ... etc (as in your existing file)
    f01 = factor1_momentum(start.strftime("%Y%m%d"), today.strftime("%Y%m%d"))
    f02 = factor2_strength(start.strftime("%Y%m%d"), today.strftime("%Y%m%d"))
    f03 = factor3_breadth(start.strftime("%Y%m%d"), today.strftime("%Y%m%d"))
    f06 = factor6_volatility(start.strftime("%Y%m%d"), today.strftime("%Y%m%d"))
    f07 = factor7_safehaven(start.strftime("%Y%m%d"), today.strftime("%Y%m%d"))
    f08 = factor8_foreign_netbuy(start.strftime("%Y%m%d"), today.strftime("%Y%m%d"))

    for df in [f01, f02, f03, f06, f07, f08]:
        safe_to_datetime(df, "date")

    # ---- Factor 4 (Put/Call) from KRX OpenAPI (recent 14 days) ----
    f04 = fetch_putcall_ratio_by_date(start, today)
    f04 = safe_to_datetime(f04, "date")

    # ---- Factor 5/10: keep from existing index parquet if present ----
    # (You can later re-add your true fetchers here.)
    f05f10 = load_existing_f05_f10_from_index(old)

    # ---- Merge base ----
    base = k200[["date", "k200_close"]].copy()
    for add in [f01, f02, f03, f04, f06, f07, f08, f05f10]:
        if add is None or add.empty:
            continue
        base = base.merge(add, on="date", how="left")

    base = base.sort_values("date").reset_index(drop=True)

    # ---- K200 derived ----
    if ("k200_close" not in base.columns) or (base["k200_close"].dropna().empty):
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

    # ---- Scores (rolling percentile 0-100) ----
    # raw columns assumed:
    # f01_raw f02_raw f03_raw f04_raw f05_raw f06_raw f07_raw f08_raw f10_raw
    for raw, score in [
        ("f01_raw", "f01_score"),
        ("f02_raw", "f02_score"),
        ("f03_raw", "f03_score"),
        ("f04_raw", "f04_score"),
        ("f05_raw", "f05_score"),
        ("f06_raw", "f06_score"),
        ("f07_raw", "f07_score"),
        ("f08_raw", "f08_score"),
        ("f10_raw", "f10_score"),
    ]:
        if raw in base.columns:
            base[raw] = pd.to_numeric(base[raw], errors="coerce")
            base[score] = rolling_percentile(base[raw], cfg.ROLLING_DAYS, cfg.MIN_OBS)
        else:
            base[score] = np.nan

    # ---- Index aggregation with per-row renormalized weights ----
    w_norm = renormalize_weights(base, cfg.W)
    score_cols = list(cfg.W.keys())
    base["index_score_total"] = (base[score_cols] * w_norm).sum(axis=1)

    # ---- Buckets (5pt) ----
    base["bucket_5pt"] = (np.floor(base["index_score_total"] / 5.0) * 5.0).clip(0, 100)

    # ---- Merge into old full history (upsert) ----
    out = upsert_timeseries(old, base, "date")
    save_parquet(out, index_path)

    # ---- Report inputs (today last available) ----
    last = out.dropna(subset=["index_score_total"]).tail(1)
    if last.empty:
        today_date = "-"
        today_score = "-"
        today_bucket = "-"
        idx_3d = idx_5d = idx_7d = "-"
        k3 = k5 = k7 = "-"
        opinion, opinion_reason = ("중립", "데이터가 아직 충분하지 않습니다.")
    else:
        row = last.iloc[0]
        today_date = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        today_score = fmt_score(float(row["index_score_total"]))
        today_bucket = fmt_score(float(row["bucket_5pt"])) + " ~ " + fmt_score(float(row["bucket_5pt"]) + 5)

        idx_3d = fmt_score(float(out["index_score_total"].pct_change(3).iloc[-1] * 100)) + "%"
        idx_5d = fmt_score(float(out["index_score_total"].pct_change(5).iloc[-1] * 100)) + "%"
        idx_7d = fmt_score(float(out["index_score_total"].pct_change(7).iloc[-1] * 100)) + "%"

        k3 = fmt_pct(float(row.get("k200_ret_3d", np.nan)))
        k5 = fmt_pct(float(row.get("k200_ret_5d", np.nan)))
        k7 = fmt_pct(float(row.get("k200_ret_7d", np.nan)))

        opinion, opinion_reason = make_opinion(float(row["index_score_total"]), float(row.get("k200_ret_5d", np.nan)))

    # ---- Plotly charts ----
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=out["date"], y=out["index_score_total"], mode="lines", name="Index Score"))
    fig1.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(range=[0, 100]))

    fig2 = go.Figure()
    for c, n in [
        ("f01_score", "F01"),
        ("f02_score", "F02"),
        ("f03_score", "F03"),
        ("f04_score", "F04 Put/Call"),
        ("f05_score", "F05"),
        ("f06_score", "F06"),
        ("f07_score", "F07"),
        ("f08_score", "F08"),
        ("f10_score", "F10"),
    ]:
        if c in out.columns:
            fig2.add_trace(go.Scatter(x=out["date"], y=out[c], mode="lines", name=n))
    fig2.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(range=[0, 100]))

    fig_index_html = fig1.to_html(full_html=False, include_plotlyjs="cdn")
    fig_components_html = fig2.to_html(full_html=False, include_plotlyjs=False)

    html = HTML_TMPL.render(
        today_date=today_date,
        today_score=today_score,
        today_bucket=today_bucket,
        idx_3d=idx_3d,
        idx_5d=idx_5d,
        idx_7d=idx_7d,
        k3=k3,
        k5=k5,
        k7=k7,
        opinion=opinion,
        opinion_reason=opinion_reason,
        fig_index=fig_index_html,
        fig_components=fig_components_html,
    )

    out_html = Path(cfg.DOCS_DIR) / "index.html"
    out_css = Path(cfg.ASSETS_DIR) / "style.css"

    # keep your existing css if present; create minimal if missing
    ensure_dir(out_css.parent)
    if not out_css.exists():
        out_css.write_text(
            """:root{--bg:#0b1220;--card:#121a2b;--txt:#e9eefc;--muted:#93a4c7;--accent:#62d2ff;}
body{background:var(--bg);color:var(--txt);font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;margin:0;}
.container{max-width:1100px;margin:0 auto;padding:24px;}
h1{margin:0 0 16px 0;}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:12px;}
.card{background:var(--card);border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:14px;}
.big{font-size:44px;font-weight:800;color:var(--accent);line-height:1.1;}
.muted{color:var(--muted);font-size:13px;}
.opinion{font-size:20px;font-weight:700;}
.footer{margin-top:18px;}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr);}}
@media(max-width:520px){.grid{grid-template-columns:repeat(1,1fr);}}
""",
            encoding="utf-8",
        )

    out_html.write_text(html, encoding="utf-8")
    print("[daily_update] OK ->", out_html)


if __name__ == "__main__":
    main()
