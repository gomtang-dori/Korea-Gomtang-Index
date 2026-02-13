# src/daily_update.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template

from lib.pykrx_factors import (
    factor1_momentum,          # (k200_df)
    factor2_strength,          # (start_date, end_date)
    factor3_breadth,           # (start_date, end_date)
    factor6_volatility,        # (k200_df)  -> will be saved as f06_alt_raw (optional)
    factor7_safe_haven,        # (k200_df, usdkrw_df)
    factor8_foreign_netbuy,    # (start_date, end_date)
)

from lib.krx_putcall import fetch_putcall_ratio_by_date
from lib.krx_kospi_index import KRXKospiIndexAPI


@dataclass
class CFG:
    REFRESH_DAYS: int = 14
    ROLLING_DAYS: int = 252 * 5
    MIN_OBS: int = 252
    DATA_DIR: str = "data"
    DOCS_DIR: str = "docs"
    ASSETS_DIR: str = "docs/assets"
    USDKRW_LEVEL_PATH: str = "data/usdkrw_level.parquet"
    VKOSPI_LEVEL_PATH: str = "data/vkospi_level.parquet"
    K200_LOOKBACK_EXTRA_DAYS: int = 30

    W: dict = None

    def __post_init__(self):
        if self.W is None:
            self.W = {
                "f01_score": 0.10,
                "f02_score": 0.10,
                "f03_score": 0.10,
                "f04_score": 0.10,
                "f05_score": 0.05,
                "f06_score": 0.125,  # VKOSPI score
                "f07_score": 0.10,
                "f08_score": 0.10,
                "f10_score": 0.10,
            }


cfg = CFG()


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def safe_to_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[col])
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_parquet(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def save_parquet(df: pd.DataFrame, path: str | Path):
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False)


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


def rolling_percentile(s: pd.Series, window: int, min_obs: int) -> pd.Series:
    def _pct(x):
        if len(x) < min_obs:
            return np.nan
        return float(pd.Series(x).rank(pct=True).iloc[-1] * 100.0)
    return s.rolling(window=window, min_periods=min_obs).apply(_pct, raw=False)


def forward_return(level: pd.Series, n: int) -> pd.Series:
    return level.shift(-n) / level - 1.0


def forward_win(level: pd.Series, n: int) -> pd.Series:
    return (forward_return(level, n) > 0).astype(float)


def renormalize_weights(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    score_cols = list(weights.keys())
    w = pd.Series(weights, dtype=float)
    avail = df[score_cols].notna().astype(float)
    w_mat = avail.mul(w, axis=1)
    w_sum = w_mat.sum(axis=1).replace(0, np.nan)
    return w_mat.div(w_sum, axis=0)


def load_existing_f05_f10(index_df: pd.DataFrame) -> pd.DataFrame:
    keep = ["date", "f05_raw", "f10_raw"]
    cols = [c for c in keep if c in index_df.columns]
    if not cols:
        return pd.DataFrame(columns=["date"])
    return index_df[cols].copy()


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
      <h2>KOSPI200 수익률</h2>
      <div>3일: {{ k3 }}</div>
      <div>5일: {{ k5 }}</div>
      <div>7일: {{ k7 }}</div>
    </div>
    <div class="card">
      <h2>최근 추세(지수)</h2>
      <div>3일: {{ idx3 }}</div>
      <div>5일: {{ idx5 }}</div>
      <div>7일: {{ idx7 }}</div>
    </div>
    <div class="card">
      <h2>데이터 소스</h2>
      <div class="muted">
        VKOSPI(코스피 200 변동성지수)는 KRX OpenAPI 파생상품지수 시세정보에서 IDX_NM 필터로 수집 (종가=CLSPRC_IDX). [Source](https://www.genspark.ai/api/files/s/uX7923Iq)
      </div>
    </div>
  </div>

  <div class="card">
    <h2>지수 라인차트</h2>
    {{ fig_index | safe }}
  </div>

  <div class="card">
    <h2>구성요소(팩터 점수) 라인차트</h2>
    {{ fig_components | safe }}
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


def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x*100:,.2f}%"


def main():
    ensure_dir(cfg.DATA_DIR)
    ensure_dir(cfg.DOCS_DIR)
    ensure_dir(cfg.ASSETS_DIR)

    index_path = Path(cfg.DATA_DIR) / "index_daily.parquet"
    old = load_parquet(index_path)
    old = safe_to_datetime(old, "date")

    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = today - pd.Timedelta(days=cfg.REFRESH_DAYS + cfg.K200_LOOKBACK_EXTRA_DAYS)
    start_str = start.strftime("%Y%m%d")
    end_str = today.strftime("%Y%m%d")

    # ---- USD/KRW level (mandatory) ----
    usd_path = Path(cfg.USDKRW_LEVEL_PATH)
    if not usd_path.exists():
        raise RuntimeError(f"Missing {usd_path}. Workflow must run usdkrw_fetch.py before daily_update.py")
    usdkrw = pd.read_parquet(usd_path)
    usdkrw = safe_to_datetime(usdkrw, "date")
    if "usdkrw" not in usdkrw.columns:
        raise RuntimeError("usdkrw_level.parquet missing 'usdkrw' column")
    usdkrw["usdkrw"] = pd.to_numeric(usdkrw["usdkrw"], errors="coerce")
    usdkrw = usdkrw.dropna(subset=["date", "usdkrw"]).sort_values("date").reset_index(drop=True)

    # ---- VKOSPI level (mandatory for f06) ----
    vko_path = Path(cfg.VKOSPI_LEVEL_PATH)
    if not vko_path.exists():
        raise RuntimeError(f"Missing {vko_path}. Workflow must run vkospi_fetch.py before daily_update.py")
    vkospi = pd.read_parquet(vko_path)
    vkospi = safe_to_datetime(vkospi, "date")
    if "vkospi" not in vkospi.columns:
        raise RuntimeError("vkospi_level.parquet missing 'vkospi' column")
    vkospi["vkospi"] = pd.to_numeric(vkospi["vkospi"], errors="coerce")
    vkospi = vkospi.dropna(subset=["date", "vkospi"]).sort_values("date").reset_index(drop=True)

    # ---- KOSPI200 Close from KRX OpenAPI (stable) ----
    api = KRXKospiIndexAPI.from_env()
    k200 = api.fetch_k200_close_range(start, today)
    k200 = safe_to_datetime(k200, "date")
    k200["k200_close"] = pd.to_numeric(k200.get("k200_close"), errors="coerce")
    k200 = k200.dropna(subset=["date", "k200_close"]).sort_values("date").reset_index(drop=True)

    if k200.empty:
        raise RuntimeError("K200 close series is empty. Check KRX_KOSPI_DD_TRD_URL / KRX_AUTH_KEY / API approval.")

    # ---- Factors ----
    f01 = factor1_momentum(k200)
    f02 = factor2_strength(start_str, end_str)
    f03 = factor3_breadth(start_str, end_str)

    # (optional) keep realized volatility as alt raw (not used in index)
    f06_alt = factor6_volatility(k200)
    if f06_alt is not None and not f06_alt.empty and "f06_raw" in f06_alt.columns:
        f06_alt = f06_alt.rename(columns={"f06_raw": "f06_alt_raw"})

    f07 = factor7_safe_haven(k200, usdkrw)
    f08 = factor8_foreign_netbuy(start_str, end_str)

    for df in [f01, f02, f03, f06_alt, f07, f08]:
        if df is None:
            continue
        safe_to_datetime(df, "date")

    # ---- Factor4 Put/Call (recent refresh) ----
    f04 = fetch_putcall_ratio_by_date(start, today)
    f04 = safe_to_datetime(f04, "date")

    # ---- Factor6 VKOSPI raw (level) ----
    f06 = vkospi[["date", "vkospi"]].copy().rename(columns={"vkospi": "f06_raw"})

    # ---- Factor5/10 existing in index parquet (optional) ----
    f05f10 = load_existing_f05_f10(old)

    # ---- Build base ----
    base = k200[["date", "k200_close"]].copy()
    for add in [f01, f02, f03, f04, f06, f06_alt, f07, f08, f05f10]:
        if add is None or add.empty:
            continue
        if "k200_close" in add.columns and "k200_close" in base.columns:
            add = add.drop(columns=["k200_close"])
        base = base.merge(add, on="date", how="left")

    base = base.sort_values("date").reset_index(drop=True)

    # ---- Derived K200 ----
    base["k200_ret_3d"] = base["k200_close"].pct_change(3)
    base["k200_ret_5d"] = base["k200_close"].pct_change(5)
    base["k200_ret_7d"] = base["k200_close"].pct_change(7)
    base["k200_fwd_10d_return"] = forward_return(base["k200_close"], 10)
    base["k200_fwd_10d_win"] = forward_win(base["k200_close"], 10)

    # ---- Scores (Fear->Greed flip applied where needed) ----
    # flip list: f04 (PCR), f05 (spread), f06 (VKOSPI), f07 (safehaven raw), f10 (fx vol) => 100 - percentile
    flip_scores = {"f04_score", "f05_score", "f06_score", "f07_score", "f10_score"}

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
            pct = rolling_percentile(base[raw], cfg.ROLLING_DAYS, cfg.MIN_OBS)
            base[score] = 100.0 - pct if score in flip_scores else pct
        else:
            base[score] = np.nan

    w_norm = renormalize_weights(base, cfg.W)
    score_cols = list(cfg.W.keys())
    base["index_score_total"] = (base[score_cols] * w_norm).sum(axis=1)
    base["bucket_5pt"] = (np.floor(base["index_score_total"] / 5.0) * 5.0).clip(0, 100)

    out = upsert_timeseries(old, base, "date")
    save_parquet(out, index_path)

    # ---- Report ----
    last = out.dropna(subset=["index_score_total"]).tail(1)
    if last.empty:
        today_date = today_score = today_bucket = "-"
        k3 = k5 = k7 = "-"
        idx3 = idx5 = idx7 = "-"
    else:
        r = last.iloc[0]
        today_date = pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
        today_score = fmt_score(float(r["index_score_total"]))
        b = float(r["bucket_5pt"])
        today_bucket = f"{b:.0f} ~ {b+5:.0f}"
        k3 = fmt_pct(float(r.get("k200_ret_3d", np.nan)))
        k5 = fmt_pct(float(r.get("k200_ret_5d", np.nan)))
        k7 = fmt_pct(float(r.get("k200_ret_7d", np.nan)))
        idx3 = fmt_pct(float(out["index_score_total"].pct_change(3).iloc[-1]))
        idx5 = fmt_pct(float(out["index_score_total"].pct_change(5).iloc[-1]))
        idx7 = fmt_pct(float(out["index_score_total"].pct_change(7).iloc[-1]))

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=out["date"], y=out["index_score_total"], mode="lines", name="Index Score"))
    fig1.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(range=[0, 100]))
    fig_index_html = fig1.to_html(full_html=False, include_plotlyjs="cdn")

    fig2 = go.Figure()
    for c, n in [
        ("f01_score", "F01"),
        ("f02_score", "F02"),
        ("f03_score", "F03"),
        ("f04_score", "F04 Put/Call"),
        ("f05_score", "F05 Spread"),
        ("f06_score", "F06 VKOSPI"),
        ("f07_score", "F07 SafeHaven"),
        ("f08_score", "F08 Foreign"),
        ("f10_score", "F10 FXVol"),
    ]:
        if c in out.columns:
            fig2.add_trace(go.Scatter(x=out["date"], y=out[c], mode="lines", name=n))
    fig2.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(range=[0, 100]))
    fig_components_html = fig2.to_html(full_html=False, include_plotlyjs=False)

    html = HTML_TMPL.render(
        today_date=today_date,
        today_score=today_score,
        today_bucket=today_bucket,
        k3=k3,
        k5=k5,
        k7=k7,
        idx3=idx3,
        idx5=idx5,
        idx7=idx7,
        fig_index=fig_index_html,
        fig_components=fig_components_html,
    )

    out_html = Path(cfg.DOCS_DIR) / "index.html"
    out_html.write_text(html, encoding="utf-8")
    print("[daily_update] OK ->", out_html)


if __name__ == "__main__":
    main()
