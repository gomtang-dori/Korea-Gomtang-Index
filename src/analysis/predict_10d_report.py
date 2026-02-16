# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


FACTOR_SCORE_COLS = {
    "f01": "f01_score",
    "f02": "f02_score",
    "f03": "f03_score",
    "f04": "f04_score",
    "f05": "f05_score",
    "f06": "f06_score",
    "f07": "f07_score",
    "f08": "f08_score",
    "f09": "f09_score",
    "f10": "f10_score",
}

INDEX_LEVEL_COLS = ["kospi_close", "kosdaq_close", "k200_close"]


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name, default) or "").strip()


def _parse_list(raw: str) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _safe_to_datetime(s: str) -> pd.Timestamp | None:
    if not s:
        return None
    return pd.to_datetime(s, errors="coerce")


def _read_factor_scores(factors_dir: Path, exclude: set[str]) -> pd.DataFrame:
    frames = []
    for tag, col in FACTOR_SCORE_COLS.items():
        if tag in exclude:
            continue
        p = factors_dir / f"{tag}.parquet"
        if not p.exists():
            continue

        df = pd.read_parquet(p)
        if "date" not in df.columns or col not in df.columns:
            continue

        out = df[["date", col]].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out = (
            out.dropna(subset=["date", col])
            .drop_duplicates("date", keep="last")
            .sort_values("date")
            .rename(columns={col: tag})
            .reset_index(drop=True)
        )
        frames.append(out)

    if not frames:
        raise RuntimeError("No factor parquet loaded. Check data/factors/*.parquet")

    base = frames[0]
    for f in frames[1:]:
        base = base.merge(f, on="date", how="inner")

    return base.sort_values("date").reset_index(drop=True)


def _read_index_levels(index_levels_path: Path) -> pd.DataFrame:
    if not index_levels_path.exists():
        raise RuntimeError(
            f"Missing index levels parquet: {index_levels_path}. "
            f"Run src/caches/cache_index_levels_fdr.py first."
        )

    df = pd.read_parquet(index_levels_path).copy()
    need = {"date", *INDEX_LEVEL_COLS}
    if not need.issubset(df.columns):
        raise RuntimeError(f"index_levels missing cols={need}. got={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in INDEX_LEVEL_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=["date"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df


def _fwd_log_return(close: pd.Series, horizon: int) -> pd.Series:
    # y(t) = log(P(t+h)/P(t))
    return np.log(close.shift(-horizon) / close)


def _rank_auc_fast(y_true01: np.ndarray, y_score: np.ndarray) -> float:
    """
    AUC with rank statistic (no sklearn).
    Handles ties approximately by average rank via pandas.
    """
    y_true01 = np.asarray(y_true01).astype(float)
    y_score = np.asarray(y_score).astype(float)

    mask = np.isfinite(y_true01) & np.isfinite(y_score)
    y_true01 = y_true01[mask]
    y_score = y_score[mask]

    if len(y_true01) < 50:
        return float("nan")

    pos = (y_true01 > 0.5)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _bucket_metrics(df: pd.DataFrame, score_col: str, y_col: str) -> dict:
    """
    score 상/하위 분위수에서 10일 상승확률 차이 등을 계산
    """
    s = pd.to_numeric(df[score_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")

    mask = s.notna() & y.notna()
    s = s[mask]
    y = y[mask]

    if len(s) < 200:
        return {
            "p_up_top20": np.nan,
            "p_up_bottom20": np.nan,
            "delta_p_up": np.nan,
            "hit_rate_mid": np.nan,
        }

    q80 = s.quantile(0.80)
    q20 = s.quantile(0.20)

    top = y[s >= q80]
    bot = y[s <= q20]

    p_top = float((top > 0).mean()) if len(top) else np.nan
    p_bot = float((bot > 0).mean()) if len(bot) else np.nan
    delta = p_top - p_bot if np.isfinite(p_top) and np.isfinite(p_bot) else np.nan

    # (참고) 중앙(전체) 상승확률
    p_mid = float((y > 0).mean())

    return {
        "p_up_top20": p_top,
        "p_up_bottom20": p_bot,
        "delta_p_up": delta,
        "p_up_all": p_mid,
    }


def main():
    # ------------ Config ------------
    factors_dir = Path(_env("FACTORS_DIR", "data/factors"))
    index_levels_path = Path(_env("INDEX_LEVELS_PATH", "data/cache/index_levels.parquet"))

    horizon = int(_env("HORIZON_DAYS", "10"))  # 10일 forward
    target_index = _env("TARGET_INDEX", "k200_close").strip()  # kospi_close | kosdaq_close | k200_close
    if target_index not in INDEX_LEVEL_COLS:
        raise RuntimeError(f"Invalid TARGET_INDEX={target_index}. choose one of {INDEX_LEVEL_COLS}")

    exclude = set(x.lower() for x in _parse_list(_env("EXCLUDE_FACTORS", "")))

    start = _safe_to_datetime(_env("START_DATE", ""))
    end = _safe_to_datetime(_env("END_DATE", ""))

    # outputs
    out_csv = Path(_env("OUT_CSV", "data/analysis/predict_10d_metrics.csv"))
    out_html = Path(_env("OUT_HTML", "docs/predict_10d_report.html"))

    # ------------ Load data ------------
    fac = _read_factor_scores(factors_dir, exclude=exclude)  # cols: date + f01..f10 tags
    idx = _read_index_levels(index_levels_path)              # cols: date + closes

    base = fac.merge(idx, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # optional date filter
    if start is not None and pd.notna(start):
        base = base[base["date"] >= start]
    if end is not None and pd.notna(end):
        base = base[base["date"] <= end]
    base = base.reset_index(drop=True)

    # ------------ Target: forward return / hit ------------
    base[f"fwd_ret{horizon}d"] = _fwd_log_return(base[target_index], horizon=horizon)
    base[f"hit{horizon}d_up"] = (base[f"fwd_ret{horizon}d"] > 0).astype(int)

    # drop last horizon rows (no forward)
    base = base.dropna(subset=[f"fwd_ret{horizon}d"]).reset_index(drop=True)

    # ------------ Evaluate each factor ------------
    rows = []
    factor_cols = [c for c in base.columns if c in FACTOR_SCORE_COLS.keys()]  # f01..f10 tags present

    for ftag in factor_cols:
        s = pd.to_numeric(base[ftag], errors="coerce")
        y_ret = pd.to_numeric(base[f"fwd_ret{horizon}d"], errors="coerce")
        y_hit = pd.to_numeric(base[f"hit{horizon}d_up"], errors="coerce")

        m = s.notna() & y_ret.notna() & y_hit.notna()
        if int(m.sum()) < 200:
            continue

        ic_spearman = float(pd.Series(s[m]).corr(pd.Series(y_ret[m]), method="spearman"))
        ic_pearson = float(pd.Series(s[m]).corr(pd.Series(y_ret[m]), method="pearson"))

        auc = _rank_auc_fast(y_hit[m].to_numpy(), s[m].to_numpy())

        b = _bucket_metrics(pd.DataFrame({ftag: s, "y": y_ret}), score_col=ftag, y_col="y")

        rows.append(
            {
                "factor": ftag,
                "n": int(m.sum()),
                "target_index": target_index,
                "horizon_days": horizon,
                "ic_spearman": ic_spearman,
                "ic_pearson": ic_pearson,
                "auc_hit_up": auc,
                "p_up_top20": b["p_up_top20"],
                "p_up_bottom20": b["p_up_bottom20"],
                "delta_p_up": b["delta_p_up"],
                "p_up_all": b["p_up_all"],
            }
        )

    if not rows:
        raise RuntimeError("No factor had enough rows to evaluate. Check factor scores / date ranges.")

    metrics = pd.DataFrame(rows)
    metrics = metrics.sort_values(["ic_spearman", "delta_p_up"], ascending=[False, False]).reset_index(drop=True)

    # ------------ Save CSV ------------
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ------------ Build simple HTML report ------------
    updated = _utc_now()
    title = f"Predictive Power Report: +{horizon}D return probability (target={target_index})"

    # small inline table (top)
    top = metrics.copy()
    for c in ["ic_spearman", "ic_pearson", "auc_hit_up", "p_up_top20", "p_up_bottom20", "delta_p_up", "p_up_all"]:
        if c in top.columns:
            top[c] = pd.to_numeric(top[c], errors="coerce")

    def fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{x:.4f}"

    # build HTML table
    cols = [
        "factor", "n",
        "ic_spearman", "ic_pearson",
        "auc_hit_up",
        "p_up_top20", "p_up_bottom20", "delta_p_up",
        "p_up_all",
    ]
    th = "".join([f"<th>{html.escape(c)}</th>" for c in cols])
    trs = []
    for _, r in top.iterrows():
        tds = []
        for c in cols:
            v = r.get(c, "")
            if c in ("factor",):
                tds.append(f"<td style='font-weight:600'>{html.escape(str(v))}</td>")
            elif c in ("n",):
                tds.append(f"<td style='text-align:right'>{int(v)}</td>")
            else:
                tds.append(f"<td style='text-align:right;font-variant-numeric:tabular-nums'>{html.escape(fmt(v))}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    tbody = "\n".join(trs)

    # recommended weights (simple heuristic)
    # w ∝ max(0, ic_spearman) + 0.5*max(0, delta_p_up)
    tmp = metrics.copy()
    tmp["w_raw"] = (
        tmp["ic_spearman"].clip(lower=0).fillna(0)
        + 0.5 * tmp["delta_p_up"].clip(lower=0).fillna(0)
    )
    if tmp["w_raw"].sum() > 0:
        tmp["w_reco"] = tmp["w_raw"] / tmp["w_raw"].sum()
    else:
        tmp["w_reco"] = np.nan

    w_lines = []
    for _, r in tmp.sort_values("w_reco", ascending=False).iterrows():
        if not np.isfinite(r["w_reco"]):
            continue
        w_lines.append(f"<li><b>{html.escape(r['factor'])}</b>: {r['w_reco']*100:.1f}%</li>")
    w_html = "<ul>" + "".join(w_lines[:12]) + "</ul>"

    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 18px; }}
  h1 {{ font-size: 18px; margin: 0 0 6px 0; }}
  .meta {{ color:#666; font-size: 12px; margin-bottom: 14px; }}
  .card {{ border:1px solid #ddd; border-radius:12px; padding:12px 14px; margin: 12px 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border-bottom:1px solid #eee; padding:8px 10px; font-size: 13px; }}
  th {{ text-align:left; position: sticky; top: 0; background: #fff; }}
  .muted {{ color:#666; font-size: 13px; line-height: 1.5; }}
  code {{ background:#f6f6f6; padding: 1px 6px; border-radius: 6px; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="meta">updated: {html.escape(updated)} · rows used: {len(base)} · factors evaluated: {len(metrics)}</div>

<div class="card">
  <div class="muted">
    목적: <b>현재 팩터 점수(score)가 높을수록</b> 향후 <b>{horizon}영업일</b> 후 수익률(로그)이 커질/상승할 확률이 증가하는지 평가합니다.<br/>
    - <code>ic_spearman</code>: score와 {horizon}일 forward 수익률의 Spearman 상관(단조 관계)<br/>
    - <code>auc_hit_up</code>: {horizon}일 후 수익률이 0보다 클지(상승) 분류 AUC (0.5=무작위)<br/>
    - <code>delta_p_up</code>: score 상위 20% 상승확률 - 하위 20% 상승확률
  </div>
</div>

<div class="card">
  <h2 style="margin:0 0 10px 0;font-size:15px;">Recommended initial weights (heuristic)</h2>
  <div class="muted">아래 가중치는 <code>max(0, ic_spearman) + 0.5*max(0, delta_p_up)</code> 기반의 초기안입니다. 최종은 검증 후 캡/그룹제약으로 안정화 권장.</div>
  {w_html}
</div>

<div class="card">
  <h2 style="margin:0 0 10px 0;font-size:15px;">Factor metrics table</h2>
  <div style="overflow:auto;">
    <table>
      <thead><tr>{th}</tr></thead>
      <tbody>
      {tbody}
      </tbody>
    </table>
  </div>
</div>

<div class="card">
  <h2 style="margin:0 0 10px 0;font-size:15px;">Files</h2>
  <div class="muted">
    CSV: <code>{html.escape(out_csv.as_posix())}</code><br/>
    HTML: <code>{html.escape(out_html.as_posix())}</code>
  </div>
</div>

</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_doc, encoding="utf-8")

    print(f"[predict10d] OK -> {out_csv}")
    print(f"[predict10d] OK -> {out_html}")


if __name__ == "__main__":
    main()
