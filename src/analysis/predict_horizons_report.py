# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
import html

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
    return np.log(close.shift(-horizon) / close)


def _rank_auc_fast(y_true01: np.ndarray, y_score: np.ndarray) -> float:
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


def _bucket_metrics(score: pd.Series, y_ret: pd.Series) -> dict:
    s = pd.to_numeric(score, errors="coerce")
    y = pd.to_numeric(y_ret, errors="coerce")
    m = s.notna() & y.notna()
    s = s[m]
    y = y[m]

    if len(s) < 200:
        return {
            "p_up_top20": np.nan,
            "p_up_bottom20": np.nan,
            "delta_p_up": np.nan,
            "p_up_all": np.nan,
        }

    q80 = s.quantile(0.80)
    q20 = s.quantile(0.20)
    top = y[s >= q80]
    bot = y[s <= q20]
    p_top = float((top > 0).mean()) if len(top) else np.nan
    p_bot = float((bot > 0).mean()) if len(bot) else np.nan
    delta = p_top - p_bot if np.isfinite(p_top) and np.isfinite(p_bot) else np.nan
    p_all = float((y > 0).mean())
    return {"p_up_top20": p_top, "p_up_bottom20": p_bot, "delta_p_up": delta, "p_up_all": p_all}


def _score_metric(ic_spearman: float, delta_p_up: float) -> float:
    ic = float(ic_spearman) if np.isfinite(ic_spearman) else 0.0
    d = float(delta_p_up) if np.isfinite(delta_p_up) else 0.0
    return max(0.0, ic) + 0.5 * max(0.0, d)


def _select_best_direction(metrics_one_h: pd.DataFrame) -> pd.DataFrame:
    """
    같은 base factor(f06 vs f06_c) 중 더 좋은 row 선택.
    우선순위: ic_spearman > delta_p_up > auc_hit_up
    """
    df = metrics_one_h.copy()
    df["base_factor"] = df["factor"].str.replace(r"_c$", "", regex=True)

    def _pick(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values(
            ["ic_spearman", "delta_p_up", "auc_hit_up"],
            ascending=[False, False, False],
        )
        return g.iloc[0]

    picked = df.groupby("base_factor", as_index=False).apply(_pick)
    if isinstance(picked.index, pd.MultiIndex):
        picked = picked.reset_index(drop=True)
    return picked.reset_index(drop=True)


def main():
    factors_dir = Path(_env("FACTORS_DIR", "data/factors"))
    index_levels_path = Path(_env("INDEX_LEVELS_PATH", "data/cache/index_levels.parquet"))

    # ✅ KOSPI 타깃
    target_index = _env("TARGET_INDEX", "kospi_close").strip()
    if target_index not in INDEX_LEVEL_COLS:
        raise RuntimeError(f"Invalid TARGET_INDEX={target_index}. choose one of {INDEX_LEVEL_COLS}")

    horizons = [int(x) for x in _parse_list(_env("HORIZONS", "5,10,20"))]
    if not horizons:
        raise RuntimeError("HORIZONS is empty. example: HORIZONS=5,10,20")

    exclude = set(x.lower() for x in _parse_list(_env("EXCLUDE_FACTORS", "")))
    add_contrarian = (_env("ADD_CONTRARIAN", "0") == "1")
    contrarian_suffix = _env("CONTRARIAN_SUFFIX", "_c")

    start = _safe_to_datetime(_env("START_DATE", ""))
    end = _safe_to_datetime(_env("END_DATE", ""))

    tag = _env("TAG", "")

    out_prefix = _env("OUT_PREFIX", "predict_h5_10_20")
    out_dir = Path(_env("OUT_DIR", "data/analysis"))
    out_html = Path(_env("OUT_HTML", f"docs/{out_prefix}.html"))
    out_csv_summary = out_dir / f"{out_prefix}_summary.csv"
    out_csv_weights = out_dir / f"{out_prefix}_weights.csv"

    fac = _read_factor_scores(factors_dir, exclude=exclude)
    idx = _read_index_levels(index_levels_path)
    base = fac.merge(idx, on="date", how="inner").sort_values("date").reset_index(drop=True)

    if start is not None and pd.notna(start):
        base = base[base["date"] >= start]
    if end is not None and pd.notna(end):
        base = base[base["date"] <= end]
    base = base.reset_index(drop=True)

    factor_cols_base = [c for c in base.columns if c in FACTOR_SCORE_COLS.keys()]
    if add_contrarian:
        for ftag in factor_cols_base:
            s = pd.to_numeric(base[ftag], errors="coerce")
            base[f"{ftag}{contrarian_suffix}"] = 100.0 - s
        factor_cols = factor_cols_base + [f"{c}{contrarian_suffix}" for c in factor_cols_base]
    else:
        factor_cols = factor_cols_base

    all_best = []

    for h in horizons:
        dfh = base.copy()
        dfh[f"fwd_ret{h}d"] = _fwd_log_return(dfh[target_index], horizon=h)
        dfh[f"hit{h}d_up"] = (dfh[f"fwd_ret{h}d"] > 0).astype(int)
        dfh = dfh.dropna(subset=[f"fwd_ret{h}d"]).reset_index(drop=True)

        y_ret = pd.to_numeric(dfh[f"fwd_ret{h}d"], errors="coerce")
        y_hit = pd.to_numeric(dfh[f"hit{h}d_up"], errors="coerce")

        rows = []
        for ftag in factor_cols:
            s = pd.to_numeric(dfh[ftag], errors="coerce")
            m = s.notna() & y_ret.notna() & y_hit.notna()
            if int(m.sum()) < 200:
                continue

            ic_s = float(pd.Series(s[m]).corr(pd.Series(y_ret[m]), method="spearman"))
            ic_p = float(pd.Series(s[m]).corr(pd.Series(y_ret[m]), method="pearson"))
            auc = _rank_auc_fast(y_hit[m].to_numpy(), s[m].to_numpy())
            b = _bucket_metrics(s[m], y_ret[m])
            score = _score_metric(ic_s, b["delta_p_up"])

            rows.append(
                {
                    "factor": ftag,
                    "base_factor": ftag.replace(contrarian_suffix, ""),
                    "is_contrarian": int(add_contrarian and ftag.endswith(contrarian_suffix)),
                    "n": int(m.sum()),
                    "target_index": target_index,
                    "horizon_days": h,
                    "ic_spearman": ic_s,
                    "ic_pearson": ic_p,
                    "auc_hit_up": auc,
                    "p_up_top20": b["p_up_top20"],
                    "p_up_bottom20": b["p_up_bottom20"],
                    "delta_p_up": b["delta_p_up"],
                    "p_up_all": b["p_up_all"],
                    "score": score,
                }
            )

        mh = pd.DataFrame(rows)
        if mh.empty:
            raise RuntimeError(f"No metrics computed for horizon={h}. Check data availability.")

        mh_best = _select_best_direction(mh) if add_contrarian else mh.copy()
        mh_best["tag"] = tag
        all_best.append(mh_best)

        out_dir.mkdir(parents=True, exist_ok=True)
        mh_best.to_csv(out_dir / f"{out_prefix}_best_h{h}d.csv", index=False, encoding="utf-8-sig")

    best_all = pd.concat(all_best, ignore_index=True)

    # 동일가중 평균 (5/10/20의 score를 평균)
    pivot = best_all.pivot_table(index="factor", columns="horizon_days", values="score", aggfunc="mean").fillna(0.0)
    pivot["score_avg"] = pivot.mean(axis=1)

    ssum = float(pivot["score_avg"].sum())
    if ssum <= 0:
        raise RuntimeError("All scores are zero. Cannot build weights.")

    pivot["w_raw_avg"] = pivot["score_avg"] / ssum

    meta = best_all.drop_duplicates(subset=["factor"]).set_index("factor")[["base_factor", "is_contrarian"]]
    weights = pivot.join(meta, how="left").reset_index().rename(columns={"index": "factor"})

    horizon_cols = [c for c in weights.columns if isinstance(c, int)]
    horizon_cols_sorted = sorted(horizon_cols)
    keep = ["factor", "base_factor", "is_contrarian"] + horizon_cols_sorted + ["score_avg", "w_raw_avg"]
    weights = weights[keep].sort_values("w_raw_avg", ascending=False).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_all.to_csv(out_csv_summary, index=False, encoding="utf-8-sig")
    weights.to_csv(out_csv_weights, index=False, encoding="utf-8-sig")

    # HTML
    updated = _utc_now()
    title = f"Predictive Power (H={horizons}, target={target_index}) {tag}".strip()

    def _fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{x:.4f}"

    cols_w = ["factor", "is_contrarian"] + horizon_cols_sorted + ["score_avg", "w_raw_avg"]
    th_w = "".join([f"<th>{html.escape(str(c))}</th>" for c in cols_w])

    trs_w = []
    for _, r in weights.iterrows():
        tds = []
        for c in cols_w:
            v = r.get(c, "")
            if c == "factor":
                tds.append(f"<td style='font-weight:600'>{html.escape(str(v))}</td>")
            elif c == "is_contrarian":
                tds.append(f"<td style='text-align:right'>{int(v) if pd.notna(v) else ''}</td>")
            elif c in horizon_cols_sorted:
                tds.append(f"<td style='text-align:right;font-variant-numeric:tabular-nums'>{html.escape(_fmt(v))}</td>")
            else:
                tds.append(f"<td style='text-align:right;font-variant-numeric:tabular-nums'>{html.escape(_fmt(v))}</td>")
        trs_w.append("<tr>" + "".join(tds) + "</tr>")
    tbody_w = "\n".join(trs_w)

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
  <div class="meta">updated: {html.escape(updated)} · horizons: {html.escape(str(horizons))} · target: {html.escape(target_index)}</div>

  <div class="card">
    <div class="muted">
      - 각 horizon별로 <b>원본 vs contrarian(100-score)</b> 중 성능이 더 좋은 방향을 선택했습니다.<br/>
      - horizon별 점수는 <code>ic_spearman + 0.5*max(0, delta_p_up)</code> 입니다.<br/>
      - 최종 <code>w_raw_avg</code>는 (5/10/20) 점수를 <b>동일가중 평균</b>해 정규화한 값입니다.<br/>
      - 다음 단계에서 <b>그룹제약 + cap=0.25</b>를 적용해 최종 가중치를 확정하세요.
    </div>
  </div>

  <div class="card">
    <h2 style="margin:0 0 10px 0;font-size:15px;">Weights (raw avg; before group/cap)</h2>
    <div style="overflow:auto;">
      <table>
        <thead><tr>{th_w}</tr></thead>
        <tbody>
          {tbody_w}
        </tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h2 style="margin:0 0 10px 0;font-size:15px;">Files</h2>
    <div class="muted">
      summary: <code>{html.escape(out_csv_summary.as_posix())}</code><br/>
      weights: <code>{html.escape(out_csv_weights.as_posix())}</code>
    </div>
  </div>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_doc, encoding="utf-8")

    print(f"[predict-horizons] OK -> {out_csv_summary}")
    print(f"[predict-horizons] OK -> {out_csv_weights}")
    print(f"[predict-horizons] OK -> {out_html}")


if __name__ == "__main__":
    main()
