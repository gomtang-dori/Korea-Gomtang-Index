# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Config ----------
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


def _read_factor_score(factors_dir: Path, tag: str, score_col: str) -> pd.DataFrame | None:
    p = factors_dir / f"{tag}.parquet"
    if not p.exists():
        return None

    df = pd.read_parquet(p)
    if "date" not in df.columns or score_col not in df.columns:
        return None

    out = df[["date", score_col]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    out = (
        out.dropna(subset=["date", score_col])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .rename(columns={score_col: tag})
        .reset_index(drop=True)
    )
    return out


def _read_index_levels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Missing index levels parquet: {path} (run cache_index_levels_fdr.py)")

    df = pd.read_parquet(path).copy()
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


def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in INDEX_LEVEL_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[f"{c}_ret1d"] = np.log(out[c]).diff()
    return out


def _corr_heatmap_html(corr: pd.DataFrame, title: str) -> str:
    """
    의존성 최소화를 위해 plotly 없이 HTML 테이블 기반 히트맵 생성.
    (PNG는 별도 옵션)
    """
    z = corr.values.astype(float)
    labels = list(corr.columns)

    # 색상: -1(파랑) ~ 0(흰) ~ +1(빨강)
    def color(v: float) -> str:
        if np.isnan(v):
            return "#f0f0f0"
        v = float(max(-1.0, min(1.0, v)))
        if v >= 0:
            # white -> red
            r = 255
            g = int(255 * (1 - v))
            b = int(255 * (1 - v))
        else:
            # white -> blue
            v2 = abs(v)
            r = int(255 * (1 - v2))
            g = int(255 * (1 - v2))
            b = 255
        return f"rgb({r},{g},{b})"

    # HTML table
    th = "".join([f"<th>{c}</th>" for c in [""] + labels])
    rows = []
    for i, rname in enumerate(labels):
        tds = [f"<th style='position:sticky;left:0;background:#fff'>{rname}</th>"]
        for j in range(len(labels)):
            v = z[i, j]
            txt = "" if np.isnan(v) else f"{v:.3f}"
            bg = color(v)
            tds.append(f"<td style='background:{bg};text-align:right;padding:6px 8px;font-variant-numeric:tabular-nums'>{txt}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    tbody = "\n".join(rows)

    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
  body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 18px; }}
  h1 {{ font-size: 18px; margin: 0 0 10px 0; }}
  .meta {{ color:#666; font-size: 12px; margin-bottom: 14px; }}
  .wrap {{ overflow:auto; border:1px solid #ddd; border-radius: 10px; }}
  table {{ border-collapse: collapse; width: max-content; min-width: 100%; }}
  th, td {{ border: 1px solid #eee; }}
  thead th {{ position: sticky; top: 0; background: #fff; z-index: 2; padding: 6px 8px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">상관계수 범위: -1 ~ +1 (파랑=음의 상관, 빨강=양의 상관)</div>
<div class="wrap">
<table>
  <thead><tr>{th}</tr></thead>
  <tbody>
  {tbody}
  </tbody>
</table>
</div>
</body>
</html>
"""
    return html


def main():
    factors_dir = Path(_env("FACTORS_DIR", "data/factors"))
    index_levels_path = Path(_env("INDEX_LEVELS_PATH", "data/cache/index_levels.parquet"))

    # outputs
    out_csv = Path(_env("OUT_CSV", "data/analysis/factor_corr.csv"))
    out_html = Path(_env("OUT_HTML", "docs/factor_corr.html"))

    # options
    method = _env("CORR_METHOD", "pearson").lower()
    mode = _env("MODE", "levels").lower()  # levels | returns
    tag = _env("TAG", "")                  # e.g., 8Y / 1Y
    exclude = set(x.strip().lower() for x in _env("EXCLUDE_FACTORS", "").split(",") if x.strip())
    start = _env("START_DATE", "")
    end = _env("END_DATE", "")
    save_png = (_env("SAVE_PNG", "0") == "1")
    out_png = Path(_env("OUT_PNG", "docs/factor_corr.png"))

    # 1) load factor scores
    frames = []
    for ftag, scol in FACTOR_SCORE_COLS.items():
        if ftag in exclude:
            continue
        df = _read_factor_score(factors_dir, ftag, scol)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError("No factor score parquet loaded. Check data/factors/*.parquet")

    base = frames[0]
    for f in frames[1:]:
        base = base.merge(f, on="date", how="inner")

    # 2) add index levels
    idx = _read_index_levels(index_levels_path)
    base = base.merge(idx, on="date", how="inner").sort_values("date").reset_index(drop=True)

    if mode == "returns":
        base = _add_returns(base)

    # 3) optional date filter
    if start:
        base = base[base["date"] >= pd.to_datetime(start)]
    if end:
        base = base[base["date"] <= pd.to_datetime(end)]
    base = base.reset_index(drop=True)

    # 4) choose columns
    factor_cols = [c for c in base.columns if c != "date"]
    corr = base[factor_cols].corr(method=method)

    # 5) write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_csv, encoding="utf-8-sig")

    # 6) write HTML heatmap (no extra deps)
    title = f"Factor Correlation Matrix {tag}".strip()
    title = f"{title} ({method.upper()} / {mode})"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(_corr_heatmap_html(corr, title=title), encoding="utf-8")
    print(f"[corr] OK rows={len(base)} cols={len(factor_cols)} method={method} mode={mode}")
    print(f"[corr] CSV  -> {out_csv}")
    print(f"[corr] HTML -> {out_html}")

    # 7) optional PNG (requires matplotlib)
    if save_png:
        try:
            import matplotlib.pyplot as plt

            fig_w = max(10, int(0.6 * len(corr.columns)))
            fig_h = max(8, int(0.55 * len(corr.columns)))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="bwr")

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.index)

            for i in range(corr.shape[0]):
                for j in range(corr.shape[1]):
                    v = corr.values[i, j]
                    if np.isnan(v):
                        continue
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="black")

            ax.set_title(title, fontsize=12)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()

            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_png, dpi=180)
            plt.close(fig)
            print(f"[corr] PNG  -> {out_png}")
        except Exception as e:
            print(f"[corr] WARN: PNG export failed: {e} (set SAVE_PNG=0 or install matplotlib)")


if __name__ == "__main__":
    main()
