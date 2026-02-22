#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
render_top10_deep_report.py
- Takes daily_top5_top5.csv (TOP10 = 5+5) and renders a deeper HTML report.
- Output: docs/stocks/reports/top10_deep_report.html
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def _fmt(x, kind="float"):
    if pd.isna(x):
        return ""
    if kind == "pct":
        return f"{x*100:,.2f}%"
    if kind == "float":
        return f"{x:,.4f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_csv", default="docs/stocks/reports/daily_top5_top5.csv")
    ap.add_argument("--out_html", default="docs/stocks/reports/top10_deep_report.html")
    args = ap.parse_args()

    top_csv = Path(args.top_csv)
    if not top_csv.exists():
        raise FileNotFoundError(f"missing top csv: {top_csv}")

    df = pd.read_csv(top_csv, encoding="utf-8-sig")
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    # basic ordering by score
    if "score" in df.columns:
        df = df.sort_values("score", ascending=False)

    sections = []
    for _, r in df.iterrows():
        name = r.get("name", "")
        ticker = r.get("ticker", "")
        market = r.get("market", "")
        sector = r.get("wics_major", "")
        size_q = r.get("size_q", "")
        score = r.get("score", "")

        def getcol(c, kind="float"):
            v = r.get(c, np.nan)
            if isinstance(v, str):
                return v
            return _fmt(v, kind)

        # Build small "reason" summary
        reason = []
        if pd.notna(r.get("ma_gap_20d", np.nan)):
            reason.append(f"ma_gap_20d={getcol('ma_gap_20d','pct')}")
        if pd.notna(r.get("foreign_net_to_value_20d", np.nan)):
            reason.append(f"foreign_net_to_value_20d={getcol('foreign_net_to_value_20d','pct')}")
        if pd.notna(r.get("turnover_20d_mean", np.nan)):
            reason.append(f"turnover_20d_mean={getcol('turnover_20d_mean','float')}")

        sec = f"""
        <div class="card">
          <h2>{ticker} {name} <span class="tag">{market}</span></h2>
          <div class="meta">Sector: <b>{sector}</b> / Size: <b>{size_q}</b> / Score: <b>{score}</b></div>
          <div class="small">핵심 요약: {", ".join(reason) if reason else "—"}</div>

          <table>
            <tr>
              <th class="left">구분</th><th>값</th><th class="left">구분</th><th>값</th>
            </tr>
            <tr>
              <td class="left">ret_5d</td><td class="num">{getcol("ret_5d","pct")}</td>
              <td class="left">ret_20d</td><td class="num">{getcol("ret_20d","pct")}</td>
            </tr>
            <tr>
              <td class="left">ret_40d</td><td class="num">{getcol("ret_40d","pct")}</td>
              <td class="left">vol_20d</td><td class="num">{getcol("vol_20d","float")}</td>
            </tr>
            <tr>
              <td class="left">turnover_20d_mean</td><td class="num">{getcol("turnover_20d_mean","float")}</td>
              <td class="left">amihud_20d</td><td class="num">{getcol("amihud_20d","float")}</td>
            </tr>
            <tr>
              <td class="left">inv_per</td><td class="num">{getcol("inv_per","float")}</td>
              <td class="left">inv_pbr</td><td class="num">{getcol("inv_pbr","float")}</td>
            </tr>
            <tr>
              <td class="left">roe_proxy</td><td class="num">{getcol("roe_proxy","float")}</td>
              <td class="left">peg_approx_20d</td><td class="num">{getcol("peg_approx_20d","float")}</td>
            </tr>
            <tr>
              <td class="left">model_prob</td><td class="num">{getcol("model_prob","float")}</td>
              <td class="left">rule_global / rule_sector</td><td class="num">{getcol("rule_global","float")} / {getcol("rule_sector","float")}</td>
            </tr>
          </table>
        </div>
        """
        sections.append(sec)

    html = f"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>TOP10 Deep Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, "Apple SD Gothic Neo", "Noto Sans KR"; margin: 22px; }}
    .card {{ border: 1px solid #e3e3e3; border-radius: 12px; padding: 12px 14px; margin-bottom: 14px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 8px; }}
    th, td {{ border: 1px solid #ededed; padding: 6px 8px; }}
    th {{ background: #fafafa; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    td.left, th.left {{ text-align: left; }}
    .small {{ color: #666; font-size: 12px; }}
    .meta {{ color:#333; font-size: 13px; margin-top: 4px; }}
    .tag {{ display:inline-block; padding:2px 8px; margin-left:8px; border-radius:999px; background:#eef; font-size:12px; }}
  </style>
</head>
<body>
  <h1>TOP10 상세 분석 리포트</h1>
  <div class="small">입력: {top_csv.as_posix()}</div>
  {"".join(sections)}
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")
    print(f"[deep-report] OK -> {out_html}")


if __name__ == "__main__":
    main()
