#!/usr/bin/env python3
"""
전종목 HTML 리포트 생성
- features.parquet(analysis)에 fundamentals(per/eps/bps/div 등)이 들어오면 화면에 표시
"""

import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")) if os.getenv("PROJECT_ROOT") else Path.cwd()
print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}")

def _fmt_num(x, digits=2, suffix=""):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "-"
        if isinstance(x, (int, float)):
            if abs(x) >= 1000 and digits == 0:
                return f"{x:,.0f}{suffix}"
            return f"{x:.{digits}f}{suffix}"
        return str(x)
    except:
        return "-"

def render_stock_report():
    print("[render_stock_report] 시작...")

    master_path = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    if not master_path.exists():
        print(f"⚠️ master not found: {master_path}")
        return

    df_master = pd.read_parquet(master_path)

    out_dir = PROJECT_ROOT / "docs/stocks"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for _, row in df_master.iterrows():
        ticker = str(row["ticker"])
        name = str(row.get("name", ""))
        market = str(row.get("market", ""))

        feat_path = PROJECT_ROOT / f"data/stocks/analysis/{ticker}/features.parquet"
        if not feat_path.exists():
            continue

        df_feat = pd.read_parquet(feat_path)
        if df_feat.empty:
            continue

        latest = df_feat.iloc[-1]

        close = latest.get("close", pd.NA)
        ret_1d = latest.get("ret_1d", pd.NA)
        ret_5d = latest.get("ret_5d", pd.NA)
        ret_20d = latest.get("ret_20d", pd.NA)

        # fundamentals (표시용)
        per = latest.get("per", pd.NA)
        eps = latest.get("eps", pd.NA)
        bps = latest.get("bps", pd.NA)
        div = latest.get("div", pd.NA)
        pbr = latest.get("pbr", pd.NA)
        dps = latest.get("dps", pd.NA)

        signal_fund = latest.get("signal_fundamentals", 0)
        signal_flow = latest.get("signal_flows", 0)
        total_signal = latest.get("signal", 0)

        # opinion (기존 유지)
        if total_signal >= 3:
            opinion = "BUY"
            opinion_color = "#4CAF50"
        elif total_signal >= 1:
            opinion = "HOLD"
            opinion_color = "#FF9800"
        else:
            opinion = "SELL"
            opinion_color = "#F44336"

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>{name} ({ticker})</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #f5f5f5; }}
.container {{ max-width: 960px; margin: auto; background: white; padding: 28px; border-radius: 10px; }}
h1 {{ color: #333; margin-bottom: 10px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.card {{ background: #fafafa; padding: 14px; border-radius: 8px; border: 1px solid #eee; }}
.opinion {{ font-size: 22px; font-weight: 800; color: {opinion_color}; }}
small {{ color: #666; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 10px; border-bottom: 1px solid #e5e5e5; text-align:left; }}
th {{ width: 140px; color:#333; }}
</style>
</head>
<body>
<div class="container">
  <h1>{name} ({ticker})</h1>
  <small>시장: {market}</small>

  <div class="grid" style="margin-top:12px;">
    <div class="card">
      <h3>가격/수익률</h3>
      <table>
        <tr><th>현재가</th><td>{_fmt_num(close, digits=0, suffix="원")}</td></tr>
        <tr><th>1일 수익률</th><td>{_fmt_num(ret_1d*100 if pd.notna(ret_1d) else pd.NA, digits=2, suffix="%")}</td></tr>
        <tr><th>5일 수익률</th><td>{_fmt_num(ret_5d*100 if pd.notna(ret_5d) else pd.NA, digits=2, suffix="%")}</td></tr>
        <tr><th>20일 수익률</th><td>{_fmt_num(ret_20d*100 if pd.notna(ret_20d) else pd.NA, digits=2, suffix="%")}</td></tr>
      </table>
    </div>

    <div class="card">
      <h3>투자 의견(임시)</h3>
      <div class="opinion">{opinion}</div>
      <small>※ SIGNAL은 추후 로직 확정 예정</small>
      <table style="margin-top:8px;">
        <tr><th>Fund 점수</th><td>{signal_fund}</td></tr>
        <tr><th>Flow 점수</th><td>{signal_flow}</td></tr>
        <tr><th>Total</th><td>{total_signal}</td></tr>
      </table>
    </div>
  </div>

  <div class="card" style="margin-top:12px;">
    <h3>Fundamentals (PyKRX)</h3>
    <table>
      <tr><th>PER</th><td>{_fmt_num(per, digits=2)}</td></tr>
      <tr><th>PBR</th><td>{_fmt_num(pbr, digits=2)}</td></tr>
      <tr><th>EPS</th><td>{_fmt_num(eps, digits=2)}</td></tr>
      <tr><th>BPS</th><td>{_fmt_num(bps, digits=2)}</td></tr>
      <tr><th>DIV(배당수익률)</th><td>{_fmt_num(div, digits=2, suffix="%")}</td></tr>
      <tr><th>DPS</th><td>{_fmt_num(dps, digits=2)}</td></tr>
    </table>
  </div>

</div>
</body>
</html>
"""

        out_path = out_dir / f"{ticker}.html"
        out_path.write_text(html, encoding="utf-8")

        summary_rows.append({
            "ticker": ticker,
            "name": name,
            "market": market,
            "close": close,
            "ret_1d": ret_1d,
            "opinion": opinion,
            "signal": total_signal,
            "per": per,
            "eps": eps,
            "div": div,
        })

    if not summary_rows:
        print("⚠️ 생성된 리포트 없음 (features.parquet가 없거나 비어있음)")
        return

    df_sum = pd.DataFrame(summary_rows)
    df_sum.sort_values("signal", ascending=False, inplace=True)

    # dashboard
    dash = f"""<!DOCTYPE html>
<html lang="ko">
<head><meta charset="UTF-8"><title>전종목 대시보드</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #f5f5f5; }}
.container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius:10px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 10px; border-bottom: 1px solid #e5e5e5; text-align:left; }}
th {{ background: #4CAF50; color: white; }}
a {{ color: #1565c0; text-decoration: none; }}
.BUY {{ color: #2e7d32; font-weight: 800; }}
.HOLD {{ color: #ef6c00; font-weight: 800; }}
.SELL {{ color: #c62828; font-weight: 800; }}
</style></head>
<body><div class="container">
<h1>전종목 대시보드</h1>
<p>총 <strong>{len(df_sum)}</strong>개 종목</p>
<table><thead><tr>
<th>종목</th><th>티커</th><th>시장</th><th>현재가</th><th>1일</th><th>의견</th><th>점수</th><th>PER</th><th>EPS</th><th>DIV</th>
</tr></thead><tbody>
"""

    for _, r in df_sum.iterrows():
        dash += f"""<tr>
<td><a href="{r['ticker']}.html">{r['name']}</a></td>
<td>{r['ticker']}</td>
<td>{r['market']}</td>
<td>{_fmt_num(r['close'], digits=0, suffix="원")}</td>
<td>{_fmt_num(r['ret_1d']*100 if pd.notna(r['ret_1d']) else pd.NA, digits=2, suffix="%")}</td>
<td class="{r['opinion']}">{r['opinion']}</td>
<td>{r['signal']}</td>
<td>{_fmt_num(r.get('per', pd.NA), digits=2)}</td>
<td>{_fmt_num(r.get('eps', pd.NA), digits=2)}</td>
<td>{_fmt_num(r.get('div', pd.NA), digits=2, suffix="%")}</td>
</tr>
"""

    dash += "</tbody></table></div></body></html>"
    (out_dir / "index.html").write_text(dash, encoding="utf-8")

    print("[render_stock_report] ✅ 완료")
    print(f"  → dashboard: {out_dir / 'index.html'}")
    print(f"  → HTML: {len(summary_rows)} files")

if __name__ == "__main__":
    render_stock_report()
