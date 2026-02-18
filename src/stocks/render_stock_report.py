#!/usr/bin/env python3
"""
전종목 HTML 리포트 생성
"""
import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()
print(f"[DEBUG] CWD: {Path.cwd()}")
print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}")

def render_stock_report():
    print("[render_stock_report] 시작...")
    
    master_path = PROJECT_ROOT / "data/stocks/master/listings.parquet"
    print(f"[DEBUG] master_path exists: {master_path.exists()}")
    
    if not master_path.exists():
        print(f"⚠️  마스터 파일 없음: {master_path}")
        return
    df_master = pd.read_parquet(master_path)
    
    out_dir = PROJECT_ROOT / "docs/stocks"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] 출력 디렉토리: {out_dir} (exists: {out_dir.exists()})")
    
    summary_rows = []
    
    for idx, row in df_master.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        market = row["market"]
        
        feat_path = PROJECT_ROOT / f"data/stocks/analysis/{ticker}/features.parquet"
        if not feat_path.exists():
            continue
        
        df_feat = pd.read_parquet(feat_path)
        if df_feat.empty:
            continue
        
        latest = df_feat.iloc[-1]
        close = latest.get("close", 0)
        ret_1d = latest.get("ret_1d", 0)
        ret_5d = latest.get("ret_5d", 0)
        
        signal_fund = latest.get("signal_fundamentals", 0)
        signal_flow = latest.get("signal_flows", 0)
        total_signal = latest.get("signal", 0)
        
        if total_signal >= 3:
            opinion = "BUY"
            position = "+10% ~ +20%"
            opinion_color = "#4CAF50"
        elif total_signal >= 1:
            opinion = "HOLD"
            position = "0% (유지)"
            opinion_color = "#FF9800"
        else:
            opinion = "SELL"
            position = "-10% ~ -30%"
            opinion_color = "#F44336"
        
        html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>{name} ({ticker})</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #f5f5f5; }}
.container {{ max-width: 900px; margin: auto; background: white; padding: 30px; }}
h1 {{ color: #333; border-bottom: 3px solid #4CAF50; }}
.card {{ background: #fafafa; padding: 15px; margin: 15px 0; border-radius: 5px; }}
.opinion {{ font-size: 24px; font-weight: bold; color: {opinion_color}; }}
</style>
</head>
<body>
<div class="container">
<h1>{name} ({ticker})</h1>
<div class="card">
<p><strong>시장:</strong> {market} | <strong>현재가:</strong> {close:,.0f}원</p>
<p><strong>수익률:</strong> 1일 {ret_1d:+.2%} | 5일 {ret_5d:+.2%}</p>
</div>
<div class="card">
<h2>투자 의견</h2>
<p class="opinion">{opinion}</p>
<p><strong>포지션:</strong> {position}</p>
<ul><li>펀더멘털: {signal_fund}</li><li>수급: {signal_flow}</li><li>종합: {total_signal}</li></ul>
</div>
</div>
</body>
</html>"""
        
        out_path = out_dir / f"{ticker}.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        summary_rows.append({
            "ticker": ticker,
            "name": name,
            "market": market,
            "close": close,
            "ret_1d": ret_1d,
            "opinion": opinion,
            "signal": total_signal
        })
        
        if len(summary_rows) == 1:
            print(f"  [첫 파일] {out_path} 생성 완료")
    
    if not summary_rows:
        print("⚠️  생성된 리포트 없음")
        return
    
    print(f"  [진행 중] {len(summary_rows)}개 종목 HTML 생성 완료")
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.sort_values("signal", ascending=False, inplace=True)
    
    dashboard_html = f"""<!DOCTYPE html>
<html lang="ko">
<head><meta charset="UTF-8"><title>전종목 대시보드</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #f5f5f5; }}
.container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; }}
h1 {{ color: #333; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
th {{ background: #4CAF50; color: white; }}
a {{ color: #2196F3; text-decoration: none; }}
.BUY {{ color: #4CAF50; font-weight: bold; }}
.HOLD {{ color: #FF9800; font-weight: bold; }}
.SELL {{ color: #F44336; font-weight: bold; }}
</style>
</head>
<body>
<div class="container">
<h1>📊 전종목 대시보드</h1>
<p>총 <strong>{len(df_summary)}</strong>개 종목</p>
<table><thead><tr>
<th>종목명</th><th>티커</th><th>시장</th><th>현재가</th><th>1일</th><th>의견</th><th>점수</th>
</tr></thead><tbody>
"""
    
    for _, r in df_summary.iterrows():
        dashboard_html += f"""<tr>
<td><a href="{r['ticker']}.html">{r['name']}</a></td>
<td>{r['ticker']}</td><td>{r['market']}</td>
<td>{r['close']:,.0f}원</td><td>{r['ret_1d']:+.2%}</td>
<td class="{r['opinion']}">{r['opinion']}</td><td>{r['signal']}</td>
</tr>
"""
    
    dashboard_html += "</tbody></table></div></body></html>"
    
    dashboard_path = out_dir / "index.html"
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(dashboard_html)
    
    print(f"[render_stock_report] ✅ 완료")
    print(f"  → 대시보드: {dashboard_path}")
    print(f"  → HTML: {len(summary_rows)}개")
    print(f"  → 출력 위치: {out_dir}")

if __name__ == "__main__":
    render_stock_report()
