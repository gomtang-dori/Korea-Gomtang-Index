#!/usr/bin/env python3
"""
전종목 HTML 리포트 생성
출력: docs/stocks/{ticker}.html, docs/stocks/index.html
"""
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def render_stock_report():
    print("[render_stock_report] 시작...")
    
    master_path = Path("data/stocks/master/listings.parquet")
    if not master_path.exists():
        print("⚠️  마스터 파일 없음")
        return
    df_master = pd.read_parquet(master_path)
    
    out_dir = Path("docs/stocks")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary_rows = []
    
    for idx, row in df_master.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        market = row["market"]
        
        # features 로드
        feat_path = Path(f"data/stocks/analysis/{ticker}/features.parquet")
        if not feat_path.exists():
            continue
        
        df_feat = pd.read_parquet(feat_path)
        if df_feat.empty:
            continue
        
        # 최근 데이터
        latest = df_feat.iloc[-1]
        close = latest.get("close", 0)
        ret_1d = latest.get("ret_1d", 0)
        ret_5d = latest.get("ret_5d", 0)
        
        # 투자 의견 (간단 예시)
        signal_fund = latest.get("signal_fundamentals", 0)
        signal_flow = latest.get("signal_flows", 0)
        total_signal = signal_fund + signal_flow
        
        if total_signal >= 5:
            opinion = "BUY"
        elif total_signal >= 2:
            opinion = "HOLD"
        else:
            opinion = "SELL"
        
        # HTML 생성 (간단 템플릿)
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head><meta charset="UTF-8"><title>{name} ({ticker})</title></head>
        <body>
        <h1>{name} ({ticker})</h1>
        <p>시장: {market} | 현재가: {close:,.0f}원 | 1D: {ret_1d:+.2%} | 5D: {ret_5d:+.2%}</p>
        <h2>투자 의견: {opinion}</h2>
        <p>펀더멘털 점수: {signal_fund} | 수급 점수: {signal_flow} | 종합: {total_signal}</p>
        </body>
        </html>
        """
        
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
        
        print(f"  [{idx+1}/{len(df_master)}] {ticker} 리포트 생성")
    
    # 대시보드 생성
    df_summary = pd.DataFrame(summary_rows)
    df_summary.sort_values("signal", ascending=False, inplace=True)
    
    dashboard_html = """
    <!DOCTYPE html>
    <html lang="ko">
    <head><meta charset="UTF-8"><title>전종목 대시보드</title></head>
    <body>
    <h1>📊 전종목 대시보드</h1>
    <table border="1">
    <tr><th>종목명</th><th>티커</th><th>시장</th><th>현재가</th><th>1D</th><th>의견</th><th>점수</th></tr>
    """
    for _, r in df_summary.head(50).iterrows():
        dashboard_html += f"""
        <tr>
        <td><a href="{r['ticker']}.html">{r['name']}</a></td>
        <td>{r['ticker']}</td>
        <td>{r['market']}</td>
        <td>{r['close']:,.0f}</td>
        <td>{r['ret_1d']:+.2%}</td>
        <td>{r['opinion']}</td>
        <td>{r['signal']}</td>
        </tr>
        """
    dashboard_html += "</table></body></html>"
    
    dashboard_path = out_dir / "index.html"
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(dashboard_html)
    
    print(f"[render_stock_report] OK → {dashboard_path}")

if __name__ == "__main__":
    render_stock_report()
