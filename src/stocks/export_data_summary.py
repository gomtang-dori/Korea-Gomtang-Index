#!/usr/bin/env python3
"""
수집된 데이터 요약 CSV 생성
1. master_summary.csv: 전체 종목 목록 + 데이터 수집 상태
2. prices_summary.csv: 가격 데이터 통합 (최근 30일)
3. flows_summary.csv: 투자자 매매 통합 (최근 30일)
4. financials_summary.csv: 재무제표 통합
5. signals_summary.csv: 투자 의견 통합
"""
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

def export_master_summary():
    """종목 목록 + 데이터 수집 상태"""
    print("[1/5] master_summary.csv 생성 중...")
    
    master_path = Path("data/stocks/master/listings.parquet")
    if not master_path.exists():
        print("  ⚠️  마스터 파일 없음")
        return
    
    df_master = pd.read_parquet(master_path)
    
    # 각 종목별 데이터 수집 상태 체크
    status_rows = []
    for _, row in df_master.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        market = row["market"]
        
        # 파일 존재 여부
        has_price = Path(f"data/stocks/raw/prices/{ticker}.csv").exists()
        has_flow = Path(f"data/stocks/raw/krx_flows/{ticker}.csv").exists()
        has_dart = Path(f"data/stocks/raw/dart/{ticker}_quarterly.json").exists()
        has_financials = Path(f"data/stocks/curated/{ticker}/financials_quarterly.parquet").exists()
        has_features = Path(f"data/stocks/analysis/{ticker}/features.parquet").exists()
        
        # 데이터 건수
        price_count = 0
        flow_count = 0
        
        if has_price:
            df_p = pd.read_csv(f"data/stocks/raw/prices/{ticker}.csv", encoding="utf-8-sig")
            price_count = len(df_p)
        
        if has_flow:
            df_f = pd.read_csv(f"data/stocks/raw/krx_flows/{ticker}.csv", encoding="utf-8-sig")
            flow_count = len(df_f)
        
        status_rows.append({
            "ticker": ticker,
            "name": name,
            "market": market,
            "has_price": has_price,
            "price_rows": price_count,
            "has_flow": has_flow,
            "flow_rows": flow_count,
            "has_dart": has_dart,
            "has_financials": has_financials,
            "has_features": has_features,
            "data_complete": all([has_price, has_flow, has_financials, has_features])
        })
    
    df_status = pd.DataFrame(status_rows)
    out_path = Path("docs/stocks/master_summary.csv")
    df_status.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ OK → {out_path} ({len(df_status)} 종목)")

def export_prices_summary():
    """가격 데이터 통합 (최근 30일)"""
    print("[2/5] prices_summary.csv 생성 중...")
    
    master_path = Path("data/stocks/master/listings.parquet")
    if not master_path.exists():
        return
    df_master = pd.read_parquet(master_path)
    
    cutoff_date = datetime.now() - timedelta(days=30)
    all_prices = []
    
    for _, row in df_master.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        
        price_path = Path(f"data/stocks/raw/prices/{ticker}.csv")
        if not price_path.exists():
            continue
        
        df_p = pd.read_csv(price_path, encoding="utf-8-sig")
        df_p["date"] = pd.to_datetime(df_p["date"])
        df_p = df_p[df_p["date"] >= cutoff_date].copy()
        
        if df_p.empty:
            continue
        
        df_p["ticker"] = ticker
        df_p["name"] = name
        all_prices.append(df_p[["date", "ticker", "name", "close", "volume"]])
    
    if all_prices:
        df_all = pd.concat(all_prices, ignore_index=True)
        df_all.sort_values(["date", "ticker"], inplace=True)
        out_path = Path("docs/stocks/prices_summary.csv")
        df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ OK → {out_path} ({len(df_all)} rows)")
    else:
        print("  ⚠️  가격 데이터 없음")

def export_flows_summary():
    """투자자 매매 통합 (최근 30일)"""
    print("[3/5] flows_summary.csv 생성 중...")
    
    master_path = Path("data/stocks/master/listings.parquet")
    if not master_path.exists():
        return
    df_master = pd.read_parquet(master_path)
    
    cutoff_date = datetime.now() - timedelta(days=30)
    all_flows = []
    
    for _, row in df_master.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        
        flow_path = Path(f"data/stocks/raw/krx_flows/{ticker}.csv")
        if not flow_path.exists():
            continue
        
        df_f = pd.read_csv(flow_path, encoding="utf-8-sig")
        
        # 날짜 컬럼 찾기
        date_col = "날짜" if "날짜" in df_f.columns else "date"
        df_f[date_col] = pd.to_datetime(df_f[date_col])
        df_f = df_f[df_f[date_col] >= cutoff_date].copy()
        
        if df_f.empty:
            continue
        
        df_f["ticker"] = ticker
        df_f["name"] = name
        
        # 필요한 컬럼만 추출 (외국인·기관)
        keep_cols = [date_col, "ticker", "name"]
        for col in df_f.columns:
            if "외국인" in col or "기관" in col:
                keep_cols.append(col)
        
        df_f_filtered = df_f[keep_cols].copy()
        df_f_filtered.rename(columns={date_col: "date"}, inplace=True)
        all_flows.append(df_f_filtered)
    
    if all_flows:
        df_all = pd.concat(all_flows, ignore_index=True)
        df_all.sort_values(["date", "ticker"], inplace=True)
        out_path = Path("docs/stocks/flows_summary.csv")
        df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ OK → {out_path} ({len(df_all)} rows)")
    else:
        print("  ⚠️  수급 데이터 없음")

def export_financials_summary():
    """재무제표 통합"""
    print("[4/5] financials_summary.csv 생성 중...")
    
    master_path = Path("data/stocks/master/listings.parquet")
    if not master_path.exists():
        return
    df_master = pd.read_parquet(master_path)
    
    all_fin = []
    
    for _, row in df_master.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        
        fin_path = Path(f"data/stocks/curated/{ticker}/financials_quarterly.parquet")
        if not fin_path.exists():
            continue
        
        df_fin = pd.read_parquet(fin_path)
        if df_fin.empty:
            continue
        
        df_fin["ticker"] = ticker
        df_fin["name"] = name
        all_fin.append(df_fin)
    
    if all_fin:
        df_all = pd.concat(all_fin, ignore_index=True)
        out_path = Path("docs/stocks/financials_summary.csv")
        df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ OK → {out_path} ({len(df_all)} rows)")
    else:
        print("  ⚠️  재무 데이터 없음")

def export_signals_summary():
    """투자 의견 통합 (최신)"""
    print("[5/5] signals_summary.csv 생성 중...")
    
    master_path = Path("data/stocks/master/listings.parquet")
    if not master_path.exists():
        return
    df_master = pd.read_parquet(master_path)
    
    all_signals = []
    
    for _, row in df_master.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        market = row["market"]
        
        feat_path = Path(f"data/stocks/analysis/{ticker}/features.parquet")
        if not feat_path.exists():
            continue
        
        df_feat = pd.read_parquet(feat_path)
        if df_feat.empty:
            continue
        
        # 최신 데이터
        latest = df_feat.iloc[-1]
        
        signal_fund = latest.get("signal_fundamentals", 0)
        signal_flow = latest.get("signal_flows", 0)
        total_signal = latest.get("signal", 0)
        
        if total_signal >= 3:
            opinion = "BUY"
            position = "+10~+20%"
        elif total_signal >= 1:
            opinion = "HOLD"
            position = "0%"
        else:
            opinion = "SELL"
            position = "-10~-30%"
        
        all_signals.append({
            "ticker": ticker,
            "name": name,
            "market": market,
            "date": latest.get("date", ""),
            "close": latest.get("close", 0),
            "ret_1d": latest.get("ret_1d", 0),
            "ret_5d": latest.get("ret_5d", 0),
            "ret_20d": latest.get("ret_20d", 0),
            "signal_fundamentals": signal_fund,
            "signal_flows": signal_flow,
            "signal_total": total_signal,
            "opinion": opinion,
            "position_guide": position
        })
    
    if all_signals:
        df_all = pd.DataFrame(all_signals)
        df_all.sort_values("signal_total", ascending=False, inplace=True)
        out_path = Path("docs/stocks/signals_summary.csv")
        df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ OK → {out_path} ({len(df_all)} 종목)")
    else:
        print("  ⚠️  투자 의견 데이터 없음")

def main():
    print("[export_data_summary] 시작...\n")
    
    # 출력 디렉토리 생성
    Path("docs/stocks").mkdir(parents=True, exist_ok=True)
    
    export_master_summary()
    export_prices_summary()
    export_flows_summary()
    export_financials_summary()
    export_signals_summary()
    
    print("\n[export_data_summary] 완료 ✅")
    print("  → docs/stocks/*.csv 파일 생성")

if __name__ == "__main__":
    main()
