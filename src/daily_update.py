import os
from datetime import datetime, timedelta
import pandas as pd
import requests
from pathlib import Path

DOCS = Path("docs")
DATA = Path("data")
DOCS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

def yyyymmdd(d): return d.strftime("%Y%m%d")

def fetch_f05(begin, end):
    # data.go.kr: 소매채권수익률요약
    url = "https://apis.data.go.kr/1160100/service/GetBondInfoService/getBondSecurityBenefitRate"
    service_key = os.environ["SERVICE_KEY"].strip()
    params = {
        "serviceKey": service_key,
        "resultType": "json",
        "numOfRows": 3000,
        "pageNo": 1,
        "beginBasDt": begin,
        "endBasDt": end,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    body = r.json()["response"]["body"]
    items = body.get("items", [])
    if isinstance(items, dict): items = items.get("item", [])
    df = pd.DataFrame(items)
    if df.empty:
        return df
    df = df.rename(columns={"basDt":"date","crdtSc":"grade","ctg":"bucket","bnfRt":"yield"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    return df[["date","grade","bucket","yield"]].dropna(subset=["date"])

def fetch_f10(begin, end):
    # ECOS StatisticSearch
    ecos_key = os.environ["ECOS_KEY"].strip()
    stat_code = "731Y003"
    cycle = "D"
    item_code1 = "0000003"
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{ecos_key}/json/kr/1/100000/{stat_code}/{cycle}/{begin}/{end}/{item_code1}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    rows = r.json().get("StatisticSearch", {}).get("row", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["TIME"], format="%Y%m%d", errors="coerce")
    df["usdkrw"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    return df[["date","usdkrw"]].dropna(subset=["date"])

def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def render_min_html(latest_date, f05_count, f10_count):
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <title>한국 곰탕 지수 - Daily Report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; max-width: 720px; }}
    h1 {{ margin: 0 0 8px; }}
    .muted {{ color:#666; }}
    ul {{ margin: 8px 0 0 18px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>한국 곰탕 지수 (자동 업데이트)</h1>
    <div class="muted">최종 업데이트(UTC): {datetime.utcnow().strftime("%Y-%m-%d %H:%M")}</div>
    <p>데이터 최신 날짜: <b>{latest_date}</b></p>
    <ul>
      <li>⑤ 소매채권수익률요약 rows(최근14일): <b>{f05_count}</b> (필드: basDt/crdtSc/ctg/bnfRt)</li>
      <li>⑩ USD/KRW rows(최근14일): <b>{f10_count}</b> (ECOS StatisticSearch)</li>
    </ul>
    <p class="muted">다음 단계: ⑨ + KOSPI200 + 히트맵/차트 + 투자의견 로직을 추가합니다.</p>
  </div>
</body>
</html>"""
    (DOCS / "index.html").write_text(html, encoding="utf-8")

def main():
    today = datetime.utcnow().date()
    begin = today - timedelta(days=14)
    begin_s, end_s = yyyymmdd(begin), yyyymmdd(today)

    f05 = fetch_f05(begin_s, end_s)
    f10 = fetch_f10(begin_s, end_s)

    if not f05.empty: save_csv(f05, DATA / "f05_recent.csv")
    if not f10.empty: save_csv(f10, DATA / "f10_recent.csv")

    latest = max(
        f05["date"].max() if not f05.empty else pd.Timestamp("1900-01-01"),
        f10["date"].max() if not f10.empty else pd.Timestamp("1900-01-01"),
    ).date()

    render_min_html(str(latest), len(f05), len(f10))

if __name__ == "__main__":
    main()
