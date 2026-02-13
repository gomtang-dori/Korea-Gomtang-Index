# src/probe_krx_vkospi.py
from __future__ import annotations

import os
import json
import pandas as pd
import requests

def call_one(url: str, auth_key: str, basDd: str) -> dict | None:
    headers = {"AUTH_KEY": auth_key, "Accept": "application/json"}
    params = {"basDd": basDd}
    try:
        # 주말/공휴일 등 데이터가 없는 날 404가 발생할 수 있으므로 try-except 처리
        r = requests.get(url, headers=headers, params=params, timeout=30)
        print(f"[vk_probe] basDd={basDd} status={r.status_code}")
        
        if r.status_code == 404:
            print(f"[vk_probe] {basDd}는 데이터가 없거나 잘못된 경로입니다. (Skip)")
            return None
            
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        print(f"[vk_probe] {basDd} 호출 중 HTTP 에러 발생: {e}")
        return None
    except Exception as e:
        print(f"[vk_probe] {basDd} 예기치 못한 에러: {e}")
        return None

def to_df(js: dict | None) -> pd.DataFrame:
    if js and isinstance(js, dict) and isinstance(js.get("OutBlock_1"), list):
        return pd.DataFrame(js["OutBlock_1"])
    return pd.DataFrame()

def main():
    # 환경 변수 로드 및 공백 제거
    url = (os.getenv("KRX_DVRPROD_DD_TRD_URL") or "").strip()
    auth = (os.getenv("KRX_AUTH_KEY") or "").strip()
    
    if not url or not auth:
        raise SystemExit("Missing env: KRX_DVRPROD_DD_TRD_URL / KRX_AUTH_KEY")

    # 오늘 기준 최근 6일간의 캘린더 데이 생성
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    days = pd.date_range(today - pd.Timedelta(days=5), today, freq="D")

    all_rows = []
    for d in days:
        basDd = d.strftime("%Y%m%d")
        js = call_one(url, auth, basDd)
        
        if js is None:
            continue

        print("[vk_probe] top_keys:", list(js.keys()) if isinstance(js, dict) else type(js))
        
        df = to_df(js)
        if df.empty:
            print(f"[vk_probe] basDd={basDd} 데이터가 비어있습니다.")
            continue

        print(f"[vk_probe] basDd={basDd} out_rows={len(df)} cols={list(df.columns)}")

        # VKOSPI/변동성 관련 키워드 탐색
        if "IDX_NM" in df.columns:
            norm = df["IDX_NM"].astype(str)
            cand = df.loc[
                norm.str.contains("VKO", case=False, na=False)
                | norm.str.contains("변동", na=False)
                | norm.str.contains("VOL", case=False, na=False),
                [c for c in ["BAS_DD", "IDX_NM", "CLSPRC_IDX", "FLUC_RT"] if c in df.columns],
            ].head(30)
            
            if not cand.empty:
                print("[vk_probe] candidates found:")
                print(cand.to_string(index=False))

        df["_basDd_req"] = basDd
        all_rows.append(df)

    if all_rows:
        big = pd.concat(all_rows, ignore_index=True)
        print("[vk_probe] total_rows gathered:", len(big))
        if "IDX_NM" in big.columns:
            top = big["IDX_NM"].astype(str).value_counts().head(50)
            print("[vk_probe] IDX_NM value_counts head50:")
            print(top.to_string())
    else:
        print("[vk_probe] 최근 6일간 수집된 데이터가 없습니다.")

if __name__ == "__main__":
    main()
