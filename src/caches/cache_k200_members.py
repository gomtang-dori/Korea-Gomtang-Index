from __future__ import annotations
import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def get_k200_members_from_krx_web(target_date: str):
    """
    KRX 정보데이터시스템 웹사이트의 API를 직접 호출하여 
    KOSPI 200 구성 종목 리스트를 가져옵니다.
    """
    url = "http://data.krx.co.kr/comm/bldMng/getDesent.do"
    
    # KRX 웹사이트 호출에 필요한 파라미터 (지수구성종목 내역)
    # 1028은 KOSPI 200의 표준 코드입니다.
    data = {
        "bld": "dbms/MVD/04/04060100/mvd04060100_01",
        "indIdx": "1",
        "indIdx2": "028",
        "trdDd": target_date,
        "money": "1",
        "csvExport": "false"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.do?menuId=MDC0201030104"
    }

    try:
        response = requests.post(url, data=data, headers=headers, timeout=30)
        js = response.json()
        output = js.get("output", [])
        
        if not output:
            return pd.DataFrame()
            
        # ISU_SRT_CD(종목코드), ISU_ABBRV(종목명) 추출
        df = pd.DataFrame(output)
        res = pd.DataFrame({
            "asof_date": target_date,
            "isu_cd": df["ISU_SRT_CD"],
            "isu_nm": df["ISU_ABBRV"]
        })
        return res
    except Exception as e:
        print(f"[Error] KRX Web Direct Call Failed: {e}")
        return pd.DataFrame()

def main():
    out_path = Path("data/k200_members.csv")
    ensure_dir(out_path.parent)

    # 1. 확실한 최근 영업일 (금요일) 설정
    target_date = "20260213"
    print(f"[cache_k200_members] Direct calling KRX for {target_date}...")

    # 2. 직접 수집 시도
    df = get_k200_members_from_krx_web(target_date)

    # 3. 만약 실패했다면 어제 날짜로 한 번 더 시도
    if df.empty:
        target_date = "20260212"
        print(f"[cache_k200_members] Retrying with {target_date}...")
        df = get_k200_members_from_krx_web(target_date)

    # 4. 최종 검증 및 저장
    if df.empty or len(df) < 150:
        raise RuntimeError("CRITICAL: KRX 웹사이트로부터 데이터를 가져오지 못했습니다. 서버 상태를 확인하세요.")

    df = df.sort_values("isu_cd")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[cache_k200_members] Success! {len(df)} rows saved to {out_path}")

if __name__ == "__main__":
    main()
