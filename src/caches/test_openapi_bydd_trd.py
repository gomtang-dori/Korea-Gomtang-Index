# src/caches/test_openapi_bydd_trd.py
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import requests


# 실서비스 base (중요)
# - 샘플: https://data-dbg.krx.co.kr/svc/sample/apis/...
# - 실서비스: https://data-dbg.krx.co.kr/svc/apis/...
BASE = "https://data-dbg.krx.co.kr/svc/apis/sto"

ENDPOINTS = {
    "KOSPI(stk)": f"{BASE}/stk_bydd_trd",
    "KOSDAQ(ksq)": f"{BASE}/ksq_bydd_trd",
}


def _env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Missing env var: {name}")
    return str(v)


def _to_int_safe(x: Any) -> int | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s == "-":
        return None
    s = s.replace(",", "")
    try:
        return int(float(s))
    except Exception:
        return None


def fetch_outblock_1(url: str, basDd: str, auth_key: str, timeout: int = 90) -> List[Dict[str, Any]]:
    # 인증은 요청 헤더 AUTH_KEY (사용자 캡쳐 근거)
    # - "Request 헤더에 인증키 값을 AUTH_KEY 필드에 추가하여 전달" [Source](https://www.genspark.ai/api/files/s/jYZ3Wt11)
    headers = {
        "AUTH_KEY": auth_key,
        "Accept": "application/json",
    }
    params = {"basDd": basDd}

    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code != 200:
        body_head = r.text[:400].replace("\n", "\\n")
        raise RuntimeError(f"HTTP {r.status_code} {r.reason} url={r.url} body_head={body_head}")

    data = r.json()
    ob = data.get("OutBlock_1")
    if not isinstance(ob, list):
        raise RuntimeError(f"Unexpected response: OutBlock_1 not list. keys={list(data.keys())}")
    return ob


def summarize_advdec(rows: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    adv = dec = unch = unknown = 0
    for row in rows:
        v = _to_int_safe(row.get("CMPPREVDD_PRC"))
        if v is None:
            unknown += 1
        elif v > 0:
            adv += 1
        elif v < 0:
            dec += 1
        else:
            unch += 1
    return adv, dec, unch, unknown


def main() -> None:
    # workflow_dispatch에서 BAS_DD를 넣거나, 로컬에서 export BAS_DD=YYYYMMDD
    basDd = os.environ.get("BAS_DD", "20200414")
    auth_key = _env("KRX_API_KEY")

    print(f"[test_openapi] basDd={basDd}")
    print(f"[test_openapi] AUTH_KEY header enabled (KRX_API_KEY length={len(auth_key)})")
    print(f"[test_openapi] base={BASE}")

    for name, url in ENDPOINTS.items():
        print(f"\n== {name} ==")
        rows = fetch_outblock_1(url=url, basDd=basDd, auth_key=auth_key)

        print(f"[{name}] row_count={len(rows)}")

        # 첫 2행 프리뷰
        for i, r in enumerate(rows[:2], start=1):
            print(f"[{name}] preview#{i} BAS_DD={r.get('BAS_DD')} ISU_CD={r.get('ISU_CD')} ISU_NM={r.get('ISU_NM')}")
            print(f"[{name}] preview#{i} CMPPREVDD_PRC={r.get('CMPPREVDD_PRC')} FLUC_RT={r.get('FLUC_RT')} MKT_NM={r.get('MKT_NM')}")

        adv, dec, unch, unknown = summarize_advdec(rows)
        denom = adv + dec
        f02_raw = None if denom == 0 else (adv - dec) / denom

        print(f"[{name}] adv={adv} dec={dec} unch={unch} unknown={unknown} adv+dec={denom} f02_raw={f02_raw}")

    print("\n[test_openapi] DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[test_openapi] ERROR: {e}", file=sys.stderr)
        raise
