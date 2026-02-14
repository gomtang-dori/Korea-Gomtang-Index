# src/caches/test_openapi_bydd_trd.py
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import requests

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
    # 숫자에 콤마가 있을 수도 있으니 제거
    s = s.replace(",", "")
    try:
        return int(float(s))
    except Exception:
        return None


def fetch_outblock_1(url: str, basDd: str, auth_key: str, timeout: int = 60) -> List[Dict[str, Any]]:
    headers = {
        "AUTH_KEY": auth_key,  # <-- 캡쳐에서 확정 [Source](https://www.genspark.ai/api/files/s/jYZ3Wt11)
        "Accept": "application/json",
    }
    params = {"basDd": basDd}

    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code != 200:
        # 에러일 때 본문이 JSON/HTML인지 확인하려고 일부 출력
        body_head = r.text[:300].replace("\n", "\\n")
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
    basDd = os.environ.get("BAS_DD", "20200414")  # 테스트용 기본값
    auth_key = _env("KRX_API_KEY")

    print(f"[test_openapi] BAS_DD={basDd}")
    print(f"[test_openapi] Using AUTH_KEY header (KRX_API_KEY length={len(auth_key)})")

    for name, url in ENDPOINTS.items():
        print(f"\n== {name} ==")
        rows = fetch_outblock_1(url=url, basDd=basDd, auth_key=auth_key)
        print(f"[{name}] row_count={len(rows)}")

        # 첫 2행 미리보기(키만 확인)
        preview = rows[:2]
        for i, r in enumerate(preview, start=1):
            keys = list(r.keys())
            print(f"[{name}] preview#{i} keys={keys}")
            print(
                f"[{name}] preview#{i} BAS_DD={r.get('BAS_DD')} ISU_CD={r.get('ISU_CD')} "
                f"CMPPREVDD_PRC={r.get('CMPPREVDD_PRC')} FLUC_RT={r.get('FLUC_RT')} MKT_NM={r.get('MKT_NM')}"
            )

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
