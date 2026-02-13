# src/probe_krx_vkospi.py
from __future__ import annotations

import os
import json
import pandas as pd
import requests


def main():
    url = (os.getenv("KRX_DVRPROD_DD_TRD_URL") or "").strip()
    auth = (os.getenv("KRX_AUTH_KEY") or "").strip()
    if not url or not auth:
        raise SystemExit("Missing env: KRX_DVRPROD_DD_TRD_URL / KRX_AUTH_KEY")

    # "어제 하루" (캘린더 1일) - UTC 기준
    basDd = (pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    headers = {"AUTH_KEY": auth, "Accept": "application/json"}
    params = {"basDd": basDd}

    print(f"[vk_probe1d] url={url}")
    print(f"[vk_probe1d] basDd={basDd}")

    r = requests.get(url, headers=headers, params=params, timeout=30)
    print(f"[vk_probe1d] status={r.status_code}")
    print(f"[vk_probe1d] content-type={r.headers.get('content-type')}")
    print(f"[vk_probe1d] text_head={r.text[:800]}")

    # 404면 여기서 끝: URL이 틀린 것
    if r.status_code == 404:
        raise SystemExit("[vk_probe1d] HTTP 404 Not Found. Check KRX_DVRPROD_DD_TRD_URL secret value.")

    r.raise_for_status()

    try:
        js = r.json()
    except Exception as e:
        raise SystemExit(f"[vk_probe1d] JSON parse failed: {e}")

    if isinstance(js, dict):
        print(f"[vk_probe1d] top_keys={list(js.keys())}")
    else:
        print(f"[vk_probe1d] json_type={type(js)}")

    rows = js.get("OutBlock_1") if isinstance(js, dict) else None
    if not isinstance(rows, list):
        print("[vk_probe1d] OutBlock_1 not found or not list")
        print("[vk_probe1d] js_head:", json.dumps(js, ensure_ascii=False)[:1000])
        raise SystemExit(1)

    df = pd.DataFrame(rows)
    print(f"[vk_probe1d] OutBlock_1 rows={len(df)} cols={list(df.columns)}")

    if df.empty:
        print("[vk_probe1d] OutBlock_1 is empty (data not provided for this date or approval issue).")
        return

    # dump a few rows (compact)
    show_cols = [c for c in ["BAS_DD", "IDX_NM", "CLSPRC_IDX", "FLUC_RT"] if c in df.columns]
    print("[vk_probe1d] sample_rows:")
    print(df[show_cols].head(20).to_string(index=False))

    if "IDX_NM" in df.columns:
        vc = df["IDX_NM"].astype(str).value_counts().head(100)
        print("[vk_probe1d] IDX_NM value_counts head100:")
        print(vc.to_string())

        # VKOSPI/변동성 후보 빠르게 찾기
        nm = df["IDX_NM"].astype(str)
        cand = df.loc[
            nm.str.contains("VKO", case=False, na=False)
            | nm.str.contains("변동", na=False)
            | nm.str.contains("VOL", case=False, na=False),
            show_cols,
        ].head(50)
        if not cand.empty:
            print("[vk_probe1d] VKOSPI candidates:")
            print(cand.to_string(index=False))


if __name__ == "__main__":
    main()
