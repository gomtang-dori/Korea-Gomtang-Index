# src/probe_krx_vkospi.py
from __future__ import annotations

import os
import json
import pandas as pd
import requests


def call_one(url: str, auth_key: str, basDd: str) -> dict:
    headers = {"AUTH_KEY": auth_key, "Accept": "application/json"}
    params = {"basDd": basDd}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    print(f"[vk_probe] basDd={basDd} status={r.status_code}")
    r.raise_for_status()
    return r.json()


def to_df(js: dict) -> pd.DataFrame:
    if isinstance(js, dict) and isinstance(js.get("OutBlock_1"), list):
        return pd.DataFrame(js["OutBlock_1"])
    return pd.DataFrame()


def main():
    url = (os.getenv("KRX_DVRPROD_DD_TRD_URL") or "").strip()
    auth = (os.getenv("KRX_AUTH_KEY") or "").strip()
    if not url or not auth:
        raise SystemExit("Missing env: KRX_DVRPROD_DD_TRD_URL / KRX_AUTH_KEY")

    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    days = pd.date_range(today - pd.Timedelta(days=5), today, freq="D")  # calendar 6 days

    all_rows = []
    for d in days:
        basDd = d.strftime("%Y%m%d")
        js = call_one(url, auth, basDd)
        print("[vk_probe] top_keys:", list(js.keys()) if isinstance(js, dict) else type(js))
        print("[vk_probe] js_head:", json.dumps(js, ensure_ascii=False)[:300])

        df = to_df(js)
        print(f"[vk_probe] basDd={basDd} out_rows={len(df)} cols={list(df.columns)}")

        if not df.empty and "IDX_NM" in df.columns:
            idx_head = df["IDX_NM"].astype(str).head(30).tolist()
            print(f"[vk_probe] basDd={basDd} IDX_NM_head30={idx_head}")

            # show candidates that look like VKOSPI/변동성
            norm = df["IDX_NM"].astype(str)
            cand = df.loc[
                norm.str.contains("VKO", case=False, na=False)
                | norm.str.contains("변동", na=False)
                | norm.str.contains("VOL", case=False, na=False),
                [c for c in ["BAS_DD", "IDX_NM", "CLSPRC_IDX", "FLUC_RT"] if c in df.columns],
            ].head(30)
            if not cand.empty:
                print("[vk_probe] candidates:")
                print(cand.to_string(index=False))

        if not df.empty:
            df["_basDd_req"] = basDd
            all_rows.append(df)

    if all_rows:
        big = pd.concat(all_rows, ignore_index=True)
        print("[vk_probe] total_rows:", len(big))
        if "IDX_NM" in big.columns:
            top = big["IDX_NM"].astype(str).value_counts().head(50)
            print("[vk_probe] IDX_NM value_counts head50:")
            print(top.to_string())
    else:
        print("[vk_probe] no data rows in last 6 calendar days")


if __name__ == "__main__":
    main()
