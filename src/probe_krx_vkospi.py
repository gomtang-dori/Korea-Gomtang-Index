from __future__ import annotations

import os
import json
import pandas as pd
import requests


def try_call(url: str, auth: str, bas: str, param_name: str) -> tuple[int, str]:
    headers = {"AUTH_KEY": auth, "Accept": "application/json"}
    r = requests.get(url, headers=headers, params={param_name: bas}, timeout=30)
    head = (r.text or "")[:300].replace("\n", " ")
    return r.status_code, head


def main():
    auth = (os.getenv("KRX_AUTH_KEY") or "").strip()
    if not auth:
        raise SystemExit("Missing env: KRX_AUTH_KEY")

    # 어제 하루(UTC 기준)
    bas = (pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    print(f"[vk_probe_find] bas={bas}")

    # 사용자가 등록한 URL(우선)
    primary = (os.getenv("KRX_DVRPROD_DD_TRD_URL") or "").strip()

    # 후보 URL들: 운영에서 종종 경로가 미세하게 다름(서비스 ID/오타/구버전)
    candidates = []
    if primary:
        candidates.append(primary)

    # “idx” 계열의 흔한 변형 후보들(탐색용)
    candidates += [
        "https://data-dbg.krx.co.kr/svc/apis/idx/dvrprod_dd_trd",
        "https://data-dbg.krx.co.kr/svc/apis/idx/drvprod_dd_trd",
        "https://data-dbg.krx.co.kr/svc/apis/idx/dvrprod_bydd_trd",
        "https://data-dbg.krx.co.kr/svc/apis/idx/drvprod_bydd_trd",
        # 일부 서비스는 idx가 아닌 다른 그룹으로 노출되는 경우가 있어 후보 포함
        "https://data-dbg.krx.co.kr/svc/apis/drv/dvrprod_dd_trd",
        "https://data-dbg.krx.co.kr/svc/apis/drv/drvprod_dd_trd",
    ]

    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for u in candidates:
        if u and u not in seen:
            uniq.append(u)
            seen.add(u)

    # param 후보: basDd / basDt
    param_names = ["basDd", "basDt"]

    ok = None
    for u in uniq:
        for p in param_names:
            st, head = try_call(u, auth, bas, p)
            print(f"[vk_probe_find] try url={u} param={p} status={st} head={head}")
            if st == 200:
                ok = (u, p)
                break
        if ok:
            break

    if not ok:
        raise SystemExit("[vk_probe_find] No working endpoint found. Need API SPEC/portal test URL confirmation.")

    ok_url, ok_param = ok
    print(f"[vk_probe_find] FOUND OK url={ok_url} param={ok_param}")

    # 성공한 URL로 실제 JSON 파싱/OutBlock_1 덤프
    headers = {"AUTH_KEY": auth, "Accept": "application/json"}
    r = requests.get(ok_url, headers=headers, params={ok_param: bas}, timeout=30)
    js = r.json()
    print("[vk_probe_find] top_keys:", list(js.keys()) if isinstance(js, dict) else type(js))
    rows = js.get("OutBlock_1") if isinstance(js, dict) else None
    if not isinstance(rows, list):
        print("[vk_probe_find] json_head:", json.dumps(js, ensure_ascii=False)[:1000])
        raise SystemExit(1)

    df = pd.DataFrame(rows)
    print(f"[vk_probe_find] OutBlock_1 rows={len(df)} cols={list(df.columns)}")
    show_cols = [c for c in ["BAS_DD", "IDX_NM", "CLSPRC_IDX", "FLUC_RT"] if c in df.columns]
    print("[vk_probe_find] sample:")
    print(df[show_cols].head(30).to_string(index=False))

    if "IDX_NM" in df.columns:
        vc = df["IDX_NM"].astype(str).value_counts().head(100)
        print("[vk_probe_find] IDX_NM value_counts head100:")
        print(vc.to_string())


if __name__ == "__main__":
    main()
