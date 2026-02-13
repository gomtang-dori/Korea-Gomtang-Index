import os
import json
import requests

def _probe_one(url: str, auth_key: str, basDd: str, tag: str):
    headers = {"AUTH_KEY": auth_key, "Accept": "application/json"}
    params = {"basDd": basDd}

    print(f"\n[{tag}] url={url}")
    r = requests.get(url, headers=headers, params=params, timeout=30)
    print(f"[{tag}] status={r.status_code}")
    print(f"[{tag}] text(head)={r.text[:800]}")

    try:
        js = r.json()
    except Exception as e:
        print(f"[{tag}] not json: {e}")
        return None

    # 첫 row 찾기(응답 구조가 달라도 최대한 견고하게)
    def find_first_row(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                got = find_first_row(v)
                if got is not None:
                    return got
        elif isinstance(obj, list) and obj:
            if isinstance(obj[0], dict):
                return obj[0]
            return find_first_row(obj[0])
        return None

    row = find_first_row(js)
    if isinstance(js, dict):
        print(f"[{tag}] top keys={list(js.keys())}")
    if row is None:
        print(f"[{tag}] first row not found. dump(head)={json.dumps(js, ensure_ascii=False)[:1500]}")
        return js

    print(f"[{tag}] first-row keys={list(row.keys())}")
    print(f"[{tag}] first-row sample={json.dumps(row, ensure_ascii=False)[:800]}")
    return js


def main():
    auth_key = (os.environ.get("KRX_AUTH_KEY") or "").strip()
    eqsop_url = (os.environ.get("KRX_EQSOP_URL") or "").strip()
    eqkop_url = (os.environ.get("KRX_EQKOP_URL") or "").strip()
    basDd = (os.environ.get("KRX_BASDD") or "20200414").strip()

    if not auth_key:
        raise SystemExit("Missing KRX_AUTH_KEY (GitHub Secrets에 등록 필요)")
    if not eqsop_url or not eqkop_url:
        raise SystemExit("Missing KRX_EQSOP_URL or KRX_EQKOP_URL (GitHub Secrets에 등록 필요)")

    _probe_one(eqsop_url, auth_key, basDd, "EQSOP(KOSPI)")
    _probe_one(eqkop_url, auth_key, basDd, "EQKOP(KOSDAQ)")


if __name__ == "__main__":
    main()
