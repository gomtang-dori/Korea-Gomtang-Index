import os
import json
import requests

def main():
    key = os.environ.get("SERVICE_KEY")
    if not key:
        raise SystemExit("SERVICE_KEY missing (GitHub Secrets에 등록 필요)")

    # TODO: 여기에 'KRX OpenAPI 파생상품(옵션) 일별 지표'의 정확한 endpoint URL을 넣어야 합니다.
    # 지금 단계에서는 URL/파라미터를 확정하기 위해 'probe'만 합니다.
    #
    # 예시 형태(가짜 URL 아님): 사용 중인 KRX OpenAPI 문서의 "호출 URL"을 그대로 넣으세요.
    url = os.environ.get("KRX_PUTCALL_URL", "").strip()
    if not url:
        raise SystemExit(
            "KRX_PUTCALL_URL env missing. \n"
            "방법: repo Settings → Secrets and variables → Actions → New secret 로\n"
            "KRX_PUTCALL_URL에 실제 호출 URL을 넣어주세요.\n"
            "(또는 이 파일에서 url=... 로 직접 하드코딩도 가능하지만 비추천)"
        )

    # 일반적으로 공공데이터포털 계열은 serviceKey를 querystring으로 받습니다.
    # KRX OpenAPI도 유사한 경우가 많아 우선 serviceKey만 붙여 probe 합니다.
    params = {
        "serviceKey": key,
        "resultType": "json",
        # 아래 날짜/기타 파라미터는 KRX API 스펙에 맞게 바꿔야 합니다.
        # probe 목적: "일단 성공 응답을 받고 outBlock 필드명을 알아내기"
    }

    print("[probe] GET", url)
    r = requests.get(url, params=params, timeout=30)
    print("[probe] status:", r.status_code)
    print("[probe] first 800 chars:\n", r.text[:800])

    try:
        js = r.json()
    except Exception as e:
        raise SystemExit(f"[probe] not json: {e}")

    # 구조를 최대한 덤프
    print("[probe] top keys:", list(js.keys()) if isinstance(js, dict) else type(js))

    # 공공데이터포털 스타일(response/body/items/item)을 가정해 최대한 찾기
    def find_first_item(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() == "item" and isinstance(v, list) and v:
                    return v[0]
                got = find_first_item(v)
                if got is not None:
                    return got
        elif isinstance(obj, list) and obj:
            return find_first_item(obj[0])
        return None

    item = find_first_item(js)
    if item is None:
        print("[probe] Could not find item list. Dump full json:")
        print(json.dumps(js, ensure_ascii=False)[:3000])
        return

    if isinstance(item, dict):
        print("[probe] item keys:", list(item.keys()))
        print("[probe] sample item:", json.dumps(item, ensure_ascii=False)[:1200])
    else:
        print("[probe] item type:", type(item))

if __name__ == "__main__":
    main()
