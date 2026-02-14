# src/caches/cache_f09_credit_deposit_datago.py
from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

BASE = "https://apis.data.go.kr/1160100/service/GetKofiaStatisticsInfoService"

def _env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        if default is None:
            raise RuntimeError(f"Missing env: {name}")
        return default
    return v

def _parse_yyyymmdd(s: str) -> pd.Timestamp:
    ts = pd.to_datetime(s, format="%Y%m%d", errors="raise")
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()

def _clamp_end(backfill_end: pd.Timestamp) -> pd.Timestamp:
    today_utc = pd.Timestamp.utcnow().tz_localize(None).normalize()
    safe_end = today_utc - pd.Timedelta(days=1)
    if backfill_end > safe_end:
        print(f"[f09_cache] WARN: BACKFILL_END {backfill_end:%Y%m%d} > safe_end {safe_end:%Y%m%d}. clamp.")
        return safe_end
    return backfill_end

def _get_all_items(endpoint: str, service_key: str, begin: str, end: str, sleep_s: float = 0.1) -> list[dict]:
    url = f"{BASE}/{endpoint}"
    items: list[dict] = []
    page_no = 1
    num_rows = 1000

    while True:
        params = {
            "serviceKey": service_key,
            "pageNo": page_no,
            "numOfRows": num_rows,
            "resultType": "json",
            "beginBasDt": begin,
            "endBasDt": end,
        }
        r = requests.get(url, params=params, timeout=30)

        if r.status_code != 200:
            raise RuntimeError(
                f"[f09_cache] HTTP {r.status_code} endpoint={endpoint} "
                f"url={url}?{urlencode({k:v for k,v in params.items() if k!='serviceKey'})} "
                f"body={r.text[:500]}"
            )

        data = r.json()
        body = data.get("response", {}).get("body", {})
        total = int(body.get("totalCount", 0) or 0)

        page_items = body.get("items", {}).get("item", [])
        if isinstance(page_items, dict):
            page_items = [page_items]
        if page_items:
            items.extend(page_items)

        if total == 0:
            break
        if page_no * num_rows >= total:
            break

        page_no += 1
        time.sleep(sleep_s)

    return items

def main():
    out_path = Path(os.environ.get("F09_CACHE_PATH", "data/cache/f09_credit_deposit.parquet"))

    service_key = _env("DATA_GO_KR_SERVICE_KEY")
    backfill_start = _parse_yyyymmdd(_env("BACKFILL_START"))
    backfill_end = _clamp_end(_parse_yyyymmdd(_env("BACKFILL_END")))
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))

    start = (backfill_start - pd.Timedelta(days=buffer_days)).normalize()
    end = backfill_end.normalize()

    begin_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")

    print(f"[f09_cache] range={begin_s}~{end_s} out={out_path}")

    # Credit: 신용거래융자 전체(crdTrFingWhl)
    credit_items = _get_all_items(
        endpoint="getGrantingOfCreditBalanceInfo",
        service_key=service_key,
        begin=begin_s,
        end=end_s,
    )
    credit = pd.DataFrame(credit_items)
    if credit.empty:
        raise RuntimeError("[f09_cache] credit empty. Check serviceKey/승인/호출 파라미터.")

    need = {"basDt", "crdTrFingWhl"}
    if not need.issubset(set(credit.columns)):
        raise RuntimeError(f"[f09_cache] credit missing cols={need}. got={list(credit.columns)}")

    credit = credit[["basDt", "crdTrFingWhl"]].copy()
    credit["date"] = pd.to_datetime(credit["basDt"], format="%Y%m%d", errors="coerce")
    credit["credit_whl"] = pd.to_numeric(credit["crdTrFingWhl"], errors="coerce")
    credit = credit.dropna(subset=["date", "credit_whl"])[["date", "credit_whl"]]
    credit = credit.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    # Deposit proxy: 투자자예탁금(invrDpsgAmt)
    dep_items = _get_all_items(
        endpoint="getSecuritiesMarketTotalCapitalInfo",
        service_key=service_key,
        begin=begin_s,
        end=end_s,
    )
    dep = pd.DataFrame(dep_items)
    if dep.empty:
        raise RuntimeError("[f09_cache] deposit empty. Check serviceKey/승인/호출 파라미터.")

    need = {"basDt", "invrDpsgAmt"}
    if not need.issubset(set(dep.columns)):
        raise RuntimeError(f"[f09_cache] deposit missing cols={need}. got={list(dep.columns)}")

    dep = dep[["basDt", "invrDpsgAmt"]].copy()
    dep["date"] = pd.to_datetime(dep["basDt"], format="%Y%m%d", errors="coerce")
    dep["deposit"] = pd.to_numeric(dep["invrDpsgAmt"], errors="coerce")
    dep = dep.dropna(subset=["date", "deposit"])[["date", "deposit"]]
    dep = dep.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    merged = pd.merge(credit, dep, on="date", how="inner")
    merged["margin_ratio_raw"] = merged["credit_whl"] / merged["deposit"]
    merged = merged.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["margin_ratio_raw"])
    merged = merged.sort_values("date").reset_index(drop=True)

    # upsert
    if out_path.exists():
        old = pd.read_parquet(out_path)
        old["date"] = pd.to_datetime(old.get("date"), errors="coerce")
        for c in ["credit_whl", "deposit", "margin_ratio_raw"]:
            old[c] = pd.to_numeric(old.get(c), errors="coerce")
        old = old.dropna(subset=["date", "margin_ratio_raw"])
        out = pd.concat([old, merged], ignore_index=True)
        out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    else:
        out = merged

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[f09_cache] OK rows={len(out)} -> {out_path}")

if __name__ == "__main__":
    main()
