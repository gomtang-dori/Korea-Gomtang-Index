from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import requests
from pathlib import Path

KRX_JSON_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"


def _env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        if default is None:
            raise RuntimeError(f"Missing env: {name}")
        return default
    return v


def _yyyymmdd(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y%m%d")


def _to_date_any(s: str) -> pd.Timestamp:
    s = str(s).strip().replace("-", "")
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.strip(): c for c in df.columns}
    for k in candidates:
        if k in cols:
            return cols[k]
    # 느슨한 매칭
    for c in df.columns:
        cc = str(c).replace(" ", "")
        for k in candidates:
            if cc == k.replace(" ", ""):
                return c
    return None


def _normalize_outblock(outblock: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(outblock)
    if df.empty:
        return df

    date_col = _pick_col(df, ["TRD_DD", "TRD_DD1", "TRD_DD2", "일자", "거래일", "DATE"])
    adv_col = _pick_col(df, ["UP", "상승", "상승종목수", "상승종목", "UP_CNT", "UP_ISSU_CNT"])
    dec_col = _pick_col(df, ["DOWN", "하락", "하락종목수", "하락종목", "DN_CNT", "DOWN_CNT", "DN_ISSU_CNT"])
    unch_col = _pick_col(df, ["UNCH", "보합", "보합종목수", "보합종목", "UNCH_CNT"])

    if not date_col or not adv_col or not dec_col:
        raise RuntimeError(f"Cannot map columns. cols={list(df.columns)}")

    out = pd.DataFrame({
        "date": df[date_col].apply(_to_date_any),
        "adv": pd.to_numeric(df[adv_col], errors="coerce"),
        "dec": pd.to_numeric(df[dec_col], errors="coerce"),
    })
    if unch_col:
        out["unch"] = pd.to_numeric(df[unch_col], errors="coerce")

    out = out.dropna(subset=["date", "adv", "dec"]).sort_values("date").reset_index(drop=True)
    return out


@dataclass
class Candidate:
    bld: str
    referer: str
    desc: str


def _candidates() -> list[Candidate]:
    # 자동탐색: KRX는 bld가 “dbms/MDC/STAT/standard/XXXX” 형태인 경우가 많습니다. [Source] 참조
    # 여기 후보들은 “상승/하락/보합 종목수”로 자주 쓰이는 통계 화면 계열에서 추정한 값들입니다.
    # 실패해도 probe가 후보를 순차 시도하므로 안전합니다.
    return [
        Candidate(
            bld="dbms/MDC/STAT/standard/MDCSTAT01501",
            referer="https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd?screenId=MDCSTAT015",
            desc="candidate-1: market issues up/down/unch (KOSPI/KOSDAQ)"
        ),
        Candidate(
            bld="dbms/MDC/STAT/standard/MDCSTAT01601",
            referer="https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd?screenId=MDCSTAT016",
            desc="candidate-2: market issues up/down/unch (alt)"
        ),
        Candidate(
            bld="dbms/MDC/STAT/standard/MDCSTAT01701",
            referer="https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd?screenId=MDCSTAT017",
            desc="candidate-3: market issues up/down/unch (alt)"
        ),
    ]


def _request_krx(bld: str, referer: str, params: dict) -> dict:
    headers = {
        "Referer": referer,
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
    }
    r = requests.post(KRX_JSON_URL, data=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    # probe 범위: 최근 20영업일 정도 커버(캘린더 기준 35일)
    end = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=35)

    strtDd = _yyyymmdd(start)
    endDd = _yyyymmdd(end)

    # KOSPI+KOSDAQ 합산 목표 → 일단 각각 받아 합산 (market id는 bld에 따라 다르므로 후보 매핑을 시도)
    # mktId 후보: STK(코스피), KSQ(코스닥) 등. 화면별로 달라질 수 있어 여러개 시도.
    mkt_ids = ["STK", "KSQ", "ALL", "KOSPI", "KOSDAQ"]

    out_dir = Path("data/cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    for cand in _candidates():
        print(f"[probe_ad] try {cand.desc} bld={cand.bld}")

        got = {}
        for mkt in mkt_ids:
            payload = {
                "bld": cand.bld,
                "locale": "ko_KR",
                "strtDd": strtDd,
                "endDd": endDd,
                "mktId": mkt,
                "csvxls_isNo": "false",
            }

            try:
                js = _request_krx(cand.bld, cand.referer, payload)
            except Exception as e:
                continue

            # outblock 후보 키들(화면마다 다름)
            outblock = None
            for k in ["OutBlock_1", "output", "block1", "OutBlock"]:
                if k in js and isinstance(js[k], list) and len(js[k]) > 0:
                    outblock = js[k]
                    break
            if outblock is None:
                continue

            try:
                df = _normalize_outblock(outblock)
            except Exception:
                continue

            if df.empty:
                continue

            got[mkt] = df
            print(f"[probe_ad] mktId={mkt} rows={len(df)} cols={list(df.columns)} date={df['date'].min().date()}~{df['date'].max().date()}")

        # 성공 조건: 코스피/코스닥을 각각 확보했거나, ALL로 한 번에 나왔거나
        if "ALL" in got:
            final = got["ALL"].copy()
            used = {"bld": cand.bld, "referer": cand.referer, "mode": "ALL"}
        elif "STK" in got and "KSQ" in got:
            a = got["STK"].rename(columns={"adv": "adv_kospi", "dec": "dec_kospi"})
            b = got["KSQ"].rename(columns={"adv": "adv_kosdaq", "dec": "dec_kosdaq"})
            final = a.merge(b, on="date", how="outer").sort_values("date").reset_index(drop=True)
            final["adv"] = final["adv_kospi"].fillna(0) + final["adv_kosdaq"].fillna(0)
            final["dec"] = final["dec_kospi"].fillna(0) + final["dec_kosdaq"].fillna(0)
            final = final[["date", "adv", "dec"]]
            used = {"bld": cand.bld, "referer": cand.referer, "mode": "STK+KSQ"}
        else:
            continue

        # 저장
        meta_path = out_dir / "probe_ad_counts_krx_meta.json"
        data_path = out_dir / "probe_ad_counts_krx.parquet"
        csv_path = out_dir / "probe_ad_counts_krx.csv"

        final.to_parquet(data_path, index=False)
        final.to_csv(csv_path, index=False, encoding="utf-8-sig")
        meta_path.write_text(json.dumps(used, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[probe_ad] SUCCESS -> {data_path} / {csv_path}")
        print(f"[probe_ad] META -> {meta_path}  ({used})")
        print(final.tail(5).to_string(index=False))
        return

    raise RuntimeError(
        "[probe_ad] FAILED: no candidate bld/referer/mktId worked.\n"
        "→ KRX 화면에서 해당 통계 페이지(상승/하락 종목수) URL(screenId)와 개발자도구 payload(bld)를 알려주시면 100% 고정 가능합니다."
    )


if __name__ == "__main__":
    main()
