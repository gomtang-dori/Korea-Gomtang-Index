# src/caches/cache_k200_ohlcv.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from lib.krx_http import KRXHTTP


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def load_members(path: Path) -> set[str]:
    df = pd.read_csv(path, dtype={"isu_cd": str})
    if "isu_cd" not in df.columns:
        raise RuntimeError("k200_members.csv must include 'isu_cd'")
    return set(df["isu_cd"].astype(str).tolist())


def upsert_parquet(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        out = new.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "isu_cd"])
    out = out.drop_duplicates(subset=["date", "isu_cd"], keep="last")
    out = out.sort_values(["date", "isu_cd"]).reset_index(drop=True)
    return out


def fetch_day(http: KRXHTTP, url: str, basDd: str) -> pd.DataFrame:
    js = http.get_json(url, {"basDd": basDd})
    out = js.get("OutBlock_1", [])
    if not isinstance(out, list):
        out = [] if out is None else [out]
    df = pd.DataFrame(out)
    return df


def main():
    url = os.environ.get("KRX_STK_BYDD_TRD_URL", "").strip()
    auth = os.environ.get("KRX_AUTH_KEY", "").strip()
    if not url or not auth:
        raise RuntimeError("Missing KRX_STK_BYDD_TRD_URL or KRX_AUTH_KEY")

    members_path = Path(os.environ.get("K200_MEMBERS_PATH", "data/k200_members.csv"))
    if not members_path.exists():
        raise RuntimeError(f"Missing {members_path} (repo에 커밋 유지로 합의)")

    start_s = os.environ.get("BACKFILL_START", "").strip()
    end_s = os.environ.get("BACKFILL_END", "").strip()
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))
    probe_only = os.environ.get("PROBE_ONLY", "0").strip() == "1"

    out_path = Path(os.environ.get("K200_OHLCV_CACHE_PATH", "data/cache/k200_ohlcv.parquet"))
    ensure_dir(out_path.parent)

    members = load_members(members_path)

    http = KRXHTTP(auth_key=auth, timeout=30)
    if probe_only:
        end = (pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1))
        # 최근 6영업일(월~금) 근사
        bas_list = []
        d = end
        while len(bas_list) < 6:
            if d.weekday() < 5:
                bas_list.append(d.strftime("%Y%m%d"))
            d -= pd.Timedelta(days=1)
        bas_list = sorted(bas_list)
    else:
        if not (start_s and end_s):
            raise RuntimeError("cache_k200_ohlcv requires BACKFILL_START/BACKFILL_END (or PROBE_ONLY=1)")
        start = pd.to_datetime(start_s, format="%Y%m%d") - pd.Timedelta(days=buffer_days)
        end = pd.to_datetime(end_s, format="%Y%m%d")
        bas_list = [d.strftime("%Y%m%d") for d in pd.date_range(start, end, freq="D")]

    old = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    new_rows = []

    # 진행률
    every = int(os.environ.get("PROGRESS_EVERY_N_DAYS", "25"))

    for i, basDd in enumerate(bas_list, start=1):
        try:
            raw = fetch_day(http, url, basDd)
            if raw.empty:
                continue

            # 스펙 컬럼 존재: ISU_CD, TDD_CLSPRC, TDD_HGPRC, TDD_LWPRC, ACC_TRDVOL, ACC_TRDVAL [Source]
            if "ISU_CD" not in raw.columns:
                continue

            raw["ISU_CD"] = raw["ISU_CD"].astype(str)
            sub = raw[raw["ISU_CD"].isin(members)].copy()
            if sub.empty:
                continue

            sub["date"] = pd.to_datetime(basDd, format="%Y%m%d")
            sub = sub.rename(columns={"ISU_CD": "isu_cd", "ISU_NM": "isu_nm"})
            sub["close"] = to_num(sub["TDD_CLSPRC"]) if "TDD_CLSPRC" in sub.columns else np.nan
            sub["high"] = to_num(sub["TDD_HGPRC"]) if "TDD_HGPRC" in sub.columns else np.nan
            sub["low"] = to_num(sub["TDD_LWPRC"]) if "TDD_LWPRC" in sub.columns else np.nan
            sub["volume"] = to_num(sub["ACC_TRDVOL"]) if "ACC_TRDVOL" in sub.columns else np.nan
            sub["trdval"] = to_num(sub["ACC_TRDVAL"]) if "ACC_TRDVAL" in sub.columns else np.nan
            sub["turnover"] = sub["close"] * sub["volume"]

            keep = ["date", "isu_cd", "isu_nm", "close", "high", "low", "volume", "turnover", "trdval"]
            keep = [c for c in keep if c in sub.columns]
            sub = sub[keep].copy()

            new_rows.append(sub)

        except Exception as e:
            # 캐시 단계에서는 에러를 누적시키지 않고 진행(필요 시 429는 KRXHTTP가 처리)
            print(f"[cache_k200_ohlcv] WARN basDd={basDd} err={repr(e)}")

        if every and (i % every == 0 or i == len(bas_list)):
            print(f"[cache_k200_ohlcv] {i}/{len(bas_list)} basDd={basDd} batch_rows={sum(len(x) for x in new_rows) if new_rows else 0}")

    if not new_rows:
        print("[cache_k200_ohlcv] no new rows (maybe holidays / probe period).")
        return

    new = pd.concat(new_rows, ignore_index=True)
    out = upsert_parquet(old, new)
    out.to_parquet(out_path, index=False)
    print(f"[cache_k200_ohlcv] OK rows={len(out)} -> {out_path}")


if __name__ == "__main__":
    import numpy as np
    main()
