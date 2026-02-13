from __future__ import annotations

import os
import time
from pathlib import Path
import pandas as pd

from lib.krx_kospi_index import KRXKospiIndexAPI


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_dt(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date"])
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def upsert_ts(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        out = new.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)
    out = safe_dt(out)
    out = out.dropna(subset=["date"])
    out = out.drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def fmt_hhmmss(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def business_days(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    # KRX는 영업일 데이터만 의미 있으므로 Business Day(B)로 제한
    # end 포함
    days = pd.date_range(start, end, freq="B")
    return [d.strftime("%Y%m%d") for d in days]


def fetch_range_by_business_days(api: KRXKospiIndexAPI, start: pd.Timestamp, end: pd.Timestamp, every: int = 25):
    bas_list = business_days(start, end)
    frames = []
    missing = []
    t0 = time.time()

    total = len(bas_list)
    for i, bas in enumerate(bas_list, start=1):
        try:
            df_day = api.fetch_k200_close_by_date(bas)  # 핵심: 일별 호출 (영업일만)
            if df_day is None or df_day.empty:
                missing.append(bas)
            else:
                frames.append(df_day)
        except Exception as e:
            missing.append(bas)
            print(f"[cache_k200_close] WARN basDt={bas} err={repr(e)}")

        if every and (i % every == 0 or i == total):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(
                f"[cache_k200_close] progress {i}/{total} basDt={bas} "
                f"elapsed={fmt_hhmmss(elapsed)} speed={rate:.2f} day/s missing={len(missing)}"
            )

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "k200_close"])
    out = safe_dt(out)
    out["k200_close"] = pd.to_numeric(out.get("k200_close"), errors="coerce")
    out = out.dropna(subset=["date", "k200_close"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return out, missing, total


def main():
    start_s = os.environ.get("BACKFILL_START", "").strip()
    end_s = os.environ.get("BACKFILL_END", "").strip()
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))
    out_path = Path(os.environ.get("K200_CACHE_PATH", "data/cache/k200_close.parquet"))

    # 진행률 로그 주기(영업일 기준)
    every = int(os.environ.get("PROGRESS_EVERY_N_DAYS", "25"))

    if not (start_s and end_s):
        raise RuntimeError("cache_k200_close requires BACKFILL_START/BACKFILL_END")

    start = pd.to_datetime(start_s, format="%Y%m%d") - pd.Timedelta(days=buffer_days)
    end = pd.to_datetime(end_s, format="%Y%m%d")

    api = KRXKospiIndexAPI.from_env()

    # ✅ 핵심 변경: 거래일만 골라서 fetch_k200_close_by_date로 가져오기
    k200, missing_days, requested = fetch_range_by_business_days(api, start, end, every=every)

    # 누락률은 로깅만(강제 실패는 backfill에서 정책적으로 처리)
    miss_rate = (len(missing_days) / requested) if requested else 0.0
    if missing_days:
        print(f"[cache_k200_close] missing sample (up to 10): {missing_days[:10]}")
    print(f"[cache_k200_close] missing={len(missing_days)}/{requested} rate={miss_rate:.2%}")

    ensure_dir(out_path.parent)
    old = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    out = upsert_ts(old, k200)
    out.to_parquet(out_path, index=False)
    print(f"[cache_k200_close] OK rows={len(out)} -> {out_path}")


if __name__ == "__main__":
    main()
