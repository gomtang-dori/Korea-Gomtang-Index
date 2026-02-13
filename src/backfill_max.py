# src/backfill_max.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ---- import your existing libs/modules here (keep your repo structure) ----
# 아래 3개는 이미 레포에 존재한다고 가정(앞서 K200/VKOSPI 작업 흐름)
from lib.krx_kospi_index import KRXKospiIndexAPI  # K200 close fetcher (kospi_dd_trd)
from lib.krx_dvrprod_index import KRXDrvProdIndexAPI  # VKOSPI fetcher (drvprod_dd_trd)
# Put/Call 및 기타(레포에 이미 구현된 함수/모듈에 맞게 import 경로 조정)
# 예: from lib.krx_putcall import fetch_put_call_by_range
# 예: from lib.ecos import fetch_usdkrw, fetch_10y
# -------------------------------------------------------------------------


# ----------------------- helpers -----------------------
def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _parse_yyyymmdd(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d")


def _fmt_hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def business_days_mon_fri(start: datetime, end: datetime) -> List[str]:
    """KRX 휴장일 캘린더 없이도 '영업일 기준'을 만족시키기 위한 월~금 리스트."""
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


@dataclass
class ProgressLogger:
    name: str
    total: int
    every: int = 25
    start_ts: float = time.time()

    def __post_init__(self):
        self.start_ts = time.time()
        self.total = max(int(self.total), 1)
        self.every = max(int(self.every), 1)

    def maybe_log(self, i: int, bas: str):
        if i % self.every != 0 and i != self.total:
            return
        elapsed = time.time() - self.start_ts
        speed = i / elapsed if elapsed > 0 else 0.0
        remain = self.total - i
        eta = remain / speed if speed > 0 else float("inf")
        pct = 100.0 * i / self.total
        print(
            f"[progress:{self.name}] {pct:6.2f}% ({i}/{self.total}) bas={bas} "
            f"elapsed={_fmt_hhmmss(elapsed)} speed={speed:.2f} days/s eta={_fmt_hhmmss(eta if np.isfinite(eta) else 0)}"
        )


def _finalize_missing(name: str, missing_days: List[str], total_days: int, max_rate: float):
    miss = len(missing_days)
    rate = miss / max(total_days, 1)
    print(f"[missing:{name}] missing={miss}/{total_days} rate={rate:.3%} max={max_rate:.3%}")
    if miss:
        head = missing_days[:200]
        print(f"[missing:{name}] days(head)={head}")
        if miss > 200:
            print(f"[missing:{name}] ... and {miss-200} more")
    if rate > max_rate:
        raise RuntimeError(f"{name} missing rate {rate:.3%} exceeds threshold {max_rate:.3%}")


def _safe_concat(frames: List[pd.DataFrame], cols: List[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


# ----------------------- K200 (index) fetch with skip -----------------------
def fetch_k200_close_range_skip(
    api: KRXKospiIndexAPI,
    bas_list: List[str],
    progress_every: int,
) -> Tuple[pd.DataFrame, List[str]]:
    missing = []
    frames = []
    prog = ProgressLogger("k200", total=len(bas_list), every=progress_every)

    for i, bas in enumerate(bas_list, start=1):
        try:
            df_day = api.fetch_k200_close_by_date(bas)  # must return columns: date, k200_close (or empty df)
            if df_day is None or df_day.empty:
                missing.append(bas)
            else:
                frames.append(df_day)
        except Exception:
            # 어떤 이유든(휴장일/일시 오류/필드 이슈) 그 날만 누락으로 처리
            missing.append(bas)

        prog.maybe_log(i, bas)

    out = _safe_concat(frames, cols=["date", "k200_close"])
    return out, missing


# ----------------------- VKOSPI fetch with skip -----------------------
def fetch_vkospi_range_skip(
    api: KRXDrvProdIndexAPI,
    bas_list: List[str],
    progress_every: int,
) -> Tuple[pd.DataFrame, List[str]]:
    missing = []
    frames = []
    prog = ProgressLogger("vkospi", total=len(bas_list), every=progress_every)

    for i, bas in enumerate(bas_list, start=1):
        try:
            df_day = api.fetch_vkospi_by_date(bas)  # must return columns: date, vkospi (or empty df)
            if df_day is None or df_day.empty:
                missing.append(bas)
            else:
                frames.append(df_day)
        except Exception:
            missing.append(bas)

        prog.maybe_log(i, bas)

    out = _safe_concat(frames, cols=["date", "vkospi"])
    return out, missing


# ----------------------- rolling percentile -----------------------
def rolling_percentile_0_100(series: pd.Series, window: int) -> pd.Series:
    """
    5y rolling percentile (0-100).
    window=1260 (대략 5년 영업일)
    """
    s = series.astype(float)

    def _pct_rank(x: np.ndarray) -> float:
        # 마지막 값이 윈도우 내에서 몇 % 위치인지
        last = x[-1]
        # nan 방어
        x = x[~np.isnan(x)]
        if len(x) == 0 or np.isnan(last):
            return np.nan
        return 100.0 * (np.sum(x <= last) - 1) / max(len(x) - 1, 1)

    return s.rolling(window, min_periods=max(60, window // 10)).apply(_pct_rank, raw=True)


# ----------------------- main -----------------------
def main():
    # ---- required envs for chunk mode ----
    start_s = _env("BACKFILL_START")
    end_s = _env("BACKFILL_END")
    chunk_out = _env("CHUNK_OUT")
    if not (start_s and end_s and chunk_out):
        raise RuntimeError("Chunk mode requires BACKFILL_START, BACKFILL_END, CHUNK_OUT env vars.")

    missing_rate_max = float(_env("MISSING_RATE_MAX", "0.02"))  # 2%
    progress_every = int(_env("PROGRESS_EVERY_N_DAYS", "25"))
    window_5y = int(_env("ROLLING_WINDOW_5Y", "1260"))

    start_dt = _parse_yyyymmdd(start_s)
    end_dt = _parse_yyyymmdd(end_s)
    bas_list = business_days_mon_fri(start_dt, end_dt)
    if not bas_list:
        raise RuntimeError("No business days generated for given range.")

    print(f"[backfill_chunk] range={start_s}..{end_s} days={len(bas_list)} out={chunk_out}")

    # ---- APIs from env (already used in your repo) ----
    k200_api = KRXKospiIndexAPI.from_env()
    vkospi_api = KRXDrvProdIndexAPI.from_env()

    # ---- 1) Fetch series with skip+missing list ----
    k200_df, miss_k200 = fetch_k200_close_range_skip(k200_api, bas_list, progress_every)
    vkospi_df, miss_vkospi = fetch_vkospi_range_skip(vkospi_api, bas_list, progress_every)

    # 누락률 체크(2% 초과 시 실패)
    _finalize_missing("k200", miss_k200, len(bas_list), missing_rate_max)
    _finalize_missing("vkospi", miss_vkospi, len(bas_list), missing_rate_max)

    # ---- 2) Merge base daily frame ----
    base = pd.DataFrame({"date": pd.to_datetime(bas_list, format="%Y%m%d")})
    base = base.merge(k200_df, on="date", how="left")
    base = base.merge(vkospi_df, on="date", how="left")

    # ---- 3) Factor scores (여기서는 예시로 f01(모멘텀) + f06(VKOSPI)만 안전하게 계산) ----
    # 실제 레포에서 이미 계산 중인 팩터 로직이 많다면,
    # "청크화" 첫 단계에서는 기존 로직을 그대로 두고,
    # empty day 스킵/누락률 체크를 통과한 뒤 점진적으로 이 구조에 맞춰 옮기는 것을 권장.

    # f06: VKOSPI (fear) -> Greed score = 100 - percentile
    base["f06_raw"] = base["vkospi"]
    base["f06_score"] = 100.0 - rolling_percentile_0_100(base["f06_raw"], window_5y)

    # f01: 예시(모멘텀) - 기존 레포 로직이 있으면 그걸 사용하세요.
    # 여기서는 placeholder: 20d return of k200_close
    base["k200_ret_20d"] = base["k200_close"].astype(float).pct_change(20)
    base["f01_raw"] = base["k200_ret_20d"]
    base["f01_score"] = rolling_percentile_0_100(base["f01_raw"], window_5y)

    # ---- 4) (중요) 기존 레포의 나머지 팩터(②~⑤,⑦~⑩)는 여기서 그대로 추가/병합 ----
    # 지금은 "청크/누락/병합 인프라"를 먼저 안정화하는 단계이므로,
    # 다음 커밋에서 기존 backfill_max.py의 factor 계산부를 이 base에 맞게 이식하는 방식으로 진행하면 안전합니다.

    # ---- 5) Save chunk parquet ----
    out_path = Path(chunk_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.sort_values("date").to_parquet(out_path, index=False)
    print(f"[backfill_chunk] wrote parquet rows={len(base)} -> {out_path}")


if __name__ == "__main__":
    main()
