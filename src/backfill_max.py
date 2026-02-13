# src/backfill_max.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from lib.pykrx_factors import (
    factor1_momentum,
    factor2_strength,
    factor3_breadth,
    factor6_volatility,
    factor7_safe_haven,
    factor8_foreign_netbuy,
)
from lib.krx_putcall import fetch_putcall_ratio_by_date
from lib.krx_kospi_index import KRXKospiIndexAPI


# ------------------- CFG -------------------
@dataclass
class CFG:
    ROLLING_DAYS: int = 252 * 5
    MIN_OBS: int = 252
    DATA_DIR: str = "data"
    USDKRW_LEVEL_PATH: str = "data/usdkrw_level.parquet"
    VKOSPI_LEVEL_PATH: str = "data/vkospi_level.parquet"
    W: dict = None

    def __post_init__(self):
        if self.W is None:
            self.W = {
                "f01_score": 0.10,
                "f02_score": 0.10,
                "f03_score": 0.10,
                "f04_score": 0.10,
                "f05_score": 0.05,
                "f06_score": 0.125,
                "f07_score": 0.10,
                "f08_score": 0.10,
                "f10_score": 0.10,
            }


cfg = CFG()


# ------------------- helpers -------------------
def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def safe_to_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[col])
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def rolling_percentile(s: pd.Series, window: int, min_obs: int) -> pd.Series:
    def _pct(x):
        if len(x) < min_obs:
            return np.nan
        return float(pd.Series(x).rank(pct=True).iloc[-1] * 100.0)
    return s.rolling(window=window, min_periods=min_obs).apply(_pct, raw=False)


def forward_return(level: pd.Series, n: int) -> pd.Series:
    return level.shift(-n) / level - 1.0


def forward_win(level: pd.Series, n: int) -> pd.Series:
    return (forward_return(level, n) > 0).astype(float)


def renormalize_weights(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    score_cols = list(weights.keys())
    w = pd.Series(weights, dtype=float)
    avail = df[score_cols].notna().astype(float)
    w_mat = avail.mul(w, axis=1)
    w_sum = w_mat.sum(axis=1).replace(0, np.nan)
    return w_mat.div(w_sum, axis=0)


def load_existing_f05_f10(index_df: pd.DataFrame) -> pd.DataFrame:
    keep = ["date", "f05_raw", "f10_raw"]
    cols = [c for c in keep if c in index_df.columns]
    if not cols:
        return pd.DataFrame(columns=["date"])
    return index_df[cols].copy()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _fmt_hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _finalize_missing(name: str, missing_days: list[str], total_days: int, max_rate: float):
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


def _progress_line(tag: str, i: int, n: int, bas: str, t0: float, every: int):
    if every <= 0:
        return
    if i % every != 0 and i != n:
        return
    elapsed = time.time() - t0
    speed = i / elapsed if elapsed > 0 else 0.0
    remain = n - i
    eta = remain / speed if speed > 0 else float("inf")
    pct = 100.0 * i / max(n, 1)
    print(
        f"[progress:{tag}] {pct:6.2f}% ({i}/{n}) bas={bas} "
        f"elapsed={_fmt_hhmmss(elapsed)} speed={speed:.2f} days/s eta={_fmt_hhmmss(eta if np.isfinite(eta) else 0)}"
    )


# ------------------- K200 chunk fetch (skip empty day) -------------------
def fetch_k200_close_range_chunked(
    api: KRXKospiIndexAPI,
    start: pd.Timestamp,
    end: pd.Timestamp,
    progress_every: int,
) -> tuple[pd.DataFrame, list[str], int]:
    """
    일단위로 basDt 요청 -> 빈 결과는 SKIP + 누락일 기록.
    반환:
      - k200_df(date,k200_close)
      - missing_days(YYYYMMDD list)
      - requested_day_count
    """
    # 캘린더 범위 (휴장일 포함 가능). 누락률 판단은 "요청일수" 대비로 계산.
    days = pd.date_range(start, end, freq="D")
    missing = []
    frames = []
    t0 = time.time()

    for i, d in enumerate(days, start=1):
        bas = d.strftime("%Y%m%d")
        try:
            df_day = api.fetch_k200_close_by_date(bas)  # 내부에서 basDt/basDd fallback 및 close col 자동 선택 처리된 상태 가정
            if df_day is None or df_day.empty:
                missing.append(bas)
            else:
                frames.append(df_day)
        except Exception:
            missing.append(bas)

        _progress_line("k200", i, len(days), bas, t0, progress_every)

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = safe_to_datetime(out, "date")
        if "k200_close" in out.columns:
            out["k200_close"] = pd.to_numeric(out["k200_close"], errors="coerce")
        out = out.dropna(subset=["date", "k200_close"]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=["date", "k200_close"])

    return out, missing, len(days)


# ------------------- main -------------------
def main():
    ensure_dir(cfg.DATA_DIR)

    # ---- chunk mode required ----
    # workflow matrix가 이 값을 주입합니다.
    start_s = _env("BACKFILL_START")
    end_s = _env("BACKFILL_END")
    chunk_out = _env("CHUNK_OUT")
    if not (start_s and end_s and chunk_out):
        raise RuntimeError("Chunk mode requires BACKFILL_START, BACKFILL_END, CHUNK_OUT env vars.")

    missing_rate_max = float(_env("MISSING_RATE_MAX", "0.02"))  # 2%
    progress_every = int(_env("PROGRESS_EVERY_N_DAYS", "25"))

    start = pd.to_datetime(start_s, format="%Y%m%d")
    end = pd.to_datetime(end_s, format="%Y%m%d")

    start_str = pd.Timestamp(start).strftime("%Y%m%d")
    end_str = pd.Timestamp(end).strftime("%Y%m%d")

    print(f"[backfill_chunk] range={start_str}..{end_str} out={chunk_out} missing_rate_max={missing_rate_max}")

    # ---- USD/KRW level mandatory ----
    usd_path = Path(cfg.USDKRW_LEVEL_PATH)
    if not usd_path.exists():
        raise RuntimeError(f"Missing {usd_path}. Backfill workflow must run usdkrw_fetch.py first.")
    usdkrw = pd.read_parquet(usd_path)
    usdkrw = safe_to_datetime(usdkrw, "date")
    if "usdkrw" not in usdkrw.columns:
        raise RuntimeError("usdkrw_level.parquet missing 'usdkrw'")
    usdkrw["usdkrw"] = pd.to_numeric(usdkrw["usdkrw"], errors="coerce")
    usdkrw = usdkrw.dropna(subset=["date", "usdkrw"]).sort_values("date").reset_index(drop=True)

    # ---- VKOSPI level mandatory ----
    vko_path = Path(cfg.VKOSPI_LEVEL_PATH)
    if not vko_path.exists():
        raise RuntimeError(f"Missing {vko_path}. Backfill workflow must run vkospi_fetch.py first.")
    vkospi = pd.read_parquet(vko_path)
    vkospi = safe_to_datetime(vkospi, "date")
    if "vkospi" not in vkospi.columns:
        raise RuntimeError("vkospi_level.parquet missing 'vkospi'")
    vkospi["vkospi"] = pd.to_numeric(vkospi["vkospi"], errors="coerce")
    vkospi = vkospi.dropna(subset=["date", "vkospi"]).sort_values("date").reset_index(drop=True)

    # ---- KOSPI200 close (daily-skip mode) ----
    api = KRXKospiIndexAPI.from_env()
    k200, miss_k200, req_days = fetch_k200_close_range_chunked(api, start, end, progress_every)

    # 누락률 검사(2% 초과 시 실패). 이번 에러의 원인인 'empty로 전체 실패'를 방지 [Source](https://www.genspark.ai/api/files/s/jxj4XzKF)
    _finalize_missing("k200", miss_k200, req_days, missing_rate_max)

    if k200.empty:
        raise RuntimeError("K200 close series is empty for the whole chunk. Check KRX_KOSPI_DD_TRD_URL / KRX_AUTH_KEY / API approval.")

    # ---- Factors ----
    f01 = factor1_momentum(k200)
    f02 = factor2_strength(start_str, end_str)
    f03 = factor3_breadth(start_str, end_str)

    # keep realized vol as alt raw
    f06_alt = factor6_volatility(k200)
    if f06_alt is not None and not f06_alt.empty and "f06_raw" in f06_alt.columns:
        f06_alt = f06_alt.rename(columns={"f06_raw": "f06_alt_raw"})

    f07 = factor7_safe_haven(k200, usdkrw)
    f08 = factor8_foreign_netbuy(start_str, end_str)

    for df in [f01, f02, f03, f06_alt, f07, f08]:
        if df is None:
            continue
        safe_to_datetime(df, "date")

    # ---- Put/Call full window ----
    f04 = fetch_putcall_ratio_by_date(pd.to_datetime(start), pd.to_datetime(end))
    f04 = safe_to_datetime(f04, "date")

    # ---- VKOSPI raw (level) ----
    f06 = vkospi[["date", "vkospi"]].copy().rename(columns={"vkospi": "f06_raw"})

    # ---- Existing f05/f10 from existing index (optional) ----
    index_path = Path(cfg.DATA_DIR) / "index_daily.parquet"
    old = pd.read_parquet(index_path) if index_path.exists() else pd.DataFrame()
    old = safe_to_datetime(old, "date")
    f05f10 = load_existing_f05_f10(old)

    base = k200[["date", "k200_close"]].copy()
    for add in [f01, f02, f03, f04, f06, f06_alt, f07, f08, f05f10]:
        if add is None or add.empty:
            continue
        if "k200_close" in add.columns and "k200_close" in base.columns:
            add = add.drop(columns=["k200_close"])
        base = base.merge(add, on="date", how="left")
    base = base.sort_values("date").reset_index(drop=True)

    # ---- Derived K200 ----
    base["k200_ret_3d"] = base["k200_close"].pct_change(3)
    base["k200_ret_5d"] = base["k200_close"].pct_change(5)
    base["k200_ret_7d"] = base["k200_close"].pct_change(7)
    base["k200_fwd_10d_return"] = forward_return(base["k200_close"], 10)
    base["k200_fwd_10d_win"] = forward_win(base["k200_close"], 10)

    # ---- Scores with flips ----
    flip_scores = {"f04_score", "f05_score", "f06_score", "f07_score", "f10_score"}

    for raw, score in [
        ("f01_raw", "f01_score"),
        ("f02_raw", "f02_score"),
        ("f03_raw", "f03_score"),
        ("f04_raw", "f04_score"),
        ("f05_raw", "f05_score"),
        ("f06_raw", "f06_score"),
        ("f07_raw", "f07_score"),
        ("f08_raw", "f08_score"),
        ("f10_raw", "f10_score"),
    ]:
        if raw in base.columns:
            base[raw] = pd.to_numeric(base[raw], errors="coerce")
            pct = rolling_percentile(base[raw], cfg.ROLLING_DAYS, cfg.MIN_OBS)
            base[score] = 100.0 - pct if score in flip_scores else pct
        else:
            base[score] = np.nan

    w_norm = renormalize_weights(base, cfg.W)
    score_cols = list(cfg.W.keys())
    base["index_score_total"] = (base[score_cols] * w_norm).sum(axis=1)
    base["bucket_5pt"] = (np.floor(base["index_score_total"] / 5.0) * 5.0).clip(0, 100)

    # ---- chunk output only (artifact upload) ----
    out_path = Path(chunk_out)
    ensure_dir(out_path.parent)
    base.to_parquet(out_path, index=False)
    print(f"[backfill_chunk] OK rows={len(base)} -> {out_path}")


if __name__ == "__main__":
    main()
