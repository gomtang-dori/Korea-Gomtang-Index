# src/caches/cache_k200_close.py
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from pykrx import stock


def _env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        if default is None:
            raise RuntimeError(f"Missing env: {name}")
        return default
    return v


def _as_yyyymmdd(d: pd.Timestamp) -> str:
    return d.strftime("%Y%m%d")


def _to_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """tz-aware/naive 혼재 방지: 항상 tz-naive(UTC 기준)로 통일"""
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def main():
    out_path = Path(os.environ.get("K200_CACHE_PATH", "data/cache/k200_close.parquet"))
    index_code = os.environ.get("K200_INDEX_CODE", "1028")  # KOSPI200

    # 입력 날짜 파싱 (대부분 tz-naive로 들어오지만, 방어적으로 통일)
    backfill_start = pd.to_datetime(_env("BACKFILL_START"), format="%Y%m%d", errors="raise")
    backfill_end = pd.to_datetime(_env("BACKFILL_END"), format="%Y%m%d", errors="raise")
    backfill_start = _to_naive_utc(backfill_start).normalize()
    backfill_end = _to_naive_utc(backfill_end).normalize()

    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))

    # "오늘"은 장중/주말/휴일 문제를 줄이기 위해 기본은 "어제(UTC)"로 둠
    today_utc = pd.Timestamp.utcnow().tz_localize(None).normalize()
    safe_end = today_utc - pd.Timedelta(days=1)

    # 미래 end 방지 (여기서 backfill_end를 먼저 클램프하고, 이후 end를 계산해야 함)
    if backfill_end > safe_end:
        print(
            f"[cache_k200_close:pykrx] WARN: BACKFILL_END {backfill_end:%Y%m%d} > safe_end {safe_end:%Y%m%d}. clamp to safe_end."
        )
        backfill_end = safe_end

    # 이제 start/end 계산 (클램프 반영됨)
    start = (backfill_start - pd.Timedelta(days=buffer_days)).normalize()
    end = backfill_end.normalize()

    start_s = _as_yyyymmdd(start)
    end_s = _as_yyyymmdd(end)

    print(f"[cache_k200_close:pykrx] fetch index_code={index_code} range={start_s}~{end_s}")

    # --- 핵심: pykrx 내부 '지수명' 매핑(KeyError) 우회 ---
    # pykrx가 IndexTicker().get_name() 호출에 실패하면 KeyError('지수명')로 죽는 케이스가 있음.
    # 우리는 지수명은 필요 없으므로, get_index_ticker_name을 안전한 fallback으로 바꾼다.
    # 관련 이슈 보고: [Source](https://github.com/sharebook-kr/pykrx/issues/229)
    try:
        import pykrx.stock.stock_api as stock_api

        _orig = stock_api.get_index_ticker_name

        def _safe_get_index_ticker_name(ticker: str) -> str:
            try:
                return _orig(ticker)
            except Exception:
                return f"INDEX_{ticker}"

        stock_api.get_index_ticker_name = _safe_get_index_ticker_name
        print("[cache_k200_close:pykrx] patched get_index_ticker_name() to avoid KeyError('지수명')")
    except Exception as e:
        print(f"[cache_k200_close:pykrx] WARN: patch failed: {type(e).__name__}: {e}")

    # 기간 한 번에 조회
    df = stock.get_index_ohlcv(start_s, end_s, index_code)

    # empty면 날짜를 하루씩 뒤로 당겨 3회 재시도 (휴일/당일 미반영/일시적 문제 방어)
    if df is None or len(df) == 0:
        print(f"[cache_k200_close:pykrx] WARN: empty result on {start_s}~{end_s}, retry by shifting end back")
        for k in (1, 2, 3):
            end_retry = (end - pd.Timedelta(days=k)).normalize()
            end_retry_s = _as_yyyymmdd(end_retry)
            print(f"[cache_k200_close:pykrx] retry{k}: range={start_s}~{end_retry_s}")
            df = stock.get_index_ohlcv(start_s, end_retry_s, index_code)
            if df is not None and len(df) > 0:
                end_s = end_retry_s
                end = end_retry
                break

    if df is None or len(df) == 0:
        raise RuntimeError(f"[cache_k200_close:pykrx] empty result index_code={index_code} range={start_s}~{end_s}")

    df = df.reset_index()
    date_col = df.columns[0]

    close_col = None
    for c in ["종가", "Close", "종가 "]:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        raise RuntimeError(f"[cache_k200_close:pykrx] cannot find close column in {list(df.columns)}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "k200_close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    ).dropna(subset=["date", "k200_close"]).sort_values("date").reset_index(drop=True)

    # 기존 파일과 upsert
    if out_path.exists():
        old = pd.read_parquet(out_path)
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
        old["k200_close"] = pd.to_numeric(old["k200_close"], errors="coerce")
        old = old.dropna(subset=["date", "k200_close"])
        merged = pd.concat([old, out], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    else:
        merged = out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"[cache_k200_close:pykrx] OK rows={len(merged)} -> {out_path}")


if __name__ == "__main__":
    main()
