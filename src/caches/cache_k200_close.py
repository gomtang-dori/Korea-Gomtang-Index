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


def main():
    out_path = Path(os.environ.get("K200_CACHE_PATH", "data/cache/k200_close.parquet"))
    index_code = os.environ.get("K200_INDEX_CODE", "1028")  # KOSPI200

    backfill_start = pd.to_datetime(_env("BACKFILL_START"), format="%Y%m%d", errors="raise")
    backfill_end = pd.to_datetime(_env("BACKFILL_END"), format="%Y%m%d", errors="raise")
    buffer_days = int(os.environ.get("CACHE_BUFFER_DAYS", "450"))

    start = (backfill_start - pd.Timedelta(days=buffer_days)).normalize()
    end = backfill_end.normalize()

    start_s = _as_yyyymmdd(start)
    end_s = _as_yyyymmdd(end)

    print(f"[cache_k200_close:pykrx] fetch index_code={index_code} range={start_s}~{end_s}")

    # --- 핵심: pykrx 내부 '지수명' 매핑(KeyError) 우회 ---
    # pykrx가 IndexTicker().get_name() 호출에 실패하면 KeyError('지수명')로 죽는 케이스가 있음.
    # 우리는 지수명은 필요 없으므로, get_index_ticker_name을 안전한 fallback으로 바꾼다.
    try:
        import pykrx.stock.stock_api as stock_api
        _orig = stock_api.get_index_ticker_name

        def _safe_get_index_ticker_name(ticker: str) -> str:
            try:
                return _orig(ticker)
            except Exception as e:
                # 이름 못 가져와도 데이터는 충분함. columns.name만 대체 문자열로.
                return f"INDEX_{ticker}"

        stock_api.get_index_ticker_name = _safe_get_index_ticker_name
        print("[cache_k200_close:pykrx] patched get_index_ticker_name() to avoid KeyError('지수명')")
    except Exception as e:
        # 패치 실패해도 일단 시도는 해보되, 여기서 실패하면 원인 로그 확인
        print(f"[cache_k200_close:pykrx] WARN: patch failed: {type(e).__name__}: {e}")

    # 기간 한 번에 조회 (월/분기보다 더 강력: 전체기간 1~수회 호출)
    df = stock.get_index_ohlcv(start_s, end_s, index_code)
    if df is None or len(df) == 0:
        raise RuntimeError(f"[cache_k200_close:pykrx] empty result index_code={index_code} range={start_s}~{end_s}")

    # pykrx index ohlcv는 인덱스가 날짜 형태로 들어오므로 reset_index
    df = df.reset_index()

    # 첫 컬럼이 날짜
    date_col = df.columns[0]

    # 종가 컬럼 탐색 (pykrx 문서 예시 기준으로 한글/영문 혼재 가능)
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
