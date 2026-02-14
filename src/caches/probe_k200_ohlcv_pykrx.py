# src/caches/probe_k200_ohlcv_pykrx.py
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


def _as_yyyymmdd(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d")


def _safe_patch_index_name():
    """
    pykrx 내부에서 지수명 매핑 테이블을 못 읽으면 KeyError('지수명')가 날 수 있어
    get_index_ticker_name()만 안전하게 우회.
    """
    try:
        import pykrx.stock.stock_api as stock_api
        _orig = stock_api.get_index_ticker_name

        def _safe_get_index_ticker_name(ticker: str) -> str:
            try:
                return _orig(ticker)
            except Exception:
                return f"INDEX_{ticker}"

        stock_api.get_index_ticker_name = _safe_get_index_ticker_name
        print("[probe_k200_ohlcv_pykrx] patched get_index_ticker_name()")
    except Exception as e:
        print(f"[probe_k200_ohlcv_pykrx] WARN patch failed: {type(e).__name__}: {e}")


def main():
    # 설정
    index_code = _env("K200_INDEX_CODE", "1028")  # KOSPI200
    days = int(_env("DAYS", "7"))

    out_path = Path(os.environ.get("OUT_PATH", "data/cache/probe_k200_ohlcv_pykrx.parquet"))

    # 날짜 범위: 안전하게 "어제"를 end로 사용 (당일/휴일 미반영 방지)
    end = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days * 2)  # 거래일 누락(주말) 감안해 여유 있게 잡음

    start_s = _as_yyyymmdd(start)
    end_s = _as_yyyymmdd(end)

    print(f"[probe_k200_ohlcv_pykrx] fetch index_code={index_code} range={start_s}~{end_s} (target last {days} trading days)")

    _safe_patch_index_name()

    # pykrx 호출 (기간 조회)
    df = stock.get_index_ohlcv(start_s, end_s, index_code)
    if df is None or len(df) == 0:
        raise RuntimeError(f"[probe_k200_ohlcv_pykrx] empty result index_code={index_code} range={start_s}~{end_s}")

    # 최근 trading days만 자르기
    df = df.tail(days).copy()

    # 보기 좋게 date 컬럼 만들기
    df = df.reset_index().rename(columns={df.index.name or "index": "date"})
    # 첫 컬럼이 날짜가 아닐 수도 있어 방어
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})

    # 타입 정리
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 출력: 컬럼/헤드/테일
    print(f"[probe_k200_ohlcv_pykrx] cols={list(df.columns)}")
    print("[probe_k200_ohlcv_pykrx] head:")
    print(df.head(3).to_string(index=False))
    print("[probe_k200_ohlcv_pykrx] tail:")
    print(df.tail(3).to_string(index=False))

    # 저장 (parquet + csv 둘 다 저장하면 확인이 편함)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"[probe_k200_ohlcv_pykrx] OK rows={len(df)} -> {out_path}")
    print(f"[probe_k200_ohlcv_pykrx] OK rows={len(df)} -> {csv_path}")


if __name__ == "__main__":
    main()
