# src/lib/krx_kospi_index.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from lib.krx_http import KRXHTTP

import pandas as pd
import requests


@dataclass
class KRXKospiIndexAPI:
    """
    KRX OpenAPI - KOSPI 시리즈 일별시세정보
    Endpoint (confirmed by user): https://data-dbg.krx.co.kr/svc/apis/idx/kospi_dd_trd
    Auth: header AUTH_KEY
    Request: basDt=YYYYMMDD (but we also try basDd as fallback)
    Response: OutBlock_1 list

    We extract KOSPI200 close by filtering IDX_NM and picking a close column.
    In real response, close column observed as 'CLSPRC_IDX' (not 'CLS_PRC').
    """
    url: str
    auth_key: str
    timeout: int = 30
    debug: bool = False

    @classmethod
    def from_env(cls) -> "KRXKospiIndexAPI":
        url = (os.getenv("KRX_KOSPI_DD_TRD_URL") or "https://data-dbg.krx.co.kr/svc/apis/idx/kospi_dd_trd").strip()
        auth = (os.getenv("KRX_AUTH_KEY") or "").strip()
        if not auth:
            raise RuntimeError("Missing env KRX_AUTH_KEY")
        debug = (os.getenv("K200_DEBUG") or "0").strip() == "1"
        return cls(url=url, auth_key=auth, debug=debug)

    def _dbg(self, *args):
        if self.debug:
            print(*args)

    def _get_json(self, params: dict) -> dict:
        # 1) http client가 없으면 만든다(안전장치)
        if not hasattr(self, "http") or self.http is None:
            self.http = KRXHTTP(auth_key=self.auth_key, timeout=self.timeout)
    
        # 2) 429/5xx 자동 재시도 + 백오프로 JSON을 받는다
        return self.http.get_json(self.url, params)
    def fetch_k200_close_range_monthly(self, start, end, progress_every: int = 3):
        """
        backfill에서 사용: start~end를 월 단위로 끊어 처리
        반환: (k200_df, missing_days, requested_day_count)
        """
        import pandas as pd
    
        start = pd.to_datetime(start).tz_localize(None).normalize()
        end = pd.to_datetime(end).tz_localize(None).normalize()
    
        months = pd.period_range(start=start, end=end, freq="M")
        frames = []
        missing = []
    
        for i, p in enumerate(months, start=1):
            y, m = int(p.year), int(p.month)
            m_start = pd.Timestamp(y, m, 1)
            m_end = (m_start + pd.offsets.MonthEnd(1)).normalize()
            days = pd.date_range(m_start, m_end, freq="D")
    
            for d in days:
                bas = d.strftime("%Y%m%d")
                try:
                    df_day = self.fetch_k200_close_by_date(bas)
                    if df_day is None or df_day.empty:
                        missing.append(bas)
                    else:
                        frames.append(df_day)
                except Exception:
                    missing.append(bas)
    
            if progress_every and (i % progress_every == 0 or i == len(months)):
                print(f"[k200_monthly] {i}/{len(months)} y={y} m={m:02d} frames={len(frames)} miss={len(missing)}")
    
        if frames:
            out = pd.concat(frames, ignore_index=True)
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out["k200_close"] = pd.to_numeric(out["k200_close"], errors="coerce")
            out = (
                out.dropna(subset=["date", "k200_close"])
                   .drop_duplicates("date", keep="last")
                   .sort_values("date")
                   .reset_index(drop=True)
            )
        else:
            out = pd.DataFrame(columns=["date", "k200_close"])
    
        requested = len(pd.date_range(start, end, freq="D"))
        missing = [d for d in missing if start.strftime("%Y%m%d") <= d <= end.strftime("%Y%m%d")]
        return out, missing, requested



    def _get(self, basDt: str) -> dict:
        """
        Spec says basDt, but we saw empty OutBlock_1 when only basDt was used.
        So: try basDt -> if OutBlock_1 empty -> try basDd (compat fallback).
        """
        js = self._get_json({"basDt": basDt})

        # If empty array, try alternate param name
        if isinstance(js, dict) and js.get("OutBlock_1") == []:
            self._dbg(f"[k200_debug] basDt={basDt} OutBlock_1 empty with basDt -> retry basDd")
            js2 = self._get_json({"basDd": basDt})
            return js2

        return js

    @staticmethod
    def _to_frame(js: dict) -> pd.DataFrame:
        if isinstance(js, dict):
            for k in ["OutBlock_1", "outBlock_1", "OutBlock1", "output", "data"]:
                if k in js and isinstance(js[k], list):
                    return pd.DataFrame(js[k])
        if isinstance(js, list):
            return pd.DataFrame(js)
        return pd.DataFrame()

    @staticmethod
    def _norm_idx_nm(s: pd.Series) -> pd.Series:
        # remove spaces; keep Korean/English as-is
        return s.astype(str).str.replace(" ", "", regex=False)

    @classmethod
    def _pick_k200_row(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        IDX_NM values observed include both '코스피 200' and '코스피200'.
        We'll normalize by removing spaces and match exactly 'KOSPI200' or '코스피200'.
        """
        if df.empty or "IDX_NM" not in df.columns:
            return pd.DataFrame()

        norm = cls._norm_idx_nm(df["IDX_NM"])
        mask = norm.isin(["KOSPI200", "코스피200"])
        return df.loc[mask].copy()

    @staticmethod
    def _parse_num_series(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")

    @staticmethod
    def _pick_close_col(df: pd.DataFrame) -> str | None:
        # Real log showed 'CLSPRC_IDX' exists; spec screenshot showed 'CLS_PRC'.
        candidates = [
            "CLSPRC_IDX",  # confirmed in your Action log
            "CLS_PRC",     # spec name (may differ)
            "CLSPRC",
            "CLOSE_PRC",
            "TDD_CLSPRC",
        ]
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def fetch_k200_close_by_date(self, basDt: str) -> pd.DataFrame:
        """
        Return columns: date, k200_close
        """
        js = self._get(basDt)

        if self.debug:
            try:
                if isinstance(js, dict):
                    self._dbg(f"[k200_debug] basDt={basDt} top_keys={list(js.keys())}")
                self._dbg("[k200_debug] js_head:", json.dumps(js, ensure_ascii=False)[:600])
            except Exception as e:
                self._dbg("[k200_debug] js_dump_failed:", e)

        df = self._to_frame(js)

        self._dbg(f"[k200_debug] basDt={basDt} out_rows={len(df)}")
        self._dbg(f"[k200_debug] basDt={basDt} cols={list(df.columns)}")

        if df.empty:
            return pd.DataFrame(columns=["date", "k200_close"])

        # Print IDX_NM head for diagnostics
        if self.debug and "IDX_NM" in df.columns:
            self._dbg(f"[k200_debug] basDt={basDt} IDX_NM_head30={df['IDX_NM'].astype(str).head(30).tolist()}")

        picked = self._pick_k200_row(df)
        if picked.empty:
            self._dbg(f"[k200_debug] basDt={basDt} pick_row=EMPTY (filter miss)")
            return pd.DataFrame(columns=["date", "k200_close"])

        # Pick date
        if "BAS_DD" in picked.columns:
            dt = pd.to_datetime(picked["BAS_DD"].iloc[0], errors="coerce")
        else:
            dt = pd.to_datetime(basDt, format="%Y%m%d", errors="coerce")

        close_col = self._pick_close_col(picked)
        if close_col is None:
            self._dbg(f"[k200_debug] basDt={basDt} close_col missing; available_cols={list(picked.columns)}")
            return pd.DataFrame(columns=["date", "k200_close"])

        close = self._parse_num_series(picked[close_col]).iloc[0]
        self._dbg(f"[k200_debug] basDt={basDt} PICKED IDX_NM={picked.get('IDX_NM').iloc[0] if 'IDX_NM' in picked.columns else 'NA'} "
                  f"using_close_col={close_col} close={close}")

        return pd.DataFrame({"date": [dt], "k200_close": [close]})

    def fetch_k200_close_range(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch per-day. Robust for stability; holidays return empty and are skipped.
        """
        start = pd.to_datetime(start).normalize()
        end = pd.to_datetime(end).normalize()
        days = pd.date_range(start, end, freq="D")

        rows = []
        for d in days:
            basDt = d.strftime("%Y%m%d")
            try:
                one = self.fetch_k200_close_by_date(basDt)
                if not one.empty and pd.notna(one["k200_close"].iloc[0]):
                    rows.append(one)
            except Exception as e:
                self._dbg(f"[k200_debug] basDt={basDt} exception={e}")
                continue

        if not rows:
            return pd.DataFrame(columns=["date", "k200_close"])

        out = pd.concat(rows, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["k200_close"] = pd.to_numeric(out["k200_close"], errors="coerce")
        out = out.dropna(subset=["date", "k200_close"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
        return out
