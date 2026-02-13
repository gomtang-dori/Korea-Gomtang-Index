from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Iterable

import pandas as pd
import requests
import json


@dataclass
class KRXKospiIndexAPI:
    """
    KRX OpenAPI - KOSPI 시리즈 일별시세정보
    Spec (user capture):
      - Server endpoint: (https)://data-dbg.krx.co.kr/svc/apis/idx/kospi_dd_trd
      - Request: basDt=YYYYMMDD
      - Response OutBlock_1, close field: CLS_PRC
      - We'll filter IDX_NM == 'KOSPI 200' or '코스피 200'
    """
    url: str
    auth_key: str
    timeout: int = 30

    @classmethod
    def from_env(cls) -> "KRXKospiIndexAPI":
        url = (os.getenv("KRX_KOSPI_DD_TRD_URL") or "https://data-dbg.krx.co.kr/svc/apis/idx/kospi_dd_trd").strip()
        auth = (os.getenv("KRX_AUTH_KEY") or "").strip()
        if not auth:
            raise RuntimeError("Missing env KRX_AUTH_KEY")
        return cls(url=url, auth_key=auth)

    def _get(self, basDt: str) -> dict:
        headers = {"AUTH_KEY": self.auth_key, "Accept": "application/json"}
        params = {"basDt": basDt}
        r = requests.get(self.url, headers=headers, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _to_frame(js: dict) -> pd.DataFrame:
        # Most KRX OpenAPI returns dict with OutBlock_1 list.
        if isinstance(js, dict):
            for k in ["OutBlock_1", "outBlock_1", "OutBlock1", "output", "data"]:
                if k in js and isinstance(js[k], list):
                    return pd.DataFrame(js[k])
        # fallback: if js itself is list
        if isinstance(js, list):
            return pd.DataFrame(js)
        return pd.DataFrame()

    @staticmethod
    def _pick_k200_row(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # Normalize column names
        cols = set(df.columns)
        if "IDX_NM" not in cols:
            return pd.DataFrame()

        mask = df["IDX_NM"].astype(str).isin(["KOSPI 200", "코스피 200"])
        out = df.loc[mask].copy()
        return out

    @staticmethod
    def _parse_close(s: pd.Series) -> pd.Series:
        # CLS_PRC may contain commas or '-' etc.
        return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")

    # --- 여기서부터 들여쓰기가 수정된 부분입니다 ---

    def fetch_k200_close_by_date(self, basDt: str) -> pd.DataFrame:
        """
        Returns: DataFrame columns: date, k200_close
        date is YYYY-MM-DD (datetime)
        """
        js = self._get(basDt)

        # ---- DEBUG (start) ----
        try:
            if isinstance(js, dict):
                print(f"[k200_debug] basDt={basDt} top_keys={list(js.keys())}")
            else:
                print(f"[k200_debug] basDt={basDt} js_type={type(js)}")
            print("[k200_debug] js_head:", json.dumps(js, ensure_ascii=False)[:800])
        except Exception as e:
            print("[k200_debug] js_dump_failed:", e)
        # ---- DEBUG (end) ----

        df = self._to_frame(js)

        # ---- DEBUG (start) ----
        print(f"[k200_debug] basDt={basDt} out_rows={len(df)}")
        print(f"[k200_debug] basDt={basDt} cols={list(df.columns)}")
        if not df.empty and "IDX_NM" in df.columns:
            idx_list = df["IDX_NM"].astype(str).head(30).tolist()
            print(f"[k200_debug] basDt={basDt} IDX_NM_head30={idx_list}")

            # KOSPI/200 관련 후보 탐색
            norm = df["IDX_NM"].astype(str).str.replace(" ", "", regex=False).str.upper()
            cand_mask = norm.str.contains("KOSPI", na=False) | norm.str.contains("코스피".upper(), na=False) | norm.str.contains("200", na=False)
            cand = df.loc[cand_mask, ["IDX_NM"] + ([c for c in ["BAS_DD", "CLS_PRC"] if c in df.columns])].head(20)
            print("[k200_debug] candidates_head20:")
            print(cand.to_string(index=False))
        # ---- DEBUG (end) ----

        df = self._pick_k200_row(df)
        if df.empty:
            print(f"[k200_debug] basDt={basDt} pick_row=EMPTY (filter miss)")
            return pd.DataFrame(columns=["date", "k200_close"])

        # 날짜 처리
        if "BAS_DD" in df.columns:
            dt = pd.to_datetime(df["BAS_DD"].iloc[0], errors="coerce")
        else:
            dt = pd.to_datetime(basDt, format="%Y%m%d", errors="coerce")

        if "CLS_PRC" not in df.columns:
            print(f"[k200_debug] basDt={basDt} CLS_PRC missing")
            return pd.DataFrame(columns=["date", "k200_close"])

        close = self._parse_close(df["CLS_PRC"]).iloc[0]
        print(f"[k200_debug] basDt={basDt} PICKED IDX_NM={df.get('IDX_NM').iloc[0] if 'IDX_NM' in df.columns else 'NA'} CLS_PRC={close}")
        return pd.DataFrame({"date": [dt], "k200_close": [close]})

    def fetch_k200_close_range(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch by calling the daily endpoint per date.
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
            except Exception:
                continue

        if not rows:
            return pd.DataFrame(columns=["date", "k200_close"])

        out = pd.concat(rows, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["k200_close"] = pd.to_numeric(out["k200_close"], errors="coerce")
        out = out.dropna(subset=["date", "k200_close"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
        return out
