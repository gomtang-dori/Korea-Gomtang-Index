# src/lib/krx_dvrprod_index.py
from __future__ import annotations

import os
from dataclasses import dataclass
import pandas as pd
import requests


@dataclass
class KRXDrvProdIndexAPI:
    """
    KRX OpenAPI - 파생상품지수 시세정보
    (실제 동작 endpoint는 probe로 확인된 drvprod_dd_trd)
      - URL (env): KRX_DVRPROD_DD_TRD_URL
      - Auth: AUTH_KEY header (env: KRX_AUTH_KEY)
      - Request param: basDd=YYYYMMDD
      - Response list: OutBlock_1
      - Fields: BAS_DD, IDX_NM, CLSPRC_IDX, ... (spec capture)
        [Source] https://www.genspark.ai/api/files/s/uX7923Iq
    """
    url: str
    auth_key: str
    timeout: int = 30

    @classmethod
    def from_env(cls) -> "KRXDrvProdIndexAPI":
        url = (os.getenv("KRX_DVRPROD_DD_TRD_URL") or "").strip()
        if not url:
            # safe default (probe-confirmed working path)
            url = "https://data-dbg.krx.co.kr/svc/apis/idx/drvprod_dd_trd"
        auth = (os.getenv("KRX_AUTH_KEY") or "").strip()
        if not auth:
            raise RuntimeError("Missing env KRX_AUTH_KEY")
        return cls(url=url, auth_key=auth)

    @staticmethod
    def _parse_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")

    def fetch_day(self, basDd: str) -> pd.DataFrame:
        headers = {"AUTH_KEY": self.auth_key, "Accept": "application/json"}
        params = {"basDd": basDd}
        r = requests.get(self.url, headers=headers, params=params, timeout=self.timeout)
        r.raise_for_status()
        js = r.json()
        rows = js.get("OutBlock_1") if isinstance(js, dict) else None
        if not isinstance(rows, list):
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def fetch_vkospi_range(self, start: pd.Timestamp, end: pd.Timestamp, idx_nm: str = "코스피 200 변동성지수") -> pd.DataFrame:
        start = pd.to_datetime(start).normalize()
        end = pd.to_datetime(end).normalize()
        days = pd.date_range(start, end, freq="D")

        out = []
        tgt = str(idx_nm).replace(" ", "")

        for d in days:
            basDd = d.strftime("%Y%m%d")
            df = self.fetch_day(basDd)
            if df.empty or "IDX_NM" not in df.columns:
                continue

            nm = df["IDX_NM"].astype(str).str.replace(" ", "", regex=False)
            picked = df.loc[nm == tgt].copy()
            if picked.empty:
                continue

            # date
            if "BAS_DD" in picked.columns:
                dt = pd.to_datetime(picked["BAS_DD"].iloc[0], errors="coerce")
            else:
                dt = pd.to_datetime(basDd, format="%Y%m%d", errors="coerce")

            # close
            if "CLSPRC_IDX" not in picked.columns:
                continue
            level = self._parse_num(picked["CLSPRC_IDX"]).iloc[0]
            out.append(pd.DataFrame({"date": [dt], "vkospi": [level]}))

        if not out:
            return pd.DataFrame(columns=["date", "vkospi"])

        res = pd.concat(out, ignore_index=True)
        res["date"] = pd.to_datetime(res["date"], errors="coerce")
        res["vkospi"] = pd.to_numeric(res["vkospi"], errors="coerce")
        res = res.dropna(subset=["date", "vkospi"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
        return res
