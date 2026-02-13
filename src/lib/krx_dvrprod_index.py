# src/lib/krx_dvrprod_index.py
from __future__ import annotations

import os
from dataclasses import dataclass
import pandas as pd
import requests


@dataclass
class KRXDvrProdIndexAPI:
    url: str
    auth_key: str
    timeout: int = 30

    @classmethod
    def from_env(cls) -> "KRXDvrProdIndexAPI":
        url = (os.getenv("KRX_DVRPROD_DD_TRD_URL") or "").strip()
        auth = (os.getenv("KRX_AUTH_KEY") or "").strip()
        if not url:
            # safe default from spec
            url = "https://data-dbg.krx.co.kr/svc/apis/idx/dvrprod_dd_trd"
        if not auth:
            raise RuntimeError("Missing env KRX_AUTH_KEY")
        return cls(url=url, auth_key=auth)

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

    @staticmethod
    def _parse_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")

    def fetch_index_level_range(self, start: pd.Timestamp, end: pd.Timestamp, idx_nm: str) -> pd.DataFrame:
        start = pd.to_datetime(start).normalize()
        end = pd.to_datetime(end).normalize()
        days = pd.date_range(start, end, freq="D")

        out = []
        for d in days:
            basDd = d.strftime("%Y%m%d")
            df = self.fetch_day(basDd)
            if df.empty:
                continue
            if "IDX_NM" not in df.columns:
                continue

            # normalize spaces for safe match
            nm = df["IDX_NM"].astype(str).str.replace(" ", "", regex=False)
            tgt = str(idx_nm).replace(" ", "")
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
            close = self._parse_num(picked["CLSPRC_IDX"]).iloc[0]
            out.append(pd.DataFrame({"date": [dt], "vkospi": [close]}))

        if not out:
            return pd.DataFrame(columns=["date", "vkospi"])

        res = pd.concat(out, ignore_index=True)
        res["date"] = pd.to_datetime(res["date"], errors="coerce")
        res["vkospi"] = pd.to_numeric(res["vkospi"], errors="coerce")
        res = res.dropna(subset=["date", "vkospi"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
        return res
