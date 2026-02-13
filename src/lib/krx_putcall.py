# src/lib/krx_putcall.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import requests


@dataclass
class KRXPutCallConfig:
    auth_key_env: str = "KRX_AUTH_KEY"
    eqsop_url_env: str = "KRX_EQSOP_URL"  # KOSPI stock options
    eqkop_url_env: str = "KRX_EQKOP_URL"  # KOSDAQ stock options
    timeout: int = 30
    sleep_sec: float = 0.15  # gentle throttling


def _must_env(name: str) -> str:
    v = (os.environ.get(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


def _to_number_series(s: pd.Series) -> pd.Series:
    # values may be "0", "-", "", None, "12,345" etc.
    s = s.astype(str).str.replace(",", "", regex=False).replace({"-": None, "": None, "None": None})
    return pd.to_numeric(s, errors="coerce")


def _fetch_outblock(url: str, auth_key: str, basDd: str, timeout: int) -> List[Dict[str, Any]]:
    headers = {"AUTH_KEY": auth_key, "Accept": "application/json"}
    params = {"basDd": basDd}
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code != 200:
        # KRX often returns JSON with respMsg/respCode on 401
        try:
            js = r.json()
        except Exception:
            js = {"text": r.text[:500]}
        raise RuntimeError(f"KRX call failed status={r.status_code} body={js}")

    js = r.json()
    if isinstance(js, dict) and "OutBlock_1" in js and isinstance(js["OutBlock_1"], list):
        return js["OutBlock_1"]
    # fallback: sometimes nested
    for k, v in (js.items() if isinstance(js, dict) else []):
        if isinstance(v, dict) and "OutBlock_1" in v and isinstance(v["OutBlock_1"], list):
            return v["OutBlock_1"]
    raise RuntimeError(f"Unexpected KRX response structure. top_keys={list(js.keys()) if isinstance(js, dict) else type(js)}")


def _daily_put_call_from_rows(rows: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    returns (put_trdval_sum, call_trdval_sum)
    Uses ACC_TRDVAL (trading value) and RGHT_TP_NM (CALL/PUT).
    """
    if not rows:
        return (float("nan"), float("nan"))

    df = pd.DataFrame(rows)
    if df.empty:
        return (float("nan"), float("nan"))

    # Required keys confirmed by probe/spec:
    # RGHT_TP_NM, ACC_TRDVAL
    if "RGHT_TP_NM" not in df.columns or "ACC_TRDVAL" not in df.columns:
        return (float("nan"), float("nan"))

    df["RGHT_TP_NM"] = df["RGHT_TP_NM"].astype(str).str.upper().str.strip()
    df["ACC_TRDVAL_NUM"] = _to_number_series(df["ACC_TRDVAL"]).fillna(0.0)

    put_sum = float(df.loc[df["RGHT_TP_NM"] == "PUT", "ACC_TRDVAL_NUM"].sum())
    call_sum = float(df.loc[df["RGHT_TP_NM"] == "CALL", "ACC_TRDVAL_NUM"].sum())
    return put_sum, call_sum


def fetch_putcall_ratio_by_date(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cfg: Optional[KRXPutCallConfig] = None,
) -> pd.DataFrame:
    """
    Fetch put/call ratio timeseries (daily) for KOSPI+KOSDAQ stock options.
    - Uses KRX endpoints from env:
      KRX_EQSOP_URL, KRX_EQKOP_URL
    - Auth header key from env: KRX_AUTH_KEY (AUTH_KEY header)
    - Query param: basDd=YYYYMMDD
    Output:
      columns: date, f04_put_trdval, f04_call_trdval, f04_raw
    """
    cfg = cfg or KRXPutCallConfig()

    auth_key = _must_env(cfg.auth_key_env)
    eqsop_url = _must_env(cfg.eqsop_url_env)
    eqkop_url = _must_env(cfg.eqkop_url_env)

    start_date = pd.to_datetime(start_date).normalize()
    end_date = pd.to_datetime(end_date).normalize()
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")

    days = pd.date_range(start_date, end_date, freq="D")
    out_rows = []
    for d in days:
        basDd = d.strftime("%Y%m%d")

        # KOSPI options
        try:
            rows1 = _fetch_outblock(eqsop_url, auth_key, basDd, cfg.timeout)
            put1, call1 = _daily_put_call_from_rows(rows1)
        except Exception:
            put1, call1 = (float("nan"), float("nan"))

        time.sleep(cfg.sleep_sec)

        # KOSDAQ options
        try:
            rows2 = _fetch_outblock(eqkop_url, auth_key, basDd, cfg.timeout)
            put2, call2 = _daily_put_call_from_rows(rows2)
        except Exception:
            put2, call2 = (float("nan"), float("nan"))

        put = pd.Series([put1, put2], dtype="float").sum(skipna=True)
        call = pd.Series([call1, call2], dtype="float").sum(skipna=True)

        if pd.isna(put) and pd.isna(call):
            f04 = float("nan")
        else:
            # avoid divide-by-zero
            f04 = float(put / call) if (call is not None and call != 0) else float("nan")

        out_rows.append(
            {
                "date": d,
                "f04_put_trdval": put,
                "f04_call_trdval": call,
                "f04_raw": f04,  # Put/Call ratio (value-based)
            }
        )

    df = pd.DataFrame(out_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df
