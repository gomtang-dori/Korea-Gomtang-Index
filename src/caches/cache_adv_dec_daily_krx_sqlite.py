from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

# =========================
# KRX OpenAPI (PROD)
# =========================
BASE = "https://data-dbg.krx.co.kr/svc/apis/sto"
URL_STK = f"{BASE}/stk_bydd_trd"  # 유가증권 일별매매정보
URL_KSQ = f"{BASE}/ksq_bydd_trd"  # 코스닥 일별매매정보


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.environ.get(name, default)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Missing env var: {name}")
    return str(v)


def _to_int_safe(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s == "-":
        return None
    s = s.replace(",", "")
    try:
        return int(float(s))
    except Exception:
        return None


def yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def is_weekend_kst(yyyymmdd_str: str) -> bool:
    dt = datetime.strptime(yyyymmdd_str, "%Y%m%d")
    return dt.weekday() >= 5  # Sat/Sun


def date_range(start_yyyymmdd: str, end_yyyymmdd: str) -> List[str]:
    s = datetime.strptime(start_yyyymmdd, "%Y%m%d")
    e = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    if e < s:
        return []
    out = []
    cur = s
    while cur <= e:
        out.append(yyyymmdd(cur))
        cur += timedelta(days=1)
    return out


# =========================
# SQLite schema
# =========================
DDL = """
CREATE TABLE IF NOT EXISTS adv_dec_daily (
  date TEXT PRIMARY KEY,         -- YYYYMMDD
  adv  INTEGER NOT NULL,
  dec  INTEGER NOT NULL,
  unch INTEGER NOT NULL,
  total INTEGER NOT NULL,
  unknown INTEGER NOT NULL,
  row_count INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_adv_dec_daily_date ON adv_dec_daily(date);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.commit()


def upsert_row(
    conn: sqlite3.Connection,
    date: str,
    adv: int,
    dec: int,
    unch: int,
    unknown: int,
    row_count: int,
) -> None:
    total = adv + dec + unch
    conn.execute(
        """
        INSERT INTO adv_dec_daily(date, adv, dec, unch, total, unknown, row_count, created_at, updated_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        ON CONFLICT(date) DO UPDATE SET
          adv=excluded.adv,
          dec=excluded.dec,
          unch=excluded.unch,
          total=excluded.total,
          unknown=excluded.unknown,
          row_count=excluded.row_count,
          updated_at=datetime('now')
        """,
        (date, adv, dec, unch, total, unknown, row_count),
    )
    conn.commit()


def get_last_saved_date(conn: sqlite3.Connection) -> Optional[str]:
    cur = conn.execute("SELECT MAX(date) FROM adv_dec_daily")
    return cur.fetchone()[0]


# =========================
# OpenAPI fetch + summarize
# =========================
def fetch_outblock_1(url: str, basDd: str, api_key: str, timeout: int = 90) -> List[Dict[str, Any]]:
    headers = {
        # 캡쳐 근거: Request 헤더에 AUTH_KEY로 전달 [Source](https://www.genspark.ai/api/files/s/jYZ3Wt11)
        "AUTH_KEY": api_key,
        "Accept": "application/json",
    }
    params = {"basDd": basDd}

    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code != 200:
        head = r.text[:300].replace("\n", "\\n")
        raise RuntimeError(f"HTTP {r.status_code} {r.reason} url={r.url} body_head={head}")

    data = r.json()
    ob = data.get("OutBlock_1")
    if not isinstance(ob, list):
        raise RuntimeError(f"Unexpected response: OutBlock_1 not list. keys={list(data.keys())}")
    return ob


def summarize_rows(rows: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    adv = dec = unch = unknown = 0
    for row in rows:
        v = _to_int_safe(row.get("CMPPREVDD_PRC"))
        if v is None:
            unknown += 1
        elif v > 0:
            adv += 1
        elif v < 0:
            dec += 1<span class="cursor">█</span>
