from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

# KRX OpenAPI (PROD)
BASE = "https://data-dbg.krx.co.kr/svc/apis/sto"
URL_STK = f"{BASE}/stk_bydd_trd"  # 유가증권
URL_KSQ = f"{BASE}/ksq_bydd_trd"  # 코스닥


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
    out: List[str] = []
    cur = s
    while cur <= e:
        out.append(yyyymmdd(cur))
        cur += timedelta(days=1)
    return out


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


def fetch_outblock_1(url: str, basDd: str, api_key: str, timeout: int = 90) -> List[Dict[str, Any]]:
    headers = {
        "AUTH_KEY": api_key,  # 인증 방식: AUTH_KEY 헤더 [Source](https://www.genspark.ai/api/files/s/jYZ3Wt11)
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
            dec += 1
        else:
            unch += 1
    return adv, dec, unch, unknown


@dataclass
class Config:
    db_path: str
    mode: str
    start: Optional[str]
    end: Optional[str]
    sleep_sec: float
    retry: int
    timeout_sec: int


def load_config() -> Config:
    return Config(
        db_path=os.environ.get("ADVDEC_DB_PATH", "data/cache/adv_dec_daily.sqlite"),
        mode=os.environ.get("ADVDEC_MODE", "backfill"),
        start=os.environ.get("ADVDEC_START"),
        end=os.environ.get("ADVDEC_END"),
        sleep_sec=float(os.environ.get("ADVDEC_SLEEP_SEC", "0.5")),
        retry=int(os.environ.get("ADVDEC_RETRY", "4")),
        timeout_sec=int(os.environ.get("ADVDEC_TIMEOUT_SEC", "90")),
    )


def main() -> None:
    cfg = load_config()
    api_key = _env("KRX_API_KEY")

    os.makedirs(os.path.dirname(cfg.db_path), exist_ok=True)

    conn = sqlite3.connect(cfg.db_path)
    try:
        ensure_schema(conn)

        if cfg.mode not in ("backfill", "incremental"):
            raise RuntimeError("ADVDEC_MODE must be backfill or incremental")

        if cfg.mode == "incremental":
            last = get_last_saved_date(conn)
            if last is None:
                raise RuntimeError("No data in DB. Run backfill first.")
            start_dt = datetime.strptime(last, "%Y%m%d") + timedelta(days=1)
            end_dt = datetime.utcnow().date() - timedelta(days=1)
            start = yyyymmdd(start_dt)
            end = end_dt.strftime("%Y%m%d")
        else:
            if not cfg.start or not cfg.end:
                raise RuntimeError("Backfill requires ADVDEC_START and ADVDEC_END")
            start, end = cfg.start, cfg.end

        dates = date_range(start, end)
        print(f"[advdec] db={cfg.db_path} mode={cfg.mode} range={start}~{end} days={len(dates)}")
        print(f"[advdec] endpoints: {URL_STK} , {URL_KSQ}")

        ok = weekend_skip = empty_skip = failed = 0

        for d in dates:
            if is_weekend_kst(d):
                weekend_skip += 1
                continue

            for attempt in range(1, cfg.retry + 1):
                try:
                    stk = fetch_outblock_1(URL_STK, d, api_key, timeout=cfg.timeout_sec)
                    ksq = fetch_outblock_1(URL_KSQ, d, api_key, timeout=cfg.timeout_sec)

                    if len(stk) == 0 and len(ksq) == 0:
                        empty_skip += 1
                        break

                    a1, de1, u1, un1 = summarize_rows(stk)
                    a2, de2, u2, un2 = summarize_rows(ksq)

                    adv = a1 + a2
                    dec = de1 + de2
                    unch = u1 + u2
                    unknown = un1 + un2
                    row_count = len(stk) + len(ksq)

                    upsert_row(conn, d, adv, dec, unch, unknown, row_count)
                    ok += 1

                    if ok % 20 == 0:
                        print(f"[advdec] ok={ok} last={d} row_count={row_count} unknown={unknown}")
                    break

                except Exception as e:
                    if attempt < cfg.retry:
                        time.sleep(1.0 * attempt)
                        continue
                    failed += 1
                    print(f"[WARN][advdec] {d} failed after {cfg.retry} retries: {e}")

            time.sleep(cfg.sleep_sec)

        print(f"[advdec] DONE ok={ok} weekend_skip={weekend_skip} empty_skip={empty_skip} failed={failed}")
        print(f"[advdec] sqlite={cfg.db_path}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
