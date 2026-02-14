# src/caches/cache_adv_dec_daily_krx_sqlite.py
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

KRX_JSON_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
DEFAULT_REFERER = "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020102"
DEFAULT_BLD = "dbms/MDC/STAT/standard/MDCSTAT01602"


def _env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        if default is None:
            raise RuntimeError(f"Missing env: {name}")
        return default
    return v


def _yyyymmdd(d: pd.Timestamp) -> str:
    return d.strftime("%Y%m%d")


def _to_int_safe(x: Any) -> int | None:
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


@dataclass
class AdvDecRow:
    date: str  # YYYYMMDD
    adv: int
    dec: int
    unch: int
    total: int
    unknown: int
    row_count: int


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS adv_dec_daily (
          date       TEXT PRIMARY KEY,
          adv        INTEGER NOT NULL,
          dec        INTEGER NOT NULL,
          unch       INTEGER NOT NULL,
          total      INTEGER NOT NULL,
          unknown    INTEGER NOT NULL,
          row_count  INTEGER NOT NULL,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_adv_dec_daily_date ON adv_dec_daily(date);")
    conn.commit()


def upsert_row(conn: sqlite3.Connection, r: AdvDecRow) -> None:
    conn.execute(
        """
        INSERT INTO adv_dec_daily (date, adv, dec, unch, total, unknown, row_count, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        ON CONFLICT(date) DO UPDATE SET
          adv=excluded.adv,
          dec=excluded.dec,
          unch=excluded.unch,
          total=excluded.total,
          unknown=excluded.unknown,
          row_count=excluded.row_count,
          updated_at=datetime('now');
        """,
        (r.date, r.adv, r.dec, r.unch, r.total, r.unknown, r.row_count),
    )
    conn.commit()


def fetch_outblock_1(date_yyyymmdd: str, bld: str, referer: str, timeout: int = 30) -> List[Dict[str, Any]]:
    headers = {
        "Referer": referer,
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
    }
    payload = {
        "bld": bld,
        "locale": "ko_KR",
        "strtDd": date_yyyymmdd,
        "endDd": date_yyyymmdd,
        "csvxls_isNo": "false",
    }
    r = requests.post(KRX_JSON_URL, data=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    ob = js.get("OutBlock_1", [])
    if not isinstance(ob, list):
        return []
    return ob


def summarize_adv_dec(date_yyyymmdd: str, outblock_1: List[Dict[str, Any]]) -> AdvDecRow | None:
    if not outblock_1:
        return None

    adv = dec = unch = unknown = 0
    row_count = len(outblock_1)

    # FLUC_TP meaning:
    # 1 상승, 2 하락, 3 보합, 4 상한가, 5 하한가
    for row in outblock_1:
        tp = row.get("FLUC_TP", None)
        tp_i = _to_int_safe(tp)
        if tp_i in (1, 4):
            adv += 1
        elif tp_i in (2, 5):
            dec += 1
        elif tp_i == 3:
            unch += 1
        else:
            unknown += 1

    total = adv + dec + unch
    # unknown은 total에 포함하지 않음(품질 체크)
    return AdvDecRow(
        date=date_yyyymmdd,
        adv=adv,
        dec=dec,
        unch=unch,
        total=total,
        unknown=unknown,
        row_count=row_count,
    )


def get_last_date(conn: sqlite3.Connection) -> str | None:
    cur = conn.execute("SELECT MAX(date) FROM adv_dec_daily;")
    v = cur.fetchone()[0]
    return v


def iter_dates(start_yyyymmdd: str, end_yyyymmdd: str):
    s = pd.to_datetime(start_yyyymmdd, format="%Y%m%d")
    e = pd.to_datetime(end_yyyymmdd, format="%Y%m%d")
    d = s
    while d <= e:
        yield _yyyymmdd(d)
        d += pd.Timedelta(days=1)


def main():
    db_path = Path(_env("ADVDEC_DB_PATH", "data/cache/adv_dec_daily.sqlite"))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    bld = _env("ADVDEC_BLD", DEFAULT_BLD)
    referer = _env("ADVDEC_REFERER", DEFAULT_REFERER)

    mode = _env("ADVDEC_MODE", "incremental").strip().lower()  # incremental | backfill

    # backfill 범위(필요시)
    backfill_start = os.environ.get("ADVDEC_START", "")
    backfill_end = os.environ.get("ADVDEC_END", "")

    # incremental 기본: 마지막 저장 다음날 ~ 어제(UTC 기준)
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    default_end = _yyyymmdd(today - pd.Timedelta(days=1))

    with sqlite3.connect(str(db_path)) as conn:
        ensure_schema(conn)

        if mode == "backfill":
            if not backfill_start or not backfill_end:
                raise RuntimeError("ADVDEC_MODE=backfill requires ADVDEC_START and ADVDEC_END (YYYYMMDD)")
            start_yyyymmdd = backfill_start
            end_yyyymmdd = backfill_end
        else:
            last = get_last_date(conn)
            if last:
                start_dt = pd.to_datetime(last, format="%Y%m%d") + pd.Timedelta(days=1)
                start_yyyymmdd = _yyyymmdd(start_dt)
            else:
                # 초기값: 2년 정도만 기본 수집(원하면 env로 바꾸세요)
                start_yyyymmdd = _yyyymmdd(today - pd.Timedelta(days=365 * 2))
            end_yyyymmdd = default_end

        print(f"[advdec] db={db_path} mode={mode} range={start_yyyymmdd}~{end_yyyymmdd} bld={bld}")

        n_ok = n_skip = 0
        for d in iter_dates(start_yyyymmdd, end_yyyymmdd):
            try:
                ob1 = fetch_outblock_1(d, bld=bld, referer=referer)
                row = summarize_adv_dec(d, ob1)
                if row is None:
                    n_skip += 1
                    continue
                upsert_row(conn, row)
                n_ok += 1
                if n_ok % 20 == 0:
                    print(f"[advdec] progress ok={n_ok} skip={n_skip} last={d}")
            except Exception as e:
                # 휴일/오류는 스킵하고 진행(운영 안정성 우선)
                n_skip += 1
                continue

        print(f"[advdec] DONE ok={n_ok} skip={n_skip} -> {db_path}")


if __name__ == "__main__":
    main()
