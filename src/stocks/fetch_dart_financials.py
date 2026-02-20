#!/usr/bin/env python3
"""
DART raw JSON cache backfill (2015~) - optimized 1 call per (ticker, year, reprt_code)
- API: fnlttSinglAcnt.json  [Source] https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS003&apiId=2019016
- Save:
  data/stocks/raw/dart/<ticker>/<year>_<reprt_code>_ALL.json
  (optional derived split caches)
"""

import os, re, io, json, time, zipfile, random, threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------- root ----------
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")) if os.getenv("PROJECT_ROOT") else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "stocks"
MASTER = DATA_DIR / "master" / "listings.parquet"
RAW_DIR = DATA_DIR / "raw" / "dart"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DART_API_KEY = os.getenv("DART_API2_KEY", "").strip()
if not DART_API_KEY:
    raise RuntimeError("Missing env var DART_API_KEY")

# ---------- env config ----------
TODAY_YEAR = datetime.utcnow().year
YEAR_FROM = int(os.getenv("DART_YEAR_FROM", "2015"))
YEAR_TO = int(os.getenv("DART_YEAR_TO", str(TODAY_YEAR)))

REPRT_CODES = [x.strip() for x in os.getenv("DART_REPRT_CODES", "11011,11012,11013,11014").split(",") if x.strip()]
MAX_WORKERS = int(os.getenv("DART_MAX_WORKERS", "5"))
CACHE_MODE = os.getenv("DART_CACHE_MODE", "skip").lower()  # skip|overwrite
RETRY = int(os.getenv("DART_RETRY", "3"))
MIN_INTERVAL = float(os.getenv("DART_MIN_INTERVAL_SEC", "0.15"))
RUN_BUDGET = int(os.getenv("DART_RUN_BUDGET", "38000"))

BACKOFF_BASE = float(os.getenv("DART_BACKOFF_BASE_SEC", "1.5"))
BACKOFF_MAX = float(os.getenv("DART_BACKOFF_MAX_SEC", "30"))

# endpoints
BASE = "https://opendart.fss.or.kr/api"
URL_CORPCODE = f"{BASE}/corpCode.xml"
URL_FNLTT = f"{BASE}/fnlttSinglAcnt.json"

# corpcode cache
CORP_CACHE = RAW_DIR / "_corpcode_cache"
CORP_CACHE.mkdir(parents=True, exist_ok=True)
CORP_XML = CORP_CACHE / "CORPCODE.xml"


def normalize_ticker6(x: str) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    digits = re.sub(r"[^0-9]", "", s)
    if not digits:
        return None
    if len(digits) > 6:
        digits = digits[-6:]
    return digits.zfill(6)


class RateLimiter:
    def __init__(self, sec: float):
        self.sec = sec
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            w = self.sec - (now - self.last)
            if w > 0:
                time.sleep(w)
            self.last = time.time()


class Budget:
    def __init__(self, budget: int):
        self.budget = budget
        self.used = 0
        self.lock = threading.Lock()

    def consume(self, n=1) -> bool:
        with self.lock:
            if self.used + n > self.budget:
                return False
            self.used += n
            return True

    def get(self) -> int:
        with self.lock:
            return self.used


rl = RateLimiter(MIN_INTERVAL)
bd = Budget(RUN_BUDGET)


def download_corpcode(session: requests.Session):
    if CORP_XML.exists() and CORP_XML.stat().st_size > 1000:
        return
    rl.wait()
    if not bd.consume(1):
        raise RuntimeError("RUN_BUDGET exceeded before corpCode.xml")

    r = session.get(URL_CORPCODE, params={"crtfc_key": DART_API_KEY}, timeout=60)
    r.raise_for_status()

    content = r.content
    # usually zip bytes
    if content[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            xml_name = next((n for n in zf.namelist() if n.upper().endswith("CORPCODE.XML")), None)
            if not xml_name:
                raise RuntimeError("CORPCODE.xml not found in zip")
            CORP_XML.write_bytes(zf.read(xml_name))
    else:
        # sometimes plain xml
        CORP_XML.write_bytes(content)


def parse_corp_map() -> Dict[str, str]:
    root = ET.fromstring(CORP_XML.read_bytes())
    m: Dict[str, str] = {}
    for el in root.findall("list"):
        corp_code = (el.findtext("corp_code") or "").strip()
        stock_code = (el.findtext("stock_code") or "").strip()
        if not corp_code or not stock_code:
            continue
        t6 = normalize_ticker6(stock_code)
        if t6:
            m[t6] = corp_code
    return m


def out_path(ticker: str, year: int, reprt_code: str) -> Path:
    d = RAW_DIR / str(ticker)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{year}_{reprt_code}_ALL.json"


def request_one(session: requests.Session, corp_code: str, year: int, reprt_code: str) -> dict:
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": str(reprt_code),
    }

    last_exc = None
    for attempt in range(1, RETRY + 1):
        try:
            rl.wait()
            if not bd.consume(1):
                raise RuntimeError("RUN_BUDGET exceeded")

            r = session.get(URL_FNLTT, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            status = str(data.get("status", "")).strip()

            if status == "000" or status == "013":
                return data

            if status == "020":
                wait = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1)) + random.random())
                time.sleep(wait)
                last_exc = RuntimeError(f"status=020 rate limit wait={wait:.2f}")
                continue

            wait = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1)) + random.random())
            time.sleep(wait)
            last_exc = RuntimeError(f"status={status} msg={data.get('message')}")
            continue

        except Exception as e:
            last_exc = e
            wait = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1)) + random.random())
            time.sleep(wait)

    raise last_exc


def job_one(session: requests.Session, ticker: str, name: str, corp_code: str, year: int, reprt_code: str) -> Tuple[str, str]:
    p = out_path(ticker, year, reprt_code)
    if CACHE_MODE == "skip" and p.exists() and p.stat().st_size > 50:
        return ticker, "skip"

    data = request_one(session, corp_code, year, reprt_code)
    meta = {
        "_meta": {
            "ticker": str(ticker),
            "ticker6": normalize_ticker6(ticker),
            "name": str(name),
            "corp_code": str(corp_code),
            "bsns_year": int(year),
            "reprt_code": str(reprt_code),
            "fetched_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    }
    merged = dict(meta)
    merged.update(data)

    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)

    status = str(merged.get("status", "")).strip()
    if status == "000":
        n = len(merged.get("list", []) or [])
        return ticker, f"OK n={n}"
    if status == "013":
        return ticker, "NO_DATA"
    return ticker, f"status={status}"


def main():
    print("[fetch_dart_financials] start")
    print(f"  PROJECT_ROOT={PROJECT_ROOT}")
    print(f"  years={YEAR_FROM}..{YEAR_TO}, reprt_codes={REPRT_CODES}")
    print(f"  workers={MAX_WORKERS}, cache_mode={CACHE_MODE}, run_budget={RUN_BUDGET}, min_interval={MIN_INTERVAL}")

    if not MASTER.exists():
        raise FileNotFoundError(f"missing {MASTER} (run fetch_listings first)")

    df = pd.read_parquet(MASTER)
    df["ticker"] = df["ticker"].astype(str)
    df["ticker6"] = df["ticker"].map(normalize_ticker6)

    session = requests.Session()
    download_corpcode(session)
    corp_map = parse_corp_map()
    print(f"  corp_map size={len(corp_map)}")

    df["corp_code"] = df["ticker6"].map(corp_map)
    missing = df["corp_code"].isna().sum()
    if missing:
        print(f"  missing corp_code tickers={missing} (will skip)")

    todo = df.dropna(subset=["corp_code"]).copy()
    years = list(range(YEAR_FROM, YEAR_TO + 1))

    jobs = []
    for _, r in todo.iterrows():
        for y in years:
            for rc in REPRT_CODES:
                jobs.append((r["ticker"], r.get("name", ""), r["corp_code"], y, rc))

    print(f"  jobs={len(jobs):,}")

    ok = skip = nodata = err = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(job_one, session, *j) for j in jobs]
        done = 0
        for f in as_completed(futures):
            done += 1
            try:
                t, msg = f.result()
                if msg == "skip":
                    skip += 1
                elif msg.startswith("OK"):
                    ok += 1
                elif msg == "NO_DATA":
                    nodata += 1
                else:
                    err += 1
            except Exception as e:
                err += 1

            if done % 500 == 0:
                print(f"  [progress] {done:,}/{len(futures):,} ok={ok:,} skip={skip:,} nodata={nodata:,} err={err:,} calls_used={bd.get():,}")

    print(f"[fetch_dart_financials] done ok={ok:,} skip={skip:,} nodata={nodata:,} err={err:,} calls_used={bd.get():,}")
    print(f"[fetch_dart_financials] raw dir={RAW_DIR}")

if __name__ == "__main__":
    main()
