#!/usr/bin/env python3
"""
DART raw JSON cache backfill (2015~) - optimized 1 call per (ticker, year, reprt_code)
- API: fnlttSinglAcnt.json  [Source] https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS003&apiId=2019016
- Save:
  data/stocks/raw/dart/<ticker>/<year>_<reprt_code>_ALL.json

Key rotation (B-2):
- Use DART_API_KEY + optional DART_API2_KEY
- Round-robin key selection per request attempt
- Per-key budget: DART_RUN_BUDGET_PER_KEY (fallback: DART_RUN_BUDGET)
- If one key budget is exhausted, automatically failover to the other key

Notes:
- Each HTTP request consumes 1 unit from the selected key's budget (including retries and corpCode.xml download).
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

# ---------- API keys ----------
DART_API_KEY = os.getenv("DART_API_KEY", "").strip()
DART_API2_KEY = os.getenv("DART_API2_KEY", "").strip()

API_KEYS = [k for k in [DART_API_KEY, DART_API2_KEY] if k]
if not API_KEYS:
    raise RuntimeError("Missing env var DART_API_KEY (and optional DART_API2_KEY)")

# ---------- env config ----------
TODAY_YEAR = datetime.utcnow().year
YEAR_FROM = int(os.getenv("DART_YEAR_FROM", "2015"))
YEAR_TO = int(os.getenv("DART_YEAR_TO", str(TODAY_YEAR)))

REPRT_CODES = [x.strip() for x in os.getenv("DART_REPRT_CODES", "11011,11012,11013,11014").split(",") if x.strip()]
MAX_WORKERS = int(os.getenv("DART_MAX_WORKERS", "5"))
CACHE_MODE = os.getenv("DART_CACHE_MODE", "skip").lower()  # skip|overwrite
RETRY = int(os.getenv("DART_RETRY", "3"))
MIN_INTERVAL = float(os.getenv("DART_MIN_INTERVAL_SEC", "0.15"))

# Budget:
# - preferred: DART_RUN_BUDGET_PER_KEY
# - fallback:  DART_RUN_BUDGET (for backward compatibility; treated as per-key budget)
RUN_BUDGET_PER_KEY = int(os.getenv("DART_RUN_BUDGET_PER_KEY", os.getenv("DART_RUN_BUDGET", "38000")))

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


class KeyBudgetManager:
    """
    Round-robin key selection with per-key budget.
    - acquire(): returns (key_index, key_string) or (None, None) if all budgets exhausted.
    - consume(key_index, n=1): consumes budget.
    """
    def __init__(self, keys: List[str], budget_per_key: int):
        self.keys = keys
        self.lock = threading.Lock()
        self.budget_total = {i: int(budget_per_key) for i in range(len(keys))}
        self.used = {i: 0 for i in range(len(keys))}
        self.next_idx = 0

    def remaining(self, i: int) -> int:
        return self.budget_total[i] - self.used[i]

    def acquire(self) -> Tuple[Optional[int], Optional[str]]:
        with self.lock:
            n = len(self.keys)
            for _ in range(n):
                i = self.next_idx
                self.next_idx = (self.next_idx + 1) % n
                if self.remaining(i) > 0:
                    return i, self.keys[i]
            return None, None

    def consume(self, i: int, n: int = 1) -> bool:
        with self.lock:
            if i not in self.used:
                return False
            if self.used[i] + n > self.budget_total[i]:
                return False
            self.used[i] += n
            return True

    def used_total(self) -> int:
        with self.lock:
            return sum(self.used.values())

    def snapshot(self) -> Dict[str, int]:
        with self.lock:
            out = {}
            for i in range(len(self.keys)):
                out[f"key{i+1}_used"] = self.used[i]
                out[f"key{i+1}_remain"] = self.budget_total[i] - self.used[i]
            out["total_used"] = sum(self.used.values())
            out["total_budget"] = sum(self.budget_total.values())
            return out


rl = RateLimiter(MIN_INTERVAL)
km = KeyBudgetManager(API_KEYS, RUN_BUDGET_PER_KEY)

# thread-local session (safer than sharing one Session across threads)
_tls = threading.local()


def get_session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        _tls.session = s
    return s


def download_corpcode():
    if CORP_XML.exists() and CORP_XML.stat().st_size > 1000:
        return

    # Select key + consume budget
    k_idx, k = km.acquire()
    if k is None:
        raise RuntimeError("All API key budgets exhausted before corpCode.xml")

    rl.wait()
    if not km.consume(k_idx, 1):
        raise RuntimeError("All API key budgets exhausted before corpCode.xml (consume failed)")

    s = get_session()
    r = s.get(URL_CORPCODE, params={"crtfc_key": k}, timeout=60)
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


def request_one(corp_code: str, year: int, reprt_code: str) -> dict:
    last_exc = None

    for attempt in range(1, RETRY + 1):
        try:
            # choose key (round-robin among those with remaining budget)
            k_idx, k = km.acquire()
            if k is None:
                raise RuntimeError("All API key budgets exhausted")

            rl.wait()
            if not km.consume(k_idx, 1):
                # race/edge: in case budget exhausted between acquire/consume
                last_exc = RuntimeError("Budget consume failed (exhausted)")
                continue

            params = {
                "crtfc_key": k,
                "corp_code": corp_code,
                "bsns_year": str(year),
                "reprt_code": str(reprt_code),
            }

            s = get_session()
            r = s.get(URL_FNLTT, params=params, timeout=60)
            r.raise_for_status()

            data = r.json()
            status = str(data.get("status", "")).strip()

            # 정상 or 데이터없음은 그대로 반환 (데이터없음도 캐시해두는 게 중요)
            if status == "000" or status == "013":
                return data

            # 요청제한(020): backoff 후 재시도 (다음 attempt에서 다른 키로 넘어갈 수 있음)
            if status == "020":
                wait = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1)) + random.random())
                time.sleep(wait)
                last_exc = RuntimeError(f"status=020 rate limit wait={wait:.2f}")
                continue

            # 기타 오류: backoff 후 재시도
            wait = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1)) + random.random())
            time.sleep(wait)
            last_exc = RuntimeError(f"status={status} msg={data.get('message')}")
            continue

        except Exception as e:
            last_exc = e
            wait = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1)) + random.random())
            time.sleep(wait)

    raise last_exc


def job_one(ticker: str, name: str, corp_code: str, year: int, reprt_code: str) -> Tuple[str, str]:
    p = out_path(ticker, year, reprt_code)
    if CACHE_MODE == "skip" and p.exists() and p.stat().st_size > 50:
        return ticker, "skip"

    data = request_one(corp_code, year, reprt_code)

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
    snap = km.snapshot()
    print("[fetch_dart_financials] start")
    print(f"  PROJECT_ROOT={PROJECT_ROOT}")
    print(f"  years={YEAR_FROM}..{YEAR_TO}, reprt_codes={REPRT_CODES}")
    print(f"  workers={MAX_WORKERS}, cache_mode={CACHE_MODE}, retry={RETRY}, min_interval={MIN_INTERVAL}")
    print(f"  keys={len(API_KEYS)}, budget_per_key={RUN_BUDGET_PER_KEY}, total_budget={snap['total_budget']}")
    print(f"  budget_snapshot={snap}")

    if not MASTER.exists():
        raise FileNotFoundError(f"missing {MASTER} (run fetch_listings first)")

    df = pd.read_parquet(MASTER)
    df["ticker"] = df["ticker"].astype(str)
    df["ticker6"] = df["ticker"].map(normalize_ticker6)

    # corpCode.xml (consumes 1 call budget from one of keys, only when not cached)
    download_corpcode()
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
        futures = [ex.submit(job_one, *j) for j in jobs]
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
            except Exception:
                err += 1

            if done % 500 == 0:
                snap = km.snapshot()
                print(
                    f"  [progress] {done:,}/{len(futures):,} "
                    f"ok={ok:,} skip={skip:,} nodata={nodata:,} err={err:,} "
                    f"calls_used_total={snap['total_used']:,} "
                    f"(k1_used={snap.get('key1_used',0):,} k1_rem={snap.get('key1_remain',0):,} "
                    f"k2_used={snap.get('key2_used',0):,} k2_rem={snap.get('key2_remain',0):,})"
                )

    snap = km.snapshot()
    print(f"[fetch_dart_financials] done ok={ok:,} skip={skip:,} nodata={nodata:,} err={err:,} calls_used_total={snap['total_used']:,}")
    print(f"[fetch_dart_financials] budget_snapshot={snap}")
    print(f"[fetch_dart_financials] raw dir={RAW_DIR}")


if __name__ == "__main__":
    main()
