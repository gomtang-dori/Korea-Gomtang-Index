# src/lib/krx_http.py
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class KRXHTTP:
    auth_key: str
    timeout: int = 30

    # throttle
    base_sleep_sec: float = 0.20
    jitter_sec: float = 0.15

    # retry/backoff
    max_retries: int = 8
    backoff_start_sec: float = 1.0
    backoff_max_sec: float = 60.0

    def get_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"AUTH_KEY": self.auth_key, "Accept": "application/json"}
        backoff = self.backoff_start_sec
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            time.sleep(self.base_sleep_sec + random.random() * self.jitter_sec)

            try:
                r = requests.get(url, headers=headers, params=params, timeout=self.timeout)

                if r.status_code == 429:
                    wait = min(backoff, self.backoff_max_sec)
                    time.sleep(wait + random.random() * 0.5)
                    backoff = min(backoff * 2, self.backoff_max_sec)
                    continue

                if 500 <= r.status_code < 600:
                    wait = min(backoff, self.backoff_max_sec)
                    time.sleep(wait + random.random() * 0.5)
                    backoff = min(backoff * 2, self.backoff_max_sec)
                    continue

                if r.status_code != 200:
                    raise requests.HTTPError(f"HTTP {r.status_code} url={r.url} body={r.text[:300]}")

                try:
                    return r.json()
                except Exception as e:
                    raise RuntimeError(f"Non-JSON response url={r.url}: {r.text[:500]}") from e

            except Exception as e:
                last_err = e
                if attempt == self.max_retries:
                    raise
                wait = min(backoff, self.backoff_max_sec)
                time.sleep(wait + random.random() * 0.5)
                backoff = min(backoff * 2, self.backoff_max_sec)

        if last_err:
            raise last_err
        raise RuntimeError("KRXHTTP.get_json failed unexpectedly")
