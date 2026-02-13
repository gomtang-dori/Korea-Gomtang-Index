# src/lib/krx_http.py
from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any

import requests


@dataclass
class KRXHTTP:
    auth_key: str
    timeout: int = 30

    # throttling
    base_sleep_sec: float = 0.12  # every request sleep
    jitter_sec: float = 0.08

    # retry
    max_retries: int = 8
    backoff_start_sec: float = 1.0
    backoff_max_sec: float = 60.0

    def get_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"AUTH_KEY": self.auth_key, "Accept": "application/json"}

        backoff = self.backoff_start_sec
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            # small throttle to avoid 429
            time.sleep(self.base_sleep_sec + random.random() * self.jitter_sec)

            try:
                r = requests.get(url, headers=headers, params=params, timeout=self.timeout)

                # 429: Too Many Requests -> backoff & retry
                if r.status_code == 429:
                    wait = min(backoff, self.backoff_max_sec)
                    time.sleep(wait + random.random() * 0.5)
                    backoff = min(backoff * 2, self.backoff_max_sec)
                    continue

                # 5xx -> retry
                if 500 <= r.status_code < 600:
                    wait = min(backoff, self.backoff_max_sec)
                    time.sleep(wait + random.random() * 0.5)
                    backoff = min(backoff * 2, self.backoff_max_sec)
                    continue

                # other non-200 -> fail fast with context
                if r.status_code != 200:
                    raise requests.HTTPError(f"HTTP {r.status_code} for url={r.url} body={r.text[:300]}")

                # parse json
                try:
                    return r.json()
                except Exception as e:
                    raise RuntimeError(f"Non-JSON response for url={r.url}: {r.text[:500]}") from e

            except Exception as e:
                last_err = e
                # final attempt -> raise
                if attempt == self.max_retries:
                    raise
                # generic backoff
                wait = min(backoff, self.backoff_max_sec)
                time.sleep(wait + random.random() * 0.5)
                backoff = min(backoff * 2, self.backoff_max_sec)

        # should not reach
        if last_err:
            raise last_err
        raise RuntimeError("KRXHTTP.get_json failed unexpectedly")
