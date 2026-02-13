# -*- coding: utf-8 -*-
"""
Daily_v2 quick verification runner
- Runs steps 1~10 sequentially
- Summarizes logs per step
- Stops on failure with actionable message
"""

import os
import sys
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# --------- helpers ---------

def _now_hhmmss() -> str:
    return time.strftime("%H:%M:%S")

def _print_hr():
    print("-" * 88)

def _env(name: str, default: Optional[str] = None) -> str:
    v = os.environ.get(name, default)
    return "" if v is None else str(v)

def _ensure_pythonpath_src():
    # Ensure subprocesses see PYTHONPATH=src
    pp = os.environ.get("PYTHONPATH", "")
    parts = [p for p in pp.split(os.pathsep) if p]
    if "src" not in parts:
        parts.insert(0, "src")
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)

def _run(cmd: List[str], step_name: str, timeout_sec: int = 1800) -> Tuple[int, str]:
    """
    Run a command, capture stdout+stderr, return (returncode, combined_output).
    """
    print(f"[{_now_hhmmss()}] ▶ {step_name}")
    print(f"  $ {' '.join(cmd)}")
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            env=os.environ.copy(),
        )
        out = p.stdout or ""
        return p.returncode, out
    except subprocess.TimeoutExpired:
        return 124, f"[verify_daily_v2] TIMEOUT after {timeout_sec}s: {' '.join(cmd)}\n"

def _tail_lines(text: str, n: int = 40) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= n:
        return "\n".join(lines)
    return "\n".join(lines[-n:])

def _pick_lines(text: str, contains_any: List[str], max_lines: int = 25) -> str:
    lines = (text or "").splitlines()
    picked = []
    for ln in lines:
        if any(k in ln for k in contains_any):
            picked.append(ln)
            if len(picked) >= max_lines:
                break
    return "\n".join(picked)

def _fail(step_no: int, step_name: str, out: str, hint: str):
    _print_hr()
    print(f"[FAIL] Step {step_no}: {step_name}")
    _print_hr()
    print("---- picked log ----")
    picked = _pick_lines(out, ["OK", "rows", "missing", "429", "Too Many", "ERROR", "Exception", "Traceback"], 40)
    if picked.strip():
        print(picked)
    else:
        print(_tail_lines(out, 60))
    _print_hr()
    print("---- hint ----")
    print(hint)
    _print_hr()
    raise SystemExit(1)

@dataclass
class Step:
    no: int
    name: str
    cmd: List[str]
    timeout_sec: int = 1800
    success_must_contain: Optional[List[str]] = None  # any of these must appear in output
    success_should_contain: Optional[List[str]] = None # optional signal keywords

def _check_success(step: Step, rc: int, out: str) -> Tuple[bool, str]:
    if rc != 0:
        return False, f"returncode={rc}"
    if step.success_must_contain:
        if not any(k in out for k in step.success_must_contain):
            return False, f"missing expected keywords: {step.success_must_contain}"
    return True, "ok"

def _summarize(step: Step, out: str):
    keys = ["OK", "rows", "missing", "missing_rate", "429", "Too Many Requests", "joined", "output", "docs/index.html"]
    picked = _pick_lines(out, keys, max_lines=18)
    if picked.strip():
        print("---- summary ----")
        print(picked)
    else:
        print("---- tail ----")
        print(_tail_lines(out, 30))

# --------- main verification ---------

def main():
    _ensure_pythonpath_src()

    # Paths (override by env if you want)
    members_raw = _env("K200_MEMBERS_RAW_PATH", "data/k200_members_raw.csv")
    members_std = _env("K200_MEMBERS_PATH", "data/k200_members.csv")
    ohlcv_cache = _env("K200_OHLCV_CACHE_PATH", "data/cache/k200_ohlcv.parquet")
    index_daily = _env("INDEX_DAILY_PATH", "data/index_daily.parquet")
    report_html = _env("REPORT_HTML_PATH", "docs/index.html")

    # Quick preflight: show important envs (secrets not printed)
    _print_hr()
    print("[verify_daily_v2] Preflight")
    print(f"  PYTHONPATH={os.environ.get('PYTHONPATH')}")
    print(f"  members_raw={members_raw}")
    print(f"  members_std={members_std}")
    print(f"  ohlcv_cache={ohlcv_cache}")
    print(f"  index_daily={index_daily}")
    print(f"  report_html={report_html}")
    print(f"  PROBE_ONLY={_env('PROBE_ONLY','0')} (you can set PROBE_ONLY=1 to run 6-day probe)")
    print(f"  PROGRESS_EVERY_N_DAYS={_env('PROGRESS_EVERY_N_DAYS','25')}")
    _print_hr()

    # Step plan (1~10)
    # Note: Step 2 uses PROBE_ONLY=1 if you export it; otherwise it will do normal caching.
    steps: List[Step] = [
        Step(
            1, "Normalize K200 members (raw -> standard)",
            ["python", "src/caches/normalize_k200_members.py"],
            timeout_sec=300,
            success_must_contain=["OK", "rows"],
        ),
        Step(
            2, "Cache K200 OHLCV (probe or short fill)",
            ["python", "src/caches/cache_k200_ohlcv.py"],
            timeout_sec=1800,
            success_must_contain=["OK"],
        ),
        Step(
            3, "Sanity check members/caches exist (local checks)",
            ["python", "-c",
             "import os,sys; from pathlib import Path; "
             "paths=['" + members_std + "','" + ohlcv_cache + "']; "
             "missing=[p for p in paths if not Path(p).exists()]; "
             "print('missing=',missing); sys.exit(1 if missing else 0)"],
            timeout_sec=30,
        ),
        Step(
            4, "Factor f02 Strength (from cache only)",
            ["python", "src/factors/f02_strength.py"],
            timeout_sec=1800,
            success_must_contain=["f02", "score"],
        ),
        Step(
            5, "Factor f03 Breadth (from cache only)",
            ["python", "src/factors/f03_breadth.py"],
            timeout_sec=1800,
            success_must_contain=["f03", "score"],
        ),
        Step(
            6, "Optional: check 429/backoff occurrence (no-fail)",
            ["python", "-c", "print('No-op. Search previous logs for 429 if needed.')"],
            timeout_sec=10,
        ),
        Step(
            7, "Assemble index (join factor files -> index_daily.parquet)",
            ["python", "src/assemble/assemble_index.py"],
            timeout_sec=900,
            success_must_contain=["index", "parquet"],
        ),
        Step(
            8, "Render report (index_daily -> docs/index.html)",
            ["python", "src/report/render_report.py"],
            timeout_sec=900,
            success_must_contain=["html", "docs"],
        ),
        Step(
            9, "Check outputs exist (index parquet + report html)",
            ["python", "-c",
             "from pathlib import Path; "
             f"p1=Path('{index_daily}'); p2=Path('{report_html}'); "
             "print('index_daily_exists=',p1.exists(),'size=',(p1.stat().st_size if p1.exists() else None)); "
             "print('report_exists=',p2.exists(),'size=',(p2.stat().st_size if p2.exists() else None)); "
             "import sys; sys.exit(0 if (p1.exists() and p2.exists()) else 1)"],
            timeout_sec=30,
        ),
        Step(
            10, "Print last 5 rows snapshot (index_daily parquet)",
            ["python", "-c",
             f"import pandas as pd; df=pd.read_parquet('{index_daily}'); "
             "print(df.sort_values('date').tail(5).to_string(index=False))"],
            timeout_sec=60,
        ),
    ]

    # Run
    for st in steps:
        _print_hr()
        rc, out = _run(st.cmd, f"Step {st.no} - {st.name}", timeout_sec=st.timeout_sec)
        ok, why = _check_success(st, rc, out)

        # Step 6 is "no-fail" informational
        if st.no == 6 and rc == 0:
            print("---- summary ----")
            print("Step6 is informational (no-fail).")
            continue

        if not ok:
            hint = ""
            if st.no == 1:
                hint = (
                    "members_raw 경로/인코딩/구분자 문제 가능성이 큽니다.\n"
                    f"- 파일 존재 확인: {members_raw}\n"
                    "- 헤더에 '종목코드'/'종목명'이 있는지, 탭/콤마(구분자) 확인\n"
                    "- rows가 150 미만이면 raw 파일 형식이 다르게 읽힌 것입니다."
                )
            elif st.no == 2:
                hint = (
                    "OHLCV 캐시 단계 실패입니다. 주로 아래 원인:\n"
                    "- KRX_STK_BYDD_TRD_URL / KRX_AUTH_KEY 시크릿 미설정/오타\n"
                    "- PROBE_ONLY=1인데 최근 영업일 계산/파싱 실패\n"
                    "- 429가 과도하면 base_sleep 상<span class="cursor">█</span>
