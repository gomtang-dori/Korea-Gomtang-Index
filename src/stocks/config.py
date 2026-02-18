#!/usr/bin/env python3
"""
전역 설정
- SAMPLE_MODE: True → 샘플링 (N종목만), False → 전체
- SAMPLE_SIZE: 샘플링 시 종목 수
- MAX_WORKERS: 병렬처리 worker 수
- INCREMENTAL_MODE: True → 증분 업데이트만, False → 전체 백필
"""
import os

# ✅ 샘플링 설정 (환경변수 우선)
SAMPLE_MODE = os.getenv("SAMPLE_MODE", "true").lower() == "true"
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "100"))

# ✅ 병렬처리 설정
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))

# ✅ 증분 업데이트 설정
INCREMENTAL_MODE = os.getenv("INCREMENTAL_MODE", "false").lower() == "true"
INCREMENTAL_DAYS = int(os.getenv("INCREMENTAL_DAYS", "5"))  # 최근 N일만

# 날짜 설정
START_DATE = "20200101"  # 백필 시작일

print(f"[CONFIG] SAMPLE_MODE={SAMPLE_MODE}, SAMPLE_SIZE={SAMPLE_SIZE}")
print(f"[CONFIG] MAX_WORKERS={MAX_WORKERS}, INCREMENTAL_MODE={INCREMENTAL_MODE}")
