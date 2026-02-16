# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Fixed group-sum constraints for factor weights.

- 목적: 팩터 weight를 "그룹합 고정" 방식으로 안정화
- 사용처 예:
  - predict_horizons_report 결과(w_raw_avg)를 그룹별로 재스케일링
  - 최종 가중치 확정 시 cap=0.25 적용 전/후의 기준으로 사용

그룹 개념 (base_factor 기준):
- G1: 옵션/수급/브레드(중복군) = f02, f04, f08
- G2: 모멘텀/ADR              = f01, f03
- G3: 리스크/크레딧/변동/레버리지/안전자산/FX = f05, f06, f07, f09, f10

주의:
- base_factor 단위로 그룹을 정의합니다. (f06 vs f06_c는 같은 base_factor=f06)
- 8Y/1Y에 따라 f09 포함 여부만 다르게 설정합니다.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


# -----------------------------
# 1) base_factor -> group map
# -----------------------------
BASE_FACTOR_TO_GROUP: Dict[str, str] = {
    # G1
    "f02": "G1",
    "f04": "G1",
    "f08": "G1",
    # G2
    "f01": "G2",
    "f03": "G2",
    # G3
    "f05": "G3",
    "f06": "G3",
    "f07": "G3",
    "f09": "G3",
    "f10": "G3",
}


# 그룹별 base_factor 구성(명시적으로도 제공)
GROUP_TO_BASE_FACTORS: Dict[str, List[str]] = {
    "G1": ["f02", "f04", "f08"],
    "G2": ["f01", "f03"],
    "G3": ["f05", "f06", "f07", "f09", "f10"],
}


@dataclass(frozen=True)
class GroupSumConfig:
    """
    group_sums: 그룹합(총합=1.0 권장)
    allowed_base_factors: 사용할 base_factor 집합(정책상 제외할 base_factor 제거)
    """
    name: str
    group_sums: Dict[str, float]
    allowed_base_factors: Set[str]
    cap: float = 0.25  # 최종 cap 기본값 (요청사항)


def _validate(cfg: GroupSumConfig) -> None:
    # group_sums 합이 1.0인지(약간의 float 오차 허용)
    s = sum(float(v) for v in cfg.group_sums.values())
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"[{cfg.name}] group_sums must sum to 1.0. got={s}")

    # 정의되지 않은 그룹이 들어오면 오류
    for g in cfg.group_sums.keys():
        if g not in GROUP_TO_BASE_FACTORS:
            raise ValueError(f"[{cfg.name}] unknown group in group_sums: {g}")

    # allowed_base_factors가 group map에 없는 값이면 오류(오타 방지)
    for bf in cfg.allowed_base_factors:
        if bf not in BASE_FACTOR_TO_GROUP:
            raise ValueError(f"[{cfg.name}] unknown base_factor in allowed_base_factors: {bf}")

    # 그룹에 속한 base_factor가 전부 제외된 경우 경고 수준이지만, 여기서는 오류로 막음
    for g, members in GROUP_TO_BASE_FACTORS.items():
        if g not in cfg.group_sums:
            continue
        alive = [m for m in members if m in cfg.allowed_base_factors]
        if len(alive) == 0 and cfg.group_sums[g] > 0:
            raise ValueError(f"[{cfg.name}] group {g} has sum>0 but no allowed members.")



# -----------------------------
# 2) 정책별 그룹합 (고정)
# -----------------------------
# 사용자가 합의한 “안정적인 예측력” 관점 권장값
# - 1Y: G1=0.30, G2=0.15, G3=0.55
# - 8Y: G1=0.25, G2=0.15, G3=0.60 (f09 제외)
GROUP_SUMS_1Y: Dict[str, float] = {
    "G1": 0.30,
    "G2": 0.15,
    "G3": 0.55,
}

GROUP_SUMS_8Y: Dict[str, float] = {
    "G1": 0.25,
    "G2": 0.15,
    "G3": 0.60,
}


def get_config_1y(*, cap: float = 0.25) -> GroupSumConfig:
    allowed = set(BASE_FACTOR_TO_GROUP.keys())  # f01~f10 모두 가능
    cfg = GroupSumConfig(
        name="1Y",
        group_sums=GROUP_SUMS_1Y,
        allowed_base_factors=allowed,
        cap=cap,
    )
    _validate(cfg)
    return cfg


def get_config_8y(*, cap: float = 0.25) -> GroupSumConfig:
    allowed = set(BASE_FACTOR_TO_GROUP.keys())
    allowed.discard("f09")  # ✅ 8Y 정책: f09 제외
    cfg = GroupSumConfig(
        name="8Y",
        group_sums=GROUP_SUMS_8Y,
        allowed_base_factors=allowed,
        cap=cap,
    )
    _validate(cfg)
    return cfg


# -----------------------------
# 3) Helper utilities
# -----------------------------
def base_factor_of(factor_name: str) -> str:
    """
    factor_name 예:
      - 'f06'    -> 'f06'
      - 'f06_c'  -> 'f06'
      - 'f06__x' 같이 확장해도 base는 f06로 보고 싶으면 여기서 규칙 확장 가능
    """
    if factor_name.endswith("_c"):
        return factor_name[:-2]
    return factor_name


def group_of_base_factor(base_factor: str) -> str:
    if base_factor not in BASE_FACTOR_TO_GROUP:
        raise KeyError(f"Unknown base_factor={base_factor}")
    return BASE_FACTOR_TO_GROUP[base_factor]


def list_groups() -> List[str]:
    return list(GROUP_TO_BASE_FACTORS.keys())


def list_base_factors(group: str) -> List[str]:
    if group not in GROUP_TO_BASE_FACTORS:
        raise KeyError(f"Unknown group={group}")
    return list(GROUP_TO_BASE_FACTORS[group])
