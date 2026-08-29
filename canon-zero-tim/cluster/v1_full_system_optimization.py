"""Reviewed system-optimization additions for registered V1 full recipes."""

from __future__ import annotations

from types import MappingProxyType


_BASE_ADDITIONS = MappingProxyType({
    "CANON_P59_CHECKED_VMA": "1",
    "CANON_V1_HP_FIRST_UPDATE_GATE": "1",
    # These selectors reduce host receipt traffic without changing gradients.
    "CANON_DP_COMPARE_MODE": "fingerprint-hybrid",
    "CANON_DP_DISTINCT_SCHEDULE": "first-group-warmup",
    "CANON_DP_FINITE_FETCH": "batched-commit",
    # P71 fwd is hardware-certified on a non-unit TP axis. P71 bwd is not
    # admitted by this bundle.
    "CANON_P71_SCAN": "fwd",
})

_P59_ONLY_WORKLOADS = frozenset({
    "frozenlake-p45",
    "frozenlake-m15",
    "deepswe-qwen4b",
})

REGISTERED_FULL_WORKLOADS = frozenset({
    "gsm8k",
    *_P59_ONLY_WORKLOADS,
})

FULL_SYSTEM_OPTIMIZATION_ENV_NAMES = tuple(_BASE_ADDITIONS) + (
    "CANON_P67_P66_VMA_P59_ONLY",
)


def full_system_optimization_additions(workload: str) -> dict[str, str]:
  """Returns a fresh exact env tuple for one registered production full job."""
  if workload not in REGISTERED_FULL_WORKLOADS:
    raise ValueError(
        f"unregistered V1 full system-optimization workload: {workload!r}"
    )
  additions = dict(_BASE_ADDITIONS)
  if workload in _P59_ONLY_WORKLOADS:
    additions["CANON_P67_P66_VMA_P59_ONLY"] = "1"
  return additions
