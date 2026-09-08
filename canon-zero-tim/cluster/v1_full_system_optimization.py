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

# The streamed kept tape (tasks/v1_forward_dedup): the reverse pass consumes
# the forward's own tape group by group instead of recomputing it, with at
# most two groups' tapes alive.  Hardware-certified on 2026-09-02 on the
# one-host DP2xTP2 and DP4xTP1 GSM8K carriers (gradient anchors bitwise,
# forward programs halved / thirded, peak HBM at or below the flag-off
# baseline).  DeepSWE is not armed until its geometry has been certified.
_KEEP_TAPE_WORKLOADS = frozenset({
    "gsm8k",
    "frozenlake-p45",
    "frozenlake-m15",
})

# Reduce-once deliberately changes the cross-group/rank summation order.  It
# is admitted on the same three geometries as the streamed tape, with the
# registered per-geometry anchors.  DeepSWE remains excluded until its Phase 2
# gradient and row-mapping certification is complete.
_REDUCE_ONCE_WORKLOADS = _KEEP_TAPE_WORKLOADS

REGISTERED_FULL_WORKLOADS = frozenset({
    "gsm8k",
    *_P59_ONLY_WORKLOADS,
})

FULL_SYSTEM_OPTIMIZATION_ENV_NAMES = tuple(_BASE_ADDITIONS) + (
    "CANON_P67_P66_VMA_P59_ONLY",
    "CANON_P32_KEEP_TAPE",
    "CANON_DP_REDUCE_ONCE",
)


def full_system_optimization_base_additions(workload: str) -> dict[str, str]:
  """Returns the common exact tuple without workload-specific admitted knives."""
  if workload not in REGISTERED_FULL_WORKLOADS:
    raise ValueError(
        f"unregistered V1 full system-optimization workload: {workload!r}"
    )
  additions = dict(_BASE_ADDITIONS)
  if workload in _P59_ONLY_WORKLOADS:
    additions["CANON_P67_P66_VMA_P59_ONLY"] = "1"
  return additions


def full_system_optimization_additions(workload: str) -> dict[str, str]:
  """Returns a fresh exact env tuple for one registered production full job."""
  additions = full_system_optimization_base_additions(workload)
  if workload in _KEEP_TAPE_WORKLOADS:
    additions["CANON_P32_KEEP_TAPE"] = "stream"
  if workload in _REDUCE_ONCE_WORKLOADS:
    additions["CANON_DP_REDUCE_ONCE"] = "1"
  return additions
