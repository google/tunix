"""Reviewed system-optimization additions for registered V1 full recipes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
import shlex
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

# These values are resolved by the exact full profile, not duplicated in each
# rendered recipe. Explicit controls retain their old presence semantics.
FULL_PROFILE_DEFAULT_NAMES = ("CANON_P32_KEEP_TAPE", "CANON_DP_REDUCE_ONCE")


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


def full_system_optimization_render_additions(workload: str) -> dict[str, str]:
  """Returns raw recipe additions; the exact full profile derives defaults."""
  return {
      name: value
      for name, value in full_system_optimization_additions(workload).items()
      if name not in FULL_PROFILE_DEFAULT_NAMES
  }


def _pair_defaults(values: Mapping[str, str], workload: str) -> dict[str, str]:
  if any(name in values for name in FULL_PROFILE_DEFAULT_NAMES):
    return {}
  bundle = full_system_optimization_additions(workload)
  return {name: bundle[name] for name in FULL_PROFILE_DEFAULT_NAMES}


def onehost_defaults(values: Mapping[str, str], *, arm: str, geometry: str,
                     run_stage: str) -> dict[str, str]:
  """Resolves the pair before one-host Docker and census argument delivery.

  This is not workload admission. The common launcher validates capture and
  negative-control signatures before calling this policy; the inner profile
  retains its exact runtime admission. Native and specialized diagnostic
  routes never acquire defaults. Do not import the full target's P71 mode:
  the ordinary one-host carrier has its own existing forward program.
  """
  if arm not in ("native", "zero-hp"):
    raise ValueError("one-host defaults require native or zero-hp arm")
  if geometry not in (
      "dp4-tp1", "dp2-tp2", "dp2-tp2-long", "dp2-tp2-long8k", "dp2-tp2-p45",
  ):
    raise ValueError("one-host defaults require a registered GSM8K geometry")
  if run_stage not in ("three-update", "six-update"):
    raise ValueError("one-host defaults require a committed update stage")
  if (arm == "native" or geometry == "dp2-tp2-p45"
      or values.get("V2_P0_CAPTURE_FULL_TREE", "") not in ("", "0")
      or values.get("V2_P0_NEGATIVE_CONTROL", "")):
    return {}
  return _pair_defaults(values, "gsm8k")


def full_profile_defaults(values: Mapping[str, str]) -> dict[str, str]:
  """Derives the registered pair only for an exact full-profile identity.

  If either option is present, preserve both raw values, including missing,
  empty or explicit0. This avoids silently constructing a different partial
  override. Existing runtime and full-run checks still judge manual controls.
  No model, geometry, Native or diagnostic inherits a default by prefix.
  """
  expected = {
      "CANON_V1_HP_FULL": "1",
      "CANON_P33_RUN_STAGE": "full",
      "CANON_P33_NO_COMMIT": "0",
  }
  profile = values.get("CANON_PROFILE_FILE", "")
  gsm8k = "qwen3-1p7b-dp16-tp4-gsm8k-v1-hp"
  frozen_profiles = {
      "qwen3-8b-dp8-tp8-frozenlake-v1-hp": ("8", "64"),
      "qwen3-8b-dp4-tp8-frozenlake-v1-hp": ("4", "32"),
  }
  if profile == f"cluster/profiles/{gsm8k}.env":
    workload = "gsm8k"
    expected.update({
        "CANON_PROFILE": gsm8k,
        "CANON_P32_WORKLOAD": "gsm8k",
        "CANON_MODEL_DIR_NAME": "qwen1p7b",
        "CANON_DP_SIZE": "16", "CANON_TP_SIZE": "4",
        "CANON_TOTAL_DEVICES": "64", "CANON_GSM8K_TRAIN": "1",
    })
    if values.get("CANON_GSM8K_VANILLA", "") not in ("", "0"):
      raise ValueError("full-profile defaults refuse Native GSM8K")
  else:
    match = next((name for name in frozen_profiles
                  if profile == f"cluster/profiles/{name}.env"), None)
    if match is None:
      raise ValueError("full-profile defaults require an exact registered profile")
    dp, devices = frozen_profiles[match]
    candidate = (values.get("CANON_P57_WORKLOAD_CANDIDATE", ""),
                 values.get("CANON_P57_DATA_SPLIT", ""))
    if candidate not in (("", ""), ("m15", "main")):
      raise ValueError("full-profile defaults require P45 or M15/main identity")
    workload = "frozenlake-m15" if candidate[0] else "frozenlake-p45"
    expected.update({
        "CANON_PROFILE": match,
        "CANON_P32_WORKLOAD": f"frozenlake-dp{dp}-tp8",
        "CANON_MODEL_DIR_NAME": "qwen8b_tp8",
        "CANON_DP_SIZE": dp, "CANON_TP_SIZE": "8",
        "CANON_TOTAL_DEVICES": devices,
        "CANON_P57_TIM_ARM": "zero", "CANON_P57_RUN_KIND": "train",
    })
  wrong = [name for name, value in expected.items() if values.get(name) != value]
  if wrong:
    raise ValueError("full-profile defaults identity mismatch: " + ",".join(wrong))
  return _pair_defaults(values, workload)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  route = parser.add_mutually_exclusive_group(required=True)
  route.add_argument("--profile-defaults", action="store_true")
  route.add_argument("--onehost-defaults", nargs=3,
                     metavar=("ARM", "GEOMETRY", "STAGE"))
  args = parser.parse_args()
  if args.onehost_defaults:
    arm, geometry, stage = args.onehost_defaults
    defaults = onehost_defaults(os.environ, arm=arm, geometry=geometry,
                                run_stage=stage)
  else:
    defaults = full_profile_defaults(os.environ)
  for name, value in defaults.items():
    # Only fixed, allowlisted names and registered constant values reach stdout.
    print(f"export {name}={shlex.quote(value)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
