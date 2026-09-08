#!/usr/bin/env python3
"""Judge the P45 DP2xTP2 r2 recovery control without admitting program reuse.

The standard classifier and its verdict are retained verbatim. This separate
contract accepts only the exact conservative-key control, not a performance
candidate or a fresh-rollout/optimizer/target certification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any


STANDARD_PATH = Path(__file__).with_name("classify_frozenlake_dp2tp2.py")
_SPEC = importlib.util.spec_from_file_location(
    "v2_fl_standard_for_recovery", STANDARD_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
standard = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = standard
_SPEC.loader.exec_module(standard)

CONTRACT = "p45-dp2tp2-r2-recovery-control-v1"
CONTROL_REUSE = [(0, 36, 36, 36, 1, 1, 0)]
SHARED_REUSE = [(1, 36, 1, 1, 1, 1, 0)]
DEFERRED_REASON = (
    f"p59_layer_program_reuse_receipts={CONTROL_REUSE};expected={SHARED_REUSE}"
)
_CLEAN_DIFF = hashlib.sha256(b"").hexdigest()
_CLEAN_END = re.compile(
    r"^\[V2\.FL\.ONEHOST\] RUN_END docker_exit=0 elapsed_seconds=\d+ "
    r"contention=0(?: timeout=0)?[ \t]*$", re.MULTILINE
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm_bits(values: Any) -> list[str] | None:
  if not isinstance(values, list) or not values:
    return None
  if any(type(value) not in (int, float) or not math.isfinite(value)
         for value in values):
    return None
  return [struct.pack(">d", float(value)).hex() for value in values]


def classify(
    root: Path, *, docker_exit: int, anchor_registry: Path,
    expected_source_commit: str,
) -> dict[str, Any]:
  """Return scoped control evidence; never change the standard classification."""
  if re.fullmatch(r"[0-9a-f]{40}", expected_source_commit) is None:
    raise ValueError("recovery control requires the pre-registered full source SHA")
  original = standard.classify(
      root, workload="p45", geometry="dp2-tp2", arm="r2",
      docker_exit=docker_exit, anchor_registry=anchor_registry,
      require_anchor=True,
  )
  result = {
      "schema": "canon.v2-frozenlake-onehost.recovery-control.v1",
      "contract": CONTRACT,
      "verdict": "INCONCLUSIVE",
      "expected_source_commit": expected_source_commit,
      "claim_level": "p45-dp2tp2-r2-frozen-capsule-recovery-control",
      "claim_excludes": [
          "one-program-optimization-admission", "fresh-rollout-zero-tim",
          "full-gradient-leaf-parity", "optimizer-commit", "convergence",
          "new-speedup", "other-geometries", "Pathways", "GKE",
      ],
      "standard_classification": original,
      "program_sharing_optimization": {
          "admission": "NOT_ADMITTED", "disposition": "DEFERRED",
          "expected_reuse_receipts": SHARED_REUSE,
      },
      "reasons": list(original["reasons"]),
      "evaluator_sha256": {
          "standard_classifier": _sha256(STANDARD_PATH),
          "recovery_classifier": _sha256(Path(__file__)),
          "anchor_registry": _sha256(anchor_registry),
      },
  }
  if original["verdict"] == "INCONCLUSIVE":
    return result

  observed = original["receipts"]["p59_layer_program_reuse"]
  # Account only for this exact goal mismatch. No prefix-based error filtering,
  # no numerical exception, and no auto-selection from observed program counts.
  exact_control = observed == CONTROL_REUSE
  reasons = [reason for reason in original["reasons"]
             if not (exact_control and reason == DEFERRED_REASON)]

  def require(condition: bool, reason: str) -> None:
    if not condition:
      reasons.append(reason)

  require(original["verdict"] == "FAIL", "recovery_standard_contract_not_rejected")
  require(exact_control, "recovery_reuse_contract")
  require(DEFERRED_REASON in original["reasons"], "recovery_deferred_goal_missing")
  manifest = json.loads((root / "run_manifest.json").read_text())
  require(manifest.get("source_commit") == expected_source_commit,
          "recovery_source_commit")
  require(manifest.get("source_diff_sha256") == _CLEAN_DIFF,
          "recovery_source_not_clean")
  raw = (root / "raw.log").read_text(encoding="utf-8", errors="replace")
  require(
      len(_CLEAN_END.findall(raw)) == 1
      and len(re.findall(r"^\[V2\.FL\.ONEHOST\] RUN_END\b", raw, re.MULTILINE)) == 1,
      "recovery_clean_terminal",
  )

  entry = json.loads(anchor_registry.read_text())["anchors"].get(
      "p45:dp2-tp2:r2", {}
  )
  gradient = original["gradient"]
  observed_bits = _norm_bits(gradient["micro_gradient_norms"])
  expected_bits = _norm_bits(entry.get("micro_gradient_norms"))
  observed_update = _norm_bits([gradient["update_gradient_norm"]])
  expected_update = _norm_bits([entry.get("update_gradient_norm")])
  require(
      observed_bits is not None and observed_bits == expected_bits
      and observed_update is not None and observed_update == expected_update,
      "recovery_anchor_bits",
  )
  result.update(
      verdict="FAIL" if reasons else "RECOVERY_CONTROL_PASS",
      reasons=reasons,
      observed_reuse_receipts=observed,
      expected_reuse_receipts=CONTROL_REUSE,
      source_commit=manifest.get("source_commit"),
  )
  return result


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", required=True, type=Path)
  parser.add_argument("--docker-exit", required=True, type=int)
  parser.add_argument("--anchor-registry", required=True, type=Path)
  parser.add_argument("--expected-source-commit", required=True)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.resolve().is_relative_to(args.root.resolve()):
    raise ValueError("recovery output must be outside the immutable input run")
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  result = classify(
      args.root, docker_exit=args.docker_exit, anchor_registry=args.anchor_registry,
      expected_source_commit=args.expected_source_commit,
  )
  with args.output.open("x", encoding="utf-8") as output:
    output.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
  print(json.dumps({
      "contract": CONTRACT, "verdict": result["verdict"],
      "standard_verdict": result["standard_classification"]["verdict"],
      "reasons": result["reasons"],
  }, sort_keys=True))
  return 0 if result["verdict"] == "RECOVERY_CONTROL_PASS" else 1


if __name__ == "__main__":
  sys.exit(main())
