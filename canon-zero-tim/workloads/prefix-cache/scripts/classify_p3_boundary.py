#!/usr/bin/env python3
"""Fail-closed classifier for the fixed-prefix Phase 3 APC boundary probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREFIX_LENGTHS = [
    1535,
    1536,
    1537,
    1685,
    1686,
    1687,
    1788,
    1792,
    2047,
    2048,
    2049,
]


class ClassificationError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ClassificationError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(
    report_path: Path, expect_apc: bool, expect_dirty_page: bool = False
) -> dict:
  _require(report_path.is_file(), f"report is absent: {report_path}")
  report = json.loads(report_path.read_text(encoding="utf-8"))
  _require(
      report.get("schema") == "phase3-apc-boundary-probe-v2",
      "report schema drifted",
  )
  _require(report.get("apc_enabled") is expect_apc, "APC arm marker drifted")
  _require(report.get("prefix_lengths") == PREFIX_LENGTHS,
           "prefix boundary set drifted")
  _require(report.get("topology") == "DP1xTP4", "topology drifted")
  _require(report.get("canonical_m") == 256, "canonical M drifted")
  _require(report.get("backward") == 0, "probe executed backward")
  _require(report.get("optimizer_commits") == 0,
           "probe executed an optimizer commit")
  _require(
      report.get("token_source")
      == "fixed-arange-prefix-v1:a-decode-completion-v1",
      "token source drifted",
  )
  _require(
      report.get("a_request_contract")
      == {
          "max_tokens": 16,
          "sampled_logprobs": 1,
          "prompt_logprobs": None,
          "skip_reading_prefix_cache": False,
          "ignore_eos": True,
      },
      "A request is not a cache-readable production decode",
  )
  weight = report.get("weight_attestation", {})
  _require(weight.get("equal") is True, "actor/engine weights are not exact")
  _require(weight.get("mismatch_indices") == [],
           "actor/engine weight mismatches are present")

  cases = report.get("cases", [])
  _require(len(cases) == len(PREFIX_LENGTHS), "case count drifted")
  _require(
      [case.get("prefix_length") for case in cases] == PREFIX_LENGTHS,
      "case order drifted",
  )

  differing_bytes = []
  cached_tokens = []
  for case in cases:
    _require(case.get("target_length") == 16, "target length drifted")
    _require(case.get("finite") is True, "non-finite logprob observed")
    _require(case.get("b_reset_prefix_cache") is True,
             "B did not attest full reset")
    _require(case.get("b_num_cached_tokens") == 0,
             "B consumed cached tokens")
    _require(bool(case.get("input_sha256")), "input hash is absent")
    _require(len(case.get("target_tokens", [])) == 16,
             "A-returned target token IDs are absent")
    _require(bool(case.get("target_sha256")), "target hash is absent")
    _require(bool(case.get("a_sha256")), "A hash is absent")
    _require(bool(case.get("b_sha256")), "B hash is absent")
    differing = int(case.get("differing_bytes", -1))
    elements = int(case.get("differing_elements", -1))
    _require(differing >= 0 and elements >= 0, "difference counts are invalid")
    _require((differing == 0) == (elements == 0),
             "byte/element difference signals disagree")
    _require((case.get("first_mismatch") is None) == (elements == 0),
             "first mismatch signal disagrees")
    differing_bytes.append(differing)
    cached_tokens.append(int(case.get("a_num_cached_tokens", -1)))

  dirty_control = report.get("dirty_page_control")
  if expect_dirty_page:
    _require(expect_apc, "dirty-page control requires APC-on")
    _require(isinstance(dirty_control, dict), "dirty-page evidence is absent")
    _require(dirty_control.get("enabled") is True,
             "dirty-page marker is not enabled")
    _require(dirty_control.get("target_prefix_length") == PREFIX_LENGTHS[0],
             "dirty-page target prefix drifted")
    page = dirty_control.get("page")
    _require(isinstance(page, dict), "dirty-page mutation receipt is absent")
    _require(page.get("layer_index") == 0, "dirty-page layer drifted")
    _require(page.get("logical_token_extent") == 256,
             "dirty-page logical extent drifted")
    _require(page.get("page_dtype") == "bfloat16",
             "dirty-page dtype drifted")
    _require(page.get("mutation") in ("fill-zero", "fill-one"),
             "dirty-page mutation kind drifted")
    _require(page.get("before_sha256") != page.get("after_sha256"),
             "dirty-page hashes did not change")
    _require(int(page.get("differing_bytes", 0)) > 0,
             "dirty-page byte mutation was ineffective")
    _require(int(page.get("differing_elements", 0)) > 0,
             "dirty-page element mutation was ineffective")
    _require(differing_bytes[0] > 0,
             "A-B gate did not catch the dirty cache page")
    _require(all(value == 0 for value in differing_bytes[1:]),
             "dirty-page mutation contaminated non-target cases")
    status = "DIRTY_PAGE_GATE_CAUGHT"
    first_red_prefix = PREFIX_LENGTHS[0]
    preceding_green_prefix = None
  elif dirty_control is not None:
    _require(
        dirty_control == {
            "enabled": False,
            "target_prefix_length": None,
            "page": None,
        },
        "unexpected dirty-page evidence in a normal arm",
    )

  if expect_dirty_page:
    pass
  elif expect_apc:
    _require(all(value > 0 for value in cached_tokens),
             "APC-on did not hit the prefix cache in every boundary case")
    red_indices = [
        index for index, value in enumerate(differing_bytes) if value > 0
    ]
    if red_indices:
      first_red_index = red_indices[0]
      status = "BOUNDARY_REPRODUCED_RED"
      first_red_prefix = PREFIX_LENGTHS[first_red_index]
      preceding_green_prefix = (
          PREFIX_LENGTHS[first_red_index - 1] if first_red_index else None
      )
    else:
      status = "BOUNDARY_DEEP_EXACT_NO_RED"
      first_red_prefix = None
      preceding_green_prefix = PREFIX_LENGTHS[-1]
  else:
    _require(all(value == 0 for value in cached_tokens),
             "APC-off unexpectedly reported cached tokens")
    _require(all(value == 0 for value in differing_bytes),
             "APC-off boundary control is not byte-exact")
    status = "BOUNDARY_CONTROL_GREEN"
    first_red_prefix = None
    preceding_green_prefix = PREFIX_LENGTHS[-1]

  return {
      "schema": "phase3-apc-boundary-classification-v2",
      "status": status,
      "expect_apc": expect_apc,
      "expect_dirty_page": expect_dirty_page,
      "prefix_lengths": PREFIX_LENGTHS,
      "a_num_cached_tokens": cached_tokens,
      "a_b_differing_bytes": differing_bytes,
      "first_red_prefix": first_red_prefix,
      "preceding_green_prefix": preceding_green_prefix,
      "report_sha256": _sha256(report_path),
      "claim": (
          "P3.1 fixed-prefix G-A boundary evidence only; not APC certification"
      ),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--report", type=Path, required=True)
  parser.add_argument("--expect-apc", choices=("0", "1"), required=True)
  parser.add_argument(
      "--expect-dirty-page", choices=("0", "1"), default="0"
  )
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise SystemExit(f"refusing to overwrite classification: {args.output}")
  try:
    result = classify(
        args.report,
        args.expect_apc == "1",
        expect_dirty_page=args.expect_dirty_page == "1",
    )
  except (ClassificationError, json.JSONDecodeError, OSError) as exc:
    result = {
        "schema": "phase3-apc-boundary-classification-v2",
        "status": "INCONCLUSIVE",
        "error": str(exc),
    }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps(result, sort_keys=True))
  if result["status"] == "INCONCLUSIVE":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
