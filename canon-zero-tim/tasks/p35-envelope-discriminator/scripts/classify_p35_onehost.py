#!/usr/bin/env python3
"""Classify an immutable one-host P35 raw log without rerunning TPU work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


TERMINAL_RE = re.compile(
    r"^\[rank0\]: tunix\.rl\.envelope_probe\.EnvelopeProbeError: "
    r"known A-C red was not reproduced in the current batch$",
    re.MULTILINE,
)


def classify_text(text: str) -> dict[str, object]:
  counts = {
      "fixed_ar": text.count("CANON_FIXED_AR=1 fixed-order tree"),
      "fixed_ar_embed": text.count(
          "CANON_FIXED_AR_EMBED=1 fixed-order embed gather"
      ),
      "logprob_m": text.count("CANON_LOGPROB_M on"),
      "p35_complete": len(
          re.findall(
              r"^\[CANON_P35\] REPORT_COMPLETE .*STOP_BEFORE_BACKWARD$",
              text,
              re.MULTILINE,
          )
      ),
      "p35_replay": len(
          re.findall(
              r"^\[CANON_P35\.3\] REPLAY_COMPLETE.*$", text, re.MULTILINE
          )
      ),
      "known_red_not_reproduced": len(TERMINAL_RE.findall(text)),
      "known_red_text_occurrences": text.count(
          "known A-C red was not reproduced in the current batch"
      ),
      "postflight_clean": text.count(
          "[postflight] C7/C8 violations: none"
      ),
      "contract_red": text.count("C7/C8 violation [post-import]"),
  }
  local_exact = (
      counts["fixed_ar"] > 0
      and counts["fixed_ar_embed"] > 0
      and counts["logprob_m"] > 0
      and counts["postflight_clean"] == 1
      and counts["contract_red"] == 0
      and counts["p35_complete"] == 0
      and counts["p35_replay"] == 0
      and counts["known_red_not_reproduced"] == 1
  )
  return {
      "verdict": "LOCAL_NOT_REPRODUCED" if local_exact else "INCONCLUSIVE",
      "counts": counts,
      "claims": {
          "action_ac_bitwise_mismatch_found": False if local_exact else None,
          "six_arm_replay_executed": False,
          "backward_executed": False,
          "optimizer_update": False,
          "pathways_target_verdict": False,
      },
  }


def self_test() -> None:
  prefix = "\n".join(
      (
          "[PATHTRACE] CANON_FIXED_AR=1 fixed-order tree at x",
          "[PATHTRACE] CANON_FIXED_AR_EMBED=1 fixed-order embed gather at x",
          "[PATHTRACE] CANON_LOGPROB_M on rows=256",
          "[rank0]:     raise EnvelopeProbeError(\"known A-C red was not reproduced in the current batch\")",
          "[rank0]: tunix.rl.envelope_probe.EnvelopeProbeError: known A-C red was not reproduced in the current batch",
          "[P35.ONEHOST.POSTFLIGHT] [postflight] C7/C8 violations: none",
      )
  )
  assert classify_text(prefix)["verdict"] == "LOCAL_NOT_REPRODUCED"
  assert classify_text(prefix.replace("CANON_LOGPROB_M on", "missing"))["verdict"] == "INCONCLUSIVE"
  assert classify_text(prefix + "\n" + TERMINAL_RE.findall(prefix)[0])["verdict"] == "INCONCLUSIVE"
  assert classify_text(prefix + "\n[CANON_P35.3] REPLAY_COMPLETE x")["verdict"] == "INCONCLUSIVE"


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--source-commit", default="")
  parser.add_argument("--self-test", action="store_true")
  args = parser.parse_args()
  if args.self_test:
    self_test()
    print("P35 one-host classifier self-test: PASS")
    return 0
  if args.raw is None or args.output is None:
    parser.error("--raw and --output are required unless --self-test is used")
  raw_bytes = args.raw.read_bytes()
  payload = {
      "schema_version": 2,
      "source_commit": args.source_commit,
      "topology": "DP1xTP4-direct-attached",
      "raw_log": str(args.raw),
      "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
      **classify_text(raw_bytes.decode("utf-8", errors="replace")),
  }
  args.output.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(payload, sort_keys=True))
  return 0 if payload["verdict"] == "LOCAL_NOT_REPRODUCED" else 1


if __name__ == "__main__":
  sys.exit(main())
