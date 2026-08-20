#!/usr/bin/env python3
"""Fail closed unless every admitted fixed-head executable is evidenced."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REQUEST_M = (16, 32, 64, 128, 256)
LEARNER_M = 4096
ENDPOINTS = ("untied_lm_head", "tied_embed")


def _fields(line: str) -> dict[str, str]:
  result = {}
  for token in line.split():
    if "=" not in token:
      continue
    name, value = token.split("=", 1)
    result[name] = value
  return result


def _matches_primal(
    record: dict[str, str], *, semantic_m: int, hidden: int, endpoint: str
) -> bool:
  chunks = 16 if semantic_m == LEARNER_M else 1
  expected = {
      "semantic_M": str(semantic_m),
      "fixed_M": "256",
      "K": str(hidden),
      "local_N": "37984",
      "fixed_N": "38144",
      "BM": "128",
      "BN": "256",
      "BK": "256",
      "chunks": str(chunks),
      "endpoint": endpoint,
  }
  return all(record.get(name) == value for name, value in expected.items())


def _matches_vjp(
    record: dict[str, str], *, hidden: int, endpoint: str
) -> bool:
  expected = {
      "semantic_M": "4096",
      "fixed_M": "256",
      "chunks": "16",
      "accumulation": "lax.scan",
      "order": "ascending",
      "K": str(hidden),
      "endpoint": endpoint,
  }
  return all(record.get(name) == value for name, value in expected.items())


def classify(
    text: str, *, endpoint: str, hidden: int, require_vjp: bool
) -> dict[str, object]:
  if endpoint not in ENDPOINTS:
    raise ValueError(f"unsupported fixed-head endpoint: {endpoint!r}")
  if hidden not in (2048, 4096):
    raise ValueError(f"unsupported fixed-head hidden width: {hidden}")

  primal = []
  vjp = []
  for line in text.splitlines():
    if "[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 " in line:
      primal.append(_fields(line))
    if "[PATHTRACE] CANON_P38_FIXED_LM_HEAD_VJP=1 " in line:
      vjp.append(_fields(line))

  missing_m = [
      semantic_m
      for semantic_m in (*REQUEST_M, LEARNER_M)
      if not any(
          _matches_primal(
              record,
              semantic_m=semantic_m,
              hidden=hidden,
              endpoint=endpoint,
          )
          for record in primal
      )
  ]
  foreign_endpoints = sorted({
      record.get("endpoint", "missing")
      for record in primal
      if record.get("K") == str(hidden)
      and record.get("endpoint") != endpoint
  })
  vjp_count = sum(
      _matches_vjp(record, hidden=hidden, endpoint=endpoint)
      for record in vjp
  )
  tied_marker_count = text.count("[P28.G5C] TIED_EMBEDDING_HEAD on")

  reasons = []
  if missing_m:
    reasons.append("missing_primal_M=" + ",".join(map(str, missing_m)))
  if foreign_endpoints:
    reasons.append("foreign_endpoints=" + ",".join(foreign_endpoints))
  if require_vjp and vjp_count < 1:
    reasons.append("missing_fixed_order_vjp")
  if endpoint == "tied_embed" and tied_marker_count < 1:
    reasons.append("missing_tied_embedding_adapter_marker")
  if endpoint == "untied_lm_head" and tied_marker_count:
    reasons.append("unexpected_tied_embedding_adapter_marker")

  verdict = (
      "P38_FIXED_LM_HEAD_RECEIPTS_PASS"
      if not reasons
      else "P38_FIXED_LM_HEAD_RECEIPTS_FAIL"
  )
  return {
      "schema": "p38-fixed-lm-head-receipts-v1",
      "verdict": verdict,
      "endpoint": endpoint,
      "hidden": hidden,
      "request_M": list(REQUEST_M),
      "learner_M": LEARNER_M,
      "require_vjp": require_vjp,
      "primal_records": len(primal),
      "vjp_records": len(vjp),
      "matching_vjp_records": vjp_count,
      "tied_marker_count": tied_marker_count,
      "missing_M": missing_m,
      "foreign_endpoints": foreign_endpoints,
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--log", required=True, type=Path)
  parser.add_argument("--endpoint", required=True, choices=ENDPOINTS)
  parser.add_argument("--hidden", required=True, type=int, choices=(2048, 4096))
  parser.add_argument("--require-vjp", action="store_true")
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()

  text = args.log.read_text(errors="replace")
  report = classify(
      text,
      endpoint=args.endpoint,
      hidden=args.hidden,
      require_vjp=args.require_vjp,
  )
  report["log_sha256"] = hashlib.sha256(args.log.read_bytes()).hexdigest()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  temporary = args.output.with_name(args.output.name + ".tmp")
  temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  temporary.replace(args.output)

  marker = (
      "[P38.FIXED_LM_HEAD] RECEIPTS_"
      + ("PASS" if not report["reasons"] else "FAIL")
      + f" endpoint={args.endpoint} K={args.hidden} "
      + "request_M="
      + ",".join(map(str, REQUEST_M))
      + f" learner_M={LEARNER_M} vjp={report['matching_vjp_records']}"
  )
  if report["reasons"]:
    marker += " reasons=" + ";".join(report["reasons"])
  print(marker, flush=True)
  return 0 if not report["reasons"] else 1


if __name__ == "__main__":
  sys.exit(main())
