#!/usr/bin/env python3
"""Fail closed unless every admitted fixed-head executable is evidenced."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REQUEST_M = (16, 32, 64, 128, 256)
DEFAULT_LEARNER_M = 4096
LEARNER_M = DEFAULT_LEARNER_M  # Historical import retained for old readers.
ENDPOINTS = ("untied_lm_head", "tied_embed")
GEOMETRIES = {
    ("tied_embed", 2048, 4): (37984, 38144),
    ("untied_lm_head", 4096, 4): (37984, 38144),
    ("untied_lm_head", 4096, 8): (18992, 19200),
    ("tied_embed", 2560, 8): (18992, 19200),
    ("untied_lm_head", 5120, 8): (18992, 19200),
}


def _fields(line: str) -> dict[str, str]:
  result = {}
  for token in line.split():
    if "=" not in token:
      continue
    name, value = token.split("=", 1)
    result[name] = value
  return result


def _matches_primal(
    record: dict[str, str],
    *,
    semantic_m: int,
    hidden: int,
    tp_size: int,
    endpoint: str,
    local_vocab: int,
    padded_local_vocab: int,
    learner_m: int,
) -> bool:
  chunks = semantic_m // 256 if semantic_m == learner_m else 1
  expected = {
      "semantic_M": str(semantic_m),
      "fixed_M": "256",
      "K": str(hidden),
      "TP": str(tp_size),
      "local_N": str(local_vocab),
      "fixed_N": str(padded_local_vocab),
      "BM": "128",
      "BN": "256",
      "BK": "256",
      "chunks": str(chunks),
      "endpoint": endpoint,
  }
  return (
      "p59_local" not in record
      and all(record.get(name) == value for name, value in expected.items())
  )


def _matches_p59_local_primal(
    record: dict[str, str],
    *,
    global_m: int,
    local_m: int,
    dp_size: int,
    hidden: int,
    tp_size: int,
    endpoint: str,
    local_vocab: int,
    padded_local_vocab: int,
) -> bool:
  expected = {
      "semantic_M": str(local_m),
      "fixed_M": "256",
      "K": str(hidden),
      "TP": str(tp_size),
      "local_N": str(local_vocab),
      "fixed_N": str(padded_local_vocab),
      "BM": "128",
      "BN": "256",
      "BK": "256",
      "chunks": "1",
      "endpoint": endpoint,
      "p59_local": "1",
      "global_M": str(global_m),
      "dp": str(dp_size),
  }
  return all(record.get(name) == value for name, value in expected.items())


def _matches_vjp(
    record: dict[str, str],
    *,
    hidden: int,
    tp_size: int,
    endpoint: str,
    local_vocab: int,
    padded_local_vocab: int,
    learner_m: int,
    p59_local_dp_size: int | None,
) -> bool:
  expected = {
      "semantic_M": str(learner_m),
      "fixed_M": "256",
      "chunks": str(learner_m // 256),
      "accumulation": "lax.scan",
      "order": "ascending",
      "K": str(hidden),
      "TP": str(tp_size),
      "local_N": str(local_vocab),
      "fixed_N": str(padded_local_vocab),
      "endpoint": endpoint,
  }
  if p59_local_dp_size is not None:
    expected.update({
        "local_M": str(learner_m // p59_local_dp_size),
        "chunks": "1",
        "tp_input_reduction": "all_gather_rank_order_f32_barrier",
    })
  return all(record.get(name) == value for name, value in expected.items())


def classify(
    text: str,
    *,
    endpoint: str,
    hidden: int,
    tp_size: int,
    require_vjp: bool,
    include_learner: bool = True,
    learner_m: int = DEFAULT_LEARNER_M,
    p59_local_dp_size: int | None = None,
) -> dict[str, object]:
  if endpoint not in ENDPOINTS:
    raise ValueError(f"unsupported fixed-head endpoint: {endpoint!r}")
  try:
    local_vocab, padded_local_vocab = GEOMETRIES[(endpoint, hidden, tp_size)]
  except KeyError as error:
    raise ValueError(
        "unsupported fixed-head geometry: "
        f"endpoint={endpoint} hidden={hidden} tp={tp_size}"
    ) from error
  if learner_m not in (2048, 4096) or learner_m % 256:
    raise ValueError(f"unsupported fixed-head learner M: {learner_m}")
  if learner_m == 2048 and (endpoint, hidden, tp_size) != (
      "untied_lm_head", 4096, 8
  ):
    raise ValueError(
        "fixed-head learner M2048 is registered only for "
        "Qwen3-8B/TP8 untied_lm_head"
    )
  if p59_local_dp_size is not None:
    if not include_learner or not require_vjp:
      raise ValueError(
          "P59 local receipt mode requires learner primal and VJP receipts"
      )
    if p59_local_dp_size not in (8, 16):
      raise ValueError(
          f"unsupported P59 local DP size: {p59_local_dp_size}"
      )
    if learner_m % p59_local_dp_size or (
        learner_m // p59_local_dp_size != 256
    ):
      raise ValueError(
          "P59 local fixed-head receipt requires global/local learner rows "
          f"{learner_m}/{learner_m // p59_local_dp_size} with local_M=256"
      )

  primal = []
  vjp = []
  for line in text.splitlines():
    if "[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 " in line:
      primal.append(_fields(line))
    if "[PATHTRACE] CANON_P38_FIXED_LM_HEAD_VJP=1 " in line:
      vjp.append(_fields(line))

  required_primal_m = (
      *REQUEST_M,
      *((learner_m,) if include_learner and p59_local_dp_size is None else ()),
  )
  missing_m = [
      semantic_m
      for semantic_m in required_primal_m
      if not any(
          _matches_primal(
              record,
              semantic_m=semantic_m,
              hidden=hidden,
              tp_size=tp_size,
              endpoint=endpoint,
              local_vocab=local_vocab,
              padded_local_vocab=padded_local_vocab,
              learner_m=learner_m,
          )
          for record in primal
      )
  ]
  p59_local_m = (
      learner_m // p59_local_dp_size
      if p59_local_dp_size is not None
      else None
  )
  p59_local_primal_count = 0
  if include_learner and p59_local_dp_size is not None:
    p59_local_primal_count = sum(
        _matches_p59_local_primal(
            record,
            global_m=learner_m,
            local_m=p59_local_m,
            dp_size=p59_local_dp_size,
            hidden=hidden,
            tp_size=tp_size,
            endpoint=endpoint,
            local_vocab=local_vocab,
            padded_local_vocab=padded_local_vocab,
        )
        for record in primal
    )
  foreign_endpoints = sorted({
      record.get("endpoint", "missing")
      for record in primal
      if record.get("K") == str(hidden)
      and record.get("endpoint") != endpoint
  })
  vjp_count = sum(
      _matches_vjp(
          record,
          hidden=hidden,
          tp_size=tp_size,
          endpoint=endpoint,
          local_vocab=local_vocab,
          padded_local_vocab=padded_local_vocab,
          learner_m=learner_m,
          p59_local_dp_size=p59_local_dp_size,
      )
      for record in vjp
  )
  tied_marker_count = text.count("[P28.G5C] TIED_EMBEDDING_HEAD on")

  reasons = []
  if missing_m:
    reasons.append("missing_primal_M=" + ",".join(map(str, missing_m)))
  if include_learner and p59_local_dp_size is not None and (
      p59_local_primal_count < 1
  ):
    reasons.append("missing_p59_local_primal")
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
      "tp_size": tp_size,
      "local_vocab": local_vocab,
      "padded_local_vocab": padded_local_vocab,
      "request_M": list(REQUEST_M),
      "learner_M": learner_m if include_learner else None,
      "p59_local_dp_size": p59_local_dp_size,
      "p59_local_M": p59_local_m,
      "matching_p59_local_primal_records": p59_local_primal_count,
      "include_learner": include_learner,
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
  parser.add_argument(
      "--hidden", required=True, type=int, choices=(2048, 2560, 4096, 5120)
  )
  parser.add_argument("--tp-size", required=True, type=int, choices=(4, 8))
  parser.add_argument(
      "--learner-m", type=int, choices=(2048, 4096), default=4096
  )
  parser.add_argument("--require-vjp", action="store_true")
  parser.add_argument(
      "--p59-local-dp-size",
      type=int,
      choices=(8, 16),
      help=(
          "Require the P59 rank-local learner receipt, its global/local row "
          "identity, one local M256 chunk, and fixed TP input reduction."
      ),
  )
  parser.add_argument(
      "--request-only",
      action="store_true",
      help="Require serving request buckets but not learner or VJP receipts.",
  )
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()

  text = args.log.read_text(errors="replace")
  report = classify(
      text,
      endpoint=args.endpoint,
      hidden=args.hidden,
      tp_size=args.tp_size,
      require_vjp=args.require_vjp,
      include_learner=not args.request_only,
      learner_m=args.learner_m,
      p59_local_dp_size=args.p59_local_dp_size,
  )
  report["log_sha256"] = hashlib.sha256(args.log.read_bytes()).hexdigest()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  temporary = args.output.with_name(args.output.name + ".tmp")
  temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  temporary.replace(args.output)

  marker = (
      "[P38.FIXED_LM_HEAD] RECEIPTS_"
      + ("PASS" if not report["reasons"] else "FAIL")
      + f" endpoint={args.endpoint} K={args.hidden} TP={args.tp_size} "
      + "request_M="
      + ",".join(map(str, REQUEST_M))
      + f" learner_M={report['learner_M']} "
      + f"p59_dp={report['p59_local_dp_size']} "
      + f"local_M={report['p59_local_M']} "
      + f"vjp={report['matching_vjp_records']}"
  )
  if report["reasons"]:
    marker += " reasons=" + ";".join(report["reasons"])
  print(marker, flush=True)
  return 0 if not report["reasons"] else 1


if __name__ == "__main__":
  sys.exit(main())
