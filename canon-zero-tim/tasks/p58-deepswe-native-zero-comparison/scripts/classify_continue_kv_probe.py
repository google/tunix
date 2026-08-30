#!/usr/bin/env python3
"""Classify the P58.22 live-continue versus clean-rescore KV probe."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PKG = HERE.parents[2]


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load classifier module: {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


DECODE = _load("p58_decode_classifier", HERE / "classify_decode_prefill_probe.py")
KV = _load(
    "p58_kv_classifier",
    PKG / "tasks/p38-pathways-decode-prefill-carrier/scripts"
    / "classify_p38_kv_observer.py",
)


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ValueError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _trajectory_prefix_matches(root: Path, target: np.ndarray) -> list[int]:
  paths = sorted(root.glob("batch-*.trajectories.jsonl.gz"))
  _require(len(paths) == 1, "continue-KV probe requires one trajectory journal")
  matches = []
  with gzip.open(paths[0], "rt", encoding="utf-8") as source:
    for index, line in enumerate(source):
      record = json.loads(line)
      trajectory = record.get("trajectory")
      _require(isinstance(trajectory, dict), "trajectory object is absent")
      prompt = np.asarray(trajectory.get("prompt_tokens"), dtype=np.int32)
      completion = np.asarray(
          trajectory.get("conversation_tokens"), dtype=np.int32
      )
      prompt_length = trajectory.get("prompt_length")
      if prompt_length is None:
        _require(
            record.get("compact_filtered") is True
            and prompt.size == 0
            and completion.size == 0,
            "only an empty compact-filtered trajectory may omit prompt_length",
        )
        continue
      _require(
          isinstance(prompt_length, int)
          and 0 <= prompt_length <= prompt.size,
          "trajectory prompt_length is absent or outside prompt_tokens",
      )
      # Durable DeepSWE trajectories store the prompt at the right edge of a
      # fixed-width, left-padded tensor.  The live vLLM request contains only
      # the semantic prompt, so strip that storage padding before joining the
      # exact sampled and environment token stream.
      prompt = prompt[-prompt_length:] if prompt_length else prompt[:0]
      tokens = np.concatenate((prompt, completion))
      if tokens.size >= target.size and np.array_equal(tokens[:target.size], target):
        matches.append(index)
  return matches


def classify(
    root: Path, *, source_sha: str | None, expected_hostname: str | None
) -> dict[str, Any]:
  root = root.resolve()
  decode = DECODE.classify(
      root,
      source_sha=source_sha,
      expected_hostname=expected_hostname,
  )
  _require(decode.get("zero_admission") is True, "not a Zero admission carrier")
  provenance = decode.get("carrier_provenance", {})
  _require(
      provenance.get("q4_tp4_continue_kv_diagnostic") is True,
      "continue-KV manifest selector is absent",
  )
  _require(provenance.get("q4_tp4_seam_diagnostic") == "", "standard decode leaked")
  _require(provenance.get("continue_decode_steps") == "8", "continue decode is not 8")
  outcome = decode.get("outcome")
  _require(
      outcome in ("ZERO_TIM_ALIGNMENT_RED", "ZERO_TIM_ALIGNMENT_ONLY_PASS"),
      "continue-KV run is neither the historical RED nor repaired exact arm",
  )
  repaired_exact = outcome == "ZERO_TIM_ALIGNMENT_ONLY_PASS"
  boundaries = decode.get("pre_alignment_boundaries", {})
  a_b_differing = int(
      boundaries["S_decode_vs_S_prefill"]["differing_bytes"]
  )
  _require(
      a_b_differing == 0 if repaired_exact else a_b_differing > 0,
      "continue-KV A-B boundary disagrees with the classified outcome",
  )
  _require(
      int(boundaries["S_prefill_vs_T_old"]["differing_bytes"]) == 0,
      "continue-KV diagnostic has B-C RED",
  )
  process = decode.get("process_status")
  controlled_precheck = (
      provenance.get("alignment_precheck_only") is True
      and provenance.get("alignment_controlled_exit") is True
  )
  expected_status = 42 if controlled_precheck else 1
  _require(
      isinstance(process, dict)
      and int(process.get("training_process_status", 0))
      == expected_status,
      "strict alignment diagnostic did not stop with the expected status",
  )
  _require(
      not (root / "backward_no_commit.json").exists(),
      "continue-KV diagnostic unexpectedly reached backward",
  )

  kv_root = root / "continue-kv"
  kv_report = KV.classify(kv_root, capsules=[], require_red_join=False)
  _require(kv_report.get("status") == "PASS", "KV observer evidence is invalid")
  _require(kv_report.get("records") == 2 and kv_report.get("pairs") == 1,
           "continue-KV probe requires exactly one A/B pair")
  records = KV._load_records(kv_root)
  pairs = KV._pair_records(records)
  _require(len(pairs) == 1, "continue-KV pair count drifted")
  live, clean = pairs[0]
  _require(
      isinstance(live.get("cache_effective_sharding"), dict)
      and isinstance(clean.get("cache_effective_sharding"), dict),
      "continue-KV probe lacks the effective device sharding contract",
  )
  target = np.asarray(live["arrays"]["token_ids"], dtype=np.int32).reshape(-1)
  _require(_trajectory_prefix_matches(root, target) == [0],
           "live KV token prefix does not join trajectory row 0")
  first_mismatch = decode["S_decode_vs_S_prefill"].get("first_mismatch")
  if repaired_exact:
    _require(first_mismatch is None, "exact A-B boundary has a first mismatch")
    first_prefix = None
  else:
    _require(isinstance(first_mismatch, dict), "A-B RED has no first mismatch")
    first_prefix = int(first_mismatch.get("logical_kv_prefix_length", -1))
  _require(
      2280 <= int(live.get("tag_prefix", -1)) < 3072,
      "continue-KV candidate was selected outside the signed seam window",
  )
  if first_prefix is not None:
    _require(
        int(live["target_seq_len"]) > first_prefix,
        "live KV capture ends before the observed first mismatch",
    )
  comparison = kv_report["comparisons"][0]
  fingerprint_equal = comparison.get("fingerprint_equal") is True
  classification = (
      "EXACT_TOKEN_CONTINUITY_ALIGNMENT_PASS"
      if repaired_exact
      else "LIVE_KV_FINGERPRINT_EQUAL_READ_PROGRAM_SUSPECT"
      if fingerprint_equal
      else "LIVE_KV_FINGERPRINT_DIFFERS_WRITE_STATE_SUSPECT"
  )
  return {
      "schema": "canon.p58.continue-kv-probe.classification.v1",
      "verdict": "PASS",
      "classification": classification,
      "source_commit": decode.get("source_commit"),
      "carrier_provenance": provenance,
      "decode_alignment": {
          "outcome": decode.get("outcome"),
          "N_action": decode.get("N_action"),
          "boundaries": boundaries,
          "first_mismatch": first_mismatch,
      },
      "kv_observer": {
          "classification": kv_report.get("classification"),
          "comparison": comparison,
          "live_request_id": live["request_id"],
          "clean_request_id": clean["request_id"],
          "tag_prefix": int(live["tag_prefix"]),
          "target_seq_len": int(live["target_seq_len"]),
          "token_history_sha256": live["token_history_sha256"],
          "live_json": live["path"],
          "live_json_sha256": live["json_sha256"],
          "live_npz": live["npz_path"],
          "live_npz_sha256": live["npz_sha256"],
          "clean_json": clean["path"],
          "clean_json_sha256": clean["json_sha256"],
          "clean_npz": clean["npz_path"],
          "clean_npz_sha256": clean["npz_sha256"],
      },
      "process_status": process,
      "claim": (
          "This certifies strict A=B=C and exact-token continuity through a "
          "controlled pre-backward stop on one Qwen3-4B DP1xTP4 "
          "continue-decode request. It does not certify backward, TP8, or "
          "production."
          if repaired_exact
          else
          "This is a bounded non-cryptographic KV fingerprint discriminator "
          "on one Qwen3-4B DP1xTP4 continue-decode request. It does not "
          "certify a repair, backward, TP8, or production."
      ),
  }


def _package(root: Path, report: dict[str, Any]) -> None:
  note = root / "RETURN_TO_AGENT.md"
  note.write_text(
      "# P58.22 continue-decode KV return\n\n"
      f"Classification: `{report['classification']}`\n\n"
      "The strict alignment gate stopped before backward. Read the claim in "
      "the classification JSON: an exact arm is alignment-only evidence; a "
      "RED arm remains diagnostic evidence only.\n",
      encoding="utf-8",
  )
  required = [
      "raw.log",
      "run_manifest.json",
      "probe_process_status.json",
      "pre_alignment.jsonl",
      "batch_metrics.jsonl",
      "continue_kv_probe.classification.json",
      "RETURN_TO_AGENT.md",
  ]
  trajectories = sorted(path.name for path in root.glob("batch-*.trajectories.jsonl.gz"))
  kv_files = sorted(
      str(path.relative_to(root))
      for path in (root / "continue-kv").glob("p38_kv_observer_*")
      if path.is_file()
  )
  files = required + trajectories + kv_files
  for name in files:
    _require((root / name).is_file(), f"return artifact is absent: {name}")
  (root / "SHA256SUMS").write_text(
      "".join(f"{_sha256(root / name)}  {name}\n" for name in sorted(files)),
      encoding="utf-8",
  )
  return_files = sorted(files + ["SHA256SUMS", "RETURN_FILES"])
  (root / "RETURN_FILES").write_text(
      "".join(f"{name}\n" for name in return_files), encoding="utf-8"
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--artifact-dir", type=Path, required=True)
  parser.add_argument("--source-sha")
  parser.add_argument("--expected-hostname")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--package", action="store_true")
  args = parser.parse_args()
  root = args.artifact_dir.resolve()
  output = (args.output or root / "continue_kv_probe.classification.json").resolve()
  if output.parent != root:
    raise SystemExit("classification output must live inside --artifact-dir")
  try:
    report = classify(
        root,
        source_sha=args.source_sha,
        expected_hostname=args.expected_hostname,
    )
  except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
    report = {
        "schema": "canon.p58.continue-kv-probe.classification.v1",
        "verdict": "FAIL",
        "classification": "MALFORMED_OR_INCOMPLETE_EVIDENCE",
        "error": str(exc),
    }
  output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  if args.package and report["verdict"] == "PASS":
    _package(root, report)
  print(json.dumps(report, sort_keys=True, separators=(",", ":")))
  raise SystemExit(0 if report["verdict"] == "PASS" else 1)


if __name__ == "__main__":
  main()
