#!/usr/bin/env python3
"""Classify a bounded DeepSWE decode-vs-prefill evidence package.

This tool never re-scores or fabricates log probabilities.  It joins the
bounded mismatch records in ``pre_alignment.jsonl`` back to the durable
trajectory journal and proves that reported token ids and S_decode values are
the values actually stored for the trajectory.  A hardware carrier can return
either an exact boundary or a reproduced finite RED; both are useful probe
outcomes.  Missing, malformed, non-finite, or unjoinable evidence fails closed.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


_TRAJECTORY_GLOB = "batch-*.trajectories.jsonl.gz"
_VALID_SCHEMAS = {
    "canon.p58.deepswe.trajectory.v1",
    "canon.local.deepswe.trajectory.v1",
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as source:
    value = json.load(source)
  if not isinstance(value, dict):
    raise ValueError(f"{path.name} must contain one JSON object")
  return value


def _load_last_jsonl(path: Path) -> dict[str, Any]:
  records = []
  with path.open(encoding="utf-8") as source:
    for line_number, line in enumerate(source, 1):
      if not line.strip():
        continue
      value = json.loads(line)
      if not isinstance(value, dict):
        raise ValueError(f"{path.name}:{line_number} is not a JSON object")
      records.append(value)
  if not records:
    raise ValueError(f"{path.name} contains no records")
  return records[-1]


def _load_trajectories(path: Path) -> list[dict[str, Any]]:
  records = []
  with gzip.open(path, "rt", encoding="utf-8") as source:
    for line_number, line in enumerate(source, 1):
      value = json.loads(line)
      if not isinstance(value, dict):
        raise ValueError(f"{path.name}:{line_number} is not a JSON object")
      if value.get("schema") not in _VALID_SCHEMAS:
        raise ValueError(
            f"{path.name}:{line_number} has unsupported schema "
            f"{value.get('schema')!r}"
        )
      records.append(value)
  if not records:
    raise ValueError(f"{path.name} contains no trajectory records")
  return records


def _boundary_counters(
    boundaries: dict[str, Any], name: str, n_action: int
) -> dict[str, Any]:
  boundary = boundaries.get(name)
  if not isinstance(boundary, dict):
    raise ValueError(f"{name} boundary is absent")
  if boundary.get("valid") is not True or boundary.get("finite") is not True:
    raise ValueError(f"{name} is invalid or non-finite")
  differing_elements = int(boundary.get("differing_elements", -1))
  differing_bytes = int(boundary.get("differing_bytes", -1))
  total_elements = int(boundary.get("total_elements", -1))
  if min(differing_elements, differing_bytes, total_elements) < 0:
    raise ValueError(f"{name} counters are invalid")
  if total_elements != n_action or differing_elements > total_elements:
    raise ValueError(f"{name} counters differ from admitted action tokens")
  return {
      "valid": True,
      "finite": True,
      "differing_elements": differing_elements,
      "differing_bytes": differing_bytes,
      "total_elements": total_elements,
      "element_fraction": boundary.get("element_fraction"),
      "byte_fraction": boundary.get("byte_fraction"),
      "max_abs": boundary.get("max_abs"),
  }


def _validate_zero_admission_backward(root: Path, n_action: int) -> dict[str, Any]:
  report_path = root / "backward_no_commit.json"
  alignment_path = root / "alignment.jsonl"
  for path in (report_path, alignment_path):
    if not path.is_file() or path.stat().st_size <= 0:
      raise ValueError(f"zero admission artifact is absent or empty: {path}")
  report = _load_json(report_path)
  for key, expected in {
      "verdict": "PASS",
      "commits": 0,
      "gradient_finite": True,
      "gradient_nonzero": True,
      "gradient_repeat_exact": True,
      "repeat_count": 2,
      "xprof_arm": "zero-hp",
  }.items():
    if report.get(key) != expected:
      raise ValueError(
          f"zero admission backward report {key}={report.get(key)!r}, "
          f"expected {expected!r}"
      )
  for key in (
      "model_changed_paths",
      "optimizer_changed_paths",
      "accumulator_changed_paths",
      "reference_changed_paths",
  ):
    if report.get(key) != []:
      raise ValueError(f"zero admission backward report changed {key}")
  if report.get("train_steps_before") != report.get("train_steps_after"):
    raise ValueError("zero admission backward changed train step")
  work_hashes = report.get("work_hashes")
  if not isinstance(work_hashes, dict) or work_hashes.get(
      "actor_update_calls"
  ) != 2:
    raise ValueError("zero admission backward work hashes are incomplete")

  alignment = _load_last_jsonl(alignment_path)
  boundaries = alignment.get("boundaries")
  if not isinstance(boundaries, dict):
    raise ValueError("post-backward alignment has no boundaries")
  post = {
      name: _boundary_counters(boundaries, name, n_action)
      for name in ("S_decode_vs_S_prefill", "S_prefill_vs_T_old")
  }
  if any(value["differing_bytes"] != 0 for value in post.values()):
    raise ValueError("zero admission post-backward boundaries are not exact")
  return {
      "report": report,
      "post_backward_boundaries": post,
      "artifacts": {
          "backward_no_commit": report_path.name,
          "backward_no_commit_sha256": _sha256(report_path),
          "alignment": alignment_path.name,
          "alignment_sha256": _sha256(alignment_path),
      },
  }


def _trajectory_arrays(
    record: dict[str, Any], row_index: int
) -> tuple[list[int], list[int], list[float] | None]:
  trajectory = record.get("trajectory")
  if not isinstance(trajectory, dict):
    raise ValueError(f"trajectory row {row_index} has no trajectory object")
  tokens = trajectory.get("conversation_tokens")
  masks = trajectory.get("conversation_masks")
  logps = trajectory.get("old_logprobs")
  if not isinstance(tokens, list) or not isinstance(masks, list):
    raise ValueError(f"trajectory row {row_index} token/mask arrays are absent")
  if len(tokens) != len(masks):
    raise ValueError(f"trajectory row {row_index} token/mask lengths differ")
  if logps is not None:
    if not isinstance(logps, list) or len(logps) != len(tokens):
      raise ValueError(f"trajectory row {row_index} logprob length differs")
    if not all(math.isfinite(float(value)) for value in logps):
      raise ValueError(f"trajectory row {row_index} has non-finite logprobs")
  if any(value not in (0, 1) for value in masks):
    raise ValueError(f"trajectory row {row_index} has a non-binary action mask")
  return tokens, masks, logps


def _percentile(values: list[float], fraction: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  return float(ordered[int(fraction * (len(ordered) - 1))])


def _shift_stats(
    mismatches: Iterable[dict[str, Any]], shift: int
) -> dict[str, Any]:
  by_row: dict[int, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
  for mismatch in mismatches:
    row, position = mismatch["coordinate"]
    by_row[int(row)][int(position)] = mismatch
  deltas = []
  for positions in by_row.values():
    for position, mismatch in positions.items():
      peer = positions.get(position + shift)
      if peer is not None:
        deltas.append(abs(float(mismatch["a"]) - float(peer["b"])))
  return {
      "shift": shift,
      "pairs": len(deltas),
      "mean_abs": (sum(deltas) / len(deltas)) if deltas else None,
      "median_abs": statistics.median(deltas) if deltas else None,
      "p90_abs": _percentile(deltas, 0.9),
  }


def _join_mismatch(
    mismatch: dict[str, Any],
    records: list[dict[str, Any]],
    candidates: dict[tuple[int, int], list[int]],
) -> int:
  required = {
      "coordinate",
      "completion_position",
      "completion_valid_length",
      "prompt_length",
      "token_id",
      "a",
      "b",
  }
  missing = sorted(required - mismatch.keys())
  if missing:
    raise ValueError(f"mismatch record is missing fields: {missing}")
  coordinate = mismatch["coordinate"]
  if (
      not isinstance(coordinate, list)
      or len(coordinate) != 2
      or int(coordinate[1]) != int(mismatch["completion_position"])
  ):
    raise ValueError("mismatch coordinate is malformed")
  position = int(mismatch["completion_position"])
  key = (
      int(mismatch["prompt_length"]),
      int(mismatch["completion_valid_length"]),
  )
  matches = []
  for row_index in candidates.get(key, []):
    tokens, masks, logps = _trajectory_arrays(records[row_index], row_index)
    if (
        logps is not None
        and 0 <= position < len(tokens)
        and int(tokens[position]) == int(mismatch["token_id"])
        and int(masks[position]) == 1
        and float(logps[position]) == float(mismatch["a"])
    ):
      matches.append(row_index)
  if len(matches) != 1:
    raise ValueError(
        "mismatch does not join exactly one durable trajectory row: "
        f"coordinate={coordinate} key={key} matches={matches}"
    )
  return matches[0]


def classify(
    root: Path,
    *,
    source_sha: str | None = None,
    expected_hostname: str | None = None,
    prealignment_path: Path | None = None,
) -> dict[str, Any]:
  root = root.resolve()
  manifest_path = root / "run_manifest.json"
  prealignment_path = (
      prealignment_path.resolve()
      if prealignment_path is not None
      else root / "pre_alignment.jsonl"
  )
  metrics_path = root / "batch_metrics.jsonl"
  trajectory_paths = sorted(root.glob(_TRAJECTORY_GLOB))
  if len(trajectory_paths) != 1:
    raise ValueError(
        f"probe requires exactly one trajectory journal, got {trajectory_paths}"
    )
  for path in (manifest_path, prealignment_path, metrics_path):
    if not path.is_file() or path.stat().st_size <= 0:
      raise ValueError(f"required probe artifact is absent or empty: {path}")

  manifest = _load_json(manifest_path)
  whitelist_sha256 = manifest.get("whitelist_sha256")
  if (
      not isinstance(whitelist_sha256, str)
      or len(whitelist_sha256) != 64
      or any(character not in "0123456789abcdef" for character in whitelist_sha256)
  ):
    raise ValueError("probe manifest has no lowercase SHA-256 whitelist identity")
  contract_name = manifest.get("contract_name")
  zero_admission = contract_name == "local-qwen4b-dp1-tp4-zero-admission"
  if contract_name in (
      "local-qwen4b-dp1-tp4-seam-probe",
      "local-qwen4b-dp1-tp4-zero-admission",
  ):
    if (
        manifest.get("onehost_seam_probe") is not True
        or manifest.get("onehost_xprof_arm") != "zero-hp"
        or manifest.get("stage") != "backward-no-commit"
    ):
      raise ValueError("one-host seam manifest selectors are inconsistent")
  if zero_admission:
    if manifest.get("q4_tp4_zero_admission") is not True:
      raise ValueError("Qwen3-4B TP4 Zero admission selector is absent")
    trajectory_replay = manifest.get("q4_tp4_trajectory_replay", False)
    if not isinstance(trajectory_replay, bool):
      raise ValueError("trajectory replay identity is not boolean")
    expected_sampling_temperature = 1.0 if trajectory_replay else 0.7
    if manifest.get("sampling_contract") != {
        "source": "explicit-cli",
        "temperature": expected_sampling_temperature,
        "top_k": 0,
        "top_p": 1.0,
    }:
      raise ValueError("Qwen3-4B TP4 Zero admission sampling differs")
    seam_diagnostic = manifest.get("q4_tp4_seam_diagnostic", "")
    if seam_diagnostic not in ("", "standard-decode"):
      raise ValueError("Qwen3-4B TP4 seam diagnostic identity is invalid")
    expected_continue_decode = "" if seam_diagnostic else "8"
    if manifest.get("continue_decode_steps") != expected_continue_decode:
      raise ValueError(
          "Qwen3-4B TP4 continue-decode identity differs from the seam arm"
      )
    continue_kv_diagnostic = manifest.get(
        "q4_tp4_continue_kv_diagnostic", False
    )
    if not isinstance(continue_kv_diagnostic, bool):
      raise ValueError("Qwen3-4B TP4 continue-KV identity is not boolean")
    if continue_kv_diagnostic and seam_diagnostic:
      raise ValueError("continue-KV diagnostic cannot use standard decode")
    short_backward = manifest.get("q4_tp4_short_backward", False)
    if not isinstance(short_backward, bool):
      raise ValueError("Qwen3-4B TP4 short-backward identity is not boolean")
    if trajectory_replay and not short_backward:
      raise ValueError("trajectory replay requires short backward identity")
    if short_backward:
      if seam_diagnostic or continue_kv_diagnostic:
        raise ValueError("short backward overlaps a diagnostic arm")
      replay_geometry = {
          "max_prompt_length": 2048,
          "max_response_length": 512,
          "max_turns": 16,
      }
      live_geometry = {
          "max_prompt_length": 1792,
          "max_response_length": 2880,
          "max_turns": 16,
      }
      for key, expected in (
          replay_geometry if trajectory_replay else live_geometry
      ).items():
        if manifest.get(key) != expected:
          raise ValueError(
              f"short backward {key}={manifest.get(key)!r}, expected {expected}"
          )
      expected_task_image = (
          "namanjain12/scrapy_final:"
          "439a3e59b8e858441f8d97dbc32f398db392330d"
          if trajectory_replay
          else "namanjain12/pillow_final:"
          "52079cb2975fda98476c7a7f172e5519e67ba612"
      )
      if manifest.get("task_image") != expected_task_image:
        raise ValueError("short backward clean task image changed")
      expected_whitelist = (
          "26e06ab7469987b4bc0c66d683e8468c"
          "2f10ae7d6842b0e138e563adcf87e257"
          if trajectory_replay
          else "7294da90559ebace771b7bd3fd8be01de"
          "87e0ae9bcb7ae1e317dbe5a6ed0db9f"
      )
      if manifest.get("whitelist_sha256") != expected_whitelist:
        raise ValueError("short backward clean whitelist changed")
      expected_cache = (
          "/mnt/disks/tunix-data/jax-compilation-cache/"
          "p58-q4-tp4-systemopt-b2g2-k2560"
          if trajectory_replay
          else "/mnt/disks/tunix-data/jax-compilation-cache/"
          "p58-q4-tp4-short-backward"
      )
      if manifest.get("compilation_cache_dir") != expected_cache:
        raise ValueError("short backward persistent compilation cache changed")
      if trajectory_replay:
        if manifest.get("task_image_id") != (
            "not-applicable-recorded-trajectory-replay"
        ):
          raise ValueError("trajectory replay unexpectedly binds a sandbox image")
        if manifest.get("replay_journal_sha256") != (
            "091a9273c2067876fbee1996ee853e3c8"
            "e861352e307cd5fb94fea2563aec456"
        ):
          raise ValueError("trajectory replay journal identity changed")
        if manifest.get("global_prompts") != 2 or manifest.get(
            "global_trajectories"
        ) != 4:
          raise ValueError("trajectory replay must be B2xG2, never batch one")
        if manifest.get("system_optimization") != {
            "carrier": "P28+P30+P71-fwd",
            "p59_rank_parallel_backward": False,
            "p59_reason": (
                "DP1 one-host cannot execute rank-parallel backward"
            ),
            "p28_segmented_forward": True,
            "p28_segmented_train": True,
            "p30_sparse_grad_assembly": True,
            "p30_reuse_segmented_engine": True,
            "p71_scan": "fwd",
        }:
          raise ValueError("trajectory replay system-optimization tuple changed")
    precheck_only = manifest.get("alignment_precheck_only", False)
    controlled_exit = manifest.get("alignment_controlled_exit", False)
    if not isinstance(precheck_only, bool) or not isinstance(
        controlled_exit, bool
    ):
      raise ValueError("alignment diagnostic provenance is not boolean")
    if precheck_only != controlled_exit:
      raise ValueError("alignment precheck and controlled exit must be paired")
    if precheck_only and not continue_kv_diagnostic:
      raise ValueError(
          "one-host alignment-only stop is restricted to continue-KV diagnostic"
      )
  else:
    seam_diagnostic = ""
    continue_kv_diagnostic = False
    precheck_only = False
    controlled_exit = False
    short_backward = False
    trajectory_replay = False
  if source_sha is not None and manifest.get("source_commit") != source_sha:
    raise ValueError(
        "probe source SHA differs from the requested source: "
        f"{manifest.get('source_commit')!r} != {source_sha!r}"
    )
  if expected_hostname is not None:
    observed = manifest.get("expected_hostname")
    if observed != expected_hostname:
      raise ValueError(
          f"probe hostname provenance differs: {observed!r} != "
          f"{expected_hostname!r}"
      )

  records = _load_trajectories(trajectory_paths[0])
  metrics = _load_last_jsonl(metrics_path)
  trajectory_sha256 = _sha256(trajectory_paths[0])
  if metrics.get("trajectory_sha256") not in (None, trajectory_sha256):
    raise ValueError("batch metrics trajectory SHA-256 differs from journal")
  metric_path = metrics.get("trajectory_path")
  if (
      metric_path is not None
      and Path(str(metric_path)).name != trajectory_paths[0].name
  ):
    raise ValueError("batch metrics trajectory path differs from journal")
  if metrics.get("trajectories") not in (None, len(records)):
    raise ValueError("batch metrics trajectory count differs from journal")
  if manifest.get("global_trajectories") not in (None, len(records)):
    raise ValueError("manifest trajectory count differs from journal")
  candidates: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
  admitted_action_tokens = 0
  statuses = collections.Counter()
  compact_rows = 0
  for row_index, record in enumerate(records):
    tokens, masks, _ = _trajectory_arrays(record, row_index)
    trajectory = record["trajectory"]
    statuses[str(record.get("status"))] += 1
    if record.get("compact_filtered") is True:
      compact_rows += 1
      # Timeout/overlong rows may be deliberately compacted to empty arrays.
      # They are durable audit records, but they are not admitted to alignment
      # and therefore need neither a prompt length nor a mismatch join key.
      continue
    prompt_length = trajectory.get("prompt_length")
    if not isinstance(prompt_length, int) or prompt_length <= 0:
      raise ValueError(f"trajectory row {row_index} has invalid prompt_length")
    candidates[(prompt_length, len(tokens))].append(row_index)
    admitted_action_tokens += sum(int(value) for value in masks)

  prealignment = _load_last_jsonl(prealignment_path)
  boundaries = prealignment.get("boundaries")
  if not isinstance(boundaries, dict):
    raise ValueError("pre-alignment record has no boundaries")
  n_action = int(prealignment.get("N_action", -1))
  if n_action != admitted_action_tokens:
    raise ValueError(
        "pre-alignment action count differs from durable trajectories: "
        f"{n_action} != {admitted_action_tokens}"
    )

  boundary_summaries = {
      name: _boundary_counters(boundaries, name, n_action)
      for name in (
          ("S_decode_vs_S_prefill", "S_prefill_vs_T_old")
          if zero_admission
          else ("S_decode_vs_S_prefill",)
      )
  }
  boundary = boundaries["S_decode_vs_S_prefill"]
  differing_elements = boundary_summaries["S_decode_vs_S_prefill"][
      "differing_elements"
  ]
  differing_bytes = boundary_summaries["S_decode_vs_S_prefill"][
      "differing_bytes"
  ]
  total_elements = boundary_summaries["S_decode_vs_S_prefill"][
      "total_elements"
  ]
  mismatches = boundary.get("mismatches")
  if not isinstance(mismatches, list):
    raise ValueError("pre-alignment mismatches must be a list")
  if differing_elements == 0 and mismatches:
    raise ValueError("exact boundary contains mismatch records")
  if differing_elements > 0 and not mismatches:
    raise ValueError("red boundary contains no bounded mismatch records")
  if mismatches and boundary.get("first_mismatch") != mismatches[0]:
    raise ValueError("first mismatch differs from bounded mismatch record 0")

  joined_rows = collections.Counter()
  action_starts = 0
  previous_environment = 0
  for mismatch in mismatches:
    if not isinstance(mismatch, dict):
      raise ValueError("mismatch entry is not an object")
    row_index = _join_mismatch(mismatch, records, candidates)
    joined_rows[row_index] += 1
    action_starts += int(mismatch.get("action_run_start") is True)
    previous_environment += int(
        mismatch.get("previous_token_is_environment") is True
    )

  process_status = None
  process_status_path = root / "probe_process_status.json"
  if process_status_path.is_file():
    process_status = _load_json(process_status_path)
    if process_status.get("profile") != "seam":
      raise ValueError("probe process status has the wrong profile")
    if not isinstance(process_status.get("training_process_status"), int):
      raise ValueError("probe process status has no integer exit status")

  backward = None
  zero_red_boundaries = [
      name
      for name, value in boundary_summaries.items()
      if value["differing_bytes"] != 0
  ]
  if zero_admission:
    if not n_action:
      outcome = "INCONCLUSIVE_NO_ACTION_TOKENS"
      verdict = "INCONCLUSIVE"
    elif zero_red_boundaries:
      outcome = (
          "ZERO_TIM_STANDARD_DECODE_ALIGNMENT_RED"
          if seam_diagnostic
          else "ZERO_TIM_ALIGNMENT_RED"
      )
      verdict = "FAIL"
    elif process_status is None:
      raise ValueError("zero admission has no process status")
    elif precheck_only:
      raw_path = root / "raw.log"
      if not raw_path.is_file():
        raise ValueError("alignment-only admission has no raw log")
      raw = raw_path.read_text(encoding="utf-8", errors="replace")
      if process_status["training_process_status"] != 42:
        raise ValueError("alignment-only admission did not use controlled exit 42")
      for marker in (
          "[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD",
          "[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0",
      ):
        if marker not in raw:
          raise ValueError(f"alignment-only admission lacks marker: {marker}")
      if (root / "backward_no_commit.json").exists():
        raise ValueError("alignment-only admission unexpectedly reached backward")
      outcome = "ZERO_TIM_ALIGNMENT_ONLY_PASS"
      verdict = "PASS"
    elif process_status["training_process_status"] != 0:
      outcome = "ZERO_TIM_BACKWARD_INCOMPLETE"
      verdict = "FAIL"
    else:
      backward = _validate_zero_admission_backward(root, n_action)
      outcome = (
          "ZERO_TIM_STANDARD_DECODE_CONTROL_PASS"
          if seam_diagnostic
          else "ZERO_TIM_BACKWARD_NO_COMMIT_PASS"
      )
      verdict = "PASS"
  else:
    if differing_elements:
      outcome = "FINITE_RED_REPRODUCED"
    elif n_action:
      outcome = "EXACT_ON_THIS_CARRIER"
    else:
      outcome = "INCONCLUSIVE_NO_ACTION_TOKENS"
    verdict = (
        "PASS" if outcome != "INCONCLUSIVE_NO_ACTION_TOKENS" else "INCONCLUSIVE"
    )

  return {
      "schema": "canon.p58.decode-prefill-probe.classification.v1",
      "verdict": verdict,
      "outcome": outcome,
      "source_commit": manifest.get("source_commit"),
      "model_id": manifest.get("model_id"),
      "contract_name": manifest.get("contract_name"),
      "role_topology": manifest.get("role_topology"),
      "carrier_provenance": {
          key: manifest.get(key)
          for key in (
              "run_id",
              "expected_hostname",
              "model_snapshot",
              "r2egym_commit",
              "task_image",
              "task_images",
              "task_image_id",
              "runner_sha256",
              "whitelist_sha256",
              "stage",
              "checked_vma_diagnostic",
              "onehost_xprof_arm",
              "onehost_seam_probe",
              "q4_tp4_zero_admission",
              "q4_tp4_seam_diagnostic",
              "q4_tp4_continue_kv_diagnostic",
              "q4_tp4_short_backward",
              "q4_tp4_trajectory_replay",
              "replay_journal_sha256",
              "max_prompt_length",
              "max_response_length",
              "alignment_precheck_only",
              "alignment_controlled_exit",
              "continue_decode_steps",
              "sampling_contract",
              "system_optimization",
              "global_prompts",
              "global_trajectories",
              "max_response_length",
              "max_turns",
          )
      },
      "process_status": process_status,
      "trajectory_rows": len(records),
      "compact_filtered_rows": compact_rows,
      "status_histogram": dict(sorted(statuses.items())),
      "N_action": n_action,
      "zero_admission": zero_admission,
      "zero_red_boundaries": zero_red_boundaries,
      "pre_alignment_boundaries": boundary_summaries,
      "backward_no_commit": backward,
      "S_decode_vs_S_prefill": {
          "differing_elements": differing_elements,
          "differing_bytes": differing_bytes,
          "total_elements": total_elements,
          "element_fraction": boundary.get("element_fraction"),
          "byte_fraction": boundary.get("byte_fraction"),
          "max_abs": boundary.get("max_abs"),
          "first_mismatch": boundary.get("first_mismatch"),
          "reported_mismatches": len(mismatches),
          "mismatches_truncated": boundary.get("mismatches_truncated"),
          "joined_artifact_rows": {
              str(key): value for key, value in sorted(joined_rows.items())
          },
          "reported_action_run_starts": action_starts,
          "reported_previous_environment": previous_environment,
          "shift_discriminator": [
              _shift_stats(mismatches, shift) for shift in (-1, 0, 1)
          ],
      },
      "artifacts": {
          "manifest": manifest_path.name,
          "manifest_sha256": _sha256(manifest_path),
          "pre_alignment": str(prealignment_path),
          "pre_alignment_sha256": _sha256(prealignment_path),
          "batch_metrics": metrics_path.name,
          "batch_metrics_sha256": _sha256(metrics_path),
          "trajectory": trajectory_paths[0].name,
          "trajectory_sha256": trajectory_sha256,
      },
      "claim": (
          "A standard-decode PASS is a P58.21 causal control only. It does "
          "not certify the continue-decode baseline, TP8, or production."
          if seam_diagnostic
          else
          "An alignment-only PASS certifies strict A=B=C and a controlled "
          "stop before backward only for this direct one-host Qwen3-4B "
          "DP1xTP4 carrier. It does not certify backward or TP8."
          if outcome == "ZERO_TIM_ALIGNMENT_ONLY_PASS"
          else
          "A PASS certifies strict A=B=C and backward-no-commit only for this "
          "direct one-host Qwen3-4B DP1xTP4 carrier. It does not certify TP8."
          if zero_admission
          else
          "A finite RED reproduces decode-vs-prefill divergence on this exact "
          "carrier only. An exact result does not certify other TP/DP geometry."
      ),
  }


def _write_return_note(root: Path, report: dict[str, Any]) -> None:
  boundary = report["S_decode_vs_S_prefill"]
  text = f"""# P58 decode-vs-prefill probe return

Verdict: `{report['verdict']}`
Outcome: `{report['outcome']}`
Source: `{report.get('source_commit')}`
Topology: `{json.dumps(report.get('role_topology'), sort_keys=True)}`

- trajectory rows: {report['trajectory_rows']}
- action tokens: {report['N_action']}
- differing elements: {boundary['differing_elements']}
- differing bytes: {boundary['differing_bytes']}
- max abs: {boundary['max_abs']}
- reported mismatches joined to durable rows: {sum(boundary['joined_artifact_rows'].values())}
- training process status: {json.dumps(report.get('process_status'), sort_keys=True)}

Return only `P58_SEAM_PROBE_RETURN.tar.gz` and its adjacent `.sha256` file.
After extraction, `SHA256SUMS` must verify from inside the directory. This
carrier does not certify a different TP/DP geometry. The runner is configured
backward-no-commit; this classifier does not authorize or certify any
optimizer commit.
"""
  (root / "RETURN_TO_AGENT.md").write_text(text, encoding="utf-8")


def _write_checksums(root: Path, names: Iterable[str]) -> None:
  lines = []
  for name in sorted(set(names)):
    path = root / name
    if path.is_file():
      lines.append(f"{_sha256(path)}  {name}\n")
  (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--artifact-dir", type=Path, required=True)
  parser.add_argument("--source-sha")
  parser.add_argument("--expected-hostname")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--package", action="store_true")
  args = parser.parse_args()

  root = args.artifact_dir.resolve()
  output = (args.output or (root / "decode_prefill_probe.classification.json")).resolve()
  if output.parent != root:
    raise SystemExit("classification output must live inside --artifact-dir")
  try:
    report = classify(
        root,
        source_sha=args.source_sha,
        expected_hostname=args.expected_hostname,
    )
  except (OSError, ValueError, json.JSONDecodeError) as exc:
    report = {
        "schema": "canon.p58.decode-prefill-probe.classification.v1",
        "verdict": "FAIL",
        "outcome": "MALFORMED_OR_INCOMPLETE_EVIDENCE",
        "error": str(exc),
    }
  output.write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  if args.package and report["verdict"] in ("PASS", "INCONCLUSIVE"):
    _write_return_note(root, report)
    names = [
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    _write_checksums(root, names)
  print(json.dumps(report, sort_keys=True, separators=(",", ":")))
  raise SystemExit(0 if report["verdict"] == "PASS" else 3 if report["verdict"] == "INCONCLUSIVE" else 1)


if __name__ == "__main__":
  main()
