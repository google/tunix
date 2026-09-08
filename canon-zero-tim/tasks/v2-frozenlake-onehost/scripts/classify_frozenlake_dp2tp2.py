#!/usr/bin/env python3
"""Fail-closed classifier for a FrozenLake DP2xTP2 no-commit carrier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


_FORWARD_RE = re.compile(
    r"^\[P32\.DP2\] (forward_group_(?:issued|done)) "
    r"group=(\d+)/8 .* n_real=\(([^)]*)\)$",
    re.MULTILINE,
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_RE = re.compile(r"[0-9a-f]{40}\Z")
_BUCKET_RE = re.compile(
    r"^\[P59\.DP2\] reducer_bucket_schedule "
    r"programs=(\d+) max_local_bytes=(\d+) "
    r"peak_local_bytes=(\d+) total_local_bytes=(\d+)$",
    re.MULTILINE,
)
_HBM_STAGE_PREFIX = "[V2.FL.HBM_STAGE] "
_P75_BUCKET_HBM_RE = re.compile(
    r"^report_bucket_(\d+)_(before_execute|after_execute|after_delete)$"
)
_P75_GROUP_HBM_STAGE = "report_group_after_bucket_0"
_REPORT_ADJOINT_MEMORY_PREFIX = "[V2.FL.REPORT_ADJOINT_MEMORY] "
_P75_BUCKET_RE = re.compile(
    r"^\[P75\.REPORT_ADJOINT_BUCKETS\] enabled=1 "
    r"programs=(\d+) max_local_bytes=(\d+) "
    r"peak_local_bytes=(\d+) total_local_bytes=(\d+) host_blocks=(\d+)$",
    re.MULTILINE,
)
_P76_TICKET_RE = re.compile(
    r"^\[P76\.CHUNK_DEPENDENCY\] enabled=1 "
    r"leaves=(\d+) checked_vma=(\d+) "
    r"scalar_collectives=(\d+) host_transfers=(\d+)$",
    re.MULTILINE,
)
_P77_BACKPRESSURE_RE = re.compile(
    r"^\[P77\.CHUNK_BACKPRESSURE\] enabled=1 "
    r"pullback_waits=(\d+) accumulation_waits=(\d+) leaves=(\d+) "
    r"wait_api=(block_until_ready) host_transfers=(\d+)$",
    re.MULTILINE,
)
_KNOWN_POST_TRAINING_FINALIZER_RE = re.compile(
    r"Exception ignored in: <finalize object at 0x[0-9a-f]+; dead>\n"
    r"Traceback \(most recent call last\):\n"
    r"  File \"/usr/local/lib/python3\.12/weakref\.py\", line 590, in __call__\n"
    r"    return info\.func\(\*info\.args, \*\*\(info\.kwargs or \{\}\)\)\n"
    r" {11}\^{44}\n"
    r"  File \"/usr/local/lib/python3\.12/site-packages/vllm/v1/engine/llm_engine\.py\", line 441, in _cleanup_instance_caches\n"
    r"    for module in model\.modules\(\):\n"
    r" {18}\^{13}\n"
    r"AttributeError: 'Qwen3ForCausalLM' object has no attribute 'modules'\n"
)
_TRAINING_DONE = "[CANON_FROZENLAKE_P27] TRAINING_DONE"
_V2_CAPSULE_CAPTURE_RE = re.compile(
    r"^\[V2\.CAPSULE\] capture_ready path=(\S+) "
    r"sha256=([0-9a-f]{64}) rows=(\d+) arrays=(\d+) "
    r"logical_bytes=(\d+) certification=strict-prealignment-source$",
    re.MULTILINE,
)
_V2_CAPSULE_MODEL_RE = re.compile(
    r"^\[V2\.CAPSULE\] model_(bound|verified) "
    r"mode=(capture|replay) capsule_sha256=([0-9a-f]{64}) "
    r"binding_sha256=([0-9a-f]{64}) "
    r"model_fingerprint_sha256=([0-9a-f]{64}) sampled_model=1$",
    re.MULTILINE,
)
_V2_CAPSULE_REPLAY_RE = re.compile(
    r"^\[V2\.CAPSULE\] diagnostic_replay_ready path=(\S+) "
    r"sha256=([0-9a-f]{64}) rows=(\d+) "
    r"capture_run=(\S+) capture_source=([0-9a-f]{40}) "
    r"replay_source=([0-9a-f]{40}) "
    r"rollout=skipped rescore_b=skipped certification=0$",
    re.MULTILINE,
)
_V2_CAPSULE_PRODUCER_BYPASS = (
    "[V2.CAPSULE] producer_bypass verdict=PASS "
    "environment=0 rollout=0 rescore_b=0"
)
_REDUCE_ONCE_REPORT_ACCUMULATE_RE = re.compile(
    r"^\[V2\.REDUCE_ONCE\.REPORT_ACCUMULATE\] enabled=1 "
    r"group=(\d+)/(\d+) leaves=(\d+) buckets=(\d+) "
    r"executables=(\d+) "
    r"peak_local_bytes=(\d+) device_dependencies=(\d+) "
    r"host_transfers=(\d+)$",
    re.MULTILINE,
)
_REDUCE_ONCE_ACCUMULATOR_LOAN_RE = re.compile(
    r"^\[V2\.REDUCE_ONCE\.ACCUMULATOR_LOAN\] enabled=1 "
    r"leaves=(\d+) local_bytes=(\d+) transition=(\S+) "
    r"base_handles_retired=(\d+) staged_handles_retired=(\d+) "
    r"check_vma=(\d+) host_transfers=(\d+)$",
    re.MULTILINE,
)
_REDUCE_ONCE_ACCUMULATOR_RESET_RE = re.compile(
    r"^\[V2\.REDUCE_ONCE\.ACCUMULATOR_RESET\] enabled=1 "
    r"leaves=(\d+) transition=(\S+) alias_mode=(\S+) "
    r"host_transfers=(\d+)$",
    re.MULTILINE,
)
_HBM_OUTER_PREFIX = (
    "before_replay",
    "after_replay",
)
_HBM_OUTER_SUFFIX = (
    "after_model_backward",
    "after_report_adjoint",
    "after_reduce_compare",
    "after_sink_delete",
)
_REDUCER_MAX_LOCAL_BYTES = 2 * 1024**3
_QWEN8B_TP2_BUCKET_RECEIPT = {
    "programs": 8,
    "max_local_bytes": _REDUCER_MAX_LOCAL_BYTES,
    "peak_local_bytes": 2_130_875_392,
    "total_local_bytes": 16_382_087_168,
}
_ARMS = {
    "r0": {
        "keep_tape": "0",
        "reduce_once": "0",
        "length_sort": "0",
        "report_adjoint_buckets": "0",
        "chunk_dependency_ticket": "0",
        "chunk_backpressure": "0",
    },
    "r0b": {
        "keep_tape": "0",
        "reduce_once": "0",
        "length_sort": "0",
        "report_adjoint_buckets": "1",
        "chunk_dependency_ticket": "0",
        "chunk_backpressure": "0",
    },
    "r0c": {
        "keep_tape": "0",
        "reduce_once": "0",
        "length_sort": "0",
        "report_adjoint_buckets": "1",
        "chunk_dependency_ticket": "1",
        "chunk_backpressure": "0",
    },
    "r0d": {
        "keep_tape": "0",
        "reduce_once": "0",
        "length_sort": "0",
        "report_adjoint_buckets": "1",
        "chunk_dependency_ticket": "0",
        "chunk_backpressure": "1",
    },
    "r1": {
        "keep_tape": "stream",
        "reduce_once": "0",
        "length_sort": "0",
        "report_adjoint_buckets": "0",
        "chunk_dependency_ticket": "0",
        "chunk_backpressure": "0",
    },
    "r2": {
        "keep_tape": "stream",
        "reduce_once": "1",
        "length_sort": "0",
        "report_adjoint_buckets": "0",
        "chunk_dependency_ticket": "0",
        "chunk_backpressure": "0",
    },
    "r3": {
        "keep_tape": "stream",
        "reduce_once": "1",
        "length_sort": "1",
        "report_adjoint_buckets": "0",
        "chunk_dependency_ticket": "0",
        "chunk_backpressure": "0",
    },
}
_WORKLOADS = {
    "p45": {
        "name": "frozenlake-p45-onehost-dp2-tp2",
        "prompt": 4096,
        "response": 2048,
        "max_turns": 5,
        "vllm_hbm_utilization": 0.35,
        # The backward carrier dispatches fixed M=256 programs. Eight chunks
        # is the operation-count gate; requiring the final chunk to contain
        # all 256 real tokens does not exercise another program or allocation.
        "min_group_chunks": 8,
        "min_action_tokens": 1024,
    },
    "m15": {
        "name": "frozenlake-m15-onehost-dp2-tp2",
        "prompt": 4096,
        "response": 8192,
        "max_turns": 15,
        "vllm_hbm_utilization": 0.37,
        # This is a landed-work threshold, not a claim that the 12,288-token
        # static cap was reached.  The classifier reports cap_coverage
        # separately and never promotes a CLI cap to measured work.
        "min_landed_n_real": 4096,
        "min_action_tokens": 4096,
    },
}


def _remove_known_post_training_finalizer(raw: str) -> tuple[str, int]:
  """Removes only the exact vLLM/JAX weakref failure after successful work."""
  matches = list(_KNOWN_POST_TRAINING_FINALIZER_RE.finditer(raw))
  if len(matches) != 1:
    return raw, 0
  match = matches[0]
  if raw.rfind(_TRAINING_DONE, 0, match.start()) < 0:
    return raw, 0
  return raw[:match.start()] + raw[match.end():], 1


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected JSON object: {path}")
  return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
      continue
    value = json.loads(line)
    if not isinstance(value, dict):
      raise ValueError(f"expected JSON object at {path}:{number}")
    rows.append(value)
  if not rows:
    raise ValueError(f"empty JSONL: {path}")
  return rows


def _exact_boundaries(rows: list[dict[str, Any]]) -> bool:
  if not rows:
    return False
  for row in rows:
    if row.get("verdict") != "PASS":
      return False
    if row.get("blocking_reds", []) or row.get("reds", []):
      return False
    boundaries = row.get("boundaries")
    if not isinstance(boundaries, dict) or not boundaries:
      return False
    for boundary in boundaries.values():
      if not isinstance(boundary, dict):
        return False
      if (
          boundary.get("valid") is not True
          or boundary.get("finite") is not True
          or boundary.get("differing_bytes") != 0
          or boundary.get("differing_elements") != 0
          or boundary.get("max_abs") != 0.0
      ):
        return False
  return True


def _peak_hbm(update: dict[str, Any]) -> tuple[int | None, int | None]:
  peaks = []
  limits = []
  for field in ("hbm_before", "hbm_after_reverse"):
    for snapshot in update.get(field, []):
      if not isinstance(snapshot, dict):
        continue
      peak = snapshot.get("peak_bytes_in_use")
      limit = snapshot.get("bytes_limit")
      if isinstance(peak, int):
        peaks.append(peak)
      if isinstance(limit, int):
        limits.append(limit)
  return (max(peaks) if peaks else None, min(limits) if limits else None)


def _forward_lengths(text: str) -> tuple[list[int], list[int], str | None]:
  matches = list(_FORWARD_RE.finditer(text))
  groups = []
  lengths = []
  marker = None
  for match in matches:
    this_marker = match.group(1)
    marker = this_marker if marker is None else marker
    if marker != this_marker:
      raise ValueError("mixed forward_group_issued/forward_group_done markers")
    groups.append(int(match.group(2)))
    fields = [field.strip() for field in match.group(3).split(",")]
    if len(fields) != 2:
      raise ValueError("DP2 forward receipt does not contain two rank lengths")
    lengths.extend(int(field) for field in fields)
  return groups, lengths, marker


def _hbm_stage_receipts(text: str) -> list[dict[str, Any]]:
  receipts = []
  for line in text.splitlines():
    if line.startswith(_HBM_STAGE_PREFIX):
      value = json.loads(line[len(_HBM_STAGE_PREFIX):])
      if not isinstance(value, dict):
        raise ValueError("HBM stage receipt is not a JSON object")
      receipts.append(value)
  return receipts


def _report_adjoint_memory_receipts(text: str) -> list[dict[str, Any]]:
  receipts = []
  for line in text.splitlines():
    if line.startswith(_REPORT_ADJOINT_MEMORY_PREFIX):
      value = json.loads(line[len(_REPORT_ADJOINT_MEMORY_PREFIX):])
      if not isinstance(value, dict):
        raise ValueError("report-adjoint memory receipt is not a JSON object")
      receipts.append(value)
  return receipts


def _valid_report_adjoint_memory_receipt(receipt: dict[str, Any]) -> bool:
  byte_fields = (
      "argument_size_in_bytes",
      "output_size_in_bytes",
      "alias_size_in_bytes",
      "temp_size_in_bytes",
      "host_argument_size_in_bytes",
      "host_output_size_in_bytes",
      "host_alias_size_in_bytes",
      "host_temp_size_in_bytes",
  )
  if (
      receipt.get("schema") != "canon-v2-p59-report-adjoint-memory-v1"
      or not isinstance(receipt.get("staged_engine_leaves"), int)
      or receipt["staged_engine_leaves"] <= 0
      or not isinstance(receipt.get("trainer_leaves"), int)
      or receipt["trainer_leaves"] <= 0
      or any(
          not isinstance(receipt.get(field), int) or receipt[field] < 0
          for field in byte_fields
      )
      or receipt["argument_size_in_bytes"] <= 0
      or receipt["output_size_in_bytes"] <= 0
  ):
    return False
  return receipt["alias_size_in_bytes"] <= min(
      receipt["argument_size_in_bytes"], receipt["output_size_in_bytes"]
  )


def _valid_bucketed_report_memory_receipt(receipt: dict[str, Any]) -> bool:
  buckets = receipt.get("buckets")
  if (
      receipt.get("schema")
      != "canon-v2-p75-report-adjoint-buckets-memory-v1"
      or not isinstance(receipt.get("source_leaves"), int)
      or receipt["source_leaves"] <= 0
      or not isinstance(receipt.get("target_leaves"), int)
      or receipt["target_leaves"] <= 0
      or not isinstance(receipt.get("bucket_count"), int)
      or receipt["bucket_count"] <= 1
      or receipt.get("max_local_bytes") != _REDUCER_MAX_LOCAL_BYTES
      or receipt.get("host_blocks") != receipt["bucket_count"]
      or not isinstance(buckets, list)
      or len(buckets) != receipt["bucket_count"]
  ):
    return False
  byte_fields = (
      "argument_size_in_bytes",
      "output_size_in_bytes",
      "alias_size_in_bytes",
      "temp_size_in_bytes",
      "host_argument_size_in_bytes",
      "host_output_size_in_bytes",
      "host_alias_size_in_bytes",
      "host_temp_size_in_bytes",
  )
  for index, bucket in enumerate(buckets):
    if (
        not isinstance(bucket, dict)
        or bucket.get("bucket") != index
        or not isinstance(bucket.get("source_leaves"), int)
        or bucket["source_leaves"] <= 0
        or not isinstance(bucket.get("target_leaves"), int)
        or bucket["target_leaves"] <= 0
        or not isinstance(bucket.get("local_output_bytes"), int)
        or not 0 < bucket["local_output_bytes"] <= _REDUCER_MAX_LOCAL_BYTES
        or any(
            not isinstance(bucket.get(field), int) or bucket[field] < 0
            for field in byte_fields
        )
        or bucket["argument_size_in_bytes"] <= 0
        or bucket["output_size_in_bytes"] <= 0
        or bucket["alias_size_in_bytes"]
        > min(
            bucket["argument_size_in_bytes"],
            bucket["output_size_in_bytes"],
        )
        or any(bucket[field] != 0 for field in byte_fields[4:])
    ):
      return False
  local_bytes = [bucket["local_output_bytes"] for bucket in buckets]
  return (
      sum(bucket["source_leaves"] for bucket in buckets)
      == receipt["source_leaves"]
      and sum(bucket["target_leaves"] for bucket in buckets)
      == receipt["target_leaves"]
      and max(local_bytes) == receipt.get("peak_local_output_bytes")
      and sum(local_bytes) == receipt.get("total_local_output_bytes")
  )


def _expected_hbm_stages(num_chunks: int) -> tuple[str, ...]:
  chunk_stages = []
  for chunk in reversed(range(num_chunks)):
    chunk_stages.extend((
        f"model_before_chunk_{chunk}",
        f"model_after_pullbacks_chunk_{chunk}",
        f"model_after_accumulate_chunk_{chunk}",
    ))
  return _HBM_OUTER_PREFIX + tuple(chunk_stages) + _HBM_OUTER_SUFFIX


def _valid_hbm_stage_receipts(
    receipts: list[dict[str, Any]], *, group_index: int, num_chunks: int
) -> bool:
  if [item.get("stage") for item in receipts] != list(
      _expected_hbm_stages(num_chunks)
  ):
    return False
  previous_peaks = None
  for item in receipts:
    if item.get("group") != group_index:
      return False
    devices = item.get("devices")
    if not isinstance(devices, list) or len(devices) != 4:
      return False
    if [device.get("device") for device in devices] != list(range(4)):
      return False
    peaks = []
    for device in devices:
      current = device.get("bytes_in_use")
      peak = device.get("peak_bytes_in_use")
      limit = device.get("bytes_limit")
      if (
          not isinstance(current, int)
          or not isinstance(peak, int)
          or not isinstance(limit, int)
          or current < 0
          or current > peak
          or peak > limit
      ):
        return False
      peaks.append(peak)
    if previous_peaks is not None and any(
        current < previous
        for current, previous in zip(peaks, previous_peaks, strict=True)
    ):
      return False
    previous_peaks = peaks
  return True


def _valid_p75_bucket_hbm_receipts(
    receipts: list[dict[str, Any]], *, group_indices: tuple[int, ...]
) -> bool:
  expected = [
      (group, bucket, stage)
      for group in group_indices
      for bucket in range(_QWEN8B_TP2_BUCKET_RECEIPT["programs"])
      for stage in ("before_execute", "after_execute", "after_delete")
  ]
  actual = []
  previous_peaks = None
  per_bucket = {}
  for item in receipts:
    match = _P75_BUCKET_HBM_RE.fullmatch(str(item.get("stage", "")))
    if match is None:
      return False
    group = item.get("group")
    bucket = int(match.group(1))
    stage = match.group(2)
    actual.append((group, bucket, stage))
    devices = item.get("devices")
    if not isinstance(group, int) or not isinstance(devices, list):
      return False
    if len(devices) != 4:
      return False
    if [device.get("device") for device in devices] != list(range(4)):
      return False
    currents = []
    peaks = []
    for device in devices:
      current = device.get("bytes_in_use")
      peak = device.get("peak_bytes_in_use")
      limit = device.get("bytes_limit")
      if (
          not isinstance(current, int)
          or not isinstance(peak, int)
          or not isinstance(limit, int)
          or current < 0
          or current > peak
          or peak > limit
      ):
        return False
      currents.append(current)
      peaks.append(peak)
    if previous_peaks is not None and any(
        current < previous
        for current, previous in zip(peaks, previous_peaks, strict=True)
    ):
      return False
    previous_peaks = peaks
    per_bucket[(group, bucket, stage)] = currents
  if actual != expected:
    return False
  for group in group_indices:
    for bucket in range(_QWEN8B_TP2_BUCKET_RECEIPT["programs"]):
      before = per_bucket[(group, bucket, "before_execute")]
      executed = per_bucket[(group, bucket, "after_execute")]
      deleted = per_bucket[(group, bucket, "after_delete")]
      if any(
          not (before_value <= executed_value and deleted_value <= executed_value)
          for before_value, executed_value, deleted_value in zip(
              before, executed, deleted, strict=True
          )
      ):
        return False
  return True


def _valid_p75_group_hbm_receipts(receipts: list[dict[str, Any]]) -> bool:
  if [item.get("group") for item in receipts] != list(range(8)):
    return False
  previous_peaks = None
  for item in receipts:
    if item.get("stage") != _P75_GROUP_HBM_STAGE:
      return False
    devices = item.get("devices")
    if not isinstance(devices, list) or len(devices) != 4:
      return False
    if [device.get("device") for device in devices] != list(range(4)):
      return False
    peaks = []
    for device in devices:
      current = device.get("bytes_in_use")
      peak = device.get("peak_bytes_in_use")
      limit = device.get("bytes_limit")
      if (
          not isinstance(current, int)
          or not isinstance(peak, int)
          or not isinstance(limit, int)
          or current < 0
          or current > peak
          or peak > limit
      ):
        return False
      peaks.append(peak)
    if previous_peaks is not None and any(
        current < previous
        for current, previous in zip(peaks, previous_peaks, strict=True)
    ):
      return False
    previous_peaks = peaks
  return True


def _anchor(
    registry: dict[str, Any], workload: str, arm: str
) -> tuple[list[float] | None, float | None, str | None, str | None]:
  if registry.get("schema") != "canon.v2-frozenlake-onehost.gradient-anchors.v1":
    raise ValueError("gradient anchor registry schema changed")
  anchors = registry.get("anchors")
  if not isinstance(anchors, dict):
    raise ValueError("gradient anchor registry has no anchors object")
  entry = anchors.get(f"{workload}:{arm}")
  if entry is None:
    return None, None, None, None
  if not isinstance(entry, dict):
    raise ValueError("gradient anchor entry is not an object")
  norms = entry.get("micro_gradient_norms")
  update_norm = entry.get("update_gradient_norm")
  run_id = entry.get("run_id")
  capsule_sha = entry.get("training_capsule_sha256")
  if (
      not isinstance(norms, list)
      or len(norms) != 8
      or any(not isinstance(value, (int, float)) for value in norms)
      or not isinstance(run_id, str)
      or not run_id
      or (
          update_norm is not None
          and not isinstance(update_norm, (int, float))
      )
  ):
    raise ValueError("gradient anchor entry is incomplete")
  if capsule_sha is not None and _SHA256_RE.fullmatch(str(capsule_sha)) is None:
    raise ValueError("gradient anchor capsule SHA is invalid")
  if (arm in ("r2", "r3")) != (update_norm is not None):
    raise ValueError(
        "gradient anchor update norm does not match reduce-once arm"
    )
  return (
      [float(value) for value in norms],
      None if update_norm is None else float(update_norm),
      run_id,
      capsule_sha,
  )


def classify(
    root: Path,
    *,
    workload: str,
    arm: str,
    docker_exit: int,
    anchor_registry: Path,
    require_anchor: bool = False,
) -> dict[str, Any]:
  if workload not in _WORKLOADS:
    raise ValueError(f"unknown workload {workload!r}")
  if arm not in _ARMS:
    raise ValueError(f"unknown arm {arm!r}")
  spec = _WORKLOADS[workload]
  arm_spec = _ARMS[arm]
  required = {
      name: root / name
      for name in (
          "raw.log",
          "run_manifest.json",
          "runtime.json",
          "pre_alignment.jsonl",
          "alignment.jsonl",
          "updates.json",
      )
  }
  missing = [
      name
      for name, path in required.items()
      if not path.is_file() or path.stat().st_size == 0
  ]
  if missing:
    raw = ""
    if required["raw.log"].is_file():
      raw = required["raw.log"].read_text(
          encoding="utf-8", errors="replace"
      )
    reasons = [f"missing_or_empty:{name}" for name in missing]
    if (
        "RuntimeProgramAllocationFailure" in raw
        and "jit_reduce_local" in raw
    ):
      reasons.append("runtime_hbm_exhausted:jit_reduce_local")
    return {
        "schema": "canon.v2-frozenlake-onehost.classification.v1",
        "verdict": "INCONCLUSIVE",
        "workload": workload,
        "arm": arm,
        "reasons": reasons,
    }

  raw = required["raw.log"].read_text(encoding="utf-8", errors="replace")
  manifest = _json(required["run_manifest.json"])
  runtime = _json(required["runtime.json"])
  pre = _jsonl(required["pre_alignment.jsonl"])
  alignment = _jsonl(required["alignment.jsonl"])
  update = _json(required["updates.json"])
  reasons = []

  def require(condition: bool, reason: str) -> None:
    if not condition:
      reasons.append(reason)

  expected_manifest = {
      "schema": "canon.v2-frozenlake-onehost.run.v1",
      "workload": workload,
      "workload_name": spec["name"],
      "arm": arm,
      "stage": "backward-no-commit",
      "model_id": "Qwen/Qwen3-8B",
      "model_dir_name": "qwen8b_tp2",
      "topology": {"dp": 2, "tp": 2, "devices": 4},
      "global_prompts": 4,
      "num_generations": 4,
      "global_trajectories": 16,
      "gradient_groups": 8,
      "local_m": 256,
      "global_m": 512,
      "max_prompt_length": spec["prompt"],
      "max_response_length": spec["response"],
      "max_turns": spec["max_turns"],
      "data_shuffle_seed": 42,
      "vllm_global_seed": 0,
      "vllm_hbm_utilization": spec["vllm_hbm_utilization"],
      "selectors": arm_spec,
      "reducer_schedule": {
          "kind": "fixed-local-byte-buckets",
          "max_local_bytes": _REDUCER_MAX_LOCAL_BYTES,
      },
      "checked_vma": True,
      "wandb_mode": "disabled",
      "classification_mode": "certify" if require_anchor else "measure",
      "hbm_stage_diagnostic": (
          not require_anchor and arm in ("r0", "r0b")
      ),
  }
  changed = {
      key: manifest.get(key)
      for key, expected in expected_manifest.items()
      if manifest.get(key) != expected
  }
  require(not changed, f"manifest:{changed}")
  capsule_manifest = manifest.get("training_capsule")
  if capsule_manifest is None:
    capsule_manifest = {
        "mode": "none",
        "capture_run": None,
        "sha256": None,
        "model_binding_sha256": None,
    }
  valid_capsule_manifest = (
      isinstance(capsule_manifest, dict)
      and capsule_manifest.get("mode") in ("none", "capture", "replay")
      and set(capsule_manifest)
      == {"mode", "capture_run", "sha256", "model_binding_sha256"}
  )
  require(valid_capsule_manifest, f"training_capsule={capsule_manifest}")
  capsule_values = (
      capsule_manifest if isinstance(capsule_manifest, dict) else {}
  )
  capsule_mode = (
      capsule_values.get("mode") if valid_capsule_manifest else "invalid"
  )
  if capsule_mode == "none":
    require(
        all(
            capsule_values.get(name) is None
            for name in ("capture_run", "sha256", "model_binding_sha256")
        ),
        f"training_capsule_none={capsule_values}",
    )
  elif capsule_mode == "capture":
    require(not require_anchor, "capture_requires_measure_mode")
    require(
        capsule_values.get("capture_run") is None
        and capsule_values.get("sha256") is None
        and capsule_values.get("model_binding_sha256") is None,
        f"training_capsule_capture_manifest={capsule_values}",
    )
  elif capsule_mode == "replay":
    require(require_anchor, "replay_requires_certify_mode")
    require(
        isinstance(capsule_values.get("capture_run"), str)
        and bool(
            re.fullmatch(
                r"[a-z0-9][a-z0-9_-]+", capsule_values["capture_run"]
            )
        )
        and _SHA256_RE.fullmatch(str(capsule_values.get("sha256")))
        is not None
        and _SHA256_RE.fullmatch(
            str(capsule_values.get("model_binding_sha256"))
        )
        is not None,
        f"training_capsule_replay_manifest={capsule_values}",
    )
  if require_anchor:
    require(
        capsule_mode == "replay",
        "certify_requires_training_capsule_replay",
    )
  require(
      arm not in ("r0b", "r0c", "r0d") or workload == "p45",
      f"p75_p76_p77_workload={workload}",
  )
  require(_SOURCE_RE.fullmatch(str(manifest.get("source_commit"))) is not None,
          "manifest.source_commit")
  for key in (
      "source_diff_sha256",
      "image_id_sha256",
      "model_snapshot_sha256",
      "dataset_train_sha256",
      "dataset_test_sha256",
      "runner_sha256",
  ):
    require(_SHA256_RE.fullmatch(str(manifest.get(key))) is not None,
            f"manifest.{key}")

  require(docker_exit == 0, f"docker_exit={docker_exit}")
  require(runtime.get("verdict") == "PASS", "runtime")
  require(_exact_boundaries(pre), "pre_alignment_not_exact")
  require(_exact_boundaries(alignment), "alignment_not_exact")
  require(len(pre) == 1, f"pre_alignment_records={len(pre)}")
  require(len(alignment) == 8, f"alignment_records={len(alignment)}")
  require(
      "S_decode_vs_S_prefill" in pre[0].get("boundaries", {})
      and "S_prefill_vs_T_old" in pre[0].get("boundaries", {}),
      "pre_alignment_boundary_inventory",
  )
  action_tokens = pre[0].get("N_action")
  input_hashes = pre[0].get("hashes")
  valid_input_hashes = (
      isinstance(input_hashes, dict)
      and all(
          _SHA256_RE.fullmatch(str(input_hashes.get(key))) is not None
          for key in (
              "tokens",
              "action_mask",
              "S_decode",
              "S_prefill",
              "T_old",
          )
      )
  )
  require(valid_input_hashes, "input_hash_inventory")
  require(
      isinstance(action_tokens, int)
      and action_tokens >= spec["min_action_tokens"],
      f"landed_action_tokens={action_tokens}",
  )

  expected_update = {
      "contract_name": spec["name"],
      "dp_size": 2,
      "tp_size": 2,
      "global_m": 512,
      "verdict": "PASS",
      "mode": "backward-no-commit",
      "microsteps": 8,
      "commits": 0,
      "train_steps_before": 0,
      "train_steps_after": 0,
      "gradient_finite": True,
      "dp_replicas_exact": True,
      "dp_axis": "dp",
      "dp_reduction_transactions": 1 if arm in ("r2", "r3") else 8,
      # Fixed DPRank reduction is one ordered reduce plus one broadcast.
      "dp_reduction_rounds_per_transaction": 2,
      "dp_rank_pullbacks_per_transaction": 2,
      "dp_pullback_invocations_per_transaction": 1,
      "model_changed_paths": [],
      "optimizer_changed_paths": [],
      "accumulator_changed_paths": [],
      "reference_changed_paths": [],
      "optimizer_placement": "pinned-host-offload",
      "optimizer_memory_kinds_before": ["pinned_host"],
  }
  wrong_update = {
      key: update.get(key)
      for key, expected in expected_update.items()
      if update.get(key) != expected
  }
  require(not wrong_update, f"update:{wrong_update}")
  activity = update.get("gradient_activity")
  norms = update.get("micro_gradient_norms")
  valid_norms = (
      isinstance(activity, list)
      and len(activity) == 8
      and all(isinstance(value, bool) for value in activity)
      and any(activity)
      and isinstance(norms, list)
      and len(norms) == 8
      and all(
          isinstance(value, (int, float)) and math.isfinite(value)
          for value in norms
      )
      and all((float(value) > 0.0) == active for value, active in zip(norms, activity))
  )
  require(valid_norms, "gradient_activity_or_norms")
  update_gradient_norm = update.get("update_gradient_norm")
  if arm in ("r2", "r3"):
    require(
        isinstance(update_gradient_norm, (int, float))
        and math.isfinite(update_gradient_norm)
        and update_gradient_norm > 0.0,
        f"update_gradient_norm={update_gradient_norm}",
    )
  else:
    require(
        update_gradient_norm is None,
        "unexpected_update_gradient_norm",
    )

  try:
    groups, n_real, forward_marker = _forward_lengths(raw)
  except ValueError as exc:
    groups, n_real, forward_marker = [], [], None
    reasons.append(f"forward_receipts:{exc}")
  require(groups == list(range(1, 9)), f"forward_groups={groups}")
  require(len(n_real) == 16, f"n_real_count={len(n_real)}")
  expected_marker = (
      "forward_group_done"
      if arm in ("r0", "r0b", "r0c", "r0d")
      else "forward_group_issued"
  )
  require(forward_marker == expected_marker, f"forward_marker={forward_marker}")
  max_n_real = max(n_real) if n_real else 0
  max_static = spec["prompt"] + spec["response"]
  require(max_n_real <= max_static, f"n_real_over_cap={max_n_real}>{max_static}")
  group_chunks = [
      (max(n_real[index:index + 2]) + 255) // 256
      for index in range(0, len(n_real), 2)
  ] if len(n_real) == 16 else []
  if "min_group_chunks" in spec:
    max_group_chunks = max(group_chunks, default=0)
    require(
        max_group_chunks >= spec["min_group_chunks"],
        f"landed_group_chunks_max={max_group_chunks}",
    )
  else:
    require(
        max_n_real >= spec["min_landed_n_real"],
        f"landed_n_real_max={max_n_real}",
    )

  p66_receipts = raw.count("[P66.VMA] outer_check_enabled")
  require(p66_receipts >= 1, f"p66_outer_check_receipts={p66_receipts}")
  seed_receipt = raw.count(
      "[V2.FL.SEED] CONTRACT_PASS data_shuffle_seed=42 "
      "vllm_global_seed=0 per_request_seed=unsupported"
  )
  require(seed_receipt == 1, f"seed_receipts={seed_receipt}")
  disabled_wandb_receipt = raw.count("[V2.FL.WANDB] DISABLED_LOCAL_PASS")
  require(
      disabled_wandb_receipt == 1,
      f"disabled_wandb_receipts={disabled_wandb_receipt}",
  )
  sampler_receipt = raw.count(
      "[V2.FL.SAMPLER] CONTRACT_PASS sampler_is=none "
      "use_rollout_logps=1 tis_weights=absent"
  )
  expected_sampler_receipts = 0 if capsule_mode == "replay" else 1
  require(
      sampler_receipt == expected_sampler_receipts,
      f"sampler_receipts={sampler_receipt};expected={expected_sampler_receipts}",
  )
  bucket_matches = list(_BUCKET_RE.finditer(raw))
  bucket_receipt = None
  if len(bucket_matches) == 1:
    bucket_receipt = {
        "programs": int(bucket_matches[0].group(1)),
        "max_local_bytes": int(bucket_matches[0].group(2)),
        "peak_local_bytes": int(bucket_matches[0].group(3)),
        "total_local_bytes": int(bucket_matches[0].group(4)),
    }
  require(
      len(bucket_matches) == 1,
      f"reducer_bucket_receipts={len(bucket_matches)}",
  )
  require(
      bucket_receipt == _QWEN8B_TP2_BUCKET_RECEIPT,
      f"reducer_bucket_schedule={bucket_receipt}",
  )
  report_accumulate_receipts = [
      tuple(int(value) for value in match)
      for match in _REDUCE_ONCE_REPORT_ACCUMULATE_RE.findall(raw)
  ]
  if arm in ("r2", "r3"):
    require(
        report_accumulate_receipts
        == [
            (group, 8, 399, 8, 1, 2130875392, 7, 0)
            for group in range(2, 9)
        ],
        f"reduce_once_report_accumulate={report_accumulate_receipts}",
    )
  else:
    require(
        not report_accumulate_receipts,
        "unexpected_reduce_once_report_accumulate",
    )
  accumulator_loan_receipts = [
      (
          int(match[0]),
          int(match[1]),
          match[2],
          int(match[3]),
          int(match[4]),
          int(match[5]),
          int(match[6]),
      )
      for match in _REDUCE_ONCE_ACCUMULATOR_LOAN_RE.findall(raw)
  ]
  if arm in ("r2", "r3"):
    require(
        accumulator_loan_receipts
        == [(399, 16382087168, "base-to-staged", 399, 399, 1, 0)],
        f"reduce_once_accumulator_loan={accumulator_loan_receipts}",
    )
  else:
    require(
        not accumulator_loan_receipts,
        "unexpected_reduce_once_accumulator_loan",
    )
  accumulator_reset_receipts = [
      (int(match[0]), match[1], match[2], int(match[3]))
      for match in _REDUCE_ONCE_ACCUMULATOR_RESET_RE.findall(raw)
  ]
  if arm in ("r2", "r3"):
    require(
        accumulator_reset_receipts
        == [(399, "adopted-to-idle", "bitwise-zero", 0)],
        f"reduce_once_accumulator_reset={accumulator_reset_receipts}",
    )
  else:
    require(
        not accumulator_reset_receipts,
        "unexpected_reduce_once_accumulator_reset",
    )
  try:
    raw_hbm_stages = _hbm_stage_receipts(raw)
  except (json.JSONDecodeError, ValueError) as exc:
    raw_hbm_stages = []
    reasons.append(f"hbm_stage_receipts:{exc}")
  try:
    report_adjoint_memory = _report_adjoint_memory_receipts(raw)
  except (json.JSONDecodeError, ValueError) as exc:
    report_adjoint_memory = []
    reasons.append(f"report_adjoint_memory_receipt:{exc}")
  p75_bucket_matches = [
      tuple(int(value) for value in match)
      for match in _P75_BUCKET_RE.findall(raw)
  ]
  if arm in ("r0b", "r0c", "r0d"):
    require(
        len(p75_bucket_matches) == 8
        and len(set(p75_bucket_matches)) == 1
        and p75_bucket_matches[0][0] > 1
        and p75_bucket_matches[0][1] == _REDUCER_MAX_LOCAL_BYTES
        and 0 < p75_bucket_matches[0][2] <= _REDUCER_MAX_LOCAL_BYTES
        and p75_bucket_matches[0][3]
        == _QWEN8B_TP2_BUCKET_RECEIPT["total_local_bytes"]
        and p75_bucket_matches[0][4] == p75_bucket_matches[0][0],
        f"p75_report_adjoint_bucket_receipts={len(p75_bucket_matches)}",
    )
  else:
    require(
        not p75_bucket_matches,
        "unexpected_p75_report_adjoint_bucket_receipt",
    )
  p76_ticket_matches = [
      tuple(int(value) for value in match)
      for match in _P76_TICKET_RE.findall(raw)
  ]
  if arm == "r0c":
    require(
        len(p76_ticket_matches) == 1
        and p76_ticket_matches[0][0] > 0
        and p76_ticket_matches[0][1:] == (1, 1, 0),
        f"p76_chunk_dependency_receipts={p76_ticket_matches}",
    )
  else:
    require(
        not p76_ticket_matches,
        "unexpected_p76_chunk_dependency_receipt",
    )
  p77_backpressure_matches = [
      (
          int(pullback_waits),
          int(accumulation_waits),
          int(leaves),
          wait_api,
          int(host_transfers),
      )
      for (
          pullback_waits,
          accumulation_waits,
          leaves,
          wait_api,
          host_transfers,
      ) in _P77_BACKPRESSURE_RE.findall(raw)
  ]
  if arm == "r0d":
    require(
        len(p77_backpressure_matches) == 8
        and [item[0] for item in p77_backpressure_matches] == group_chunks
        and [item[1] for item in p77_backpressure_matches] == group_chunks
        and all(item[2] > 0 for item in p77_backpressure_matches)
        and all(
            item[3:] == ("block_until_ready", 0)
            for item in p77_backpressure_matches
        ),
        f"p77_chunk_backpressure_receipts={p77_backpressure_matches}",
    )
  else:
    require(
        not p77_backpressure_matches,
        "unexpected_p77_chunk_backpressure_receipt",
    )
  update_hbm_stages = update.get("hbm_stage_receipts")
  if expected_manifest["hbm_stage_diagnostic"]:
    hbm_stage_group = (
        1
        if arm == "r0b" and len(group_chunks) == 8
        else group_chunks.index(max(group_chunks)) if group_chunks else None
    )
    outer_hbm_stages = [
        item
        for item in raw_hbm_stages
        if _P75_BUCKET_HBM_RE.fullmatch(str(item.get("stage", ""))) is None
        and item.get("stage") != _P75_GROUP_HBM_STAGE
    ]
    bucket_hbm_stages = [
        item
        for item in raw_hbm_stages
        if _P75_BUCKET_HBM_RE.fullmatch(str(item.get("stage", ""))) is not None
    ]
    group_hbm_stages = [
        item
        for item in raw_hbm_stages
        if item.get("stage") == _P75_GROUP_HBM_STAGE
    ]
    require(
        len(raw_hbm_stages)
        == len(outer_hbm_stages)
        + len(bucket_hbm_stages)
        + len(group_hbm_stages),
        "unknown_hbm_stage_receipts",
    )
    require(
        isinstance(update_hbm_stages, list)
        and raw_hbm_stages == update_hbm_stages
        and len(group_chunks) == 8
        and _valid_hbm_stage_receipts(
            outer_hbm_stages,
            group_index=hbm_stage_group,
            num_chunks=group_chunks[hbm_stage_group],
        ),
        "hbm_stage_receipts",
    )
    if arm == "r0b":
      observer_groups = tuple(dict.fromkeys((0, hbm_stage_group)))
      require(
          _valid_p75_bucket_hbm_receipts(
              bucket_hbm_stages, group_indices=observer_groups
          ),
          "p75_bucket_hbm_receipts",
      )
      require(
          _valid_p75_group_hbm_receipts(group_hbm_stages),
          "p75_group_hbm_receipts",
      )
      require(
          len(report_adjoint_memory) == 1
          and _valid_bucketed_report_memory_receipt(
              report_adjoint_memory[0]
          ),
          f"bucketed_report_memory_receipts={len(report_adjoint_memory)}",
      )
    else:
      require(not bucket_hbm_stages, "unexpected_p75_bucket_hbm_receipts")
      require(not group_hbm_stages, "unexpected_p75_group_hbm_receipts")
      require(
          len(report_adjoint_memory) == 1
          and _valid_report_adjoint_memory_receipt(report_adjoint_memory[0]),
          f"report_adjoint_memory_receipts={len(report_adjoint_memory)}",
      )
  else:
    require(
        not raw_hbm_stages and update_hbm_stages is None,
        "unexpected_hbm_stage_receipts",
    )
    require(
        not report_adjoint_memory,
        "unexpected_report_adjoint_memory_receipt",
    )
  traceback_scan, ignored_post_training_finalizers = (
      _remove_known_post_training_finalizer(raw)
  )
  require(
      "Traceback (most recent call last)" not in traceback_scan,
      "traceback",
  )
  require("NUMERICAL_REJECT" not in raw, "numerical_reject")

  capsule_receipt = None
  capsule_models = _V2_CAPSULE_MODEL_RE.findall(raw)
  if capsule_mode == "capture":
    captures = _V2_CAPSULE_CAPTURE_RE.findall(raw)
    capsule_path = root / "training_capsule.npz"
    binding_path = root / "training_capsule.npz.model.json"
    capsule_sha = _sha256(capsule_path) if capsule_path.is_file() else None
    binding_sha = _sha256(binding_path) if binding_path.is_file() else None
    require(
        len(captures) == 1
        and captures[0][0] == str(capsule_path)
        and int(captures[0][2]) == 16
        and int(captures[0][3]) > 0
        and int(captures[0][4]) > 0
        and captures[0][1] == capsule_sha,
        f"training_capsule_capture_receipts={captures}",
    )
    require(
        len(capsule_models) == 1
        and capsule_models[0][0:2] == ("bound", "capture")
        and capsule_models[0][2] == capsule_sha
        and capsule_models[0][3] == binding_sha,
        f"training_capsule_model_receipts={capsule_models}",
    )
    capsule_receipt = {
        "mode": "capture",
        "sha256": capsule_sha,
        "model_binding_sha256": binding_sha,
    }
  elif capsule_mode == "replay":
    replays = _V2_CAPSULE_REPLAY_RE.findall(raw)
    expected_capsule_sha = capsule_values.get("sha256")
    expected_binding_sha = capsule_values.get("model_binding_sha256")
    require(
        len(replays) == 1
        and int(replays[0][2]) == 16
        and replays[0][1] == expected_capsule_sha
        and replays[0][3] == capsule_values.get("capture_run")
        and replays[0][5] == manifest.get("source_commit"),
        f"training_capsule_replay_receipts={replays}",
    )
    require(
        raw.count(_V2_CAPSULE_PRODUCER_BYPASS) == 1,
        "training_capsule_producer_bypass",
    )
    require(
        len(capsule_models) == 1
        and capsule_models[0][0:2] == ("verified", "replay")
        and capsule_models[0][2] == expected_capsule_sha
        and capsule_models[0][3] == expected_binding_sha,
        f"training_capsule_model_receipts={capsule_models}",
    )
    capsule_receipt = dict(capsule_values)
  else:
    require(
        not _V2_CAPSULE_CAPTURE_RE.search(raw)
        and not _V2_CAPSULE_REPLAY_RE.search(raw)
        and not capsule_models
        and _V2_CAPSULE_PRODUCER_BYPASS not in raw,
        "unexpected_training_capsule_receipt",
    )

  peak_hbm, hbm_limit = _peak_hbm(update)
  hbm_ratio = (
      peak_hbm / hbm_limit
      if isinstance(peak_hbm, int) and isinstance(hbm_limit, int) and hbm_limit
      else None
  )
  require(hbm_ratio is not None, "hbm_receipt")
  require(hbm_ratio is not None and hbm_ratio <= 0.90,
          f"hbm_headroom_ratio={hbm_ratio}")

  (
      anchor_norms,
      anchor_update_norm,
      anchor_run_id,
      anchor_capsule_sha,
  ) = _anchor(
      _json(anchor_registry), workload, arm
  )
  anchor_exact = (
      anchor_norms is not None
      and norms == anchor_norms
      and (
          update_gradient_norm == anchor_update_norm
          if arm in ("r2", "r3")
          else anchor_update_norm is None
      )
  )
  if require_anchor and anchor_norms is None:
    reasons.append("gradient_anchor_unregistered")
  if anchor_norms is not None:
    require(anchor_exact, "gradient_anchor_bitwise")
  if require_anchor and anchor_norms is not None:
    require(
        anchor_capsule_sha == capsule_values.get("sha256"),
        "gradient_anchor_capsule_sha",
    )
    require(
        anchor_run_id == capsule_values.get("capture_run"),
        "gradient_anchor_capture_run",
    )

  verdict = (
      "FAIL"
      if reasons
      else "PASS"
      if require_anchor and anchor_norms is not None and capsule_mode == "replay"
      else "MEASUREMENT_ONLY"
  )
  return {
      "schema": "canon.v2-frozenlake-onehost.classification.v1",
      "verdict": verdict,
      "workload": workload,
      "arm": arm,
      "claim_level": "onehost-qwen8b-dp2-tp2-no-commit",
      "classification_mode": "certify" if require_anchor else "measure",
      "claim_excludes": [
          "DP8xTP8",
          "Pathways",
          "GKE",
          "optimizer-commit",
          "convergence",
          "target-performance",
      ],
      "reasons": reasons,
      "zero_tim": {
          "pre_alignment_records": len(pre),
          "post_alignment_records": len(alignment),
          "strict_exact": _exact_boundaries(pre + alignment),
          "input_hashes": input_hashes,
          "action_tokens": action_tokens,
      },
      "gradient": {
          "micro_gradient_norms": norms,
          "update_gradient_norm": update_gradient_norm,
          "activity": activity,
          "anchor_registered": anchor_norms is not None,
          "anchor_exact": anchor_exact,
          "anchor_run_id": anchor_run_id,
          "anchor_training_capsule_sha256": anchor_capsule_sha,
          "anchor_update_gradient_norm": anchor_update_norm,
      },
      "landed_shape": {
          "n_real": n_real,
          "min_n_real": min(n_real) if n_real else None,
          "max_n_real": max_n_real if n_real else None,
          "group_chunks": group_chunks,
          "max_group_chunks": max(group_chunks) if group_chunks else None,
          "static_cap": max_static,
          "cap_coverage": max_n_real == max_static,
      },
      "hbm": {
          "peak_bytes": peak_hbm,
          "limit_bytes": hbm_limit,
          "peak_to_limit": hbm_ratio,
          "max_admitted_ratio": 0.90,
      },
      "receipts": {
          "p66_outer_check_enabled": p66_receipts,
          "deterministic_rollout_seed": seed_receipt,
          "disabled_local_wandb": disabled_wandb_receipt,
          "sampler_contract": sampler_receipt,
          "reducer_bucket_schedule": bucket_receipt,
          "reduce_once_report_accumulate": report_accumulate_receipts,
          "reduce_once_accumulator_loan": accumulator_loan_receipts,
          "reduce_once_accumulator_reset": accumulator_reset_receipts,
          "hbm_stage_receipts": raw_hbm_stages,
          "report_adjoint_memory": (
              report_adjoint_memory[0] if report_adjoint_memory else None
          ),
          "p76_chunk_dependency": (
              p76_ticket_matches[0] if p76_ticket_matches else None
          ),
          "p77_chunk_backpressure": p77_backpressure_matches,
          "ignored_post_training_finalizers": (
              ignored_post_training_finalizers
          ),
          "training_capsule": capsule_receipt,
          "forward_marker": forward_marker,
          "dp_reduction_transactions": update.get("dp_reduction_transactions"),
      },
      "artifacts": {
          name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
          for name, path in required.items()
      },
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", required=True, type=Path)
  parser.add_argument("--workload", required=True, choices=sorted(_WORKLOADS))
  parser.add_argument("--arm", required=True, choices=sorted(_ARMS))
  parser.add_argument("--docker-exit", required=True, type=int)
  parser.add_argument("--anchor-registry", required=True, type=Path)
  parser.add_argument("--require-anchor", action="store_true")
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  result = classify(
      args.root,
      workload=args.workload,
      arm=args.arm,
      docker_exit=args.docker_exit,
      anchor_registry=args.anchor_registry,
      require_anchor=args.require_anchor,
  )
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(result, sort_keys=True, separators=(",", ":")))
  return 0 if result["verdict"] in ("PASS", "MEASUREMENT_ONLY") else 1


if __name__ == "__main__":
  sys.exit(main())
