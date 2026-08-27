# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Durable, human-readable artifacts for bounded DeepSWE debug launches."""

from __future__ import annotations

import collections
import dataclasses
import enum
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np


TRAJECTORY_SCHEMA = "canon.p43.deepswe.trajectory.v1"
METRICS_SCHEMA = "canon.p43.deepswe.batch-metrics.v1"
MANIFEST_SCHEMA = "canon.p43.deepswe.run-manifest.v1"
P34_TRAJECTORY_SCHEMA = "canon.p34.deepswe.trajectory.v1"
P34_METRICS_SCHEMA = "canon.p34.deepswe.batch-metrics.v1"
P34_MANIFEST_SCHEMA = "canon.p34.deepswe.run-manifest.v1"
P44_TRAJECTORY_SCHEMA = "canon.p44.deepswe.trajectory.v1"
P44_METRICS_SCHEMA = "canon.p44.deepswe.batch-metrics.v1"
P44_MANIFEST_SCHEMA = "canon.p44.deepswe.run-manifest.v1"
P58_TRAJECTORY_SCHEMA = "canon.p58.deepswe.trajectory.v1"
P58_METRICS_SCHEMA = "canon.p58.deepswe.batch-metrics.v1"
P58_MANIFEST_SCHEMA = "canon.p58.deepswe.run-manifest.v1"
ONEHOST_TRAJECTORY_SCHEMA = "canon.local.deepswe.trajectory.v1"
ONEHOST_METRICS_SCHEMA = "canon.local.deepswe.batch-metrics.v1"
ONEHOST_MANIFEST_SCHEMA = "canon.local.deepswe.run-manifest.v1"
SOLVE_DEFINITION = "r2egym_final_reward_eq_1"
_COMPLETE_STATUS = "SUCCEEDED"
_COMPACT_FILTER_STATUSES = frozenset({
    "MAX_STEPS_REACHED",
    "MAX_CONTEXT_LIMIT_REACHED",
    "TIMEOUT",
    "ENV_TIMEOUT",
    "MODEL_TIMEOUT",
    "REWARD_TIMEOUT",
})
_TIMEOUT_STATUSES = frozenset({
    "TIMEOUT",
    "ENV_TIMEOUT",
    "MODEL_TIMEOUT",
    "REWARD_TIMEOUT",
})
_TIMEOUT_STAGES = frozenset({
    "environment_unknown",
    "sandbox_start",
    "environment_reset",
    "environment_step",
    "model_generation",
    "final_reward",
    "trajectory_deadline",
})
_TIMEOUT_SCHEDULER_REASONS = frozenset({
    "",
    "scheduling_gated",
    "unschedulable",
})
_TIMEOUT_RESOURCES = frozenset({
    "",
    "cpu",
    "memory",
    "ephemeral_storage",
    "other",
})
_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|auth|credential|password|secret|token)$", re.I
)
_SECRET_VALUE = re.compile(
    r"(?:(?:ghp|github_pat|hf|sk)-[A-Za-z0-9_-]{12,})"
)


def _mode(values: Mapping[str, str]) -> str:
  p43 = values.get("CANON_P43_DEEPSWE_DEBUG", "0")
  p44 = values.get("CANON_P44_DEEPSWE_PARITY", "0")
  onehost = values.get("CANON_DEEPSWE_ONEHOST_SMOKE", "0")
  p34 = values.get("CANON_P34_TRAJECTORY_CAPTURE", "0")
  p58 = values.get("CANON_P58_DEEPSWE_TIM", "0")
  if p43 not in ("0", "1"):
    raise ValueError("CANON_P43_DEEPSWE_DEBUG must be exactly 0 or 1")
  if p44 not in ("0", "1"):
    raise ValueError("CANON_P44_DEEPSWE_PARITY must be exactly 0 or 1")
  if onehost not in ("0", "1"):
    raise ValueError("CANON_DEEPSWE_ONEHOST_SMOKE must be exactly 0 or 1")
  if p34 not in ("0", "1"):
    raise ValueError("CANON_P34_TRAJECTORY_CAPTURE must be exactly 0 or 1")
  if p58 not in ("0", "1"):
    raise ValueError("CANON_P58_DEEPSWE_TIM must be exactly 0 or 1")
  if sum(raw == "1" for raw in (p34, p43, p44, p58, onehost)) > 1:
    raise ValueError(
        "P34, P43, P44, P58, and one-host artifacts are mutually exclusive"
    )
  if onehost == "1":
    return "onehost"
  if p34 == "1":
    return "p34"
  if p58 == "1":
    return "p58"
  return "p44" if p44 == "1" else "p43"


def enabled(values: Mapping[str, str] | None = None) -> bool:
  environ = os.environ if values is None else values
  mode = _mode(environ)
  key = {
      "onehost": "CANON_DEEPSWE_ONEHOST_SMOKE",
      "p34": "CANON_P34_TRAJECTORY_CAPTURE",
      "p44": "CANON_P44_DEEPSWE_PARITY",
      "p58": "CANON_P58_DEEPSWE_TIM",
      "p43": "CANON_P43_DEEPSWE_DEBUG",
  }[mode]
  return environ.get(key, "0") == "1"


def onehost(values: Mapping[str, str] | None = None) -> bool:
  """Returns whether the default-off local integration contract is active."""
  environ = os.environ if values is None else values
  return _mode(environ) == "onehost" and enabled(environ)


def no_commit(values: Mapping[str, str] | None = None) -> bool:
  """Returns the fail-closed one-host backward-without-commit selection."""
  environ = os.environ if values is None else values
  raw = environ.get("CANON_DEEPSWE_ONEHOST_NO_COMMIT", "0")
  if raw not in ("0", "1"):
    raise ValueError(
        "CANON_DEEPSWE_ONEHOST_NO_COMMIT must be exactly 0 or 1"
    )
  if raw == "1" and not onehost(environ):
    raise ValueError("one-host no-commit requires the one-host smoke mode")
  return raw == "1"


def onehost_xprof_arm(
    values: Mapping[str, str] | None = None,
) -> str:
  """Returns the signed P58 one-host profiling arm, or an empty string.

  The selector is deliberately narrower than the general one-host smoke. It
  admits only the mutation-free backward carrier so a profile cannot silently
  commit an optimizer update or overlap a production P58 arm.
  """
  environ = os.environ if values is None else values
  arm = environ.get("CANON_P58_ONEHOST_XPROF_ARM", "")
  if arm not in ("", "native", "zero-hp"):
    raise ValueError(
        "CANON_P58_ONEHOST_XPROF_ARM must be empty, native, or zero-hp"
    )
  if not arm:
    return ""
  exact = (
      onehost(environ)
      and no_commit(environ)
      and environ.get("CANON_DEEPSWE_ONEHOST_STAGE", "")
      == "backward-no-commit"
      and environ.get("CANON_DEEPSWE_ONEHOST_ROLLOUT_ONLY", "0") == "0"
      and environ.get("CANON_P58_DEEPSWE_TIM", "0") == "0"
  )
  if not exact:
    raise ValueError(
        "P58 one-host XProf requires the exclusive backward-no-commit "
        "DP1xTP4 carrier"
    )
  return arm


def onehost_seam_probe(
    values: Mapping[str, str] | None = None,
) -> bool:
  """Returns the exclusive long-context P58 Zero-HP one-host probe.

  This is a diagnostic extension of the mutation-free one-host carrier.  It
  may reproduce a finite decode-vs-prefill RED, but it cannot certify TP8 or
  the disaggregated DP8xTP8 production geometry.
  """
  environ = os.environ if values is None else values
  raw = environ.get("CANON_P58_ONEHOST_SEAM_PROBE", "0")
  if raw not in ("0", "1"):
    raise ValueError(
        "CANON_P58_ONEHOST_SEAM_PROBE must be exactly 0 or 1"
    )
  if raw == "0":
    return False
  if onehost_xprof_arm(environ) != "zero-hp":
    raise ValueError(
        "P58 one-host seam probe requires the exclusive Zero-HP "
        "backward-no-commit carrier"
    )
  return True


def artifact_directory(values: Mapping[str, str] | None = None) -> str:
  environ = os.environ if values is None else values
  key = {
      "onehost": "CANON_DEEPSWE_ONEHOST_DEBUG_DIR",
      "p34": "CANON_P34_DEBUG_DIR",
      "p44": "CANON_P44_DEBUG_DIR",
      "p58": "CANON_P58_DEBUG_DIR",
      "p43": "CANON_P43_DEBUG_DIR",
  }[_mode(environ)]
  return environ.get(key, "")


def rollout_only(values: Mapping[str, str] | None = None) -> bool:
  environ = os.environ if values is None else values
  if _mode(environ) == "p34":
    return False
  if _mode(environ) == "p58":
    return False
  key = {
      "onehost": "CANON_DEEPSWE_ONEHOST_ROLLOUT_ONLY",
      "p44": "CANON_P44_ROLLOUT_ONLY",
      "p43": "CANON_P43_ROLLOUT_ONLY",
  }[_mode(environ)]
  raw = environ.get(key, "0")
  if raw not in ("0", "1"):
    raise ValueError(f"{key} must be exactly 0 or 1")
  return raw == "1"


def marker_prefix(values: Mapping[str, str] | None = None) -> str:
  environ = os.environ if values is None else values
  return {
      "onehost": "DEEPSWE.ONEHOST",
      "p34": "P34",
      "p44": "P44",
      "p58": "P58",
      "p43": "P43",
  }[_mode(environ)]


def _schemas(values: Mapping[str, str]) -> tuple[str, str, str]:
  mode = _mode(values)
  if mode == "onehost":
    return (
        ONEHOST_TRAJECTORY_SCHEMA,
        ONEHOST_METRICS_SCHEMA,
        ONEHOST_MANIFEST_SCHEMA,
    )
  if mode == "p34":
    return P34_TRAJECTORY_SCHEMA, P34_METRICS_SCHEMA, P34_MANIFEST_SCHEMA
  if mode == "p44":
    return P44_TRAJECTORY_SCHEMA, P44_METRICS_SCHEMA, P44_MANIFEST_SCHEMA
  if mode == "p58":
    return P58_TRAJECTORY_SCHEMA, P58_METRICS_SCHEMA, P58_MANIFEST_SCHEMA
  return TRAJECTORY_SCHEMA, METRICS_SCHEMA, MANIFEST_SCHEMA


def _serializable(value: Any, *, key: str = "") -> Any:
  """Converts a trajectory value to JSON while redacting credential fields."""
  if key and _SENSITIVE_KEY.search(key):
    return "<redacted>"
  if value is None or isinstance(value, (bool, int, str)):
    if isinstance(value, str):
      return _SECRET_VALUE.sub("<redacted>", value)
    return value
  if isinstance(value, float):
    return value if math.isfinite(value) else str(value)
  if isinstance(value, enum.Enum):
    return value.name
  if isinstance(value, np.generic):
    return _serializable(value.item(), key=key)
  if isinstance(value, np.ndarray):
    return _serializable(value.tolist(), key=key)
  if dataclasses.is_dataclass(value):
    return _serializable(dataclasses.asdict(value), key=key)
  if isinstance(value, Mapping):
    return {
        str(item_key): _serializable(item_value, key=str(item_key))
        for item_key, item_value in value.items()
    }
  if isinstance(value, (list, tuple, set)):
    return [_serializable(item, key=key) for item in value]
  return repr(value)


def _json_bytes(value: Any) -> bytes:
  return (
      json.dumps(
          _serializable(value), sort_keys=True, separators=(",", ":")
      )
      + "\n"
  ).encode("utf-8")


def _atomic_write_new(path: Path, payload: bytes) -> None:
  """Atomically publishes a new file without overwriting prior evidence."""
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.exists():
    raise FileExistsError(f"refusing to overwrite P43 evidence: {path}")
  temporary = path.with_name(
      f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
  )
  try:
    with temporary.open("xb") as output:
      output.write(payload)
      output.flush()
      os.fsync(output.fileno())
    os.link(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  finally:
    if temporary.exists():
      temporary.unlink()


def _atomic_write_gzip_jsonl(path: Path, records: Iterable[Any]) -> str:
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.exists():
    raise FileExistsError(f"refusing to overwrite P43 evidence: {path}")
  temporary = path.with_name(
      f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
  )
  try:
    with temporary.open("xb") as raw:
      with gzip.GzipFile(
          filename="", mode="wb", fileobj=raw, mtime=0
      ) as compressed:
        for record in records:
          compressed.write(_json_bytes(record))
      raw.flush()
      os.fsync(raw.fileno())
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    os.link(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
    return digest
  finally:
    if temporary.exists():
      temporary.unlink()


def _append_fsync(path: Path, record: Mapping[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("ab") as output:
    output.write(_json_bytes(record))
    output.flush()
    os.fsync(output.fileno())


def next_batch_index(output_dir: str | os.PathLike[str]) -> int:
  """Returns the next durable batch index, rejecting partial journals."""
  root = Path(output_dir)
  trajectory_paths = sorted(root.glob("batch-*.trajectories.jsonl.gz"))
  observed_indices = []
  for path in trajectory_paths:
    match = re.fullmatch(r"batch-(\d{6})\.trajectories\.jsonl\.gz", path.name)
    if match is None:
      raise ValueError(f"unexpected DeepSWE trajectory artifact: {path}")
    observed_indices.append(int(match.group(1)))
  expected_indices = list(range(len(trajectory_paths)))
  if observed_indices != expected_indices:
    raise ValueError(
        "DeepSWE trajectory journal is not contiguous: "
        f"expected={expected_indices} actual={observed_indices}"
    )

  metrics_path = root / "batch_metrics.jsonl"
  metrics = []
  if metrics_path.exists():
    for line_number, line in enumerate(
        metrics_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
      if not line.strip():
        continue
      try:
        record = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid DeepSWE metrics JSON at {metrics_path}:{line_number}"
        ) from exc
      metrics.append(record)
  metric_indices = [record.get("step") for record in metrics]
  if metric_indices != expected_indices or len(metrics) != len(trajectory_paths):
    raise ValueError(
        "DeepSWE trajectory/metrics journal is partial: "
        f"trajectories={observed_indices} metrics={metric_indices}"
    )
  for record, path in zip(metrics, trajectory_paths):
    if (
        Path(str(record.get("trajectory_path", ""))).name != path.name
        or record.get("trajectory_sha256")
        != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
      raise ValueError(f"DeepSWE journal digest mismatch for {path}")
  return len(trajectory_paths)


def _manifest(
    values: Mapping[str, str], *, model_id: str, output_dir: Path
) -> dict[str, Any]:
  trajectory_schema, metrics_schema, manifest_schema = _schemas(values)
  mode = _mode(values)
  xprof_arm = ""
  if mode == "onehost":
    if model_id != "Qwen/Qwen3-4B-Instruct-2507":
      raise ValueError(
          "one-host DeepSWE artifacts require "
          "Qwen/Qwen3-4B-Instruct-2507"
      )
    seam_probe = onehost_seam_probe(values)
    contract_name = (
        "local-qwen4b-dp1-tp4-seam-probe"
        if seam_probe
        else "local-qwen4b-dp1-tp4"
    )
    slice_topology = "direct-attached-v5p-4"
    role_topology = {"dp": 1, "tp": 4, "devices": 4}
    global_prompts = 1
    generations = 2
    max_turns = 16 if seam_probe else 2
    max_response_length = 4096 if seam_probe else 512
    stage = values.get("CANON_DEEPSWE_ONEHOST_STAGE", "")
    xprof_arm = onehost_xprof_arm(values)
  elif mode == "p34":
    if model_id != "Qwen/Qwen3-32B":
      raise ValueError("P34 production artifacts require Qwen/Qwen3-32B")
    p46_train = values.get("CANON_P46_DEEPSWE_TRAIN", "0") == "1"
    p46_topology = values.get("CANON_P46_TOPOLOGY", "none")
    if p46_train and p46_topology == "64":
      contract_name = "p46-qwen32b-train-64"
      slice_topology = "4x4x4"
      role_topology = {"dp": 4, "tp": 8, "devices": 32}
    elif p46_train and p46_topology == "256":
      contract_name = "p46-qwen32b-train-256"
      slice_topology = "4x8x8"
      role_topology = {"dp": 16, "tp": 8, "devices": 128}
    elif p46_train:
      raise ValueError("P46 Qwen3-32B artifact topology must be 64 or 256")
    else:
      contract_name = "p34-production"
      slice_topology = "4x8x8"
      role_topology = {"dp": 16, "tp": 8, "devices": 128}
    global_prompts = 8
    generations = 8
    max_turns = 50
    max_response_length = 16384
    stage = values.get("CANON_P34_RUN_STAGE", "")
    if stage != "full":
      raise ValueError("P34 production artifacts require the full stage")
  elif mode == "p58":
    if model_id != "Qwen/Qwen3-4B-Instruct-2507":
      raise ValueError(
          "P58 artifacts require Qwen/Qwen3-4B-Instruct-2507"
      )
    arm = values.get("CANON_P58_TIM_ARM", "")
    if arm not in ("native", "zero"):
      raise ValueError("P58 artifact arm must be native or zero")
    contract_name = "p58-qwen4b-tim-128"
    slice_topology = "4x4x8"
    role_topology = {"dp": 8, "tp": 8, "devices": 64}
    global_prompts = 8
    generations = 16
    max_turns = 50
    max_response_length = 16384
    stage = values.get("CANON_P34_RUN_STAGE", "")
  elif mode == "p44":
    topology = values.get("CANON_P44_TOPOLOGY", "")
    if topology == "64":
      contract_name = "p44-qwen4b-parity-64"
      slice_topology = "4x4x4"
      role_topology = {"dp": 4, "tp": 8, "devices": 32}
    elif topology == "128":
      contract_name = "p44-qwen4b-parity-128"
      slice_topology = "4x4x8"
      role_topology = {"dp": 8, "tp": 8, "devices": 64}
    else:
      raise ValueError("P44 artifact topology must be exactly 64 or 128")
    if model_id != "Qwen/Qwen3-4B-Instruct-2507":
      raise ValueError(
          "P44 artifacts require Qwen/Qwen3-4B-Instruct-2507"
      )
    global_prompts = 4
    generations = 4
    max_turns = 50
    max_response_length = 16384
    stage = values.get("CANON_P34_RUN_STAGE", "")
  else:
    contract_name = "p43-64chip-debug"
    slice_topology = "4x4x4"
    role_topology = {"dp": 4, "tp": 8, "devices": 32}
    global_prompts = 4
    generations = 4
    max_turns = 5
    max_response_length = 4096
    stage = values.get("CANON_P34_RUN_STAGE", "")
  return {
      "schema": manifest_schema,
      "trajectory_schema": trajectory_schema,
      "metrics_schema": metrics_schema,
      "solve_definition": SOLVE_DEFINITION,
      "source_commit": values.get("CANON_EXPECT_COMMIT", ""),
      "source_branch": values.get("CANON_SOURCE_BRANCH", ""),
      "source_diff_sha256": values.get("CANON_P58_SOURCE_DIFF_SHA256", ""),
      "run_id": values.get("CANON_RUN_ID", ""),
      "expected_hostname": values.get("CANON_P58_EXPECT_HOSTNAME", ""),
      "model_snapshot": values.get("CANON_P58_MODEL_SNAPSHOT", ""),
      "r2egym_commit": values.get("CANON_P58_R2EGYM_COMMIT", ""),
      "task_image": values.get("CANON_DEEPSWE_ONEHOST_TASK_IMAGE", ""),
      "task_image_id": values.get("CANON_P58_TASK_IMAGE_ID", ""),
      "runner_sha256": values.get("CANON_P58_RUNNER_SHA256", ""),
      "stage": stage,
      "model_id": model_id,
      "contract_name": contract_name,
      "tim_arm": values.get("CANON_P58_TIM_ARM", "none"),
      "checked_vma_diagnostic": values.get(
          "CANON_P58_CHECKED_VMA_DIAGNOSTIC", "none"
      ),
      "onehost_xprof_arm": xprof_arm if mode == "onehost" else "none",
      "onehost_seam_probe": (
          onehost_seam_probe(values) if mode == "onehost" else False
      ),
      "slice_topology": slice_topology,
      "role_topology": role_topology,
      "global_prompts": global_prompts,
      "generations": generations,
      "global_trajectories": global_prompts * generations,
      "max_turns": max_turns,
      "max_response_length": max_response_length,
      "dataset_seed": 42,
      "rollout_seed": (
          42 if mode == "p58" or bool(xprof_arm) else None
      ),
      "seed_scope": (
          "engine-global; async completion order not claimed"
          if mode == "p58" or bool(xprof_arm)
          else "dataset-only"
      ),
      "dataset_name": values.get("CANON_P34_DATASET_NAME", ""),
      "dataset_revision": values.get("CANON_P34_DATASET_REVISION", ""),
      "dataset_split": values.get("CANON_P34_DATASET_SPLIT", ""),
      "dataset_rows": values.get("CANON_P34_DATASET_ROWS", ""),
      "clean_rows": values.get("CANON_P34_CLEAN_ROWS", ""),
      "timeouts_seconds": {
          "per_turn": values.get(
              "CANON_DEEPSWE_PER_TURN_TIMEOUT_SECS", ""
          ),
          "trajectory": values.get(
              "CANON_DEEPSWE_TRAJECTORY_TIMEOUT_SECS", ""
          ),
          "step": values.get("CANON_DEEPSWE_STEP_TIMEOUT_SECS", ""),
          "reward": values.get("CANON_DEEPSWE_REWARD_TIMEOUT_SECS", ""),
          "cleanup": values.get(
              "CANON_DEEPSWE_CLEANUP_TIMEOUT_SECS", ""
          ),
          "rollout_batch": values.get(
              "CANON_DEEPSWE_ROLLOUT_BATCH_TIMEOUT_SECS", ""
          ),
          "sandbox_active_deadline": values.get(
              "R2E_ACTIVE_DEADLINE_SECONDS", ""
          ),
      },
      "whitelist_sha256": values.get("CANON_P34_WHITELIST_SHA256", ""),
      "artifact_directory": str(output_dir),
  }


def ensure_manifest(
    output_dir: str | os.PathLike[str],
    *,
    model_id: str,
    values: Mapping[str, str] | None = None,
) -> dict[str, Any]:
  environ = os.environ if values is None else values
  root = Path(output_dir)
  if not root.is_absolute():
    raise ValueError("P43 debug artifact directory must be absolute")
  record = _manifest(environ, model_id=model_id, output_dir=root)
  path = root / "run_manifest.json"
  if path.exists():
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != record:
      raise ValueError("P43 run manifest changed within one run directory")
  else:
    _atomic_write_new(path, json.dumps(record, indent=2, sort_keys=True).encode(
        "utf-8"
    ) + b"\n")
  return record


def _status_name(trajectory: Mapping[str, Any]) -> str:
  status = trajectory.get("status", "UNKNOWN")
  if isinstance(status, enum.Enum):
    return status.name
  return str(status)


def _finite_float(value: Any, *, label: str) -> float:
  result = float(np.asarray(value).item())
  if not math.isfinite(result):
    raise ValueError(f"P43 {label} must be finite, got {result!r}")
  return result


def _timeout_metadata(
    trajectory: Mapping[str, Any], status: str
) -> tuple[str, str, str]:
  """Returns bounded timeout dimensions, rejecting high-cardinality values."""
  stage = str(trajectory.get("timeout_stage", "") or "")
  scheduler_reason = str(
      trajectory.get("timeout_scheduler_reason", "") or ""
  )
  resource = str(trajectory.get("timeout_resource", "") or "")
  if status not in _TIMEOUT_STATUSES:
    if stage or scheduler_reason or resource:
      raise ValueError("non-timeout trajectory contains timeout metadata")
    return "", "", ""
  if not stage:
    stage = {
        "TIMEOUT": "trajectory_deadline",
        "ENV_TIMEOUT": "environment_unknown",
        "MODEL_TIMEOUT": "model_generation",
        "REWARD_TIMEOUT": "final_reward",
    }[status]
  if stage not in _TIMEOUT_STAGES:
    raise ValueError(f"unsupported timeout stage: {stage!r}")
  if scheduler_reason not in _TIMEOUT_SCHEDULER_REASONS:
    raise ValueError(
        f"unsupported timeout scheduler reason: {scheduler_reason!r}"
    )
  if resource not in _TIMEOUT_RESOURCES:
    raise ValueError(f"unsupported timeout resource: {resource!r}")
  if stage != "sandbox_start" and (scheduler_reason or resource):
    raise ValueError(
        "scheduler timeout metadata is valid only for sandbox_start"
    )
  return stage, scheduler_reason, resource


def timeout_wandb_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
  """Builds the fixed low-cardinality timeout dashboard dimensions."""
  trajectories = int(metrics["trajectories"])
  if trajectories <= 0:
    raise ValueError("DeepSWE timeout metrics require trajectories > 0")
  fields = (
      "timeout_trajectories",
      "env_timeout_trajectories",
      "sandbox_start_timeout_trajectories",
      "scheduling_gated_trajectories",
      "unschedulable_trajectories",
      "insufficient_cpu_trajectories",
      "insufficient_memory_trajectories",
  )
  result = {}
  for field in fields:
    count = int(metrics[field])
    if not 0 <= count <= trajectories:
      raise ValueError(f"invalid DeepSWE timeout count {field}={count}")
    name = field.removesuffix("_trajectories")
    result[f"deepswe/{field}"] = float(count)
    result[f"deepswe/{name}_ratio"] = count / trajectories
  result["deepswe/all_env_timeout_batch"] = float(
      int(metrics["env_timeout_trajectories"]) == trajectories
  )
  result["deepswe/all_sandbox_start_timeout_batch"] = float(
      int(metrics["sandbox_start_timeout_trajectories"]) == trajectories
  )
  for status in sorted(_TIMEOUT_STATUSES):
    count = int(metrics["status_histogram"].get(status, 0))
    key = status.lower()
    result[f"deepswe/status/{key}_count"] = float(count)
    result[f"deepswe/status/{key}_ratio"] = count / trajectories
  return result


def persist_batch(
    trajectories: Sequence[Any],
    rewards: Sequence[Any],
    advantages: Sequence[Any],
    *,
    expected_step: int,
    optimizer_step: int | None = None,
    output_dir: str | os.PathLike[str],
    model_id: str,
    values: Mapping[str, str] | None = None,
) -> dict[str, Any]:
  """Persists one contract-sized real DeepSWE rollout batch and metrics."""
  if expected_step < 0:
    raise ValueError("P43 expected_step must be nonnegative")
  if optimizer_step is not None and optimizer_step < 0:
    raise ValueError("DeepSWE optimizer_step must be nonnegative")
  environ = os.environ if values is None else values
  root = Path(output_dir)
  manifest = ensure_manifest(root, model_id=model_id, values=environ)
  expected_trajectories = int(manifest["global_trajectories"])
  expected_groups = int(manifest["global_prompts"])
  expected_generations = int(manifest["generations"])
  if (
      len(trajectories) != expected_trajectories
      or len(rewards) != expected_trajectories
      or len(advantages) != expected_trajectories
  ):
    raise ValueError(
        "DeepSWE artifact batch requires exactly "
        f"{expected_trajectories} trajectories, rewards, and advantages"
    )
  trajectory_schema, metrics_schema, _ = _schemas(
      environ
  )

  records = []
  groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
  status_histogram: collections.Counter[str] = collections.Counter()
  reward_histogram: collections.Counter[str] = collections.Counter()
  timeout_stage_histogram: collections.Counter[str] = collections.Counter()
  timeout_scheduler_histogram: collections.Counter[str] = (
      collections.Counter()
  )
  timeout_resource_histogram: collections.Counter[str] = collections.Counter()
  for item, training_reward_value, advantage_value in zip(
      trajectories, rewards, advantages
  ):
    trajectory = item.traj
    if not isinstance(trajectory, Mapping):
      raise TypeError("P43 Token-mode trajectory must be a mapping")
    group_id = str(item.group_id)
    pair_index = int(item.pair_index)
    raw_reward = _finite_float(
        trajectory.get("trajectory_reward"), label="raw final reward"
    )
    training_reward = _finite_float(
        training_reward_value, label="training reward"
    )
    advantage = _finite_float(advantage_value, label="advantage")
    status = _status_name(trajectory)
    timeout_stage, timeout_scheduler_reason, timeout_resource = (
        _timeout_metadata(trajectory, status)
    )
    complete = status == _COMPLETE_STATUS
    compact_filtered = status in _COMPACT_FILTER_STATUSES
    solved = complete and raw_reward == 1.0
    status_histogram[status] += 1
    if timeout_stage:
      timeout_stage_histogram[timeout_stage] += 1
    if timeout_scheduler_reason:
      timeout_scheduler_histogram[timeout_scheduler_reason] += 1
    if timeout_resource:
      timeout_resource_histogram[timeout_resource] += 1
    reward_histogram[format(raw_reward, ".12g")] += 1
    record = {
        "schema": trajectory_schema,
        "step": expected_step,
        "group_id": group_id,
        "pair_index": pair_index,
        "status": status,
        "timeout_stage": timeout_stage,
        "timeout_scheduler_reason": timeout_scheduler_reason,
        "timeout_resource": timeout_resource,
        "complete": complete,
        "compact_filtered": compact_filtered,
        "solve_definition": SOLVE_DEFINITION,
        "solved": solved,
        "raw_final_reward": raw_reward,
        "training_reward": training_reward,
        "advantage": advantage,
        # Compact-filtered rows retain their raw advantage for audit but have
        # an all-zero policy mask, so they do not constitute training signal.
        "advantage_nonzero": advantage != 0.0 and not compact_filtered,
        "raw_advantage_nonzero": advantage != 0.0,
        "task_identity": _serializable(
            getattr(item, "metadata", {}).get("task_identity", {})
        ),
        "trajectory": trajectory,
    }
    if _mode(environ) == "p58":
      record["optimizer_step"] = (
          expected_step if optimizer_step is None else optimizer_step
      )
    records.append(record)
    groups[group_id].append(record)

  if len(groups) != expected_groups or any(
      len(group) != expected_generations for group in groups.values()
  ):
    sizes = {group_id: len(group) for group_id, group in groups.items()}
    raise ValueError(
        "DeepSWE artifact group geometry changed: "
        f"expected={expected_groups}x{expected_generations} actual={sizes}"
    )
  if any(
      sorted(record["pair_index"] for record in group)
      != list(range(expected_generations))
      for group in groups.values()
  ):
    raise ValueError(
        "DeepSWE artifact pair indices must cover each generation exactly"
    )

  group_records = []
  category_counts: collections.Counter[str] = collections.Counter()
  for group_id, group in sorted(groups.items()):
    complete_count = sum(record["complete"] for record in group)
    compact_filtered_count = sum(
        record["compact_filtered"] for record in group
    )
    solved_count = sum(record["solved"] for record in group)
    if complete_count != expected_generations:
      category = "incomplete"
    elif solved_count == expected_generations:
      category = "all_solved"
    elif solved_count == 0:
      category = "all_failed"
    else:
      category = "mixed"
    category_counts[category] += 1
    group_records.append({
        "group_id": group_id,
        "category": category,
        "complete_trajectories": complete_count,
        "compact_filtered_trajectories": compact_filtered_count,
        "solved_trajectories": solved_count,
        "nonzero_advantages": sum(
            record["advantage_nonzero"] for record in group
        ),
        "raw_nonzero_advantages": sum(
            record["raw_advantage_nonzero"] for record in group
        ),
        "raw_rewards": [record["raw_final_reward"] for record in group],
    })

  solved_trajectories = sum(record["solved"] for record in records)
  complete_trajectories = sum(record["complete"] for record in records)
  compact_filtered_trajectories = sum(
      record["compact_filtered"] for record in records
  )
  nonzero_advantages = sum(record["advantage_nonzero"] for record in records)
  raw_nonzero_advantages = sum(
      record["raw_advantage_nonzero"] for record in records
  )
  timeout_trajectories = sum(status_histogram[s] for s in _TIMEOUT_STATUSES)
  env_timeout_trajectories = status_histogram["ENV_TIMEOUT"]
  sandbox_start_timeout_trajectories = timeout_stage_histogram[
      "sandbox_start"
  ]
  metrics = {
      "schema": metrics_schema,
      "step": expected_step,
      "solve_definition": SOLVE_DEFINITION,
      "trajectories": expected_trajectories,
      "complete_trajectories": complete_trajectories,
      "incomplete_trajectories": expected_trajectories - complete_trajectories,
      "compact_filtered_trajectories": compact_filtered_trajectories,
      "compact_filtered_trajectory_ratio": (
          compact_filtered_trajectories / expected_trajectories
      ),
      "compact_filtered_prompt_groups": sum(
          item["compact_filtered_trajectories"] > 0 for item in group_records
      ),
      "solved_trajectories": solved_trajectories,
      "trajectory_solve_ratio": solved_trajectories / expected_trajectories,
      "complete_trajectory_solve_ratio": (
          solved_trajectories / complete_trajectories
          if complete_trajectories
          else 0.0
      ),
      "prompt_groups": expected_groups,
      "all_solved_prompt_groups": category_counts["all_solved"],
      "all_failed_prompt_groups": category_counts["all_failed"],
      "mixed_prompt_groups": category_counts["mixed"],
      "incomplete_prompt_groups": category_counts["incomplete"],
      "effective_prompt_groups": sum(
          item["nonzero_advantages"] > 0 for item in group_records
      ),
      "nonzero_advantages": nonzero_advantages,
      "nonzero_advantage_ratio": nonzero_advantages / expected_trajectories,
      "raw_nonzero_advantages": raw_nonzero_advantages,
      "raw_nonzero_advantage_ratio": (
          raw_nonzero_advantages / expected_trajectories
      ),
      "nonbinary_final_rewards": sum(
          record["raw_final_reward"] not in (0.0, 1.0) for record in records
      ),
      "status_histogram": dict(sorted(status_histogram.items())),
      "timeout_stage_histogram": dict(
          sorted(timeout_stage_histogram.items())
      ),
      "timeout_scheduler_reason_histogram": dict(
          sorted(timeout_scheduler_histogram.items())
      ),
      "timeout_resource_histogram": dict(
          sorted(timeout_resource_histogram.items())
      ),
      "timeout_trajectories": timeout_trajectories,
      "timeout_trajectory_ratio": (
          timeout_trajectories / expected_trajectories
      ),
      "env_timeout_trajectories": env_timeout_trajectories,
      "env_timeout_trajectory_ratio": (
          env_timeout_trajectories / expected_trajectories
      ),
      "sandbox_start_timeout_trajectories": (
          sandbox_start_timeout_trajectories
      ),
      "sandbox_start_timeout_trajectory_ratio": (
          sandbox_start_timeout_trajectories / expected_trajectories
      ),
      "scheduling_gated_trajectories": timeout_scheduler_histogram[
          "scheduling_gated"
      ],
      "unschedulable_trajectories": timeout_scheduler_histogram[
          "unschedulable"
      ],
      "insufficient_cpu_trajectories": timeout_resource_histogram["cpu"],
      "insufficient_memory_trajectories": timeout_resource_histogram[
          "memory"
      ],
      "all_env_timeout_batch": (
          env_timeout_trajectories == expected_trajectories
      ),
      "all_sandbox_start_timeout_batch": (
          sandbox_start_timeout_trajectories == expected_trajectories
      ),
      "raw_final_reward_histogram": dict(sorted(reward_histogram.items())),
      "groups": group_records,
  }
  if _mode(environ) == "p58":
    metrics["optimizer_step"] = (
        expected_step if optimizer_step is None else optimizer_step
    )

  records.sort(key=lambda record: (record["group_id"], record["pair_index"]))
  trajectory_path = root / f"batch-{expected_step:06d}.trajectories.jsonl.gz"
  digest = _atomic_write_gzip_jsonl(trajectory_path, records)
  metrics["trajectory_path"] = str(trajectory_path)
  metrics["trajectory_sha256"] = digest
  _append_fsync(root / "batch_metrics.jsonl", metrics)
  payload = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
  prefix = marker_prefix(os.environ if values is None else values)
  print(
      f"[{prefix}.TRAJECTORY_BATCH] step={expected_step} "
      f"path={trajectory_path} sha256={digest}",
      flush=True,
  )
  print(f"[{prefix}.BATCH_METRICS_JSON] {payload}", flush=True)
  return metrics
