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
SOLVE_DEFINITION = "r2egym_final_reward_eq_1"
_COMPLETE_STATUS = "SUCCEEDED"
_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|auth|credential|password|secret|token)$", re.I
)
_SECRET_VALUE = re.compile(
    r"(?:(?:ghp|github_pat|hf|sk)-[A-Za-z0-9_-]{12,})"
)


def enabled(values: Mapping[str, str] | None = None) -> bool:
  environ = os.environ if values is None else values
  raw = environ.get("CANON_P43_DEEPSWE_DEBUG", "0")
  if raw not in ("0", "1"):
    raise ValueError("CANON_P43_DEEPSWE_DEBUG must be exactly 0 or 1")
  return raw == "1"


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


def _manifest(
    values: Mapping[str, str], *, model_id: str, output_dir: Path
) -> dict[str, Any]:
  return {
      "schema": MANIFEST_SCHEMA,
      "trajectory_schema": TRAJECTORY_SCHEMA,
      "metrics_schema": METRICS_SCHEMA,
      "solve_definition": SOLVE_DEFINITION,
      "source_commit": values.get("CANON_EXPECT_COMMIT", ""),
      "source_branch": values.get("CANON_SOURCE_BRANCH", ""),
      "run_id": values.get("CANON_RUN_ID", ""),
      "stage": values.get("CANON_P34_RUN_STAGE", ""),
      "model_id": model_id,
      "contract_name": "p43-64chip-debug",
      "slice_topology": "4x4x4",
      "role_topology": {"dp": 4, "tp": 8, "devices": 32},
      "global_prompts": 4,
      "generations": 4,
      "global_trajectories": 16,
      "max_turns": 5,
      "max_response_length": 4096,
      "dataset_seed": 42,
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


def persist_batch(
    trajectories: Sequence[Any],
    rewards: Sequence[Any],
    advantages: Sequence[Any],
    *,
    expected_step: int,
    output_dir: str | os.PathLike[str],
    model_id: str,
    values: Mapping[str, str] | None = None,
) -> dict[str, Any]:
  """Persists one exact 4x4 real DeepSWE rollout batch and its metrics."""
  if expected_step < 0:
    raise ValueError("P43 expected_step must be nonnegative")
  if len(trajectories) != 16 or len(rewards) != 16 or len(advantages) != 16:
    raise ValueError(
        "P43 requires exactly 16 trajectories, rewards, and advantages"
    )
  root = Path(output_dir)
  ensure_manifest(root, model_id=model_id, values=values)

  records = []
  groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
  status_histogram: collections.Counter[str] = collections.Counter()
  reward_histogram: collections.Counter[str] = collections.Counter()
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
    complete = status == _COMPLETE_STATUS
    solved = complete and raw_reward == 1.0
    status_histogram[status] += 1
    reward_histogram[format(raw_reward, ".12g")] += 1
    record = {
        "schema": TRAJECTORY_SCHEMA,
        "step": expected_step,
        "group_id": group_id,
        "pair_index": pair_index,
        "status": status,
        "complete": complete,
        "solve_definition": SOLVE_DEFINITION,
        "solved": solved,
        "raw_final_reward": raw_reward,
        "training_reward": training_reward,
        "advantage": advantage,
        "advantage_nonzero": advantage != 0.0,
        "trajectory": trajectory,
    }
    records.append(record)
    groups[group_id].append(record)

  if len(groups) != 4 or any(len(group) != 4 for group in groups.values()):
    sizes = {group_id: len(group) for group_id, group in groups.items()}
    raise ValueError(f"P43 requires four groups of four trajectories: {sizes}")
  if any(
      sorted(record["pair_index"] for record in group) != [0, 1, 2, 3]
      for group in groups.values()
  ):
    raise ValueError("P43 each group must contain pair indices 0,1,2,3")

  group_records = []
  category_counts: collections.Counter[str] = collections.Counter()
  for group_id, group in sorted(groups.items()):
    complete_count = sum(record["complete"] for record in group)
    solved_count = sum(record["solved"] for record in group)
    if complete_count != 4:
      category = "incomplete"
    elif solved_count == 4:
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
        "solved_trajectories": solved_count,
        "nonzero_advantages": sum(
            record["advantage_nonzero"] for record in group
        ),
        "raw_rewards": [record["raw_final_reward"] for record in group],
    })

  solved_trajectories = sum(record["solved"] for record in records)
  complete_trajectories = sum(record["complete"] for record in records)
  nonzero_advantages = sum(record["advantage_nonzero"] for record in records)
  metrics = {
      "schema": METRICS_SCHEMA,
      "step": expected_step,
      "solve_definition": SOLVE_DEFINITION,
      "trajectories": 16,
      "complete_trajectories": complete_trajectories,
      "incomplete_trajectories": 16 - complete_trajectories,
      "solved_trajectories": solved_trajectories,
      "trajectory_solve_ratio": solved_trajectories / 16,
      "complete_trajectory_solve_ratio": (
          solved_trajectories / complete_trajectories
          if complete_trajectories
          else 0.0
      ),
      "prompt_groups": 4,
      "all_solved_prompt_groups": category_counts["all_solved"],
      "all_failed_prompt_groups": category_counts["all_failed"],
      "mixed_prompt_groups": category_counts["mixed"],
      "incomplete_prompt_groups": category_counts["incomplete"],
      "effective_prompt_groups": sum(
          item["nonzero_advantages"] > 0 for item in group_records
      ),
      "nonzero_advantages": nonzero_advantages,
      "nonzero_advantage_ratio": nonzero_advantages / 16,
      "nonbinary_final_rewards": sum(
          record["raw_final_reward"] not in (0.0, 1.0) for record in records
      ),
      "status_histogram": dict(sorted(status_histogram.items())),
      "raw_final_reward_histogram": dict(sorted(reward_histogram.items())),
      "groups": group_records,
  }

  records.sort(key=lambda record: (record["group_id"], record["pair_index"]))
  trajectory_path = root / f"batch-{expected_step:06d}.trajectories.jsonl.gz"
  digest = _atomic_write_gzip_jsonl(trajectory_path, records)
  metrics["trajectory_path"] = str(trajectory_path)
  metrics["trajectory_sha256"] = digest
  _append_fsync(root / "batch_metrics.jsonl", metrics)
  payload = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
  print(
      f"[P43.TRAJECTORY_BATCH] step={expected_step} "
      f"path={trajectory_path} sha256={digest}",
      flush=True,
  )
  print(f"[P43.BATCH_METRICS_JSON] {payload}", flush=True)
  return metrics
