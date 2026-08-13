# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fail-closed artifacts and reports for resumable DeepSWE evaluation."""

from __future__ import annotations

import collections
import dataclasses
import enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CONFIG_SCHEMA = "canon.p46.deepswe-eval.config.v2"
TRAJECTORY_SCHEMA = "canon.p46.deepswe-eval.trajectory.v2"
REPORT_SCHEMA = "canon.p46.deepswe-eval.task-report.v2"
SUMMARY_SCHEMA = "canon.p46.deepswe-eval.summary.v2"
REWARD_ONLY = "reward_only"
REWARD_ONLY_TRAJECTORY_MODE = "reward_only_no_logprobs"
LOGPROB_OBSERVER = "logprob_observer"
LOGPROB_OBSERVER_TRAJECTORY_MODE = "observer_with_sampled_logprobs"
VALID_STATUSES = frozenset({"SUCCEEDED", "MAX_STEPS_REACHED"})
_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DIGEST_IMAGE = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}")
_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|auth|credential|password|secret|token)$", re.I
)
_SECRET_VALUE = re.compile(
    r"(?:(?:ghp|github_pat|hf|sk)-[A-Za-z0-9_-]{12,})"
)
_LOGPROB_KEY = re.compile(r"log[_-]?probs?", re.I)


@dataclasses.dataclass(frozen=True, slots=True)
class EvalConfig:
  """The exact Qwen3-4B clean-data evaluation contract."""

  model_id: str
  model_path: str
  dataset_name: str
  dataset_revision: str
  dataset_split: str
  dataset_rows: int
  whitelist_path: str
  whitelist_sha256: str
  whitelist_rows: int
  source_commit: str
  client_image: str
  topology: str
  evaluation_mode: str = REWARD_ONLY
  onehost_probe: bool = False
  parity_canary: bool = False
  max_model_len: int = 20_480
  max_response_length: int = 16_384
  max_steps: int = 50
  temperature: float = 1.0
  top_p: float = 1.0
  top_k: int = 0
  n_sample: int = 16
  logical_tasks: int = 32
  shard_tasks: int = 4
  shard_index: int = 0
  max_concurrency: int = 64
  trajectory_timeout_secs: int = 3000
  per_turn_timeout_secs: int = 300
  step_timeout_secs: int = 600
  reward_timeout_secs: int = 600
  cleanup_timeout_secs: int = 300
  shard_timeout_secs: int = 3600
  seed_base: int = 42
  prefix_cache: bool = False

  def validate(self) -> None:
    expected = {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "dataset_name": "R2E-Gym/R2E-Gym-Subset",
        "dataset_split": "train",
        "dataset_rows": 4578,
        "whitelist_rows": 1851,
    }
    if self.onehost_probe:
      expected.update({
          "evaluation_mode": REWARD_ONLY,
          "parity_canary": False,
          "max_model_len": 4096,
          "max_response_length": 512,
          "max_steps": 1,
          "n_sample": 1,
          "logical_tasks": 1,
          "shard_tasks": 1,
          "max_concurrency": 1,
          "trajectory_timeout_secs": 900,
          "per_turn_timeout_secs": 300,
          "step_timeout_secs": 300,
          "reward_timeout_secs": 300,
          "cleanup_timeout_secs": 120,
          "shard_timeout_secs": 1200,
      })
    else:
      expected.update({
          "max_model_len": 20_480,
          "max_response_length": 16_384,
          "max_steps": 50,
          "n_sample": 16,
          "logical_tasks": 1 if self.parity_canary else 32,
          "shard_tasks": 1 if self.parity_canary else 4,
          "max_concurrency": 16 if self.parity_canary else 64,
          "trajectory_timeout_secs": 3000,
          "per_turn_timeout_secs": 300,
          "step_timeout_secs": 600,
          "reward_timeout_secs": 600,
          "cleanup_timeout_secs": 300,
          "shard_timeout_secs": 3600,
      })
      if self.parity_canary:
        if self.evaluation_mode not in (REWARD_ONLY, LOGPROB_OBSERVER):
          raise ValueError("unsupported parity evaluation_mode")
      elif self.evaluation_mode != REWARD_ONLY:
        raise ValueError(
            "logprob_observer is restricted to the 64-chip parity canary"
        )
    actual = dataclasses.asdict(self)
    wrong = {
        key: actual[key]
        for key, value in expected.items()
        if actual[key] != value
    }
    if wrong:
      raise ValueError(f"P46 evaluation contract mismatch: {wrong}")
    if not _SHA.fullmatch(self.dataset_revision):
      raise ValueError("dataset_revision must be a lowercase 40-character SHA")
    if not _SHA.fullmatch(self.source_commit):
      raise ValueError("source_commit must be a lowercase 40-character SHA")
    if not _SHA256.fullmatch(self.whitelist_sha256):
      raise ValueError("whitelist_sha256 must be a lowercase SHA-256 digest")
    if not _DIGEST_IMAGE.fullmatch(self.client_image):
      raise ValueError("client_image must be pinned by sha256 digest")
    admitted_topologies = (
        ("4",)
        if self.onehost_probe
        else (("64",) if self.parity_canary else ("64", "256"))
    )
    if self.topology not in admitted_topologies:
      raise ValueError(
          "evaluation topology does not match its production/one-host mode"
      )
    if not os.path.isabs(self.model_path):
      raise ValueError("evaluation model_path must be absolute")
    if not os.path.isabs(self.whitelist_path):
      raise ValueError("evaluation whitelist_path must be absolute")
    if self.shard_index < 0:
      raise ValueError("shard_index must be nonnegative")
    if self.logical_tasks % self.shard_tasks:
      raise ValueError("logical_tasks must be divisible by shard_tasks")
    if self.shard_tasks * self.n_sample != self.max_concurrency:
      raise ValueError(
          "one physical evaluation shard must equal one concurrency wave"
      )
    if self.max_response_length >= self.max_model_len:
      raise ValueError("response budget must leave positive prompt capacity")
    if self.temperature <= 0 or self.top_p != 1.0 or self.top_k != 0:
      raise ValueError("P46 evaluation sampling policy changed")
    if self.prefix_cache:
      raise ValueError("P46 evaluation requires prefix cache disabled")
    if (
        self.trajectory_timeout_secs + self.cleanup_timeout_secs
        >= self.shard_timeout_secs
    ):
      raise ValueError("evaluation timeout nesting lost its abort margin")

  def canonical_record(self) -> dict[str, Any]:
    self.validate()
    return {
        "schema": CONFIG_SCHEMA,
        **dataclasses.asdict(self),
        "trajectory_mode": self.trajectory_mode,
        "sampled_by": self.sampled_by,
        "sampling_rng_mode": self.sampling_rng_mode,
    }

  @property
  def trajectory_mode(self) -> str:
    if self.evaluation_mode == REWARD_ONLY:
      return REWARD_ONLY_TRAJECTORY_MODE
    if self.evaluation_mode == LOGPROB_OBSERVER and self.parity_canary:
      return LOGPROB_OBSERVER_TRAJECTORY_MODE
    raise ValueError("unsupported DeepSWE evaluation_mode")

  @property
  def sampled_by(self) -> str:
    # The standalone evaluator calls the stock vLLM sampler. The source SHA
    # pins the exact Tunix request construction that selected that path.
    return f"stock@{self.source_commit}"

  @property
  def sampling_rng_mode(self) -> str:
    # The TPU/JAX vLLM backend rejects SamplingParams.seed. Sampling therefore
    # uses the engine-level key and its ordered split stream; it is incorrect
    # to claim independently replayable per-request seeds on this backend.
    return "engine_global_sequential"

  @property
  def collect_logprobs(self) -> bool:
    if self.evaluation_mode == REWARD_ONLY:
      return False
    if self.evaluation_mode == LOGPROB_OBSERVER and self.parity_canary:
      return True
    raise ValueError("unsupported DeepSWE evaluation_mode")

  @property
  def fingerprint(self) -> str:
    payload = json.dumps(
        self.canonical_record(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

  @property
  def run_tag(self) -> str:
    if self.onehost_probe:
      prefix = "q4i512-n1-onehost"
    elif self.parity_canary:
      prefix = f"q4i16k-n16-parity-{self.evaluation_mode}"
    else:
      prefix = "q4i16k-n16"
    return f"{prefix}-{self.topology}-{self.fingerprint[:16]}"

  def sample_nonce(self, task_key: str, sample_index: int) -> int:
    """Stable identity nonce; deliberately not passed as a vLLM seed."""
    if not 0 <= sample_index < self.n_sample:
      raise ValueError("sample_index is outside the signed n-sample range")
    payload = f"{self.seed_base}:{task_key}:{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def sha256_file(path: str | os.PathLike[str]) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def task_key(entry: Mapping[str, Any]) -> str:
  key = entry.get("docker_image")
  if not isinstance(key, str) or not key:
    raise ValueError("every P46 evaluation task requires docker_image")
  return key


def serializable(value: Any, *, key: str = "") -> Any:
  """Converts a trajectory to JSON while redacting credential-like values."""
  if key and _SENSITIVE_KEY.search(key):
    return "<redacted>"
  if value is None or isinstance(value, (bool, int, str)):
    return _SECRET_VALUE.sub("<redacted>", value) if isinstance(value, str) else value
  if isinstance(value, float):
    return value if math.isfinite(value) else str(value)
  if isinstance(value, enum.Enum):
    return value.name
  if isinstance(value, np.generic):
    return serializable(value.item(), key=key)
  if isinstance(value, np.ndarray):
    return serializable(value.tolist(), key=key)
  if dataclasses.is_dataclass(value):
    return serializable(dataclasses.asdict(value), key=key)
  if isinstance(value, Mapping):
    return {
        str(item_key): serializable(item_value, key=str(item_key))
        for item_key, item_value in value.items()
    }
  if isinstance(value, (list, tuple, set)):
    return [serializable(item, key=key) for item in value]
  return repr(value)


def reward_only_trajectory(value: Any, *, key: str = "") -> Any:
  """Normalizes absent logprob fields and rejects every numeric payload."""
  if key and _LOGPROB_KEY.search(key):
    if value is None:
      return None
    if isinstance(value, (list, tuple)) and not value:
      return None
    if isinstance(value, np.ndarray) and value.size == 0:
      return None
    raise ValueError(
        "reward-only evaluation artifact contains a logprob payload; "
        "missing logprobs must be absent or null, never numeric"
    )
  if dataclasses.is_dataclass(value):
    value = dataclasses.asdict(value)
  if isinstance(value, Mapping):
    return {
        str(item_key): reward_only_trajectory(
            item_value, key=str(item_key)
        )
        for item_key, item_value in value.items()
    }
  if isinstance(value, (list, tuple)):
    return [reward_only_trajectory(item, key=key) for item in value]
  return value


def trajectory_record(
    config: EvalConfig,
    *,
    entry: Mapping[str, Any],
    sample_index: int,
    trajectory: Any,
    elapsed_secs: float,
) -> dict[str, Any]:
  """Builds one complete, redacted and resume-addressable trajectory record."""
  config.validate()
  if hasattr(trajectory, "to_dict"):
    raw_trajectory = trajectory.to_dict()
  elif isinstance(trajectory, Mapping):
    raw_trajectory = dict(trajectory)
  else:
    raise TypeError("evaluation trajectory must be a mapping or expose to_dict()")
  if config.evaluation_mode == REWARD_ONLY:
    raw_trajectory = reward_only_trajectory(raw_trajectory)
  status = raw_trajectory.get("status", "UNKNOWN")
  if isinstance(status, enum.Enum):
    status = status.name
  reward = float(raw_trajectory.get("reward", 0.0))
  if not math.isfinite(reward):
    raise ValueError("evaluation reward must be finite")
  key = task_key(entry)
  return {
      "schema": TRAJECTORY_SCHEMA,
      "config_fingerprint": config.fingerprint,
      "run_tag": config.run_tag,
      "trajectory_mode": config.trajectory_mode,
      "sampled_by": config.sampled_by,
      "sampling_rng_mode": config.sampling_rng_mode,
      "engine_seed": config.seed_base,
      "task_key": key,
      "instance_id": entry.get("instance_id"),
      "docker_image": key,
      "sample_index": sample_index,
      "sample_nonce": config.sample_nonce(key, sample_index),
      "status": str(status),
      "reward": reward,
      "solved": str(status) in VALID_STATUSES and reward == 1.0,
      "valid": str(status) in VALID_STATUSES,
      "elapsed_secs": float(elapsed_secs),
      "trajectory": serializable(raw_trajectory),
  }


def append_record(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
  target = Path(path)
  target.parent.mkdir(parents=True, exist_ok=True)
  payload = (
      json.dumps(serializable(record), sort_keys=True, separators=(",", ":"))
      + "\n"
  ).encode("utf-8")
  with target.open("ab") as output:
    output.write(payload)
    output.flush()
    os.fsync(output.fileno())


def load_records(
    paths: Iterable[str | os.PathLike[str]],
    *,
    config: EvalConfig,
    allowed_task_keys: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
  """Loads exact-fingerprint records and rejects ambiguous resume state."""
  allowed = None if allowed_task_keys is None else set(allowed_task_keys)
  records: list[dict[str, Any]] = []
  seen: set[tuple[str, int]] = set()
  for path in sorted(Path(item) for item in paths):
    try:
      lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
      continue
    for line_number, line in enumerate(lines, 1):
      if not line.strip():
        continue
      try:
        record = json.loads(line)
      except json.JSONDecodeError:
        if line_number == len(lines):
          continue
        raise ValueError(f"invalid JSON before trailing line in {path}")
      if record.get("schema") != TRAJECTORY_SCHEMA:
        raise ValueError(f"unexpected evaluation schema in {path}")
      if record.get("config_fingerprint") != config.fingerprint:
        raise ValueError(f"evaluation resume fingerprint mismatch in {path}")
      key = str(record.get("task_key", ""))
      sample_index = record.get("sample_index")
      if not key or not isinstance(sample_index, int):
        raise ValueError(f"evaluation resume key is malformed in {path}")
      if allowed is not None and key not in allowed:
        raise ValueError(f"evaluation resume contains an out-of-shard task: {key}")
      if not 0 <= sample_index < config.n_sample:
        raise ValueError("evaluation resume sample index is out of range")
      identity = (key, sample_index)
      if identity in seen:
        raise ValueError(f"duplicate evaluation sample identity: {identity}")
      seen.add(identity)
      records.append(record)
  return records


def remaining_samples(
    entries: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    config: EvalConfig,
) -> list[tuple[Mapping[str, Any], int]]:
  existing = {
      (str(record["task_key"]), int(record["sample_index"]))
      for record in records
  }
  result = []
  for entry in entries:
    key = task_key(entry)
    for sample_index in range(config.n_sample):
      if (key, sample_index) not in existing:
        result.append((entry, sample_index))
  return result


def aggregate_tasks(
    entries: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    config: EvalConfig,
) -> list[dict[str, Any]]:
  """Classifies exact-N task results without treating invalid runs as failures."""
  by_task: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
  for record in records:
    by_task[str(record["task_key"])].append(record)
  reports = []
  for entry in entries:
    key = task_key(entry)
    samples = sorted(by_task[key], key=lambda item: int(item["sample_index"]))
    n = len(samples)
    valid = [item for item in samples if item.get("valid") is True]
    valid_n = len(valid)
    k = sum(item.get("solved") is True for item in valid)
    if n < config.n_sample:
      category = "incomplete"
    elif valid_n == 0:
      category = "broken"
    elif valid_n != config.n_sample:
      category = "incomplete"
    elif k == 0:
      category = "all_fail"
    elif k == config.n_sample:
      category = "all_pass"
    else:
      category = "partial"
    reports.append({
        "schema": REPORT_SCHEMA,
        "config_fingerprint": config.fingerprint,
        "trajectory_mode": config.trajectory_mode,
        "sampled_by": config.sampled_by,
        "sampling_rng_mode": config.sampling_rng_mode,
        "engine_seed": config.seed_base,
        "task_key": key,
        "instance_id": entry.get("instance_id"),
        "docker_image": key,
        "category": category,
        "k": k,
        "n": n,
        "valid_n": valid_n,
        "invalid_n": n - valid_n,
        "missing_n": config.n_sample - n,
        "solve_ratio": k / valid_n if valid_n else None,
        "status_histogram": dict(sorted(collections.Counter(
            str(item.get("status", "UNKNOWN")) for item in samples
        ).items())),
    })
  return reports


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
  payload = b"".join(
      (
          json.dumps(serializable(record), sort_keys=True, separators=(",", ":"))
          + "\n"
      ).encode("utf-8")
      for record in records
  )
  digest = hashlib.sha256(payload).hexdigest()
  path.parent.mkdir(parents=True, exist_ok=True)
  try:
    with path.open("xb") as output:
      output.write(payload)
      output.flush()
      os.fsync(output.fileno())
  except FileExistsError:
    if path.read_bytes() != payload:
      raise ValueError(
          f"existing evaluation evidence differs from exact payload: {path}"
      )
  return digest


def write_reports(
    output_dir: str | os.PathLike[str],
    reports: Sequence[Mapping[str, Any]],
    *,
    config: EvalConfig,
) -> dict[str, Any]:
  """Writes immutable task tiers and a digest-bearing campaign summary."""
  root = Path(output_dir)
  by_category = collections.Counter(str(item["category"]) for item in reports)
  q4_learnable = [item for item in reports if item["category"] == "partial"]
  q32_candidates = [
      item for item in reports if item["category"] in ("partial", "all_fail")
  ]
  sets = {
      "complete": list(reports),
      "q4_learnable": q4_learnable,
      "q32_candidates": q32_candidates,
      "all_pass": [item for item in reports if item["category"] == "all_pass"],
      "all_fail": [item for item in reports if item["category"] == "all_fail"],
      "broken": [item for item in reports if item["category"] == "broken"],
      "incomplete": [item for item in reports if item["category"] == "incomplete"],
  }
  digests = {}
  paths = {}
  for name, items in sets.items():
    path = root / f"{config.run_tag}.{name}.jsonl"
    paths[name] = str(path)
    digests[name] = _write_jsonl(path, items)
  valid_trajectories = sum(int(item["valid_n"]) for item in reports)
  solved_trajectories = sum(int(item["k"]) for item in reports)
  summary = {
      "schema": SUMMARY_SCHEMA,
      "config": config.canonical_record(),
      "config_fingerprint": config.fingerprint,
      "trajectory_mode": config.trajectory_mode,
      "sampled_by": config.sampled_by,
      "sampling_rng_mode": config.sampling_rng_mode,
      "engine_seed": config.seed_base,
      "run_tag": config.run_tag,
      "tasks": len(reports),
      "category_counts": dict(sorted(by_category.items())),
      "valid_trajectories": valid_trajectories,
      "solved_trajectories": solved_trajectories,
      "solve_ratio": (
          solved_trajectories / valid_trajectories
          if valid_trajectories
          else None
      ),
      "paths": paths,
      "sha256": digests,
  }
  summary_path = root / f"{config.run_tag}.summary.json"
  payload = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
  try:
    with summary_path.open("xb") as output:
      output.write(payload)
      output.flush()
      os.fsync(output.fileno())
  except FileExistsError:
    if summary_path.read_bytes() != payload:
      raise ValueError(
          "existing evaluation summary differs from exact payload: "
          f"{summary_path}"
      )
  summary["summary_path"] = str(summary_path)
  summary["summary_sha256"] = hashlib.sha256(payload).hexdigest()
  return summary
