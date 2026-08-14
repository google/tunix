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

try:
  from .r2egym_action_compat import Q4_R2EGYM_COMPAT_MODE
  from .r2egym_action_compat import canonicalize_r2egym_action
except ImportError:
  from r2egym_action_compat import Q4_R2EGYM_COMPAT_MODE
  from r2egym_action_compat import canonicalize_r2egym_action


CONFIG_SCHEMA = "canon.p46.deepswe-eval.config.v3"
TRAJECTORY_SCHEMA = "canon.p46.deepswe-eval.trajectory.v5"
REPORT_SCHEMA = "canon.p46.deepswe-eval.task-report.v3"
SUMMARY_SCHEMA = "canon.p46.deepswe-eval.summary.v3"
CAMPAIGN_SCHEMA = "canon.p46.deepswe-eval.campaign-summary.v1"
REWARD_ONLY = "reward_only"
REWARD_ONLY_TRAJECTORY_MODE = "reward_only_no_logprobs"
LOGPROB_OBSERVER = "logprob_observer"
LOGPROB_OBSERVER_TRAJECTORY_MODE = "observer_with_sampled_logprobs"
# These are completed model outcomes under the signed evaluation budgets.  A
# retry would resample the same identity after an observed failure and bias the
# N16 solve rate. MODEL_TIMEOUT is also a completed unsolved result because the
# signed campaign measures success under a fixed per-call wall-clock budget.
# Env/reward failures remain invalid and retry.
VALID_STATUSES = frozenset({
    "SUCCEEDED",
    "MAX_STEPS_REACHED",
    "MAX_CONTEXT_LIMIT_REACHED",
    "MODEL_TIMEOUT",
    "TIMEOUT",
})
_MODEL_ACTION_OBSERVATION_ERROR = re.compile(
    r"(?:"
    r"(?:file_editor|search): error: unrecognized arguments:.*"
    r"(?:--command(?:=|\s)|--(?:view|create|str_replace|insert|undo_edit)\b|"
    r"<parameter=)"
    r"|cannot open /parameter"
    r")",
    re.I,
)
_MODEL_ACTION_SYNTAX = re.compile(
    r"(?:"
    r"<parameter\s*=\s*[A-Za-z_][A-Za-z0-9_-]*=[^>\r\n]+>"
    r"|<function=file_editor>.*?"
    r"<parameter\s*=\s*(?:view|create|str_replace|insert|undo_edit)\s*>"
    r")",
    re.I | re.S,
)
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
  action_compat_mode: str = Q4_R2EGYM_COMPAT_MODE

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
          "action_compat_mode": Q4_R2EGYM_COMPAT_MODE,
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
          "action_compat_mode": Q4_R2EGYM_COMPAT_MODE,
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
        else (("64",) if self.parity_canary else ("64", "128"))
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


def trajectory_infrastructure_error(
    trajectory: Mapping[str, Any],
) -> str | None:
  """Returns a reason only for structurally malformed harness output."""
  steps = trajectory.get("steps", [])
  if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
    return "malformed_steps"
  return None


def trajectory_action_diagnostics(
    trajectory: Mapping[str, Any],
) -> tuple[int, int]:
  """Counts deterministic repairs and model-visible tool syntax failures."""
  repairs = 0
  model_action_errors = 0
  steps = trajectory.get("steps", [])
  if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
    return repairs, model_action_errors
  for step in steps:
    if not isinstance(step, Mapping):
      continue
    response = step.get("model_response")
    if isinstance(response, str):
      _, count = canonicalize_r2egym_action(response)
      repairs += count
      if _MODEL_ACTION_SYNTAX.search(response):
        model_action_errors += 1
    observation = step.get("observation")
    if (
        isinstance(observation, str)
        and _MODEL_ACTION_OBSERVATION_ERROR.search(observation)
    ):
      model_action_errors += 1
  return repairs, model_action_errors


def trajectory_record(
    config: EvalConfig,
    *,
    entry: Mapping[str, Any],
    sample_index: int,
    attempt_index: int = 0,
    trajectory: Any,
    elapsed_secs: float,
) -> dict[str, Any]:
  """Builds one complete, redacted and resume-addressable trajectory record."""
  config.validate()
  if attempt_index < 0:
    raise ValueError("evaluation attempt_index must be nonnegative")
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
  infrastructure_error = trajectory_infrastructure_error(raw_trajectory)
  action_compat_repairs, model_action_errors = trajectory_action_diagnostics(
      raw_trajectory
  )
  status = str(status)
  valid = status in VALID_STATUSES and infrastructure_error is None
  if valid and status == "MODEL_TIMEOUT":
    validity_reason = "completed_model_timeout"
  elif valid:
    validity_reason = (
        "completed_with_model_action_errors"
        if model_action_errors
        else "completed_under_signed_budget"
    )
  else:
    validity_reason = infrastructure_error or "retryable_runtime_failure"
  return {
      "schema": TRAJECTORY_SCHEMA,
      "config_fingerprint": config.fingerprint,
      "run_tag": config.run_tag,
      "trajectory_mode": config.trajectory_mode,
      "action_compat_mode": config.action_compat_mode,
      "action_compat_repairs": action_compat_repairs,
      "model_action_errors": model_action_errors,
      "sampled_by": config.sampled_by,
      "sampling_rng_mode": config.sampling_rng_mode,
      "engine_seed": config.seed_base,
      "task_key": key,
      "instance_id": entry.get("instance_id"),
      "docker_image": key,
      "sample_index": sample_index,
      "attempt_index": attempt_index,
      "sample_nonce": config.sample_nonce(key, sample_index),
      "status": status,
      "reward": reward,
      "solved": valid and reward == 1.0,
      "valid": valid,
      "validity_reason": validity_reason,
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
  """Loads exact-fingerprint attempts and rejects ambiguous resume state.

  An invalid attempt is durable evidence, but it does not complete its sample
  identity. A later consecutive attempt may retry that identity. Once one
  attempt is valid, every further attempt for the identity is rejected.
  """
  allowed = None if allowed_task_keys is None else set(allowed_task_keys)
  records: list[dict[str, Any]] = []
  attempts: dict[tuple[str, int], list[dict[str, Any]]] = (
      collections.defaultdict(list)
  )
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
      prior = attempts[identity]
      attempt_index = record.get("attempt_index", 0)
      if not isinstance(attempt_index, int) or attempt_index < 0:
        raise ValueError(f"evaluation attempt index is malformed: {identity}")
      if attempt_index != len(prior):
        raise ValueError(
            "evaluation attempt indices must be consecutive: "
            f"identity={identity} expected={len(prior)} actual={attempt_index}"
        )
      if prior and prior[-1].get("valid") is True:
        raise ValueError(
            f"duplicate valid evaluation sample identity: {identity}"
        )
      prior.append(record)
      records.append(record)
  return records


def remaining_samples(
    entries: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    config: EvalConfig,
) -> list[tuple[Mapping[str, Any], int, int]]:
  valid = {
      (str(record["task_key"]), int(record["sample_index"]))
      for record in records
      if record.get("valid") is True
  }
  attempts = collections.Counter(
      (str(record["task_key"]), int(record["sample_index"]))
      for record in records
  )
  result = []
  for entry in entries:
    key = task_key(entry)
    for sample_index in range(config.n_sample):
      identity = (key, sample_index)
      if identity not in valid:
        result.append((entry, sample_index, attempts[identity]))
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
    task_attempts = by_task[key]
    by_sample: dict[int, Mapping[str, Any]] = {}
    for record in task_attempts:
      sample_index = int(record["sample_index"])
      selected = by_sample.get(sample_index)
      if selected is None or record.get("valid") is True:
        by_sample[sample_index] = record
    samples = [by_sample[index] for index in sorted(by_sample)]
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
        "attempts": len(task_attempts),
        "invalid_attempts": sum(
            item.get("valid") is not True for item in task_attempts
        ),
        "missing_n": config.n_sample - n,
        "solve_ratio": k / valid_n if valid_n else None,
        "status_histogram": dict(sorted(collections.Counter(
            str(item.get("status", "UNKNOWN")) for item in task_attempts
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
  attempts = sum(int(item["attempts"]) for item in reports)
  invalid_attempts = sum(int(item["invalid_attempts"]) for item in reports)
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
      "attempts": attempts,
      "invalid_attempts": invalid_attempts,
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
  records = []
  for line_number, line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), 1
  ):
    if not line.strip():
      continue
    try:
      value = json.loads(line)
    except json.JSONDecodeError as error:
      raise ValueError(
          f"invalid campaign report JSON at {path}:{line_number}"
      ) from error
    if not isinstance(value, dict):
      raise ValueError(f"campaign report row is not an object: {path}")
    records.append(value)
  return records


def _finalize_campaign(
    summary_paths: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    *,
    expected_tasks: int,
    expected_logical_shards: int,
    tasks_per_logical_shard: int,
) -> dict[str, Any]:
  """Merges only a complete, exact-N, single-contract evaluation campaign."""
  if (
      expected_tasks <= 0
      or expected_logical_shards <= 0
      or tasks_per_logical_shard <= 0
  ):
    raise ValueError("campaign expectations must be positive")
  if len(summary_paths) != expected_logical_shards:
    raise ValueError(
        "campaign requires every logical summary: "
        f"expected={expected_logical_shards} actual={len(summary_paths)}"
    )

  summaries: dict[int, dict[str, Any]] = {}
  common_config: dict[str, Any] | None = None
  all_reports: list[dict[str, Any]] = []
  input_evidence = []
  seen_tasks: set[str] = set()
  task_order: dict[str, tuple[int, int]] = {}
  config_fields = {field.name for field in dataclasses.fields(EvalConfig)}

  for item in summary_paths:
    summary_path = Path(item).resolve()
    summary_bytes = summary_path.read_bytes()
    try:
      summary = json.loads(summary_bytes)
    except json.JSONDecodeError as error:
      raise ValueError(f"invalid campaign summary JSON: {summary_path}") from error
    if not isinstance(summary, dict) or summary.get("schema") != SUMMARY_SCHEMA:
      raise ValueError(f"unexpected campaign input schema: {summary_path}")
    config_record = summary.get("config")
    if not isinstance(config_record, dict):
      raise ValueError(f"campaign summary lacks config: {summary_path}")
    try:
      config = EvalConfig(**{
          name: config_record[name] for name in config_fields
      })
    except (KeyError, TypeError) as error:
      raise ValueError(
          f"campaign summary config is malformed: {summary_path}"
      ) from error
    config.validate()
    if config.onehost_probe or config.parity_canary:
      raise ValueError("campaign finalization rejects probe/canary summaries")
    shard_index = config.shard_index
    if shard_index in summaries:
      raise ValueError(f"duplicate campaign logical shard: {shard_index}")
    normalized = config.canonical_record()
    normalized.pop("shard_index")
    if common_config is None:
      common_config = normalized
    elif normalized != common_config:
      raise ValueError("campaign logical summaries changed evaluation contract")
    if summary.get("config_fingerprint") != config.fingerprint:
      raise ValueError("campaign summary config fingerprint mismatch")
    for field_name, expected in (
        ("trajectory_mode", config.trajectory_mode),
        ("sampled_by", config.sampled_by),
        ("sampling_rng_mode", config.sampling_rng_mode),
        ("engine_seed", config.seed_base),
    ):
      if summary.get(field_name) != expected:
        raise ValueError(f"campaign summary {field_name} mismatch")

    expected_shard_tasks = min(
        tasks_per_logical_shard,
        expected_tasks - shard_index * tasks_per_logical_shard,
    )
    if expected_shard_tasks <= 0 or summary.get("tasks") != expected_shard_tasks:
      raise ValueError(
          f"campaign logical shard {shard_index} has wrong task count"
      )
    expected_valid = expected_shard_tasks * config.n_sample
    attempts = summary.get("attempts")
    invalid_attempts = summary.get("invalid_attempts")
    if (
        summary.get("valid_trajectories") != expected_valid
        or not isinstance(attempts, int)
        or not isinstance(invalid_attempts, int)
        or attempts != expected_valid + invalid_attempts
    ):
      raise ValueError(
          f"campaign logical shard {shard_index} is not exact valid N"
      )
    categories = summary.get("category_counts")
    if not isinstance(categories, dict) or sum(categories.values()) != expected_shard_tasks:
      raise ValueError("campaign category counts are malformed")
    if categories.get("broken", 0) or categories.get("incomplete", 0):
      raise ValueError("campaign contains broken or incomplete task reports")

    paths = summary.get("paths")
    digests = summary.get("sha256")
    if not isinstance(paths, dict) or not isinstance(digests, dict):
      raise ValueError("campaign summary lacks report evidence")
    complete_path = Path(str(paths.get("complete", "")))
    if not complete_path.is_absolute():
      complete_path = summary_path.parent / complete_path
    complete_path = complete_path.resolve()
    complete_digest = str(digests.get("complete", ""))
    if not _SHA256.fullmatch(complete_digest):
      raise ValueError("campaign complete-report digest is malformed")
    if sha256_file(complete_path) != complete_digest:
      raise ValueError("campaign complete-report digest mismatch")
    reports = _read_jsonl(complete_path)
    if len(reports) != expected_shard_tasks:
      raise ValueError("campaign complete-report task count mismatch")
    report_categories: collections.Counter[str] = collections.Counter()
    report_solved = 0
    report_attempts = 0
    report_invalid_attempts = 0
    for local_index, report in enumerate(reports):
      key = str(report.get("task_key", ""))
      if not key or key in seen_tasks:
        raise ValueError(f"campaign task identity is missing or duplicate: {key}")
      seen_tasks.add(key)
      if (
          report.get("schema") != REPORT_SCHEMA
          or report.get("config_fingerprint") != config.fingerprint
          or report.get("valid_n") != config.n_sample
          or report.get("n") != config.n_sample
          or report.get("invalid_n") != 0
          or report.get("missing_n") != 0
          or report.get("category") not in ("partial", "all_fail", "all_pass")
      ):
        raise ValueError(f"campaign task report is not exact valid N: {key}")
      k = report.get("k")
      report_attempt_count = report.get("attempts")
      report_invalid_count = report.get("invalid_attempts")
      expected_category = (
          "all_fail"
          if k == 0
          else ("all_pass" if k == config.n_sample else "partial")
      )
      if (
          not isinstance(k, int)
          or not 0 <= k <= config.n_sample
          or report.get("category") != expected_category
          or not isinstance(report_attempt_count, int)
          or not isinstance(report_invalid_count, int)
          or report_attempt_count != config.n_sample + report_invalid_count
          or report.get("trajectory_mode") != config.trajectory_mode
          or report.get("sampled_by") != config.sampled_by
          or report.get("sampling_rng_mode") != config.sampling_rng_mode
          or report.get("engine_seed") != config.seed_base
      ):
        raise ValueError(f"campaign task report metrics are inconsistent: {key}")
      task_order[key] = (shard_index, local_index)
      report_categories[str(report["category"])] += 1
      report_solved += k
      report_attempts += report_attempt_count
      report_invalid_attempts += report_invalid_count
    if (
        dict(report_categories) != categories
        or summary.get("solved_trajectories") != report_solved
        or attempts != report_attempts
        or invalid_attempts != report_invalid_attempts
    ):
      raise ValueError("campaign summary disagrees with complete task reports")
    summaries[shard_index] = summary
    all_reports.extend(reports)
    input_evidence.append({
        "logical_shard_index": shard_index,
        "summary_path": str(summary_path),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "complete_path": str(complete_path),
        "complete_sha256": complete_digest,
    })

  if set(summaries) != set(range(expected_logical_shards)):
    raise ValueError("campaign logical shard indices are incomplete")
  if len(all_reports) != expected_tasks or len(seen_tasks) != expected_tasks:
    raise ValueError("campaign global task cardinality mismatch")

  input_evidence.sort(key=lambda item: item["logical_shard_index"])
  all_reports.sort(key=lambda item: task_order[str(item["task_key"])])
  categories = collections.Counter(
      str(report["category"]) for report in all_reports
  )
  sets = {
      "complete": all_reports,
      "q4_learnable": [
          report for report in all_reports if report["category"] == "partial"
      ],
      "q32_candidates": [
          report
          for report in all_reports
          if report["category"] in ("partial", "all_fail")
      ],
      "all_pass": [
          report for report in all_reports if report["category"] == "all_pass"
      ],
      "all_fail": [
          report for report in all_reports if report["category"] == "all_fail"
      ],
  }
  root = Path(output_dir).resolve()
  paths = {}
  digests = {}
  for name, reports in sets.items():
    path = root / f"p46-campaign.{name}.jsonl"
    paths[name] = str(path)
    digests[name] = _write_jsonl(path, reports)

  valid_trajectories = sum(
      int(summary["valid_trajectories"]) for summary in summaries.values()
  )
  attempts = sum(int(summary["attempts"]) for summary in summaries.values())
  invalid_attempts = sum(
      int(summary["invalid_attempts"]) for summary in summaries.values()
  )
  solved = sum(int(report["k"]) for report in all_reports)
  campaign = {
      "schema": CAMPAIGN_SCHEMA,
      "config_without_logical_shard_index": common_config,
      "logical_shards": expected_logical_shards,
      "tasks": expected_tasks,
      "n_sample": 16,
      "valid_trajectories": valid_trajectories,
      "solved_trajectories": solved,
      "attempts": attempts,
      "invalid_attempts": invalid_attempts,
      "solve_ratio": solved / valid_trajectories,
      "category_counts": dict(sorted(categories.items())),
      "input_summaries": input_evidence,
      "paths": paths,
      "sha256": digests,
  }
  summary_path = root / "p46-campaign.summary.json"
  payload = (json.dumps(campaign, indent=2, sort_keys=True) + "\n").encode(
      "utf-8"
  )
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  try:
    with summary_path.open("xb") as output:
      output.write(payload)
      output.flush()
      os.fsync(output.fileno())
  except FileExistsError:
    if summary_path.read_bytes() != payload:
      raise ValueError("existing campaign summary differs from exact payload")
  campaign["summary_path"] = str(summary_path)
  campaign["summary_sha256"] = hashlib.sha256(payload).hexdigest()
  return campaign


def finalize_campaign(
    summary_paths: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
  """Finalizes the signed 1851-task x N16 production evaluation campaign."""
  return _finalize_campaign(
      summary_paths,
      output_dir,
      expected_tasks=1851,
      expected_logical_shards=58,
      tasks_per_logical_shard=32,
  )
