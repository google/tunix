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

"""Immutable P64 FrozenLake training-capsule capture and diagnostic replay.

The capsule freezes the exact tensorized input to the segmented trainer.  It
does not preserve the serving scheduler and a replay is therefore diagnostic
evidence, never a fresh Zero-TIM certification.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


MODE_ENV = "CANON_P64_TRAINING_CAPSULE_MODE"
PATH_ENV = "CANON_P64_TRAINING_CAPSULE"
GCS_URI_ENV = "CANON_P64_TRAINING_CAPSULE_GCS_URI"
SHA256_ENV = "CANON_P64_TRAINING_CAPSULE_SHA256"
MODEL_BINDING_SHA256_ENV = "CANON_P64_MODEL_BINDING_SHA256"

SCHEMA = "canon-p64-training-capsule-v1"
MODEL_BINDING_SCHEMA = "canon-p64-model-binding-v1"
_SHA_RE = re.compile(r"[0-9a-f]{64}")

_TRAIN_FIELDS = (
    "prompt_ids",
    "prompt_mask",
    "completion_ids",
    "completion_mask",
    "advantages",
    "ref_per_token_logps",
    "old_per_token_logps",
    "segment_ids",
    "segment_positions",
    "is_update_step",
    "sampler_is_weights",
    "policy_version",
    "completion_valid_mask",
)
_OBSERVED_FIELDS = (
    "s_decode",
    "s_prefill",
    "t_old",
    "action_mask",
    "completion_valid_mask",
    "prompt_mask",
    "tokens",
    "policy_version",
    "sampling_values",
)


class P64TrainingCapsuleError(RuntimeError):
  """Raised when a P64 capsule cannot satisfy its fail-closed contract."""


def mode(environ: Mapping[str, str] | None = None) -> str:
  values = os.environ if environ is None else environ
  value = values.get(MODE_ENV, "")
  if value not in ("", "capture", "replay"):
    raise P64TrainingCapsuleError(
        f"{MODE_ENV} must be unset/capture/replay, got {value!r}"
    )
  return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
  return bool(mode(environ))


def is_replay(environ: Mapping[str, str] | None = None) -> bool:
  return mode(environ) == "replay"


def reverse_group_limit(
    group_count: int, environ: Mapping[str, str] | None = None
) -> int:
  """Returns the admitted P64 reverse scope without changing forward scope."""
  active_mode = mode(environ)
  if active_mode == "replay":
    if group_count != 32:
      raise P64TrainingCapsuleError(
          f"P64 replay requires 32 registered groups, got {group_count}"
      )
    return 1
  return group_count


def _sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
  return _sha256_bytes(np.ascontiguousarray(value).tobytes())


def file_sha256(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def model_binding_path(path: str | Path) -> Path:
  return Path(f"{Path(path)}.model.json")


def _require_path(environ: Mapping[str, str]) -> Path:
  raw = environ.get(PATH_ENV, "")
  if not raw or not os.path.isabs(raw) or not raw.endswith(".npz"):
    raise P64TrainingCapsuleError(
        f"{PATH_ENV} must be an absolute .npz path"
    )
  return Path(raw)


def _require_p64_identity(environ: Mapping[str, str]) -> None:
  exact = {
      "CANON_P64_P45_NUMERIC_DEBUG": "1",
      "CANON_PROFILE_FILE": (
          "cluster/profiles/"
          "qwen3-8b-dp8-tp8-frozenlake-p64-debug.env"
      ),
      "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
      "CANON_DP_SIZE": "8",
      "CANON_TP_SIZE": "8",
      "CANON_GLOBAL_TRAJECTORIES": "256",
      "CANON_LOCAL_TRAJECTORIES": "32",
      "CANON_LOGPROB_M": "256",
      "MIN_TOKEN_BUCKET": "2048",
      "FL_SHARED_MESH": "8,8",
      "CANON_P33_RUN_STAGE": "backward-no-commit",
      "CANON_P33_NO_COMMIT": "1",
      "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
      "CANON_P38_FIXED_LM_HEAD": "1",
      "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
      "CANON_V1_HP_FULL": "0",
  }
  changed = {
      name: environ.get(name)
      for name, expected in exact.items()
      if environ.get(name) != expected
  }
  if changed:
    raise P64TrainingCapsuleError(
        f"P64 training-capsule identity drifted: {changed}"
    )
  uri = environ.get(GCS_URI_ENV, "")
  if not re.fullmatch(
      r"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p64/"
      r"[a-z0-9][a-z0-9-]*/training-capsule\.npz",
      uri,
  ):
    raise P64TrainingCapsuleError(
        f"{GCS_URI_ENV} is outside the registered P64 evidence root"
    )


def _as_array(value: Any) -> np.ndarray:
  return np.ascontiguousarray(np.asarray(value))


def _identity(environ: Mapping[str, str]) -> dict[str, Any]:
  return {
      "capture_source_commit": environ.get("CANON_EXPECT_COMMIT", ""),
      "capture_run_id": environ.get("CANON_RUN_ID", ""),
      "profile": environ.get("CANON_PROFILE_FILE", ""),
      "workload": environ.get("CANON_P32_WORKLOAD", ""),
      "model_dir_name": environ.get("CANON_MODEL_DIR_NAME", ""),
      "mesh": environ.get("FL_SHARED_MESH", ""),
      "dp": 8,
      "tp": 8,
      "global_trajectories": 256,
      "local_trajectories": 32,
      "global_M": 2048,
      "local_M": 256,
  }


def _atomic_write_npz(
    path: Path, *, arrays: Mapping[str, np.ndarray], metadata: bytes
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.exists():
    raise P64TrainingCapsuleError(f"refusing to overwrite capsule: {path}")
  temporary = Path(f"{path}.tmp-{os.getpid()}")
  try:
    with temporary.open("xb") as output:
      np.savez_compressed(
          output,
          metadata_json=np.frombuffer(metadata, dtype=np.uint8),
          **arrays,
      )
      output.flush()
      os.fsync(output.fileno())
    try:
      os.link(temporary, path)
    except FileExistsError as exc:
      raise P64TrainingCapsuleError(
          f"refusing to overwrite capsule: {path}"
      ) from exc
  finally:
    if temporary.exists():
      temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.exists():
    raise P64TrainingCapsuleError(f"refusing to overwrite receipt: {path}")
  rendered = (
      json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
  ).encode()
  temporary = Path(f"{path}.tmp-{os.getpid()}")
  try:
    with temporary.open("xb") as output:
      output.write(rendered)
      output.flush()
      os.fsync(output.fileno())
    try:
      os.link(temporary, path)
    except FileExistsError as exc:
      raise P64TrainingCapsuleError(
          f"refusing to overwrite receipt: {path}"
      ) from exc
  finally:
    if temporary.exists():
      temporary.unlink()


def persist(observed: Any, environ: Mapping[str, str] | None = None) -> dict:
  """Atomically freezes one exact 256-row pre-backward P45 train batch."""
  values = os.environ if environ is None else environ
  if mode(values) != "capture":
    raise P64TrainingCapsuleError("P64 capsule persist requires capture mode")
  _require_p64_identity(values)
  path = _require_path(values)
  train_example = observed.train_example
  arrays: dict[str, np.ndarray] = {}
  presence: dict[str, bool] = {}
  for field in _TRAIN_FIELDS:
    value = getattr(train_example, field, None)
    presence[f"train__{field}"] = value is not None
    if value is not None:
      arrays[f"train__{field}"] = _as_array(value)
  for field in _OBSERVED_FIELDS:
    value = getattr(observed, field, None)
    presence[f"observed__{field}"] = value is not None
    if value is not None:
      arrays[f"observed__{field}"] = _as_array(value)

  required = {
      "train__prompt_ids",
      "train__prompt_mask",
      "train__completion_ids",
      "train__completion_mask",
      "train__advantages",
      "train__old_per_token_logps",
      "train__policy_version",
      "train__completion_valid_mask",
      "observed__s_decode",
      "observed__s_prefill",
      "observed__t_old",
      "observed__action_mask",
      "observed__completion_valid_mask",
      "observed__prompt_mask",
      "observed__tokens",
      "observed__policy_version",
      "observed__sampling_values",
  }
  missing = sorted(required - arrays.keys())
  if missing:
    raise P64TrainingCapsuleError(
        f"P64 training capsule is missing required arrays: {missing}"
    )
  prompt_shape = arrays["train__prompt_ids"].shape
  completion_shape = arrays["train__completion_ids"].shape
  if prompt_shape != (256, 4096) or completion_shape != (256, 2048):
    raise P64TrainingCapsuleError(
        "P64 training capsule requires physical P45 tensors "
        f"prompt=(256,4096) completion=(256,2048), got "
        f"{prompt_shape}/{completion_shape}"
    )
  for name in (
      "train__prompt_mask",
      "observed__prompt_mask",
  ):
    if arrays[name].shape != prompt_shape:
      raise P64TrainingCapsuleError(
          f"P64 prompt-aligned array changed shape: {name}={arrays[name].shape}"
      )
  for name in (
      "train__completion_mask",
      "train__old_per_token_logps",
      "train__completion_valid_mask",
      "observed__s_decode",
      "observed__s_prefill",
      "observed__t_old",
      "observed__action_mask",
      "observed__completion_valid_mask",
      "observed__tokens",
  ):
    if arrays[name].shape != completion_shape:
      raise P64TrainingCapsuleError(
          "P64 completion-aligned array changed shape: "
          f"{name}={arrays[name].shape}"
      )
  if not np.array_equal(
      arrays["train__completion_ids"], arrays["observed__tokens"]
  ):
    raise P64TrainingCapsuleError(
        "P64 observed tokens differ from trainer completion ids"
    )
  if not np.array_equal(
      arrays["train__completion_mask"], arrays["observed__action_mask"]
  ):
    raise P64TrainingCapsuleError(
        "P64 observed action mask differs from trainer completion mask"
    )

  metadata = {
      "schema": SCHEMA,
      "evidence_kind": "capture",
      "claim_ceiling": (
          "Replay is backward localization only and is not a fresh "
          "Zero-TIM certification."
      ),
      **_identity(values),
      "rows": 256,
      "source_name": observed.source_name,
      "all_compact_filtered": bool(observed.all_compact_filtered),
      "presence": presence,
      "arrays": {
          name: {
              "shape": list(value.shape),
              "dtype": str(value.dtype),
              "sha256": _array_sha256(value),
              "finite": (
                  bool(np.all(np.isfinite(value)))
                  if np.issubdtype(value.dtype, np.number)
                  else None
              ),
          }
          for name, value in arrays.items()
      },
  }
  metadata_json = json.dumps(
      metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
  ).encode()
  _atomic_write_npz(path, arrays=arrays, metadata=metadata_json)
  result = {
      "path": str(path),
      "sha256": file_sha256(path),
      "rows": 256,
      "arrays": len(arrays),
      "logical_bytes": sum(value.nbytes for value in arrays.values()),
  }
  print(
      "[P64.CAPSULE] capture_ready "
      f"path={path} sha256={result['sha256']} rows=256 "
      f"arrays={result['arrays']} logical_bytes={result['logical_bytes']} "
      "certification=strict-prealignment-source",
      flush=True,
  )
  return result


@dataclass(frozen=True)
class VerifiedTrainingCapsule:
  path: Path
  sha256: str
  metadata: Mapping[str, Any]
  arrays: Mapping[str, np.ndarray]

  def build(self, train_example_cls: Any, observed_cls: Any) -> Any:
    presence = self.metadata["presence"]

    def train_value(field: str) -> Any:
      key = f"train__{field}"
      return self.arrays[key] if presence.get(key) else None

    train_example = train_example_cls(
        **{field: train_value(field) for field in _TRAIN_FIELDS}
    )
    return observed_cls(
        train_example=train_example,
        **{
            field: self.arrays[f"observed__{field}"]
            for field in _OBSERVED_FIELDS
        },
        source_name=str(self.metadata["source_name"]),
        all_compact_filtered=bool(self.metadata["all_compact_filtered"]),
    )


def load_verified(
    environ: Mapping[str, str] | None = None,
) -> VerifiedTrainingCapsule:
  """Loads a hash-bound capsule for diagnostic replay."""
  values = os.environ if environ is None else environ
  if mode(values) != "replay":
    raise P64TrainingCapsuleError("P64 capsule load requires replay mode")
  _require_p64_identity(values)
  path = _require_path(values)
  expected_file_sha = values.get(SHA256_ENV, "")
  if not _SHA_RE.fullmatch(expected_file_sha):
    raise P64TrainingCapsuleError(
        f"{SHA256_ENV} must be exactly 64 lowercase hex in replay mode"
    )
  observed_file_sha = file_sha256(path)
  if observed_file_sha != expected_file_sha:
    raise P64TrainingCapsuleError(
        "P64 capsule file hash mismatch: "
        f"{observed_file_sha}/{expected_file_sha}"
    )
  try:
    with np.load(path, allow_pickle=False) as archive:
      if "metadata_json" not in archive.files:
        raise P64TrainingCapsuleError("P64 capsule metadata is absent")
      metadata = json.loads(archive["metadata_json"].tobytes())
      arrays = {
          name: np.ascontiguousarray(archive[name])
          for name in archive.files
          if name != "metadata_json"
      }
  except (OSError, ValueError, json.JSONDecodeError) as exc:
    if isinstance(exc, P64TrainingCapsuleError):
      raise
    raise P64TrainingCapsuleError(
        f"cannot load P64 training capsule {path}: {exc}"
    ) from exc
  if metadata.get("schema") != SCHEMA:
    raise P64TrainingCapsuleError(
        f"unexpected P64 capsule schema: {metadata.get('schema')!r}"
    )
  identity = _identity(values)
  for name in (
      "profile",
      "workload",
      "model_dir_name",
      "mesh",
      "dp",
      "tp",
      "global_trajectories",
      "local_trajectories",
      "global_M",
      "local_M",
  ):
    if metadata.get(name) != identity[name]:
      raise P64TrainingCapsuleError(
          f"P64 replay identity mismatch for {name}: "
          f"{metadata.get(name)!r}/{identity[name]!r}"
      )
  expected_arrays = metadata.get("arrays")
  presence = metadata.get("presence")
  if not isinstance(expected_arrays, dict) or not isinstance(presence, dict):
    raise P64TrainingCapsuleError(
        "P64 capsule array or presence table is absent"
    )
  expected_names = {
      name for name, present in presence.items() if present is True
  }
  if set(arrays) != expected_names or set(expected_arrays) != expected_names:
    raise P64TrainingCapsuleError(
        "P64 capsule array inventory differs from metadata"
    )
  for name, value in arrays.items():
    expected = expected_arrays.get(name, {})
    if (
        expected.get("shape") != list(value.shape)
        or expected.get("dtype") != str(value.dtype)
        or expected.get("sha256") != _array_sha256(value)
    ):
      raise P64TrainingCapsuleError(
          f"P64 capsule array receipt mismatch: {name}"
      )
  print(
      "[P64.CAPSULE] diagnostic_replay_ready "
      f"path={path} sha256={observed_file_sha} rows={metadata.get('rows')} "
      f"capture_run={metadata.get('capture_run_id')} "
      f"capture_source={metadata.get('capture_source_commit')} "
      f"replay_source={values.get('CANON_EXPECT_COMMIT', '')} "
      "rollout=skipped rescore_b=skipped certification=0",
      flush=True,
  )
  return VerifiedTrainingCapsule(
      path=path,
      sha256=observed_file_sha,
      metadata=metadata,
      arrays=arrays,
  )


def bind_or_verify_model(
    fingerprint: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
  """Binds capture to, or verifies replay against, the live model sample."""
  values = os.environ if environ is None else environ
  active_mode = mode(values)
  if active_mode not in ("capture", "replay"):
    raise P64TrainingCapsuleError(
        "P64 model binding requires capture or replay mode"
    )
  _require_p64_identity(values)
  capsule = _require_path(values)
  capsule_sha = file_sha256(capsule)
  canonical_fingerprint = json.dumps(
      fingerprint, sort_keys=True, separators=(",", ":"), allow_nan=False
  )
  fingerprint_sha = _sha256_bytes(canonical_fingerprint.encode())
  binding = model_binding_path(capsule)
  if active_mode == "capture":
    payload = {
        "schema": MODEL_BINDING_SCHEMA,
        "capsule_sha256": capsule_sha,
        "capture_source_commit": values.get("CANON_EXPECT_COMMIT", ""),
        "capture_run_id": values.get("CANON_RUN_ID", ""),
        "model_dir_name": values.get("CANON_MODEL_DIR_NAME", ""),
        "model_fingerprint_sha256": fingerprint_sha,
        "model_fingerprint": fingerprint,
    }
    _atomic_write_json(binding, payload)
    binding_sha = file_sha256(binding)
  else:
    expected_binding_sha = values.get(MODEL_BINDING_SHA256_ENV, "")
    if not _SHA_RE.fullmatch(expected_binding_sha):
      raise P64TrainingCapsuleError(
          f"{MODEL_BINDING_SHA256_ENV} must be 64 lowercase hex in replay"
      )
    observed_binding_sha = file_sha256(binding)
    if observed_binding_sha != expected_binding_sha:
      raise P64TrainingCapsuleError(
          "P64 model-binding file hash mismatch: "
          f"{observed_binding_sha}/{expected_binding_sha}"
      )
    try:
      payload = json.loads(binding.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
      raise P64TrainingCapsuleError(
          f"cannot read P64 model binding {binding}: {exc}"
      ) from exc
    if (
        payload.get("schema") != MODEL_BINDING_SCHEMA
        or payload.get("capsule_sha256") != capsule_sha
        or payload.get("model_dir_name")
        != values.get("CANON_MODEL_DIR_NAME", "")
        or payload.get("model_fingerprint_sha256") != fingerprint_sha
        or payload.get("model_fingerprint") != fingerprint
    ):
      raise P64TrainingCapsuleError(
          "P64 replay live-model fingerprint differs from capture"
      )
    binding_sha = observed_binding_sha
  result = {
      "mode": active_mode,
      "capsule_sha256": capsule_sha,
      "binding_path": str(binding),
      "binding_sha256": binding_sha,
      "model_fingerprint_sha256": fingerprint_sha,
  }
  print(
      f"[P64.CAPSULE] model_{'bound' if active_mode == 'capture' else 'verified'} "
      f"mode={active_mode} capsule_sha256={capsule_sha} "
      f"binding_sha256={binding_sha} "
      f"model_fingerprint_sha256={fingerprint_sha} sampled_model=1",
      flush=True,
  )
  return result
