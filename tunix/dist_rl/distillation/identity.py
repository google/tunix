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

"""Checkpoint contract for an OPD objective and its declared teacher identity.

A revision is the caller's declaration of the frozen teacher weights, such as
an immutable snapshot digest. Neither a revision nor a local tokenizer digest
proves which weights a remote endpoint actually serves. An endpoint URL is not
a model identity and is deliberately excluded from this contract.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import dataclasses
import json
from typing import Any

from tunix.dist_rl.distillation import on_policy

METADATA_KEY = "opd_manifest"
_SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True, kw_only=True)
class TeacherIdentity:
  """Caller-declared frozen model revision and local tokenizer semantics hash.

  Omitted fields allow a fresh run, but its checkpoints cannot be resumed safely
  with a positive OPD coefficient. Declare both before the first run when resume
  is required; changing the declaration later is an objective change.
  """

  declared_revision: str | None = None
  tokenizer_fingerprint: str | None = None

  def __post_init__(self) -> None:
    for name in ("declared_revision", "tokenizer_fingerprint"):
      value = getattr(self, name)
      if value is not None and (
          not isinstance(value, str) or not value.strip()
      ):
        raise ValueError(f"OPD teacher {name} must be a nonempty string.")


def build_manifest(
    config: on_policy.OnPolicyDistillationConfig | None,
    *,
    sampling_contract: Mapping[str, Any],
    teacher_identity: TeacherIdentity | None = None,
) -> dict[str, Any] | None:
  """Snapshots the effective objective and sampling config for checkpointing."""
  if config is None:
    if teacher_identity is not None:
      raise ValueError("opd_teacher_identity requires an OPD configuration.")
    return None
  sampling = dict(sampling_contract)
  # These two spellings both disable top-k filtering.
  if sampling.get("top_k") in (-1, 0):
    sampling["top_k"] = 0
  teacher = None
  if config.coefficient > 0:
    identity = teacher_identity or TeacherIdentity()
    teacher = dataclasses.asdict(identity)
  manifest = {
      "schema_version": _SCHEMA_VERSION,
      "estimator": "sampled_token_reverse_kl",
      "coefficient": float(config.coefficient),
      "task_reward_coefficient": float(config.task_reward_coefficient),
      "sampling": sampling,
      "teacher": teacher,
  }
  # Reject non-serializable or non-finite contract values now, rather than at
  # checkpoint time, and detach all nested values from the caller.
  return json.loads(json.dumps(manifest, allow_nan=False))


def has_resumable_identity(manifest: Mapping[str, Any] | None) -> bool:
  """Whether a manifest records all identity fields required for resume."""
  if manifest is None or manifest["coefficient"] == 0:
    return True
  teacher = manifest["teacher"]
  return isinstance(teacher, Mapping) and all(
      isinstance(teacher.get(name), str) and bool(teacher[name].strip())
      for name in ("declared_revision", "tokenizer_fingerprint")
  )


def checkpoint_metadata(
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
  """Returns a detached OPD metadata entry; disabled OPD adds no keys."""
  return {} if manifest is None else {METADATA_KEY: copy.deepcopy(manifest)}


def _changed_fields(expected: Any, actual: Any, prefix: str = "") -> list[str]:
  if isinstance(expected, Mapping) and isinstance(actual, Mapping):
    changed = []
    for key in sorted(expected.keys() | actual.keys()):
      path = f"{prefix}.{key}" if prefix else key
      if key not in expected or key not in actual:
        changed.append(path)
      else:
        changed.extend(_changed_fields(expected[key], actual[key], path))
    return changed
  return [] if expected == actual else [prefix]


def validate_resume_manifest(
    current_manifest: Mapping[str, Any] | None,
    restored_metadata: Mapping[str, Any] | None,
) -> None:
  """Rejects unknown or changed OPD targets before any rollout is dispatched.

  Legacy checkpoints without this metadata remain compatible with ordinary RL.
  They cannot establish an OPD target, so resuming them with OPD is rejected.
  Deliberately changing a target requires a fresh run rather than resume.
  """
  metadata = restored_metadata if isinstance(restored_metadata, Mapping) else {}
  stored = metadata.get(METADATA_KEY)
  if current_manifest is None:
    if METADATA_KEY in metadata:
      raise ValueError("Cannot resume an OPD checkpoint with OPD disabled.")
    return
  if METADATA_KEY not in metadata:
    raise ValueError("Cannot resume OPD: checkpoint has no opd_manifest.")
  if (
      not isinstance(stored, Mapping)
      or type(stored.get("schema_version")) is not int
      or stored["schema_version"] != _SCHEMA_VERSION
  ):
    raise ValueError("Cannot resume OPD: unsupported or invalid opd_manifest.")
  changed = _changed_fields(current_manifest, stored)
  if changed:
    raise ValueError(
        "Cannot resume OPD: checkpoint objective differs in "
        + ", ".join(changed)
        + ". Start a fresh run to change the objective."
    )
  if not has_resumable_identity(current_manifest):
    raise ValueError(
        "Cannot resume OPD: a positive coefficient requires a declared teacher"
        " revision and tokenizer fingerprint recorded before the first run."
        " The checkpoint's remote teacher identity is unknown."
    )
