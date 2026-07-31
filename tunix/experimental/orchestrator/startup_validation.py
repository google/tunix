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

"""Checking that separate worker processes agree before the run starts.

In one process the workers cannot disagree about the tokenizer, the padding
token, or the sampling temperature, because there is one config object and
they all read it. Once they are separate processes each loads its own, and a
disagreement produces no error anywhere: tokens tokenized one way are scored
under another vocabulary, padding is masked at the wrong id, log-probabilities
are taken at a temperature the tokens were not sampled at. Training proceeds
and the numbers are quietly wrong.

So workers declare these facts through the resources map on their info, and
the fleet reconciles them once, before any work is dispatched. Every
disagreement is collected and reported together, because finding them one
restart at a time is its own kind of failure.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any, Iterable, Mapping, Optional, Sequence

# What a worker must declare about itself before it can be trusted with work.
# Chosen because each one silently corrupts training if it differs across
# workers, rather than failing.
REQUIRED_RESOURCE_KEYS = (
    "tokenizer_hash",
    "pad_id",
    "eos_id",
    "vocab_size",
)

# Checked when present: not every role has a sampling temperature, but those
# that do must agree with the scorers.
OPTIONAL_AGREEMENT_KEYS = (
    "bos_id",
    "temperature",
    "segment_attention",
)


class StartupValidationError(ValueError):
  """One or more workers are misconfigured relative to the rest."""

  def __init__(self, problems: Sequence[str]):
    self.problems = list(problems)
    joined = "\n  - ".join(self.problems)
    super().__init__(
        f"workers do not agree on their configuration:\n  - {joined}"
    )


@dataclasses.dataclass(frozen=True)
class Disagreement:
  """One fact the workers report differently.

  Attributes:
    key: The resource key in question.
    values_by_worker: What each worker said.
  """

  key: str
  values_by_worker: Mapping[str, Any]

  def describe(self) -> str:
    listed = ", ".join(
        f"{worker}={value!r}"
        for worker, value in sorted(self.values_by_worker.items())
    )
    return f"{self.key} differs across workers: {listed}"


def validate_workers(
    workers: Iterable[Any],
    *,
    required_keys: Sequence[str] = REQUIRED_RESOURCE_KEYS,
    agreement_keys: Sequence[str] = OPTIONAL_AGREEMENT_KEYS,
    expected: Optional[Mapping[str, Any]] = None,
) -> list[str]:
  """Returns every configuration problem across `workers`.

  Args:
    workers: Objects exposing `info()`.
    required_keys: Keys every worker must declare, and agree on.
    agreement_keys: Keys that need not be declared, but must agree wherever
      they are.
    expected: Values the orchestrator itself requires, checked against every
      worker that declares them. Catches the case where all workers agree with
      each other and none of them agrees with the run.

  Returns:
    Human-readable problems; empty when everything lines up.
  """
  declared = [(_worker_id(worker), _resources(worker)) for worker in workers]
  if not declared:
    return ["no workers to validate."]

  problems: list[str] = []
  for worker_id, resources in declared:
    missing = [key for key in required_keys if key not in resources]
    if missing:
      problems.append(
          f"{worker_id} does not declare {sorted(missing)}; a worker that"
          " cannot say how it tokenizes cannot be checked against the others."
      )

  for key in (*required_keys, *agreement_keys):
    reported = {
        worker_id: resources[key]
        for worker_id, resources in declared
        if key in resources
    }
    if len(set(map(_hashable, reported.values()))) > 1:
      problems.append(Disagreement(key, reported).describe())

  for key, wanted in (expected or {}).items():
    mismatched = {
        worker_id: resources[key]
        for worker_id, resources in declared
        if key in resources and resources[key] != wanted
    }
    if mismatched:
      listed = ", ".join(
          f"{worker}={value!r}" for worker, value in sorted(mismatched.items())
      )
      problems.append(
          f"{key} should be {wanted!r} for this run, but {listed}."
      )

  return problems


def require_agreement(workers: Iterable[Any], **kwargs) -> None:
  """Raises unless every worker agrees; reports all problems at once.

  Args:
    workers: Objects exposing `info()`.
    **kwargs: Forwarded to `validate_workers`.

  Raises:
    StartupValidationError: If any problem is found.
  """
  problems = validate_workers(workers, **kwargs)
  if problems:
    raise StartupValidationError(problems)


def describe_resources(
    tokenizer_hash: str,
    *,
    pad_id: int,
    eos_id: int,
    vocab_size: int,
    **extra: Any,
) -> dict[str, Any]:
  """Builds the resources map a worker should report.

  Args:
    tokenizer_hash: Identifies the exact vocabulary in use.
    pad_id: Padding token id.
    eos_id: End-of-sequence token id.
    vocab_size: Size of the vocabulary.
    **extra: Optional keys such as `bos_id`, `temperature`, or
      `segment_attention`.

  Returns:
    The map to hand to `WorkerInfo(resources=...)`.
  """
  resources = {
      "tokenizer_hash": tokenizer_hash,
      "pad_id": pad_id,
      "eos_id": eos_id,
      "vocab_size": vocab_size,
  }
  resources.update(extra)
  return resources


def _resources(worker: Any) -> Mapping[str, Any]:
  info = getattr(worker, "info", None)
  if not callable(info):
    return {}
  return getattr(info(), "resources", None) or {}


def _worker_id(worker: Any) -> str:
  info = getattr(worker, "info", None)
  if callable(info):
    try:
      return info().worker_id
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  return str(getattr(worker, "worker_id", worker))


def _hashable(value: Any) -> Any:
  """Makes a reported value comparable even if it arrived as a container."""
  if isinstance(value, (list, tuple)):
    return tuple(_hashable(item) for item in value)
  if isinstance(value, collections.abc.Mapping):
    return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
  return value
