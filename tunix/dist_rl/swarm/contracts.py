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
"""Option 5 unified Swarm RL contracts: ownership, budgets, and serving fidelity.

Separates three orthogonal concerns (L1 user contracts) plus explicit numerical
serving fidelity checks (L3):
  1. `LearnerSpec` owns disjoint mutable parameters (full or shared-base LoRA).
  2. `PolicySpec` binds logical roles to immutable (`fixed`) or evolving
     (`follow`) snapshots with explicit `ServingFidelity` requirements.
  3. `ObjectiveBatch` chooses which roles/actions receive gradient updates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import enum
import math
from typing import Any

import numpy as np


class ServingFidelityMode(str, enum.Enum):
  """How learner weights are materialized on the rollout serving pool."""

  FULL_BF16 = "full_bf16"
  UNMERGED_DYNAMIC_LORA = "unmerged_dynamic_lora"
  FP32_NATIVE = "fp32_native"
  MERGED_BF16 = "merged_bf16"


@dataclasses.dataclass(frozen=True, kw_only=True)
class ServingFidelity:
  """L3 numerical fidelity contract between a learner and its rollout sampler."""

  mode: ServingFidelityMode = ServingFidelityMode.FULL_BF16
  max_top1_mismatch_rate: float = 0.01
  max_logprob_abs_diff: float = 1e-3
  require_attestation: bool = True
  allow_lossy_lora_merge: bool = False

  def __post_init__(self) -> None:
    if isinstance(self.mode, str):
      object.__setattr__(self, "mode", ServingFidelityMode(self.mode))
    if not 0.0 <= self.max_top1_mismatch_rate <= 1.0:
      raise ValueError("max_top1_mismatch_rate must be in [0, 1].")
    if self.max_logprob_abs_diff < 0.0:
      raise ValueError("max_logprob_abs_diff must be nonnegative.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class FidelityReport:
  """Startup numerical parity report comparing learner vs. serving sampler."""

  passed: bool
  mode: ServingFidelityMode
  top1_agreement: float
  max_logprob_abs_diff: float
  mean_logprob_abs_diff: float
  checked_tokens: int
  reason: str = "ok"


def verify_serving_fidelity(
    *,
    learner_top1_ids: Sequence[int],
    sampler_top1_ids: Sequence[int],
    learner_logprobs: Sequence[float],
    sampler_logprobs: Sequence[float],
    fidelity: ServingFidelity,
    raise_on_fail: bool = True,
) -> FidelityReport:
  """Verifies Learner <-> Sampler greedy Top-1 and logprob numerical parity."""
  lt = np.asarray(learner_top1_ids, dtype=np.int64)
  st = np.asarray(sampler_top1_ids, dtype=np.int64)
  ll = np.asarray(learner_logprobs, dtype=np.float64)
  sl = np.asarray(sampler_logprobs, dtype=np.float64)
  if not (lt.size > 0 and lt.shape == st.shape and ll.shape == sl.shape):
    raise ValueError("Fidelity audit requires nonempty aligned token/logprob arrays.")
  if not (np.all(np.isfinite(ll)) and np.all(np.isfinite(sl))):
    raise ValueError("Non-finite logprobs encountered during fidelity audit.")
  agreement = float(np.mean(lt == st))
  abs_diffs = np.abs(ll - sl)
  max_diff = float(np.max(abs_diffs))
  mean_diff = float(np.mean(abs_diffs))
  reasons: list[str] = []
  if (1.0 - agreement) > fidelity.max_top1_mismatch_rate:
    reasons.append(
        f"top1_mismatch={1.0 - agreement:.4f} > {fidelity.max_top1_mismatch_rate:.4f}"
    )
  if max_diff > fidelity.max_logprob_abs_diff:
    reasons.append(
        f"max_logprob_abs_diff={max_diff:.6g} > {fidelity.max_logprob_abs_diff:.6g}"
    )
  passed = not reasons
  report = FidelityReport(
      passed=passed,
      mode=fidelity.mode,
      top1_agreement=agreement,
      max_logprob_abs_diff=max_diff,
      mean_logprob_abs_diff=mean_diff,
      checked_tokens=int(ll.size),
      reason="ok" if passed else "; ".join(reasons),
  )
  if raise_on_fail and not passed:
    raise RuntimeError(
        f"ServingFidelity gate failed ({fidelity.mode.value}): {report.reason}"
    )
  return report


@dataclasses.dataclass(frozen=True, kw_only=True)
class SnapshotRef:
  """Immutable identity of a served model weight snapshot."""

  model_id: str
  revision: int
  digest: str
  base_digest: str | None = None
  adapter_id: str | None = None

  def __post_init__(self) -> None:
    if not self.model_id or not self.digest or self.revision < 0:
      raise ValueError("SnapshotRef needs model_id, digest, and revision >= 0.")
    if self.base_digest is None:
      object.__setattr__(self, "base_digest", self.digest)


@dataclasses.dataclass(frozen=True, kw_only=True)
class LearnerSpec:
  """Declares mutable parameters owned by one independent optimizer."""

  name: str
  initial: SnapshotRef
  is_lora: bool = False
  adapter_id: str | None = None
  lora_rank: int = 0
  lora_alpha: float = 0.0
  fidelity: ServingFidelity = dataclasses.field(default_factory=ServingFidelity)

  def __post_init__(self) -> None:
    if not self.name:
      raise ValueError("LearnerSpec requires a nonempty name.")
    if self.is_lora:
      if not self.adapter_id or self.lora_rank <= 0:
        raise ValueError("LoRA learner requires adapter_id and lora_rank > 0.")
      if (
          self.fidelity.mode == ServingFidelityMode.MERGED_BF16
          and not self.fidelity.allow_lossy_lora_merge
      ):
        raise ValueError(
            "MERGED_BF16 silently zeroes small LoRA updates on TPU; use "
            "FP32_NATIVE or UNMERGED_DYNAMIC_LORA (or opt in explicitly with "
            "allow_lossy_lora_merge=True)."
        )

  @classmethod
  def full(
      cls,
      name: str,
      initial: SnapshotRef,
      *,
      fidelity: ServingFidelity | None = None,
  ) -> "LearnerSpec":
    return cls(
        name=name,
        initial=initial,
        is_lora=False,
        fidelity=fidelity or ServingFidelity(mode=ServingFidelityMode.FULL_BF16),
    )

  @classmethod
  def lora(
      cls,
      name: str,
      *,
      base: SnapshotRef,
      adapter: str,
      rank: int = 64,
      alpha: float = 128.0,
      fidelity: ServingFidelity | None = None,
  ) -> "LearnerSpec":
    return cls(
        name=name,
        initial=base,
        is_lora=True,
        adapter_id=adapter,
        lora_rank=rank,
        lora_alpha=alpha,
        fidelity=fidelity or ServingFidelity(mode=ServingFidelityMode.FP32_NATIVE),
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class PolicySpec:
  """Role-to-policy binding separating serving weights from loss inclusion."""

  kind: str  # "fixed" | "follow"
  snapshot: SnapshotRef | None = None
  learner: str | None = None
  capture_logprobs: bool = False
  fidelity: ServingFidelity | None = None

  def __post_init__(self) -> None:
    if self.kind == "fixed" and self.snapshot is None:
      raise ValueError("PolicySpec.fixed requires an immutable SnapshotRef.")
    if self.kind == "follow" and not self.learner:
      raise ValueError("PolicySpec.follow requires a learner name.")
    if self.kind not in ("fixed", "follow"):
      raise ValueError(f"Unsupported PolicySpec kind: {self.kind!r}.")

  @classmethod
  def fixed(
      cls,
      snapshot: SnapshotRef,
      *,
      capture_logprobs: bool = False,
      fidelity: ServingFidelity | None = None,
  ) -> "PolicySpec":
    return cls(
        kind="fixed",
        snapshot=snapshot,
        capture_logprobs=capture_logprobs,
        fidelity=fidelity,
    )

  @classmethod
  def follow(
      cls,
      learner: str,
      *,
      fidelity: ServingFidelity | None = None,
  ) -> "PolicySpec":
    return cls(
        kind="follow",
        learner=learner,
        capture_logprobs=True,
        fidelity=fidelity,
    )


def _register_learner_ownership(
    name: str,
    spec: LearnerSpec,
    full_bases: dict[str, str],
    lora_keys: dict[tuple[str, str], str],
) -> None:
  if spec.name != name:
    raise ValueError(f"Learner key {name!r} mismatches spec.name {spec.name!r}.")
  base_id = spec.initial.base_digest or spec.initial.digest
  if spec.is_lora:
    key = (base_id, str(spec.adapter_id))
    if key in lora_keys:
      raise ValueError(f"Duplicate LoRA adapter ownership for {key}.")
    if base_id in full_bases:
      raise ValueError(f"LoRA learner {name!r} collides with full base {base_id!r}.")
    lora_keys[key] = name
    return
  if base_id in full_bases or any(b == base_id for b, _ in lora_keys):
    raise ValueError(f"Full learner {name!r} collides on base {base_id!r}.")
  full_bases[base_id] = name


def validate_ownership(
    learners: Mapping[str, LearnerSpec],
    bindings: Mapping[str, PolicySpec],
) -> None:
  """Validates disjoint parameter ownership and complete role bindings."""
  full_bases: dict[str, str] = {}
  lora_keys: dict[tuple[str, str], str] = {}
  for name, spec in learners.items():
    _register_learner_ownership(name, spec, full_bases, lora_keys)
  for role, binding in bindings.items():
    if not role:
      raise ValueError("Policy bindings require nonempty role names.")
    if binding.kind == "follow" and binding.learner not in learners:
      raise ValueError(f"Role {role!r} follows undeclared learner {binding.learner!r}.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class SnapshotBundle:
  """Pinned policy snapshots resolved for one step or async window."""

  step_id: int
  role_snapshots: Mapping[str, SnapshotRef]
  learner_snapshots: Mapping[str, SnapshotRef] = dataclasses.field(
      default_factory=dict
  )
  references: Mapping[str, SnapshotRef] = dataclasses.field(default_factory=dict)

  def snapshot_for_role(self, role: str) -> SnapshotRef:
    if role not in self.role_snapshots:
      raise KeyError(f"Role {role!r} missing from SnapshotBundle.")
    return self.role_snapshots[role]


@dataclasses.dataclass(frozen=True, kw_only=True)
class CriticalStepsBudget:
  """PARL latency-oriented budget charging main_steps + max(child_steps)."""

  limit: int

  def __post_init__(self) -> None:
    if self.limit <= 0:
      raise ValueError("CriticalStepsBudget.limit must be positive.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class WallClockBudget:
  """Physical time and optional call cap budget for dynamic peer swarms."""

  deadline_s: float
  max_total_calls: int | None = None

  def __post_init__(self) -> None:
    if not math.isfinite(self.deadline_s) or self.deadline_s <= 0.0:
      raise ValueError("WallClockBudget.deadline_s must be finite and positive.")
    if self.max_total_calls is not None and self.max_total_calls <= 0:
      raise ValueError("WallClockBudget.max_total_calls must be positive.")


EpisodeBudget = CriticalStepsBudget | WallClockBudget


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelAction:
  """Immutable record of one participant model invocation."""

  action_id: str
  participant_id: str
  role: str
  policy_id: str
  served_snapshot: SnapshotRef
  prompt_ids: tuple[int, ...]
  completion_ids: tuple[int, ...]
  rollout_logprobs: tuple[float, ...]
  text: str
  parent_action_id: str | None = None
  cohort_id: str | None = None
  finish_reason: str = "stop"
  loss_eligible: bool = True
  annotations: dict[str, Any] = dataclasses.field(default_factory=dict)

  def __post_init__(self) -> None:
    for field_name in ("prompt_ids", "completion_ids", "rollout_logprobs"):
      object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
    if not self.action_id or not self.participant_id or not self.role:
      raise ValueError("ModelAction needs action_id, participant_id, and role.")
    if not self.prompt_ids or not self.completion_ids:
      raise ValueError("ModelAction needs nonempty prompt and completion IDs.")
    if self.rollout_logprobs and len(self.rollout_logprobs) != len(
        self.completion_ids
    ):
      raise ValueError("rollout_logprobs must align with completion_ids.")
    if any(not math.isfinite(x) or x > 1e-5 for x in self.rollout_logprobs):
      raise ValueError("rollout_logprobs must be finite log probabilities.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompactedSequence:
  """Automatically compacted TITO sequence for one participant & snapshot."""

  participant_id: str
  role: str
  policy_id: str
  served_snapshot: SnapshotRef
  action_ids: tuple[str, ...]
  prompt_ids: tuple[int, ...]
  completion_ids: tuple[int, ...]
  rollout_logprobs: tuple[float, ...]
  loss_mask: tuple[int, ...]
  attention_mask: tuple[int, ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ChildRunRef:
  """Deterministic lineage and completion summary for one spawned child."""

  child_id: str
  participant_id: str
  parent_action_id: str | None
  cohort_id: str | None
  status: str  # "completed" | "budget_exhausted" | "cancelled" | "failed"
  critical_steps: int
  total_steps: int
  cost: float
  output: Any = None
  error: str | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class EpisodeRecord:
  """One terminal attempt record returned by EpisodeRuntime."""

  episode_id: str
  task_id: str
  group_id: str
  slot_index: int
  condition: str
  snapshots: SnapshotBundle
  status: str  # "completed" | "budget_exhausted" | "cancelled"
  public_output: Any
  actions: tuple[ModelAction, ...]
  children: tuple[ChildRunRef, ...]
  critical_steps: int
  total_steps: int
  wall_time_s: float
  worker_cost: float = 0.0
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ScoreRecord:
  """Post-collection private grading outcome for one EpisodeRecord."""

  episode_id: str
  task_id: str
  reward: float
  components: Mapping[str, float] = dataclasses.field(default_factory=dict)
  action_annotations: Mapping[str, Mapping[str, Any]] = dataclasses.field(
      default_factory=dict
  )
  grader_revision: str = "v1"

  def __post_init__(self) -> None:
    if not math.isfinite(self.reward):
      raise ValueError("ScoreRecord.reward must be finite.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class ObjectiveBatch:
  """Typed batch staged into one learner via `StepContext.update`."""

  target: str
  objective_kind: str  # "token_grpo" | "candidate_pair_pg" | "opd"
  examples: tuple[dict[str, Any], ...]
  reduction: str = "token_mean"
  roles: tuple[str, ...] = ()
  max_policy_lag: int = 0
  provenance: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, kw_only=True)
class StepResult:
  """Return value of `train_step(step)` carrying next committed recipe state."""

  state: Any = None
  metrics: Mapping[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Recipe:
  """Top-level Option 5 specification combining workflow, policies, and step."""

  workflow: Callable[..., Any]
  learners: Mapping[str, LearnerSpec]
  bindings: Mapping[str, PolicySpec]
  grade: Callable[[EpisodeRecord, Any], ScoreRecord]
  train_step: Callable[..., Any]
  budget: EpisodeBudget | None = None
  references: Mapping[str, SnapshotRef] = dataclasses.field(default_factory=dict)
  initial_state: Any = None

  def __post_init__(self) -> None:
    validate_ownership(self.learners, self.bindings)
