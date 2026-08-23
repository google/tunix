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

"""Fail-closed checkpoint contract for the P45 FrozenLake carrier."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import re
from typing import Any


GCS_ROOT = (
    "gs://yuxzhang-tunix-models/canon-zero-tim/checkpoints/frozenlake"
)
SCHEMA = "p45-frozenlake-checkpoint-v1"
_TAG_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_ENV_KEYS = (
    "CANON_FROZENLAKE_CKPT_ROOT",
    "CANON_FROZENLAKE_CKPT_TAG",
    "CANON_FROZENLAKE_CKPT_INTERVAL",
    "CANON_FROZENLAKE_CKPT_MAX_TO_KEEP",
    "CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL",
)
P57_STOCK_SYNC_RECEIPT = {
    "completed": True,
    "transport": "update_params",
    "exact_weight_attestation": "unavailable-by-design",
}
P45_CHECKPOINT_INTERVAL = 10
P57_PRIMARY_CHECKPOINT_INTERVAL = 300


def registered_checkpoint_interval(env: Mapping[str, str]) -> int:
  """Returns the exact checkpoint interval registered for this workload.

  The active P57 concept-study arms run uninterrupted for 300 updates and
  produce their learning curve in-process.  They need only the final durable
  recovery point.  Historical P45 and P57 discovery carriers retain their
  ten-update recovery cadence.
  """
  primary_workload = (
      (
          env.get("CANON_P57_WORKLOAD_CANDIDATE", "") == ""
          and env.get("CANON_P57_DATA_SPLIT", "") == ""
      )
      or (
          env.get("CANON_P57_WORKLOAD_CANDIDATE", "") == "m15"
          and env.get("CANON_P57_DATA_SPLIT", "") == "main"
      )
  )
  if (
      env.get("CANON_P57_RUN_KIND", "") in ("train", "eval")
      and env.get("CANON_P57_TIM_ARM", "") in ("zero", "mismatch", "is")
      and env.get("CANON_P57_EXPECTED_UPDATES", "") == "300"
      and primary_workload
  ):
    return P57_PRIMARY_CHECKPOINT_INTERVAL
  return P45_CHECKPOINT_INTERVAL


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
  """Resolved P45 checkpoint configuration."""

  mode: str
  root: str = ""
  tag: str = ""
  interval: int = 0
  max_to_keep: int = 0
  milestone_interval: int = 0

  @property
  def enabled(self) -> bool:
    return self.mode != "disabled"

  @property
  def directory(self) -> str | None:
    if not self.enabled:
      return None
    return f"{self.root}/{self.tag}"


def sync_rollout_for_no_update(
    learner: Any, *, stock_fast: bool
) -> dict[str, Any]:
  """Synchronizes rollout weights and applies the available proof contract.

  The untreated stock engine deliberately has no canonical engine adapter, so
  it cannot expose the live-leaf comparison used by canonical resume/eval.
  Successful ``update_params`` completion is the strongest honest stock
  receipt.  Canonical callers retain the exact live-weight gate.
  """
  # Admission belongs to RLLearner/GRPOLearner, while the transport and exact
  # comparison belong to RLCluster.  Accept the learner so this ownership
  # topology cannot be flattened into a fake cluster-only interface.
  if not learner.should_sync_weights:
    raise ValueError(
        "P45 resume/P57 no-update run requires an explicit rollout weight sync"
    )
  rl_cluster = learner.rl_cluster
  rl_cluster.sync_weights_for_resume()
  if stock_fast:
    return dict(P57_STOCK_SYNC_RECEIPT)

  exact = dict(rl_cluster.attest_actor_anchor_matches_engine())
  if exact.get("equal") is not True:
    raise ValueError(
        "P45 restored actor did not match vLLM after resume sync: "
        f"{exact}"
    )
  return {
      "completed": True,
      "transport": "update_params",
      "exact_weight_attestation": "pass",
      "attestation": exact,
  }


def from_env(env: Mapping[str, str]) -> Config:
  """Resolves the explicit checkpoint mode and rejects partial contracts."""
  mode = env.get("CANON_FROZENLAKE_CKPT_MODE", "").strip() or "disabled"
  if mode == "disabled":
    residual = {key: env.get(key, "") for key in _ENV_KEYS if env.get(key, "")}
    if residual:
      raise ValueError(
          "FrozenLake checkpoint fields require an explicit new/resume mode: "
          f"{sorted(residual)}"
      )
    return Config(mode="disabled")
  if mode not in ("new", "resume"):
    raise ValueError(
        "CANON_FROZENLAKE_CKPT_MODE must be disabled, new, or resume"
    )

  root = env.get("CANON_FROZENLAKE_CKPT_ROOT", "").rstrip("/")
  tag = env.get("CANON_FROZENLAKE_CKPT_TAG", "")
  if root != GCS_ROOT:
    raise ValueError(
        "FrozenLake checkpoint root drifted: "
        f"expected {GCS_ROOT!r}, got {root!r}"
    )
  if not _TAG_RE.fullmatch(tag):
    raise ValueError(
        "FrozenLake checkpoint tag must be lowercase, Kubernetes-safe, and "
        "at most 63 characters"
    )
  try:
    interval = int(env.get("CANON_FROZENLAKE_CKPT_INTERVAL", ""))
    max_to_keep = int(env.get("CANON_FROZENLAKE_CKPT_MAX_TO_KEEP", ""))
    milestone_interval = int(
        env.get("CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL", "0") or "0"
    )
  except ValueError as exc:
    raise ValueError("FrozenLake checkpoint bounds must be integers") from exc
  expected_interval = registered_checkpoint_interval(env)
  if interval != expected_interval:
    raise ValueError(
        "FrozenLake checkpoint interval drifted: "
        f"expected {expected_interval}, got {interval}"
    )
  if max_to_keep != 1:
    raise ValueError("P45 checkpoint retention must be exactly one")
  if milestone_interval not in (0, 50):
    raise ValueError(
        "FrozenLake checkpoint milestone interval must be disabled or 50"
    )
  if milestone_interval:
    try:
      expected_updates = int(env.get("CANON_P57_EXPECTED_UPDATES", ""))
    except ValueError as exc:
      raise ValueError(
          "P57 checkpoint milestones require an integer signed horizon"
      ) from exc
    if (
        env.get("CANON_P57_RUN_KIND", "") not in ("train", "eval")
        or env.get("CANON_P57_TIM_ARM", "")
        not in ("zero", "mismatch", "is")
        or expected_updates != 450
    ):
      raise ValueError(
          "50-step checkpoint milestones are isolated to the registered "
          "P57 450-update train/eval arms"
      )
  if env.get("ENABLE_PATHWAYS_PERSISTENCE", "") != "1":
    raise ValueError("P45 checkpointing requires Pathways persistence")
  return Config(
      mode=mode,
      root=root,
      tag=tag,
      interval=interval,
      max_to_keep=max_to_keep,
      milestone_interval=milestone_interval,
  )


def require_p45(config: Config, env: Mapping[str, str]) -> None:
  """Restricts the GCS contract to committed P45 DP8xTP8 full training."""
  if not config.enabled:
    raise ValueError("P45 full training requires checkpointing enabled")
  required = {
      "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
      "CANON_P33_RUN_STAGE": "full",
      "CANON_P33_NO_COMMIT": "0",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
  }
  wrong = {
      key: env.get(key, "")
      for key, expected in required.items()
      if env.get(key, "") != expected
  }
  if wrong:
    raise ValueError(f"P45 checkpoint workload contract drifted: {wrong}")


def build_contract(config: Config, values: Mapping[str, Any]) -> dict[str, Any]:
  """Builds the exact metadata stored with every P45 actor checkpoint."""
  if not config.enabled:
    raise ValueError("cannot build a disabled checkpoint contract")
  contract = {
      "schema": SCHEMA,
      "checkpoint_root": config.root,
      "checkpoint_tag": config.tag,
      "checkpoint_interval": config.interval,
      "checkpoint_max_to_keep": config.max_to_keep,
      **dict(values),
  }
  # Keep the historical P45 contract byte-for-byte when milestone retention
  # is disabled, so pre-P57 checkpoints remain resumable.
  if config.milestone_interval:
    contract["checkpoint_milestone_interval"] = config.milestone_interval
  # Checkpoint custom metadata must remain portable JSON. Round-tripping also
  # rejects arrays and other objects whose equality would be ambiguous.
  return json.loads(json.dumps(contract, sort_keys=True))


def build_preservation_policy(config: Config) -> Any:
  """Keeps one rolling recovery point plus registered P57 milestones."""
  if not config.enabled:
    raise ValueError("cannot build preservation policy for disabled checkpoints")
  # Import lazily so the pure checkpoint-contract tests remain lightweight.
  from orbax.checkpoint import v1 as ocp  # pylint: disable=g-import-not-at-top

  latest = ocp.training.preservation_policies.LatestN(config.max_to_keep)
  if not config.milestone_interval:
    return latest
  milestones = ocp.training.preservation_policies.EveryNSteps(
      config.milestone_interval,
      exact_interval=True,
  )
  return ocp.training.preservation_policies.AnyPreservationPolicy(
      (latest, milestones)
  )


def contract_json(contract: Mapping[str, Any]) -> str:
  return json.dumps(dict(contract), sort_keys=True, separators=(",", ":"))


def validate_latest(config: Config, latest_step: int | None) -> None:
  """Validates the durable prefix before allocating the model."""
  if not config.enabled:
    return
  if config.mode == "new":
    if latest_step is not None:
      raise ValueError(
          "new P45 campaign refuses an existing complete checkpoint: "
          f"step={latest_step}"
      )
    return
  if latest_step is None:
    raise ValueError("P45 resume requires an existing complete checkpoint")
  if latest_step <= 0 or latest_step % config.interval:
    raise ValueError(
        "P45 latest checkpoint is not a registered committed boundary: "
        f"step={latest_step} interval={config.interval}"
    )


def validate_restored(
    config: Config,
    *,
    restored_step: int,
    optimizer_restored: bool,
    metadata: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
) -> None:
  """Checks the post-restore actor, optimizer, step, and provenance contract."""
  if not config.enabled:
    return
  if config.mode == "new":
    if restored_step != 0 or metadata:
      raise ValueError(
          "new P45 campaign unexpectedly restored checkpoint state: "
          f"step={restored_step} metadata={dict(metadata)}"
      )
    if optimizer_restored:
      raise ValueError("new P45 campaign unexpectedly restored optimizer state")
    return

  if restored_step <= 0 or restored_step % config.interval:
    raise ValueError(
        "P45 restored step is not a registered checkpoint boundary: "
        f"step={restored_step}"
    )
  if not optimizer_restored:
    raise ValueError("P45 resume did not restore optimizer state")
  if metadata.get("global_step") != restored_step:
    raise ValueError(
        "P45 restored global step mismatch: "
        f"metadata={metadata.get('global_step')!r} restored={restored_step}"
    )
  if metadata.get("role") != "actor":
    raise ValueError(
        f"P45 checkpoint role mismatch: {metadata.get('role')!r}"
    )
  actual_contract = metadata.get("canon_resume_contract")
  if actual_contract != dict(expected_contract):
    raise ValueError("P45 checkpoint resume contract mismatch")


def validate_p57_evaluation_restored(
    config: Config,
    *,
    restored_step: int,
    metadata: Mapping[str, Any],
    env: Mapping[str, str],
) -> None:
  """Validates an actor checkpoint consumed by an isolated P57 evaluator."""
  try:
    expected_step = int(env.get("CANON_P57_EVAL_CHECKPOINT_STEP", ""))
  except ValueError as exc:
    raise ValueError("P57 evaluation checkpoint step must be an integer") from exc
  if expected_step < 0 or expected_step % config.interval:
    raise ValueError(
        "P57 evaluation checkpoint step must be zero or a checkpoint boundary"
    )
  if expected_step == 0:
    if config.mode != "new":
      raise ValueError("P57 step-0 evaluation requires checkpoint mode=new")
    if restored_step != 0 or metadata:
      raise ValueError(
          "P57 step-0 evaluation unexpectedly restored checkpoint state: "
          f"restored={restored_step} metadata={dict(metadata)}"
      )
    return
  if config.mode != "resume":
    raise ValueError("P57 checkpoint evaluation requires checkpoint mode=resume")
  if restored_step != expected_step or metadata.get("global_step") != expected_step:
    raise ValueError(
        "P57 evaluation restored the wrong checkpoint: "
        f"restored={restored_step} metadata={metadata.get('global_step')!r} "
        f"expected={expected_step}"
    )
  if metadata.get("role") != "actor":
    raise ValueError("P57 evaluation requires an actor checkpoint")
  contract = metadata.get("canon_resume_contract")
  if not isinstance(contract, Mapping):
    raise ValueError("P57 evaluation checkpoint lacks its training contract")
  active_inprocess_eval = (
      env.get("CANON_P57_EXPECTED_UPDATES", "") == "300"
      and (
          (
              env.get("CANON_P57_WORKLOAD_CANDIDATE", "") == ""
              and env.get("CANON_P57_DATA_SPLIT", "") == ""
          )
          or (
              env.get("CANON_P57_WORKLOAD_CANDIDATE", "") == "m15"
              and env.get("CANON_P57_DATA_SPLIT", "") == "main"
          )
      )
  )
  required = {
      "schema": SCHEMA,
      "checkpoint_root": config.root,
      "checkpoint_tag": config.tag,
      "checkpoint_interval": config.interval,
      "checkpoint_max_to_keep": config.max_to_keep,
      "source_commit": env.get("CANON_EXPECT_COMMIT", ""),
      "profile": "qwen3-8b-dp8-tp8-frozenlake-tim",
      "workload": "frozenlake-dp8-tp8",
      "model_version": "Qwen/Qwen3-8B",
      "model_dir_name": "qwen8b_tp8",
      "mesh_dp": 8,
      "mesh_tp": 8,
      "seed": 42,
      "p57_tim_arm": env.get("CANON_P57_TIM_ARM", ""),
      "p57_fixed_lm_head": env.get("CANON_P38_FIXED_LM_HEAD", "0"),
      "p57_workload_candidate": env.get(
          "CANON_P57_WORKLOAD_CANDIDATE", ""
      ),
      "p57_data_split": env.get("CANON_P57_DATA_SPLIT", ""),
      "eval_enabled": active_inprocess_eval,
  }
  if config.milestone_interval:
    required["checkpoint_milestone_interval"] = config.milestone_interval
  wrong = {
      key: contract.get(key)
      for key, expected in required.items()
      if contract.get(key) != expected
  }
  if wrong:
    raise ValueError(f"P57 evaluation checkpoint provenance drifted: {wrong}")
