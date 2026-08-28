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

"""Base class for Agentic RL Learners."""

from __future__ import annotations
import abc
import time
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import contextlib
import copy
import dataclasses
import hashlib
import itertools
import json
import os
import queue
import threading
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Type,
    TypeVar,
)

from absl import logging
import flax
import jax
from tunix.sft import profiler as sft_profiler
from jax import typing
import jax.numpy as jnp
import numpy as np
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import alignment
from tunix.rl import common
from tunix.rl import deepswe_contract
from tunix.rl import deepswe_debug
from tunix.rl import dp_workloads
from tunix.perf.experimental import constants as perf_constants
from tunix.rl import function_registry
from tunix.rl import gsm8k_xprof
from tunix.rl import host_memory as host_memory_lib
from tunix.rl import p64_training_capsule
from tunix.rl import v1_first_update_gate
from tunix.rl import perf_log
from tunix.rl import reward_manager  # pylint: disable=unused-import
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.agents import model_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.environments import task_environment
from tunix.rl.agentic.pipeline import rollout_orchestrator
from tunix.rl.agentic.rewards import reward  # pylint: disable=unused-import
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import utils as sft_utils

ArrayLike = typing.ArrayLike


def _p61_capture_tree(
    directory: str, capture_name: str, tree: Any
) -> dict[str, Any]:
  """Writes one complete P61 tree without replacing prior evidence."""
  from flax import nnx  # pylint: disable=g-import-not-at-top

  if not directory or not os.path.isabs(directory):
    raise alignment.AlignmentGateError(
        "P61 numerical capture directory must be an absolute path"
    )
  if capture_name not in ("gradient", "model_before", "model_after"):
    raise alignment.AlignmentGateError(
        f"unsupported P61 numerical capture name: {capture_name!r}"
    )
  if os.path.lexists(directory):
    if os.path.islink(directory) or not os.path.isdir(directory):
      raise alignment.AlignmentGateError(
          f"P61 numerical capture root is not a real directory: {directory}"
      )
  else:
    os.makedirs(directory, mode=0o755)
  capture_dir = os.path.join(directory, capture_name)
  try:
    os.mkdir(capture_dir, mode=0o755)
  except FileExistsError as exc:
    raise alignment.AlignmentGateError(
        f"refusing to overwrite P61 numerical capture: {capture_dir}"
    ) from exc

  flattened = jax.tree_util.tree_flatten_with_path(
      tree, is_leaf=lambda value: isinstance(value, nnx.Variable)
  )[0]
  if not flattened:
    raise alignment.AlignmentGateError(
        f"P61 numerical capture {capture_name!r} has no leaves"
    )
  leaves = []
  total_bytes = 0
  for index, (path, value) in enumerate(flattened):
    array = value[...] if isinstance(value, nnx.Variable) else value
    host = np.ascontiguousarray(np.asarray(jax.device_get(array)))
    if host.dtype.hasobject:
      raise alignment.AlignmentGateError(
          "P61 numerical capture refuses object-valued leaves"
      )
    filename = f"leaf_{index:05d}.npy"
    output_path = os.path.join(capture_dir, filename)
    with open(output_path, "xb") as output_file:
      np.save(output_file, host, allow_pickle=False)
    file_digest = hashlib.sha256()
    with open(output_path, "rb") as input_file:
      while chunk := input_file.read(8 * 1024 * 1024):
        file_digest.update(chunk)
    byte_count = int(host.nbytes)
    total_bytes += byte_count
    leaves.append({
        "index": index,
        "path": jax.tree_util.keystr(path),
        "file": filename,
        "shape": list(host.shape),
        "dtype": str(host.dtype),
        "elements": int(host.size),
        "data_bytes": byte_count,
        "data_sha256": hashlib.sha256(host.tobytes(order="C")).hexdigest(),
        "file_sha256": file_digest.hexdigest(),
    })
  manifest = {
      "schema": "canon-p61-full-tree-capture-v1",
      "capture": capture_name,
      "leaves": leaves,
      "leaf_count": len(leaves),
      "total_data_bytes": total_bytes,
  }
  manifest_path = os.path.join(capture_dir, "manifest.json")
  with open(manifest_path, "x", encoding="utf-8") as manifest_file:
    json.dump(manifest, manifest_file, indent=2, sort_keys=True)
    manifest_file.write("\n")
  print(
      "[P61.NUMERICAL] capture_complete "
      f"name={capture_name} leaves={len(leaves)} bytes={total_bytes} "
      f"manifest={manifest_path}",
      flush=True,
  )
  return manifest


def _emit_p45_host_memory(
    *, phase: str, step: int, gc_collected: int | None = None
) -> None:
  print(
      "[P45.HOST_MEMORY] "
      + json.dumps(
          host_memory_lib.record(
              phase=phase, step=step, gc_collected=gc_collected
          ),
          sort_keys=True,
          separators=(",", ":"),
      ),
      flush=True,
  )


def _p38_diagnostic_consumer_contract(
    *,
    enabled: bool,
    full_batch_size: int,
    mini_batch_size: int,
    train_micro_batch_size: int,
    num_generations: int,
    process_in_consumer: bool,
    onehost_rehearsal: bool = False,
    m15_target_debug: bool = False,
    v1_fl_tp8_ab: bool = False,
    p58_vma_diagnostic: bool = False,
    p58_seam_localization: bool = False,
) -> tuple[int, bool, int]:
  """Return the P38 full-coverage consumer geometry.

  Historical P38 keeps four prompts per producer unit and consumes all eight
  units.  The M15 target carrier and V1 FrozenLake TP8 A/B diagnostic instead
  preserve the production 32-prompt unit and consume that one complete unit.
  Both paths cover all 256 trajectories before calling ``_process_results``;
  neither admits a partial tail or the old P38s10 first-four-prompt subset.
  """
  if not enabled:
    return train_micro_batch_size, False, 0
  if onehost_rehearsal and m15_target_debug:
    raise ValueError("M15 target debug is not a one-host rehearsal")
  if onehost_rehearsal and v1_fl_tp8_ab:
    raise ValueError("V1 FrozenLake TP8 A/B is not a one-host rehearsal")
  if onehost_rehearsal and p58_vma_diagnostic:
    raise ValueError("P58 VMA diagnostic is not a one-host rehearsal")
  if onehost_rehearsal and p58_seam_localization:
    raise ValueError("P58 seam localization is not a one-host rehearsal")
  expected = (
      (2, 2, 2)
      if onehost_rehearsal
      else (8, 8, 16)
      if (p58_vma_diagnostic or p58_seam_localization)
      else (32, 32, 8)
      if (m15_target_debug or v1_fl_tp8_ab)
      else (32, 4, 8)
  )
  observed = (full_batch_size, mini_batch_size, num_generations)
  if observed != expected:
    raise ValueError(
        "P38 diagnostic coverage geometry changed: "
        f"observed={observed} expected={expected}"
    )
  if not process_in_consumer:
    raise ValueError(
        "P38 diagnostic coverage requires raw trajectories to be processed "
        "in the consumer"
    )
  if full_batch_size % mini_batch_size:
    raise ValueError(
        "P38 diagnostic prompt coverage is not an integer number of units: "
        f"{full_batch_size} vs {mini_batch_size}"
    )
  return full_batch_size, True, full_batch_size // mini_batch_size


def _frozenlake_evaluation_metrics(
    rewards: Any, *, wall_seconds: float, policy_step: int
) -> dict[str, float | int]:
  """Returns finite, complete held-out evaluation summary metrics."""
  values = np.asarray(rewards, dtype=np.float32).reshape(-1)
  if values.size == 0 or not np.all(np.isfinite(values)):
    raise ValueError("FrozenLake evaluation rewards must be nonempty and finite")
  if not np.isfinite(wall_seconds) or wall_seconds < 0.0:
    raise ValueError(
        "FrozenLake evaluation wall time must be finite and nonnegative"
    )
  if policy_step < 0:
    raise ValueError("FrozenLake evaluation policy step must be nonnegative")
  return {
      "reward": float(values.mean()),
      "solve": float((values > 0.1).mean()),
      "n": int(values.size),
      "wall_seconds": float(wall_seconds),
      "policy_step": int(policy_step),
  }


def _segmented_update_geometry(environ) -> tuple[int, int, str, bool]:
  """Returns the fail-closed trajectory contract for one segmented update."""
  p33_workload = environ.get("CANON_P33_WORKLOAD_LAUNCH_ADMITTED", "") == "1"
  p34_workload = environ.get("CANON_P34_DEEPSWE", "") == "1"
  p41_optimizer_bench = environ.get("CANON_P41_OPTIMIZER_BENCH", "") == "1"
  if p41_optimizer_bench and (
      environ.get("CANON_GSM8K_L3", "") != "1"
      or environ.get("CANON_GSM8K_UPDATE_CANARY", "") != "1"
      or p33_workload
      or p34_workload
  ):
    raise ValueError(
        "P41 optimizer benchmark requires the bounded GSM8K L3 update "
        "canary and cannot overlap P33 or P34"
    )
  p31_convergence = environ.get("CANON_P31_CONVERGENCE", "") == "1"
  if p34_workload:
    workload = deepswe_contract.active_workload(environ)
    return (
        workload.global_trajectories,
        workload.local_trajectories,
        f"[CANON_P34_DP{workload.dp_size}]",
        True,
    )
  if p33_workload:
    workload = dp_workloads.active_workload(environ)
    if workload is None:
      raise ValueError("P33 update requires an active canonical workload")
    return (
        workload.global_trajectories,
        workload.dp_size,
        f"[CANON_P33_DP{workload.dp_size}]",
        True,
    )
  if p41_optimizer_bench:
    return 2, 2, "[P41.OPTIMIZER]", False
  if environ.get("CANON_GSM8K_TRAIN", "") == "1":
    # One-host real-geometry GSM8K training (P51): 32 trajectories per
    # update chunked 16x2. The P33/P34 cluster branches return earlier.
    return 32, 2, "[CANON_GSM8K_TRAIN]", False
  if p31_convergence:
    return 32, 2, "[CANON_FROZENLAKE_P31]", False
  return 8, 2, "[CANON_FROZENLAKE_P27]", False


def _eval_schedule_step(
    *,
    segmented_update: bool,
    pre_update_train_step: int,
    current_train_step: int,
) -> int:
  """Returns the policy step whose rollout weights evaluation will consume.

  The P28/P31 segmented path commits actor gradients before reaching the
  shared evaluation block, but rollout weights are not synchronized until
  after that block.  Evaluation therefore still consumes the pre-update
  rollout policy and must be scheduled/labeled with its pre-update step.
  """
  return pre_update_train_step if segmented_update else current_train_step


def _p57_eval_cycle_enclosing_step(
    *,
    policy_step: int,
    actor_train_steps: int,
    cluster_global_steps: int,
) -> int:
  """Returns the completed wall-row step for a pre-update P57 evaluation.

  This receipt is emitted after the actor update but before ``sync_weights``.
  At that boundary ``actor_trainer.train_steps`` has advanced, while
  ``rl_cluster.global_steps`` deliberately still identifies the rollout
  policy that was evaluated.  Both the standard update and P28/G6 update use
  this lifecycle; reading the deferred cluster counter as the completed row
  produces a false drift failure.
  """
  policy_step = int(policy_step)
  committed_train_step = int(actor_train_steps)
  deferred_global_step = int(cluster_global_steps)
  expected_completed_step = policy_step + 1
  if (
      committed_train_step != expected_completed_step
      or deferred_global_step != policy_step
  ):
    raise RuntimeError(
        "P57 evaluation cycle mapping drifted: "
        f"policy_step={policy_step} "
        f"committed_train_step={committed_train_step} "
        f"deferred_global_step={deferred_global_step}"
    )
  return committed_train_step


def _should_run_eval(
    *,
    prompt_count: int,
    schedule_step: int,
    eval_every_n_steps: int,
    last_eval_train_step: int,
) -> bool:
  """Pure exactly-once predicate for the held-out rollout schedule."""
  if eval_every_n_steps <= 0:
    return False
  return bool(
      prompt_count
      and schedule_step % eval_every_n_steps == 0
      and schedule_step != last_eval_train_step
  )


def _p28_reference_state(rl_cluster):
  """Returns the authoritative state consumed by reference inference.

  RLCluster splits and transfers the frozen reference into InferenceWorker;
  reference scoring thereafter consumes the worker's saved graph/state pair.
  Fingerprinting the module shell or applying the actor-only ``nnx.Param``
  filter can therefore produce an empty tree.  The immutability gate must hash
  the exact saved state used by ``get_ref_per_token_logps``.
  """
  direct_reference = getattr(rl_cluster, "reference", None)
  if direct_reference is not None:
    return flax.nnx.state(direct_reference)
  worker = getattr(rl_cluster, "inference_worker", None)
  if worker is None:
    return None
  return worker.get_model_state("reference")


def _p30_sharding_inventory(tree):
  """Summarizes logical versus local addressable bytes without reading values."""
  arrays = [
      value for value in jax.tree.leaves(tree)
      if isinstance(value, jax.Array)
  ]
  logical_bytes = 0
  addressable_bytes = 0
  by_device = {}
  by_sharding = {}
  for value in arrays:
    itemsize = int(value.dtype.itemsize)
    logical = int(value.size) * itemsize
    logical_bytes += logical
    memory_kind = getattr(value.sharding, "memory_kind", None) or "unspecified"
    spec = getattr(value.sharding, "spec", None)
    spec_text = repr(spec) if spec is not None else type(value.sharding).__name__
    key = f"memory_kind={memory_kind}|spec={spec_text}"
    group = by_sharding.setdefault(
        key,
        {
            "arrays": 0,
            "logical_bytes": 0,
            "addressable_bytes": 0,
            "addressable_shards": 0,
        },
    )
    group["arrays"] += 1
    group["logical_bytes"] += logical
    for shard in value.addressable_shards:
      shard_bytes = int(shard.data.size) * itemsize
      addressable_bytes += shard_bytes
      group["addressable_bytes"] += shard_bytes
      group["addressable_shards"] += 1
      device_id = str(shard.device.id)
      by_device[device_id] = by_device.get(device_id, 0) + shard_bytes
  return {
      "arrays": len(arrays),
      "logical_bytes": logical_bytes,
      "addressable_bytes": addressable_bytes,
      "replication_factor": (
          addressable_bytes / logical_bytes if logical_bytes else 0.0
      ),
      "addressable_bytes_by_device": dict(sorted(by_device.items())),
      "by_sharding": dict(sorted(by_sharding.items())),
  }


def _split_train_example_by_trajectory(
    train_example: Any,
    *,
    total_trajectories: int,
    trajectory_micro_batch_size: int,
) -> list[Any]:
  """Slices every batch-shaped leaf into equal trajectory micro-batches."""
  if total_trajectories <= 0 or trajectory_micro_batch_size <= 0:
    raise ValueError("trajectory batch sizes must be positive")
  if total_trajectories % trajectory_micro_batch_size != 0:
    raise ValueError(
        "total trajectories must be divisible by trajectory micro-batch size:"
        f" {total_trajectories} vs {trajectory_micro_batch_size}"
    )
  return [
      jax.tree_util.tree_map(
          lambda x: (
              x[i : i + trajectory_micro_batch_size]
              if hasattr(x, "shape")
              and x.shape
              and x.shape[0] == total_trajectories
              else x
          ),
          train_example,
      )
      for i in range(0, total_trajectories, trajectory_micro_batch_size)
  ]


def _advance_unpacked_microsteps(
    counter: int, num_microsteps: int, steps_per_update: int
) -> tuple[int, bool]:
  """Advances a host cadence counter and reports one optimizer boundary."""
  if counter < 0 or num_microsteps <= 0 or steps_per_update <= 0:
    raise ValueError("microstep cadence values must be positive")
  next_counter = counter + num_microsteps
  updates = next_counter // steps_per_update - counter // steps_per_update
  if updates > 1:
    raise ValueError(
        "one learner chunk crossed multiple optimizer boundaries:"
        f" counter={counter} microsteps={num_microsteps}"
        f" steps_per_update={steps_per_update}"
    )
  return next_counter, updates == 1


def _p58_all_filtered_no_commit_contract(
    values: Mapping[str, str],
    *,
    all_compact_filtered: bool,
    train_steps_before: int,
    train_steps_after: int,
) -> bool:
  """Validate the P58 all-filtered boundary before skipping weight sync."""
  if not all_compact_filtered:
    return False
  signed = (
      values.get("CANON_P34_DEEPSWE") == "1"
      and values.get("CANON_P58_DEEPSWE_TIM") == "1"
      and values.get("CANON_P58_TIM_ADMITTED") == "1"
      and values.get("CANON_P58_TIM_ARM") in ("native", "zero")
      and values.get("CANON_ALIGNMENT_TRAIN") == "1"
  )
  if not signed:
    raise alignment.AlignmentGateError(
        "all-compact-filtered no-commit sync suppression is restricted to "
        "signed P58 training"
    )
  if train_steps_after != train_steps_before:
    raise alignment.AlignmentGateError(
        "P58 all-compact-filtered batch advanced optimizer train_steps: "
        f"before={train_steps_before} after={train_steps_after}"
    )
  return True


TrainingInputT = Dict[str, List[str] | ArrayLike]
RewardFn = Callable[..., List[float]]
MetricFn = Callable[..., rl_cluster_lib.MetricsT]


@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  policy_version: np.ndarray | None = None
  # ``completion_mask`` is the policy/action mask: environment-injected and
  # parser-appended tokens are excluded from the loss.  They are nevertheless
  # real causal context for later assistant tokens.  Keep that independent
  # sequence-validity contract instead of making model execution compact away
  # every non-action token in a multi-turn trajectory.
  completion_valid_mask: jax.Array | None = None


@dataclasses.dataclass(slots=True, kw_only=True)
class AgenticRLConfig(algo_config_lib.AlgorithmConfig):
  """Base configuration for Agentic RL algorithms.

  Parameters:
    system_prompt: System prompt for the agent.
    max_response_length: Maximum number of tokens for each episode.
    max_concurrency: Maximum number of concurrent requests to the rollout
      engines.
    off_policy_steps: Number of off-policy steps can be accepted before a
      policy update.
    num_generations: Number of samples per prompt.
    num_iterations: Number of iterations per batch.
    episode_timeout: Timeout for each episode in seconds.
    per_turn_timeout: Hard timeout for one model generation in seconds.
    cleanup_timeout: Maximum time allowed to close one environment.
    rollout_batch_timeout: Hard wall-clock timeout for one prompt batch.
  """

  system_prompt: str = ""
  # TODO(tsbao): we need to update the scripts that uses max_tokens_to_generate
  # once this new agentic_rl_learner is used.
  reward_manager: str = "agentic-sequence-level"
  max_response_length: int = 1024
  max_concurrency: int = 32
  off_policy_steps: int = 0
  num_generations: int = 1
  num_iterations: int = 1
  episode_timeout: float = 1800.0
  per_turn_timeout: float | None = None
  cleanup_timeout: float = 150.0
  rollout_batch_timeout: float | None = None
  filter_statuses: Optional[Set] = None
  overlong_filter: bool = False
  use_rollout_logps: bool = True


TConfig = TypeVar("TConfig", bound=AgenticRLConfig)


class AgenticRLLearner(abc.ABC, Generic[TConfig]):
  """Base class for Agentic RL Learners using asynchronous rollouts."""

  class _AsyncQueueIterator:
    """Async iterator that yields items from a sync queue."""

    def __init__(
        self,
        q: queue.Queue[TrainingInputT | None],
        loop: asyncio.AbstractEventLoop,
    ):
      self.q = q
      self.loop = loop

    def __aiter__(self):
      return self

    async def __anext__(self):
      item = await self.loop.run_in_executor(None, self.q.get)
      if item is None:
        raise StopAsyncIteration
      return item

  def _run_p28_g5c_gate(self, observed_train_example) -> bool:
    """Runs the default-off complete segmented loss gate without an update."""
    import json  # pylint: disable=g-import-not-at-top
    from flax import nnx  # pylint: disable=g-import-not-at-top
    from tunix.rl import alignment  # pylint: disable=g-import-not-at-top
    from tunix.rl import canonical_forward  # pylint: disable=g-import-not-at-top

    train_example, sidecar = alignment.unwrap_train_example(
        observed_train_example
    )
    if sidecar is None:
      raise alignment.AlignmentGateError(
          "P28 G5c requires the four-boundary ObservedTrainExample"
      )
    actor_trainer = self.rl_cluster.actor_trainer
    _, trainer_state = nnx.split(actor_trainer.model)
    adapter = canonical_forward.require_registered()
    def memory_snapshot():
      snapshots = []
      for device in jax.local_devices():
        stats = {}
        try:
          stats = device.memory_stats() or {}
        except Exception:
          pass
        snapshots.append({
            "device": int(device.id),
            "bytes_in_use": stats.get("bytes_in_use"),
            "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
            "bytes_limit": stats.get("bytes_limit"),
        })
      return tuple(snapshots)

    def block_all(tree):
      for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
          leaf.block_until_ready()

    def tree_exact(left, right):
      left_leaves = jax.tree.leaves(left)
      right_leaves = jax.tree.leaves(right)
      return len(left_leaves) == len(right_leaves) and all(
          bool(np.asarray(jnp.array_equal(a, b)))
          for a, b in zip(left_leaves, right_leaves, strict=True)
      )

    before = {
        "model": actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
            nnx.state(actor_trainer.model, nnx.Param)
        ),
        "optimizer": actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
            nnx.state(actor_trainer.optimizer, nnx.optimizer.OptState)
        ),
        "accumulator": actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
            nnx.state(actor_trainer.grad_accumulator), min_elements=1
        ),
    }
    hbm_before = memory_snapshot()
    start = time.perf_counter()
    first = adapter.segmented_grpo_value_and_grad(
        trainer_state=trainer_state,
        train_example=train_example,
        algo_config=self.algo_config,
        pad_id=self.rl_cluster.rollout.pad_id(),
        eos_id=self.rl_cluster.rollout.eos_id(),
    )
    block_all((first["loss"], first["per_token_logps"], first["gradients"]))
    first_seconds = time.perf_counter() - start
    hbm_after_first = memory_snapshot()
    start = time.perf_counter()
    second = adapter.segmented_grpo_value_and_grad(
        trainer_state=trainer_state,
        train_example=train_example,
        algo_config=self.algo_config,
        pad_id=self.rl_cluster.rollout.pad_id(),
        eos_id=self.rl_cluster.rollout.eos_id(),
    )
    block_all((second["loss"], second["per_token_logps"], second["gradients"]))
    repeat_seconds = time.perf_counter() - start
    hbm_after_repeat = memory_snapshot()
    repeat_exact = (
        bool(np.asarray(jnp.array_equal(first["loss"], second["loss"])))
        and bool(np.asarray(jnp.array_equal(
            first["per_token_logps"], second["per_token_logps"]
        )))
        and tree_exact(first["gradients"], second["gradients"])
    )
    gradient_sample = actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
        first["gradients"]
    )
    grad_norm = jnp.sqrt(sum(
        jnp.sum(jnp.square(value.astype(jnp.float32)))
        for value in jax.tree.leaves(first["gradients"])
    ))
    reports = []
    for index in range(8):
      row_sidecar = jax.tree.map(
          lambda value: (
              value[index : index + 1]
              if hasattr(value, "shape")
              and value.shape
              and value.shape[0] == 8
              else value
          ),
          sidecar,
      )
      record = alignment.check_batch(
          row_sidecar,
          t_current=first["per_token_logps"][index : index + 1],
          gradient_norm=grad_norm,
          optimizer_skipped=jnp.asarray(1, jnp.int32),
          step=index,
          fail_closed=False,
      )
      print(
          "[P28.G5C] TRAJECTORY " + json.dumps({
              "alignment": record,
              "reverse": first["reports"][index],
          }, sort_keys=True),
          flush=True,
      )
      reports.append(record)

    after = {
        "model": actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
            nnx.state(actor_trainer.model, nnx.Param)
        ),
        "optimizer": actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
            nnx.state(actor_trainer.optimizer, nnx.optimizer.OptState)
        ),
        "accumulator": actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
            nnx.state(actor_trainer.grad_accumulator), min_elements=1
        ),
    }
    changed = {
        name: actor_trainer._canon_changed_paths(  # pylint: disable=protected-access
            before[name], after[name]
        )
        for name in before
    }
    result = {
        "trajectories": len(reports),
        "cadence": tuple(
            report["boundary"] for report in first["reports"]
        ),
        "all_alignment_pass": all(
            report["verdict"] == "PASS" for report in reports
        ),
        "repeat_exact": repeat_exact,
        "gradient_sha256": tuple(
            leaf["sha256"] for leaf in gradient_sample["leaves"].values()
        ),
        "gradient_sampled_leaves": gradient_sample["sampled_leaves"],
        "gradient_norm": float(np.asarray(grad_norm)),
        "first_seconds": first_seconds,
        "repeat_seconds": repeat_seconds,
        "state_changed": changed,
        "hbm_before": hbm_before,
        "hbm_after_first": hbm_after_first,
        "hbm_after_repeat": hbm_after_repeat,
        "shared_logsoftmax": bool(
            getattr(adapter, "_p28_g5c_shared_logsoftmax", False)
        ),
    }
    print(f"[P28.G5C] RESULT {result}", flush=True)
    if (
        not repeat_exact
        or any(changed.values())
        or not np.isfinite(result["gradient_norm"])
        or result["gradient_norm"] <= 0
    ):
      raise alignment.AlignmentGateError(f"P28 G5c red: {result}")
    return result["all_alignment_pass"]

  def _run_p66_backward_gate(self, observed_train_example) -> dict[str, Any]:
    """Captures one ordinary or segmented full gradient without an update.

    P66 deliberately bypasses both the optimizer and the persistent gradient
    accumulator. The historical DP4xTP1 arms capture complete trees. The
    full-depth TP4 discriminator instead captures bounded state fingerprints,
    the complete layerwise max-abs profile, and the group-0 engine receipt so
    a causal magnitude failure never spends tens of GiB on a dead arm.
    """
    from flax import nnx  # pylint: disable=g-import-not-at-top
    from tunix.rl import canonical_forward  # pylint: disable=g-import-not-at-top

    arm = os.environ.get("CANON_P66_BACKWARD_ARM", "")
    capture_dir = os.environ.get("CANON_P66_BACKWARD_CAPTURE_DIR", "")
    workload = dp_workloads.active_workload(os.environ)
    required = {
        "CANON_P32_TRAIN_ADMITTED": "1",
        "CANON_P32_DP_REDUCTION_ADMITTED": "1",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
        "CANON_P33_RUN_STAGE": "backward-no-commit",
        "CANON_P33_NO_COMMIT": "1",
        "CANON_P28_SEGMENTED_TRAIN": "1",
        "CANON_P28_G6_UPDATE": "1",
        "CANON_P29_FULL_TRAIN": "1",
        "CANON_ALIGNMENT_GATE": "1",
        "CANON_ALIGNMENT_TRAIN": "1",
        "CANON_PRE_ALIGN_GATE": "1",
        "CANON_P60_DETERMINISTIC_AB": "1",
    }
    tp4_arm = arm in (
        "tp4-serial",
        "tp4-p59-old",
        "tp4-p59",
        "tp4-gather-off",
        "tp4-vma-oracle",
    )
    arm_required = {
        "ordinary": ("0", "0", "1"),
        "segmented": ("0", "0", "1"),
        "tp4-serial": ("0", "0", "1"),
        "tp4-p59-old": ("1", "0", "1"),
        "tp4-p59": ("1", "1", "1"),
        "tp4-gather-off": ("1", "1", "0"),
        "tp4-vma-oracle": ("1", "1", "1"),
    }
    rank_parallel, check_vma, fixed_gather = arm_required.get(
        arm, (None, None, None)
    )
    required.update({
        "CANON_P59_RANK_PARALLEL_BACKWARD": rank_parallel,
        "CANON_P66_P59_CHECK_VMA": check_vma,
        "CANON_FIXED_AR_GATHER": fixed_gather,
    })
    wrong = {
        name: os.environ.get(name)
        for name, expected in required.items()
        if os.environ.get(name) != expected
    }
    if (
        arm not in arm_required
        or not capture_dir
        or not os.path.isabs(capture_dir)
        or workload is None
        or (
            (workload.name, workload.dp_size, workload.tp_size)
            != (
                ("gsm8k-p66-dp1-tp4", 1, 4)
                if tp4_arm
                else ("gsm8k-p59-dp4-tp1", 4, 1)
            )
        )
        or os.environ.get("CANON_P61_BACKWARD_NUMERICAL_DIR", "")
        or os.environ.get("CANON_P62_BACKWARD_NUMERIC_DEBUG", "0") != "0"
        or os.environ.get("CANON_P64_P45_NUMERIC_DEBUG", "0") != "0"
        or wrong
    ):
      raise alignment.AlignmentGateError(
          "P66 backward gate requires one exact deterministic registered "
          "GSM8K no-commit carrier: "
          f"arm={arm!r} capture={capture_dir!r} "
          f"workload={getattr(workload, 'name', None)!r} wrong={wrong}"
      )

    train_example, sidecar = alignment.unwrap_train_example(
        observed_train_example
    )
    if sidecar is None:
      raise alignment.AlignmentGateError(
          "P66 backward gate requires an ObservedTrainExample sidecar"
      )
    actor_trainer = self.rl_cluster.actor_trainer
    _, trainer_state = nnx.split(actor_trainer.model)
    reference_state = _p28_reference_state(self.rl_cluster)

    def fingerprint(value, *, min_elements=128):
      return actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
          value, min_elements=min_elements
      )

    before = {
        "model": fingerprint(nnx.state(actor_trainer.model, nnx.Param)),
        "optimizer": fingerprint(
            nnx.state(actor_trainer.optimizer, nnx.optimizer.OptState)
        ),
        "accumulator": fingerprint(
            nnx.state(actor_trainer.grad_accumulator), min_elements=1
        ),
        "reference": (
            fingerprint(reference_state) if reference_state is not None else None
        ),
        "train_steps": actor_trainer.train_steps,
    }
    if not tp4_arm:
      _p61_capture_tree(
          capture_dir,
          "model_before",
          nnx.state(actor_trainer.model, nnx.Param),
      )

    started = time.perf_counter()
    with self.rl_cluster._get_mesh_and_logical_axis_rules_cm(  # pylint: disable=protected-access
        rl_cluster_lib.Role.ACTOR
    ):
      if arm == "ordinary":

        def ordinary_value_and_grad(model, current_train_example):
          def diff_fn(differentiated_model):
            output = actor_trainer.loss_fn(
                differentiated_model,
                train_example=current_train_example,
                algo_config=self.algo_config,
            )
            if not isinstance(output, sft_utils.LossOutput):
              raise TypeError(
                  "P66 ordinary backward requires the GRPO LossOutput"
              )
            return output.primary_loss.unreduced_sum, output

          grad_fn = nnx.value_and_grad(
              diff_fn,
              argnums=(
                  nnx.DiffState(0, nnx.LoRAParam)
                  if actor_trainer._lora_enabled  # pylint: disable=protected-access
                  else 0
              ),
              has_aux=True,
          )
          return grad_fn(model)

        compiled = nnx.jit(ordinary_value_and_grad)
        (_, loss_output), gradients = compiled(
            actor_trainer.model, train_example
        )
        scale = loss_output.primary_loss.compute_scale()
        gradients = jax.tree.map(lambda value: value * scale, gradients)
        per_token_logps = loss_output.aux_metrics.get("canon/T_current")
      else:
        adapter = canonical_forward.require_registered()
        result = adapter.segmented_dp_grpo_value_and_grad(
            trainer_state=trainer_state,
            train_example=train_example,
            algo_config=self.algo_config,
            pad_id=self.rl_cluster.rollout.pad_id(),
            eos_id=self.rl_cluster.rollout.eos_id(),
            gradient_microbatch_sink=None,
        )
        gradients = result["gradients"]
        per_token_logps = result["per_token_logps"]

    if per_token_logps is None:
      raise alignment.AlignmentGateError(
          "P66 backward arm did not return canonical T_current"
      )
    gradient_stats = sft_utils.tree_numeric_receipt(gradients)
    gradient_norm = jnp.asarray(
        gradient_stats["stable_norm"], dtype=jnp.float32
    )
    if (
        not gradient_stats["all_finite"]
        or not gradient_stats["any_nonzero"]
        or not np.isfinite(float(gradient_stats["stable_norm"]))
        or (tp4_arm and gradient_stats["stable_norm"] > 1.0e6)
    ):
      raise alignment.AlignmentGateError(
          f"P66 {arm} backward emitted an invalid gradient: {gradient_stats}"
      )
    gradient_sample = fingerprint(gradients)
    if not tp4_arm:
      _p61_capture_tree(capture_dir, "gradient", gradients)

    contract = workload.training_contract()
    records = []
    for index, rows in enumerate(contract.rank_major_reverse_groups()):
      row_indices = np.asarray(rows, dtype=np.int32)
      group_sidecar = jax.tree.map(
          lambda value: (
              value[row_indices]
              if hasattr(value, "shape")
              and value.shape
              and value.shape[0] == contract.global_trajectories
              else value
          ),
          sidecar,
      )
      record = alignment.check_batch(
          group_sidecar,
          t_current=per_token_logps[row_indices],
          gradient_norm=gradient_norm,
          optimizer_skipped=jnp.asarray(1, jnp.int32),
          step=index,
          fail_closed=True,
      )
      records.append(record)

    after = {
        "model": fingerprint(nnx.state(actor_trainer.model, nnx.Param)),
        "optimizer": fingerprint(
            nnx.state(actor_trainer.optimizer, nnx.optimizer.OptState)
        ),
        "accumulator": fingerprint(
            nnx.state(actor_trainer.grad_accumulator), min_elements=1
        ),
        "reference": (
            fingerprint(_p28_reference_state(self.rl_cluster))
            if reference_state is not None
            else None
        ),
        "train_steps": actor_trainer.train_steps,
    }
    changed = {
        name: actor_trainer._canon_changed_paths(  # pylint: disable=protected-access
            before[name], after[name]
        )
        for name in ("model", "optimizer", "accumulator", "reference")
        if before[name] is not None
    }
    unchanged = (
        not any(changed.values())
        and after["train_steps"] == before["train_steps"]
    )
    report = {
        "schema": "canon-p66-backward-gate-v1",
        "arm": arm,
        "verdict": "PASS" if unchanged else "FAIL",
        "commits": 0,
        "dp_size": workload.dp_size,
        "tp_size": workload.tp_size,
        "global_trajectories": workload.global_trajectories,
        "gradient_groups": workload.gradient_groups,
        "gradient": gradient_stats,
        "gradient_sample": gradient_sample,
        "model_before_sample": before["model"],
        "engine_vjp": result.get("p66_engine_receipt") if tp4_arm else None,
        "layerwise_profile": (
            result.get("p66_layerwise_profile") if tp4_arm else None
        ),
        "row_cotangent_summary": (
            result.get("p66_row_cotangent_summary") if tp4_arm else None
        ),
        "vjp_oracle": (
            result.get("p66_vjp_oracle") if tp4_arm else None
        ),
        "alignment_hashes": [record["hashes"] for record in records],
        "alignment_verdicts": [record["verdict"] for record in records],
        "state_changed_paths": changed,
        "train_steps_before": before["train_steps"],
        "train_steps_after": after["train_steps"],
        "seconds": time.perf_counter() - started,
    }
    report_path = os.environ.get("CANON_UPDATE_REPORT", "")
    if not report_path:
      raise alignment.AlignmentGateError(
          "P66 backward gate requires CANON_UPDATE_REPORT"
      )
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "x", encoding="utf-8") as output_file:
      json.dump(report, output_file, indent=2, sort_keys=True)
      output_file.write("\n")
    print(
        "[P66.BACKWARD] "
        f"arm={arm} verdict={report['verdict']} commits=0 "
        f"alignments={len(records)}/{workload.gradient_groups} "
        f"gradient_norm={gradient_stats['stable_norm']} "
        f"seconds={report['seconds']:.3f}",
        flush=True,
    )
    if not unchanged:
      raise alignment.AlignmentGateError(
          f"P66 {arm} backward mutated training state: {report}"
      )
    return report

  def _run_p28_g6_update(
      self, observed_train_example, *, xprof_train_schedule=None
  ) -> dict[str, Any]:
    """Streams the proven segmented gradient into an attested real update."""
    import json  # pylint: disable=g-import-not-at-top
    from flax import nnx  # pylint: disable=g-import-not-at-top
    from tunix.rl import alignment  # pylint: disable=g-import-not-at-top
    from tunix.rl import canonical_forward  # pylint: disable=g-import-not-at-top
    from tunix.rl import deepswe_contract  # pylint: disable=g-import-not-at-top

    p31_convergence = os.environ.get("CANON_P31_CONVERGENCE", "") == "1"
    p33_workload = (
        os.environ.get("CANON_P33_WORKLOAD_LAUNCH_ADMITTED", "") == "1"
    )
    p34_workload = os.environ.get("CANON_P34_DEEPSWE", "") == "1"
    canonical_workload = p33_workload or p34_workload
    (
        expected_trajectories,
        expected_trajectory_micro,
        marker_prefix,
        geometry_is_canonical,
    ) = _segmented_update_geometry(os.environ)
    if geometry_is_canonical != canonical_workload:
      raise alignment.AlignmentGateError(
          "segmented update geometry classification is inconsistent"
      )
    p33_no_commit = (
        canonical_workload
        and (
            os.environ.get("CANON_P34_NO_COMMIT", "") == "1"
            if p34_workload
            else os.environ.get("CANON_P33_NO_COMMIT", "") == "1"
        )
    )
    run_stage = os.environ.get(
        "CANON_P34_RUN_STAGE" if p34_workload else "CANON_P33_RUN_STAGE",
        "",
    )
    gsm8k_train = os.environ.get("CANON_GSM8K_TRAIN", "") == "1"
    expected_mode = (
        "train"
        if p31_convergence or canonical_workload or gsm8k_train
        else "update-canary"
    )
    if alignment.execution_mode() != expected_mode:
      raise alignment.AlignmentGateError(
          f"segmented update requires alignment mode {expected_mode!r}"
      )
    train_example, sidecar = alignment.unwrap_train_example(
        observed_train_example
    )
    if sidecar is None:
      raise alignment.AlignmentGateError(
          "P28 G6 requires the four-boundary ObservedTrainExample"
      )
    actor_trainer = self.rl_cluster.actor_trainer
    resident_requested = (
        os.environ.get("CANON_OPT_STATE_RESIDENT", "0") == "1"
    )
    if resident_requested and actor_trainer.config.optimizer_offload:
      raise alignment.AlignmentGateError(
          "optimizer resident and offload modes are mutually exclusive"
      )
    if p33_workload:
      optimizer_placement = dp_workloads.canonical_optimizer_placement(
          os.environ, require_explicit=True
      )
    elif actor_trainer.config.optimizer_offload:
      optimizer_placement = "pinned-host-offload"
    elif resident_requested:
      optimizer_placement = "device-resident"
    else:
      optimizer_placement = "device-unattested"
    full_train = os.environ.get("CANON_P29_FULL_TRAIN", "") == "1"
    num_trajectories = int(train_example.completion_ids.shape[0])
    if p34_workload:
      deepswe_contract.validate_environment(os.environ)
      workload = deepswe_contract.active_workload(os.environ)
      trajectory_micro = workload.local_trajectories
    elif p33_workload:
      workload = dp_workloads.active_workload()
      if workload is None:
        raise alignment.AlignmentGateError(
            "P33 update requires an active canonical workload"
        )
      dp_workloads.validate_environment(
          workload, require_reduction_admission=True
      )
      trajectory_micro = workload.local_trajectories
    else:
      trajectory_micro = expected_trajectory_micro
    p61_capture_dir = os.environ.get(
        "CANON_P61_BACKWARD_NUMERICAL_DIR", ""
    )
    if p61_capture_dir:
      p61_contract = (
          p33_workload
          and workload.name == "gsm8k-p59-dp4-tp1"
          and workload.dp_size == 4
          and workload.tp_size == 1
          and run_stage == "one-update"
          and not p33_no_commit
          and full_train
          and os.environ.get("CANON_P60_DETERMINISTIC_AB", "") == "1"
          and os.environ.get("CANON_P59_DP4_TAIL8", "0") == "0"
      )
      if not p61_contract or not os.path.isabs(p61_capture_dir):
        raise alignment.AlignmentGateError(
            "CANON_P61_BACKWARD_NUMERICAL_DIR requires exact committed "
            "gsm8k-p59-dp4-tp1 one-update deterministic DP4xTP1 geometry"
        )
    p62_value = os.environ.get("CANON_P62_BACKWARD_NUMERIC_DEBUG", "")
    if p62_value not in ("", "0", "1"):
      raise alignment.AlignmentGateError(
          "CANON_P62_BACKWARD_NUMERIC_DEBUG must be unset/0/1"
      )
    p62_numeric_debug = p62_value == "1"
    p64_value = os.environ.get("CANON_P64_P45_NUMERIC_DEBUG", "")
    if p64_value not in ("", "0", "1"):
      raise alignment.AlignmentGateError(
          "CANON_P64_P45_NUMERIC_DEBUG must be unset/0/1"
      )
    p64_numeric_debug = p64_value == "1"
    if p62_numeric_debug and p64_numeric_debug:
      raise alignment.AlignmentGateError(
          "P62 and P64 numerical observers are mutually exclusive"
      )
    p64_capsule_mode = p64_training_capsule.mode()
    if p64_numeric_debug != bool(p64_capsule_mode):
      raise alignment.AlignmentGateError(
          "P64 numerical debug and training-capsule mode must be enabled "
          "together"
      )
    if p62_numeric_debug and not (
        p33_workload
        and workload.name == "gsm8k"
        and (workload.dp_size, workload.tp_size) == (16, 4)
        and run_stage == "backward-no-commit"
        and p33_no_commit
        and full_train
        and os.environ.get("CANON_P59_RANK_PARALLEL_BACKWARD", "") == "1"
        and os.environ.get("CANON_P38_FIXED_LM_HEAD", "") == "1"
        and os.environ.get("CANON_V1_HP_FULL", "0") == "0"
    ):
      raise alignment.AlignmentGateError(
          "P62 numerical debug requires exact strict GSM8K DP16xTP4 "
          "P59 fixed-head backward-no-commit geometry"
      )
    if p64_numeric_debug and not (
        p33_workload
        and workload.name == "frozenlake-dp8-tp8"
        and (workload.dp_size, workload.tp_size) == (8, 8)
        and run_stage == "backward-no-commit"
        and p33_no_commit
        and full_train
        and os.environ.get("CANON_P59_RANK_PARALLEL_BACKWARD", "") == "1"
        and os.environ.get("CANON_P38_FIXED_LM_HEAD", "") == "1"
        and os.environ.get("CANON_V1_HP_FULL", "0") == "0"
    ):
      raise alignment.AlignmentGateError(
          "P64 numerical debug requires exact strict P45 DP8xTP8 "
          "P59 fixed-head backward-no-commit geometry"
      )
    numeric_debug = p62_numeric_debug or p64_numeric_debug
    numeric_marker = "P62" if p62_numeric_debug else "P64"
    numeric_schema = (
        "canon-p62" if p62_numeric_debug else "canon-p64"
    )
    if num_trajectories % trajectory_micro:
      raise alignment.AlignmentGateError(
          "segmented update trajectory cadence changed: "
          f"trajectories={num_trajectories} micro={trajectory_micro}"
      )
    registered_microbatches = (
        workload.local_trajectories
        if canonical_workload
        else num_trajectories // trajectory_micro
    )
    expected_microbatches = (
        p64_training_capsule.reverse_group_limit(registered_microbatches)
        if p64_numeric_debug
        else registered_microbatches
    )
    if canonical_workload:
      expected_trajectories = workload.global_trajectories
    if num_trajectories != expected_trajectories:
      raise alignment.AlignmentGateError(
          "segmented update trajectory contract changed: "
          f"{num_trajectories} != {expected_trajectories}"
      )
    _, trainer_state = nnx.split(actor_trainer.model)
    adapter = canonical_forward.require_registered()
    sharding_profile_enabled = (
        os.environ.get("CANON_P30_SHARDING_PROFILE", "") == "1"
    )

    def memory_snapshot():
      snapshots = []
      for device in jax.local_devices():
        stats = {}
        try:
          stats = device.memory_stats() or {}
        except Exception:
          pass
        snapshots.append({
            "device": int(device.id),
            "bytes_in_use": stats.get("bytes_in_use"),
            "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
            "bytes_limit": stats.get("bytes_limit"),
        })
      return tuple(snapshots)

    def fingerprint(value, *, min_elements=128):
      return actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
          value, min_elements=min_elements
      )

    def emit_sharding_inventory(boundary):
      if not sharding_profile_enabled:
        return
      inventory = {
          "model": _p30_sharding_inventory(
              nnx.state(actor_trainer.model, nnx.Param)
          ),
          "optimizer": _p30_sharding_inventory(
              nnx.state(actor_trainer.optimizer, nnx.optimizer.OptState)
          ),
          "accumulator": _p30_sharding_inventory(
              nnx.state(actor_trainer.grad_accumulator)
          ),
      }
      print(
          "[P30.G2] SHARDING_INVENTORY "
          f"boundary={boundary} "
          f"inventory={json.dumps(inventory, sort_keys=True)}",
          flush=True,
      )

    ref_state = _p28_reference_state(self.rl_cluster)
    before = {
        "model": fingerprint(nnx.state(actor_trainer.model, nnx.Param)),
        "optimizer": fingerprint(
            nnx.state(actor_trainer.optimizer, nnx.optimizer.OptState)
        ),
        "accumulator": fingerprint(
            nnx.state(actor_trainer.grad_accumulator), min_elements=1
        ),
        "reference": (
            fingerprint(ref_state) if ref_state is not None else None
        ),
        "train_steps": actor_trainer.train_steps,
    }
    if p64_numeric_debug:
      p64_training_capsule.bind_or_verify_model(before["model"])
    hbm_before = memory_snapshot()
    emit_sharding_inventory("before_reverse")
    optimizer_memory_kinds_before = (
        actor_trainer.optimizer_state_memory_kinds()
    )
    expected_optimizer_memory_kind = (
        "pinned_host"
        if optimizer_placement == "pinned-host-offload"
        else "device"
    )
    if optimizer_placement != "device-unattested":
      if optimizer_memory_kinds_before != (expected_optimizer_memory_kind,):
        raise alignment.AlignmentGateError(
            "optimizer state placement before reverse is invalid: "
            f"{optimizer_memory_kinds_before!r}"
        )
      print(
          "[P41.OPTIMIZER] before_reverse "
          f"placement={optimizer_placement} "
          f"memory_kind={expected_optimizer_memory_kind}",
          flush=True,
      )
    micro_norms = []
    fused_pair_accumulation = (
        os.environ.get("CANON_P30_FUSED_PAIR_ACCUMULATION", "") == "1"
    )
    if canonical_workload and fused_pair_accumulation:
      raise alignment.AlignmentGateError(
          "P33 uses rank-reduced scaled groups, not pair accumulation"
      )
    if fused_pair_accumulation:
      print(
          "[P30.G2] FUSED_PAIR_ACCUMULATION on order=(left+right)*scale",
          flush=True,
      )

    def consume_microbatch(index, gradients):
      norm = actor_trainer.accumulate_precomputed_gradient_microbatch(
          gradients, microbatch_index=index
      )
      micro_norms.append(norm)
      if index < expected_microbatches - 1:
        print(
            f"{marker_prefix} update_accumulation_pending "
            f"train_steps={actor_trainer.train_steps} "
            f"microstep={index + 1}/{expected_microbatches}",
            flush=True,
        )

    def consume_pair(index, left, right, multiplier):
      norm = actor_trainer.accumulate_precomputed_gradient_pair_microbatch(
          left,
          right,
          multiplier,
          microbatch_index=index,
      )
      micro_norms.append(norm)
      if index < expected_microbatches - 1:
        print(
            f"{marker_prefix} update_accumulation_pending "
            f"train_steps={actor_trainer.train_steps} "
            f"microstep={index + 1}/{expected_microbatches}",
            flush=True,
        )

    def consume_scaled(index, gradients, multiplier):
      if p33_no_commit:
        multiplier = jnp.asarray(multiplier, jnp.float32)
        if numeric_debug:
          numeric_stats = sft_utils.scaled_tree_numeric_stats(
              gradients, multiplier
          )
          numeric_receipt = sft_utils.tree_numeric_receipt(
              gradients, stats=numeric_stats
          )
          numeric_record = {
              "schema": f"{numeric_schema}-tree-numeric-v1",
              "stage": "scaled_microgradient",
              "group": index,
              "groups": registered_microbatches,
              "executed_groups": expected_microbatches,
              "multiplier": float(np.asarray(multiplier)),
              **numeric_receipt,
          }
          if (
              index in (0, expected_microbatches - 1)
              or not numeric_receipt["all_finite"]
              or not numeric_receipt["naive_norm_finite"]
          ):
            print(
                f"[{numeric_marker}.NUMERIC] "
                + json.dumps(
                    numeric_record,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
          if not numeric_receipt["all_finite"]:
            raise alignment.AlignmentGateError(
                f"{numeric_marker} first non-finite numerical boundary: "
                f"stage=scaled_microgradient group={index} "
                f"first={numeric_receipt['first_nonfinite']}"
            )
          actor_trainer.accumulate_precomputed_scaled_gradient_microbatch(
              gradients,
              multiplier,
              microbatch_index=index,
          )
          norm = jnp.asarray(
              numeric_receipt["stable_norm"], dtype=jnp.float32
          )
        else:
          norm = jnp.sqrt(sum(
              jnp.sum(jnp.square(
                  value.astype(jnp.float32) * multiplier
              ))
              for value in jax.tree.leaves(gradients)
          ))
          norm.block_until_ready()
      else:
        norm = actor_trainer.accumulate_precomputed_scaled_gradient_microbatch(
            gradients,
            multiplier,
            microbatch_index=index,
        )
      micro_norms.append(norm)
      if index < expected_microbatches - 1:
        print(
            f"{marker_prefix} update_accumulation_pending "
            f"train_steps={actor_trainer.train_steps} "
            f"microstep={index + 1}/{expected_microbatches}",
            flush=True,
        )

    if p61_capture_dir:
      if before["train_steps"] != 0:
        raise alignment.AlignmentGateError(
            "P61 numerical capture requires the first optimizer transaction"
        )
      _p61_capture_tree(
          p61_capture_dir,
          "model_before",
          nnx.state(actor_trainer.model, nnx.Param),
      )
    start = time.perf_counter()
    with self.rl_cluster._get_mesh_and_logical_axis_rules_cm(  # pylint: disable=protected-access
        rl_cluster_lib.Role.ACTOR
    ):
      common = {
          "trainer_state": trainer_state,
          "train_example": train_example,
          "algo_config": self.algo_config,
          "pad_id": self.rl_cluster.rollout.pad_id(),
          "eos_id": self.rl_cluster.rollout.eos_id(),
      }
      value_and_grad_start = time.perf_counter()
      if canonical_workload:
        segmented_kwargs = {
            "gradient_microbatch_sink": consume_scaled,
            "deterministic_repeat": (p34_workload and p33_no_commit),
        }
        if xprof_train_schedule is not None:
          segmented_kwargs["xprof_train_schedule"] = xprof_train_schedule
        result = adapter.segmented_dp_grpo_value_and_grad(
            **common,
            **segmented_kwargs,
        )
      else:
        result = adapter.segmented_grpo_value_and_grad(
            **common,
            gradient_microbatch_sink=(
                None if fused_pair_accumulation else consume_microbatch
            ),
            gradient_pair_sink=(
                consume_pair if fused_pair_accumulation else None
            ),
        )
      value_and_grad_call_done = time.perf_counter()
    result["loss"].block_until_ready()
    if perf_log.enabled():
      value_and_grad_done = time.perf_counter()
      print(
          "[PERF] stage=segmented_value_and_grad seconds=%.3f call=%.3f"
          " block=%.3f microbatches=%d"
          % (
              value_and_grad_done - value_and_grad_start,
              value_and_grad_call_done - value_and_grad_start,
              value_and_grad_done - value_and_grad_call_done,
              int(result["gradient_microbatches"]),
          ),
          flush=True,
      )
    if (
        result["gradient_microbatches"] != expected_microbatches
        or len(micro_norms) != expected_microbatches
    ):
      raise alignment.AlignmentGateError(
          "segmented update gradient microbatch count changed: "
          f"result={result['gradient_microbatches']} "
          f"norms={len(micro_norms)} expected={expected_microbatches}"
      )
    checked_vma_value = os.environ.get("CANON_P59_CHECKED_VMA", "0")
    first_update_gate_value = os.environ.get(
        "CANON_V1_HP_FIRST_UPDATE_GATE", "0"
    )
    if checked_vma_value not in ("0", "1"):
      raise alignment.AlignmentGateError(
          "CANON_P59_CHECKED_VMA must be exactly 0 or 1"
      )
    if first_update_gate_value not in ("0", "1"):
      raise alignment.AlignmentGateError(
          "CANON_V1_HP_FIRST_UPDATE_GATE must be exactly 0 or 1"
      )
    checked_vma_full = checked_vma_value == "1"
    first_update_gate_enabled = first_update_gate_value == "1"
    workload_identity = (
        workload.contract_name if p34_workload else workload.name
    )
    if checked_vma_full or first_update_gate_enabled:
      exact_checked_vma_geometry = (
          full_train
          and not p33_no_commit
          and os.environ.get("CANON_V1_HP_FULL", "0") == "1"
          and os.environ.get("CANON_P59_RANK_PARALLEL_BACKWARD", "0")
          == "1"
          and os.environ.get("CANON_P66_P59_CHECK_VMA", "0") == "1"
          and (
              (
                  workload_identity == "gsm8k"
                  and (workload.dp_size, workload.tp_size) == (16, 4)
                  and workload.global_m == 4096
              )
              or (
                  workload_identity == "frozenlake-dp8-tp8"
                  and (workload.dp_size, workload.tp_size) == (8, 8)
                  and workload.global_m == 2048
              )
              or (
                  workload_identity == "p58-qwen4b-tim-128"
                  and (workload.dp_size, workload.tp_size) == (8, 8)
                  and workload.global_m == 2048
                  and run_stage == "full"
                  and os.environ.get("CANON_P58_DEEPSWE_TIM", "0") == "1"
                  and os.environ.get("CANON_P58_TIM_ADMITTED", "0") == "1"
                  and os.environ.get("CANON_P58_TIM_ARM", "") == "zero"
                  and os.environ.get(
                      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY", "1"
                  ) == "0"
              )
          )
      )
      if not exact_checked_vma_geometry or not (
          checked_vma_full and first_update_gate_enabled
      ):
        raise alignment.AlignmentGateError(
            "checked-VMA/first-update gate requires the exact complete "
            "registered full bundle"
        )
      print(
          "[P59.CHECKED_VMA] enabled=1 "
          f"workload={workload_identity} dp={workload.dp_size} "
          f"tp={workload.tp_size} global_M={workload.global_m} "
          "manual_axes=data,model compatibility_alias=1",
          flush=True,
      )
    gradient_deterministic = result.get("gradient_deterministic_repeat")
    if p34_workload and p33_no_commit:
      if gradient_deterministic is not True:
        raise alignment.AlignmentGateError(
            "P34 repeated backward-no-commit gradients are not array-exact"
        )
      print(
          f"[CANON_P34_DP{workload.dp_size}] "
          "deterministic_repeat array_exact=1 repeats=2",
          flush=True,
      )
    hbm_after_accumulation = memory_snapshot()
    emit_sharding_inventory("after_accumulation")

    records = []
    activity = []
    for index in range(expected_microbatches):
      rows = (
          tuple(result["reports"][index]["trajectory_rows"])
          if canonical_workload
          else tuple(range(
              index * trajectory_micro,
              (index + 1) * trajectory_micro,
          ))
      )
      pair_sidecar = jax.tree.map(
          lambda value: (
              value[np.asarray(rows, dtype=np.int32)]
              if hasattr(value, "shape")
              and value.shape
              and value.shape[0] == num_trajectories
              else value
          ),
          sidecar,
      )
      active = (
          result["reports"][index]["gradient_nonzero"] > 0
          if canonical_workload
          else any(
              report["loss_cotangent"]["nonzero"] > 0
              for report in result["reports"][rows[0] : rows[-1] + 1]
          )
      )
      gradient_finite = (
          bool(result["reports"][index].get("gradient_finite", False))
          if canonical_workload
          else True
      )
      norm_value = float(np.asarray(micro_norms[index]))
      if (
          not gradient_finite
          or (norm_value > 0.0) != active
          or not np.isfinite(norm_value)
      ):
        raise alignment.AlignmentGateError(
            "P28 G6 microgradient activity mismatch: "
            f"index={index} active={active} finite={gradient_finite} "
            f"norm={norm_value}"
        )
      activity.append(active)
      record = alignment.check_batch(
          pair_sidecar,
          t_current=result["per_token_logps"][
              np.asarray(rows, dtype=np.int32)
          ],
          gradient_norm=micro_norms[index],
          optimizer_skipped=jnp.asarray(
              1 if p33_no_commit else 0, jnp.int32
          ),
          step=(
              before["train_steps"] * expected_microbatches + index
              if full_train
              else index
          ),
          fail_closed=True,
      )
      records.append(record)
    if not any(activity) and (not full_train or p33_no_commit):
      raise alignment.AlignmentGateError(
          "INCONCLUSIVE_NO_SIGNAL: all four G6 microgradients are zero"
      )

    p62_accumulator_denominator = None
    if numeric_debug:
      accumulator_tree = actor_trainer.grad_accumulator.get()
      accumulator_receipt = sft_utils.tree_numeric_receipt(accumulator_tree)
      p62_accumulator_denominator = float(np.asarray(jax.device_get(
          actor_trainer.grad_accumulator.denom[...]
      )))
      accumulator_record = {
          "schema": f"{numeric_schema}-tree-numeric-v1",
          "stage": "final_accumulator",
          "group": expected_microbatches - 1,
          "groups": registered_microbatches,
          "executed_groups": expected_microbatches,
          "accumulator_denominator": p62_accumulator_denominator,
          **accumulator_receipt,
      }
      print(
          f"[{numeric_marker}.NUMERIC] "
          + json.dumps(
              accumulator_record, sort_keys=True, separators=(",", ":")
          ),
          flush=True,
      )
      if (
          not accumulator_receipt["all_finite"]
          or p62_accumulator_denominator != float(expected_microbatches)
      ):
        raise alignment.AlignmentGateError(
            f"{numeric_marker} final accumulator contract failed: "
            f"{accumulator_record}"
        )
      with self.rl_cluster._get_mesh_and_logical_axis_rules_cm(  # pylint: disable=protected-access
          rl_cluster_lib.Role.ACTOR
      ):
        discarded_denominator = actor_trainer.discard_precomputed_gradients()
      if float(np.asarray(discarded_denominator)) != float(
          expected_microbatches
      ):
        raise alignment.AlignmentGateError(
            f"{numeric_marker} discarded accumulator denominator changed: "
            f"{float(np.asarray(discarded_denominator))} != "
            f"{expected_microbatches}"
        )
      print(
          f"[{numeric_marker}.NUMERIC] discard_complete optimizer_commits=0 "
          f"microsteps={expected_microbatches} "
          f"denominator={p62_accumulator_denominator} "
          f"diagnostic_replay={int(p64_capsule_mode == 'replay')}",
          flush=True,
      )

    p58_all_filtered = (
        p34_workload
        and workload.contract_name == "p58-qwen4b-tim-128"
        and float(np.asarray(
            result["loss_output"].primary_loss.denominator
        )) == 0.0
    )
    if p58_all_filtered:
      with self.rl_cluster._get_mesh_and_logical_axis_rules_cm(  # pylint: disable=protected-access
          rl_cluster_lib.Role.ACTOR
      ):
        discarded_denominator = actor_trainer.discard_precomputed_gradients()
      if float(np.asarray(discarded_denominator)) != float(
          expected_microbatches
      ):
        raise alignment.AlignmentGateError(
            "P58 all-filtered accumulator cadence changed: "
            f"{float(np.asarray(discarded_denominator))} != "
            f"{expected_microbatches}"
        )
      after_discard = {
          "model": fingerprint(nnx.state(actor_trainer.model, nnx.Param)),
          "optimizer": fingerprint(
              nnx.state(actor_trainer.optimizer, nnx.optimizer.OptState)
          ),
          "accumulator": fingerprint(
              nnx.state(actor_trainer.grad_accumulator), min_elements=1
          ),
          "reference": (
              fingerprint(_p28_reference_state(self.rl_cluster))
              if ref_state is not None else None
          ),
          "train_steps": actor_trainer.train_steps,
      }
      changed = {
          name: actor_trainer._canon_changed_paths(  # pylint: disable=protected-access
              before[name], after_discard[name]
          )
          for name in ("model", "optimizer", "accumulator", "reference")
          if before[name] is not None
      }
      unchanged = (
          not any(changed.values())
          and actor_trainer.train_steps == before["train_steps"]
      )
      skip_record = {
          "contract_name": workload.contract_name,
          "dp_size": workload.dp_size,
          "tp_size": workload.tp_size,
          "global_m": workload.global_m,
          "verdict": "PASS" if unchanged else "FAIL",
          "mode": "compact-filtered-no-commit",
          "microsteps": expected_microbatches,
          "commits": 0,
          "train_steps_before": before["train_steps"],
          "train_steps_after": actor_trainer.train_steps,
          "loss_denominator": 0.0,
          "gradient_activity": activity,
          "gradient_finite": all(
              np.isfinite(float(np.asarray(value))) for value in micro_norms
          ),
          "dp_replicas_exact": result["replica_equality"],
          "dp_axis": result["dp_axis"],
          "dp_reduction_transactions": result["dp_reduction_transactions"],
          "dp_reduction_rounds_per_transaction": result[
              "dp_reduction_rounds_per_transaction"
          ],
          "dp_rank_pullbacks_per_transaction": result[
              "dp_rank_pullbacks_per_transaction"
          ],
          "micro_gradient_norms": [
              float(np.asarray(value)) for value in micro_norms
          ],
          "changed_paths": changed,
          "alignment_hashes": [record["hashes"] for record in records],
          "hbm_before": hbm_before,
          "hbm_after_accumulation": hbm_after_accumulation,
          "optimizer_memory_kinds_before": list(
              optimizer_memory_kinds_before
          ),
          "optimizer_memory_kinds_after": list(
              actor_trainer.optimizer_state_memory_kinds()
          ),
          "optimizer_placement": optimizer_placement,
      }
      report_path = os.environ.get("CANON_UPDATE_REPORT", "")
      if not report_path:
        raise alignment.AlignmentGateError("CANON_UPDATE_REPORT is required")
      os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
      with open(report_path, "a", encoding="utf-8") as update_file:
        update_file.write(json.dumps(skip_record, sort_keys=True) + "\n")
      print(
          "[DEEPSWE.COMPACT_FILTER] canonical all_filtered=1 "
          f"train_steps={actor_trainer.train_steps} optimizer_commits=0",
          flush=True,
      )
      if not unchanged:
        raise alignment.AlignmentGateError(
            f"P58 all-filtered discard mutated state: {skip_record}"
        )
      return skip_record

    if p33_no_commit:
      after_no_commit = {
          "model": fingerprint(nnx.state(actor_trainer.model, nnx.Param)),
          "optimizer": fingerprint(
              nnx.state(actor_trainer.optimizer, nnx.optimizer.OptState)
          ),
          "accumulator": fingerprint(
              nnx.state(actor_trainer.grad_accumulator), min_elements=1
          ),
          "reference": (
              fingerprint(_p28_reference_state(self.rl_cluster))
              if ref_state is not None else None
          ),
      }
      changed = {
          name: actor_trainer._canon_changed_paths(  # pylint: disable=protected-access
              before[name], after_no_commit[name]
          )
          for name in ("model", "optimizer", "accumulator")
      }
      reference_changed = (
          actor_trainer._canon_changed_paths(  # pylint: disable=protected-access
              before["reference"], after_no_commit["reference"]
          )
          if before["reference"] is not None else []
      )
      unchanged = (
          not any(changed.values())
          and not reference_changed
          and actor_trainer.train_steps == before["train_steps"]
      )
      no_commit_record = {
          "contract_name": (
              workload.contract_name
              if p34_workload
              else workload.name
              if p33_workload
              else "legacy-segmented"
          ),
          "dp_size": workload.dp_size if canonical_workload else 1,
          "tp_size": workload.tp_size if canonical_workload else 1,
          "global_m": workload.global_m if canonical_workload else 256,
          "verdict": "PASS" if unchanged else "FAIL",
          "mode": run_stage,
          "microsteps": expected_microbatches,
          "commits": 0,
          "train_steps_before": before["train_steps"],
          "train_steps_after": actor_trainer.train_steps,
          "gradient_activity": activity,
          "gradient_finite": all(
              np.isfinite(float(np.asarray(value))) for value in micro_norms
          ),
          "gradient_deterministic": gradient_deterministic,
          "dp_replicas_exact": result["replica_equality"],
          "dp_axis": result["dp_axis"],
          "dp_reduction_transactions": result["dp_reduction_transactions"],
          "dp_reduction_rounds_per_transaction": result[
              "dp_reduction_rounds_per_transaction"
          ],
          "dp_rank_pullbacks_per_transaction": result[
              "dp_rank_pullbacks_per_transaction"
          ],
          "dp_pullback_invocations_per_transaction": result.get(
              "dp_pullback_invocations_per_transaction",
              result["dp_rank_pullbacks_per_transaction"],
          ),
          "micro_gradient_norms": [
              float(np.asarray(value)) for value in micro_norms
          ],
          "p62_numeric_debug": p62_numeric_debug,
          "p64_numeric_debug": p64_numeric_debug,
          "p62_accumulator_denominator": p62_accumulator_denominator,
          "model_changed_paths": changed["model"],
          "optimizer_changed_paths": changed["optimizer"],
          "accumulator_changed_paths": changed["accumulator"],
          "reference_changed_paths": reference_changed,
          "alignment_hashes": [record["hashes"] for record in records],
          "hbm_before": hbm_before,
          "hbm_after_reverse": hbm_after_accumulation,
          "optimizer_memory_kinds_before": list(
              optimizer_memory_kinds_before
          ),
            "optimizer_placement": optimizer_placement,
            "state_fingerprints_before": before,
            "state_fingerprints_after": after_no_commit,
        }
      report_path = os.environ.get("CANON_UPDATE_REPORT", "")
      if not report_path:
        raise alignment.AlignmentGateError("CANON_UPDATE_REPORT is required")
      os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
      with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(no_commit_record, report_file, indent=2, sort_keys=True)
        report_file.write("\n")
      print(
          f"{marker_prefix} backward_no_commit "
          f"verdict={no_commit_record['verdict']} commits=0 "
          f"microsteps={expected_microbatches}",
          flush=True,
      )
      if not unchanged:
        raise alignment.AlignmentGateError(
            f"P33 backward no-commit mutated training state: {no_commit_record}"
        )
      return no_commit_record

    first_update_admission = bool(
        first_update_gate_enabled and before["train_steps"] == 0
    )
    if first_update_admission:
      accumulator_tree = actor_trainer.grad_accumulator.get()
      accumulator_receipt = sft_utils.tree_numeric_receipt(accumulator_tree)
      accumulator_denominator = float(np.asarray(jax.device_get(
          actor_trainer.grad_accumulator.denom[...]
      )))
      first_precommit_record = {
          "schema": "canon-v1-first-update-precommit-v1",
          "update": 0,
          "workload": workload_identity,
          "dp": int(workload.dp_size),
          "tp": int(workload.tp_size),
          "microsteps": int(expected_microbatches),
          "accumulator_denominator": accumulator_denominator,
          "stable_norm_max": v1_first_update_gate.STABLE_NORM_MAX,
          **accumulator_receipt,
      }
      print(
          "[V1.FIRST_UPDATE] "
          + json.dumps(
              first_precommit_record,
              sort_keys=True,
              separators=(",", ":"),
          ),
          flush=True,
      )
      first_precommit_reasons = v1_first_update_gate.validate_precommit(
          first_precommit_record,
          workload=workload_identity,
          dp=int(workload.dp_size),
          tp=int(workload.tp_size),
          microsteps=int(expected_microbatches),
      )
      if first_precommit_reasons:
        raise alignment.AlignmentGateError(
            "V1 first-update precommit gradient gate failed before AdamW: "
            f"reasons={first_precommit_reasons} record={first_precommit_record}"
        )

    if p61_capture_dir:
      _p61_capture_tree(
          p61_capture_dir,
          "gradient",
          actor_trainer.grad_accumulator.get(),
      )
    optimizer_annotation = (
        xprof_train_schedule.optimizer_commit()
        if xprof_train_schedule is not None
        else gsm8k_xprof.trace_annotation(
            "optimizer_commit", update_step=self.rl_cluster.global_steps
        )
    )
    with optimizer_annotation:
      with self.rl_cluster._get_mesh_and_logical_axis_rules_cm(  # pylint: disable=protected-access
          rl_cluster_lib.Role.ACTOR
      ):
        commit_norm = actor_trainer.commit_precomputed_gradients()
      commit_norm.block_until_ready()
    commit_evidence = actor_trainer.consume_precomputed_commit_evidence()
    if p61_capture_dir:
      if (
          commit_evidence.get("effective_learning_rate") is None
          or commit_evidence["effective_learning_rate"] <= 0.0
          or commit_evidence.get("parameter_changed_elements", 0) <= 0
      ):
        raise alignment.AlignmentGateError(
            "P61 numerical capture requires a positive learning rate and a "
            f"material parameter update: {commit_evidence}"
        )
      _p61_capture_tree(
          p61_capture_dir,
          "model_after",
          nnx.state(actor_trainer.model, nnx.Param),
      )
    elapsed = time.perf_counter() - start
    hbm_after_commit = memory_snapshot()
    emit_sharding_inventory("after_commit")
    opt_timing = commit_evidence.get("optimizer_timing") or {}
    if perf_log.enabled() and opt_timing:
      print(
          "[PERF] stage=optimizer_transaction seconds=%.3f h2d=%.3f"
          " adam=%.3f d2h=%.3f"
          % (
              float(opt_timing.get("optimizer_transaction_seconds", 0.0)),
              float(opt_timing.get("optimizer_h2d_seconds", 0.0)),
              float(opt_timing.get("adam_commit_seconds", 0.0)),
              float(opt_timing.get("optimizer_d2h_seconds", 0.0)),
          ),
          flush=True,
      )
      self.rl_cluster.buffer_metrics_async(
          {
              f"perf/opt_{key}": (float(value), np.mean)
              for key, value in opt_timing.items()
              if isinstance(value, (int, float))
          },
          mode=rl_cluster_lib.Mode.TRAIN,
          step=int(self.rl_cluster.global_steps),
      )
    memory_profile_path = os.environ.get(
        "CANON_P30_MEMORY_PROFILE_PATH", ""
    )
    if memory_profile_path:
      if os.path.exists(memory_profile_path):
        raise alignment.AlignmentGateError(
            f"P30 memory profile path already exists: {memory_profile_path}"
        )

      def inventory(tree):
        arrays = [
            value for value in jax.tree.leaves(tree)
            if isinstance(value, jax.Array)
        ]
        by_kind = {}
        for value in arrays:
          kind = value.sharding.memory_kind
          by_kind[kind] = by_kind.get(kind, 0) + int(
              value.size * value.dtype.itemsize
          )
        return {
            "arrays": len(arrays),
            "logical_bytes": sum(by_kind.values()),
            "memory_kind_bytes": by_kind,
        }

      profile_inventory = {
          "model": inventory(nnx.state(actor_trainer.model, nnx.Param)),
          "optimizer": inventory(
              nnx.state(actor_trainer.optimizer, nnx.optimizer.OptState)
          ),
          "accumulator": inventory(
              nnx.state(actor_trainer.grad_accumulator)
          ),
          "hbm_after_commit": hbm_after_commit,
      }
      jax.profiler.save_device_memory_profile(memory_profile_path)
      print(
          "[P30.G2] MEMORY_PROFILE_SAVED "
          f"path={memory_profile_path} "
          f"inventory={json.dumps(profile_inventory, sort_keys=True)}",
          flush=True,
      )
    optimizer_memory_kinds_after = (
        actor_trainer.optimizer_state_memory_kinds()
    )
    if optimizer_placement != "device-unattested" and (
        optimizer_memory_kinds_after != (expected_optimizer_memory_kind,)
    ):
      raise alignment.AlignmentGateError(
          "optimizer state placement after commit is invalid: "
          f"{optimizer_memory_kinds_after!r}"
      )
    if optimizer_placement != "device-unattested":
      print(
          "[P41.OPTIMIZER] after_commit "
          f"placement={optimizer_placement} "
          f"memory_kind={expected_optimizer_memory_kind}",
          flush=True,
      )
    print(
        f"{marker_prefix} update_step_committed "
        f"train_steps={actor_trainer.train_steps}",
        flush=True,
    )

    after = {
        "model": fingerprint(nnx.state(actor_trainer.model, nnx.Param)),
        "optimizer": fingerprint(
            nnx.state(actor_trainer.optimizer, nnx.optimizer.OptState)
        ),
        "accumulator": fingerprint(
            nnx.state(actor_trainer.grad_accumulator), min_elements=1
        ),
        "reference": (
            fingerprint(_p28_reference_state(self.rl_cluster))
            if ref_state is not None else None
        ),
    }
    changed = {
        name: actor_trainer._canon_changed_paths(  # pylint: disable=protected-access
            before[name], after[name]
        )
        for name in ("model", "optimizer", "accumulator")
    }
    reference_changed = (
        actor_trainer._canon_changed_paths(  # pylint: disable=protected-access
            before["reference"], after["reference"]
        )
        if before["reference"] is not None else []
    )
    has_learning_signal = any(activity)
    effective_learning_rate = commit_evidence["effective_learning_rate"]
    schedule_required = workload_identity == "gsm8k"
    schedule_known = effective_learning_rate is not None
    zero_learning_rate = schedule_known and effective_learning_rate == 0.0
    parameter_changed_elements = commit_evidence[
        "parameter_changed_elements"
    ]
    parameter_fingerprint_consistent = (
        not changed["model"] or parameter_changed_elements > 0
    )
    zero_lr_model_unchanged = (
        not zero_learning_rate or parameter_changed_elements == 0
    )
    optimizer_transaction_valid = (
        (not schedule_required or schedule_known)
        and commit_evidence["gradient_finite"]
        and commit_evidence["parameter_delta_finite"]
        and parameter_fingerprint_consistent
        and zero_lr_model_unchanged
        and (bool(changed["optimizer"]) if has_learning_signal else True)
    )
    parameter_mutation = (
        "zero_lr_unchanged"
        if zero_learning_rate and parameter_changed_elements == 0
        else "observed_nonzero"
        if parameter_changed_elements > 0
        else "positive_lr_quantized_zero"
        if schedule_known and effective_learning_rate > 0.0
        else "unregistered_schedule_no_change"
    )
    update_record = {
        "contract_name": (
            workload.contract_name
            if p34_workload
            else workload.name
            if p33_workload
            else "legacy-segmented"
        ),
        "dp_size": workload.dp_size if canonical_workload else 1,
        "tp_size": workload.tp_size if canonical_workload else 1,
        "global_m": workload.global_m if canonical_workload else 256,
        "verdict": (
          "PASS"
            if optimizer_transaction_valid
            and not changed["accumulator"]
            and not reference_changed
            and actor_trainer.train_steps == before["train_steps"] + 1
            else "FAIL"
        ),
        "microsteps": expected_microbatches,
        "commits": 1,
        "train_steps_before": before["train_steps"],
        "train_steps_after": actor_trainer.train_steps,
        "gradient_activity": activity,
        "gradient_finite": all(
            np.isfinite(float(np.asarray(value))) for value in micro_norms
        ),
        "dp_replicas_exact": result["replica_equality"],
        "dp_axis": result["dp_axis"],
        "dp_reduction_transactions": result["dp_reduction_transactions"],
        "dp_reduction_rounds_per_transaction": result[
            "dp_reduction_rounds_per_transaction"
        ],
        "dp_rank_pullbacks_per_transaction": result[
            "dp_rank_pullbacks_per_transaction"
        ],
        "dp_pullback_invocations_per_transaction": result.get(
            "dp_pullback_invocations_per_transaction",
            result["dp_rank_pullbacks_per_transaction"],
        ),
        "has_learning_signal": has_learning_signal,
        "micro_gradient_norms": [float(np.asarray(x)) for x in micro_norms],
        "commit_gradient_norm": float(np.asarray(commit_norm)),
        "optimizer_transaction_valid": optimizer_transaction_valid,
        "parameter_mutation": parameter_mutation,
        "commit_evidence": commit_evidence,
        "model": {
            "changed_count": len(changed["model"]),
            "changed_paths": changed["model"],
        },
        "optimizer": {
            "changed_count": len(changed["optimizer"]),
            "changed_paths": changed["optimizer"],
        },
        "accumulator_changed_paths": changed["accumulator"],
        "reference_present": ref_state is not None,
        "reference_changed_paths": reference_changed,
        "checkpoint_enabled": actor_trainer.config.checkpoint_root_directory is not None,
        "alignment_hashes": [record["hashes"] for record in records],
        "elapsed_seconds": elapsed,
        "hbm_before": hbm_before,
        "hbm_after_accumulation": hbm_after_accumulation,
        "hbm_after_commit": hbm_after_commit,
        "optimizer_memory_kinds_before": list(
            optimizer_memory_kinds_before
        ),
        "optimizer_memory_kinds_after": list(
            optimizer_memory_kinds_after
        ),
        "optimizer_placement": optimizer_placement,
        "state_fingerprints_before": before,
        "state_fingerprints_after": after,
    }
    update_path = os.environ.get("CANON_UPDATE_REPORT", "")
    if not update_path:
      raise alignment.AlignmentGateError("CANON_UPDATE_REPORT is required")
    os.makedirs(os.path.dirname(update_path) or ".", exist_ok=True)
    update_mode = "a" if full_train else "w"
    with open(update_path, update_mode, encoding="utf-8") as update_file:
      if full_train:
        update_file.write(json.dumps(update_record, sort_keys=True) + "\n")
      else:
        json.dump(update_record, update_file, indent=2, sort_keys=True)
        update_file.write("\n")
    if full_train:
      max_differing_bytes = max(
          boundary["differing_bytes"]
          for record in records
          for boundary in record["boundaries"].values()
      )
      canonical_update_metrics = {
          "canonical/segmented_loss": (
              float(np.asarray(result["loss"])), np.mean
          ),
          "canonical/commit_gradient_norm": (
              float(np.asarray(commit_norm)), np.mean
          ),
          "canonical/alignment_max_differing_bytes": (
              max_differing_bytes, np.max
          ),
          "canonical/active_microbatches": (sum(activity), np.mean),
          "canonical/effective_learning_rate": (
              float(effective_learning_rate or 0.0), np.mean
          ),
          "canonical/parameter_changed_elements": (
              parameter_changed_elements, np.mean
          ),
          "canonical/max_abs_parameter_delta": (
              commit_evidence["parameter_delta_max_abs"], np.max
          ),
      }
      p58_clip_evidence = (
          commit_evidence.get("overflow_safe_clip")
          if p34_workload
          else None
      )
      if p58_clip_evidence is not None:
        canonical_update_metrics.update({
            "canonical/gradient_stable_norm": (
                float(p58_clip_evidence["stable_norm"]), np.mean
            ),
            "canonical/gradient_naive_norm_finite": (
                int(p58_clip_evidence["naive_norm_finite"]), np.mean
            ),
            "canonical/overflow_safe_clip_fallback": (
                int(p58_clip_evidence["fallback_used"]), np.sum
            ),
            "canonical/gradient_clip_factor": (
                float(p58_clip_evidence["clip_factor"]), np.mean
            ),
        })
      self.rl_cluster.buffer_metrics_async(
          canonical_update_metrics,
          mode=rl_cluster_lib.Mode.TRAIN,
          step=before["train_steps"],
      )
    print(
        f"{marker_prefix} post_update_snapshot "
        f"verdict={update_record['verdict']} "
        f"model_changed={len(changed['model'])} "
        f"optimizer_changed={len(changed['optimizer'])} "
        f"parameter_changed_elements={parameter_changed_elements} "
        f"effective_lr={effective_learning_rate} "
        f"reference_changed={len(reference_changed)}",
        flush=True,
    )
    if update_record["verdict"] != "PASS":
      raise alignment.AlignmentGateError(
          f"P28 G6 update transaction red: {update_record}"
      )
    if first_update_admission:
      first_commit_record = {
          "schema": "canon-v1-first-update-commit-v1",
          "update": 0,
          "workload": workload_identity,
          "dp": int(workload.dp_size),
          "tp": int(workload.tp_size),
          "train_steps_before": int(before["train_steps"]),
          "train_steps_after": int(actor_trainer.train_steps),
          "optimizer_transaction_valid": bool(
              optimizer_transaction_valid
          ),
          "gradient_finite": bool(commit_evidence["gradient_finite"]),
          "parameter_delta_finite": bool(
              commit_evidence["parameter_delta_finite"]
          ),
          "parameter_changed_elements": int(parameter_changed_elements),
          "effective_learning_rate": effective_learning_rate,
          "outer_weight_sync_pending": True,
      }
      first_commit_reasons = v1_first_update_gate.validate_commit(
          first_commit_record,
          workload=workload_identity,
          dp=int(workload.dp_size),
          tp=int(workload.tp_size),
      )
      print(
          "[V1.FIRST_UPDATE] "
          + json.dumps(
              first_commit_record,
              sort_keys=True,
              separators=(",", ":"),
          ),
          flush=True,
      )
      if first_commit_reasons:
        raise alignment.AlignmentGateError(
            "V1 first-update optimizer admission failed before outer weight "
            f"sync/checkpoint: reasons={first_commit_reasons} "
            f"record={first_commit_record}"
        )
    return update_record

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      algo_config: TConfig,
      reward_fns: RewardFn | List[RewardFn] | None = None,
      chat_parser: Any | None = None,
      metric_fns: Sequence[MetricFn] | None = None,
      agent_class: Type[
          base_agent.ConversationAgentBase
      ] = model_agent.ModelAgent,
      agent_kwargs: Dict[str, Any] | None = None,
      env_class: Type[
          base_environment.BaseTaskEnv
      ] = task_environment.TaskEnvironment,
      env_kwargs: Dict[str, Any] | None = None,
  ):
    """Initializes the `AgenticRLLearner`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      algo_config: Configuration object.
      reward_fns: Reward functions.
      chat_parser: A parser to handle chat message formatting.
      metric_fns: A sequence of callables that compute metrics for the
        completions. Each callable should accept ``prompts``, ``completions``,
        ``rewards``, ``advantages`` and optional keyword arguments, and return
        a dictionary of metric names to tuples of
        ``(metric_value, aggregation_fn)``:

           >>> def metric_fn(
           ...     prompts, completions, rewards, advantages, **kargs
           ... ):
           ...     return {
           ...       # ...
           ...       "prompt_min_len": (min(len(p) for p in prompts), np.min),
           ...       # ... }
      agent_class: User defined agent class.
      agent_kwargs: Keyword arguments for the agent class.
      env_class: User defined environment class.
      env_kwargs: Keyword arguments for the environment class.
    """
    self.rl_cluster = rl_cluster
    self.algo_config = algo_config
    self._validate_rollout_config()
    reward_manager_fn = function_registry.get_reward_manager(
        algo_config.reward_manager
    )
    self.reward_manager = reward_manager_fn(
        reward_fns=reward_fns,
        algo_config=algo_config,
    )
    self.metric_fns = metric_fns or []
    self.rl_cluster.actor_trainer.is_managed_externally = True
    if hasattr(self.rl_cluster, "critic_trainer"):
      self.rl_cluster.critic_trainer.is_managed_externally = True

    self.agent_class = agent_class
    self.agent_kwargs = agent_kwargs or {}
    self.env_class = env_class
    self.env_kwargs = env_kwargs or {}

    self._training_config = self.rl_cluster.cluster_config.training_config

    self.rl_cluster.global_steps = (
        self.rl_cluster.actor_trainer.restored_global_step()
    )
    # Current iter steps for micro-batch based training.
    self._iter_steps = self.rl_cluster.actor_trainer.iter_steps
    self._eval_iter_steps = 0
    # Tracks the last train_step value at which evaluation was run. The
    # optimizer is wrapped in ``optax.MultiSteps(grad_accum_steps)``, which
    # keeps ``actor_trainer.train_steps`` constant for ``grad_accum_steps``
    # consecutive micro-iterations. Without this guard, the
    # ``train_steps % eval_every_n_steps == 0`` check would fire at every
    # micro-iteration during an eval boundary, causing the full evaluation
    # rollout to be replayed ``grad_accum_steps`` times for the same step.
    self._last_eval_train_step = -1

    # Sync weights if the actor model and rollout model are not sharing weights.
    self.should_sync_weights = not (
        rl_utils.is_sharing_weights(
            self.rl_cluster.actor_trainer.model,
            self.rl_cluster.rollout.model(),
        )
    )

    # Enable async rollout if trainer and rollout are not on the same mesh.
    # If they do, then doesn't make sense for the interleave because they will
    # have resource contention.
    self.can_enable_async_rollout = (
        self.rl_cluster.cluster_config.role_to_mesh[rl_cluster_lib.Role.ACTOR]
        != self.rl_cluster.cluster_config.role_to_mesh[
            rl_cluster_lib.Role.ROLLOUT
        ]
    )

    self._rollout_micro_batch_size = (
        self._training_config.rollout_micro_batch_size
    )
    self._compute_logps_micro_batch_size = (
        self._training_config.compute_logps_micro_batch_size or 1
    )
    sft_utils.show_hbm_usage(title="AgenticRLLearner init")

    self.chat_parser = chat_parser
    self.tokenizer = rl_cluster.tokenizer
    self.policy_version = self.rl_cluster.global_steps
    self._rollout_sync_lock = agentic_utils.RolloutSyncLock()
    self._background_tasks: Set[asyncio.Task] = set()
    self._full_batch_size = 0
    self._process_in_consumer: bool = False

    loop_queue = queue.Queue()

    def run_loop_forever():
      loop = agentic_utils.get_or_create_loop()
      loop.set_default_executor(
          ThreadPoolExecutor(max_workers=algo_config.max_concurrency + 1)
      )
      loop_queue.put(loop)
      loop.run_forever()

    loop_thread = threading.Thread(target=run_loop_forever, daemon=True)
    loop_thread.start()
    self.loop = loop_queue.get()
    self._global_step_start_time = time.time()

    # Per-step reward accumulators populated inside ``_compute_rewards``.
    # Drained at the global-step boundary to emit a one-line per-step
    # summary that mirrors what an external metric logger would show.
    # Each bin keeps at most ``full_batch_size``-worth of recent values
    # so a producer that races one batch ahead of the consumer does not
    # double-count.
    self._train_rewards_window: List[float] = []
    self._eval_rewards_window: List[float] = []
    self._rewards_window_lock = threading.Lock()

  def _validate_rollout_config(self):
    """Validates that the rollout config is properly aligned with the algo config."""
    rollout_config = self.rl_cluster.cluster_config.rollout_config
    if not isinstance(rollout_config, dict):
      configs_to_check = {"train": rollout_config}
    else:
      configs_to_check = rollout_config

    for mode, config in configs_to_check.items():
      if config.max_tokens_to_generate != self.algo_config.max_response_length:
        raise ValueError(
            f"RolloutConfig ({mode}) max_tokens_to_generate "
            f"({config.max_tokens_to_generate}) must match AgenticRLConfig "
            f"max_response_length ({self.algo_config.max_response_length}). "
            "Please align these configurations before initializing RLCluster."
        )
      if self.algo_config.use_rollout_logps and not config.return_logprobs:
        raise ValueError(
            f"RolloutConfig ({mode}) must have return_logprobs=True for "
            "AgenticRLLearner when use_rollout_logps=True. Please set this "
            "before initializing RLCluster."
        )
      if (
          self.rl_cluster.cluster_config.rollout_engine == "vllm"
          and not config.rollout_vllm_server_mode
      ):
        raise ValueError(
            f"RolloutConfig ({mode}) must have rollout_vllm_server_mode set to "
            "True for AgenticRLLearner if using vLLM engine. Please set this "
            "before initializing RLCluster."
        )

  def _compute_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      mode: rl_cluster_lib.Mode,
      expected_step: int | None = None,
      **kwargs,
  ) -> np.ndarray:
    """Computes the rewards for completions using the provided reward functions.

    Args:
      prompts: A list of input prompts.
      completions: A list of generated text completions.
      mode: The mode to use for logging metrics.
      expected_step: The expected training step.
      **kwargs: Additional keyword arguments passed to the reward functions.

    Returns:
      A JAX array (shape `[num_prompts]`) of scalar rewards for each
      prompt-completion pair. The rewards are the sum across all the provided
      reward functions.

    Raises:
        RuntimeError: If 'r' reward is None, indicating a failure to obtain the
        result, or if the length of 'r' reward does not match the length of
        'prompts'.
    """
    if "mode" in kwargs:
      raise ValueError(f"kwargs already contains mode as a key: {kwargs}")
    kwargs["mode"] = str(mode)

    rewards_info = self.reward_manager(
        prompts=prompts,
        completions=completions,
        **kwargs,
    )

    # Pass the expected_step explicitly because it is calculated based on
    # the batch index (predicted step) to align metrics with the correct
    # training step in the asynchronous execution.
    expected_step = 0 if expected_step is None else expected_step
    self.rl_cluster.buffer_metrics_async(
        rewards_info["log_metrics"], mode=mode, step=expected_step
    )

    rewards_array = np.asarray(rewards_info["rewards"])
    with self._rewards_window_lock:
      target = (
          self._train_rewards_window
          if mode == rl_cluster_lib.Mode.TRAIN
          else self._eval_rewards_window
      )
      target.extend(rewards_array.tolist())
      # Cap train window at full_batch_size * num_generations (one full step's
      # worth of per-sequence rewards) to bound the producer-vs-consumer
      # race: the producer can race up to ``off_policy_steps + 1`` batches
      # ahead, so without a cap the window would over-count next-step rewards
      # at the current step's boundary.
      if mode == rl_cluster_lib.Mode.TRAIN and self._full_batch_size > 0:
        cap = self._full_batch_size * self.algo_config.num_generations
        excess = len(target) - cap
        if excess > 0:
          del target[:excess]

    return rewards_info["rewards"]

  def _create_micro_batch_iterator(
      self,
      full_batch_iterator: Iterator[TrainingInputT],
      micro_batch_size: int,
  ) -> Iterator[TrainingInputT]:
    """Re-batches large inputs into an iterator of micro-batches.

    Args:
      full_batch_iterator: Iterator yielding large `TrainingInputT` batches.
      micro_batch_size: The desired size of the micro-batches.

    Yields:
      `TrainingInputT` dicts, each with `micro_batch_size` samples.
    """
    buffer = {}

    def get_buffer_len(buf: dict[str, list[Any]]) -> int:
      if not buf:
        return 0
      return len(next(iter(buf.values())))

    for large_batch in full_batch_iterator:
      for key, values in large_batch.items():
        if key not in buffer:
          buffer[key] = []

        if isinstance(values, (np.ndarray, jax.Array)):
          buffer[key].extend(list(values.flatten()))
        elif isinstance(values, (list, tuple)):
          buffer[key].extend(values)
        else:
          buffer[key].append(values)

      while get_buffer_len(buffer) >= micro_batch_size:
        micro_batch = {}
        for key in buffer:
          micro_batch_list_slice = buffer[key][:micro_batch_size]
          micro_batch[key] = np.array(micro_batch_list_slice)
          buffer[key] = buffer[key][micro_batch_size:]

        yield micro_batch

  def _create_agent_env_pair(
      self, single_example: TrainingInputT, group_id: int, pair_index: int
  ) -> tuple[base_agent.ConversationAgentBase, base_environment.BaseTaskEnv]:
    """Constructs an (agent, environment) pair for a single input sample.

    This is used to set up a rollout for one generation within a group.

    Args:
      single_example: A training input containing a single prompt.
      group_id: An identifier for group generations from the same original
        prompt.
      pair_index: The index of the pair within the group.

    Returns:
      A tuple of agent and environment.
    """

    agent = self.agent_class(
        **{"system_prompt": self.algo_config.system_prompt, **self.agent_kwargs}
    )  # if agent_kwargs contains "system_prompt", it will be honored.

    assert "group_id" not in self.env_kwargs
    assert "pair_index" not in self.env_kwargs
    env = self.env_class(
        single_example,
        **{"group_id": group_id, "pair_index": pair_index, **self.env_kwargs},  # pyrefly: ignore[bad-argument-type]
    )
    # Seed provenance before ``env.reset()``. Reset can fail before the first
    # model call (for example while a Kubernetes sandbox is being admitted),
    # but the resulting filtered trajectory still belongs to this policy
    # version and must remain journalable.
    env.task["policy_version"] = self.policy_version

    return agent, env

  def _model_call(
      self,
      chat_lists: List[Dict[str, str]],
      env: Any = None,
      max_generation_steps: int | None = None,
      request_timeout_s: float | None = None,
  ) -> base_rollout.RolloutOutput:
    """Calls model generation."""
    if env:
      env.task["policy_version"] = self.policy_version

    if self.chat_parser:
      chat_lists = self.chat_parser.parse(
          messages=chat_lists,
          add_generation_prompt=True,
          is_first_msg=True,  # no op if system msg is populated in reset
      )
    tags = {}
    if env and hasattr(env, "extra_kwargs"):
      if "group_id" in env.extra_kwargs:
        tags[perf_constants.GROUP_ID] = env.extra_kwargs["group_id"]
        if self._full_batch_size > 0:
          tags[perf_constants.STEP] = (
              env.extra_kwargs["group_id"] // self._full_batch_size
          )
      if "pair_index" in env.extra_kwargs:
        tags[perf_constants.PAIR_INDEX] = env.extra_kwargs["pair_index"]

    prompts = [chat_lists]
    result = self.rl_cluster.generate(
        prompts=prompts,
        apply_chat_template=False if self.chat_parser else True,
        mode=rl_cluster_lib.Mode.TRAIN,
        trace_tags=tags,
        max_generation_steps=max_generation_steps,
        request_timeout_s=request_timeout_s,
    )

    return result

  def _build_orchestrator(self) -> rollout_orchestrator.RolloutOrchestrator:
    """Builds and configures a RolloutOrchestrator for parallel rollouts."""
    engine_kwargs = dict(
        model_call=self._model_call,
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
        timeout=self.algo_config.episode_timeout,
        per_turn_timeout=self.algo_config.per_turn_timeout,
        cleanup_timeout=self.algo_config.cleanup_timeout,
        max_response_length=self.algo_config.max_response_length,
        overlong_filter=self.algo_config.overlong_filter,
        filter_statuses=self.algo_config.filter_statuses,
        perf_v2=self.rl_cluster.perf_v2,
    )
    return rollout_orchestrator.RolloutOrchestrator(
        engine_cls=trajectory_collect_engine.TrajectoryCollectEngine,
        engine_kwargs=engine_kwargs,
        max_concurrency=self.algo_config.max_concurrency,
        rollout_sync_lock=self._rollout_sync_lock,
    )

  async def _orchestrator_producer(
      self,
      orchestrator: rollout_orchestrator.RolloutOrchestrator,
      prompt_iterator: Iterable[TrainingInputT] | AsyncIterator[TrainingInputT],
      num_generations: int = 1,
      collect_mode: str = "Token",
  ):
    """Generates trajectory groups using the orchestrator pattern.

    Args:
      orchestrator: The RolloutOrchestrator instance to use.
      prompt_iterator: An iterable yielding single `TrainingInputT` examples.
      num_generations: The number of episodes to run per agent-environment pair.
      collect_mode: The mode for trajectory collection (e.g., "Token").

    Yields:
      A list of trajectories for a group.
    """
    is_async_iterator = hasattr(prompt_iterator, "__aiter__")

    async def pairs_stream_generator():
      """Yield (agent, env) pairs with unique group_id per original prompt."""
      # TODO (tsbao): fix the group id when we can resume from mid global step
      # with mini-batch.
      group_id = self.rl_cluster.global_steps * self._full_batch_size
      if is_async_iterator:
        async for single_example in prompt_iterator:  # pyrefly: ignore[not-iterable]
          # Create agent-env pairs in parallel for a group to handle potential
          # cold start latency on env creation.
          agent_env_pairs = await asyncio.gather(*[
              self.loop.run_in_executor(
                  None,
                  self._create_agent_env_pair,
                  copy.deepcopy(single_example),
                  group_id,
                  pair_index,
              )
              for pair_index in range(num_generations)
          ])
          for agent, env in agent_env_pairs:
            yield agent, env
          group_id += 1
      else:
        for single_example in prompt_iterator:  # pyrefly: ignore[not-iterable]
          agent_env_pairs = await asyncio.gather(*[
              self.loop.run_in_executor(
                  None,
                  self._create_agent_env_pair,
                  copy.deepcopy(single_example),
                  group_id,
                  pair_index,
              )
              for pair_index in range(num_generations)
          ])
          for agent, env in agent_env_pairs:
            yield agent, env
          group_id += 1

    # Start producers in the background.
    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pairs_stream_generator(),
            group_size=self.algo_config.num_generations,
            group_key_fn=lambda i, env, traj: env.extra_kwargs["group_id"],
            collect_mode=collect_mode,
        )
    )

    # Let the producer start and initialize its manager before consuming.
    await asyncio.sleep(0)

    # Consume full groups and yield them with their original input.
    async_generator = orchestrator.yield_batches(
        batch_size=self.algo_config.num_generations
    )
    prompt_groups_in_batch = 0
    batch_started = self.loop.time()
    try:
      async with contextlib.aclosing(async_generator) as stream:
        while True:
          timeout = self.algo_config.rollout_batch_timeout
          if timeout is None:
            remaining = None
          else:
            remaining = timeout - (self.loop.time() - batch_started)
            if remaining <= 0:
              raise TimeoutError(
                  "rollout batch exceeded hard timeout before completion: "
                  f"timeout={timeout:.1f}s "
                  f"completed_prompt_groups={prompt_groups_in_batch}/"
                  f"{self._full_batch_size}"
              )
          try:
            if remaining is None:
              group = await anext(stream)
            else:
              group = await asyncio.wait_for(anext(stream), timeout=remaining)
          except StopAsyncIteration:
            break
          except asyncio.TimeoutError as exc:
            raise TimeoutError(
                "rollout batch exceeded hard timeout while waiting for "
                f"trajectories: timeout={timeout:.1f}s "
                f"completed_prompt_groups={prompt_groups_in_batch}/"
                f"{self._full_batch_size}"
            ) from exc
          if group:
            # Retrieve the original input embedded in the task.
            yield group
            prompt_groups_in_batch += 1
            if prompt_groups_in_batch == self._full_batch_size:
              logging.info(
                  "[DEEPSWE.ROLLOUT_DEADLINE] batch_complete "
                  "prompt_groups=%d elapsed_secs=%.1f deadline_secs=%s",
                  prompt_groups_in_batch,
                  self.loop.time() - batch_started,
                  timeout,
              )
              prompt_groups_in_batch = 0
              batch_started = self.loop.time()
    except (GeneratorExit, asyncio.CancelledError):
      # This is the normal shutdown path for a generator.
      return
    finally:
      # Ensure the background producer task is cancelled and cleaned up.
      if not producer_task.done():
        producer_task.cancel()

        async def await_cancellation():
          with contextlib.suppress(asyncio.CancelledError):
            await producer_task

        cancellation_task = asyncio.create_task(await_cancellation())
        self._background_tasks.add(cancellation_task)
        cancellation_task.add_done_callback(self._background_tasks.discard)

  def _batch_to_train_example(
      self,
      batch_results: list[Any],
      mode: rl_cluster_lib.Mode,
  ) -> List[TrainExample]:
    """Converts a group of trajectories into a list of `TrainExample`s.

    Args:
      batch_results: A list of trajectories from the same generation group.
      mode: The current mode (TRAIN or EVAL).

    Returns:
      A list of `TrainExample` instances, ready for training.
    """
    # Create a merged training_input where each field from the original input
    # is repeated G times to align with the G completions.
    if mode == rl_cluster_lib.Mode.TRAIN:
      expected_step = batch_results[0].group_id // self._full_batch_size
    else:
      expected_step = self.rl_cluster.global_steps

    return self._process_results(
        trajectories=batch_results,
        mode=mode,
        expected_step=expected_step,
    )

  @abc.abstractmethod
  def _process_results(
      self,
      trajectories: List[Any],
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
      expected_step: int | None = None,
  ) -> List[TrainExample]:
    """Processes generation results, computes rewards and advantages."""
    pass

  def _generate_and_compute_advantage(
      self,
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> TrainExample:
    """Unused in AgenticRLLearner."""
    raise NotImplementedError(
        "_generate_and_compute_advantage is not used in AgenticRLLearner"
    )

  def _num_iterations(self) -> int:
    """Returns the number of iterations per batch."""
    return self.algo_config.num_iterations

  def _num_generations(self) -> int:
    """Returns the number of generations per prompt."""
    return self.algo_config.num_generations

  async def _producer(
      self,
      orchestrator,
      prompt_queue: queue.Queue[TrainingInputT | None],
      train_data_queue,
  ):
    """Produces training examples from prompts in the dataset_iterator."""
    loop = asyncio.get_running_loop()
    async_queue_iter = self._AsyncQueueIterator(prompt_queue, loop)

    async def _iterate_micro_batches():
      async for item in async_queue_iter:
        for prompt in self._create_micro_batch_iterator(iter([item]), 1):
          yield prompt

    prompt_iterator = _iterate_micro_batches()
    try:
      async for batch in self._orchestrator_producer(
          orchestrator=orchestrator,
          prompt_iterator=prompt_iterator,
          num_generations=self.algo_config.num_generations,
          collect_mode="Token",
      ):
        try:
          if self._process_in_consumer:
            # Put raw batch (list of trajectories) into queue.
            # We put it once, and consumer will handle iterations.
            train_data_queue.put(batch)
          else:
            train_examples = self._batch_to_train_example(
                batch_results=batch,
                mode=rl_cluster_lib.Mode.TRAIN,
            )
            for train_example in train_examples:
              train_data_queue.put(train_example)
        except Exception as e:
          if not isinstance(e, RuntimeError):
            logging.exception(
                "Exception in _producer while processing batch: %s", e
            )
          raise
    finally:
      # Signal production is complete for this batch, even if errors occurred.
      train_data_queue.put(None)
      # Ensure that any background threads waiting on the prompt queue are
      # unblocked.
      prompt_queue.put(None)

  def _data_consumer_batch_generator(
      self,
      queue: queue_lib.AbstractDataQueue,
      batch_size: int,
      *,
      require_full_batch: bool = False,
  ):
    """Yields micro-batches from a queue until a None is received."""
    item_iterator = iter(lambda: queue.get(block=True), None)
    while True:
      batch = list(itertools.islice(item_iterator, batch_size))
      if not batch:
        return  # The iterator is exhausted.
      if require_full_batch and len(batch) != batch_size:
        raise RuntimeError(
            "P38 diagnostic full-coverage consumer received a partial "
            f"prompt batch: got={len(batch)} expected={batch_size}; "
            "refusing subset alignment"
        )
      yield batch

  def evaluate_only(
      self,
      eval_dataset: Iterable[TrainingInputT],
      *,
      policy_step: int,
  ) -> dict[str, Any]:
    """Runs one isolated held-out rollout without training or checkpoint writes."""
    all_eval_prompts = list(
        self._create_micro_batch_iterator(iter(eval_dataset), 1)
    )
    if not all_eval_prompts:
      raise ValueError("P57 isolated evaluation dataset is empty")
    if policy_step < 0:
      raise ValueError("P57 isolated evaluation policy step must be nonnegative")
    # ``train`` normally initializes this from its first dataset batch.  The
    # isolated path never enters ``train``, so bind the full evaluation
    # inventory explicitly before group ids and watchdog accounting use it.
    self._full_batch_size = len(all_eval_prompts)
    with self._rewards_window_lock:
      self._eval_rewards_window.clear()
    self._eval_iter_steps = 0
    eval_orchestrator = self._build_orchestrator()
    started = time.perf_counter()

    async def _run():
      batches = 0
      async for batch in self._orchestrator_producer(
          eval_orchestrator,
          all_eval_prompts,
          num_generations=self._num_generations(),
      ):
        # Converting through the ordinary EVAL path computes the authoritative
        # environment rewards and metrics without constructing an optimizer
        # update. Do not retain the full trajectories after each batch.
        self._batch_to_train_example(batch, rl_cluster_lib.Mode.EVAL)
        batches += 1
      return batches

    batches = asyncio.run_coroutine_threadsafe(_run(), self.loop).result()
    with self._rewards_window_lock:
      rewards = np.asarray(self._eval_rewards_window, dtype=np.float32)
      self._eval_rewards_window.clear()
    expected_rewards = len(all_eval_prompts) * self._num_generations()
    if rewards.size != expected_rewards:
      raise RuntimeError(
          "P57 isolated evaluation coverage mismatch: "
          f"prompts={len(all_eval_prompts)} "
          f"generations={self._num_generations()} rewards={rewards.size} "
          f"expected={expected_rewards}"
      )
    metrics = _frozenlake_evaluation_metrics(
        rewards,
        wall_seconds=time.perf_counter() - started,
        policy_step=policy_step,
    )
    return {
        **metrics,
        "prompts": len(all_eval_prompts),
        "generations": self._num_generations(),
        "batches": batches,
      "rewards": rewards.tolist(),
    }

  def rollout_only_evaluate(
      self,
      eval_dataset: Iterable[TrainingInputT],
      *,
      policy_step: int,
  ) -> dict[str, Any]:
    """Collects raw held-out trajectories without trainer-side recomputation.

    This is the P57 workload-calibration path.  It intentionally does not call
    ``_batch_to_train_example``: that ordinary EVAL conversion recomputes
    trainer log-probabilities and pads full training tensors, neither of which
    is needed to choose a FrozenLake workload.  The returned receipt contains
    only scalar trajectory facts and identifiers; no token contents are
    retained.
    """
    all_eval_prompts = list(
        self._create_micro_batch_iterator(iter(eval_dataset), 1)
    )
    if not all_eval_prompts:
      raise ValueError("P57 rollout-only evaluation dataset is empty")
    if policy_step < 0:
      raise ValueError("P57 rollout-only policy step must be nonnegative")
    self._full_batch_size = len(all_eval_prompts)
    before_train_steps = self.rl_cluster.actor_trainer.train_steps
    before_global_steps = self.rl_cluster.global_steps
    first_group_id = before_global_steps * len(all_eval_prompts)

    def _prompt_scalar(prompt_index: int, key: str):
      try:
        value = all_eval_prompts[prompt_index][key]
      except (IndexError, KeyError) as exc:
        raise RuntimeError(
            f"P57 rollout-only prompt provenance is missing {key!r}"
        ) from exc
      array = np.asarray(value)
      if array.size != 1:
        raise RuntimeError(
            "P57 rollout-only prompt provenance must be scalar: "
            f"index={prompt_index} key={key} shape={array.shape}"
        )
      return array.reshape(-1)[0].item()

    orchestrator = self._build_orchestrator()
    started = time.perf_counter()

    async def _run():
      batches = 0
      records = []
      async for batch in self._orchestrator_producer(
          orchestrator,
          all_eval_prompts,
          num_generations=self._num_generations(),
      ):
        if len(batch) != self._num_generations():
          raise RuntimeError(
              "P57 rollout-only group coverage mismatch: "
              f"got={len(batch)} expected={self._num_generations()}"
          )
        for item in batch:
          traj = item.traj
          prompt_tokens = np.asarray(traj.get("prompt_tokens", []))
          completion_tokens = np.asarray(
              traj.get("conversation_tokens", [])
          )
          completion_masks = np.asarray(
              traj.get("conversation_masks", [])
          )
          if completion_tokens.size != completion_masks.size:
            raise RuntimeError(
                "P57 rollout-only token/mask lengths diverged: "
                f"tokens={completion_tokens.size} masks={completion_masks.size}"
            )
          prompt_length = traj.get("prompt_length")
          prompt_length = (
              int(prompt_tokens.size)
              if prompt_length is None
              else int(prompt_length)
          )
          conversation = traj.get("conversation_text") or []
          turns = sum(
              1
              for message in conversation
              if isinstance(message, dict)
              and message.get("role") == "assistant"
          )
          group_id = int(item.group_id)
          prompt_index = group_id - first_group_id
          if prompt_index not in range(len(all_eval_prompts)):
            raise RuntimeError(
                "P57 rollout-only group id cannot join its prompt: "
                f"group_id={group_id} first={first_group_id} "
                f"prompts={len(all_eval_prompts)}"
            )
          reward_value = float(traj.get("trajectory_reward", 0.0))
          if not np.isfinite(reward_value):
            raise RuntimeError("P57 rollout-only trajectory reward is not finite")
          records.append({
              "group_id": group_id,
              "pair_index": int(item.pair_index),
              "policy_version": int(traj.get("policy_version", policy_step)),
              "status": str(traj.get("status", "")),
              "reward": reward_value,
              "invalid_actions": int(traj.get("invalid_action_count", 0)),
              "ineffective_actions": int(
                  traj.get("ineffective_action_count", 0)
              ),
              "turns": turns,
              "prompt_tokens": prompt_length,
              "completion_tokens": int(completion_tokens.size),
              "assistant_tokens": int(np.count_nonzero(completion_masks)),
              "context_tokens": prompt_length + int(completion_tokens.size),
              # The agent trajectory's first observation is presentation text,
              # not the original Grain row. Join through the orchestrator's
              # exact group-id construction instead of pretending that
              # ``trajectory.task`` retained dataset metadata.
              "p57_index": int(_prompt_scalar(prompt_index, "p57_index")),
              "grid_side": int(_prompt_scalar(prompt_index, "size")),
              "shortest_path": int(
                  _prompt_scalar(prompt_index, "shortest_path")
              ),
              "map_sha256": str(
                  _prompt_scalar(prompt_index, "map_sha256")
              ),
          })
        batches += 1
      return batches, records

    batches, records = asyncio.run_coroutine_threadsafe(
        _run(), self.loop
    ).result()
    expected_records = len(all_eval_prompts) * self._num_generations()
    if len(records) != expected_records:
      raise RuntimeError(
          "P57 rollout-only coverage mismatch: "
          f"prompts={len(all_eval_prompts)} "
          f"generations={self._num_generations()} records={len(records)} "
          f"expected={expected_records}"
      )
    if (
        self.rl_cluster.actor_trainer.train_steps != before_train_steps
        or self.rl_cluster.global_steps != before_global_steps
    ):
      raise RuntimeError("P57 rollout-only evaluation mutated training steps")
    return {
        "policy_step": policy_step,
        "prompts": len(all_eval_prompts),
        "generations": self._num_generations(),
        "batches": batches,
        "trajectories": len(records),
        "wall_seconds": time.perf_counter() - started,
        "train_steps_before": before_train_steps,
        "train_steps_after": self.rl_cluster.actor_trainer.train_steps,
        "records": records,
    }

  def train(
      self,
      train_dataset: Iterable[TrainingInputT],
      eval_dataset: Iterable[TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """Main training loop for the AgenticRLLearner."""
    full_batch_iterator = iter(train_dataset)

    if self.rl_cluster.global_steps > 0:
      logging.info(
          "Skipping %d batches from train_dataset to fast-forward to step %d",
          self.rl_cluster.global_steps,
          self.rl_cluster.global_steps,
      )
      # TODO(b/483779605): Current implementation of fast-forwarding does not
      # take into account the mini-batch size. Follow-up CL will address this.
      for _ in range(self.rl_cluster.global_steps):
        try:
          next(full_batch_iterator)
        except StopIteration:
          logging.warning("Train dataset exhausted while skipping batches.")
          self.rl_cluster.close()
          return

    try:
      first_item = next(full_batch_iterator)
    except StopIteration:
      logging.warning("Training dataset is empty.")
      self.rl_cluster.close()
      return

    full_batch_size = len(next(iter(first_item.values())))  # pyrefly: ignore[bad-argument-type]
    self._full_batch_size = full_batch_size
    # Initialize batch sizes.
    mini_batch_size = self._training_config.mini_batch_size or full_batch_size
    train_micro_batch_size = (
        self._training_config.train_micro_batch_size or mini_batch_size
    )
    # Rollout micro batch size has to be 1 since we only process individual
    # prompts.
    self._rollout_micro_batch_size = 1
    self._process_in_consumer = False

    if self._compute_logps_micro_batch_size > 1:
      if self._compute_logps_micro_batch_size != train_micro_batch_size:
        raise ValueError(
            "compute_logps_micro_batch_size"
            f" ({self._compute_logps_micro_batch_size}) must be equal to"
            f" train_micro_batch_size ({train_micro_batch_size})"
        )
      self._process_in_consumer = True

    for v, n in [
        (self._rollout_micro_batch_size, f"{self._rollout_micro_batch_size=}"),
        (
            self._compute_logps_micro_batch_size,
            f"{self._compute_logps_micro_batch_size=}",
        ),
        (mini_batch_size, f"{mini_batch_size=}"),
    ]:
      rl_utils.check_divisibility(v, full_batch_size, n, f"{full_batch_size=}")
    grad_acc_steps = self._training_config.get_with_default(
        "gradient_accumulation_steps", 1
    )
    trajectory_micro_batch_size = (
        self._training_config.train_trajectory_micro_batch_size
    )
    if trajectory_micro_batch_size is not None:
      expected_trajectory_mini_batch_size = (
          mini_batch_size * self.algo_config.num_generations
      )
      if (
          self._training_config.trajectory_mini_batch_size
          != expected_trajectory_mini_batch_size
      ):
        raise ValueError(
            "trajectory_mini_batch_size must equal mini_batch_size *"
            " num_generations:"
            f" {self._training_config.trajectory_mini_batch_size} vs"
            f" {expected_trajectory_mini_batch_size}"
        )
      expected_grad_acc_steps = (
          expected_trajectory_mini_batch_size
          // trajectory_micro_batch_size
      )
      if grad_acc_steps != expected_grad_acc_steps:
        raise ValueError(
            "trajectory gradient accumulation mismatch:"
            f" {grad_acc_steps=} vs {expected_grad_acc_steps=}"
        )

    logging.info(  # pylint: disable=logging-fstring-interpolation
        f"Training with {full_batch_size=}, {mini_batch_size=},"
        f" {train_micro_batch_size=}, {self._rollout_micro_batch_size=},"
        f" {self._compute_logps_micro_batch_size=}, {grad_acc_steps=},"
        f" {trajectory_micro_batch_size=}"
    )

    logging.info("Starting AgenticRLLearner training loop.")
    full_dataset_iterator = itertools.chain([first_item], full_batch_iterator)

    all_eval_prompts = (
        list(self._create_micro_batch_iterator(iter(eval_dataset), 1))
        if eval_dataset
        else []
    )
    p45_host_memory_enabled, p45_host_gc_interval = (
        host_memory_lib.contract(os.environ)
    )
    if p45_host_memory_enabled:
      _emit_p45_host_memory(
          phase="train_start", step=int(self.rl_cluster.global_steps)
      )
    p31_eval_rollout_enabled = (
        os.environ.get("CANON_P31_ENABLE_EVAL", "") == "1"
    )
    p57_inprocess_eval_enabled = (
        p31_eval_rollout_enabled
        and os.environ.get("CANON_PROFILE_FILE", "") in (
            "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env",
            "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env",
        )
        and os.environ.get("CANON_P57_RUN_KIND", "") == "train"
        and os.environ.get("CANON_P57_EXPECTED_UPDATES", "") == "300"
    )
    if p31_eval_rollout_enabled:
      expected_eval_rewards = len(all_eval_prompts) * self._num_generations()
      print(
          "[CANON_FROZENLAKE_P31] eval_input_inventory "
          f"prompts={len(all_eval_prompts)} "
          f"generations={self._num_generations()} "
          f"expected_rewards={expected_eval_rewards}",
          flush=True,
      )

    training_config = self.rl_cluster.cluster_config.training_config

    train_data_queue = queue_lib.SimpleDataQueue(maxsize=0)

    # 1. Start the rollout producer, except for the explicitly hash-bound P64
    # diagnostic replay. Replay loads the already certified train batch and
    # must never regenerate environment or serving work.
    p64_replay = p64_training_capsule.is_replay()
    prompt_queue = queue.Queue()
    replay_train_example = None
    if p64_replay:
      if eval_dataset is not None or all_eval_prompts:
        raise alignment.AlignmentGateError(
            "P64 diagnostic replay forbids evaluation inputs"
        )
      verified_capsule = p64_training_capsule.load_verified()
      replay_train_example = verified_capsule.build(
          TrainExample, alignment.ObservedTrainExample
      )
      replay_precheck = alignment.check_pre_backward(
          replay_train_example,
          step=int(self.rl_cluster.actor_trainer.train_steps),
          fail_closed=True,
      )
      if replay_precheck.get("verdict") != "PASS":
        raise alignment.AlignmentGateError(
            "P64 diagnostic replay capsule failed strict pre-alignment"
        )
      producer_future = Future()
      producer_future.set_result(None)
      print(
          "[P64.CAPSULE] producer_bypass verdict=PASS "
          "environment=0 rollout=0 rescore_b=0",
          flush=True,
      )
    else:
      orchestrator = self._build_orchestrator()
      initial_buffer_size = self.algo_config.off_policy_steps + 1
      logging.info(
          "Prefilling prompt queue with %d batches.", initial_buffer_size
      )
      for _ in range(initial_buffer_size):
        try:
          self._put_prompts_to_queue(prompt_queue, next(full_dataset_iterator))
        except StopIteration:
          prompt_queue.put(None)
          break

      producer_future = asyncio.run_coroutine_threadsafe(
          self._producer(orchestrator, prompt_queue, train_data_queue),
          self.loop,
      )

    # 2. Consume training examples and train.
    p38_precheck_only = (
        os.environ.get("CANON_P38_PRECHECK_ONLY", "0") == "1"
    )
    v1_fl_tp8_ab_arm = os.environ.get("CANON_V1_FL_TP8_AB_ARM", "")
    v1_fl_tp8_ab = v1_fl_tp8_ab_arm in ("p66-off", "serving-scope")
    p58_vma_diagnostic = (
        os.environ.get("CANON_P58_CHECKED_VMA_DIAGNOSTIC", "") in ("off", "on")
    )
    p58_seam_localization = (
        os.environ.get("CANON_P58_SEAM_LOCALIZATION", "") == "coarse"
    )
    (
        consumer_batch_size,
        require_full_consumer_batch,
        diagnostic_units,
    ) = _p38_diagnostic_consumer_contract(
        enabled=p38_precheck_only,
        full_batch_size=full_batch_size,
        mini_batch_size=mini_batch_size,
        train_micro_batch_size=train_micro_batch_size,
        num_generations=self._num_generations(),
        process_in_consumer=self._process_in_consumer,
        onehost_rehearsal=(
            os.environ.get("CANON_P38_ONEHOST_REHEARSAL", "0") == "1"
        ),
        m15_target_debug=(
            os.environ.get("CANON_APC_M15_TARGET_DEBUG", "")
            in ("off", "on")
        ),
        v1_fl_tp8_ab=v1_fl_tp8_ab,
        p58_vma_diagnostic=p58_vma_diagnostic,
        p58_seam_localization=p58_seam_localization,
    )
    if p38_precheck_only:
      m15_debug_arm = os.environ.get("CANON_APC_M15_TARGET_DEBUG", "")
      if m15_debug_arm in ("off", "on"):
        print(
            "[CAN" "ON_APC_M15_TARGET_CONTRACT] "
            f"arm={m15_debug_arm} topology=DP8xTP8 "
            "workload=m15/main backward=0 optimizer_commits=0",
            flush=True,
        )
      if v1_fl_tp8_ab:
        print(
            "[V1.FL.AB] "
            f"arm={v1_fl_tp8_ab_arm} topology=DP8xTP8 "
            "workload=p45/frozenlake backward=0 optimizer_commits=0",
            flush=True,
        )
      print(
          "[CANON_P38] DIAGNOSTIC_COVERAGE_CONTRACT "
          f"prompt_groups={full_batch_size} "
          f"unit_prompts={mini_batch_size} units={diagnostic_units} "
          f"generations={self._num_generations()} "
          f"trajectories={full_batch_size * self._num_generations()} "
          "partial_tail=reject verdict=PASS",
          flush=True,
      )
    if p64_replay:
      train_data_gen = iter([[replay_train_example]])
    else:
      train_data_gen = self._data_consumer_batch_generator(
          train_data_queue,
          consumer_batch_size,
          require_full_batch=require_full_consumer_batch,
      )
    is_packed = self._training_config.max_seq_token_per_tpu is not None
    if p64_replay and is_packed:
      raise alignment.AlignmentGateError(
          "P64 diagnostic replay requires the original unpacked P45 batch"
      )
    if is_packed:
      logging.info(
          "Using sequence packing with max_seq_token_per_tpu: %d",
          self._training_config.max_seq_token_per_tpu,
      )
      train_data_gen = rl_utils.pack_sequences(
          train_data_gen,
          self._training_config.max_seq_token_per_tpu,
          target_items_per_update=grad_acc_steps,
      )
    update_steps_since_last_sync = 0
    update_steps_per_full_batch = full_batch_size // mini_batch_size
    unpacked_micro_step_counter = 0
    did_eval_this_global_step = False
    p57_eval_policy_step_this_cycle = None
    full_batch_chunks = []
    for train_micro_batch in train_data_gen:
      if (
          self._training_config.max_steps
          and self.rl_cluster.global_steps >= self._training_config.max_steps
      ):
        logging.info(
            "Reached max_steps: %d >= %d",
            self.rl_cluster.global_steps,
            self._training_config.max_steps,
        )
        prompt_queue.put(None)
        break
      self._iter_steps += 1

      # TODO(tsbao): Re-enable this once off-policy filtering is needed.
      # Filter out examples that are too old (off-policy).
      # filtered_train_micro_batch = self._filter_outdated_offpolicy_examples(
      #     train_micro_batch
      # )
      # if not filtered_train_micro_batch:
      #   continue
      # train_micro_batch = filtered_train_micro_batch

      if p64_replay:
        if len(train_micro_batch) != 1:
          raise alignment.AlignmentGateError(
              "P64 replay must contain exactly one frozen train batch"
          )
        merged_train_micro_batch = train_micro_batch[0]
      elif self._process_in_consumer:
        # train_micro_batch is a list of lists of trajectories.
        all_trajectories = [t for group in train_micro_batch for t in group]
        try:
          train_examples = self._batch_to_train_example(
              batch_results=all_trajectories,
              mode=rl_cluster_lib.Mode.TRAIN,
          )
        except alignment.P38DiagnosticRoundComplete:
          if not p38_precheck_only:
            raise
          completed = alignment.p38_diagnostic_round_index()
          total = alignment.p38_diagnostic_rounds()
          profile_stopped = _canon_xprof_diagnostic_round_boundary(completed)
          if profile_stopped:
            # The precheck vehicle never reaches a global-step boundary, so
            # explicitly export the official semantic timeline after the
            # selected diagnostic round.  This is instrumentation only: it
            # does not enter the alignment values or training state.
            self.rl_cluster.perf_v2.export()
            print(
                "[P3.XPROF] semantic_perfetto_exported "
                f"completed_rounds={completed}",
                flush=True,
            )
          try:
            next_prompts = next(full_dataset_iterator)
          except StopIteration as exc:
            prompt_queue.put(None)
            raise RuntimeError(
                "P38 frozen-weight diagnostic exhausted the dataset after "
                f"{completed}/{total} rounds"
            ) from exc
          self._put_prompts_to_queue(prompt_queue, next_prompts)
          print(
              "[CANON_P38] DIAGNOSTIC_ROUND_SKIPPED_UPDATE "
              f"completed={completed}/{total} backward=0 "
              "optimizer_commits=0 weights=frozen next_round=queued",
              flush=True,
          )
          continue
        # GRPO returns a list with a single TrainExample.
        merged_train_micro_batch = train_examples[0]
      else:
        # TODO(b/491970038): handle seq packing case differently
        merged_train_micro_batch = jax.tree.map(
            lambda *xs: jnp.concatenate(xs, axis=0), *train_micro_batch
        )

      if (
          deepswe_debug.enabled()
          and deepswe_debug.rollout_only()
      ):
        marker = deepswe_debug.marker_prefix()
        trajectory_count = int(
            merged_train_micro_batch.completion_ids.shape[0]
        )
        print(
            f"[{marker}.ROLLOUT_ONLY] PASS "
            f"trajectories={trajectory_count} backward=0 "
            "optimizer_commits=0",
            flush=True,
        )
        prompt_queue.put(None)
        _ = producer_future.result()
        self.rl_cluster.close()
        return

      # Capture the rollout-policy step before the segmented trainer mutates
      # actor_trainer.train_steps. Rollout weights are synchronized only after
      # the evaluation block, so this remains the authoritative eval step.
      pre_update_train_step = self.rl_cluster.actor_trainer.train_steps
      pre_update_global_step = self.rl_cluster.global_steps
      receipt_train_example, _ = alignment.unwrap_train_example(
          merged_train_micro_batch
      )
      gsm8k_xprof.emit_work_receipt(
          receipt_train_example,
          train_step=int(pre_update_train_step),
          global_step=int(pre_update_global_step),
      )

      if os.environ.get("CANON_P28_G5C_ONLY", "") == "1":
        alignment_pass = self._run_p28_g5c_gate(merged_train_micro_batch)
        marker = (
            "GATE_ONLY_PASS" if alignment_pass else "CAUSAL_RED_CAPTURED"
        )
        print(
            f"[P28.G5C] {marker} no_optimizer=1 no_accumulator=1 "
            "no_checkpoint=1",
            flush=True,
        )
        prompt_queue.put(None)
        _ = producer_future.result()
        self.rl_cluster.close()
        return

      p28_g6_update = os.environ.get("CANON_P28_G6_UPDATE", "") == "1"
      if p28_g6_update:
        (
            expected_total,
            expected_trajectory_micro,
            marker_prefix,
            canonical_workload,
        ) = _segmented_update_geometry(os.environ)
        p33_workload = (
            os.environ.get("CANON_P33_WORKLOAD_LAUNCH_ADMITTED", "") == "1"
        )
        p34_workload = os.environ.get("CANON_P34_DEEPSWE", "") == "1"
        expected_grad_acc = expected_total // expected_trajectory_micro
        if (
            is_packed
            or trajectory_micro_batch_size != expected_trajectory_micro
            or grad_acc_steps != expected_grad_acc
        ):
          raise ValueError(
              "segmented update geometry changed: expected unpacked "
              f"{expected_total}->{expected_grad_acc}x"
              f"{expected_trajectory_micro} with "
              f"grad_acc={expected_grad_acc}; got "
              f"is_packed={is_packed}, "
              f"trajectory_micro_batch_size={trajectory_micro_batch_size}, "
              f"grad_acc_steps={grad_acc_steps}"
          )
        print(
            f"{marker_prefix} trajectory_microbatch "
            f"total={expected_total} size={expected_trajectory_micro} "
            f"chunks={expected_grad_acc} "
            f"grad_acc={expected_grad_acc} segmented=1",
            flush=True,
        )
        _canon_xprof_update_entry()
        # TraceAnnotation starts its TraceMe interval at construction time,
        # not at __enter__. Construct both parents only after the update-only
        # profiler window is open; otherwise their children are captured but
        # the two outer intervals begin before start_trace() and disappear.
        xprof_train_schedule = (
            gsm8k_xprof.zero_hp_train_microsteps(
                update_step=self.rl_cluster.global_steps,
                microsteps=expected_grad_acc,
            )
            if gsm8k_xprof.arm() == "zero-hp"
            else None
        )
        train_step_annotation = (
            contextlib.nullcontext()
            if xprof_train_schedule is not None
            else gsm8k_xprof.train_step_annotation(
                step_num=self.rl_cluster.global_steps
            )
        )
        update_annotation = gsm8k_xprof.trace_annotation(
            "zero_tim_update", update_step=self.rl_cluster.global_steps
        )
        # One flat PEFT_TRAIN span for the whole G6 update, mirroring the
        # official peft_trainer usage: official vocabulary, no nesting, no
        # custom names. The official call ends with async_end([train_loss])
        # because its train_step returns before the device finishes; this
        # call blocks through loss and commit_norm block_until_ready
        # internally, so the span already closes at device completion and
        # there is nothing asynchronous left to register.
        with train_step_annotation:
          with update_annotation:
            with (
                xprof_train_schedule
                if xprof_train_schedule is not None
                else contextlib.nullcontext()
            ):
              with self.rl_cluster.perf_v2.span(
                  perf_constants.PEFT_TRAIN,
                  self.rl_cluster.perf_v2.all_devices,
                  tags={
                      perf_constants.STEP: self.rl_cluster.global_steps,
                      perf_constants.ROLE: "actor",
                  },
              ):
                if os.environ.get("CANON_P66_BACKWARD_ARM", ""):
                  segmented_result = self._run_p66_backward_gate(
                      merged_train_micro_batch
                  )
                else:
                  segmented_result = self._run_p28_g6_update(
                      merged_train_micro_batch,
                      xprof_train_schedule=xprof_train_schedule,
                  )
        if (
            canonical_workload
            and (
                os.environ.get("CANON_P34_NO_COMMIT", "") == "1"
                if p34_workload
                else os.environ.get("CANON_P33_NO_COMMIT", "") == "1"
            )
        ):
          if (
              segmented_result.get("verdict") != "PASS"
              or segmented_result.get("commits") != 0
          ):
            raise RuntimeError(
                "canonical backward no-commit did not produce its signed verdict"
            )
          prompt_queue.put(None)
          _ = producer_future.result()
          self.rl_cluster.close()
          return

      # When ``train_micro_batch_size < mini_batch_size`` we want the trainer
      # to invoke ``train_step`` multiple times per outer iteration so the
      # optimizer (which fires every ``gradient_accumulation_steps`` micro-
      # steps) sees ``mini_batch_size``-shaped gradients while peak HBM is
      # only ``train_micro_batch_size``-shaped. Slice the merged train
      # example along its batch axis into chunks sized to one micro-step,
      # and pass the list to ``update_actor``; ``peft_trainer.train``
      # iterates the list and calls ``train_step`` once per chunk.
      n_total = merged_train_micro_batch.completion_ids.shape[0]
      seqs_per_chunk = trajectory_micro_batch_size or (
          train_micro_batch_size * self.algo_config.num_generations
      )
      if p28_g6_update:
        chunked_train_micro_batch = []
      elif trajectory_micro_batch_size is not None:
        chunked_train_micro_batch = _split_train_example_by_trajectory(
            merged_train_micro_batch,
            total_trajectories=n_total,
            trajectory_micro_batch_size=seqs_per_chunk,
        )
        marker = (
            "trajectory_microbatch"
            f" total={n_total} size={trajectory_micro_batch_size}"
            f" chunks={len(chunked_train_micro_batch)}"
            f" grad_acc={grad_acc_steps}"
        )
        logging.info(marker)
        if os.environ.get("CANON_ALIGNMENT_TRAIN", "") == "1":
          print(f"[CANON_GSM8K_TRAIN] {marker}", flush=True)
        elif os.environ.get("CANON_FROZENLAKE_P27", "") == "1":
          print(f"[CANON_FROZENLAKE_P27] {marker}", flush=True)
      elif n_total > seqs_per_chunk:
        chunked_train_micro_batch = [
            jax.tree_util.tree_map(
                lambda x: (
                    x[i : i + seqs_per_chunk]
                    if hasattr(x, "shape")
                    and x.shape
                    and x.shape[0] == n_total
                    else x
                ),
                merged_train_micro_batch,
            )
            for i in range(0, n_total, seqs_per_chunk)
        ]
      else:
        chunked_train_micro_batch = [merged_train_micro_batch]

      if not p28_g6_update:
        full_batch_chunks.extend(chunked_train_micro_batch)

      # --- Evaluation Logic on FIRST microbatch ---
      current_eval_dataset = None
      eval_examples = None
      eval_future = None
      eval_orchestrator = None
      p31_eval_rollout = p31_eval_rollout_enabled
      if (
          (not p28_g6_update or p31_eval_rollout)
          and update_steps_since_last_sync == 0
      ):
        current_train_step = _eval_schedule_step(
            segmented_update=p28_g6_update and p31_eval_rollout,
            pre_update_train_step=pre_update_train_step,
            current_train_step=self.rl_cluster.actor_trainer.train_steps,
        )
        if _should_run_eval(
            prompt_count=len(all_eval_prompts),
            schedule_step=current_train_step,
            eval_every_n_steps=training_config.eval_every_n_steps,
            last_eval_train_step=self._last_eval_train_step,
        ):
          self._last_eval_train_step = current_train_step
          self._eval_iter_steps = 0
          eval_orchestrator = self._build_orchestrator()
          eval_started = time.perf_counter()

          async def _eval_runner_async(current_eval_orchestrator):
            eval_examples = []
            async for batch in self._orchestrator_producer(
                current_eval_orchestrator,
                all_eval_prompts,
                num_generations=self._num_generations(),
            ):
              eval_example = self._batch_to_train_example(
                  batch,
                  rl_cluster_lib.Mode.EVAL,
              )
              eval_examples.extend(eval_example)
            return eval_examples

          eval_future = asyncio.run_coroutine_threadsafe(
              _eval_runner_async(eval_orchestrator), self.loop
          )
          eval_examples = eval_future.result()
          if p45_host_memory_enabled:
            _emit_p45_host_memory(
                phase="eval_materialized", step=int(current_train_step)
            )
          if p31_eval_rollout:
            expected_eval_rewards = (
                len(all_eval_prompts) * self._num_generations()
            )
            with self._rewards_window_lock:
              actual_eval_rewards = len(self._eval_rewards_window)
            if actual_eval_rewards != expected_eval_rewards:
              raise RuntimeError(
                  "P31 held-out eval coverage mismatch: "
                  f"prompts={len(all_eval_prompts)} "
                  f"generations={self._num_generations()} "
                  f"rewards={actual_eval_rewards} "
                  f"expected={expected_eval_rewards}"
              )
            print(
                "[CANON_FROZENLAKE_P31] eval_reward_inventory "
                f"step={current_train_step} "
                f"prompts={len(all_eval_prompts)} "
                f"generations={self._num_generations()} "
                f"rewards={actual_eval_rewards} "
                f"expected={expected_eval_rewards} verdict=PASS",
                flush=True,
            )
            with self._rewards_window_lock:
              eval_rewards_for_metrics = np.asarray(
                  self._eval_rewards_window, dtype=np.float32
              )
            eval_metrics = _frozenlake_evaluation_metrics(
                eval_rewards_for_metrics,
                wall_seconds=time.perf_counter() - eval_started,
                policy_step=current_train_step,
            )
            self.rl_cluster.buffer_metrics_async(
                {
                    f"frozenlake_eval/{name}": (value, np.mean)
                    for name, value in eval_metrics.items()
                },
                mode=rl_cluster_lib.Mode.EVAL,
                step=current_train_step,
            )
            print(
                "[CANON_FROZENLAKE_P42_JSON] "
                + json.dumps(eval_metrics, sort_keys=True, separators=(",", ":")),
                flush=True,
            )
          self._eval_iter_steps += 1
          # P57 publishes rollout-only environment rewards.  Sending these
          # examples through trainer eval would add an unrelated forward pass
          # and would contaminate the alignment/performance contract.
          current_eval_dataset = (
              None if p57_inprocess_eval_enabled else eval_examples
          )
          if p57_inprocess_eval_enabled:
            eval_examples = None
            p57_eval_policy_step_this_cycle = int(current_train_step)
          did_eval_this_global_step = True

      # --- First iteration Training Step (Parallelized with Rollout) ---
      # Note: Suppose one full batch has m minibatches, each minibatch has n
      # microbatches, and #iterations=K, we will:
      #   1. Train on the m * n microbatches once as we get them from rollout.
      #   2. When we get the full batch, repeat K-1 times on the entire batch.
      onehost_no_commit = (
          deepswe_debug.onehost() and deepswe_debug.no_commit()
      )
      onehost_xprof_arm = deepswe_debug.onehost_xprof_arm()
      onehost_before = None
      onehost_hbm_before = None
      onehost_after_warmup = None
      onehost_warmup_gradient_norms = None
      onehost_work_hashes = None
      if onehost_no_commit:
        from flax import nnx  # pylint: disable=g-import-not-at-top

        def onehost_memory_snapshot():
          snapshots = []
          for device in jax.local_devices():
            try:
              stats = device.memory_stats() or {}
            except Exception:  # pylint: disable=broad-except
              stats = {}
            snapshots.append({
                "device": int(device.id),
                "bytes_in_use": stats.get("bytes_in_use"),
                "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
                "bytes_limit": stats.get("bytes_limit"),
            })
          return snapshots

        actor_trainer = self.rl_cluster.actor_trainer
        reference_state = _p28_reference_state(self.rl_cluster)
        onehost_hbm_before = onehost_memory_snapshot()
        onehost_before = {
            "model": actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
                nnx.state(actor_trainer.model, nnx.Param)
            ),
            "optimizer": actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
                nnx.state(actor_trainer.optimizer, nnx.optimizer.OptState)
            ),
            "accumulator": actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
                nnx.state(actor_trainer.grad_accumulator), min_elements=1
            ),
            "reference": (
                actor_trainer._canon_fingerprint_state(reference_state)  # pylint: disable=protected-access
                if reference_state is not None
                else None
            ),
            "train_steps": actor_trainer.train_steps,
        }
        if onehost_xprof_arm:
          def hash_array(value):
            array = np.asarray(jax.device_get(value))
            digest = hashlib.sha256()
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(array.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(array).tobytes())
            return digest.hexdigest()

          work_fields = (
              "prompt_ids",
              "prompt_mask",
              "completion_ids",
              "completion_mask",
              "completion_valid_mask",
              "advantages",
              "policy_version",
          )
          onehost_work_hashes = {
              name: hash_array(getattr(merged_train_micro_batch, name))
              for name in work_fields
              if getattr(merged_train_micro_batch, name, None) is not None
          }
          onehost_work_hashes["shape_signature"] = hashlib.sha256(
              json.dumps(
                  {
                      name: list(
                          np.shape(getattr(merged_train_micro_batch, name))
                      )
                      for name in work_fields
                      if getattr(merged_train_micro_batch, name, None) is not None
                  },
                  sort_keys=True,
                  separators=(",", ":"),
              ).encode("utf-8")
          ).hexdigest()
          onehost_work_hashes["actor_update_calls"] = 2

      if not p28_g6_update:
        # The stock/non-segmented trainer does not pass through the G6 entry
        # above.  Open the same phase=update window immediately before its
        # actor update so P60 can compare a whole warm trainer update across
        # stock and zero-TIM without tracing rollout/decode.
        if onehost_xprof_arm:
          # Compile and execute the exact same no-commit update once before
          # tracing.  The second invocation reuses the same in-memory batch;
          # the no-commit contract below proves that neither invocation
          # changes model, optimizer, accumulator, reference, or train step.
          self.rl_cluster.update_actor(
              chunked_train_micro_batch, current_eval_dataset, skip_jit
          )
          actor_trainer = self.rl_cluster.actor_trainer
          trainer_buffer = (
              getattr(actor_trainer, "_prev_buffered_train_metrics", None)
              or getattr(actor_trainer, "_buffered_train_metrics", None)
          )
          try:
            gradient_values, _ = trainer_buffer.additional_metrics["grad_norm"]
            onehost_warmup_gradient_norms = [
                float(np.asarray(jax.device_get(value)))
                for value in gradient_values
            ]
          except (AttributeError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "P58 one-host XProf warmup gradient metric is missing"
            ) from exc
          reference_state = _p28_reference_state(self.rl_cluster)
          onehost_after_warmup = {
              "model": actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
                  nnx.state(actor_trainer.model, nnx.Param)
              ),
              "optimizer": actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
                  nnx.state(actor_trainer.optimizer, nnx.optimizer.OptState)
              ),
              "accumulator": actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
                  nnx.state(actor_trainer.grad_accumulator), min_elements=1
              ),
              "reference": (
                  actor_trainer._canon_fingerprint_state(reference_state)  # pylint: disable=protected-access
                  if reference_state is not None
                  else None
              ),
              "train_steps": actor_trainer.train_steps,
          }
          warmup_changed = {
              name: actor_trainer._canon_changed_paths(  # pylint: disable=protected-access
                  onehost_before[name], onehost_after_warmup[name]
              )
              for name in ("model", "optimizer", "accumulator")
          }
          if onehost_before["reference"] is not None:
            warmup_changed["reference"] = (
                actor_trainer._canon_changed_paths(  # pylint: disable=protected-access
                    onehost_before["reference"],
                    onehost_after_warmup["reference"],
                )
            )
          if (
              any(warmup_changed.values())
              or onehost_after_warmup["train_steps"]
              != onehost_before["train_steps"]
          ):
            raise RuntimeError(
                "P58 one-host XProf warmup mutated no-commit state: "
                f"{warmup_changed}"
            )
          print(
              "[P58.ONEHOST.XPROF] warmup_complete "
              f"arm={onehost_xprof_arm} commits=0 state_unchanged=1",
              flush=True,
          )
          # Commit the warmup semantic spans without invoking the exporter.
          # The later single-step export then selects only the newly committed
          # profiled repeat rather than merging both backward calls.
          perf_v2 = self.rl_cluster.perf_v2
          if not hasattr(perf_v2, "process_and_commit_timelines"):
            raise RuntimeError(
                "P58 one-host XProf requires the PerfMetrics v2 tracer"
            )
          perf_v2.process_and_commit_timelines()
          print(
              "[P58.ONEHOST.XPROF] semantic_warmup_discarded "
              f"arm={onehost_xprof_arm} next_export=profiled-repeat-only",
              flush=True,
          )
          _canon_xprof_onehost_update_entry(onehost_xprof_arm)
          self.rl_cluster.update_actor(
              chunked_train_micro_batch, current_eval_dataset, skip_jit
          )
          _canon_xprof_onehost_update_complete(onehost_xprof_arm)
          perf_v2.export()
        else:
          _canon_xprof_update_entry()
          self.rl_cluster.update_actor(
              chunked_train_micro_batch, current_eval_dataset, skip_jit
          )
        if hasattr(self.rl_cluster, "critic_trainer"):
          self.rl_cluster.update_critic(
              chunked_train_micro_batch, current_eval_dataset, skip_jit
          )
      p58_all_filtered_no_commit = _p58_all_filtered_no_commit_contract(
          os.environ,
          all_compact_filtered=bool(
              getattr(
                  merged_train_micro_batch, "all_compact_filtered", False
              )
          ),
          train_steps_before=int(pre_update_train_step),
          train_steps_after=int(self.rl_cluster.actor_trainer.train_steps),
      )
      if (
          p58_all_filtered_no_commit
          and self.rl_cluster.global_steps != pre_update_global_step
      ):
        raise alignment.AlignmentGateError(
            "P58 all-compact-filtered batch advanced RL global_steps before "
            "the no-commit boundary"
        )
      # Evaluation can contain hundreds of full trajectory objects. Keep it
      # alive only through the actor/critic call that consumes it; otherwise
      # Python function locals retain the most recent eval until the next
      # cadence boundary.
      current_eval_dataset = None
      eval_examples = None
      eval_future = None
      eval_orchestrator = None

      if onehost_no_commit:
        from flax import nnx  # pylint: disable=g-import-not-at-top

        if onehost_before is None:
          raise RuntimeError("one-host no-commit pre-state snapshot is missing")
        actor_trainer = self.rl_cluster.actor_trainer
        reference_state = _p28_reference_state(self.rl_cluster)
        onehost_after = {
            "model": actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
                nnx.state(actor_trainer.model, nnx.Param)
            ),
            "optimizer": actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
                nnx.state(actor_trainer.optimizer, nnx.optimizer.OptState)
            ),
            "accumulator": actor_trainer._canon_fingerprint_state(  # pylint: disable=protected-access
                nnx.state(actor_trainer.grad_accumulator), min_elements=1
            ),
            "reference": (
                actor_trainer._canon_fingerprint_state(reference_state)  # pylint: disable=protected-access
                if reference_state is not None
                else None
            ),
            "train_steps": actor_trainer.train_steps,
        }
        onehost_hbm_after = onehost_memory_snapshot()
        changed = {
            name: actor_trainer._canon_changed_paths(  # pylint: disable=protected-access
                onehost_before[name], onehost_after[name]
            )
            for name in ("model", "optimizer", "accumulator")
        }
        reference_changed = (
            actor_trainer._canon_changed_paths(  # pylint: disable=protected-access
                onehost_before["reference"], onehost_after["reference"]
            )
            if onehost_before["reference"] is not None
            else []
        )
        trainer_buffer = (
            getattr(actor_trainer, "_prev_buffered_train_metrics", None)
            or getattr(actor_trainer, "_buffered_train_metrics", None)
        )
        try:
          gradient_values, _ = trainer_buffer.additional_metrics["grad_norm"]
          gradient_norms = [
              float(np.asarray(jax.device_get(value)))
              for value in gradient_values
          ]
        except (AttributeError, KeyError, TypeError) as exc:
          raise RuntimeError(
              "one-host no-commit gradient metric is missing"
          ) from exc
        gradient_finite = bool(
            gradient_norms and np.all(np.isfinite(gradient_norms))
        )
        gradient_nonzero = bool(
            gradient_finite and any(value > 0.0 for value in gradient_norms)
        )
        state_unchanged = bool(
            not any(changed.values())
            and not reference_changed
            and onehost_after["train_steps"]
            == onehost_before["train_steps"]
        )
        gradient_repeat_exact = bool(
            not onehost_xprof_arm
            or gradient_norms == onehost_warmup_gradient_norms
        )
        if not gradient_finite or not state_unchanged:
          verdict = "FAIL"
        elif not gradient_repeat_exact:
          verdict = "FAIL"
        elif gradient_nonzero:
          verdict = "PASS"
        else:
          verdict = "INCONCLUSIVE_NO_SIGNAL"
        report = {
            "schema": "canon.local.deepswe.backward-no-commit.v1",
            "verdict": verdict,
            "commits": 0,
            "gradient_finite": gradient_finite,
            "gradient_nonzero": gradient_nonzero,
            "gradient_norms": gradient_norms,
            "gradient_warmup_norms": onehost_warmup_gradient_norms,
            "gradient_repeat_exact": gradient_repeat_exact,
            "repeat_count": 2 if onehost_xprof_arm else 1,
            "xprof_arm": onehost_xprof_arm,
            "work_hashes": onehost_work_hashes,
            "model_changed_paths": changed["model"],
            "optimizer_changed_paths": changed["optimizer"],
            "accumulator_changed_paths": changed["accumulator"],
            "reference_changed_paths": reference_changed,
            "train_steps_before": onehost_before["train_steps"],
            "train_steps_after": onehost_after["train_steps"],
            "optimizer_memory_kinds": list(
                actor_trainer.optimizer_state_memory_kinds()
            ),
            "hbm_before_backward": onehost_hbm_before,
            "hbm_after_backward": onehost_hbm_after,
            "state_fingerprints_before": onehost_before,
            "state_fingerprints_after_warmup": onehost_after_warmup,
            "state_fingerprints_after": onehost_after,
        }
        report_path = os.environ.get("CANON_DEEPSWE_ONEHOST_REPORT", "")
        if not report_path or not os.path.isabs(report_path):
          raise RuntimeError(
              "CANON_DEEPSWE_ONEHOST_REPORT must be an absolute path"
          )
        if os.path.exists(report_path):
          raise FileExistsError(
              f"refusing to overwrite one-host report: {report_path}"
          )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "x", encoding="utf-8") as report_file:
          json.dump(report, report_file, indent=2, sort_keys=True)
          report_file.write("\n")
        print(
            "[DEEPSWE.ONEHOST] backward_no_commit "
            f"verdict={verdict} commits=0 "
            f"gradient_norms={gradient_norms} "
            f"optimizer_memory_kinds={report['optimizer_memory_kinds']}",
            flush=True,
        )
        if verdict == "FAIL":
          raise RuntimeError(f"one-host backward no-commit failed: {report}")
        prompt_queue.put(None)
        _ = producer_future.result()
        self.rl_cluster.close()
        return

      # --- Weight Sync Logic ---
      if p28_g6_update:
        is_update = True
      elif is_packed:
        # `merged_train_micro_batch.is_update_step` is a 0-d jax scalar set
        # by `pack_sequences`; pull the host-side value before deciding.
        is_update = bool(
            np.asarray(merged_train_micro_batch.is_update_step).item()
        )
      else:
        # Mirror `peft_trainer._train_step`'s derivation:
        # `is_update_step` flips True every `grad_acc_steps` micro-batches.
        unpacked_micro_step_counter, is_update = _advance_unpacked_microsteps(
            unpacked_micro_step_counter,
            len(chunked_train_micro_batch),
            grad_acc_steps,
        )
        
      if is_update:
        update_steps_since_last_sync += 1
        
      if update_steps_since_last_sync == update_steps_per_full_batch:
        # --- Remaining Iterations Training Step ---
        iterations = self._num_iterations()
        if p58_all_filtered_no_commit and iterations != 1:
          raise alignment.AlignmentGateError(
              "P58 all-compact-filtered no-commit requires num_iterations=1"
          )

        for i in range(1, iterations):
          # TODO(b/483779605) Sub-step checkpointing.
          self._iter_steps += len(full_batch_chunks)

          # TODO(yixuanm): Eval during iteration too. Skipping for now as we 
          # will refactor the learner soon.
          self.rl_cluster.update_actor(
              full_batch_chunks, None, skip_jit
          )
          if hasattr(self.rl_cluster, "critic_trainer"):
            self.rl_cluster.update_critic(
                full_batch_chunks, None, skip_jit
            )
        full_batch_chunks.clear()


        global_step_time = time.time() - self._global_step_start_time
        if p58_all_filtered_no_commit:
          logging.info(
              "P58 all-compact-filtered rollout batch completed without "
              "advancing global step %d in %.2f seconds.",
              self.rl_cluster.global_steps,
              global_step_time,
          )
        else:
          if p57_eval_policy_step_this_cycle is not None:
            enclosing_global_step = _p57_eval_cycle_enclosing_step(
                policy_step=p57_eval_policy_step_this_cycle,
                actor_train_steps=self.rl_cluster.actor_trainer.train_steps,
                cluster_global_steps=self.rl_cluster.global_steps,
            )
            print(
                "[P57.EVAL.CYCLE] "
                f"policy_step={p57_eval_policy_step_this_cycle} "
                f"enclosing_global_step={enclosing_global_step}",
                flush=True,
            )
          logging.info(
              f"Global step {self.rl_cluster.global_steps} completed in"
              f" {global_step_time:.2f} seconds."
          )
          _canon_xprof_step_boundary()
        # One-line per-step diagnostic: raw rewards, solve rate, completion
        # length, advantage scale, and eval (when an eval just fired this
        # step). Mirrors the per-iter view a wandb dashboard would show
        # without depending on the async metric logger pipeline.
        with self._rewards_window_lock:
          train_rewards = np.asarray(self._train_rewards_window, dtype=np.float32)
          eval_rewards = np.asarray(self._eval_rewards_window, dtype=np.float32)
          self._train_rewards_window.clear()
          if did_eval_this_global_step:
            self._eval_rewards_window.clear()
        adv = np.asarray(merged_train_micro_batch.advantages, dtype=np.float32)
        cmask = np.asarray(
            merged_train_micro_batch.completion_mask, dtype=np.float32
        )
        compl_len = cmask.sum(axis=-1).mean() if cmask.size else 0.0
        valid_mask = np.asarray(
            getattr(
                merged_train_micro_batch,
                "completion_valid_mask",
                merged_train_micro_batch.completion_mask,
            ),
            dtype=np.float32,
        )
        raw_lengths = (
            valid_mask.sum(axis=-1)
            if valid_mask.size
            else np.asarray([], dtype=np.float32)
        )
        raw_compl_len = raw_lengths.mean() if raw_lengths.size else 0.0
        trunc_ratio = (
            float(
                (raw_lengths >= self.algo_config.max_response_length).mean()
            )
            if raw_lengths.size
            else 0.0
        )
        adv_abs_mean = float(np.abs(adv).mean()) if adv.size else float("nan")
        train_r_mean = (
            float(train_rewards.mean()) if train_rewards.size else float("nan")
        )
        train_solve = (
            float((train_rewards > 0.1).mean())
            if train_rewards.size
            else float("nan")
        )
        if eval_rewards.size and did_eval_this_global_step:
          eval_r_mean = float(eval_rewards.mean())
          eval_solve = float((eval_rewards > 0.1).mean())
          eval_str = (
              f" eval_reward={eval_r_mean:.3f}"
              f" eval_solve={eval_solve:.3f}"
              f" eval_n={eval_rewards.size}"
          )
        else:
          eval_str = ""
        # Best-effort read of trainer-side per-step metrics (grad_norm,
        # pg_loss, entropy, kl) directly from the actor trainer's metric
        # buffer so they appear in the per-step absl log alongside the
        # rollout metrics, independently of any external metric logger.
        trainer_str = ""
        try:
          actor_trainer = self.rl_cluster.actor_trainer
          trainer_buf = (
              getattr(actor_trainer, "_prev_buffered_train_metrics", None)
              or getattr(actor_trainer, "_buffered_train_metrics", None)
          )
          if trainer_buf is not None:
            extras = []
            if trainer_buf.losses:
              extras.append(f"loss={float(trainer_buf.loss):.4f}")
            am = trainer_buf.additional_metrics
            for key, label in (
                ("grad_norm", "grad_norm"),
                ("reduced_pg_loss", "reduced_pg_loss"),
                ("entropy", "entropy"),
                ("kl", "kl"),
                ("log_ratio/abs_mean", "log_ratio_abs"),
                ("pg_clipfrac", "clipfrac"),
            ):
              if key in am:
                vals, _ = am[key]
                if vals:
                  v = float(
                      np.mean([
                          np.asarray(common._metric_scalar(x)) for x in vals
                      ])
                  )
                  extras.append(f"{label}={v:.4f}")
            if extras:
              trainer_str = " " + " ".join(extras)
        except Exception as e:  # pylint: disable=broad-except
          logging.debug("Failed to read trainer buffered metrics: %s", e)
        logging.info(
            "[step %d] train_reward=%.3f train_solve=%.3f n=%d"
            " adv_abs_mean=%.3f compl_len=%.1f raw_compl_len=%.1f"
            " trunc_ratio=%.3f time=%.1fs%s%s",
            self.rl_cluster.global_steps,
            train_r_mean,
            train_solve,
            int(train_rewards.size),
            adv_abs_mean,
            float(compl_len),
            float(raw_compl_len),
            trunc_ratio,
            global_step_time,
            trainer_str,
            eval_str,
        )
        did_eval_this_global_step = False
        p57_eval_policy_step_this_cycle = None
        self.rl_cluster.buffer_metrics_async(
            {"perf/global_step_time": (global_step_time, np.mean)},
            mode=rl_cluster_lib.Mode.TRAIN,
            step=self.rl_cluster.global_steps,
        )
        if p58_all_filtered_no_commit:
          if (
              self.rl_cluster.global_steps != pre_update_global_step
              or self.rl_cluster.actor_trainer.train_steps
              != pre_update_train_step
          ):
            raise alignment.AlignmentGateError(
                "P58 all-compact-filtered no-commit state advanced before "
                "the next rollout batch"
            )
          print(
              "[P58.COMPACT_FILTER] all_filtered=1 optimizer_commits=0 "
              f"train_steps={pre_update_train_step} "
              f"global_steps={pre_update_global_step} "
              f"policy_version={self.policy_version} weight_sync=0",
              flush=True,
          )
          try:
            with self.rl_cluster.perf_v2.span(
                perf_constants.DATA_LOADING,
                tags={perf_constants.STEP: self.rl_cluster.global_steps},
            ):
              batch = next(full_dataset_iterator)
            self._put_prompts_to_queue(prompt_queue, batch)
          except StopIteration:
            prompt_queue.put(None)
        elif self.should_sync_weights:
          logging.info("Requesting sync lock to sync weights...")
          self._rollout_sync_lock.acquire_weight_sync()
          try:
            logging.info("Sync lock acquired. Syncing weights.")
            with perf_log.phase(
                "weight_sync",
                step=int(self.rl_cluster.global_steps),
                sink=lambda stage, seconds: self.rl_cluster.buffer_metrics_async(
                    {f"perf/{stage}_seconds": (seconds, np.mean)},
                    mode=rl_cluster_lib.Mode.TRAIN,
                    step=int(self.rl_cluster.global_steps),
                ),
            ), self.rl_cluster.perf_v2.span(
                perf_constants.WEIGHT_SYNC,
                self.rl_cluster.perf_v2.all_devices,
                tags={
                    perf_constants.STEP: self.rl_cluster.global_steps,
                },
            ):
              self.rl_cluster.sync_weights()
            self.policy_version += 1
            if p28_g6_update:
              print(
                  "[P28.G6] weight_sync_committed count=1 "
                  f"policy_version={self.policy_version} "
                  f"global_steps={self.rl_cluster.global_steps}",
                  flush=True,
              )
            logging.info(
                "Weights synced. Policy version incremented to %d.",
                self.policy_version,
            )
            try:
              with self.rl_cluster.perf_v2.span(
                  perf_constants.DATA_LOADING,
                  tags={
                      perf_constants.STEP: self.rl_cluster.global_steps,
                  },
              ):
                batch = next(full_dataset_iterator)
              self._put_prompts_to_queue(prompt_queue, batch)
            except StopIteration:
              prompt_queue.put(None)
          finally:
            self._rollout_sync_lock.release_weight_sync()
            logging.info("Sync lock released.")
        else:
          self.rl_cluster.global_steps += 1
          try:
            with self.rl_cluster.perf_v2.span(
                perf_constants.DATA_LOADING,
                tags={
                    perf_constants.STEP: self.rl_cluster.global_steps,
                },
            ):
              batch = next(full_dataset_iterator)
            self._put_prompts_to_queue(prompt_queue, batch)
          except StopIteration:
            prompt_queue.put(None)

        if p45_host_memory_enabled:
          # These are complete-step host references, not device state. Drop
          # them before cyclic GC so the post-GC record is a useful long-run
          # baseline. JAX compilation caches are deliberately left intact.
          del train_micro_batch
          del merged_train_micro_batch
          del chunked_train_micro_batch
          del train_rewards
          del eval_rewards
          del adv
          del cmask
          del valid_mask
          del raw_lengths
          completed_step = int(self.rl_cluster.actor_trainer.train_steps)
          collected = host_memory_lib.maybe_collect_garbage(
              step=completed_step, interval=p45_host_gc_interval
          )
          _emit_p45_host_memory(
              phase="global_step_complete",
              step=completed_step,
              gc_collected=collected,
          )

        self.rl_cluster.buffer_metrics(
            self.rl_cluster.perf_v2.export(),
            mode=rl_cluster_lib.Mode.TRAIN,
        )
        update_steps_since_last_sync = 0
        did_eval_this_global_step = False
        self._global_step_start_time = time.time()

    _ = producer_future.result()
    final_policy_step = int(self.rl_cluster.actor_trainer.train_steps)
    if p57_inprocess_eval_enabled and _should_run_eval(
        prompt_count=len(all_eval_prompts),
        schedule_step=final_policy_step,
        eval_every_n_steps=training_config.eval_every_n_steps,
        last_eval_train_step=self._last_eval_train_step,
    ):
      final_evaluation = self.evaluate_only(
          all_eval_prompts,
          policy_step=final_policy_step,
      )
      self._last_eval_train_step = final_policy_step
      final_metrics = {
          name: final_evaluation[name]
          for name in ("reward", "solve", "n", "wall_seconds", "policy_step")
      }
      self.rl_cluster.buffer_metrics_async(
          {
              f"frozenlake_eval/{name}": (value, np.mean)
              for name, value in final_metrics.items()
          },
          mode=rl_cluster_lib.Mode.EVAL,
          step=final_policy_step,
      )
      print(
          "[CANON_" "FROZENLAKE_P42_JSON] "
          + json.dumps(final_metrics, sort_keys=True, separators=(",", ":")),
          flush=True,
      )
      print(
          "[P57.EVAL.CYCLE] "
          f"policy_step={final_policy_step} enclosing_global_step=none",
          flush=True,
      )
      print(
          "[P57.EVAL] FINAL "
          f"policy_step={final_policy_step} "
          f"prompts={final_evaluation['prompts']} "
          f"generations={final_evaluation['generations']} "
          f"n={final_evaluation['n']} "
          f"reward={final_evaluation['reward']:.6f} "
          f"solve={final_evaluation['solve']:.6f} "
          "backward=0 optimizer_commits=0 evaluation_checkpoint_writes=0",
          flush=True,
      )
    self.rl_cluster.close()

  def _put_prompts_to_queue(
      self,
      prompt_queue: queue.Queue[TrainingInputT | None],
      batch,
  ):
    """Puts a batch of prompts into the queue.

    If the batch size does not match the expected full batch size, a warning is
    logged, and a StopIteration is raised to signal the end of the dataset.
    A None is put into the queue upon StopIteration to signal completion.

    Args:
      prompt_queue: The queue to put the batch into.
      batch: The batch of prompts (TrainingInputT).
    """
    current_batch_size = len(next(iter(batch.values())))
    if (
        self._training_config.max_steps
        and self.rl_cluster.global_steps >= self._training_config.max_steps
    ):
      logging.info(
          "Reached max_steps: %d >= %d",
          self.rl_cluster.global_steps,
          self._training_config.max_steps,
      )
      prompt_queue.put(None)
    elif current_batch_size != self._full_batch_size:
      logging.warning(
          "partial batch %d vs %d detected. The rest of the batch will be"
          " skipped.",
          current_batch_size,
          self._full_batch_size,
      )
      prompt_queue.put(None)
    else:
      prompt_queue.put(batch)
      if os.environ.get("CANON_ANCHOR_OVERLAP", "") == "1":
        # Every prompt flow passes through here (priming included), so
        # the anchor exists before any step's rescore-era consumers. The
        # queue is already loaded: the ~3s pinned-host copy below runs
        # while the rollout generates.
        self.rl_cluster.snapshot_anchor_policy()

  def _filter_outdated_offpolicy_examples(
      self,
      train_micro_batch: List[TrainExample],
  ) -> List[TrainExample]:
    """Filters out outdated off-policy examples."""
    filtered_train_micro_batch = []
    for train_example in train_micro_batch:
      if train_example.policy_version is not None and (
          train_example.policy_version[0] == -1
          or (
              self.policy_version - train_example.policy_version[0]
              <= self.algo_config.off_policy_steps
          )
      ):
        filtered_train_micro_batch.append(train_example)
    if not filtered_train_micro_batch:
      logging.warning(
          "Skipping microbatch: all %d examples are too old."
          " Current policy version: %d, data versions: %s,"
          " off_policy_steps: %d",
          len(train_micro_batch),
          self.policy_version,
          str([
              train_example.policy_version[0]  # pyrefly: ignore[unsupported-operation]
              for train_example in train_micro_batch
          ]),
          self.algo_config.off_policy_steps,
      )
    return filtered_train_micro_batch


_CANON_XPROF = {
    "completed_steps": 0,
    "configured": False,
    "profiler": None,
    "mode": "step",
    "skip": 0,
    "steps": 0,
    "directory": "",
    "host_tracer": 1,
    "python_tracer": 0,
    "tpu_trace_mode": "",
    "armed": False,
    "started": False,
}


_CANON_XPROF_TPU_TRACE_MODES = (
    "",
    "TRACE_ONLY_XLA",
    "TRACE_COMPUTE",
    "TRACE_COMPUTE_AND_SYNC",
)


def _canon_xprof_profile_options(
    *, host_tracer: int, python_tracer: int, tpu_trace_mode: str
) -> jax.profiler.ProfileOptions:
  """Builds fail-closed JAX profiler options for an update capture."""
  if tpu_trace_mode not in _CANON_XPROF_TPU_TRACE_MODES:
    raise ValueError(
        "CANON_XPROF_TPU_TRACE_MODE must be empty or one of "
        f"{_CANON_XPROF_TPU_TRACE_MODES[1:]}, got {tpu_trace_mode!r}"
    )
  options = jax.profiler.ProfileOptions()
  options.host_tracer_level = host_tracer
  options.python_tracer_level = python_tracer
  if tpu_trace_mode:
    options.advanced_configuration = {"tpu_trace_mode": tpu_trace_mode}
  return options


def _canon_xprof_update_entry():
  """Update-entry anchor: opens the phase=update trace window when armed.

  The G6 update call passes through here once per global step, in the
  trainer thread, at the same spot the semantic peft_train span opens, so
  the device window and the semantic span cover the same region. Opening
  here instead of at the step boundary keeps the rollout out of the
  capture entirely: the engine decode loop emits ~110k device ops per
  second per core and fills the device trace buffer (~2.8M events/core)
  in ~25s, silently dropping everything after -- which is how a whole-step
  window ends up with no backward in it. Anchored here, the buffer is
  spent on the update: forward, the sixteen accumulation dispatches, the
  pipelined reverse, and the optimizer commit.
  """
  if not _CANON_XPROF["armed"]:
    return
  _CANON_XPROF["armed"] = False
  options = _canon_xprof_profile_options(
      host_tracer=_CANON_XPROF["host_tracer"],
      python_tracer=_CANON_XPROF["python_tracer"],
      tpu_trace_mode=_CANON_XPROF["tpu_trace_mode"],
  )
  jax.profiler.start_trace(
      log_dir=_CANON_XPROF["directory"], profiler_options=options
  )
  _CANON_XPROF["started"] = True
  print(
      "[P51.XPROF] phase=update started "
      f"step={_CANON_XPROF['skip']} anchor=update_entry "
      f"tpu_trace_mode={_CANON_XPROF['tpu_trace_mode'] or 'default'}",
      flush=True,
  )


def _canon_xprof_onehost_update_entry(arm: str) -> None:
  """Opens the profiled repeat of the P58 no-commit one-host carrier."""
  from tunix.rl import deepswe_debug  # pylint: disable=g-import-not-at-top

  if deepswe_debug.onehost_xprof_arm() != arm:
    raise RuntimeError("P58 one-host XProf arm changed before update entry")
  if not _canon_xprof_configure():
    raise RuntimeError("P58 one-host XProf requires a non-empty trace directory")
  exact = (
      _CANON_XPROF["mode"] == "update"
      and _CANON_XPROF["skip"] == 0
      and _CANON_XPROF["steps"] == 1
      and _CANON_XPROF["host_tracer"] == 1
      and _CANON_XPROF["python_tracer"] == 0
      and _CANON_XPROF["tpu_trace_mode"] == "TRACE_COMPUTE"
  )
  if not exact or _CANON_XPROF["armed"] or _CANON_XPROF["started"]:
    raise RuntimeError(
        "P58 one-host XProf requires a fresh immediate update window with "
        "host_tracer=1 and python_tracer=0"
    )
  _CANON_XPROF["armed"] = True
  print(
      f"[P51.XPROF] phase=update armed step=0 arm={arm} "
      "anchor=onehost_profiled_repeat",
      flush=True,
  )
  _canon_xprof_update_entry()


def _canon_xprof_onehost_update_complete(arm: str) -> None:
  """Closes the no-commit one-host trace without inventing a global step."""
  from tunix.rl import deepswe_debug  # pylint: disable=g-import-not-at-top

  if deepswe_debug.onehost_xprof_arm() != arm:
    raise RuntimeError("P58 one-host XProf arm changed before trace stop")
  if not _CANON_XPROF["started"]:
    raise RuntimeError("P58 one-host XProf trace never started")
  jax.profiler.stop_trace()
  _CANON_XPROF["started"] = False
  print(
      f"[P51.XPROF] phase=update stopped step=0 arm={arm} "
      "anchor=onehost_profiled_repeat_complete",
      flush=True,
  )


def _canon_xprof_step_boundary():
  """Drives the xprof capture window at global-step boundaries.

  CANON_XPROF_DIR arms it (empty or unset stays off, so the docker -e K=""
  idiom is inert). Both window modes hang off this hook because the G6
  segmented update path never enters PeftTrainer.train(), where the
  trainer-level activation lives:

  - step (default): the official tunix.sft Profiler opens and closes at
    step boundaries. The window spans the whole global step, but the
    device planes only retain what fits the device trace buffer -- at the
    certified geometry that is the first ~25s of engine decode -- so this
    mode profiles the rollout fabric, not the trainer.
  - update: this hook arms at completed==skip; the update-entry hook opens
    the window at the G6 update call and completed==skip+steps closes it.
    The rollout never enters the capture, so the buffer retains the whole
    backward.

  Counting starts after the first completed step, so a skip below one
  could never fire; that is rejected loudly instead of leaving an empty
  profile directory behind. An update window still unopened at its stop
  boundary (the armed step never reached the G6 update call) is a hard
  error for the same reason.
  """
  if not _canon_xprof_configure():
    return
  if _CANON_XPROF["mode"] == "diagnostic":
    return

  _CANON_XPROF["completed_steps"] += 1
  completed = _CANON_XPROF["completed_steps"]
  if _CANON_XPROF["mode"] == "step":
    _CANON_XPROF["profiler"].maybe_activate(completed)
    _CANON_XPROF["profiler"].maybe_deactivate(completed)
    return
  if completed == _CANON_XPROF["skip"]:
    _CANON_XPROF["armed"] = True
    print(
        f"[P51.XPROF] phase=update armed step={completed}",
        flush=True,
    )
    return
  if completed == _CANON_XPROF["skip"] + _CANON_XPROF["steps"]:
    if not _CANON_XPROF["started"]:
      raise RuntimeError(
          "phase=update capture window never started before its stop "
          "boundary; the armed step never reached the G6 update call"
      )
    jax.profiler.stop_trace()
    _CANON_XPROF["started"] = False
    print(
        "[P51.XPROF] phase=update stopped "
        f"step={completed} anchor=step_completed",
        flush=True,
    )


def _canon_xprof_configure() -> bool:
  """Configures one of the mutually exclusive XProf window modes."""
  directory = os.environ.get("CANON_XPROF_DIR", "")
  if not directory:
    return False
  if not _CANON_XPROF["configured"]:
    skip = int(os.environ.get("CANON_XPROF_SKIP_STEPS", "") or "2")
    steps = int(os.environ.get("CANON_XPROF_STEPS", "") or "1")
    from tunix.rl import deepswe_debug  # pylint: disable=g-import-not-at-top

    onehost_immediate = bool(deepswe_debug.onehost_xprof_arm())
    if skip < 0 or steps < 1 or (skip == 0 and not onehost_immediate):
      raise ValueError(
          "CANON_XPROF_STEPS must be >= 1 and CANON_XPROF_SKIP_STEPS must "
          "be >= 1 except for the signed P58 one-host profiled repeat "
          f"(got skip={skip}, steps={steps})"
      )
    mode = os.environ.get("CANON_XPROF_PHASE", "") or "step"
    if mode not in ("step", "update", "diagnostic"):
      raise ValueError(
          "CANON_XPROF_PHASE must be 'step', 'update', or 'diagnostic': "
          f"{mode!r}"
      )
    if mode in ("update", "diagnostic") and steps != 1:
      raise ValueError(
          f"CANON_XPROF_PHASE={mode} requires CANON_XPROF_STEPS=1 "
          f"(got steps={steps})"
      )
    host_tracer = int(os.environ.get("CANON_XPROF_HOST_TRACER", "") or "1")
    python_tracer = int(
        os.environ.get("CANON_XPROF_PYTHON_TRACER", "") or "0"
    )
    tpu_trace_mode = os.environ.get("CANON_XPROF_TPU_TRACE_MODE", "")
    if tpu_trace_mode not in _CANON_XPROF_TPU_TRACE_MODES:
      raise ValueError(
          "CANON_XPROF_TPU_TRACE_MODE must be empty or one of "
          f"{_CANON_XPROF_TPU_TRACE_MODES[1:]}, got {tpu_trace_mode!r}"
      )
    if tpu_trace_mode and mode != "update":
      raise ValueError(
          "CANON_XPROF_TPU_TRACE_MODE is admitted only for "
          "CANON_XPROF_PHASE=update"
      )
    _CANON_XPROF.update(
        configured=True,
        mode=mode,
        skip=skip,
        steps=steps,
        directory=directory,
        host_tracer=host_tracer,
        python_tracer=python_tracer,
        tpu_trace_mode=tpu_trace_mode,
    )
    if mode == "step":
      _CANON_XPROF["profiler"] = sft_profiler.Profiler(
          initial_step=0,
          max_step=None,
          profiler_options=sft_profiler.ProfilerOptions(
              log_dir=directory,
              skip_first_n_steps=skip,
              profiler_steps=steps,
              host_tracer_level=host_tracer,
              python_tracer_level=python_tracer,
          ),
      )
  return True


def _canon_xprof_diagnostic_round_boundary(completed_rounds: int) -> bool:
  """Captures one frozen-weight diagnostic round and reports a stop event.

  `skip=1, steps=1` opens after round 0 has fully passed A/B/C and closes
  after round 1 has fully passed A/B/C.  Starting at this boundary excludes
  model/engine initialization and the first C compilation while retaining the
  production A rollout, independent B full rescore, and trainer-old C forward.
  """
  if not _canon_xprof_configure() or _CANON_XPROF["mode"] != "diagnostic":
    return False
  if completed_rounds == _CANON_XPROF["skip"]:
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = _CANON_XPROF["host_tracer"]
    options.python_tracer_level = _CANON_XPROF["python_tracer"]
    jax.profiler.start_trace(
        log_dir=_CANON_XPROF["directory"], profiler_options=options
    )
    _CANON_XPROF["started"] = True
    print(
        "[P3.XPROF] phase=diagnostic started "
        f"completed_rounds={completed_rounds} "
        f"capture_round={completed_rounds}",
        flush=True,
    )
    return False
  if completed_rounds == _CANON_XPROF["skip"] + _CANON_XPROF["steps"]:
    if not _CANON_XPROF["started"]:
      raise RuntimeError(
          "phase=diagnostic capture window never started before its stop "
          "boundary"
      )
    jax.profiler.stop_trace()
    _CANON_XPROF["started"] = False
    print(
        "[P3.XPROF] phase=diagnostic stopped "
        f"completed_rounds={completed_rounds} "
        f"captured_round={completed_rounds - 1}",
        flush=True,
    )
    return True
  return False
