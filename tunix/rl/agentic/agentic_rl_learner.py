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
import inspect
import json
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
import copy
import dataclasses
import itertools
import queue
import threading
from collections.abc import Hashable
from typing import Any, AsyncIterator, Callable, Dict, Generic, Iterable, Iterator, List, Sequence, Type, TypeVar, Optional, Set

from absl import logging
from flax import nnx
import flax
import jax
from jax import typing
import jax.numpy as jnp
import numpy as np
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import common
from tunix.perf.experimental import constants as perf_constants
from tunix.rl import function_registry
from tunix.rl import reward_manager  # pylint: disable=unused-import
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl.rollout import base_rollout
from tunix.rl import sub_batch_checkpoint
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.agents import model_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.environments import task_environment
from tunix.utils import compat
from tunix.rl.agentic.pipeline import rollout_orchestrator
from tunix.rl.agentic.rewards import reward  # pylint: disable=unused-import
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import checkpoint_manager as sft_checkpoint_manager
from tunix.sft import utils as sft_utils

ArrayLike = typing.ArrayLike
TrainingInputT = Dict[str, List[str] | ArrayLike]
RewardFn = Callable[..., List[float]]
MetricFn = Callable[..., rl_engine_lib.MetricsT]


@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  policy_version: np.ndarray | None = None


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
    sub_batch_checkpointing: Whether to checkpoint the sample rollout components
      and the live gradient-accumulation buffer at every trainer micro-step, so
      a preemption mid-global-step resumes without discarding completed rollouts
      or partially accumulated gradients. Requires 
      `training_config.checkpoint_root_directory` to be set (there is nothing
      to reconcile against otherwise) and is incompatible with sequence packing
      (`training_config.max_seq_token_per_tpu`) at the moment.
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
  filter_statuses: Optional[Set] = None
  overlong_filter: bool = False
  use_rollout_logps: bool = True
  sub_batch_checkpointing: bool = False


TConfig = TypeVar("TConfig", bound=AgenticRLConfig)


class AgenticRLLearner(abc.ABC, Generic[TConfig]):
  """Base class for Agentic RL Learners using asynchronous rollouts."""

  # Sub-batch checkpointing: inert CLASS-LEVEL defaults. Test subclasses
  # (e.g. the mock learners in agentic_grpo_learner_test) replace __init__
  # wholesale without calling super(), then drive producer/consumer paths
  # that read this state; they must observe the feature OFF, never
  # AttributeError. __init__ shadows these with instance attributes.
  _sb_mgr = None
  _sb_pending_state = None
  _sb_restored_trainer_state = False
  _sb_identity_queue = None
  _sb_preempt_at = frozenset()
  _bench_event_log = None
  _sb_chaos_prob = 0.0
  _bench_metrics = False
  _bench_anchor_pinned = False
  _experiment_start_time = 0.0
  _global_step_start_time = 0.0

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

  @compat.alias_init_param("rl_cluster", "rl_engine")
  def __init__(
      self,
      rl_engine: rl_engine_lib.RLEngine,
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
      rl_engine: RL engine containing actor, reference and reward models.
      algo_config: Configuration object.
      reward_fns: Reward functions.
      chat_parser: A parser to handle chat message formatting.
      metric_fns: A sequence of callables that compute metrics for the
        completions. Each callable should accept ``prompts``, ``completions``,
        ``rewards``, ``advantages`` and optional keyword arguments, and return a
        dictionary of metric names to tuples of ``(metric_value,
        aggregation_fn)``:  >>> def metric_fn( ...     prompts, completions,
        rewards, advantages, **kargs ... ): ...     return { ...       # ... ...
        "prompt_min_len": (min(len(p) for p in prompts), np.min), ...       #
        ... }
      agent_class: User defined agent class.
      agent_kwargs: Keyword arguments for the agent class.
      env_class: User defined environment class.
      env_kwargs: Keyword arguments for the environment class.
    """
    self.rl_engine = rl_engine
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
    self.rl_engine.actor_trainer.is_managed_externally = True
    if hasattr(self.rl_engine, "critic_trainer"):
      self.rl_engine.critic_trainer.is_managed_externally = True

    self.agent_class = agent_class
    self.agent_kwargs = agent_kwargs or {}
    self.env_class = env_class
    self.env_kwargs = env_kwargs or {}

    self._training_config = self.rl_engine.cluster_config.training_config

    self.rl_engine.global_steps = (
        self.rl_engine.actor_trainer.restored_global_step()
    )

    # --- Sub-batch checkpointing ---
    self._sb_mgr: sub_batch_checkpoint.SubBatchCheckpointManager | None = None
    self._sb_lock = threading.Lock()
    self._sb_completed: set[Hashable] = set()
    self._sb_counts: dict[tuple[Hashable, int], int] = {}
    self._sb_active: list[agent_types.TrajectoryItem] = []
    self._sb_pending_state: sub_batch_checkpoint.SubBatchState | None = None
    self._sb_identity_queue: queue_lib.AbstractDataQueue | None = None
    # Same-run snapshot-key monotonicity guard (saves use overwrite=True, so
    # a cadence wiring bug would otherwise clobber silently). Compares the
    # encoded sub-batch key, which is globally monotonic by construction.
    self._sb_last_snapshot_key = -1
    # The encoded key's two halves, maintained by _sb_snapshot: the window's
    # parent train_steps and the within-window order counter (resets when the
    # observed train_steps advances). -1 sentinels force a reset on the first
    # snapshot of a run and after a geometry rollback.
    self._sb_window_train_steps = -1
    self._sb_local = -1
    # The trainer's micro-step counter at the last snapshot. `local` is
    # self-incrementing, so key monotonicity alone cannot notice a snapshot
    # taken without a training call in between; the trainer's own counter
    # can (it ticks once per trained chunk in both modes).
    self._sb_last_snapshot_trainer_iter = -1
    # O(1) producer-side skip membership (mirrors _sb_active's group ids).
    self._sb_active_gids: set = set()
    # global_step whose step_complete=True snapshot was written, checked at
    # the boundary so a step that finished unmarked is loud rather than a
    # phantom step on the next restore. Deliberately a step STAMP, not a
    # bool: a bool would have to be re-armed inside _sb_step_boundary, so
    # dropping that call would latch it True and silently disable the guard
    # for the rest of the run -- the one failure it exists to catch.
    self._sb_step_complete_for_step: int | None = None
    # True when the trainer restored ANY weight checkpoint: every such
    # restart needs the rollout engine refreshed on disaggregated setups,
    # not just mid-step resumes.
    self._sb_restored_trainer_state = False
    # full_batch_size the restored snapshot was saved under, validated in
    # train() once the dataset reveals the current value.
    self._sb_restored_full_batch_size: int | None = None
    # BENCHMARK instrumentation (temporary, commit 4): fixed wall-clock
    # anchor for cross-restart cumulative-time metrics. Precedence: the
    # supervisor-pinned TUNIX_BENCH_START_TIME env (survives every process
    # death, so BASELINE runs -- no snapshot to carry an anchor -- get
    # continuous curves too), else the snapshot's anchor on resume, else
    # now.
    _bench_anchor = os.environ.get("TUNIX_BENCH_START_TIME")
    self._bench_anchor_pinned = _bench_anchor is not None
    self._experiment_start_time = (
        float(_bench_anchor) if _bench_anchor else time.time()
    )
    # Env-gated preemption chaos: probability per consumer micro-batch of a
    # HARD kill (os._exit), exercising real mid-step resume. 0 disables.
    # Independent of sub-batch enablement, so baseline runs crash the same
    # way and recover with whatever the stock path provides.
    self._sb_chaos_prob = float(os.environ.get("TUNIX_SB_CHAOS_PROB", "0"))
    # Deterministic preemption for the validation harness: comma-separated
    # trainer iter_steps values. The process hard-exits right after the
    # snapshot for that micro-step is DURABLE (see _sb_snapshot), so the
    # resume provably lands on that exact key -- unlike the probabilistic
    # chaos above, this can target mid-window micro-steps by name.
    self._sb_preempt_at = {
        int(v)
        for v in os.environ.get("TUNIX_SB_PREEMPT_AT", "").split(",")
        if v.strip()
    }
    # Wandb-independent evidence channel: every snapshot/preempt/resume
    # appends one JSON line here for the validation report to read.
    self._bench_event_log = os.environ.get("TUNIX_SB_EVENT_LOG")
    self._init_sub_batch_checkpointing()
    # Emit perf/* progress metrics even with sub-batch DISABLED when the
    # supervisor asks (the baseline arm of the comparison).
    self._bench_metrics = (
        self._sb_enabled or os.environ.get("TUNIX_SB_BENCH_METRICS") == "1"
    )
    # --- End sub-batch checkpointing section ---

    # Current iter steps for micro-batch based training.
    self._iter_steps = self.rl_engine.actor_trainer.iter_steps
    if not self._sb_enabled and self._iter_steps > 0:
      # TEMPORARY (bench): with sub-batch disabled the trainer restore above
      # is the only recovery path; record where it landed so the two-arm
      # report can measure re-executed work against the kill point (the
      # sub-batch arm logs its richer resume event in
      # _init_sub_batch_checkpointing).
      self._bench_event(
          "resume",
          stock=True,
          iter_steps=self._iter_steps,
          global_step=self.rl_engine.global_steps,
      )
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
            self.rl_engine.actor_trainer.model,
            self.rl_engine.rollout.model(),
        )
    )

    # Enable async rollout if trainer and rollout are not on the same mesh.
    # If they do, then doesn't make sense for the interleave because they will
    # have resource contention.
    self.can_enable_async_rollout = (
        self.rl_engine.cluster_config.role_to_mesh[rl_engine_lib.Role.ACTOR]
        != self.rl_engine.cluster_config.role_to_mesh[
            rl_engine_lib.Role.ROLLOUT
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
    self.tokenizer = rl_engine.tokenizer
    self.policy_version = self.rl_engine.global_steps
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

  # --- Sub-batch checkpointing -------------------------------------------
  @property
  def _sb_enabled(self) -> bool:
    return self._sb_mgr is not None

  def _init_sub_batch_checkpointing(self) -> None:
    """Constructs the sub-batch manager."""
    if not self.algo_config.sub_batch_checkpointing:
      return
    if self._training_config.max_seq_token_per_tpu is not None:
      raise ValueError(
          "sub_batch_checkpointing does not support sequence packing"
          " (max_seq_token_per_tpu) yet."
      )
    if not self._training_config.checkpoint_root_directory:
      raise ValueError(
          "sub_batch_checkpointing requires"
          " training_config.checkpoint_root_directory to be set: the"
          " sub-batch ledger is reconciled against the trainer's own weight"
          " checkpoint (by train_steps), so there must be one to reconcile"
          " against."
      )
    if hasattr(self.rl_engine, "critic_trainer"):
      raise ValueError(
          "sub_batch_checkpointing does not currently support a critic trainer:"
          " snapshots capture only the ACTOR's grad-accum buffer and step"
          " counters, so a resumed run would silently lose the critic's"
          " in-flight accumulator and desync its apply schedule."
      )
    if self.algo_config.off_policy_steps > 0:
      raise ValueError(
          "sub_batch_checkpointing requires off_policy_steps == 0: prompt"
          " prefetch interleaves groups from different global steps into the"
          " same micro-batches (completion-order release), which the"
          " per-step ledger accounting cannot represent. Disable one of the"
          f" two (off_policy_steps={self.algo_config.off_policy_steps})."
      )
    if getattr(self.rl_engine.cluster_config, "offload_to_cpu", False):
      logging.warning(
          "sub_batch_checkpointing feeds the trainer one chunk per"
          " update_actor call, and offload_to_cpu round-trips the full model"
          " between host and device on every call: expect roughly"
          " gradient_accumulation_steps times more transfers per mini-batch"
          " than the stock loop. Consider disabling offload_to_cpu with this"
          " feature."
      )
    grad_accum_steps = self._training_config.get_with_default(
        "gradient_accumulation_steps", 1
    )
    root_directory = os.path.join(
        self._training_config.checkpoint_root_directory, "sub_batch"
    )
    self._sb_mgr = sub_batch_checkpoint.SubBatchCheckpointManager(
        root_directory=root_directory,
        # The manager owns its own defaults (per-micro-step saves,
        # window-counting retention) and takes what carries over from the run
        # options (async toggle/timeouts) itself.
        run_options=self._training_config.checkpointing_options,
    )
    self._sb_check_trainer_checkpoint_cadence()

    actor_trainer = self.rl_engine.actor_trainer
    if actor_trainer.checkpoint_manager.latest_step() is None:
      self._sb_mgr.purge_steps_above(-1)
      return

    # Any restart that restored trainer weights needs the rollout engine
    # refreshed before generation on disaggregated setups, whether or not a
    # sub-batch snapshot is usable (see _sb_resync_rollout_weights).
    self._sb_restored_trainer_state = True

    train_steps = actor_trainer.train_steps
    state = self._sb_mgr.try_restore(
        train_steps,
        grad_accum_steps,
        target_training_state=self._sb_build_abstract_training_state(),
        num_generations=self._num_generations(),
        num_iterations=self._num_iterations(),
        mini_batch_size=self._training_config.mini_batch_size,
    )
    if state is None:
      return

    # Tracks the highest key written so far. Restoring this ensures that the
    # next snapshot overwrites or increments past this point, preventing
    # accidental regressions or identical duplicate saves.
    self._sb_last_snapshot_key = state.sub_batch_key

    # Tracks the trainer's exact micro-step iteration count (iter_steps) at
    # the time of the snapshot. Ensures that we don't save a new snapshot
    # unless an actual training call ran in between.
    self._sb_last_snapshot_trainer_iter = state.iter_steps

    # Derives the parent train_steps (window) and the local micro-step index 
    # from the mathematical key. This ensures the first snapshot taken after 
    # resume perfectly continues the numbering sequence (e.g., local + 1).
    self._sb_window_train_steps, self._sb_local = divmod(
        state.sub_batch_key, sub_batch_checkpoint.KEY_BASE
    )

    # TODO(angelmau): Handle step_complete case for resuming from checkpoint.
    if state.step_complete:
      # The previous step finished cleanly before the crash. The snapshot
      # still carries that step's ENTIRE retained ledger (payloads and
      # mu-capped counts survive until the in-memory boundary reset, which is
      # deliberate, see _sb_snapshot), so it must NOT be seeded or
      # re-injected: re-feeding a finished step's groups would let the
      # consumer's full-batch accounting fire a boundary after zero training,
      # producing a phantom global step (global_steps drift, a spurious
      # weight sync, and a silently skipped dataset batch). Start the next
      # step fresh: the trainer's restored global_steps stamp (N + 1) is
      # already correct.
      logging.info(
          "Sub-batch resume: previous step %d completed cleanly at"
          " iter_steps=%d. Starting step %d fresh.",
          state.global_step,
          state.iter_steps,
          self.rl_engine.global_steps,
      )
      return

    # Temporary holding buffer. We hold this restored state until train() 
    # initializes the dataloader so we can validate its full_batch_size.
    # If the batch size matches, we re-inject the active rollouts.
    self._sb_pending_state = state
    # BENCHMARK instrumentation (temporary): keep the cumulative clock
    # continuous across the restart and credit the wall-clock the resumed
    # step does NOT have to re-spend (rollouts + training already banked in
    # the snapshot). An estimate by construction -- re-rolling to measure
    # the true saving would defeat the feature being measured.
    if state.experiment_start_time > 0 and not self._bench_anchor_pinned:
      self._experiment_start_time = state.experiment_start_time
    if state.sub_step_elapsed > 0:
      logging.info(
          "Sub-batch resume: ~%.2fs of in-step work restored rather than"
          " re-done.",
          state.sub_step_elapsed,
      )
      self.rl_engine.buffer_metrics_async(
          {"perf/sub_batch_time_saved": (state.sub_step_elapsed, np.sum)},
          mode=rl_engine_lib.Mode.TRAIN,
          step=self.rl_engine.global_steps,
      )
    self._sb_completed = set(state.completed_group_ids)
    self._sb_counts = dict(state.trained_trajectory_counts)
    self._sb_active = list(state.active_group_trajectories)
    self._sb_active_gids = {t.group_id for t in self._sb_active}
    self._sb_restored_full_batch_size = state.full_batch_size
    if state.training_state is not None:
      # None means the snapshot was taken at an apply boundary: the buffer
      # was provably empty there and the freshly restored optimizer already
      # has an empty accumulator, so there is nothing to inject.
      self._sb_inject_training_state(state.training_state)
    # The trainer's own restore set `_iter_steps = train_steps * k` (the
    # window start, accumulator empty). The snapshot may be mid-window;
    # advance the trainer's counter to match or its apply/save boo.
    actor_trainer._iter_steps = state.iter_steps  # pylint: disable=protected-access

    # Mid-step crash: the trainer's custom_checkpoint_metadata_fn stamps
    # `global_steps + 1` (see rl_cluster.py), anticipating the increment
    # that only actually happens at a clean step boundary. So a restored
    # global_steps is one ahead of the step that was actually in progress.
    # Rewind to resume that same step instead of skipping past it.
    self.rl_engine.global_steps = state.global_step
    logging.info(
        "Sub-batch resume @ global_step=%d iter_steps=%d:"
        " %d completed groups, %d active trajectories.",
        state.global_step,
        state.iter_steps,
        len(self._sb_completed),
        len(self._sb_active),
    )
    self._bench_event(
        "resume",
        key=getattr(state, "sub_batch_key", None),
        iter_steps=state.iter_steps,
        global_step=state.global_step,
        completed_groups=len(self._sb_completed),
        active_trajectories=len(self._sb_active),
        had_training_state=state.training_state is not None,
    )

  def _sb_check_trainer_checkpoint_cadence(self) -> None:
    """Raises ValueError when the trainer's own weight-checkpoint cadence is coarser than every apply."""
    policy = getattr(
        self._training_config.checkpointing_options,
        "save_decision_policy",
        None,
    )
    interval = (
        getattr(policy, "interval", None)
        if policy is not None
        else getattr(
            self._training_config.checkpointing_options,
            "save_interval_steps",
            None,
        )
    )
    if interval != 1:
      raise ValueError(
          "sub_batch_checkpointing is enabled but the actor trainer's own"
          " checkpointing_options.save_decision_policy is not"
          f" FixedIntervalPolicy(interval=1) (got {policy!r}). Please configure"
          " the trainer for per-apply saves to coordinate with the sub-batch"
          " checkpoint cadence."
      )

  def _sb_accumulator(self, trainer) -> Any | None:
    """The trainer's `GradientAccumulator` module, or None if absent."""
    return getattr(trainer, "grad_accumulator", None)

  def _sb_build_abstract_for(
      self, trainer, diff: dict[tuple[Any, ...], Any]
  ) -> dict[tuple[Any, ...], Any]:
    """Builds the sharded abstract target for the trainer's accumulator."""
    ga_state = nnx.state(self._sb_accumulator(trainer))
    # Leaves come back as ShapeDtypeStructs when a NamedSharding mesh exists
    # and as the state's own arrays otherwise; both expose .sharding, which
    # is the only part needed here (shapes come from `diff`, i.e. from what
    # was actually saved).
    fixed = sft_checkpoint_manager.fix_sharding(ga_state)
    paths = [path for path, _ in ga_state.flat_state()]
    shardings = [leaf.sharding for leaf in jax.tree_util.tree_leaves(fixed)]
    assert len(paths) == len(shardings), (
        "fix_sharding's output did not preserve the accumulator's leaf"
        f" order/count ({len(paths)} paths vs {len(shardings)} shardings);"
        " the path-to-sharding zip below would silently mismatch."
    )
    sharding_by_path = dict(zip(paths, shardings))
    return {
        path: jax.ShapeDtypeStruct(
            np.asarray(value).shape,
            np.asarray(value).dtype,
            sharding=sharding_by_path[path],
        )
        for path, value in diff.items()
    }

  def _sb_build_abstract_training_state(
      self,
  ) -> dict[tuple[Any, ...], Any] | None:
    """Builds the abstract restore target mirroring what `save` persisted."""
    actor_diff = self._sb_extract_accumulator(self.rl_engine.actor_trainer)
    if actor_diff is None:
      return None
    return self._sb_build_abstract_for(
        self.rl_engine.actor_trainer, actor_diff
    )

  def _sb_extract_accumulator(
      self, trainer
  ) -> dict[tuple[Any, ...], Any] | None:
    """Extracts the trainer's live accumulator.

    Returns as a flat, tuple-path-keyed dict of plain numpy arrays: the
    `('grads', ...)` leaves plus `('denom',)`.

    Deliberately does NOT round-trip through `nnx.State`'s pure-dict helpers.
    Reading via `nnx.flat_state()` and `var[...]` gives plain host arrays for
    ease of read/update upon save/restore. Returns None when the trainer has
    no accumulator module.
    """
    accumulator = self._sb_accumulator(trainer)
    if accumulator is None:
      return None
    return {
        path: np.asarray(var[...])
        for path, var in nnx.state(accumulator).flat_state()
    }

  def _sb_peek_denom(self) -> float | None:
    """Cheaply reads the accumulator's scalar `denom` (no parameter-sized

    transfers), or None when the trainer has no accumulator. `denom == 0`
    means the accumulator was just reset at an apply: the buffer is provably
    empty and a snapshot omits the parameter-sized payload entirely.
    """
    accumulator = self._sb_accumulator(self.rl_engine.actor_trainer)
    if accumulator is None:
      return None
    return float(np.asarray(accumulator.denom[...]))

  def _sb_extract_training_state(self) -> Any | None:
    """The buffer payload for a snapshot: the actor accumulator's flat state

    (the only trainer in a Phase-1 GRPO run; the critic extension is
    deferred, see docs/designs/active/002-sub-batch-critic-support.md).
    """
    return self._sb_extract_accumulator(self.rl_engine.actor_trainer)

  def _sb_inject_accumulator(
      self, trainer, diff: dict[tuple[Any, ...], Any], role: str
  ) -> None:
    """Injects one restored accumulator payload into one trainer's live

    `GradientAccumulator`, in place, via the same `flat_state()`/`[...]`
    mechanism the extract uses (the one mechanism verified to survive nnx's
    structural expectations). No-ops with a warning if the trainer has no
    accumulator to inject into (a trainer-stack change across the restart).
    """
    accumulator = self._sb_accumulator(trainer)
    if accumulator is None:
      logging.warning(
          "Sub-batch snapshot carried a %s accumulator buffer but the"
          " current %s trainer has no GradientAccumulator to inject it into"
          " (trainer stack changed across the restart?). Discarding the"
          " buffer; training resumes with a fresh accumulator, re-training"
          " the in-flight micro-steps from scratch.",
          role,
          role,
      )
      return
    ga_state = nnx.state(accumulator)
    # Placement contract, learned the hard way on a v5e-8. The FRESH
    # accumulator is entirely single-device: grads are full-size zeros on
    # the default chip (6.4G there for a 1.7B fp32 actor) and denom is an
    # uncommitted scalar; `_shard_optimizer` only distributes them at the
    # first jit, grads functionally via the model's partition specs, denom
    # via an in-place set that is legal only on an uncommitted or already
    # mesh-consistent array. Orbax hands back the restored tree as
    # COMMITTED single-device arrays on that same nearly-full chip. So the
    # inject (a) must not double-hold the tree -- copy each leaf to host
    # and FREE orbax's device buffer before uploading, (b) must place each
    # leaf at its FINAL sharding, derived exactly the way the trainer
    # derives it (nnx.get_partition_spec + the actor mesh) -- the live
    # placement is useless (uniformly default-device) and blanket
    # replication would put the full grad tree on every chip, and
    # (c) must REPLACE buffers via set_value, never the indexed
    # `var[...] =` write -- that is an at[].set scatter which forces
    # co-located compute between the old single-device zeros and the new
    # sharded value (and materializes full leaves on the full chip).
    # Replacement also frees the fresh zeros as the loop walks the tree.
    try:
      mesh = self.rl_engine.cluster_config.role_to_mesh[
          rl_engine_lib.Role.ACTOR
      ]
    except (AttributeError, KeyError, TypeError):
      mesh = None
    if mesh is not None and getattr(mesh, "empty", False):
      mesh = None
    pspecs = {}
    if mesh is not None:
      for spath, svar in nnx.get_partition_spec(ga_state).flat_state():
        spec = svar.get_value() if hasattr(svar, "get_value") else svar
        pspecs[spath] = (
            spec
            if isinstance(spec, jax.sharding.PartitionSpec)
            else jax.sharding.PartitionSpec()
        )
    placements = set()
    for path, var in ga_state.flat_state():
      restored = diff[path]
      host = np.asarray(restored)
      if isinstance(restored, jax.Array):
        restored.delete()
        # Later readers of the payload (the geometry-mismatch rollback
        # builds its zeros from it) must see the host copy, not a
        # deleted device buffer.
        diff[path] = host
      if mesh is None:
        var.set_value(jnp.asarray(host))
        placements.add("default-device")
        continue
      spec = pspecs.get(path) or jax.sharding.PartitionSpec()
      with jax.transfer_guard("allow"):
        var.set_value(
            jax.device_put(host, jax.sharding.NamedSharding(mesh, spec))
        )
      placements.add(str(spec))
    nnx.update(accumulator, ga_state)
    logging.info(
        "Sub-batch %s accumulator inject: leaf placements %s.",
        role,
        sorted(placements),
    )

  def _sb_inject_training_state(self, training_state: Any) -> None:
    """Injects a restored buffer payload into the actor's accumulator."""
    self._sb_inject_accumulator(
        self.rl_engine.actor_trainer, training_state, "actor"
    )

  def _sb_item_step(self, item: agent_types.TrajectoryItem) -> int:
    """Which global step an active item belongs to.

    This learner assigns integer group ids as
    ``global_step * full_batch_size + prompt_index``
    (pairs_stream_generator), so the owning step is recoverable by integer
    division. Needed because the producer prefetches ``off_policy_steps + 1``
    prompt batches and can therefore register NEXT-step groups while the
    consumer is still mid-CURRENT-step: without step scoping those raced-ahead
    registrations would be serialized into the current step's snapshots (a
    mid-step resume would then re-inject next-step rollouts into this step and
    inflate its micro-batch accounting) and wiped by the current step's
    boundary reset (a later crash would then regenerate groups that already
    carry trained counts, mixing rollout lineages within a group). Non-int
    group ids (a custom orchestrator group_key_fn) fall back to the current
    step, which degrades to the unscoped behavior only for configurations
    this learner does not itself produce.
    """
    if isinstance(item.group_id, int) and self._full_batch_size > 0:
      return item.group_id // self._full_batch_size
    return self.rl_engine.global_steps

  def _sb_skip_group(self, group_id: Hashable) -> bool:
    """Returns True if the group was already generated, so the orchestrator skips regenerating it."""
    if not self._sb_enabled:
      return False
    with self._sb_lock:
      return group_id in self._sb_completed or group_id in self._sb_active_gids

  def _sb_register(self, batch: list[agent_types.TrajectoryItem]) -> None:
    """Registers a freshly generated group into the active set."""
    if not self._sb_enabled:
      return
    with self._sb_lock:
      self._sb_active.extend(batch)
      self._sb_active_gids.update(item.group_id for item in batch)

  def _sb_chunk_epoch(self, identities: list[tuple[Hashable, int]]) -> int:
    """Returns how many epochs `identities`' rows have already been trained

    (0 if never). Only meaningful when every row shares the same count --
    `_sb_split_by_epoch` is what guarantees that by construction before this
    is ever called on a chunk with more than one group, so `min` here is a
    defensive aggregate over an already-uniform list, not a substitute for
    that split.
    """
    if not self._sb_enabled or not identities:
      return 0
    with self._sb_lock:
      return min(self._sb_counts.get(key, 0) for key in identities)

  def _sb_split_by_epoch(
      self,
      chunk: Any,
      identities: list[tuple[Hashable, int]],
  ) -> list[tuple[Any, list[tuple[Hashable, int]]]]:
    """Splits a chunk if it contains rows needing different training epochs.

    In a perfectly lockstep pipeline, rows bound into a chunk are trained and
    checkpointed as an atomic unit, meaning they always share the same epoch
    count on restart. However, under Heterogeneous Sequence Packing (where parts
    of disparate prompts are binned into single rows) or dataset shuffling
    (where epoch 2+ chunks are scrambled), the chunking boundaries change across
    restarts. A mid-sweep crash then results in newly re-injected chunks holding
    rows with misaligned epoch histories.

    Because you cannot partially evaluate a tensor inside the TPU `update_actor`
    gradient step, we must physically slice the mixed tensor array here to
    prevent double-training or under-training individual rows. Returns the chunk
    unchanged in the common lockstep case.
    """
    if not identities:
      return [(chunk, identities)]
    with self._sb_lock:
      epochs = [self._sb_counts.get(key, 0) for key in identities]
    if len(set(epochs)) <= 1:
      return [(chunk, identities)]
    boundaries = (
        [0]
        + [i for i in range(1, len(epochs)) if epochs[i] != epochs[i - 1]]
        + [len(epochs)]
    )
    n_total = len(epochs)
    out = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
      sub_chunk = jax.tree_util.tree_map(
          lambda x: (
              x[start:end]
              if hasattr(x, "shape") and x.shape and x.shape[0] == n_total
              else x
          ),
          chunk,
      )
      out.append((sub_chunk, identities[start:end]))
    return out

  def _bench_event(self, event: str, **fields) -> None:
    """TEMPORARY (bench): appends one JSON line to TUNIX_SB_EVENT_LOG.

    The validation harness reads this file to prove which micro-steps
    trained (snapshot events carry key/window/local/denom), where a
    deterministic preemption landed, and where the resume picked up --
    independent of wandb availability. Write failures are logged, never
    raised: evidence capture must not take down training.
    """
    if not self._bench_event_log:
      return
    fields["event"] = event
    fields["ts"] = time.time()
    fields["pid"] = os.getpid()
    try:
      with open(self._bench_event_log, "a") as f:
        f.write(json.dumps(fields) + "\n")
    except OSError:
      logging.exception("Sub-batch bench event log write failed.")

  def _bench_wandb_flush(self) -> None:
    """TEMPORARY (bench): best-effort wandb flush before a DETERMINISTIC
    kill. A real preemption grants no such grace (which is why the chaos
    injector stays brutal), but wandb is observability, not the system
    under test -- without this the killed attempt's final metric rows,
    still queued in the client process, are lost with os._exit, and the
    benchmark reads cleaner with them present. Failures are swallowed:
    the kill must proceed regardless.
    """
    try:
      import wandb  # pylint: disable=g-import-not-at-top

      if wandb.run is not None:
        wandb.run.finish(exit_code=42)
    except Exception:  # pylint: disable=broad-except
      logging.info("Sub-batch bench: wandb flush before kill failed.")

  def _bench_stock_iter_hook(self, n_chunks: int) -> None:
    """TEMPORARY (bench): stock-baseline twin of _sb_snapshot's bench tail.

    With sub-batch checkpointing disabled there are no per-micro-step
    snapshots, so the two-arm benchmark gets its per-iter evidence and its
    deterministic kill point from here instead: one `stock_iter` event per
    stock training call, and -- when a named preemption target falls inside
    the iters that call just trained -- a hard exit AFTER the actor
    checkpoint queue is durable. The wait mirrors the sub-batch arm's
    durable-snapshot-then-die semantics: both arms die with their last
    recovery point committed, so the resume delta measures the DESIGN's
    lost work, not the async save queue's depth at kill time.
    """
    if self._bench_event_log is None and not self._sb_preempt_at:
      return  # inert outside the benchmark harness
    it = self.rl_engine.actor_trainer.iter_steps
    self._bench_event(
        "stock_iter",
        iter_steps=it,
        global_step=self.rl_engine.global_steps,
        chunks=n_chunks,
    )
    # One stock update call trains n_chunks iters at once; a target strictly
    # inside that range still fires, just at the call's end (with micro-batch
    # granularity n_chunks == 1 and the kill lands exactly on the target).
    fired = self._sb_preempt_at.intersection(range(it - n_chunks + 1, it + 1))
    if not fired:
      return
    self.rl_engine.actor_trainer.checkpoint_manager.wait()
    for target in sorted(fired):
      self._bench_event(
          "preempt",
          stock=True,
          iter_steps=target,
          killed_after_iter=it,
          global_step=self.rl_engine.global_steps,
      )
    logging.info(
        "SUB-BATCH VALIDATE (stock baseline): deterministic preemption after"
        " durable actor checkpoints (targets %s, iter_steps=%d,"
        " global_step=%d); os._exit(42).",
        sorted(fired),
        it,
        self.rl_engine.global_steps,
    )
    self._bench_wandb_flush()
    os._exit(42)

  def _sb_snapshot(
      self, identities: list[tuple[Hashable, int]], *, step_complete: bool
  ) -> None:
    """Snapshots the sub-batch state to disk after a micro-batch executes.

    Records one trained epoch for the pairs in `identities`, marks fully trained
    groups as completed, and persists the ledger and gradient accumulator
    buffer. This must be called immediately after `update_actor` so the saved
    epoch counts perfectly match the state of the saved gradient buffer.

    Active Set Retention (Deadlock Prevention):
      Fully trained trajectories are NOT evicted from the active set mid-step.
      They must be retained and re-injected on a restart because the trainer's
      step boundary logic blindly counts the number of chunks processed. If
      finished trajectories were evicted early, the boundary counter would wait
      for micro-batches that never arrive, causing a resume deadlock. The entire
      ledger is instead cleared wholesale at `_sb_step_boundary`.

    Buffer Optimization & Apply Boundaries:
      The gradient buffer payload is massive. If `denom == 0`, the accumulator
      was just reset by a weight update (apply boundary), so the buffer is empty
      and we omit saving it entirely to save disk I/O. Furthermore, snapshots
      at apply boundaries block synchronously. This ensures the sub-batch stream
      never falls behind the trainer's primary weight checkpoint, which would
      cause a stale buffer to be restored on top of updated weights.
    """
    if not self._sb_enabled:
      return
    assert self._sb_mgr is not None
    actor_trainer = self.rl_engine.actor_trainer
    iter_now = actor_trainer.iter_steps
    if iter_now <= self._sb_last_snapshot_trainer_iter:
      raise RuntimeError(
          "Sub-batch snapshot without an intervening training call: trainer"
          f" iter_steps {iter_now} <= last snapshot's"
          f" {self._sb_last_snapshot_trainer_iter}. This is a wiring bug;"
          " refusing to overwrite a same-run snapshot."
      )
    train_steps = actor_trainer.train_steps
    if train_steps != self._sb_window_train_steps:
      # First snapshot of a new window: the chunk just trained contained the
      # apply (or this is the run's first snapshot), so the local counter
      # restarts and the key encodes T-dot-0. `local` is pure within-window
      # ORDER, not a micro-step position -- which is all reconciliation
      # needs (max local wins) and what keeps this identical for fixed-k and
      # packed (ragged) windows.
      self._sb_window_train_steps = train_steps
      self._sb_local = 0
    else:
      self._sb_local += 1
    sub_batch_key = train_steps * sub_batch_checkpoint.KEY_BASE + self._sb_local
    if sub_batch_key <= self._sb_last_snapshot_key:
      # Saves use overwrite=True (a resumed run legitimately re-produces a
      # crashed run's keys), so a same-run duplicate would be silently
      # clobbered instead of surfacing. The encoding is globally monotonic
      # by construction (a new window's T-dot-0 exceeds the previous
      # window's last key since local < KEY_BASE), so a non-advancing key
      # can only mean the snapshot cadence is miswired (e.g. a snapshot
      # without a training call in between, or the trainer's apply counter
      # went backwards).
      raise RuntimeError(
          "Sub-batch snapshot key did not advance:"
          f" {sub_batch_key} <= last snapshot key"
          f" {self._sb_last_snapshot_key}. This is a wiring bug; refusing"
          " to overwrite a same-run snapshot."
      )
    mu = self._num_iterations()
    pairs_per_group = self._num_generations()
    step_now = self.rl_engine.global_steps
    with self._sb_lock:
      for key in identities:
        self._sb_counts[key] = min(self._sb_counts.get(key, 0) + 1, mu)
      for group_id in {gid for gid, _ in identities}:
        pairs = [(group_id, p) for p in range(pairs_per_group)]
        if all(self._sb_counts.get(p, 0) >= mu for p in pairs):
          self._sb_completed.add(group_id)
      completed_snapshot = list(self._sb_completed)
      counts_snapshot = dict(self._sb_counts)
      # Serialize only THIS step's payloads. The producer's prompt prefetch
      # can have registered next-step groups already (see _sb_item_step);
      # persisting them here would let a mid-step resume re-inject another
      # step's rollouts into this one.
      active_snapshot = [
          t for t in self._sb_active if self._sb_item_step(t) == step_now
      ]

    grad_accum_steps = self._training_config.get_with_default(
        "gradient_accumulation_steps", 1
    )
    # Peek the scalar denom BEFORE extracting: at apply boundaries the
    # accumulator was just reset (denom == 0, which is EVERY snapshot when
    # k == 1), the buffer is provably empty and omitted, so paying the
    # parameter-sized host transfer just to discard it would waste
    # per-micro-step bandwidth.
    denom = self._sb_peek_denom()
    at_apply_boundary = denom is None or denom == 0.0
    training_state = (
        None if at_apply_boundary else self._sb_extract_training_state()
    )
    self._sb_mgr.save(
        train_steps,
        self._sb_local,
        iter_steps=actor_trainer.iter_steps,
        global_step=self.rl_engine.global_steps,
        grad_accum_steps=grad_accum_steps,
        step_complete=step_complete,
        completed_group_ids=completed_snapshot,
        trained_trajectory_counts=counts_snapshot,
        active_group_trajectories=active_snapshot,
        training_state=training_state,
        num_generations=self._num_generations(),
        full_batch_size=self._full_batch_size,
        num_iterations=self._num_iterations(),
        mini_batch_size=self._training_config.mini_batch_size,
        sub_step_elapsed=time.time() - self._global_step_start_time,
        experiment_start_time=self._experiment_start_time,
    )
    self._sb_last_snapshot_key = sub_batch_key
    self._sb_last_snapshot_trainer_iter = iter_now
    if step_complete:
      self._sb_step_complete_for_step = self.rl_engine.global_steps
    if at_apply_boundary:
      # Durable barrier at applies: the sub-batch stream must never be
      # durably behind the trainer's own checkpoint for the same apply.
      self._sb_mgr.wait()
    self._bench_event(
        "snapshot",
        key=sub_batch_key,
        window=train_steps,
        local=self._sb_local,
        iter_steps=iter_now,
        global_step=step_now,
        denom=denom,
        step_complete=step_complete,
        carried_buffer=training_state is not None,
    )
    if iter_now in self._sb_preempt_at:
      # Deterministic validation preemption: make THIS snapshot durable
      # first, then die. A preemption landing just after a mid-window
      # commit is the scenario under test, and the explicit wait removes
      # the race with the async save queue (minutes deep on slow disks) --
      # the resume is then guaranteed to land on exactly this key.
      self._sb_mgr.wait()
      self._bench_event(
          "preempt",
          key=sub_batch_key,
          window=train_steps,
          local=self._sb_local,
          iter_steps=iter_now,
          global_step=step_now,
      )
      logging.info(
          "SUB-BATCH VALIDATE: deterministic preemption after durable"
          " snapshot key %d (iter_steps=%d, window=%d, local=%d);"
          " os._exit(42).",
          sub_batch_key,
          iter_now,
          train_steps,
          self._sb_local,
      )
      self._bench_wandb_flush()
      os._exit(42)

  def _sb_reinject(self, train_data_queue) -> None:
    """Pushes the restored active trajectories back onto the training queue.

    We reinject before any fresh generation starts, so they resume training
    without being regenerated (`_sb_skip_group` keeps the producer from creating
    new rollouts for these same groups).

    On the `process_in_consumer=False` path the queue carries processed
    `TrainExample`s, not raw trajectories, so each restored group is run
    through `_batch_to_train_example` here -- identical to what the normal
    producer path does for a freshly generated group -- before queueing.
    """
    if not self._sb_pending_state or not self._sb_active:
      return
    groups: dict[Hashable, list[agent_types.TrajectoryItem]] = {}
    for item in self._sb_active:
      groups.setdefault(item.group_id, []).append(item)

    for _, items in groups.items():
      if self._process_in_consumer:
        train_data_queue.put(items)
      else:
        train_examples = self._batch_to_train_example(
            batch_results=items, mode=rl_engine_lib.Mode.TRAIN
        )
        for train_example in train_examples:
          train_data_queue.put(train_example)
        if self._sb_identity_queue is not None:
          # Only the False path's consumer needs this side-channel (see the
          # matching comment in _producer); the True path derives identity
          # directly from the raw TrajectoryItems already in the queue item.
          self._sb_identity_queue.put(
              [(item.group_id, item.pair_index) for item in items]
          )
    logging.info("Sub-batch resume: re-injected %d groups.", len(groups))

  def _sb_dequeue_identities(
      self, count: int
  ) -> list[list[tuple[Hashable, int]]]:
    """Dequeues `count` per-group identity lists.

    This is done in lockstep with the `count` items
    `_data_consumer_batch_generator` just pulled off `train_data_queue` (both
    queues receive exactly one put per group, in the same order, from the single
    producer thread).
    """
    q = self._sb_identity_queue
    assert q is not None
    return [q.get(block=True) for _ in range(count)]

  def _sb_resync_rollout_weights(self) -> None:
    """Pushes the restored actor weights to the rollout engine."""
    if not self._sb_enabled or not self._sb_restored_trainer_state:
      return
    if not self.should_sync_weights:
      return  # colocated: rollout shares the actor's (restored) weights
    steps_before = self.rl_engine.global_steps
    self.rl_engine.sync_weights()
    self.rl_engine.global_steps = steps_before
    logging.info(
        "Sub-batch restart: restored actor weights pushed to the rollout"
        " engine before generation for step %d.",
        steps_before,
    )

  def _sb_step_boundary(self) -> None:
    """Resets the in-memory ledger pillars at the full-batch boundary."""
    if not self._sb_enabled:
      return
    step_now = self.rl_engine.global_steps
    if self._sb_step_complete_for_step != step_now:
      # Reaching the boundary means THIS step finished, so one of its
      # snapshots must have carried step_complete=True. If none did, the next
      # restore reads the final snapshot as mid-step and re-feeds an
      # already-trained ledger: a zero-training phantom step (global_steps
      # drift, spurious weight sync, silently skipped dataset batch).
      # Comparing stamps rather than clearing a flag keeps this self-arming --
      # a stale stamp from an earlier step still trips it. Report rather than
      # raise: the damage is one bounded step on a LATER restart, which is not
      # worth killing a live run over, but it is always a bug here.
      logging.error(
          "Sub-batch: global step %d reached its boundary without any"
          " snapshot marked step_complete (last marked step: %s). A restart"
          " from this step would resume it as mid-step and burn a phantom"
          " step. This is a snapshot-cadence bug (see the step_complete call"
          " sites).",
          step_now,
          self._sb_step_complete_for_step,
      )
    with self._sb_lock:
      self._sb_completed = set()
      self._sb_counts = {}
      self._sb_active = [
          t for t in self._sb_active if self._sb_item_step(t) > step_now
      ]
      self._sb_active_gids = {t.group_id for t in self._sb_active}

  def _sb_validate_batch_geometry(self, full_batch_size: int) -> None:
    """Rejects a full_batch_size change on sub-batch resume."""
    if not self._sb_enabled or self._sb_pending_state is None:
      return
    saved = self._sb_restored_full_batch_size
    if saved is None or saved == full_batch_size:
      return
    raise sub_batch_checkpoint.SubBatchGeometryError(
        f"Sub-batch snapshot was saved with full_batch_size={saved} but the"
        f" dataset provides {full_batch_size}; the ledger's group-id"
        " arithmetic is invalid across this change. Relaunch with the"
        " original configuration, or start a fresh checkpoint root to"
        " change geometry deliberately."
    )

  def _validate_rollout_config(self):
    """Validates that the rollout config is properly aligned with the algo config."""
    rollout_config = self.rl_engine.cluster_config.rollout_config
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
            "Please align these configurations before initializing RLEngine."
        )
      if self.algo_config.use_rollout_logps and not config.return_logprobs:
        raise ValueError(
            f"RolloutConfig ({mode}) must have return_logprobs=True for "
            "AgenticRLLearner when use_rollout_logps=True. Please set this "
            "before initializing RLEngine."
        )
      if (
          self.rl_engine.cluster_config.rollout_engine == "vllm"
          and not config.rollout_vllm_server_mode
      ):
        raise ValueError(
            f"RolloutConfig ({mode}) must have rollout_vllm_server_mode set to "
            "True for AgenticRLLearner if using vLLM engine. Please set this "
            "before initializing RLEngine."
        )

  def _compute_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      mode: rl_engine_lib.Mode,
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
    self.rl_engine.buffer_metrics_async(
        rewards_info["log_metrics"], mode=mode, step=expected_step
    )

    rewards_array = np.asarray(rewards_info["rewards"])
    with self._rewards_window_lock:
      target = (
          self._train_rewards_window
          if mode == rl_engine_lib.Mode.TRAIN
          else self._eval_rewards_window
      )
      target.extend(rewards_array.tolist())
      # Cap train window at full_batch_size * num_generations (one full step's
      # worth of per-sequence rewards) to bound the producer-vs-consumer
      # race: the producer can race up to ``off_policy_steps + 1`` batches
      # ahead, so without a cap the window would over-count next-step rewards
      # at the current step's boundary.
      if mode == rl_engine_lib.Mode.TRAIN and self._full_batch_size > 0:
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

    return agent, env

  def _model_call(
      self,
      chat_lists: List[Dict[str, str]],
      env: Any = None,
      max_generation_steps: int | None = None,
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
    result = self.rl_engine.generate(
        prompts=prompts,  # pytype: disable=wrong-arg-types
        apply_chat_template=False if self.chat_parser else True,
        mode=rl_engine_lib.Mode.TRAIN,
        trace_tags=tags,
        max_generation_steps=max_generation_steps,
    )

    return result

  def _build_orchestrator(self) -> rollout_orchestrator.RolloutOrchestrator:
    """Builds and configures a RolloutOrchestrator for parallel rollouts."""
    engine_kwargs = dict(
        model_call=self._model_call,
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
        timeout=self.algo_config.episode_timeout,
        max_response_length=self.algo_config.max_response_length,
        overlong_filter=self.algo_config.overlong_filter,
        filter_statuses=self.algo_config.filter_statuses,
        perf_v2=self.rl_engine.perf_v2,
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
      apply_sub_batch_skip: bool = False,
  ):
    """Generates trajectory groups using the orchestrator pattern.

    Args:
      orchestrator: The RolloutOrchestrator instance to use.
      prompt_iterator: An iterable yielding single `TrainingInputT` examples.
      num_generations: The number of episodes to run per agent-environment pair.
      collect_mode: The mode for trajectory collection (e.g., "Token").
      apply_sub_batch_skip: Whether to consult the sub-batch ledger to skip
        pre-crash groups. Must be True ONLY for the training stream. Because 
        this generator is shared with evaluation, if eval passed True, its
        independently numbered group IDs could accidentally collide with the
        training ledger  and cause eval prompts to be silently dropped.

    Yields:
      A list of trajectories for a group.
    """
    is_async_iterator = hasattr(prompt_iterator, "__aiter__")

    async def pairs_stream_generator():
      """Yield (agent, env) pairs with unique group_id per original prompt."""
      # TODO (tsbao): fix the group id when we can resume from mid global step
      # with mini-batch.
      group_id = self.rl_engine.global_steps * self._full_batch_size
      if is_async_iterator:
        async for single_example in prompt_iterator:  # pyrefly: ignore[not-iterable]
          # Sub-batch resume (train stream only): this group was already
          # generated pre-crash (fully consumed, or restored and pending
          # re-injection via _sb_reinject) -- do not regenerate it, just keep
          # the group_id counter advancing in lockstep with the original run.
          if apply_sub_batch_skip and self._sb_skip_group(group_id):
            group_id += 1
            continue
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
          if apply_sub_batch_skip and self._sb_skip_group(group_id):
            group_id += 1
            continue
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
    try:
      async with contextlib.aclosing(async_generator) as stream:
        async for group in stream:
          if group:
            # Retrieve the original input embedded in the task.
            yield group
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
      mode: rl_engine_lib.Mode,
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
    if mode == rl_engine_lib.Mode.TRAIN:
      expected_step = batch_results[0].group_id // self._full_batch_size
    else:
      expected_step = self.rl_engine.global_steps

    return self._process_results(
        trajectories=batch_results,
        mode=mode,
        expected_step=expected_step,
    )

  def _compute_packed_logps(self, example: TrainExample) -> TrainExample:
    # pack-first hook: algorithms that defer old/ref logp under packing compute
    # them on the packed buffer here. Base is a no-op.
    return example

  @abc.abstractmethod
  def _process_results(
      self,
      trajectories: List[Any],
      mode: rl_engine_lib.Mode = rl_engine_lib.Mode.TRAIN,
      expected_step: int | None = None,
  ) -> List[TrainExample]:
    """Processes generation results, computes rewards and advantages."""
    pass

  def _generate_and_compute_advantage(
      self,
      training_input: TrainingInputT,
      mode: rl_engine_lib.Mode = rl_engine_lib.Mode.TRAIN,
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
      # Push snapshotted rollouts into the queue before generating anything new.
      # The orchestrator is configured to skip regenerating these groups.
      self._sb_reinject(train_data_queue)
      producer_kwargs = dict(
          orchestrator=orchestrator,
          prompt_iterator=prompt_iterator,
          num_generations=self.algo_config.num_generations,
          collect_mode="Token",
      )
      # `_orchestrator_producer` is an override point (upstream tests and
      # custom learners replace it) whose original signature predates
      # `apply_sub_batch_skip`: pass the flag only to implementations that
      # declare it, so older-signature overrides keep working -- the
      # sub-batch skip is simply inert for them, which is correct (their
      # rollouts never enter the ledger either). Instance state cannot
      # carry this flag instead: the eval producer runs CONCURRENTLY with
      # this train producer and must not consult the ledger (id-collision
      # hazard documented on the parameter).
      if "apply_sub_batch_skip" in inspect.signature(
          self._orchestrator_producer
      ).parameters:
        producer_kwargs["apply_sub_batch_skip"] = True
      async for batch in self._orchestrator_producer(**producer_kwargs):
        try:
          # Register freshly generated rollouts in the ledger before they enter
          # the queue. This is the last point where raw string metadata exists
          # before the path below strips it away to create numeric
          # TrainExample tensors. The parallel identity queue (run 1:1) is
          # populated now so these string IDs can safely bypass the tensor
          # pipeline.
          self._sb_register(batch)
          if (
              self._sb_identity_queue is not None
              and not self._process_in_consumer
          ):
            # The process_in_consumer=True path captures identity in the
            # consumer's _to_train_examples wrapper instead (the raw
            # trajectories still exist there) -- pushing here too would
            # double-count. Push only on the producer-processing path.
            self._sb_identity_queue.put(
                [(item.group_id, item.pair_index) for item in batch]
            )
          if self._process_in_consumer:
            # Put raw batch (list of trajectories) into queue.
            # We put it once, and consumer will handle iterations.
            train_data_queue.put(batch)
          else:
            train_examples = self._batch_to_train_example(
                batch_results=batch,
                mode=rl_engine_lib.Mode.TRAIN,
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
      self, queue: queue_lib.AbstractDataQueue, batch_size: int
  ):
    """Yields micro-batches from a queue until a None is received."""
    item_iterator = iter(lambda: queue.get(block=True), None)
    while True:
      batch = list(itertools.islice(item_iterator, batch_size))
      if not batch:
        return  # The iterator is exhausted.
      yield batch

  def train(
      self,
      train_dataset: Iterable[TrainingInputT],
      eval_dataset: Iterable[TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """Main training loop for the AgenticRLLearner."""
    full_batch_iterator = iter(train_dataset)

    if self.rl_engine.global_steps > 0:
      logging.info(
          "Skipping %d batches from train_dataset to fast-forward to step %d",
          self.rl_engine.global_steps,
          self.rl_engine.global_steps,
      )
      # TODO(b/483779605): Current implementation of fast-forwarding does not
      # take into account the mini-batch size. Follow-up CL will address this.
      for _ in range(self.rl_engine.global_steps):
        try:
          next(full_batch_iterator)
        except StopIteration:
          logging.warning("Train dataset exhausted while skipping batches.")
          self.rl_engine.close()
          return

    try:
      first_item = next(full_batch_iterator)
    except StopIteration:
      logging.warning("Training dataset is empty.")
      self.rl_engine.close()
      return

    full_batch_size = len(next(iter(first_item.values())))  # pyrefly: ignore[bad-argument-type]
    self._full_batch_size = full_batch_size
    # A restored sub-batch state is only valid under the batch geometry it was
    # saved with.
    self._sb_validate_batch_geometry(full_batch_size)
    # Initialize batch sizes.
    mini_batch_size = self._training_config.mini_batch_size or full_batch_size
    train_micro_batch_size = (
        self._training_config.train_micro_batch_size or mini_batch_size
    )
    # Rollout micro batch size has to be 1 since we only process individual
    # prompts.
    self._rollout_micro_batch_size = 1
    self._process_in_consumer = False

    # pack-first: with packing on, the pack is the logp batch, so logp is
    # computed on the packed buffer after pack_sequences; do not defer
    # conversion to the consumer (which would enqueue raw lists pack_sequences
    # cannot consume).
    packing_enabled = self._training_config.max_seq_token_per_tpu is not None
    if self._compute_logps_micro_batch_size > 1 and not packing_enabled:
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

    logging.info(  # pylint: disable=logging-fstring-interpolation
        f"Training with {full_batch_size=}, {mini_batch_size=},"
        f" {train_micro_batch_size=}, {self._rollout_micro_batch_size=},"
        f" {self._compute_logps_micro_batch_size=}, {grad_acc_steps=}"
    )

    logging.info("Starting AgenticRLLearner training loop.")
    full_dataset_iterator = itertools.chain([first_item], full_batch_iterator)

    all_eval_prompts = (
        list(self._create_micro_batch_iterator(iter(eval_dataset), 1))
        if eval_dataset
        else []
    )

    training_config = self.rl_engine.cluster_config.training_config

    train_data_queue = queue_lib.SimpleDataQueue(maxsize=0)
    if self._sb_enabled:
      # A parallel side-channel queue that passes trajectory identities
      # (group_id, pair_index) perfectly in step with the main
      # train_data_queue.
      self._sb_identity_queue = queue_lib.SimpleDataQueue(maxsize=0)

    # 1. Start producer thread to generate rollouts and training examples.
    orchestrator = self._build_orchestrator()

    prompt_queue = queue.Queue()
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

    # Mid-step resume on a disaggregated setup: the rollout engine must get
    # the restored actor weights before it generates anything (see
    # _sb_resync_rollout_weights). Must run before the producer starts.
    self._sb_resync_rollout_weights()

    producer_future = asyncio.run_coroutine_threadsafe(
        self._producer(orchestrator, prompt_queue, train_data_queue),
        self.loop,
    )

    # 2. Consume training examples and train.
    train_data_gen = self._data_consumer_batch_generator(
        train_data_queue, train_micro_batch_size
    )
    if self._process_in_consumer:
      # Convert raw Trajectory groups into TrainExamples up front, before
      # `pack_sequences` wraps the generator below. Otherwise `pack_sequences`
      # (via `unpad_train_example`) receives raw lists and crashes on
      # `.prompt_ids`.
      def _to_train_examples(raw_gen):
        for group_batch in raw_gen:
          all_trajectories = [t for group in group_batch for t in group]
          if self._sb_identity_queue is not None:
            # Consumer-path identity capture: this is the only point on
            # this path where raw trajectories still exist (the yielded
            # TrainExamples drop identity at the merge to tensors). One
            # identity list per yield, row order matching the concatenated
            # example rows; the merge dequeues exactly one per iteration.
            self._sb_identity_queue.put(
                [(t.group_id, t.pair_index) for t in all_trajectories]
            )
          yield self._batch_to_train_example(
              batch_results=all_trajectories,
              mode=rl_engine_lib.Mode.TRAIN,
          )

      train_data_gen = _to_train_examples(train_data_gen)
    is_packed = self._training_config.max_seq_token_per_tpu is not None
    if is_packed:
      mesh = self.rl_engine.cluster_config.role_to_mesh[
          rl_engine_lib.Role.ACTOR
      ]
      # The packed batch size must be a multiple of the FSDP and DP mesh axis
      # sizes.
      pack_size = rl_utils.compute_pack_size(mesh)

      logging.info(
          "Using sequence packing with max_seq_token_per_tpu: %d, "
          " pack_size: %d",
          self._training_config.max_seq_token_per_tpu,
          pack_size,
      )

      # Update boundary in sequences (mini-batch semantics): packing is
      # independent of any micro-batch/streaming granularity.
      train_data_gen = rl_utils.pack_sequences(
          train_data_gen,
          self._training_config.max_seq_token_per_tpu,  # pyrefly: ignore[bad-argument-type]
          sequences_per_update=mini_batch_size * self._num_generations(),
          pack_size=pack_size,
          max_segments_per_packed_row=getattr(
              self._training_config, "max_segments_per_packed_row", None
          ),
      )
    update_steps_since_last_sync = 0
    update_steps_per_full_batch = full_batch_size // mini_batch_size
    unpacked_micro_step_counter = self.rl_engine.actor_trainer.iter_steps
    # To detect step boundaries during sub-batching, we count data chunks
    # consumed rather than model applies. Upon a mid-step resume, earlier
    # applies are already baked into the restored weights and won't run again.
    # By counting every data chunk (whether trained or skipped), we guarantee
    # the step always finishes correctly regardless of where the crash occurred.
    sb_units_seen = 0
    sb_units_per_step = max(1, full_batch_size // train_micro_batch_size)
    did_eval_this_global_step = False
    full_batch_chunks = []
    full_batch_chunk_identities = []
    # Hold the eval dataset (created at the start of a step) across the entire
    # global step until we hit a micro-batch that actually trains (isn't
    # skipped). This ensures evals aren't lost if the first micro-batch is
    # skipped during a sub-batch resume.
    pending_eval_dataset = None
    for train_micro_batch in train_data_gen:
      if (
          self._training_config.max_steps
          and self.rl_engine.global_steps >= self._training_config.max_steps
      ):
        logging.info(
            "Reached max_steps: %d >= %d",
            self.rl_engine.global_steps,
            self._training_config.max_steps,
        )
        prompt_queue.put(None)
        break
      # BENCHMARK instrumentation (temporary): random HARD preemption at
      # micro-batch granularity -- mid-window kills are the case sub-batch
      # checkpointing exists for, so the injector must be able to land
      # there, not only at step boundaries. os._exit skips every finally/
      # atexit, mimicking a real preemption; the supervisor restarts the
      # process until it exits cleanly.
      if self._sb_chaos_prob > 0 and np.random.random() < self._sb_chaos_prob:
        logging.warning(
            "SUB-BATCH BENCH: simulating hard preemption (os._exit(42)) at"
            " iter_steps=%d, global_step=%d.",
            self._iter_steps,
            self.rl_engine.global_steps,
        )
        os._exit(42)
      self._iter_steps += 1

      # TODO(tsbao): Re-enable this once off-policy filtering is needed.
      # Filter out examples that are too old (off-policy).
      # filtered_train_micro_batch = self._filter_outdated_offpolicy_examples(
      #     train_micro_batch
      # )
      # if not filtered_train_micro_batch:
      #   continue
      # train_micro_batch = filtered_train_micro_batch

      # `train_micro_batch` is always a Sequence[TrainExample] now:
      #  - _process_in_consumer: converted up front (GRPO -> single-element list)
      #  - producer-side processing: TrainExamples straight from the queue
      #  - is_packed: a single packed TrainExample from pack_sequences
      # jax.tree.map(concatenate) over a single-element list is a no-op, so this
      # equals the old `train_examples[0]` for the GRPO consumer path.

      # Merge micro-batches safely by concatenating all their JAX arrays.
      # Sequence packing introduces a 0-dimensional scalar (`num_segments`)
      # which crashes `jnp.concatenate` (you can't concatenate 0-d values).
      # We bypass this by passing scalars through untouched (just taking the
      # first one, since they are identical across all micro-batches anyway).
      merged_train_micro_batch = jax.tree.map(
          lambda *xs: (
              jnp.concatenate(xs, axis=0) if np.ndim(xs[0]) else xs[0]
          ),
          *train_micro_batch,
      )

      if is_packed:
        # pack-first: old/ref logp were deferred (left None) so they can be
        # computed here on the packed buffer (segment-aware forward), sharing
        # the same packed representation training uses.
        merged_train_micro_batch = self._compute_packed_logps(
            merged_train_micro_batch
        )
        # Packing efficiency: segment_ids==0 marks padding and dummy packs, so
        # this is the wasted fraction. As a WeightedMetric it reduces by
        # Sum(pad)/Sum(total), which stays correct if pack shapes ever differ.
        seg = np.asarray(merged_train_micro_batch.segment_ids)
        dummy_ratio = sft_utils.WeightedMetric(
            jnp.array(float((seg == 0).sum())), jnp.array(float(seg.size))
        )
        self.rl_engine.buffer_metrics_async(
            {  # pyrefly: ignore[bad-argument-type]
                "packing/dummy_ratio": (
                    dummy_ratio,
                    common.global_weighted_mean,
                ),
                "packing/seqs_per_pack": (
                    float(seg.max(axis=-1).mean()),
                    np.mean,
                ),
            },
            mode=rl_engine_lib.Mode.TRAIN,
            step=self.rl_engine.global_steps,
        )

      # Identity recovery: one identity list per queue item, dequeued in
      # lockstep with the batch generator's grouping (sub-batch + packing
      # is guarded out currently).
      if self._sb_enabled:
        row_identities = list(
            itertools.chain.from_iterable(
                self._sb_dequeue_identities(
                    1 if self._process_in_consumer else len(train_micro_batch)
                )
            )
        )
      else:
        row_identities = []

      # When ``train_micro_batch_size < mini_batch_size`` we want the trainer
      # to invoke ``train_step`` multiple times per outer iteration so the
      # optimizer (which fires every ``gradient_accumulation_steps`` micro-
      # steps) sees ``mini_batch_size``-shaped gradients while peak HBM is
      # only ``train_micro_batch_size``-shaped. Slice the merged train
      # example along its batch axis into chunks sized to one micro-step,
      # and pass the list to ``update_actor``; ``peft_trainer.train``
      # iterates the list and calls ``train_step`` once per chunk.
      is_packed = (
          hasattr(merged_train_micro_batch, "segment_ids")
          and getattr(merged_train_micro_batch, "segment_ids") is not None
      )
      n_total = merged_train_micro_batch.completion_ids.shape[0]
      if self._sb_enabled and len(row_identities) != n_total:
        raise RuntimeError(
            "Sub-batch identity tracking out of sync with the training"
            f" batch: {len(row_identities)} identities for {n_total} rows."
        )
      if not is_packed:
        seqs_per_chunk = (
            train_micro_batch_size * self.algo_config.num_generations
        )
        if n_total > seqs_per_chunk:
          chunked_train_micro_batch = [
              jax.tree_util.tree_map(
                  lambda x: (
                      x[i : i + seqs_per_chunk]
                      if hasattr(x, "shape") and x.shape and x.shape[0] == n_total
                      else x
                  ),
                  merged_train_micro_batch,
              )
              for i in range(0, n_total, seqs_per_chunk)
          ]
          # Mirrors the row slicing above exactly (same range/step), so
          # chunk_identities[c] names the (group_id, pair_index) pairs in
          # chunked_train_micro_batch[c].
          chunk_identities = [
              row_identities[i : i + seqs_per_chunk]
              for i in range(0, n_total, seqs_per_chunk)
          ]
        else:
          chunked_train_micro_batch = [merged_train_micro_batch]
          chunk_identities = [row_identities]
      else:
        chunked_train_micro_batch = [merged_train_micro_batch]
        chunk_identities = [row_identities]

      if self._sb_enabled:
        # Re-split any chunk that mixes rows needing different epochs (see
        # _sb_split_by_epoch) before it ever reaches full_batch_chunks, so
        # neither training loop below has to think about mixing at all.
        split_chunks = []
        split_identities = []
        for sub_chunk, ids in zip(chunked_train_micro_batch, chunk_identities):
          for s_chunk, s_ids in self._sb_split_by_epoch(sub_chunk, ids):
            split_chunks.append(s_chunk)
            split_identities.append(s_ids)
        chunked_train_micro_batch = split_chunks
        chunk_identities = split_identities

      full_batch_chunks.extend(chunked_train_micro_batch)
      full_batch_chunk_identities.extend(chunk_identities)

      # --- Evaluation Logic on FIRST microbatch ---
      current_eval_dataset = None
      if (
          sb_units_seen if self._sb_enabled else update_steps_since_last_sync
      ) == 0:
        current_train_step = self.rl_engine.actor_trainer.train_steps
        if (
            all_eval_prompts
            and current_train_step % training_config.eval_every_n_steps == 0
            and current_train_step != self._last_eval_train_step
        ):
          self._last_eval_train_step = current_train_step
          self._eval_iter_steps = 0
          eval_orchestrator = self._build_orchestrator()

          async def _eval_runner_async(current_eval_orchestrator):
            eval_examples = []
            async for batch in self._orchestrator_producer(
                current_eval_orchestrator,
                all_eval_prompts,
                num_generations=self._num_generations(),
            ):
              eval_example = self._batch_to_train_example(
                  batch,
                  rl_engine_lib.Mode.EVAL,
              )
              eval_examples.extend(eval_example)
            return eval_examples

          eval_future = asyncio.run_coroutine_threadsafe(
              _eval_runner_async(eval_orchestrator), self.loop
          )
          eval_examples = eval_future.result()
          self._eval_iter_steps += 1
          current_eval_dataset = eval_examples
          did_eval_this_global_step = True

      # --- First iteration Training Step (Parallelized with Rollout) ---
      # Note: Suppose one full batch has m minibatches, each minibatch has n
      # microbatches, and #iterations=K, we will:
      #   1. Train on the m * n microbatches once as we get them from rollout.
      #   2. When we get the full batch, repeat K-1 times on the entire batch.
      # Determine if this micro-batch is the last one for the current step's
      # first epoch. `mb_closes_step` is True if we have seen all requested
      # micro-batch units for the current global step.
      if self._sb_enabled:
        mb_closes_step = sb_units_seen + 1 == sb_units_per_step
        # Ensure eval datasets are delivered even if early chunks are count
        # skipped. The eval payload attaches to the FIRST chunk actually run
        # through the TPU in this epoch.
        if current_eval_dataset is not None:
          pending_eval_dataset = current_eval_dataset
        # To accurately set `step_complete=True` during snapshots, we must
        # attach it to the LAST chunk that actually runs through the TPU within
        # this micro-batch (and only if this micro-batch closes the step,
        # handled by `mb_closes_step`).
        # We cannot simply use `c == len(chunk_identities) - 1` because the
        # final chunk(s) might be fully trained (skipped here). If we attach
        # `step_complete` to a skipped chunk, no snapshot is written, breaking
        # resume logic.
        last_trainable = max(
            (
                c
                for c, cids in enumerate(chunk_identities)
                if self._sb_chunk_epoch(cids) < 1
            ),
            default=-1,
        )
        for c, sub_chunk in enumerate(chunked_train_micro_batch):
          ids = chunk_identities[c]
          if self._sb_chunk_epoch(ids) >= 1:
            # Already trained in epoch 1 pre-crash (a resumed,
            # re-injected chunk, or a fresh chunk that was already
            # processed before the crash); still counted into
            # full_batch_chunks above for the replay epochs below, just
            # not re-trained here.
            continue
          eval_ds = pending_eval_dataset
          pending_eval_dataset = None
          self.rl_engine.update_actor([sub_chunk], eval_ds, skip_jit)
          if hasattr(self.rl_engine, "critic_trainer"):
            self.rl_engine.update_critic([sub_chunk], eval_ds, skip_jit)
          self._sb_snapshot(
              ids,
              step_complete=(
                  self._num_iterations() == 1
                  and c == last_trainable
                  and mb_closes_step
              ),
          )
      else:
        self.rl_engine.update_actor(
            chunked_train_micro_batch, current_eval_dataset, skip_jit
        )
        if hasattr(self.rl_engine, "critic_trainer"):
          self.rl_engine.update_critic(
              chunked_train_micro_batch, current_eval_dataset, skip_jit
          )
        self._bench_stock_iter_hook(len(chunked_train_micro_batch))

      # --- Weight Sync Logic ---
      if self._sb_enabled:
        sb_units_seen += 1
      else:
        if is_packed:
          # `merged_train_micro_batch.is_update_step` is a size-1 jax array
          # set by `pack_sequences`; pull the host-side value.
          is_update = bool(
              np.asarray(merged_train_micro_batch.is_update_step).item()
          )
        else:
          # Mirror `peft_trainer._train_step`'s derivation:
          # `is_update_step` flips True every `grad_acc_steps` micro-batches.
          unpacked_micro_step_counter += 1
          is_update = unpacked_micro_step_counter % grad_acc_steps == 0

        if is_update:
          update_steps_since_last_sync += 1

      _step_boundary_reached = (
          sb_units_seen == sb_units_per_step
          if self._sb_enabled
          else update_steps_since_last_sync == update_steps_per_full_batch
      )
      if _step_boundary_reached:
        # --- Remaining Iterations Training Step ---
        iterations = self._num_iterations()

        for i in range(1, iterations):
          # TODO(b/483779605) Sub-step checkpointing.
          self._iter_steps += len(full_batch_chunks)

          # TODO(yixuanm): Eval during iteration too. Skipping for now as we
          # will refactor the learner soon.
          if self._sb_enabled:
            # To accurately set `step_complete=True` during snapshots, we must
            # attach it to the LAST chunk that actually runs through the TPU in
            # this mini-batch for the current epoch `i+1`. We cannot simply use
            # `c == len(full_batch_chunk_identities) - 1` because the final
            # chunk(s) might be fully trained (skipped here). If we attach
            # `step_complete` to a skipped chunk, no snapshot is written,
            # breaking resume logic.
            last_trainable = max(
                (
                    c
                    for c, cids in enumerate(full_batch_chunk_identities)
                    if self._sb_chunk_epoch(cids) < i + 1
                ),
                default=-1,
            )
            for c, sub_chunk in enumerate(full_batch_chunks):
              ids = full_batch_chunk_identities[c]
              if self._sb_chunk_epoch(ids) >= i + 1:
                # This chunk was already trained for epoch `i+1` pre-crash.
                continue

              # If a resume count-skips every epoch 1 chunk entirely, the step's
              # eval dataset might be undelivered. Ensure it rides the first
              # chunk actually trained in the remaining epochs instead of being dropped.
              eval_ds = pending_eval_dataset
              pending_eval_dataset = None
              self.rl_engine.update_actor([sub_chunk], eval_ds, skip_jit)
              if hasattr(self.rl_engine, "critic_trainer"):
                self.rl_engine.update_critic([sub_chunk], eval_ds, skip_jit)
              self._sb_snapshot(
                  ids,
                  step_complete=(i == iterations - 1 and c == last_trainable),
              )
          else:
            self.rl_engine.update_actor(full_batch_chunks, None, skip_jit)
            if hasattr(self.rl_engine, "critic_trainer"):
              self.rl_engine.update_critic(full_batch_chunks, None, skip_jit)
        full_batch_chunks.clear()
        full_batch_chunk_identities.clear()
        self._sb_step_boundary()
        # A seeded resume always has at least one untrained (chunk, epoch)
        # (else the snapshot would have been step_complete and never seeded),
        # so the pending eval must have been delivered by now; cleared anyway
        # so a violated invariant can never mislabel eval across steps.
        pending_eval_dataset = None

        global_step_time = time.time() - self._global_step_start_time
        logging.info(
            f"Global step {self.rl_engine.global_steps} completed in"
            f" {global_step_time:.2f} seconds."
        )
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
          actor_trainer = self.rl_engine.actor_trainer
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
            " adv_abs_mean=%.3f compl_len=%.1f time=%.1fs%s%s",
            self.rl_engine.global_steps,
            train_r_mean,
            train_solve,
            int(train_rewards.size),
            adv_abs_mean,
            float(compl_len),
            global_step_time,
            trainer_str,
            eval_str,
        )
        did_eval_this_global_step = False
        self.rl_engine.buffer_metrics_async(
            {"perf/global_step_time": (global_step_time, np.mean)},
            mode=rl_engine_lib.Mode.TRAIN,
            step=self.rl_engine.global_steps,
        )
        if self.should_sync_weights:
          logging.info("Requesting sync lock to sync weights...")
          self._rollout_sync_lock.acquire_weight_sync()
          try:
            logging.info("Sync lock acquired. Syncing weights.")
            with self.rl_engine.perf_v2.span(
                perf_constants.WEIGHT_SYNC,
                self.rl_engine.perf_v2.all_devices,
                tags={
                    perf_constants.STEP: self.rl_engine.global_steps,
                },
            ):
              self.rl_engine.sync_weights()
            self.policy_version += 1
            logging.info(
                "Weights synced. Policy version incremented to %d.",
                self.policy_version,
            )
            try:
              with self.rl_engine.perf_v2.span(
                  perf_constants.DATA_LOADING,
                  tags={
                      perf_constants.STEP: self.rl_engine.global_steps,
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
          self.rl_engine.global_steps += 1
          try:
            with self.rl_engine.perf_v2.span(
                perf_constants.DATA_LOADING,
                tags={
                    perf_constants.STEP: self.rl_engine.global_steps,
                },
            ):
              batch = next(full_dataset_iterator)
            self._put_prompts_to_queue(prompt_queue, batch)
          except StopIteration:
            prompt_queue.put(None)

        self.rl_engine.buffer_metrics(
            self.rl_engine.perf_v2.export(),
            mode=rl_engine_lib.Mode.TRAIN,
        )
        update_steps_since_last_sync = 0
        sb_units_seen = 0
        did_eval_this_global_step = False
        self._global_step_start_time = time.time()

      # BENCHMARK instrumentation (temporary): per-micro-batch progress vs
      # cumulative wall-clock, continuous across restarts via the pinned
      # anchor. Aggregated np.max per global step, so each step's point is
      # its latest reading and the curve is monotone; plotting
      # progress_micro_steps against cumulative_time gives the recovery
      # profile (design doc section 11).
      if self._bench_metrics:
        self.rl_engine.buffer_metrics_async(
            {
                "perf/sub_batch_cumulative_time": (
                    time.time() - self._experiment_start_time,
                    np.max,
                ),
                "perf/sub_batch_progress_micro_steps": (
                    self._iter_steps,
                    np.max,
                ),
            },
            mode=rl_engine_lib.Mode.TRAIN,
            step=self.rl_engine.global_steps,
        )

    _ = producer_future.result()
    self.rl_engine.close()

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
        and self.rl_engine.global_steps >= self._training_config.max_steps
    ):
      logging.info(
          "Reached max_steps: %d >= %d",
          self.rl_engine.global_steps,
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
