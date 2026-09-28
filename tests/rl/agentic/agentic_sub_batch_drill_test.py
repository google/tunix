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

"""CPU preemption drill: drives the REAL AgenticRLLearner.train() end to end
with a fake-but-contract-faithful cluster, injects crashes at controlled
points, restarts the whole stack from real checkpoints, and asserts the
exactly-mu contract across preemptions.

This is the layer the unit and hook-level suites cannot cover: the actual
producer/consumer loop, the real orchestrator, the real chunking + skip +
snapshot integration, and the real trainer-restore reconciliation. It is the
CPU stand-in for a TPU preemption drill.

Fidelity notes (what is real vs faked):
- REAL: AgenticRLLearner.train(), _producer, _orchestrator_producer,
  RolloutOrchestrator + GroupQueueManager, SimpleDataQueue, every _sb_* hook,
  SubBatchCheckpointManager against a real Orbax tmpdir, the trainer's weight
  checkpoint via the real sft CheckpointManager (so restore, including
  fix_sharding, runs the production code), a real nnx.Optimizer driven from the
  real GradientAccumulator with data-dependent gradients.
- FAKED: the rollout engine (deterministic Token-mode dict trajectories),
  reward/logp processing (_process_results builds real TrainExamples from the
  fake trajectories), metrics/perf plumbing (recorders), colocated weights
  (should_sync_weights False, so no rollout engine to resync).
"""

import asyncio
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
import concurrent.futures.thread as _futures_thread
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl.testing import absltest
from orbax.checkpoint import v1 as ocp
from tunix.rl.agentic import agentic_rl_learner
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.pipeline import rollout_orchestrator
from tunix.sft import checkpoint_manager as sft_checkpoint_manager
from tunix.sft import checkpoint_options
from tunix.sft import peft_trainer


# Upstream head's CheckpointManager may call orbax APIs the locally installed
# orbax build does not have yet (use_persistence_array_handler). The drill is
# the only suite that constructs the real sft CheckpointManager; skip it
# loudly instead of failing every scenario on a dependency skew that has
# nothing to do with sub-batch checkpointing.
def _probe_checkpoint_manager():
  import tempfile as _tf, shutil as _sh, unittest as _ut
  d = _tf.mkdtemp()
  try:
    sft_checkpoint_manager.CheckpointManager(d).close()
  except TypeError as e:
    raise _ut.SkipTest(
        "sft CheckpointManager is incompatible with the installed orbax"
        f" ({e}); reconcile the orbax checkout with upstream head to run"
        " the preemption drill."
    )
  finally:
    _sh.rmtree(d, ignore_errors=True)

# _probe_checkpoint_manager() runs in SubBatchPreemptionDrillTest.setUpClass

G = 2  # num_generations
SEQ = 4  # completion length (also the tiny model's feature dim)


class _CrashError(RuntimeError):
  """Injected preemption."""


class _TinyModel(nnx.Module):

  def __init__(self, rngs):
    self.w = nnx.Param(jnp.zeros((SEQ,)))

  def __call__(self, x):
    return jnp.sum(self.w * x)


def _loss_fn(model, x, target):
  return (model(x) - target) ** 2


class _FakeTrainer:
  """peft_trainer contract subset the learner + drill cluster rely on:
  counters, the REAL GradientAccumulator (add/get/update/reset cycle exactly
  as peft_trainer._train_step drives it -- the accumulator is NOT part of
  the trainer checkpoint, which is precisely the gap sub-batch snapshots
  cover), per-apply weight checkpointing through the REAL sft
  CheckpointManager, and restore via the REAL maybe_restore (exercising
  fix_sharding on the optimizer state)."""

  def __init__(self, ckpt_dir: str, k: int, cluster_ref, mesh):
    self._k = k
    self._cluster_ref = cluster_ref
    self.model = _TinyModel(nnx.Rngs(0))
    self.optimizer = nnx.Optimizer(self.model, optax.adam(0.1), wrt=nnx.Param)
    # SYNCHRONOUS trainer saves, deliberately: the drill models preemption
    # by abandoning a learner in-process, but real preemption kills the
    # async commit thread with the process. In-process those threads live
    # on, and an abandoned commit racing the restarted trainer's re-save of
    # the same step trips orbax >= 0.12.1's stricter ArrayMetadata
    # validation -- a harness artifact no real restart can produce. Sync
    # saves are the faithful in-process approximation for the TRAINER
    # stream; the sub-batch manager stays async (that is product surface).
    # Per-apply saves (the cadence check reads the resolved policy) and the
    # DEFAULT LatestN(3) retention: at 4 applies per step it would evict the
    # step-start checkpoint, which the learner's pin keeps.
    self.checkpoint_manager = sft_checkpoint_manager.CheckpointManager(
        ckpt_dir,
        options=checkpoint_options.TunixCheckpointingOptions(
            save_decision_policy=(
                ocp.training.save_decision_policies.FixedIntervalPolicy(1)
            ),
            enable_async_checkpointing=False,
        ),
    )
    self._train_steps, self._restored_custom_metadata = (
        self.checkpoint_manager.maybe_restore(self.model, self.optimizer)
    )
    # Like a production model, the weights live on the ACTOR mesh (the
    # injected sub-batch buffer is device_put onto that mesh; grads must
    # match it). The accumulator derives its buffers from the placed model.
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    for module in (self.model, self.optimizer):
      nnx.update(
          module,
          jax.tree.map(
              lambda a: jax.device_put(a, replicated), nnx.state(module)
          ),
      )
    self.grad_accumulator = peft_trainer.GradientAccumulator(
        self.model, nnx.Param
    )
    self._iter_steps = self._train_steps * k
    self.is_managed_externally = True

  @property
  def train_steps(self):
    return self._train_steps

  @property
  def iter_steps(self):
    return self._iter_steps

  def restored_global_step(self):
    return self._restored_custom_metadata.get("global_step", 0)

  def train(self, train_ds, eval_ds=None, skip_jit=False):
    """Mimics peft_trainer.train's per-chunk loop: one real gradient
    micro-step per list item, apply + weight checkpoint at every k-th."""
    del eval_ds, skip_jit
    grad_fn = nnx.value_and_grad(_loss_fn, argnums=0)
    for chunk in train_ds:
      # Data-dependent gradient so resumed-buffer correctness is observable;
      # reduced to a scalar first so the tiny model is width-agnostic
      # (standard chunks are SEQ wide, packs are budget wide).
      xval = float(np.asarray(chunk.completion_ids, dtype=np.float32).mean())
      x = jnp.full((SEQ,), xval, dtype=jnp.float32)
      target = float(np.asarray(chunk.advantages, dtype=np.float32).mean())
      _, grads = grad_fn(self.model, x, target)
      self.grad_accumulator.add(grads)
      self._iter_steps += 1
      # peft_trainer precedence: the example's is_update_step flag when
      # present (set by pack_sequences), the iter % k derivation otherwise.
      flag = getattr(chunk, "is_update_step", None)
      if flag is not None:
        do_apply = bool(np.asarray(flag).item())
      else:
        do_apply = self._iter_steps % self._k == 0
      if do_apply:
        self.optimizer.update(self.model, self.grad_accumulator.get())
        self.grad_accumulator.reset()
        self._train_steps += 1
        self.checkpoint_manager.save(
            self._train_steps,
            self.model,
            self.optimizer,
            force=True,
            custom_metadata={
                "global_step": self._cluster_ref().global_steps + 1
            },
        )


class _FakeClusterConfig(SimpleNamespace):
  pass


class _NoopSpan:

  def __enter__(self):
    return self

  def __exit__(self, *a):
    return False

  def async_end(self, *a, **k):
    pass


class _FakePerf:

  def span(self, *a, **k):
    return _NoopSpan()

  def export(self):
    return {}

  @property
  def all_devices(self):
    return []


class _FakeCluster:
  """The rl_cluster surface train() + the _sb_* hooks touch, colocated
  variant (should_sync_weights False upstream of this object)."""

  eval_log = []  # (run, chunk index) per eval delivery, class-level

  def __init__(self, training_config, actor_ckpt_dir, k, crash_plan):
    self.perf_v2 = _FakePerf()
    # A real mesh: the resume places the injected buffer on the ACTOR mesh.
    from tunix.rl import rl_cluster as rl_cluster_lib  # local: heavy import
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]), axis_names=("fsdp",)
    )
    self.cluster_config = _FakeClusterConfig(
        training_config=training_config,
        offload_to_cpu=False,
        role_to_mesh={rl_cluster_lib.Role.ACTOR: mesh},
    )
    self.actor_trainer = _FakeTrainer(actor_ckpt_dir, k, lambda: self, mesh)
    self.global_steps = self.actor_trainer.restored_global_step()
    self._crash_plan = crash_plan  # dict: micro_update_index -> "before"|"after"
    self.micro_updates = 0  # trained chunks across the run (this process)
    self.run_idx = 0  # set by _build_learner; stamps eval deliveries
    self.closed = False

  def update_actor(self, train_ds, eval_ds, skip_jit=False):
    if eval_ds:
      # (run, chunk index within the run) of each non-empty eval delivery.
      type(self).eval_log.append((self.run_idx, self.micro_updates))
    for chunk in train_ds:
      if self._crash_plan.get(self.micro_updates) == "before":
        raise _CrashError(f"injected before micro-update {self.micro_updates}")
      self.actor_trainer.train([chunk], eval_ds, skip_jit)
      self.micro_updates += 1
      if self._crash_plan.get(self.micro_updates - 1) == "between":
        # The trainer's apply checkpoint (synchronous here) is durable; the
        # post-train snapshot has not been written. For an apply chunk this
        # is exactly the commit-order race the pre-commit snapshot closes.
        raise _CrashError(
            "injected between train and snapshot of micro-update"
            f" {self.micro_updates - 1}"
        )
    # "after"-crashes are injected post-snapshot via the learner subclass.

  def buffer_metrics(self, *a, **k):
    pass

  def buffer_metrics_async(self, *a, **k):
    pass

  def close(self):
    self.closed = True


class _FakeEngine:
  """Deterministic Token-mode trajectory collector: same prompt -> same
  tokens, always. Records generation counts for regeneration assertions."""

  generation_log = []  # (group_id, pair_index), class-level across restarts

  def __init__(self, agent, env, **kwargs):
    self._env = env

  async def collect(self, mode=None):
    gid = self._env.extra_kwargs["group_id"]
    pair = self._env.extra_kwargs["pair_index"]
    type(self).generation_log.append((gid, pair))
    base = (gid * 31 + pair * 7) % 97
    # Variable REAL completion length (via the mask; unpad_train_example
    # strips by mask sums), so sequence packing is non-degenerate: packs
    # hold different item counts and windows are genuinely ragged.
    real_len = 1 + (gid * 7 + pair * 3) % SEQ
    mask = np.zeros(SEQ, dtype=np.int32)
    mask[:real_len] = 1
    return {
        "conversation_text": [
            {"role": "assistant", "content": f"resp-{gid}-{pair}"}
        ],
        "prompt_tokens": np.arange(SEQ) + base,
        "conversation_tokens": (np.arange(SEQ) + base) % 13,
        "conversation_masks": mask,
        "old_logprobs": np.zeros(SEQ, dtype=np.float32),
        "trajectory_reward": float(gid % 5) / 5.0,
        "policy_version": 0,
        "original_input": {"prompts": self._env.task["prompts"]},
        "group_id": gid,
    }


class _DrillLearner(agentic_rl_learner.AgenticRLLearner):
  """Concrete learner: deterministic processing of fake trajectories into
  real TrainExamples, fake agent/env pairs, fake engine, crash-after hook."""

  # (run_index, post_update_iter, group_id, pair_index) per trained chunk.
  trained_log = []
  # Element r is the iter_steps run r resumed at (0 for a fresh start).
  run_starts = []

  def _process_results(self, trajectories, mode=None, expected_step=None):
    completion_ids = np.stack(
        [np.asarray(t.traj["conversation_tokens"]) for t in trajectories]
    )
    masks = np.stack(
        [np.asarray(t.traj["conversation_masks"]) for t in trajectories]
    )
    rewards = np.asarray(
        [t.traj["trajectory_reward"] for t in trajectories], dtype=np.float32
    )
    adv = np.repeat(
        (rewards - rewards.mean())[:, None], SEQ, axis=1
    ).astype(np.float32)
    return [
        agentic_rl_learner.TrainExample(
            prompt_ids=jnp.asarray(
                np.stack(
                    [np.asarray(t.traj["prompt_tokens"]) for t in trajectories]
                )
            ),
            # Real mask, not None: the packer's unpad strips by mask sums.
            prompt_mask=jnp.ones((len(trajectories), SEQ), dtype=jnp.int32),
            completion_ids=jnp.asarray(completion_ids),
            completion_mask=jnp.asarray(masks),
            advantages=jnp.asarray(adv),
            ref_per_token_logps=None,
            old_per_token_logps=None,
            policy_version=None,
        )
    ]

  def _create_agent_env_pair(self, single_example, group_id, pair_index):
    env = SimpleNamespace(
        extra_kwargs={"group_id": group_id, "pair_index": pair_index},
        task={"prompts": str(single_example.get("prompts"))},
    )
    return None, env

  def _build_orchestrator(self):
    return rollout_orchestrator.RolloutOrchestrator(
        engine_cls=_FakeEngine,
        engine_kwargs={},
        max_concurrency=4,
        rollout_sync_lock=self._rollout_sync_lock,
    )

  def _sb_snapshot(self, identities, *, step_complete, anticipated_apply=False):
    super()._sb_snapshot(
        identities,
        step_complete=step_complete,
        anticipated_apply=anticipated_apply,
    )
    # Apply chunks snapshot BEFORE training: record their post-training
    # counter so the lineage rule in _assert_contract sees the same value a
    # post-train snapshot would carry.
    post_iter = self.rl_engine.actor_trainer.iter_steps + (
        1 if anticipated_apply else 0
    )
    type(self).trained_log.extend(
        (self._drill_run, post_iter, gid, pair) for gid, pair in identities
    )
    plan = self.rl_engine._crash_plan
    # Index of the chunk this snapshot belongs to: about to train (apply
    # chunk) or just trained (mid-window chunk).
    idx = self.rl_engine.micro_updates - (0 if anticipated_apply else 1)
    if plan.get(idx) == "after":
      raise _CrashError(f"injected after snapshot of micro-update {idx}")


def _make_training_config(root, *, mini, micro, max_steps, eval_every=10_000):
  k = mini // micro

  def get_with_default(key, default):
    if key == "gradient_accumulation_steps":
      return k
    return default

  return SimpleNamespace(
      max_steps=max_steps,
      mini_batch_size=mini,
      train_micro_batch_size=micro,
      rollout_micro_batch_size=None,
      compute_logps_micro_batch_size=None,
      max_inflight_computations=2,
      max_seq_token_per_tpu=None,
      eval_every_n_steps=eval_every,
      checkpoint_root_directory=root,
      checkpointing_options=SimpleNamespace(
          save_decision_policy=SimpleNamespace(interval=1),
          # Sync orbax I/O throughout the drill: ~20 manager lifecycles per
          # process trip orbax >= 0.12.1's process-global signaling registry
          # under async (barrier timeouts no production topology can see).
          # Async internals are manager-unit-tested; the drill's job is loop
          # semantics.
          enable_async_checkpointing=False,
      ),
      get_with_default=get_with_default,
  )


class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
  """Executor whose workers are daemon threads, excluded from the exit join.

  Why: a preempted learner's producer stays parked in prompt_queue.get()
  inside an executor worker forever -- production relies on the scheduler's
  SIGKILL to reap the process. Since Python 3.9 executor workers are
  non-daemon and joined at interpreter exit, so without this the drill
  prints PASS and then the pytest process never exits.
  """

  def _adjust_thread_count(self):
    real_thread = threading.Thread

    class _DaemonThread(real_thread):

      def start(self):
        self.daemon = True
        real_thread.start(self)

    # ThreadPoolExecutor hardcodes threading.Thread; patch it for the tiny
    # window in which the worker is constructed (held under the executor's
    # internal lock, and any Thread daemonized by accident is test-local).
    threading.Thread = _DaemonThread
    try:
      super()._adjust_thread_count()
    finally:
      threading.Thread = real_thread
    # _python_exit() joins every registered worker at exit; a worker blocked
    # inside a work item never returns, so deregister ours.
    for t in self._threads:
      _futures_thread._threads_queues.pop(t, None)


def _build_learner(root, *, fbs, mini, micro, mu, max_steps, crash_plan,
                   eval_every=10_000, process_in_consumer=False):
  """Builds a drill learner around the real train() dependencies, bypassing
  AgenticRLLearner.__init__ (which needs the full production cluster) but
  replicating every attribute train() and the _sb_* hooks read, in the same
  order __init__ establishes them (sub-batch init AFTER the trainer restore,
  BEFORE the loop thread)."""
  k = mini // micro
  training_config = _make_training_config(root, mini=mini, micro=micro,
                                          max_steps=max_steps,
                                          eval_every=eval_every)
  cluster = _FakeCluster(training_config, f"{root}/actor", k, crash_plan)

  learner = _DrillLearner.__new__(_DrillLearner)
  learner.rl_engine = cluster
  learner.algo_config = SimpleNamespace(
      num_generations=G,
      num_iterations=mu,
      off_policy_steps=0,
      max_concurrency=4,
      sub_batch_checkpointing=True,
  )
  learner._training_config = training_config
  # compute_logps_micro_batch_size == train_micro_batch_size > 1 is what
  # makes train() defer conversion to the consumer (raw groups on the queue).
  learner._compute_logps_micro_batch_size = micro if process_in_consumer else 1
  learner._rollout_micro_batch_size = 1
  learner._process_in_consumer = False
  learner._full_batch_size = 0
  learner.should_sync_weights = False
  learner._background_tasks = set()
  learner._rollout_sync_lock = agentic_utils.RolloutSyncLock()
  learner._eval_iter_steps = 0
  learner._last_eval_train_step = -1
  learner._train_rewards_window = []
  learner._eval_rewards_window = []
  learner._rewards_window_lock = threading.Lock()
  learner._global_step_start_time = time.time()

  # Sub-batch state + resume decision, mirroring __init__ ordering.
  learner._sb_mgr = None
  learner._sb_lock = threading.Lock()
  learner._sb_completed = set()
  learner._sb_counts = {}
  learner._sb_active = []
  learner._sb_pending_state = None
  learner._sb_last_snapshot_key = -1
  learner._sb_window_train_steps = -1
  learner._sb_local = -1
  learner._sb_last_snapshot_trainer_iter = -1
  learner._sb_active_gids = set()
  learner._sb_step_complete_for_step = None
  learner._sb_restored_trainer_state = False
  learner._sb_restored_full_batch_size = None
  learner._sb_resumed_mid_step = False
  learner._init_sub_batch_checkpointing()

  learner._iter_steps = cluster.actor_trainer.iter_steps

  # Lineage bookkeeping for the exactly-mu contract: record where this run
  # resumed (trainer restore + sub-batch snapshot fixup combined).
  learner._drill_run = len(_DrillLearner.run_starts)
  cluster.run_idx = learner._drill_run
  _DrillLearner.run_starts.append(learner._iter_steps)

  loop_queue = queue.Queue()

  def run_loop_forever():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_default_executor(_DaemonThreadPoolExecutor(max_workers=5))
    loop_queue.put(loop)
    loop.run_forever()

  threading.Thread(target=run_loop_forever, daemon=True).start()
  learner.loop = loop_queue.get()
  return learner


def _dataset(fbs, n_batches):
  return [
      {"prompts": [f"p{b}-{i}" for i in range(fbs)]} for b in range(n_batches)
  ]


def _run_until_crash_or_done(learner, dataset, timeout=None, eval_dataset=None):
  """Runs the real train() on a watchdog thread. Returns 'done', 'crash', or
  raises on hang/unexpected error."""
  timeout = timeout or float(os.environ.get("DRILL_TIMEOUT", 5))
  result = {}

  def _target():
    try:
      learner.train(iter(dataset), eval_dataset=eval_dataset)
      result["outcome"] = "done"
    except _CrashError as e:
      result["outcome"] = "crash"
      result["error"] = e
    except BaseException as e:  # pylint: disable=broad-except
      result["outcome"] = "error"
      result["error"] = e

  t = threading.Thread(target=_target, daemon=True)
  t.start()
  t.join(timeout)
  if t.is_alive():
    import faulthandler
    faulthandler.dump_traceback()
    raise AssertionError(
        "train() hung (deadlock): trained so far ="
        f" {learner.rl_engine.micro_updates}, global_steps ="
        f" {learner.rl_engine.global_steps}"
    )
  if result["outcome"] == "error":
    raise result["error"]
  return result["outcome"]


class _ErrorCapture(logging.Handler):
  """Collects ERROR records emitted anywhere during a drill run.

  The learner already prints the precise phantom-step diagnostic when a step
  reaches its boundary unmarked; without capturing it the loudest available
  signal is thrown away and the drill can only notice the damage indirectly
  (and, as measured, usually not at all).
  """

  def __init__(self):
    super().__init__(level=logging.ERROR)
    self.messages = []

  def emit(self, record):
    self.messages.append(record.getMessage())


class SubBatchPreemptionDrillTest(absltest.TestCase):
  """The contract assertions, per scenario: exactly-mu trained epochs per
  (group, pair); no deadlock; global_steps advances by exactly max_steps with
  no phantom steps; each dataset batch consumed once."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    _probe_checkpoint_manager()

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    _FakeEngine.generation_log = []
    _FakeCluster.eval_log = []
    _DrillLearner.trained_log = []
    _DrillLearner.run_starts = []

  def tearDown(self):
    # Crashed learners are abandoned mid-write on purpose (the drill models a
    # SIGKILL), so their async Orbax writer can still be creating files under
    # self.root while this runs -- rmtree would then race and raise. Every
    # assertion has already run by here, so cleanup failures are noise.
    shutil.rmtree(self.root, ignore_errors=True)
    super().tearDown()

  def _assert_contract(self, *, fbs, mu, max_steps):
    """Exactly-mu over the SURVIVING lineage, not raw update calls.

    Work from run r whose post-update iter exceeds run r+1's resume point was
    rolled back by that restart (e.g. the T=0 fresh-run guard purging
    pre-first-checkpoint accumulation) — retraining it is correct recovery,
    not over-training, so those entries must not count toward mu.
    """
    starts = _DrillLearner.run_starts
    if os.environ.get("DRILL_DUMP"):
      print("\nrun_starts:", starts)
      for run, post_iter, gid, pair in _DrillLearner.trained_log:
        surv = run == len(starts) - 1 or post_iter <= starts[run + 1]
        print(f"  run={run} post={post_iter} gid={gid} pair={pair}"
              f" {'SURVIVES' if surv else 'DISCARDED'}")
    counts = {}
    for run, post_iter, gid, pair in _DrillLearner.trained_log:
      if run < len(starts) - 1 and post_iter > starts[run + 1]:
        continue  # discarded by the successor's rollback
      counts[(gid, pair)] = counts.get((gid, pair), 0) + 1
    expected_pairs = {
        (step * fbs + i, p)
        for step in range(max_steps)
        for i in range(fbs)
        for p in range(G)
    }
    self.assertEqual(set(counts), expected_pairs, "trained pair set mismatch")
    over = {k: v for k, v in counts.items() if v != mu}
    self.assertEqual(over, {}, f"exactly-mu violated (mu={mu}): {over}")

  def _drill(self, *, fbs, mini, micro, mu, max_steps, crash_plans,
             n_batches=None, eval_every=10_000, eval_dataset=None,
             process_in_consumer=False):
    """Runs a chain of (re)starts sharing the checkpoint root. crash_plans is
    a list: element i is the crash plan for process i (the last one should be
    empty so the run completes). Returns the learners, one per process."""
    dataset = _dataset(fbs, n_batches or max_steps + 2)
    outcome = None
    learners = []
    capture = _ErrorCapture()
    root_logger = logging.getLogger()
    root_logger.addHandler(capture)
    try:
      for i, plan in enumerate(crash_plans):
        learner = _build_learner(
            self.root, fbs=fbs, mini=mini,
            micro=micro, mu=mu,
            max_steps=max_steps, crash_plan=plan, eval_every=eval_every,
            process_in_consumer=process_in_consumer,
        )
        learners.append(learner)
        outcome = _run_until_crash_or_done(
            learner, dataset, eval_dataset=eval_dataset
        )
        if outcome == "done":
          break
    finally:
      root_logger.removeHandler(capture)
    learner = learners[-1]
    self.assertEqual(outcome, "done", "run never completed")
    self._assert_contract(fbs=fbs, mu=mu, max_steps=max_steps)

    # The exactly-mu multiset alone cannot see a phantom step: re-feeding an
    # already-trained ledger trains NOTHING, so the trained-pair counts stay
    # perfect while global_steps drifts, a spurious weight sync fires, and a
    # dataset batch is silently skipped. These two assertions are what make
    # the step_complete wiring in train() actually covered -- without them,
    # hard-coding step_complete=False passes the whole drill.
    phantom = [m for m in capture.messages if "step_complete" in m]
    self.assertEqual(
        phantom, [],
        "a step reached its boundary unmarked (phantom step on next restart)",
    )
    self.assertEqual(
        learner.rl_engine.global_steps, max_steps,
        "global_steps drifted from the number of real steps (phantom step)",
    )
    # Proves train() actually reaches _sb_step_boundary. Without this the
    # call site is deletable with every test green: group ids are namespaced
    # per step, so a ledger that is never cleared still yields exactly-mu.
    # In production it would grow by full_batch_size*num_generations items per
    # step, and _sb_snapshot rescans the whole active list under the lock on
    # every micro-step -- quadratic host work plus unbounded memory.
    self.assertEqual(
        learner._sb_counts, {},
        "per-pair counts were never cleared: the step boundary did not run",
    )
    self.assertEqual(
        [t for t in learner._sb_active
         if learner._sb_item_step(t) <= max_steps - 1], [],
        "finished steps' trajectories were never dropped from the active set",
    )
    return learners

  def test_baseline_no_crash(self):
    """Harness sanity: the real train() completes cleanly with the feature
    enabled and the contract holds with zero preemptions."""
    self._drill(
        fbs=4, mini=2, micro=2, mu=2, max_steps=2, crash_plans=[{}]
    )

  def test_crash_mid_streaming_epoch(self):
    self._drill(
        fbs=4, mini=2, micro=2, mu=2, max_steps=2,
        crash_plans=[{1: "after"}, {}],
    )

  def test_crash_before_first_snapshot(self):
    """Crash between the optimizer update and the snapshot: under-count is
    corrected by retraining (count-after-update)."""
    self._drill(
        fbs=4, mini=2, micro=2, mu=2, max_steps=2,
        crash_plans=[{0: "before"}, {}],
    )

  def test_crash_mid_replay_sweep(self):
    # Step layout (fbs=4, micro=2, mu=2): micro-updates 0,1 = streaming,
    # 2,3 = replay sweep. Crash inside the sweep.
    self._drill(
        fbs=4, mini=2, micro=2, mu=2, max_steps=2,
        crash_plans=[{2: "after"}, {}],
    )

  def test_crash_after_step_complete_snapshot(self):
    """The phantom-step scenario: the step's final chunk trained and applied
    (its stamped snapshot was pre-committed), the crash lands before the
    boundary/global_steps increment finishes the step in memory."""
    self._drill(
        fbs=4, mini=2, micro=2, mu=2, max_steps=2,
        crash_plans=[{3: "between"}, {}],
    )

  def test_repeated_preemptions(self):
    """Three chained preemptions across two global steps."""
    self._drill(
        fbs=4, mini=2, micro=2, mu=2, max_steps=2,
        crash_plans=[{0: "after"}, {2: "before"}, {5: "after"}, {}],
    )

  def test_eval_runs_once_per_step_across_a_mid_step_resume(self):
    """A step's eval runs at its first micro-batch, before any training, so
    a mid-step resume implies it already ran. The resumed process must not
    run it again (that would evaluate one step twice, the second time on
    mid-step weights), while the NEXT step's eval fires normally."""
    self._drill(
        fbs=4, mini=4, micro=1, mu=1, max_steps=2,
        crash_plans=[{1: "after"}, {}],
        eval_every=1,
        eval_dataset=[{"prompts": [f"eval-{i}" for i in range(2)]}],
    )
    deliveries = _FakeCluster.eval_log
    self.assertEqual(_DrillLearner.run_starts, [0, 2])
    self.assertEqual([d for d in deliveries if d[0] == 0], [(0, 0)])
    self.assertNotIn((1, 0), deliveries)  # the resumed step is not re-evaled
    self.assertEqual(len(deliveries), 2)  # exactly one eval per step

  def test_eval_reruns_when_a_step_restarts_from_scratch(self):
    """A crash before the step's first snapshot loses the whole step; the
    restart redoes it from scratch, eval included. The duplicate eval is the
    cost of losing the step, not a resume defect."""
    self._drill(
        fbs=4, mini=4, micro=1, mu=1, max_steps=2,
        crash_plans=[{0: "before"}, {}],
        eval_every=1,
        eval_dataset=[{"prompts": [f"eval-{i}" for i in range(2)]}],
    )
    deliveries = _FakeCluster.eval_log
    self.assertIn((0, 0), deliveries)
    self.assertIn((1, 0), deliveries)  # step 0 redone from scratch
    self.assertEqual(len(deliveries), 3)

  def test_std_micro4(self):
    """train_micro_batch_size (4) above the production default
    max_inflight_computations (2): the consumer needs four groups before it
    reads any identity, so identity has to ride the data item."""
    self._drill(fbs=4, mini=4, micro=4, mu=1, max_steps=2, crash_plans=[{}])

  def test_consumer_processing_mid_step_resume(self):
    """process_in_consumer=True: raw groups on the queue, identity derived
    in the consumer's _to_train_examples wrapper -- for the reinjected
    groups of a mid-step resume as well as fresh ones. k=2, so the crash
    lands after the anticipated snapshot of the first apply: the restart
    resumes at iter 1 and trains only the three remaining chunks (same
    lineage as the producer-processing path on this geometry)."""
    learners = self._drill(
        fbs=4, mini=4, micro=2, mu=1, max_steps=2,
        crash_plans=[{1: "after"}, {}], process_in_consumer=True,
    )
    self.assertTrue(all(l._process_in_consumer for l in learners))
    self.assertEqual(_DrillLearner.run_starts, [0, 1])
    self.assertEqual([l.rl_engine.micro_updates for l in learners], [1, 3])

  def test_std_micro1_eval_no_crash(self):
    """Eval runs on the shared event loop while the consumer blocks on it:
    with the PR's bounded train_data_queue (fbs > micro + max_inflight) the
    producer parked the loop thread in put and the eval never ran."""
    self._drill(fbs=4, mini=4, micro=1, mu=1, max_steps=2, crash_plans=[{}],
                eval_every=1,
                eval_dataset=[{"prompts": [f"eval-{i}" for i in range(2)]}])
    self.assertEqual(len(_FakeCluster.eval_log), 2)  # one eval per step

  def test_std_micro2_fbs8_eval_no_crash(self):
    self._drill(fbs=8, mini=4, micro=2, mu=1, max_steps=2, crash_plans=[{}],
                eval_every=1,
                eval_dataset=[{"prompts": [f"eval-{i}" for i in range(2)]}])
    self.assertEqual(len(_FakeCluster.eval_log), 2)

  def test_grad_accum_mid_window_crash(self):
    """k=2, two chained crashes covering both mid-window cases.

    Crash 1 ({0: "after"}): mid-window in window 0, before the first apply.
    The fresh run anchored the initial weights at step 0, so the restart
    restores that anchor, selects the window-0 snapshot and resumes at
    iter 1 -- it must NOT retrain the window.
    Crash 2 ({1: "after"}, indexed within the RESTARTED process, which
    resumed at iter 1): its micro-update 1 is iter 3, mid-window WITH a
    per-apply trainer checkpoint (T=1). The restart must inject the
    snapshotted grad-accum buffer, skip the trained chunk, and finish the
    window without retraining.
    """
    self._drill(
        fbs=4, mini=4, micro=2, mu=1, max_steps=2,
        crash_plans=[{0: "after"}, {1: "after"}, {}],
    )
    self.assertEqual(
        _DrillLearner.run_starts, [0, 1, 3],
        "each restart must resume at the crashed run's last snapshot",
    )

  def test_crash_between_apply_and_its_snapshot_first_apply(self):
    """The commit-order race: the trainer's apply checkpoint is durable and
    the crash lands before a post-train snapshot could be. With the boundary
    snapshot pre-committed, the restart resumes the step right after the
    apply and groups 4-7 still train; without it the restart restored T=1,
    found no snapshot, and silently skipped the rest of the step."""
    self._drill(
        fbs=8, mini=4, micro=2, mu=1, max_steps=2,
        crash_plans=[{1: "between"}, {}],
    )
    self.assertEqual(_DrillLearner.run_starts, [0, 2])

  def test_crash_between_apply_and_its_snapshot_k1(self):
    """k=1: every chunk applies, so every between-crash is this race."""
    self._drill(
        fbs=4, mini=2, micro=2, mu=2, max_steps=2,
        crash_plans=[{0: "between"}, {}],
    )
    self.assertEqual(_DrillLearner.run_starts, [0, 1])

  def test_crash_between_apply_and_its_snapshot_mid_step_mu2(self):
    self._drill(
        fbs=4, mini=4, micro=2, mu=2, max_steps=2,
        crash_plans=[{1: "between"}, {}],
    )
    self.assertEqual(_DrillLearner.run_starts, [0, 2])

  def test_crash_before_first_apply_resumes_window_zero(self):
    """Window 0 is resumable: a preemption after most of the first window
    (k=4, crash after micro-update 2) must resume at iter 3 from the
    step-0 anchor, and the groups already trained before the crash must
    be neither regenerated nor retrained. Before the anchor existed this
    scenario restarted from scratch, discarding the whole window."""
    self._drill(
        fbs=4, mini=4, micro=1, mu=1, max_steps=2,
        crash_plans=[{2: "after"}, {}],
    )
    self.assertEqual(_DrillLearner.run_starts, [0, 3])
    trained_before_crash = {
        gid for run, _, gid, _ in _DrillLearner.trained_log if run == 0
    }
    self.assertEqual(len(trained_before_crash), 3)
    generated = {}
    for gid, pair in _FakeEngine.generation_log:
      generated[(gid, pair)] = generated.get((gid, pair), 0) + 1
    for gid in trained_before_crash:
      for pair in range(G):
        self.assertEqual(
            generated.get((gid, pair)), 1,
            f"group {gid} was regenerated after the window-0 resume",
        )

  # --- Sequence packing: the crash matrix under a token budget. Every
  # sequence is SEQ prompt tokens plus 1-4 real completion tokens (mask-
  # driven), so under budget 12 a pack holds one or two of them: windows are
  # ragged (data-dependent pack counts), applies ride the packer's
  # is_update flag, and identity flows through the packer's tags.


if __name__ == "__main__":
  absltest.main()
