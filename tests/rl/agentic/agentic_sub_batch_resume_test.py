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

import asyncio
import dataclasses
import functools
import itertools
import os
import queue
import threading
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax.checkpoint import v1 as ocp
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl import sub_batch_checkpoint
from tunix.rl.agentic import agentic_rl_learner
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import checkpoint_options


class TinyModel(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    del rngs
    self.w = nnx.Param(jnp.zeros((4,)))

  def __call__(self, x):
    return jnp.sum(self.w * x)


class Learner(agentic_rl_learner.AgenticRLLearner):
  """Concrete subclass: stubs the one abstract method so __new__ works."""

  def _process_results(self, *a, **k):
    return []


class FakeCheckpointManager:
  """The trainer checkpoint manager surface the sub-batch hooks use."""

  def __init__(self, latest_step: int | None):
    self.latest = latest_step
    self.saved: list[int] = []
    self.saved_optimizers: list[nnx.Optimizer | None] = []
    self.forced: list[bool] = []
    self.metadata: list[dict[str, int | str] | None] = []
    # Whether every save so far is durable.
    self.durable = True

  def latest_step(self) -> int | None:
    return self.latest

  def save(
      self,
      step: int,
      model: nnx.Module | None = None,
      optimizer: nnx.Optimizer | None = None,
      *,
      save_only_lora_params: bool = False,
      force: bool = False,
      custom_metadata: dict[str, int | str] | None = None,
  ) -> bool:
    del model, save_only_lora_params
    self.saved.append(step)
    self.saved_optimizers.append(optimizer)
    self.forced.append(force)
    self.metadata.append(custom_metadata)
    self.latest = step
    self.durable = False
    return True

  def wait(self) -> None:
    self.durable = True


class FakeEngine(SimpleNamespace):
  """Records weight syncs by whether they end a global step."""

  def sync_weights(self, *, increment_global_steps=True):
    self.syncs.append(increment_global_steps)


def make_learner(
    *,
    root: str,
    train_steps: int = 0,
    latest_step: int | None = 0,
    global_steps: int = 0,
    k: int = 2,
    num_generations: int = 2,
    num_iterations: int = 1,
    full_batch_size: int = 4,
    process_in_consumer: bool = False,
    save_interval: int | None = None,
    enabled: bool = True,
    packed: bool = False,
) -> Learner:
  """Builds a `Learner` without `__init__` around fake collaborators.

  Args:
    root: `training_config.checkpoint_root_directory`.
    train_steps: The trainer's restored apply count.
    latest_step: The trainer's latest checkpoint step, None if none.
    global_steps: The engine's global step.
    k: `gradient_accumulation_steps`.
    num_generations: `algo_config.num_generations`.
    num_iterations: `algo_config.num_iterations`.
    full_batch_size: The learner's resolved full batch size.
    process_in_consumer: Which queue path the learner uses.
    save_interval: The trainer's checkpoint interval. Defaults to every global
      step of the tests' mini-batches of 2 groups.
    enabled: `algo_config.sub_batch_checkpointing`.
    packed: Whether sequence packing is on.

  Returns:
    The learner, with sub-batch checkpointing configured but not initialized.
  """
  if save_interval is None:
    save_interval = num_iterations * full_batch_size // 2
  learner = Learner.__new__(Learner)
  learner.algo_config = agentic_rl_learner.AgenticRLConfig(
      sub_batch_checkpointing=enabled,
      num_generations=num_generations,
      num_iterations=num_iterations,
  )
  learner._training_config = SimpleNamespace(
      get_with_default=lambda key, default: (
          k if key == "gradient_accumulation_steps" else default
      ),
      checkpoint_root_directory=root,
      checkpointing_options=checkpoint_options.TunixCheckpointingOptions(
          save_decision_policy=(
              ocp.training.save_decision_policies.FixedIntervalPolicy(
                  save_interval
              )
          ),
          enable_async_checkpointing=False,
      ),
  )
  learner._packing_enabled = packed
  learner._process_in_consumer = process_in_consumer
  learner._full_batch_size = full_batch_size
  learner.should_sync_weights = True
  learner._sub_batch = None
  learner._sub_batch_resume = None
  learner._iter_steps = train_steps * k
  model = TinyModel(nnx.Rngs(0))
  learner.rl_engine = FakeEngine(
      global_steps=global_steps,
      syncs=[],
      actor_trainer=SimpleNamespace(
          model=model,
          optimizer=nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param),
          train_steps=train_steps,
          iter_steps=train_steps * k,
          checkpoint_manager=FakeCheckpointManager(latest_step),
      ),
  )
  return learner


def make_group(prompt_id: int, num_generations: int = 2):
  return [
      agent_types.TrajectoryItem(
          prompt_id=prompt_id,
          group_index=i,
          traj={"conversation_tokens": np.array([1, 2, 3]), "reward": 1.0},
          metadata={"generation_id": i},
      )
      for i in range(num_generations)
  ]


# Two updates per epoch: full_batch_size // mini_batch_size.
GEOMETRY = sub_batch_checkpoint.BatchGeometry(
    full_batch_size=4,
    mini_batch_size=2,
    train_micro_batch_size=1,
    gradient_accumulation_steps=2,
    num_generations=2,
    num_iterations=1,
)


def _manager(root):
  return sub_batch_checkpoint.SubBatchCheckpointManager(
      os.path.join(root, "sub_batch"), enable_async=False
  )


def _plant(root, *, global_step, prompt_ids, num_iterations=1):
  manager = _manager(root)
  manager.set_geometry(
      dataclasses.replace(GEOMETRY, num_iterations=num_iterations)
  )
  for prompt_id in prompt_ids:
    manager.record_rollout_group(global_step, make_group(prompt_id))
  manager.close()


def _prompt_ids(groups):
  return [group[0].prompt_id for group in groups]


class SubBatchInitTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.root = self.create_tempdir().full_path

  def _init(self, **kwargs):
    learner = make_learner(root=self.root, **kwargs)
    learner._init_sub_batch_checkpointing(GEOMETRY)
    self.addCleanup(learner._sub_batch.close)
    return learner

  def _trainer_checkpoints(self, learner):
    return learner.rl_engine.actor_trainer.checkpoint_manager

  def test_fresh_run_saves_its_starting_weights(self):
    learner = self._init(latest_step=None)
    checkpoints = self._trainer_checkpoints(learner)
    self.assertEqual(checkpoints.saved, [0])
    self.assertEqual(checkpoints.forced, [True])
    # Weights only: a relaunch rebuilds the initial optimizer state.
    self.assertEqual(checkpoints.saved_optimizers, [None])
    # A restart from it resumes global step 0.
    self.assertEqual(
        checkpoints.metadata,
        [{"global_step": 0, "role": rl_engine_lib.Role.ACTOR.value}],
    )
    # Durable before the first rollout is recorded.
    self.assertTrue(checkpoints.durable)
    self.assertIsNone(learner._sub_batch_resume)

  def test_a_failed_starting_save_raises(self):
    learner = make_learner(root=self.root, latest_step=None)
    checkpoints = self._trainer_checkpoints(learner)
    with mock.patch.object(checkpoints, "save", return_value=False):
      with self.assertRaisesRegex(RuntimeError, "initial weights"):
        learner._init_sub_batch_checkpointing(GEOMETRY)
    learner._sub_batch.close()

  def test_fresh_run_with_snapshots_raises(self):
    _plant(self.root, global_step=0, prompt_ids=[0])
    learner = make_learner(root=self.root, latest_step=None)
    with self.assertRaisesRegex(ValueError, "no trainer checkpoint"):
      learner._init_sub_batch_checkpointing(GEOMETRY)
    learner._sub_batch.close()

  def test_restart_requeues_every_saved_group(self):
    _plant(self.root, global_step=4, prompt_ids=[16, 17, 18])
    learner = self._init(train_steps=8, latest_step=8, global_steps=4)
    self.assertEqual(
        _prompt_ids(learner._sub_batch_resume.active_groups), [16, 17, 18]
    )
    self.assertTrue(learner._sub_batch.should_skip_group(18))
    self.assertEqual(self._trainer_checkpoints(learner).saved, [])

  def test_snapshots_of_a_finished_step_are_not_restored(self):
    _plant(self.root, global_step=3, prompt_ids=[12])
    learner = self._init(train_steps=8, latest_step=8, global_steps=4)
    self.assertIsNone(learner._sub_batch_resume)


class SubBatchConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "critic",
          lambda learner: setattr(learner.rl_engine, "critic_trainer", None),
          "critic",
      ),
      (
          "off_policy",
          lambda learner: setattr(learner.algo_config, "off_policy_steps", 1),
          "off_policy_steps",
      ),
      (
          "no_root",
          lambda learner: setattr(
              learner._training_config, "checkpoint_root_directory", None
          ),
          "checkpoint_root_directory",
      ),
  )
  def test_rejected(self, mutate, message):
    learner = make_learner(root=self.create_tempdir().full_path)
    mutate(learner)
    with self.assertRaisesRegex(ValueError, message):
      learner._check_sub_batch_config()

  def test_multi_process_rejected(self):
    learner = make_learner(root=self.create_tempdir().full_path)
    with mock.patch.object(jax, "process_count", return_value=2):
      with self.assertRaisesRegex(ValueError, "single-process"):
        learner._check_sub_batch_config()


class SubBatchCadenceTest(parameterized.TestCase):
  """The trainer must checkpoint at each global-step boundary and only there."""

  @parameterized.named_parameters(
      ("every_update", 1, 4, 2),
      ("every_other_step", 4, 4, 2),
      ("inside_a_step", 2, 3, 1),
      # A trainer checkpoint without its global step restores global step 0.
      ("no_global_step", 2, 8, 0),
  )
  def test_rejects_the_trainer_cadence(
      self, save_interval, train_steps, global_steps
  ):
    root = self.create_tempdir().full_path
    _plant(root, global_step=4, prompt_ids=[16])
    learner = make_learner(
        root=root,
        train_steps=train_steps,
        latest_step=train_steps,
        global_steps=global_steps,
        save_interval=save_interval,
    )
    with self.assertRaisesRegex(ValueError, "every global-step boundary"):
      learner._init_sub_batch_checkpointing(GEOMETRY)
    # Rejected before a restore deletes the snapshots of later steps.
    manager = _manager(root)
    self.addCleanup(manager.close)
    self.assertEqual(
        _prompt_ids(manager.restore(global_step=4).active_groups), [16]
    )


class _RecordingOrchestrator:
  """Records the group ids the producer streams agent-env pairs for."""

  def __init__(self):
    self.group_ids = []
    self._done = asyncio.Event()

  async def run_producers_from_stream(self, pairs_stream, **unused_kwargs):
    async for _, env in pairs_stream:
      self.group_ids.append(env.extra_kwargs["group_id"])
    self._done.set()

  async def yield_batches(self, batch_size):
    del batch_size
    await self._done.wait()
    return
    yield  # pylint: disable=unreachable


class SubBatchProducerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.root = self.create_tempdir().full_path

  @parameterized.parameters(False, True)
  def test_skipped_groups_are_not_generated(self, stream):
    learner = make_learner(root=self.root, global_steps=5, full_batch_size=4)
    learner._background_tasks = set()
    learner._create_agent_env_pair = lambda example, group_id, pair_index: (
        None,
        SimpleNamespace(extra_kwargs={"group_id": group_id}),
    )

    async def prompt_stream():
      for prompt in [{}, {}, {}]:
        yield prompt

    async def run():
      learner.loop = asyncio.get_running_loop()
      orchestrator = _RecordingOrchestrator()
      async for _ in learner._orchestrator_producer(
          orchestrator,
          # Training streams its prompts.
          prompt_stream() if stream else [{}, {}, {}],
          skip_group=lambda group_id: group_id == 9,
          first_group_id=8,
      ):
        pass
      return orchestrator.group_ids

    self.assertEqual(asyncio.run(run()), [8, 10])

  def test_restored_groups_queue_first_and_new_groups_are_saved(self):
    # Group 9 finished first, so it trained first.
    _plant(self.root, global_step=2, prompt_ids=[9, 8])
    learner = make_learner(root=self.root, global_steps=2)
    manager = _manager(self.root)
    learner._sub_batch = manager
    learner._sub_batch_resume = manager.restore(global_step=2)
    manager.set_geometry(GEOMETRY)
    learner._batch_to_train_example = lambda batch_results, mode: [
        batch_results[0].prompt_id
    ]

    async def orchestrator_producer(skip_group, first_group_id, **unused):
      for group_id in range(first_group_id, first_group_id + 3):
        if not skip_group(group_id):
          yield make_group(group_id)

    learner._orchestrator_producer = orchestrator_producer
    data = queue_lib.SimpleDataQueue(maxsize=0)
    asyncio.run(learner._producer(None, queue.Queue(), data))
    self.assertEqual(list(iter(data.get, None)), [9, 8, 10])
    # The producer doesn't hold on to the restored groups.
    self.assertIsNone(learner._sub_batch_resume)
    manager.close()
    restored = _manager(self.root)
    self.addCleanup(restored.close)
    self.assertEqual(
        _prompt_ids(restored.restore(global_step=2).active_groups),
        [9, 8, 10],
    )


class _TrainEngine(FakeEngine):
  """Logs the trained chunks; the trainer applies every `k` of them.

  A pack instead applies when its update flag is set, and is logged with its
  groups and that flag.
  """

  def update_actor(self, chunks, eval_dataset, skip_jit):
    del skip_jit
    trainer = self.actor_trainer
    for chunk in chunks:
      trainer.iter_steps += 1
      if chunk.segment_ids is None:
        group_id = int(chunk.prompt_ids[0, 0])
        self.log.append(("train", group_id))
        if eval_dataset:
          self.evals.append(group_id)
        applies = trainer.iter_steps % self.k == 0
      else:
        applies = bool(np.asarray(chunk.is_update_step).item())
        # Every token of a packed group is its id + 1; padding is 0.
        tokens = np.unique(np.asarray(chunk.completion_ids))
        groups = tuple(int(token) - 1 for token in tokens if token)
        self.log.append(("train", groups, applies))
      if applies:
        trainer.train_steps += 1
        trainer.checkpoint_manager.save(trainer.train_steps)

  def sync_weights(self, *, increment_global_steps=True):
    super().sync_weights(increment_global_steps=increment_global_steps)
    if increment_global_steps:
      self.global_steps += 1

  def buffer_metrics(self, *unused_args, **unused_kwargs):
    pass

  buffer_metrics_async = buffer_metrics
  close = buffer_metrics


async def _orchestrator_producer(
    orchestrator,
    prompt_iterator,
    num_generations=1,
    collect_mode="Token",
    skip_group=None,
    first_group_id=None,
    *,
    learner,
    generated,
):
  """Yields one group per prompt, numbered like the real producer."""
  del orchestrator, num_generations, collect_mode
  if isinstance(prompt_iterator, list):  # Eval prompts.
    for _ in prompt_iterator:
      yield make_group(-1)
    return
  group_id = (
      learner.rl_engine.global_steps * learner._full_batch_size
      if first_group_id is None
      else first_group_id
  )
  async for _ in prompt_iterator:
    if skip_group is None or not skip_group(group_id):
      generated.append(group_id)
      yield make_group(group_id)
    group_id += 1


def _train_examples(batch_results, rewards):
  rows = len(batch_results)
  return [
      agentic_rl_learner.TrainExample(
          prompt_ids=np.array([[item.prompt_id] for item in batch_results]),
          prompt_mask=np.ones((rows, 1)),
          completion_ids=np.zeros((rows, 1)),
          completion_mask=np.ones((rows, 1)),
          advantages=np.asarray(rewards, dtype=np.float32)[:, None],
          ref_per_token_logps=None,
          old_per_token_logps=None,
      )
  ]


def _packable_train_examples(batch_results, rewards):
  """Rows of 2 tokens, 4 for groups 0, 4, ...; group g's tokens are g + 1."""
  rows = len(batch_results)
  completion_ids = np.zeros((rows, 3), dtype=np.int32)
  completion_mask = np.zeros((rows, 3), dtype=np.float32)
  for row, item in enumerate(batch_results):
    length = 3 if item.prompt_id % 4 == 0 else 1
    completion_ids[row, :length] = item.prompt_id + 1
    completion_mask[row, :length] = 1
  return [
      agentic_rl_learner.TrainExample(
          prompt_ids=np.array([[item.prompt_id + 1] for item in batch_results]),
          prompt_mask=np.ones((rows, 1)),
          completion_ids=completion_ids,
          completion_mask=completion_mask,
          advantages=np.asarray(rewards, dtype=np.float32),
          ref_per_token_logps=None,
          old_per_token_logps=None,
      )
  ]


class SubBatchTrainTest(parameterized.TestCase):
  """Runs `train()`: 4 groups per step, an update every 2 of them."""

  def setUp(self):
    super().setUp()
    self.root = self.create_tempdir().full_path
    # Every reward_manager call scores anew, as a stochastic judge would.
    self._rewards = itertools.count()

  def _train(
      self,
      *,
      max_steps,
      eval_dataset=None,
      k=2,
      process_in_consumer=False,
      packed=False,
      **learner_kwargs,
  ):
    learner = make_learner(root=self.root, k=k, packed=packed, **learner_kwargs)
    learner.rl_engine = _TrainEngine(
        **vars(learner.rl_engine),
        k=k,
        log=[],
        evals=[],
        generated=[],
        perf_v2=mock.MagicMock(),
        cluster_config=SimpleNamespace(
            training_config=SimpleNamespace(eval_every_n_steps=1),
            # One packed row per pack.
            role_to_mesh={
                rl_engine_lib.Role.ACTOR: SimpleNamespace(shape={"fsdp": 1})
            },
        ),
    )
    micro = 2 // k
    learner._training_config.mini_batch_size = 2
    learner._training_config.train_micro_batch_size = micro
    learner._training_config.max_steps = max_steps
    # Packs hold up to 8 tokens.
    learner._training_config.max_seq_token_per_tpu = 8
    # `train()` processes groups in the consumer when this equals `micro`.
    learner._compute_logps_micro_batch_size = (
        micro if process_in_consumer else 1
    )
    learner._rollout_sync_lock = agentic_utils.RolloutSyncLock()
    learner._iter_steps = learner.rl_engine.actor_trainer.iter_steps
    learner._eval_iter_steps = 0
    learner._last_eval_train_step = -1
    learner._train_rewards_window = []
    learner._eval_rewards_window = []
    learner._rewards_window_lock = threading.Lock()
    learner._global_step_start_time = 0.0
    learner.policy_version = 0
    learner._build_orchestrator = lambda: None
    learner._orchestrator_producer = functools.partial(
        _orchestrator_producer,
        learner=learner,
        generated=learner.rl_engine.generated,
    )
    learner.reward_manager = mock.Mock(
        side_effect=lambda prompts, completions, **unused: {
            "rewards": np.full(len(prompts), float(next(self._rewards))),
            "log_metrics": {},
        }
    )

    def batch_to_train_example(batch_results, mode):
      rewards = learner._compute_rewards(
          prompts=["p"] * len(batch_results),
          completions=["c"] * len(batch_results),
          mode=mode,
      )
      if packed:
        return _packable_train_examples(batch_results, rewards)
      return _train_examples(batch_results, rewards)

    learner._batch_to_train_example = mock.Mock(
        side_effect=batch_to_train_example
    )
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    self.addCleanup(loop.close)
    self.addCleanup(thread.join)
    self.addCleanup(loop.call_soon_threadsafe, loop.stop)
    learner.loop = loop
    learner.train(
        [{"prompts": ["p"] * 4} for _ in range(max_steps + 1)], eval_dataset
    )
    return learner

  def test_disabled_saves_nothing(self):
    learner = self._train(max_steps=1, latest_step=None, enabled=False)
    self.assertEqual(
        learner.rl_engine.log, [("train", group_id) for group_id in range(4)]
    )
    # No starting checkpoint and no snapshots.
    trainer = learner.rl_engine.actor_trainer
    self.assertEqual(trainer.checkpoint_manager.saved, [1, 2])
    self.assertFalse(os.path.exists(os.path.join(self.root, "sub_batch")))

  def test_generated_groups_are_saved_without_touching_training(self):
    learner = self._train(max_steps=1, latest_step=None, num_iterations=2)
    # Epoch 1 and the replay epoch, as without sub-batch checkpointing.
    self.assertEqual(
        learner.rl_engine.log,
        [("train", group_id) for group_id in (0, 1, 2, 3, 0, 1, 2, 3)],
    )
    self.assertEqual(learner.rl_engine.global_steps, 1)
    manager = _manager(self.root)
    self.addCleanup(manager.close)
    self.assertEqual(
        _prompt_ids(manager.restore(global_step=0).active_groups),
        [0, 1, 2, 3],
    )

  @parameterized.parameters(False, True)
  def test_restart_retrains_the_step_on_the_saved_rollouts(
      self, process_in_consumer
  ):
    kwargs = dict(
        max_steps=1,
        num_iterations=2,
        k=1 if process_in_consumer else 2,
        process_in_consumer=process_in_consumer,
    )
    first = self._train(latest_step=None, **kwargs)
    # Restart as if the step's final trainer checkpoint never became durable:
    # the whole step is retrained from its start on the saved rollouts.
    second = self._train(latest_step=0, **kwargs)
    self.assertEqual(first.rl_engine.generated, [0, 1, 2, 3])
    self.assertEqual(second.rl_engine.generated, [])
    self.assertEqual(second.rl_engine.log, first.rl_engine.log)
    # The restored rollouts are scored again.
    self.assertEqual(
        second.reward_manager.call_count, first.reward_manager.call_count
    )
    self.assertEqual(second.rl_engine.actor_trainer.train_steps, 4)
    # The only weight sync is the step's end.
    self.assertEqual(second.rl_engine.syncs, [True])

  def test_restart_generates_only_the_missing_groups(self):
    _plant(self.root, global_step=0, prompt_ids=[0, 1])
    learner = self._train(max_steps=1, latest_step=0)
    self.assertEqual(learner.rl_engine.generated, [2, 3])
    self.assertEqual(
        learner.rl_engine.log, [("train", group_id) for group_id in range(4)]
    )
    self.assertEqual(learner.rl_engine.actor_trainer.train_steps, 2)

  def test_resumed_step_runs_its_eval_again(self):
    _plant(self.root, global_step=0, prompt_ids=[0, 1])
    learner = self._train(
        max_steps=2,
        latest_step=0,
        eval_dataset=[{"prompts": ["e"]}],
    )
    self.assertEqual(
        learner.rl_engine.log, [("train", group_id) for group_id in range(8)]
    )
    # The resumed step evaluates again, which is right: a crash drops the
    # step's buffered eval metrics.
    self.assertEqual(learner.rl_engine.evals, [0, 4])

  def test_packed_fresh_run_trains_the_packs(self):
    learner = self._train(
        max_steps=1, latest_step=None, num_iterations=2, packed=True
    )
    # Groups 0 and 1 take two packs, groups 2 and 3 one.
    self.assertEqual(
        learner.rl_engine.log,
        [
            ("train", (0,), False),
            ("train", (1,), True),
            ("train", (2, 3), True),
            ("train", (0,), False),
            ("train", (1,), True),
            ("train", (2, 3), True),
        ],
    )

  def test_packed_restart_retrains_the_same_packs(self):
    kwargs = dict(max_steps=1, num_iterations=2, packed=True)
    first = self._train(latest_step=None, **kwargs)
    # Restart as if the step's final trainer checkpoint never became durable:
    # the packer rebuilds the same packs from the restored groups and closes
    # them at the same updates.
    second = self._train(latest_step=0, **kwargs)
    self.assertEqual(second.rl_engine.generated, [])
    self.assertEqual(second.rl_engine.log, first.rl_engine.log)
    self.assertEqual(second.rl_engine.actor_trainer.train_steps, 4)

  def test_toggling_packing_on_restart_raises(self):
    _plant(self.root, global_step=0, prompt_ids=[0])
    with self.assertRaisesRegex(ValueError, "does not match"):
      self._train(max_steps=1, latest_step=0, packed=True)


if __name__ == "__main__":
  absltest.main()
