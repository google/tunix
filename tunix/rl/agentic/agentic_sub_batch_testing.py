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

"""Shared fixtures for the AgenticRLLearner sub-batch checkpointing tests.

Kept out of the `*_test.py` modules so every test that shares them imports
them from the package, in any build layout.
"""

import collections
import threading
from types import SimpleNamespace
from typing import Any

from flax import nnx
import jax.numpy as jnp
import optax
from tunix.rl.agentic import agentic_rl_learner
from tunix.sft import peft_trainer


class TinyModel(nnx.Module):
  """Minimal nnx module standing in for the actor.

  One param and a real forward pass, so nnx.value_and_grad produces grads
  shaped the way nnx.Optimizer.update expects (a hand-built plain-dict grads
  tree does not match).
  """

  def __init__(self, rngs: nnx.Rngs):
    del rngs  # Deterministic zero init.
    self.w = nnx.Param(jnp.zeros((4,)))

  def __call__(self, x):
    return jnp.sum(self.w * x)


def make_trainer_parts():
  """Builds the real trainer-side accumulation stack the learner targets.

  A model, the REAL peft_trainer.GradientAccumulator (the single
  mode-agnostic home of in-flight gradient state on this trainer build), and
  a plain Adam optimizer applied from the accumulator at window boundaries,
  matching peft_trainer._train_step's add/get/update/reset cycle.

  Returns:
    (model, accumulator, optimizer).
  """
  model = TinyModel(nnx.Rngs(0))
  accumulator = peft_trainer.GradientAccumulator(model, nnx.Param)
  optimizer = nnx.Optimizer(model, optax.adam(0.1), wrt=nnx.Param)
  return model, accumulator, optimizer


class FakeTrainer(SimpleNamespace):
  """Fake trainer with PeftTrainer's counter surface.

  `iter_steps` is a property over `_iter_steps`, which the restore path
  writes directly.
  """

  @property
  def iter_steps(self):
    return self._iter_steps

  @iter_steps.setter
  def iter_steps(self, value):
    self._iter_steps = value


# `make_learner(checkpoint_manager_latest_step=...)` default: the stub reports
# the trainer's own train_steps (durable by construction); pass None for
# "never checkpointed".
LATEST_IS_TRAIN_STEPS = object()


class Learner(agentic_rl_learner.AgenticRLLearner):
  """Concrete subclass: stubs the one abstract method so __new__ works."""

  def _process_results(self, *a, **k):
    return []


def make_learner(
    *,
    num_iterations=1,
    num_generations=2,
    mgr=None,
    global_steps=0,
    process_in_consumer=False,
    train_steps=0,
    k=1,
    checkpoint_manager_latest_step=LATEST_IS_TRAIN_STEPS,
    full_batch_size=0,
    critic_train_steps=None,
    critic_checkpoint_manager=None,
) -> Any:
  """Builds a `Learner` without `__init__`, around fake collaborators.

  Every attribute the sub-batch hooks read is set the way `__init__` sets
  it. The engine, trainer and trainer checkpoint manager are fakes with the
  surface those hooks use; the trainer's model, accumulator and optimizer are
  real (see `make_trainer_parts`).

  Args:
    num_iterations: `algo_config.num_iterations` (mu).
    num_generations: `algo_config.num_generations` (G).
    mgr: The sub-batch manager; None leaves the feature disabled.
    global_steps: The engine's global step.
    process_in_consumer: Which queue path the learner uses.
    train_steps: The trainer's apply count; `iter_steps` is `train_steps * k`.
    k: Gradient accumulation steps.
    checkpoint_manager_latest_step: What the trainer checkpoint manager
      reports as its latest step; defaults to `train_steps`.
    full_batch_size: The learner's resolved full batch size.
    critic_train_steps: When set, the engine also gets a critic trainer.
    critic_checkpoint_manager: The critic trainer's checkpoint manager.

  Returns:
    The learner.
  """
  learner: Any = Learner.__new__(Learner)
  if checkpoint_manager_latest_step is LATEST_IS_TRAIN_STEPS:
    checkpoint_manager_latest_step = train_steps
  learner.algo_config = SimpleNamespace(
      num_iterations=num_iterations,
      num_generations=num_generations,
      off_policy_steps=0,
      sub_batch_checkpointing=True,
      sub_batch_dataset_id=None,
  )
  learner._process_in_consumer = process_in_consumer
  learner._full_batch_size = full_batch_size
  learner._packing_enabled = False
  learner._sb_mgr = mgr
  learner._sb_lock = threading.Lock()
  learner._sb_completed = set()
  learner._sb_counts = {}
  learner._sb_active = []
  learner._sb_pending_state = None
  # Identity staging store (both queue paths write it; __init__ builds it).
  learner._sb_item_fifo = collections.deque()
  learner._sb_last_snapshot_key = -1
  learner._sb_window_train_steps = -1
  learner._sb_local = -1
  learner._sb_last_snapshot_trainer_iter = -1
  learner._sb_active_gids = set()
  learner._sb_step_complete_for_step = None
  learner._sb_restored_trainer_state = False
  learner._sb_restored_full_batch_size = None
  learner._sb_restored_geometry = None
  learner._sb_geo_train = None
  learner._sb_resumed_mid_step = False
  learner._sb_anchor_step = 0
  # Disaggregated by default: the step-start pin is active.
  learner.should_sync_weights = True
  model, accumulator, optimizer = make_trainer_parts()
  learner._training_config = SimpleNamespace(
      get_with_default=lambda key, default: k
      if key == "gradient_accumulation_steps"
      else default,
      checkpoint_root_directory="/tmp/unused",
      max_seq_token_per_tpu=None,
      mini_batch_size=None,
      train_micro_batch_size=None,
      checkpointing_options=SimpleNamespace(
          save_decision_policy=SimpleNamespace(interval=1)
      ),
  )
  saved: list[Any] = []  # the fresh-run step-0 anchor save lands here
  pinned: list[int] = []  # every CheckpointManager.pin_step call, in order
  learner.rl_engine = SimpleNamespace(
      global_steps=global_steps,
      cluster_config=SimpleNamespace(offload_to_cpu=False),
      actor_trainer=FakeTrainer(
          model=model,
          optimizer=optimizer,
          grad_accumulator=accumulator,
          _train_steps=train_steps,
          train_steps=train_steps,
          # SimpleNamespace writes kwargs straight into __dict__, so the
          # `iter_steps` entry only keeps vars() shaped like a plain
          # namespace for callers that copy it; the property reads
          # `_iter_steps`.
          iter_steps=train_steps * k,
          _iter_steps=train_steps * k,
          checkpoint_manager=SimpleNamespace(
              latest_step=lambda: checkpoint_manager_latest_step,
              # Every step up to the latest is on disk.
              has_step=lambda step: (
                  checkpoint_manager_latest_step is not None
                  and 0 <= step <= checkpoint_manager_latest_step
              ),
              maybe_restore=lambda *a, **kw: (kw.get("step", 0), {}),
              saved=saved,
              pinned=pinned,
              pin_step=pinned.append,
              save=lambda *a, **kw: saved.append((a, kw)),
              wait=lambda *a, **kw: None,
              # The RESOLVED options the learner's cadence check reads.
              _options=SimpleNamespace(
                  save_decision_policy=SimpleNamespace(interval=1),
              ),
          ),
      ),
  )
  if critic_train_steps is not None:
    # hasattr(rl_engine, "critic_trainer") is the real presence signal, so
    # the attribute only exists when a critic is requested.
    critic_model, critic_accumulator, critic_optimizer = make_trainer_parts()
    learner.rl_engine.critic_trainer = SimpleNamespace(
        model=critic_model,
        optimizer=critic_optimizer,
        grad_accumulator=critic_accumulator,
        _train_steps=critic_train_steps,
        train_steps=critic_train_steps,
        iter_steps=critic_train_steps * k,
        checkpoint_manager=(
            critic_checkpoint_manager
            or SimpleNamespace(
                latest_step=lambda: critic_train_steps,
                maybe_restore=lambda *a, **kw: (kw.get("step", 0), {}),
            )
        ),
    )
  return learner
