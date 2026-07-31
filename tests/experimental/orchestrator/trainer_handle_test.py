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

"""Parameter equality for the orchestrated loop over a step-level trainer.

The orchestrated learner drives training through a trainer handle. With a
handle backed by the toy trainer, every optimizer update is closed-form
arithmetic, so a full run can be compared parameter-for-parameter against a
replay of the same train examples under the accumulation rule the agentic
learner uses. That pins the dispatch schedule -- how many updates happen and
where the accumulation boundaries fall -- which the postprocess parity tests
cannot see.
"""

import os
from typing import Any

from absl.testing import absltest
import chex
from flax import nnx
import numpy as np
import optax
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import orchestrated_agentic_learner
from tunix.experimental.orchestrator import orchestrator_rl_cluster
from tunix.experimental.orchestrator import rl_orchestrator
from tunix.experimental.orchestrator import trainer_handle as trainer_handle_lib
from tunix.experimental.testing import toy_trainer
from tunix.generate import tokenizer_adapter
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"


def _reward_fn(prompts, completions, **kwargs):
  del prompts, kwargs
  return [float(i) for i in range(len(completions))]


class _MockChatParser:

  def parse(self, messages, add_generation_prompt=False, is_first_msg=False):
    del is_first_msg
    result = ""
    for message in messages:
      if message["role"] == "system":
        result += f"System: {message['content']}"
      elif message["role"] == "user":
        result += f" User: {message['content']}"
      elif message["role"] == "assistant":
        result += f" Assistant: {message['content']}"
      else:
        raise ValueError(f"Unsupported message role: {message['role']}")
    if add_generation_prompt:
      result += " " + self.assistant_token
    return result

  @property
  def assistant_token(self):
    return "Assistant: "

  def update_assistant_end_tokens(self, tokens):
    return tokens, 0


class _RecordingHandle(trainer_handle_lib.AbstractTrainerHandle):
  """Records every micro-batch it trains on, so the run can be replayed."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.seen = []

  def train(self, chunks, eval_ds=None, skip_jit=False):
    recorded = list(trainer_handle_lib._as_iterable(chunks))  # pylint: disable=protected-access
    self.seen.extend(recorded)
    super().train(recorded, eval_ds, skip_jit)


class OrchestratedLoopParameterEqualityTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    try:
      chex.set_n_cpu_devices(2)
    except RuntimeError:
      pass

  def setUp(self):
    super().setUp()
    self.vocab = test_common.MockVocab()
    self.tokenizer = tokenizer_adapter.TokenizerAdapter(self.vocab)

  def _build_cluster(self, max_steps: int) -> rl_cluster_lib.RLCluster:
    model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=self.vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    ref_model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=self.vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    from jax.interpreters import pxla  # pylint: disable=g-import-not-at-top

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-2),
            eval_every_n_steps=100,
            max_steps=max_steps,
            mini_batch_size=1,
            train_micro_batch_size=1,
            rollout_micro_batch_size=1,
            compute_logps_micro_batch_size=1,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_prompt_length=32,
            max_tokens_to_generate=10,
            return_logprobs=True,
            kv_cache_size=256,
            temperature=0.5,
        ),
    )
    return rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=self.tokenizer,
        cluster_config=cluster_config,
    )

  def _toy(self) -> Any:
    return toy_trainer.ToyAbstractTrainer(
        {"vocab_size": self.vocab.GetPieceSize(), "learning_rate": 0.05}
    )

  def test_orchestrated_run_equals_replayed_reference_schedule(self):
    grad_accumulation_steps = 2
    handle = _RecordingHandle(
        self._toy(), grad_accumulation_steps=grad_accumulation_steps
    )
    base = self._build_cluster(max_steps=2)
    cluster = orchestrator_rl_cluster.OrchestratorRLCluster(
        base, trainer_worker=handle
    )
    orchestrator = rl_orchestrator.RLOrchestrator(
        cluster,
        algorithm_adapter.GRPOAdapter(
            agentic_grpo_learner.GRPOConfig(
                num_generations=2,
                num_iterations=1,
                beta=0.0,
                max_response_length=10,
            )
        ),
    )
    learner = orchestrated_agentic_learner.OrchestratedAgenticGRPOLearner(
        orchestrator=orchestrator,
        reward_fns=_reward_fn,
        chat_parser=_MockChatParser(),
    )

    learner.train([
        {"prompts": [str(i)], "answer": [str(i)], "question": [str(i)]}
        for i in range(4)
    ])

    self.assertNotEmpty(handle.seen)
    # The training substrate really was the step-level trainer.
    self.assertGreater(handle.updates_applied, 0)
    self.assertEqual(
        handle.updates_applied,
        len(handle.seen) // grad_accumulation_steps,
    )

    # Replay the same examples on a fresh trainer under the boundary rule the
    # agentic learner applies to unpacked batches: apply every N micro-steps.
    replay = self._toy()
    micro_steps = 0
    for example in handle.seen:
      replay.fwd_bwd(trainer_handle_lib.to_trainer_payload(example))
      micro_steps += 1
      if micro_steps % grad_accumulation_steps == 0:
        replay.update()

    chex.assert_trees_all_close(
        handle.trainer.params, replay.params, atol=1e-6, rtol=1e-6
    )
    self.assertEqual(handle.trainer.train_steps, replay.train_steps)

  def test_accumulation_boundary_is_honored(self):
    handle = trainer_handle_lib.AbstractTrainerHandle(
        self._toy(), grad_accumulation_steps=3
    )
    example = _example()

    handle.train([example, example])
    self.assertEqual(handle.updates_applied, 0)
    self.assertEqual(handle.trainer.train_steps, 0)

    handle.train([example])
    self.assertEqual(handle.updates_applied, 1)
    self.assertEqual(handle.trainer.train_steps, 1)

  def test_payload_masks_the_prompt_out_of_the_loss(self):
    payload = trainer_handle_lib.to_trainer_payload(_example())

    prompt_width = 2
    np.testing.assert_array_equal(
        np.asarray(payload.loss_mask)[:, :prompt_width], np.zeros((1, 2))
    )
    np.testing.assert_array_equal(
        np.asarray(payload.loss_mask)[:, prompt_width:], np.ones((1, 3))
    )
    self.assertEqual(np.asarray(payload.token_ids).shape, (1, 5))

  def test_rejects_a_non_positive_accumulation_window(self):
    with self.assertRaises(ValueError):
      trainer_handle_lib.AbstractTrainerHandle(
          self._toy(), grad_accumulation_steps=0
      )


def _example() -> Any:
  """A one-row train example with a 2-token prompt and 3-token completion."""
  return agentic_grpo_learner.TrainExample(
      prompt_ids=np.array([[1, 2]], dtype=np.int32),
      prompt_mask=np.array([[1, 1]], dtype=np.int32),
      completion_ids=np.array([[3, 4, 5]], dtype=np.int32),
      completion_mask=np.array([[1, 1, 1]], dtype=np.int32),
      advantages=np.array([1.0], dtype=np.float32),
      ref_per_token_logps=None,
      old_per_token_logps=None,
  )


if __name__ == "__main__":
  absltest.main()
