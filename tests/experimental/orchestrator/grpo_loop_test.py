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

"""Tests for the formal GRPO loop over RLOrchestrator primitives."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
import unittest

import numpy as np
from tunix.experimental.orchestrator import grpo_loop


@dataclasses.dataclass(frozen=True)
class _FakeTrainExample:
  prompt_ids: np.ndarray
  prompt_mask: np.ndarray
  completion_ids: np.ndarray
  completion_mask: np.ndarray
  advantages: np.ndarray
  ref_per_token_logps: np.ndarray | None = None
  old_per_token_logps: np.ndarray | None = None
  policy_version: np.ndarray | None = None
  sampler_is_weights: np.ndarray | None = None

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)


class _FakeTokenizer:

  def encode(self, text, add_special_tokens=False):
    del add_special_tokens
    return [ord(text[-1])]


class _FakeOrchestrator:

  def __init__(self, *, return_logprobs=True, num_iterations=1):
    self.global_steps = 0
    self.algorithm = SimpleNamespace(
        algo_config=SimpleNamespace(
            beta=0.0,
            force_compute_kl=False,
            num_iterations=num_iterations,
            use_rollout_logps=True,
        )
    )
    self.calls = []
    self.train_chunks = None
    self.return_logprobs = return_logprobs

  def configure_trainer(self):
    self.calls.append(("configure_trainer",))

  def generate(self, prompts, max_generation_steps=None):
    self.calls.append(("generate", list(prompts), max_generation_steps))
    rollout = SimpleNamespace(
        text=[f"completion-{i}" for i in range(len(prompts))],
        tokens=[[10 + i] for i in range(len(prompts))],
    )
    if self.return_logprobs:
      rollout.logprobs = [
          [float(i), float(i + 1)] for i in range(len(prompts))
      ]
    return rollout

  def compute_advantages(self, rewards, *, num_generations):
    self.calls.append(("compute_advantages", rewards.copy(), num_generations))
    return rewards + 1.0

  def assemble_train_example(
      self,
      prompt_token_lists,
      completion_token_lists,
      advantages,
      *,
      max_prompt_length,
      max_response_length,
      pad_id,
      policy_version,
      old_per_token_logps=None,
    ):
    self.calls.append((
        "assemble_train_example",
        prompt_token_lists,
        completion_token_lists,
        max_prompt_length,
        max_response_length,
        pad_id,
        policy_version.copy(),
        None if old_per_token_logps is None else old_per_token_logps.copy(),
    ))
    batch = len(prompt_token_lists)
    return _FakeTrainExample(
        prompt_ids=np.arange(batch * 2, dtype=np.int32).reshape(batch, 2),
        prompt_mask=np.ones((batch, 2), dtype=bool),
        completion_ids=np.arange(batch * 3, dtype=np.int32).reshape(batch, 3),
        completion_mask=np.ones((batch, 3), dtype=np.int32),
        advantages=np.asarray(advantages, dtype=np.float32),
        old_per_token_logps=old_per_token_logps,
        policy_version=policy_version,
    )

  def train_step(self, chunks, skip_jit=False):
    self.calls.append(("train_step", len(chunks), skip_jit))
    self.train_chunks = chunks
    return 17

  def evaluate(self, eval_ds):
    self.calls.append(("evaluate", eval_ds))

  def sync_weights(self):
    self.calls.append(("sync_weights",))


def _reward_fn(prompts, completions, gold_answers, scale):
  del completions
  assert gold_answers == ["a", "a", "b", "b"]
  return [scale * float(i) for i in range(len(prompts))]


class GRPOLoopTest(unittest.TestCase):

  def test_train_step_runs_formal_grpo_choreography(self):
    orchestrator = _FakeOrchestrator()
    loop = grpo_loop.GRPOLoop(
        orchestrator,
        reward_fn=_reward_fn,
        tokenizer=_FakeTokenizer(),
        num_generations=2,
        max_prompt_length=8,
        max_response_length=4,
        train_micro_batch_size=3,
        pad_id=0,
        eos_id=1,
    )

    result = loop.train_step(
        ["prompt-a", "prompt-b"],
        reward_kwargs={"gold_answers": ["a", "b"], "scale": 2.0},
        step=5,
        eval_ds="eval",
        skip_jit=True,
    )

    self.assertEqual(result.step, 5)
    self.assertEqual(result.train_step, 17)
    self.assertEqual(result.global_step, 1)
    self.assertEqual(result.num_prompt_groups, 2)
    self.assertEqual(result.num_trajectories, 4)
    self.assertEqual(result.num_chunks, 2)
    np.testing.assert_allclose(result.rewards, np.array([0.0, 2.0, 4.0, 6.0]))
    np.testing.assert_allclose(
        result.advantages, np.array([1.0, 3.0, 5.0, 7.0])
    )
    self.assertEqual(len(orchestrator.train_chunks), 2)
    self.assertEqual(orchestrator.train_chunks[0].prompt_ids.shape[0], 3)
    self.assertEqual(orchestrator.train_chunks[1].prompt_ids.shape[0], 1)
    np.testing.assert_allclose(
        orchestrator.train_chunks[0].old_per_token_logps,
        np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0], [2.0, 3.0, 0.0, 0.0]]),
    )
    np.testing.assert_allclose(
        orchestrator.train_chunks[1].old_per_token_logps,
        np.array([[3.0, 4.0, 0.0, 0.0]]),
    )
    self.assertEqual(orchestrator.global_steps, 1)
    self.assertEqual(orchestrator.calls[0], ("configure_trainer",))
    self.assertEqual(orchestrator.calls[-3], ("train_step", 2, True))
    self.assertEqual(orchestrator.calls[-2], ("evaluate", "eval"))
    self.assertEqual(orchestrator.calls[-1], ("sync_weights",))

  def test_split_train_example_rejects_bad_micro_batch_size(self):
    example = _FakeTrainExample(
        prompt_ids=np.ones((1, 1), dtype=np.int32),
        prompt_mask=np.ones((1, 1), dtype=bool),
        completion_ids=np.ones((1, 1), dtype=np.int32),
        completion_mask=np.ones((1, 1), dtype=np.int32),
        advantages=np.ones((1,), dtype=np.float32),
    )
    with self.assertRaisesRegex(ValueError, "train_micro_batch_size"):
      grpo_loop.split_train_example(example, 0)

  def test_off_policy_iterations_require_rollout_logps(self):
    orchestrator = _FakeOrchestrator(
        return_logprobs=False, num_iterations=2
    )
    loop = grpo_loop.GRPOLoop(
        orchestrator,
        reward_fn=_reward_fn,
        tokenizer=_FakeTokenizer(),
        num_generations=2,
        max_prompt_length=8,
        max_response_length=4,
        train_micro_batch_size=2,
        pad_id=0,
        eos_id=1,
    )

    with self.assertRaisesRegex(RuntimeError, "old_per_token_logps"):
      loop.train_step(
          ["prompt-a"],
          reward_kwargs={"gold_answers": ["a"], "scale": 2.0},
      )


if __name__ == "__main__":
  unittest.main()
