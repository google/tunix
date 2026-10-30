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

"""Independent dense-objective oracle for nonuniform sampled-token OPD."""

import dataclasses
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.distillation import on_policy
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.rl import algo_core
from tunix.rl import algorithm_config

_COUNTS = np.array([[6, 3, 1], [2, 5, 3], [1, 2, 7]], dtype=np.int32)
_STUDENT = _COUNTS.astype(np.float32) / 10
_TEACHER = np.array(
    [[0.15, 0.70, 0.15], [0.70, 0.20, 0.10], [0.30, 0.60, 0.10]],
    dtype=np.float32,
)
_REWARDS = np.array(
    [[-1.0, 0.25, 2.0], [2.0, -1.0, 0.25], [0.25, 2.0, -1.0]],
    dtype=np.float32,
)
_BASELINE = np.sum(_STUDENT * _REWARDS, axis=-1, keepdims=True)
_TASK_ADVANTAGES = _REWARDS - _BASELINE


class _TransitionPolicy(nnx.Module):
  """A causal policy whose next-token logits depend on the preceding token."""

  def __init__(self):
    self.logits = nnx.Param(jnp.log(_STUDENT))

  def __call__(
      self, tokens, *, positions, cache, attention_mask, segment_ids=None
  ):
    del positions, attention_mask, segment_ids
    return self.logits[...][tokens], cache


def _payloads():
  """Enumerates the nonuniform behavior exactly, rather than using noisy draws."""
  payloads, teacher_scores = [], []
  for context, counts in enumerate(_COUNTS):
    for action, count in enumerate(counts):
      for _ in range(count):
        payloads.append(
            datatypes.RLTrainerPayload(
                prompt_ids=np.array([context], dtype=np.int32),
                prompt_mask=np.ones(1, dtype=np.float32),
                completion_ids=np.array([action], dtype=np.int32),
                completion_mask=np.ones(1, dtype=np.float32),
                old_per_token_logps=np.log(
                    _STUDENT[context, action : action + 1]
                ),
                advantages=_TASK_ADVANTAGES[context, action : action + 1],
            )
        )
        teacher_scores.append(np.log(_TEACHER[context, action : action + 1]))
  return payloads, teacher_scores


def _assemble(payloads, *, packed):
  if packed:
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        num_generations=10,
        mini_batch_size=3,
        max_packed_len=64,
        pad_id=0,
    )
  else:
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=30,
        num_generations=10,
        mini_batch_size=3,
        max_prompt_length=2,
        max_response_length=2,
        pad_id=0,
    )
  batches = list(assembler.feed(payloads)) + list(assembler.flush())
  if len(batches) != 1:
    raise AssertionError(f"Expected one complete batch, got {len(batches)}.")
  batch = batches[0].payload
  if not packed:
    # Each completion has one real token followed by padding. Zero is also a
    # real prompt/action ID, so token != pad would change the model context.
    batch = dataclasses.replace(
        batch,
        completion_attention_mask=np.broadcast_to([True, False], (30, 2)),
    )
  return batch


class OPDGradientTest(unittest.TestCase):

  def test_nonuniform_gradient_matches_dense_objective_with_real_assemblers(
      self,
  ):
    payloads, teacher_scores = _payloads()
    for packed in (False, True):
      for coefficient, task_scale in (
          (1.0, 0.0),
          (0.0, 0.7),
          (0.4, 0.7),
          (0.0, 0.0),
      ):
        with self.subTest(
            packed=packed, coefficient=coefficient, task_scale=task_scale
        ):
          shaped, _ = on_policy.apply_opd_to_payloads(
              payloads,
              teacher_scores if coefficient else None,
              on_policy.OnPolicyDistillationConfig(
                  coefficient=coefficient, task_reward_coefficient=task_scale
              ),
          )
          batch = _assemble(shaped, packed=packed)
          config = algorithm_config.GRPOConfig(beta=0, use_rollout_logps=True)
          config.temperature = 1.0

          def actual_loss(model):
            return algo_core.grpo_loss_fn(
                model, batch, config, pad_id=0, eos_id=0
            ).primary_loss.compute()

          def independent_objective(logits):
            logp = jax.nn.log_softmax(logits, axis=-1)
            p = jnp.exp(logp)
            kl = jnp.sum(p * (logp - jnp.log(_TEACHER)), axis=-1)
            task = -jnp.sum(p * _TASK_ADVANTAGES, axis=-1)
            return jnp.mean(coefficient * kl + task_scale * task)

          model = _TransitionPolicy()
          _, actual = nnx.value_and_grad(actual_loss)(model)
          expected = jax.grad(independent_objective)(jnp.log(_STUDENT))
          np.testing.assert_allclose(
              actual["logits"][...], expected, rtol=1e-5, atol=1e-6
          )
          if coefficient and not task_scale:
            # A real update must reduce independently evaluated reverse KL.
            before = independent_objective(jnp.log(_STUDENT))
            after = independent_objective(
                jnp.log(_STUDENT) - 0.1 * actual["logits"][...]
            )
            self.assertLess(float(after), float(before))
          if not coefficient and not task_scale:
            np.testing.assert_array_equal(actual["logits"][...], 0.0)


if __name__ == "__main__":
  unittest.main()
