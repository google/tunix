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

"""OPD numerical checks, including an independent reverse-KL gradient oracle."""

import dataclasses
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.distillation import on_policy
from tunix.rl import algo_core
from tunix.rl import algorithm_config


def _payload(**kwargs):
  values = dict(
      prompt_ids=np.array([1], dtype=np.int32),
      prompt_mask=np.array([1], dtype=np.float32),
      completion_ids=np.array([0, 1], dtype=np.int32),
      completion_mask=np.array([1, 1], dtype=np.float32),
      advantages=np.array([0, 0], dtype=np.float32),
      old_per_token_logps=np.log(np.array([0.5, 0.5], dtype=np.float32)),
  )
  values.update(kwargs)
  return datatypes.RLTrainerPayload(**values)


class _ConstantPolicy(nnx.Module):
  """A two-action policy, so enumerating the actions integrates exactly."""

  def __init__(self):
    self.logits = nnx.Param(jnp.zeros(2, dtype=jnp.float32))

  def __call__(self, tokens, *, positions, cache, attention_mask):
    del positions, attention_mask
    return jnp.broadcast_to(self.logits[...], (*tokens.shape, 2)), cache


class _PositionPolicy(_ConstantPolicy):
  """Makes missing real EOS/tool context observable in the loss gradient."""

  def __call__(self, tokens, *, positions, cache, attention_mask):
    del tokens, attention_mask
    offset = jnp.stack([0.2 * positions, -0.2 * positions], axis=-1)
    return self.logits[...] + offset, cache


class OnPolicyTest(unittest.TestCase):

  def test_loss_keeps_internal_eos_and_tool_context_via_attention_mask(self):
    # Zero is both the padding ID and a real input token. The tool observation
    # has zero loss weight but must still occupy a causal position before the
    # last assistant action. Its correct prediction positions are 2 and 4.
    positions = jnp.array([2.0, 4.0])
    expected_probs = jax.nn.softmax(
        jnp.stack([0.2 * positions, -0.2 * positions], axis=-1), axis=-1
    )
    old_scores = np.zeros((1, 4), dtype=np.float32)
    old_scores[0, [0, 2]] = np.log(np.asarray(expected_probs)[[0, 1], [0, 1]])
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[0, 1, 0, 2]], dtype=np.int32),
        prompt_mask=np.array([[0, 1, 1, 1]], dtype=np.float32),
        completion_ids=np.array([[0, 0, 1, 0]], dtype=np.int32),
        completion_mask=np.array([[1, 0, 1, 0]], dtype=np.float32),
        completion_attention_mask=np.array([[1, 1, 1, 0]], dtype=bool),
        advantages=np.array([[1, 0, -1, 0]], dtype=np.float32),
        old_per_token_logps=old_scores,
    )
    config = algorithm_config.GRPOConfig(beta=0, epsilon=1e6)
    config.temperature = 1.0

    def loss(model, batch):
      return algo_core.grpo_loss_fn(
          model, batch, config, pad_id=0, eos_id=0
      ).primary_loss.compute()

    _, grads = nnx.value_and_grad(loss)(_PositionPolicy(), payload)
    action_scores = jax.nn.one_hot(jnp.array([0, 1]), 2) - expected_probs
    expected_gradient = -jnp.mean(
        jnp.array([1.0, -1.0])[:, None] * action_scores, axis=0
    )
    np.testing.assert_allclose(
        grads["logits"][...], expected_gradient, rtol=1e-5, atol=1e-6
    )
    # Reverting to token != pad changes the contexts and cannot match the
    # independently computed gradient, even though all array shapes agree.
    _, wrong = nnx.value_and_grad(loss)(
        _PositionPolicy(),
        dataclasses.replace(payload, completion_attention_mask=None),
    )
    self.assertGreater(
        float(jnp.linalg.norm(wrong["logits"][...] - expected_gradient)), 1e-3
    )

  def test_actual_policy_loss_matches_dense_reverse_kl_gradient(self):
    # Sampling both actions once integrates the uniform behavior distribution.
    # This tests the sign and magnitude against KL(p || teacher), independently
    # of the sampled-token formula used to prepare the training advantages.
    teacher_probs = np.array([0.8, 0.2], dtype=np.float32)
    updated, _ = on_policy.apply_opd_to_payloads(
        [_payload()],
        [np.log(teacher_probs)],
        on_policy.OnPolicyDistillationConfig(),
    )
    row = updated[0]
    batch = dataclasses.replace(
        row,
        prompt_ids=row.prompt_ids[None, :],
        prompt_mask=row.prompt_mask[None, :],
        completion_ids=row.completion_ids[None, :],
        completion_mask=row.completion_mask[None, :],
        advantages=row.advantages[None, :],
        old_per_token_logps=row.old_per_token_logps[None, :],
    )
    config = algorithm_config.GRPOConfig(beta=0, use_rollout_logps=True)
    config.temperature = 1.0
    model = _ConstantPolicy()

    def loss(policy):
      return algo_core.grpo_loss_fn(
          policy, batch, config, pad_id=99, eos_id=98
      ).primary_loss.compute()

    _, grads = nnx.value_and_grad(loss)(model)
    actual = np.asarray(grads["logits"][...])

    def dense_kl(logits):
      logp = jax.nn.log_softmax(logits)
      return jnp.sum(jnp.exp(logp) * (logp - jnp.log(teacher_probs)))

    expected = jax.grad(dense_kl)(jnp.zeros(2))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    before = dense_kl(jnp.zeros(2))
    after = dense_kl(-0.1 * jnp.asarray(actual))
    self.assertLess(float(after), float(before))

  def test_masked_nonfinite_scores_never_affect_advantages_or_statistics(self):
    payload = _payload(
        completion_mask=np.array([1, 0], dtype=np.float32),
        advantages=np.array([2, np.nan], dtype=np.float32),
        old_per_token_logps=np.array([-1, np.nan], dtype=np.float32),
    )
    teacher = np.array([-2, np.nan], dtype=np.float32)
    updated, stats = on_policy.apply_opd_to_payloads(
        [payload],
        [teacher],
        on_policy.OnPolicyDistillationConfig(
            coefficient=0.5, task_reward_coefficient=0.25
        ),
    )
    np.testing.assert_array_equal(updated[0].advantages, [0, 0])
    self.assertTrue(np.isnan(payload.advantages[1]))
    self.assertTrue(np.isnan(teacher[1]))
    self.assertEqual(stats.token_count, 1)
    self.assertEqual(stats.reverse_kl_sum, 1)
    self.assertEqual(stats.teacher_logprob_sum, -2)

  def test_equal_teacher_and_student_produce_zero_pure_opd(self):
    payload = _payload(advantages=np.array([9, 5], dtype=np.float32))
    updated, stats = on_policy.apply_opd_to_payloads(
        [payload],
        [payload.old_per_token_logps],
        on_policy.OnPolicyDistillationConfig(),
    )
    np.testing.assert_array_equal(updated[0].advantages, [0, 0])
    np.testing.assert_array_equal(payload.advantages, [9, 5])
    self.assertEqual(stats.reverse_kl_sum, 0)

  def test_zero_coefficients_need_no_teacher_or_student_scores(self):
    payload = _payload(old_per_token_logps=None)
    updated, _ = on_policy.apply_opd_to_payloads(
        [payload],
        None,
        on_policy.OnPolicyDistillationConfig(coefficient=0),
    )
    np.testing.assert_array_equal(updated[0].advantages, [0, 0])

  def test_all_masked_payload_needs_no_teacher(self):
    payload = _payload(
        completion_mask=np.zeros(2, dtype=np.float32),
        old_per_token_logps=None,
    )
    updated, stats = on_policy.apply_opd_to_payloads(
        [payload], None, on_policy.OnPolicyDistillationConfig()
    )
    np.testing.assert_array_equal(updated[0].advantages, [0, 0])
    self.assertEqual(stats.token_count, 0)

  def test_rejects_invalid_scores_and_shapes(self):
    examples = (
        ([_payload()], None),
        ([_payload(old_per_token_logps=None)], [np.zeros(2)]),
        ([_payload()], []),
        ([_payload()], [np.zeros(1)]),
        ([_payload()], [np.array([0, np.nan])]),
        ([_payload(advantages=np.zeros(1))], [np.zeros(2)]),
        ([_payload(completion_mask=np.array([1, -1]))], [np.zeros(2)]),
    )
    for payloads, scores in examples:
      with self.subTest(payloads=payloads, scores=scores):
        with self.assertRaises(ValueError):
          on_policy.apply_opd_to_payloads(
              payloads, scores, on_policy.OnPolicyDistillationConfig()
          )

  def test_rejects_nonfinite_and_negative_coefficients(self):
    for field in ("coefficient", "task_reward_coefficient"):
      for value in (float("nan"), float("inf"), -0.1):
        with self.subTest(field=field, value=value):
          with self.assertRaises(ValueError):
            on_policy.OnPolicyDistillationConfig(**{field: value})


if __name__ == "__main__":
  unittest.main()
