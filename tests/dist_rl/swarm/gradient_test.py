"""Native JAX loss/gradient versus the actual pinned PyTorch PPO objective."""

import dataclasses

from flax import nnx
import numpy as np
import pytest
from examples.swarm_math import cpu_fixture
from tests.dist_rl.swarm import reference
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import swarm_program
from tunix.rl import algo_core
from tunix.rl import packing


def payload_and_weights():
  weights = np.random.default_rng(31).normal(0, 0.3, (8, 8)).astype(np.float32)
  prompts = np.array([[0, 1, 2], [1, 0, 3], [0, 0, 2]], dtype=np.int32)
  completions = np.array([[4, 0, 0], [6, 0, 7], [5, 4, 0]], dtype=np.int32)
  previous = np.concatenate([prompts[:, -1:], completions[:, :-1]], axis=1)
  logps = weights - np.log(np.exp(weights).sum(axis=-1, keepdims=True))
  sampled = np.take_along_axis(
      logps[previous], completions[..., None], axis=-1
  )[..., 0]
  ratios = np.array(
      [[1.4, 0.5, 1], [4, 1, 0.5], [1.1, 0.9, 1.05]], dtype=np.float32
  )
  return (
      datatypes.RLTrainerPayload(
          prompt_ids=prompts,
          prompt_mask=np.array(
              [[0, 1, 1], [1, 1, 1], [0, 0, 1]], dtype=np.float32
          ),
          completion_ids=completions,
          completion_mask=np.array(
              [[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=np.float32
          ),
          completion_attention_mask=np.array(
              [[1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=np.float32
          ),
          advantages=np.array(
              [[1, -1, 100], [-2, 100, 2], [0.4, -0.3, 1]], dtype=np.float32
          ),
          old_per_token_logps=sampled - np.log(ratios),
      ),
      weights,
  )


def test_native_loss_and_every_parameter_gradient_match_source():
  payload, weights = payload_and_weights()
  config = swarm_program.shared_grpo_config()

  def loss(model):
    return algo_core.grpo_loss_fn(
        model, payload, config, pad_id=0, eos_id=0
    ).primary_loss.compute()

  actual_loss, actual_gradient = nnx.value_and_grad(loss)(
      cpu_fixture.TinyPolicy(weights)
  )
  expected_loss, expected_gradient = reference.loss_and_gradient(
      weights, payload
  )
  np.testing.assert_allclose(actual_loss, expected_loss, atol=2e-6)
  np.testing.assert_allclose(
      actual_gradient["logits"][...], expected_gradient, atol=2e-6, rtol=2e-5
  )
  # Positive upper clip, negative lower clip and negative dual clip all fire.
  np.testing.assert_array_equal(actual_gradient["logits"][...][3], 0.0)
  # An independent central difference checks nonzero derivatives as well.
  for row, col in ((5, 4), (4, 0), (2, 5)):
    plus, minus = weights.copy(), weights.copy()
    plus[row, col] += 0.001
    minus[row, col] -= 0.001
    finite_difference = (
        reference.loss_and_gradient(plus, payload)[0]
        - reference.loss_and_gradient(minus, payload)[0]
    ) / 0.002
    np.testing.assert_allclose(
        actual_gradient["logits"][...][row, col], finite_difference, atol=2e-4
    )


def test_masked_values_do_not_change_loss_or_gradient():
  payload, weights = payload_and_weights()
  config = swarm_program.shared_grpo_config()

  def gradients(p):
    def loss(model):
      return algo_core.grpo_loss_fn(
          model, p, config, pad_id=0, eos_id=0
      ).primary_loss.compute()

    return nnx.value_and_grad(loss)(cpu_fixture.TinyPolicy(weights))

  baseline_loss, baseline_grad = gradients(payload)
  changed = dataclasses.replace(
      payload,
      advantages=np.where(payload.completion_mask, payload.advantages, -1e6),
      old_per_token_logps=np.where(
          payload.completion_mask, payload.old_per_token_logps, 1e5
      ),
  )
  changed_loss, changed_grad = gradients(changed)
  np.testing.assert_array_equal(changed_loss, baseline_loss)
  np.testing.assert_array_equal(
      changed_grad["logits"][...], baseline_grad["logits"][...]
  )


@pytest.mark.parametrize("python_fallback", [False, True])
def test_assembler_preserves_attention_mask_separate_from_loss_mask(
    monkeypatch, python_fallback
):
  if python_fallback:
    monkeypatch.setattr(packing, "_packing_ext", None)
  payload, _ = payload_and_weights()
  rows = [
      dataclasses.replace(
          payload,
          **{
              field: np.asarray(getattr(payload, field))[i, :length]
              for field in (
                  "completion_ids",
                  "completion_mask",
                  "completion_attention_mask",
                  "advantages",
                  "old_per_token_logps",
              )
          },
          prompt_ids=payload.prompt_ids[i, -2:],
          prompt_mask=payload.prompt_mask[i, -2:]
      )
      for i, length in enumerate((2, 3, 3))
  ]
  assembler = batch_assembly.PaddedBatchAssembler(
      batch_size=4,
      num_generations=3,
      mini_batch_size=1,
      max_prompt_length=4,
      max_response_length=4,
      pad_id=0,
  )
  batch = list(assembler.feed(rows))[0].payload
  np.testing.assert_array_equal(
      batch.completion_attention_mask,
      [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
  )
  np.testing.assert_array_equal(batch.completion_mask[1], [1, 0, 1, 0])
  assert batch.completion_ids[0, 1] == 0 and batch.completion_mask[0, 1] == 1
