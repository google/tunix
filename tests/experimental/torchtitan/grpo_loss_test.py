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

"""Numerical parity tests between PyTorch GRPO Loss and Tunix algo_core.grpo_loss_fn."""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import torch
from tunix.experimental.torchtitan import grpo_loss
from tunix.rl import algo_core
from tunix.rl import algorithm_config
from tunix.rl import common


class GRPOLossParityTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(42)
    torch.manual_seed(42)

  def test_no_item_calls_in_grpo_loss_source(self):
    source = inspect.getsource(grpo_loss)
    self.assertNotIn(
        ".item(",
        source,
        "grpo_loss.py must not contain any .item() calls (pure tensor hot-path"
        " required).",
    )

  def test_compact_left_padded_batch_np_and_static_gather(self):
    batch_size = 3
    completion_len = 4
    pad_id = 99

    prompt_ids = np.array(
        [
            [pad_id, pad_id, pad_id, 10, 11, 12, 13, 14],  # len 5
            [pad_id, pad_id, pad_id, pad_id, pad_id, 20, 21, 22],  # len 3
            [pad_id, 30, 31, 32, 33, 34, 35, 36],  # len 7
        ],
        dtype=np.int32,
    )
    prompt_mask = (prompt_ids != pad_id).astype(np.int32)

    completion_ids = np.array(
        [
            [101, 102, 103, 104],
            [201, 202, 203, 204],
            [301, 302, 303, 304],
        ],
        dtype=np.int32,
    )

    compact_tokens, positions, prompt_lens = (
        grpo_loss.compact_left_padded_batch_np(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            pad_id=pad_id,
            total_length=16,
        )
    )

    np.testing.assert_array_equal(prompt_lens, [5, 3, 7])
    self.assertEqual(compact_tokens.shape, (3, 16))
    self.assertEqual(positions.shape, (3 * 16,))

    # Row 0: 5 prompt + 4 completion = 9 tokens, rest padded
    np.testing.assert_array_equal(
        compact_tokens[0, :9], [10, 11, 12, 13, 14, 101, 102, 103, 104]
    )
    np.testing.assert_array_equal(compact_tokens[0, 9:], [pad_id] * 7)

    # Row 1: 3 prompt + 4 completion = 7 tokens, rest padded
    np.testing.assert_array_equal(
        compact_tokens[1, :7], [20, 21, 22, 201, 202, 203, 204]
    )
    np.testing.assert_array_equal(compact_tokens[1, 7:], [pad_id] * 9)

    # Test static gather extract_completion_logits
    vocab_size = 10
    simulated_logits = torch.randn(batch_size, 16, vocab_size)
    extracted = grpo_loss.extract_completion_logits(
        logits=simulated_logits,
        prompt_lens=torch.from_numpy(prompt_lens),
        completion_width=completion_len,
    )
    self.assertEqual(extracted.shape, (batch_size, completion_len, vocab_size))

    # Verify correctness against explicit slices
    for i, p_i in enumerate(prompt_lens):
      expected_slice = simulated_logits[i, p_i - 1 : p_i - 1 + completion_len]
      torch.testing.assert_close(extracted[i], expected_slice)

  @parameterized.named_parameters(
      dict(
          testcase_name="seq_mean_tok_mean_no_beta",
          loss_agg_mode="sequence-mean-token-mean",
          beta=0.0,
      ),
      dict(
          testcase_name="token_mean_no_beta",
          loss_agg_mode="token-mean",
          beta=0.0,
      ),
      dict(
          testcase_name="seq_mean_tok_mean_with_beta",
          loss_agg_mode="sequence-mean-token-mean",
          beta=0.05,
      ),
      dict(
          testcase_name="token_mean_with_beta",
          loss_agg_mode="token-mean",
          beta=0.05,
      ),
  )
  def test_numerical_parity_against_tunix(
      self, loss_agg_mode: str, beta: float
  ):
    batch_size = 4
    prompt_len = 16
    completion_len = 32
    pad_id = 0
    eos_id = 2

    # Left-padded prompts
    prompt_ids = np.random.randint(
        10, 1000, size=(batch_size, prompt_len), dtype=np.int32
    )
    prompt_mask = np.zeros((batch_size, prompt_len), dtype=np.int32)
    for i in range(batch_size):
      valid_len = np.random.randint(4, prompt_len + 1)
      prompt_mask[i, -valid_len:] = 1
      prompt_ids[i, :-valid_len] = pad_id

    # Right-padded completions with EOS
    completion_ids = np.random.randint(
        10, 1000, size=(batch_size, completion_len), dtype=np.int32
    )
    completion_mask = np.zeros((batch_size, completion_len), dtype=np.int32)
    for i in range(batch_size):
      resp_len = np.random.randint(5, completion_len - 2)
      completion_mask[i, :resp_len] = 1
      completion_ids[i, resp_len] = eos_id
      completion_ids[i, resp_len + 1 :] = pad_id

    # Simulated logps and advantages
    per_token_logps_np = np.random.uniform(
        -4.0, -0.5, size=(batch_size, completion_len)
    ).astype(np.float32)
    old_per_token_logps_np = (
        per_token_logps_np
        + np.random.uniform(-0.2, 0.2, size=(batch_size, completion_len))
    ).astype(np.float32)
    if beta > 0.0:
      ref_per_token_logps_np = (
          per_token_logps_np
          + np.random.uniform(-0.3, 0.3, size=(batch_size, completion_len))
      ).astype(np.float32)
    else:
      ref_per_token_logps_np = None
    advantages_np = np.random.uniform(-2.0, 2.0, size=(batch_size,)).astype(
        np.float32
    )

    # 1. Compute in PyTorch
    torch_logps = torch.tensor(per_token_logps_np, requires_grad=True)
    torch_mask = torch.tensor(completion_mask)
    torch_adv = torch.tensor(advantages_np)
    torch_old_logps = torch.tensor(old_per_token_logps_np)
    torch_ref_logps = (
        torch.tensor(ref_per_token_logps_np)
        if ref_per_token_logps_np is not None
        else None
    )

    algo_cfg = algorithm_config.GRPOConfig(
        loss_algo="grpo",
        loss_agg_mode=loss_agg_mode,
        epsilon=0.2,
        epsilon_high=0.2,
        beta=beta,
        use_rollout_logps=True,
    )
    algo_cfg.temperature = 1.0

    loss_output = grpo_loss.compute_grpo_loss(
        per_token_logps=torch_logps,
        completion_mask=torch_mask,
        advantages=torch_adv,
        old_per_token_logps=torch_old_logps,
        ref_per_token_logps=torch_ref_logps,
        algo_config=algo_cfg,
    )
    torch_loss_mean = loss_output.unreduced_loss / torch.clamp(
        loss_output.local_denom, min=1.0
    )
    torch_loss_mean.backward()
    torch_grad = torch_logps.grad.numpy()

    # 2. Compute in Tunix (JAX)
    jax_logps = jnp.asarray(per_token_logps_np)
    jax_mask = jnp.asarray(completion_mask)
    jax_adv = jnp.asarray(advantages_np)
    jax_old_logps = jnp.asarray(old_per_token_logps_np)
    jax_ref_logps = (
        jnp.asarray(ref_per_token_logps_np)
        if ref_per_token_logps_np is not None
        else None
    )

    class DummyModel:
      pass

    train_example = common.TrainExample(
        prompt_ids=jnp.zeros((batch_size, prompt_len), dtype=jnp.int32),
        prompt_mask=jnp.ones((batch_size, prompt_len), dtype=jnp.int32),
        completion_ids=jnp.asarray(completion_ids),
        completion_mask=jax_mask,
        advantages=jax_adv,
        old_per_token_logps=jax_old_logps,
        ref_per_token_logps=jax_ref_logps,
    )

    def jax_loss_fn(p_logps):
      with mock.patch.object(
          common,
          "compute_per_token_logps",
          return_value=(p_logps, jnp.zeros_like(p_logps)),
      ):
        with mock.patch("flax.nnx.split", return_value=(None, None)):
          loss_output_jax = algo_core.grpo_loss_fn(
              model=DummyModel(),
              train_example=train_example,
              algo_config=algo_cfg,
              pad_id=pad_id,
              eos_id=eos_id,
          )
          total_loss = loss_output_jax.primary_loss
          return total_loss.compute(), total_loss

    jax_val_grad_fn = jax.value_and_grad(jax_loss_fn, has_aux=True)
    (jax_reduced_loss, _), jax_grad = jax_val_grad_fn(jax_logps)

    # 3. Assert parity
    abs_loss_diff = abs(
        float(torch_loss_mean.detach().numpy()) - float(jax_reduced_loss)
    )
    self.assertLess(
        abs_loss_diff,
        1e-5,
        f"Loss mismatch: PyTorch {torch_loss_mean.detach().numpy()} vs JAX"
        f" {float(jax_reduced_loss)}",
    )

    valid_mask = completion_mask > 0
    t_grad_valid = torch_grad[valid_mask]
    j_grad_valid = np.asarray(jax_grad)[valid_mask]

    cos_sim = np.dot(t_grad_valid, j_grad_valid) / (
        np.linalg.norm(t_grad_valid) * np.linalg.norm(j_grad_valid) + 1e-12
    )
    self.assertGreater(
        cos_sim,
        0.99999,
        f"Gradient cosine similarity {cos_sim} < 0.99999",
    )

  def test_low_var_kl_and_clamp_and_force_compute(self):
    batch_size = 2
    completion_len = 8

    per_token_logps_np = np.full(
        (batch_size, completion_len), -1.0, dtype=np.float32
    )
    ref_per_token_logps_np = np.full(
        (batch_size, completion_len), -3.0, dtype=np.float32
    )
    old_per_token_logps_np = np.full(
        (batch_size, completion_len), -1.0, dtype=np.float32
    )
    completion_mask = np.ones((batch_size, completion_len), dtype=np.int32)
    advantages_np = np.ones((batch_size,), dtype=np.float32)

    torch_logps = torch.tensor(per_token_logps_np, requires_grad=True)
    torch_mask = torch.tensor(completion_mask)
    torch_adv = torch.tensor(advantages_np)
    torch_old_logps = torch.tensor(old_per_token_logps_np)
    torch_ref_logps = torch.tensor(ref_per_token_logps_np)

    # beta=0.0 and force_compute_kl=True should compute KL metric
    cfg = algorithm_config.GRPOConfig(
        loss_algo="grpo",
        loss_agg_mode="token-mean",
        epsilon=0.2,
        epsilon_high=None,
        beta=0.0,
        use_rollout_logps=True,
        kl_loss_mode="low_var_kl",
        kl_clamp_value=1.5,
        force_compute_kl=True,
    )
    # Also verify runtime resilience if epsilon_high is cleared after init.
    cfg.epsilon_high = None
    loss_output = grpo_loss.compute_grpo_loss(
        per_token_logps=torch_logps,
        completion_mask=torch_mask,
        advantages=torch_adv,
        old_per_token_logps=torch_old_logps,
        ref_per_token_logps=torch_ref_logps,
        algo_config=cfg,
    )
    self.assertIn("kl", loss_output.metric_sums)
    kl_val = float(
        (loss_output.metric_sums["kl"] / loss_output.token_denom)
        .detach()
        .numpy()
    )
    # log_ratio = ref - current = -3.0 - (-1.0) = -2.0
    # low_var_kl: exp(-2.0) - 1.0 - (-2.0) = ~1.1353 <= 1.5
    self.assertAlmostEqual(kl_val, 1.135335, places=4)

    # Now with high clamp value where clamping happens
    ref_large_diff = np.full(
        (batch_size, completion_len), -10.0, dtype=np.float32
    )
    torch_ref_large = torch.tensor(ref_large_diff)
    cfg_clamped = algorithm_config.GRPOConfig(
        loss_algo="grpo",
        loss_agg_mode="token-mean",
        epsilon=0.2,
        epsilon_high=0.2,
        beta=0.0,
        use_rollout_logps=True,
        kl_loss_mode="low_var_kl",
        kl_clamp_value=0.5,
        force_compute_kl=True,
    )
    loss_output_clamped = grpo_loss.compute_grpo_loss(
        per_token_logps=torch_logps,
        completion_mask=torch_mask,
        advantages=torch_adv,
        old_per_token_logps=torch_old_logps,
        ref_per_token_logps=torch_ref_large,
        algo_config=cfg_clamped,
    )
    self.assertIn("kl", loss_output_clamped.metric_sums)
    kl_clamped_val = float(
        (
            loss_output_clamped.metric_sums["kl"]
            / loss_output_clamped.token_denom
        )
        .detach()
        .numpy()
    )
    self.assertAlmostEqual(kl_clamped_val, 0.5, places=4)

  def test_grpo_config_field_coverage(self):
    grpo_fields = {
        f.name for f in dataclasses.fields(algorithm_config.GRPOConfig)
    }
    self.assertTrue(
        grpo_loss._LOSS_FIELDS_IMPLEMENTED.isdisjoint(
            grpo_loss._FIELDS_IGNORED_BY_LOSS
        )
    )
    self.assertEqual(
        grpo_fields,
        grpo_loss._LOSS_FIELDS_IMPLEMENTED | grpo_loss._FIELDS_IGNORED_BY_LOSS,
    )

  def test_torch_policy_loss_registry_and_decorator(self):
    self.assertIn("grpo", grpo_loss.list_torch_policy_losses())
    default_plugin = grpo_loss.get_torch_policy_loss("grpo")
    self.assertIsInstance(default_plugin, algorithm_config.PolicyLossPlugin)
    self.assertIsInstance(default_plugin, grpo_loss.GRPOPolicyLoss)

    with self.assertRaisesRegex(ValueError, "Unknown PyTorch policy loss"):
      grpo_loss.get_torch_policy_loss("nonexistent_loss_plugin")

    @grpo_loss.register_torch_policy_loss("test_decorated_loss")
    class _DecoratedLoss:

      def validate(
          self,
          algo_config: algorithm_config.AlgorithmConfig,
          *,
          fuse_agreement_in_loss: bool = False,
      ) -> None:
        del algo_config, fuse_agreement_in_loss

      def __call__(
          self,
          inputs: algorithm_config.PolicyLossInputs[torch.Tensor],
          algo_config: algorithm_config.AlgorithmConfig,
      ) -> algorithm_config.PolicyLossOutput[torch.Tensor]:
        del algo_config
        loss_sum = -torch.sum(inputs.per_token_logps * inputs.completion_mask)
        denom = torch.sum(inputs.completion_mask).clamp(min=1.0)
        return algorithm_config.PolicyLossOutput(
            loss_sum=loss_sum,
            denominator=denom,
        )

    try:
      self.assertIn("test_decorated_loss", grpo_loss.list_torch_policy_losses())
      resolved = grpo_loss.get_torch_policy_loss("test_decorated_loss")
      self.assertIsInstance(resolved, _DecoratedLoss)

      with self.assertRaisesRegex(ValueError, "already registered"):
        grpo_loss.register_torch_policy_loss(
            "test_decorated_loss", _DecoratedLoss()
        )

      replacement = _DecoratedLoss()
      grpo_loss.register_torch_policy_loss(
          "test_decorated_loss", replacement, overwrite=True
      )
      self.assertIs(
          grpo_loss.get_torch_policy_loss("test_decorated_loss"), replacement
      )
    finally:
      with grpo_loss._REGISTRY_LOCK:
        grpo_loss._TORCH_POLICY_LOSS_REGISTRY.pop("test_decorated_loss", None)

    with self.assertRaisesRegex(TypeError, "TorchPolicyLoss"):
      grpo_loss.register_torch_policy_loss(
          "bad_plugin",
          cast(Any, object()),
      )

  def test_grpo_policy_loss_validate_and_call(self):
    plugin = grpo_loss.GRPOPolicyLoss()
    valid_cfg = algorithm_config.GRPOConfig(
        loss_algo="grpo",
        loss_agg_mode="token-mean",
        epsilon=0.2,
    )
    plugin.validate(valid_cfg)

    with self.assertRaisesRegex(TypeError, "GRPOConfig"):
      plugin.validate(algorithm_config.AlgorithmConfig())

    @dataclasses.dataclass(kw_only=True)
    class _ExtendedGRPOConfig(algorithm_config.GRPOConfig):
      new_unhandled_knob: bool = True

    with self.assertRaisesRegex(ValueError, "new_unhandled_knob"):
      plugin.validate(_ExtendedGRPOConfig())

    with self.assertRaisesRegex(ValueError, "policy_loss_fn='grpo'"):
      plugin.validate(algorithm_config.GRPOConfig(policy_loss_fn="ppo"))

    with self.assertRaisesRegex(ValueError, "fuse_agreement_in_loss"):
      plugin.validate(valid_cfg, fuse_agreement_in_loss=True)

    logps = torch.full((2, 4), -0.5, dtype=torch.float32, requires_grad=True)
    mask = torch.ones((2, 4), dtype=torch.float32)
    adv = torch.tensor([1.0, -1.0], dtype=torch.float32)
    inputs = algorithm_config.PolicyLossInputs(
        per_token_logps=logps,
        completion_mask=mask,
        advantages=adv,
        old_per_token_logps=logps.detach(),
    )
    out = plugin(inputs, valid_cfg)
    raw = grpo_loss.compute_grpo_loss(
        per_token_logps=logps,
        completion_mask=mask,
        advantages=adv,
        old_per_token_logps=logps.detach(),
        ref_per_token_logps=None,
        algo_config=valid_cfg,
    )
    self.assertIsInstance(out, algorithm_config.PolicyLossOutput)
    torch.testing.assert_close(out.loss_sum, raw.unreduced_loss)
    torch.testing.assert_close(out.denominator, raw.local_denom)
    assert out.token_denom is not None
    torch.testing.assert_close(out.token_denom, raw.token_denom)
    self.assertEqual(set(out.metrics.keys()), set(raw.metric_sums.keys()))

    with self.assertRaisesRegex(ValueError, "segment_ids"):
      plugin(
          algorithm_config.PolicyLossInputs(
              per_token_logps=logps,
              completion_mask=mask,
              advantages=adv,
              segment_ids=torch.ones((2, 4), dtype=torch.int32),
          ),
          valid_cfg,
      )
    with self.assertRaisesRegex(ValueError, "sampler_is_weights"):
      plugin(
          algorithm_config.PolicyLossInputs(
              per_token_logps=logps,
              completion_mask=mask,
              advantages=adv,
              sampler_is_weights=torch.ones((2, 4), dtype=torch.float32),
          ),
          valid_cfg,
      )


if __name__ == "__main__":
  absltest.main()
