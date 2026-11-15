# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CISPO exact gradients, including tokens whose ratio exceeds the cap."""

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from examples.swarm_math import cpu_fixture
from tests.dist_rl.swarm.gradient_test import payload_and_weights
from tunix.rl import algorithm_config
from tunix.rl import cispo


def config():
  c = algorithm_config.GRPOConfig(
      policy_loss_fn="cispo",
      beta=0,
      epsilon_high=5,
      num_generations=16,
      loss_agg_mode="token-mean",
  )
  c.temperature = 1.0
  return c


@pytest.mark.parametrize("correction", [False, True])
def test_every_native_parameter_gradient_matches_torch(correction):
  p, weights = payload_and_weights()
  p = dataclasses.replace(p, old_per_token_logps=p.old_per_token_logps - 2.0)
  if correction:
    p = dataclasses.replace(
        p, sampler_is_weights=np.full_like(p.advantages, 0.7)
    )
  current = torch.tensor(weights, requires_grad=True)
  previous = np.concatenate(
      [p.prompt_ids[:, -1:], p.completion_ids[:, :-1]], axis=1
  )
  logps = torch.log_softmax(
      current[torch.tensor(previous, dtype=torch.long)], -1
  )
  logps = logps.gather(
      -1, torch.tensor(p.completion_ids, dtype=torch.long)[..., None]
  )[..., 0]
  ratio = (logps - torch.tensor(p.old_per_token_logps)).clamp(-20, 20).exp()
  losses = -ratio.clamp(max=5).detach() * torch.tensor(p.advantages) * logps
  if correction:
    losses *= torch.tensor(p.sampler_is_weights)
  mask = torch.tensor(p.completion_mask).bool()
  loss = losses[mask].mean()
  loss.backward()

  def actual(model, payload=p):
    return cispo.cispo_loss_fn(
        model, payload, config(), 0, 0
    ).primary_loss.compute()

  value, gradient = nnx.value_and_grad(actual)(cpu_fixture.TinyPolicy(weights))
  np.testing.assert_allclose(value, loss.detach(), rtol=1e-5, atol=1e-6)
  np.testing.assert_allclose(
      gradient["logits"][...], current.grad, rtol=1e-5, atol=1e-6
  )
  changed = dataclasses.replace(
      p, advantages=np.where(p.completion_mask, p.advantages, 1e6)
  )
  changed_value, changed_gradient = nnx.value_and_grad(
      lambda m: actual(m, changed)
  )(cpu_fixture.TinyPolicy(weights))
  np.testing.assert_array_equal(value, changed_value)
  np.testing.assert_array_equal(
      gradient["logits"][...], changed_gradient["logits"][...]
  )


def test_upper_cap_preserves_gradient_and_detaches_importance_weight():
  current = jnp.array([-1.0, -2.0, -3.0])
  old = jnp.array([-4.0, -2.0, -1.0])
  advantage = jnp.array([2.0, -1.0, 0.5])
  mask = jnp.ones(3)
  gradient = jax.grad(
      lambda x: cispo.token_objective(x, old, advantage, mask)[0].sum()
  )(current)
  expected = -np.minimum(np.exp(np.asarray(current - old)), 5.0) * advantage
  np.testing.assert_allclose(gradient, expected, rtol=1e-6)
  assert gradient[0] == -10  # PPO would drop this upper-clipped positive token.
  old_gradient = jax.grad(
      lambda x: cispo.token_objective(current, x, advantage, mask)[0].sum()
  )(old)
  np.testing.assert_array_equal(old_gradient, 0)


def test_nonfinite_padding_has_zero_loss_and_gradient():
  old = jnp.array([-1.0, jnp.nan])
  gradient = jax.grad(
      lambda x: cispo.token_objective(
          x, old, jnp.ones(2), jnp.array([1.0, 0.0])
      )[0].sum()
  )(old)
  np.testing.assert_array_equal(gradient, [-1.0, 0.0])


def test_rejects_implicit_reference_kl():
  p, weights = payload_and_weights()
  c = config()
  c.beta = 0.1
  with pytest.raises(ValueError, match="no reference KL"):
    cispo.cispo_loss_fn(cpu_fixture.TinyPolicy(weights), p, c, 0, 0)
