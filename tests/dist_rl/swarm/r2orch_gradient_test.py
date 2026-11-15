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

"""Independent PyTorch oracle for the R2Orch ordinary-token objective.

Equations checked against upstream training.py at
12ceb7563aa50439687f21ba5da60ad74a75eb89. This does not cover the upstream
finite-support relation policy, which needs a separate categorical objective.
"""

import dataclasses

from flax import nnx
import numpy as np
import pytest
import torch
from examples.swarm_math import cpu_fixture
from tests.dist_rl.swarm.gradient_test import payload_and_weights
from tunix.dist_rl.swarm import director_loss
from tunix.rl import algorithm_config


def director_config(beta=0.005):
  config = algorithm_config.GRPOConfig(
      num_generations=5,
      beta=beta,
      epsilon=0.2,
      epsilon_high=0.28,
      epsilon_c=None,
      loss_agg_mode="token-mean",
      kl_loss_mode="low_var_kl",
      use_rollout_logps=True,
  )
  config.temperature = 1.0
  return config


def torch_oracle(weights, payload, beta):
  """Independent autodiff with fixed actor-old, behavior TIS and initial ref."""
  w = torch.tensor(weights, requires_grad=True)
  previous = np.concatenate(
      [payload.prompt_ids[:, -1:], payload.completion_ids[:, :-1]], axis=1
  )
  current = torch.log_softmax(w[torch.tensor(previous, dtype=torch.long)], -1)
  current = current.gather(
      -1, torch.tensor(payload.completion_ids, dtype=torch.long)[..., None]
  )[..., 0]
  old = torch.tensor(payload.old_per_token_logps)
  advantage = torch.tensor(payload.advantages)
  ratio = (current - old).clamp(-20, 20).exp()
  pg = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(0.8, 1.28))
  delta = (torch.tensor(payload.ref_per_token_logps) - current).clamp(-20, 20)
  kl = delta.exp() - delta - 1
  mask = torch.tensor(payload.completion_mask).bool()
  loss = (pg * torch.tensor(payload.sampler_is_weights) + beta * kl)[
      mask
  ].mean()
  loss.backward()
  return loss.detach().numpy(), w.grad.numpy()


@pytest.mark.parametrize("beta", [0.0, 0.005, 0.2])
def test_asymmetric_clip_fixed_tis_and_reference_kl_match_torch(beta):
  p, w = payload_and_weights()
  p = dataclasses.replace(
      p,
      sampler_is_weights=np.array(
          [[2, 0.5, 1], [1.2, 1, 2], [0.3, 1, 1.7]], np.float32
      ),
      ref_per_token_logps=p.old_per_token_logps - 0.7,
      sampler_agreement_applied=True,
  )
  config = director_config(beta)

  def native(model, payload=p):
    return director_loss.director_loss_fn(
        model, payload, config, pad_id=0, eos_id=0
    ).primary_loss.compute()

  loss, gradient = nnx.value_and_grad(native)(cpu_fixture.TinyPolicy(w))
  expected, grad = torch_oracle(w, p, beta)
  np.testing.assert_allclose(loss, expected, atol=2e-6)
  np.testing.assert_allclose(
      gradient["logits"][...], grad, atol=2e-6, rtol=2e-5
  )

  # A second optimization evaluation must retain the original old policy and
  # TIS weights, even though the actor is no longer at collection parameters.
  updated = w - 0.03 * grad
  loss2, gradient2 = nnx.value_and_grad(native)(cpu_fixture.TinyPolicy(updated))
  expected2, grad2 = torch_oracle(updated, p, beta)
  np.testing.assert_allclose(loss2, expected2, atol=2e-6)
  np.testing.assert_allclose(
      gradient2["logits"][...], grad2, atol=2e-6, rtol=2e-5
  )

  # Prompt/context-only tokens never enter policy, TIS, or KL loss.
  changed = dataclasses.replace(
      p,
      advantages=np.where(p.completion_mask, p.advantages, 1e6),
      ref_per_token_logps=np.where(
          p.completion_mask, p.ref_per_token_logps, -50
      ),
      sampler_is_weights=np.where(p.completion_mask, p.sampler_is_weights, 1e6),
  )
  changed_loss, changed_grad = nnx.value_and_grad(lambda m: native(m, changed))(
      cpu_fixture.TinyPolicy(w)
  )
  np.testing.assert_allclose(changed_loss, loss, atol=1e-7)
  np.testing.assert_allclose(changed_grad["logits"][...], grad, atol=2e-6)


def test_zero_advantage_still_has_reference_kl_gradient():
  p, w = payload_and_weights()
  p = dataclasses.replace(
      p,
      advantages=np.zeros_like(p.advantages),
      sampler_is_weights=np.ones_like(p.advantages),
      ref_per_token_logps=p.old_per_token_logps - 0.7,
      sampler_agreement_applied=True,
  )
  config = director_config()
  loss, grad = nnx.value_and_grad(
      lambda m: director_loss.director_loss_fn(
          m, p, config, pad_id=0, eos_id=0
      ).primary_loss.compute()
  )(cpu_fixture.TinyPolicy(w))
  assert float(loss) > 0
  assert np.linalg.norm(grad["logits"][...]) > 0
