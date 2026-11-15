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

"""Clipped importance-sampling policy optimization for collective rewards.

Avalon Eq. (1) caps the ratio at epsilon_high=5 (not 1+epsilon_high).
CISPO differentiates log pi multiplied by a detached clipped ratio. This
differs from PPO clipping: an upper-clipped positive-advantage token still
contributes a gradient. Import this module before constructing its config.
"""

import math

from flax import nnx
import jax
import jax.numpy as jnp
from tunix.rl import common
from tunix.rl import function_registry
from tunix.sft import utils as sft_utils


def token_objective(current, old, advantages, mask, *, cap=5.0):
  """Detached capped importance weights times the current sampled logprob."""
  active = mask > 0
  current = jnp.where(active, current.astype(jnp.float32), 0.0)
  old = jnp.where(active, jax.lax.stop_gradient(old.astype(jnp.float32)), 0.0)
  advantage = jnp.where(active, jax.lax.stop_gradient(advantages), 0.0)
  ratio = jnp.exp(jnp.clip(current - old, -20.0, 20.0))
  weights = jax.lax.stop_gradient(jnp.minimum(ratio, cap))
  return -weights * advantage * current, weights


@function_registry.register_policy_loss_fn("cispo")
def cispo_loss_fn(model, train_example, algo_config, pad_id, eos_id, **kwargs):
  """Native trainer loss using exact sampled-action tokens and attention masks."""
  if train_example.action_support_ids is not None:
    raise ValueError("This CISPO profile requires unrestricted sampled tokens.")
  if (
      not math.isfinite(algo_config.epsilon_high)
      or algo_config.epsilon_high <= 0
  ):
    raise ValueError("CISPO requires a positive importance ratio cap.")
  if train_example.old_per_token_logps is None:
    raise ValueError("CISPO requires fixed behavior/old log probabilities.")
  if algo_config.beta != 0 or algo_config.force_compute_kl:
    raise ValueError("This collective CISPO profile has no reference KL.")
  if algo_config.sampler_is is not None or algo_config.sampler_rs is not None:
    raise ValueError("Supply detached sampler correction weights explicitly.")
  graph, state = nnx.split(model)
  attention = train_example.completion_attention_mask
  if attention is None:
    attention = train_example.completion_mask
  current, entropy = common.compute_per_token_logps(
      graph,
      state,
      train_example.prompt_ids,
      train_example.completion_ids,
      pad_id,
      eos_id,
      stop_gradient=False,
      return_entropy=True,
      token_mask=jnp.concatenate(
          [train_example.prompt_mask, attention], axis=1
      ),
      temperature=algo_config.temperature,
      chunk_size=kwargs.get("compute_logps_chunk_size", 0),
      segment_ids=getattr(train_example, "segment_ids", None),
      segment_positions=getattr(train_example, "segment_positions", None),
  )
  mask = train_example.completion_mask
  adv = train_example.advantages
  if adv.ndim == 1:
    adv = adv[:, None]
  losses, weights = token_objective(
      current,
      train_example.old_per_token_logps,
      adv,
      mask,
      cap=algo_config.epsilon_high,
  )
  if train_example.sampler_is_weights is not None:
    correction = jax.lax.stop_gradient(train_example.sampler_is_weights)
    losses *= jnp.where(mask > 0, correction, 0.0)
  denom = jnp.sum(mask)
  return sft_utils.LossOutput(
      primary_loss=common.aggregate_loss(
          losses,
          mask,
          algo_config.loss_agg_mode,
          segment_ids=getattr(train_example, "segment_ids", None),
          num_segments=getattr(train_example, "num_segments", None),
      ),
      aux_metrics={
          "entropy": sft_utils.WeightedMetric(
              jnp.sum(jnp.where(mask > 0, entropy, 0.0)), denom, min_denom=1.0
          ),
          "cispo/weight_mean": sft_utils.WeightedMetric(
              jnp.sum(jnp.where(mask > 0, weights, 0.0)), denom, min_denom=1.0
          ),
          "cispo/capped_fraction": sft_utils.WeightedMetric(
              jnp.sum((mask > 0) & (weights >= algo_config.epsilon_high)),
              denom,
              min_denom=1.0,
          ),
      },
  )
