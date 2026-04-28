# Copyright 2025 Google LLC
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

"""PPO core algorithm implementations."""

import jax
import jax.numpy as jnp
from flax import nnx

from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl.algo_core import utils


@function_registry.register_policy_loss_fn("ppo")
def ppo_policy_loss_fn(
    model,
    train_example,
    algo_config,
    pad_id,
    eos_id,
    **kwargs,
):
  """PPO policy loss function."""
  epsilon = algo_config.epsilon
  beta = algo_config.beta

  completion_ids = train_example.completion_ids
  completion_mask = train_example.completion_mask

  graphdef, state = nnx.split(model)
  per_token_logps, logits = common.compute_per_token_logps(
      graphdef,
      state,
      prompt_tokens=train_example.prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_id,
      eos_id=eos_id,
      completion_mask=completion_mask,
      stop_gradient=False,
      return_logits=True,
  )

  advantages = train_example.advantages
  old_per_token_logps = train_example.old_per_token_logps

  seq_importance_ratio = jnp.exp(per_token_logps - old_per_token_logps)

  # Compute pg_clipfrac
  pg_losses_1 = -seq_importance_ratio * advantages
  pg_losses_2 = -jnp.clip(seq_importance_ratio, 1 - epsilon, 1 + epsilon) * advantages

  # add dual clip logic
  epsilon_c = getattr(algo_config, "epsilon_c", 3.0)
  if epsilon_c is None:
    epsilon_c = 3.0
  pg_loss_3 = -epsilon_c * advantages

  per_token_loss = jnp.maximum(pg_losses_1, pg_losses_2)
  unreduced_pg_clipfrac_lower = (
      (per_token_loss > pg_loss_3) & (advantages < 0.0)
  ).astype(jnp.float32)
  pg_clipfrac_lower = utils.masked_mean(
      unreduced_pg_clipfrac_lower, completion_mask
  )

  pg_loss_clipped_dual = jnp.minimum(pg_loss_3, per_token_loss)
  pg_losses = jnp.where(advantages < 0.0, pg_loss_clipped_dual, per_token_loss)

  aux = {
      "pg_clipfrac": utils.masked_mean(
          jnp.greater(pg_losses_2, pg_losses_1), completion_mask
      ),
      "pg_clipfrac_lower": pg_clipfrac_lower,
  }

  policy_loss = utils.masked_mean(pg_losses, completion_mask)
  loss = policy_loss

  if beta is not None and beta != 0.0:
    token_entropy = utils.compute_entropy_from_logits(logits)
    entropy_loss = utils.masked_mean(token_entropy, completion_mask)
    loss = loss - beta * entropy_loss
    aux["entropy"] = entropy_loss

  # kl penalty term logic as before
  kl_coef = getattr(algo_config, 'kl_coef', 0.0)
  if kl_coef > 0.0 and train_example.ref_per_token_logps is not None:
    kl = common.compute_kl_divergence(
        per_token_logps,
        train_example.ref_per_token_logps,
        "kl"
    )
    kl_loss = utils.masked_mean(kl, completion_mask)
    loss = loss + kl_coef * kl_loss
    aux["kl"] = kl_loss

  return loss, aux


def ppo_value_loss_fn(
    model: nnx.Module,
    train_example,
    algo_config,
    clip_range_value: float | None,
    pad_id: int,
    eos_id: int,
):
  """Computes the value loss for PPO."""

  prompt_ids, completion_ids, completion_mask = (
      train_example.prompt_ids,
      train_example.completion_ids,
      train_example.completion_mask,
  )
  # ====== Loss ======
  values = train_example.old_values
  returns = train_example.returns

  segment_ids = getattr(train_example, "segment_ids", None)
  if segment_ids is not None:
    # For packed sequences, prompt_ids is empty and completion_ids holds the full sequence.
    # We predict values for token t using the model's output at t-1.
    logits_to_keep = completion_ids.shape[1] - 1
  else:
    logits_to_keep = completion_ids.shape[1]

  # Get new values.
  vpreds = common.compute_score(
      model,
      prompt_ids,
      completion_ids,
      pad_id,
      eos_id,
      stop_gradient=False,
      segment_ids=segment_ids,
      segment_positions=getattr(train_example, "segment_positions", None),
  )
  vpreds = vpreds[:, -logits_to_keep - 1 : -1]

  if segment_ids is not None:
    # Pad the first token's value with 0.0, since it has no preceding token to predict it.
    vpreds = jnp.pad(vpreds, ((0, 0), (1, 0)), constant_values=0.0)
  vpred_clipped = jnp.clip(
      vpreds, values - clip_range_value, values + clip_range_value
  )
  vf_losses1 = jnp.square(vpreds - returns)
  vf_losses2 = jnp.square(vpred_clipped - returns)

  clipped_vf_losses = jnp.maximum(vf_losses1, vf_losses2)
  # "token mean" style of normalisation.
  vf_loss = 0.5 * utils.masked_mean(clipped_vf_losses, completion_mask)

  aux = {
      "vf_loss": vf_loss,
      "vpred_mean": utils.masked_mean(vpreds, completion_mask),
      "vf_clipfrac": utils.masked_mean(
          jnp.greater(vf_losses2, vf_losses1), completion_mask
      ),
      "return_mean": utils.masked_mean(returns, completion_mask),
  }

  return vf_loss, aux
