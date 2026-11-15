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

"""R²Orch Director PPO with source-compatible ordinary and binary-action KL.

The source trains ordinary generated tokens with sampled k3 KL, and its
one-token off/on relation choices with exact KL(current || reference) over
the two-token support. Sampling, old scoring and current scoring must all
use that same support. TIS multiplies the policy term, never the KL anchor.
"""

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.swarm import proposer_loss
from tunix.rl import common
from tunix.rl import function_registry
from tunix.sft import utils as sft_utils


def policy_logprobs(model, payload, *, pad_id=0, eos_id=0, chunk_size=0):
  """Return chosen-token logps, entropy and optional binary distributions."""
  if any(
      value is not None
      for value in (
          payload.segment_ids,
          payload.segment_positions,
          payload.num_segments,
      )
  ):
    raise ValueError("Director calls currently require padded, unpacked rows.")
  graph, state = nnx.split(model)
  attention = payload.completion_attention_mask
  if attention is None:
    attention = payload.completion_mask
  values, entropy = common.compute_per_token_logps(
      graph,
      state,
      payload.prompt_ids,
      payload.completion_ids,
      pad_id,
      eos_id,
      stop_gradient=False,
      return_entropy=True,
      temperature=1.0,
      chunk_size=chunk_size,
      token_mask=jnp.concatenate([payload.prompt_mask, attention], axis=1),
  )
  support_logps = None
  if payload.action_support_ids is not None:
    ids = jnp.asarray(payload.action_support_ids)
    restricted = jnp.all(ids >= 0, axis=-1)
    safe_ids = jnp.where(restricted[:, None], ids, jnp.array([0, 1]))
    candidate_payload = dataclasses.replace(payload, completion_ids=safe_ids)
    support_logps = proposer_loss.candidate_logps(
        model,
        candidate_payload,
        pad_id=pad_id,
        eos_id=eos_id,
        chunk_size=chunk_size,
    )
    chosen = jnp.argmax(safe_ids == payload.completion_ids[:, :1], axis=-1)
    selected = jnp.take_along_axis(support_logps, chosen[:, None], axis=1)[:, 0]
    values = values.at[:, 0].set(jnp.where(restricted, selected, values[:, 0]))
    categorical_entropy = -jnp.sum(
        jnp.exp(support_logps) * support_logps, axis=-1
    )
    entropy = entropy.at[:, 0].set(
        jnp.where(restricted, categorical_entropy, entropy[:, 0])
    )
  return values.astype(jnp.float32), entropy, support_logps


def binary_logprobs(
    model, prompt_ids, support_ids, *, pad_id=0, eos_id=0, chunk_size=0
):
  """Native constrained sampling distribution for an original relation call."""
  prompt = jnp.asarray(prompt_ids, jnp.int32)[None, :]
  ids = jnp.asarray(support_ids, jnp.int32)[None, :]
  payload = datatypes.RLTrainerPayload(
      prompt_ids=prompt,
      prompt_mask=jnp.ones_like(prompt),
      completion_ids=ids,
      completion_mask=jnp.ones_like(ids),
      advantages=jnp.zeros_like(ids, dtype=jnp.float32),
  )
  return proposer_loss.candidate_logps(
      model, payload, pad_id=pad_id, eos_id=eos_id, chunk_size=chunk_size
  )[0]


def token_loss(
    current,
    old,
    reference,
    advantages,
    mask,
    *,
    beta,
    epsilon=0.2,
    epsilon_high=0.28,
    correction=None,
    support_ids=None,
    support_logps=None,
    reference_support_logps=None,
    kl_term_cap=0.0,
    reference_kl_weights=None,
):
  """Source asymmetric PPO + uncapped sampled k3 / exact categorical KL."""
  active = mask > 0
  current = jnp.where(active, current, 0.0)
  old = jnp.where(active, jax.lax.stop_gradient(old), 0.0)
  advantage = jnp.where(active, jax.lax.stop_gradient(advantages), 0.0)
  ratio = jnp.exp(jnp.clip(current - old, -20.0, 20.0))
  policy = jnp.maximum(
      -advantage * ratio,
      -advantage * jnp.clip(ratio, 1 - epsilon, 1 + epsilon_high),
  )
  if correction is not None:
    policy *= jnp.where(active, jax.lax.stop_gradient(correction), 0.0)
  kl = jnp.zeros_like(current)
  if beta:
    if reference is None:
      raise ValueError(
          "Director KL requires frozen reference log probabilities."
      )
    diff = jnp.clip(
        jnp.where(active, jax.lax.stop_gradient(reference), 0.0) - current,
        -20.0,
        20.0,
    )
    kl = jnp.exp(diff) - diff - 1.0
    if kl_term_cap > 0:
      kl = jnp.minimum(kl, kl_term_cap)
    if support_ids is not None:
      if reference_support_logps is None:
        raise ValueError(
            "Binary Director KL requires the complete reference distribution."
        )
      restricted = jnp.all(support_ids >= 0, axis=-1) & active[:, 0]
      safe_support = jnp.where(
          restricted[:, None], support_logps, -jnp.log(2.0)
      )
      safe_reference = jnp.where(
          restricted[:, None],
          jax.lax.stop_gradient(reference_support_logps),
          -jnp.log(2.0),
      )
      exact = jnp.sum(
          jnp.exp(safe_support) * (safe_support - safe_reference),
          axis=-1,
      )
      kl = kl.at[:, 0].set(jnp.where(restricted, exact, kl[:, 0]))
    if reference_kl_weights is not None:
      kl *= jnp.where(active, jax.lax.stop_gradient(reference_kl_weights), 0.0)
  return jnp.where(active, policy + beta * kl, 0.0), kl


@function_registry.register_policy_loss_fn("r2orch_director")
def director_loss_fn(
    model, train_example, algo_config, pad_id, eos_id, **kwargs
):
  if (
      algo_config.temperature != 1.0
      or algo_config.loss_agg_mode != "token-mean"
  ):
    raise ValueError(
        "The source Director objective uses temperature=1 and token-mean."
    )
  if algo_config.sampler_is is not None or algo_config.sampler_rs is not None:
    raise ValueError(
        "Precompute fixed Director importance weights before updates."
    )
  if train_example.old_per_token_logps is None:
    raise ValueError(
        "Director PPO requires fixed pre-update actor log probabilities."
    )
  current, entropy, support = policy_logprobs(
      model,
      train_example,
      pad_id=pad_id,
      eos_id=eos_id,
      chunk_size=kwargs.get("compute_logps_chunk_size", 0),
  )
  advantages = train_example.advantages
  if advantages.ndim == 1:
    advantages = advantages[:, None]
  mask = train_example.completion_mask
  losses, kl = token_loss(
      current,
      train_example.old_per_token_logps,
      train_example.ref_per_token_logps,
      advantages,
      mask,
      beta=algo_config.beta,
      epsilon=algo_config.epsilon,
      epsilon_high=algo_config.epsilon_high,
      correction=train_example.sampler_is_weights,
      support_ids=train_example.action_support_ids,
      support_logps=support,
      reference_support_logps=train_example.reference_support_logps,
      kl_term_cap=getattr(algo_config, "kl_term_cap", 0.0),
      reference_kl_weights=train_example.reference_kl_weights,
  )
  denominator = jnp.sum(mask)

  def metric(values):
    return sft_utils.WeightedMetric(
        jnp.sum(jnp.where(mask > 0, values, 0.0)), denominator, min_denom=1.0
    )

  return sft_utils.LossOutput(
      primary_loss=metric(losses),
      aux_metrics={"entropy": metric(entropy), "kl": metric(kl)},
  )
