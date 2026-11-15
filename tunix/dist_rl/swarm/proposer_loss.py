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

"""R2Orch Proposer's exact finite-label pair objective for native trainers.

Unlike autoregressive call payloads, each row's completion_ids lists candidate
label tokens, all predicted independently at the final prompt position. Pass
payloads directly to this loss; do not use the GRPO model-input builder to
interpret candidate labels as an autoregressive response. The source objective
is proposer_pair_training.reinforce_pair_loss at R2Orch 12ceb7563aa50439.
"""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.rl import function_registry
from tunix.sft import utils as sft_utils


def validate_payload(payload):
  """Admit a finite-label batch on the host, before native trainer execution.

  All candidate labels take part in the categorical normalizer and KL. These
  rows have no padded candidates: a zero advantage is not a padding indicator.
  The caller supplies reference probabilities normalized over exactly the same
  ordered label set.
  """
  labels = np.asarray(payload.completion_ids)
  prompts = np.asarray(payload.prompt_ids)
  if (
      labels.ndim != 2
      or labels.shape[0] < 1
      or labels.shape[1] < 2
      or prompts.ndim != 2
      or prompts.shape[0] != labels.shape[0]
      or prompts.shape[1] < 1
  ):
    raise ValueError(
        "Expected nonempty prompt rows and at least two candidate labels."
    )
  for values in (prompts, labels):
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
      raise ValueError("Prompt and label IDs must be nonnegative integers.")
  if any(len(set(row.tolist())) != len(row) for row in labels):
    raise ValueError("Candidate label IDs must be distinct in each row.")
  prompt_mask = np.asarray(payload.prompt_mask)
  if (
      prompt_mask.shape != prompts.shape
      or not np.isin(prompt_mask, (0, 1)).all()
      or np.any(prompt_mask.sum(axis=-1) < 1)
      or np.any(prompt_mask[:, -1] != 1)
  ):
    raise ValueError(
        "Prompts require binary masks with an active final position."
    )
  if payload.completion_mask is None:
    raise ValueError("All candidate labels must have an active mask.")
  for mask in (payload.completion_mask, payload.completion_attention_mask):
    if mask is not None and (
        np.asarray(mask).shape != labels.shape
        or not np.all(np.asarray(mask) == 1)
    ):
      raise ValueError("All candidate labels must have an active mask.")
  advantages = np.asarray(payload.advantages)
  if advantages.shape != labels.shape or not np.isfinite(advantages).all():
    raise ValueError("Pair advantages must be finite and align with labels.")
  for row in advantages:
    selected = row[row != 0]
    if len(selected) not in (0, 2) or (
        len(selected) == 2
        and (
            selected[0] * selected[1] >= 0
            or not np.isclose(selected.sum(), 0, atol=1e-7)
        )
    ):
      raise ValueError(
          "Each row needs one opposite-sign pair or all-zero advantages."
      )
  reference = np.asarray(payload.ref_per_token_logps)
  if (
      reference.shape != labels.shape
      or not np.isfinite(reference).all()
      or not np.allclose(np.exp(reference).sum(axis=-1), 1, atol=1e-5)
  ):
    raise ValueError(
        "Reference log probabilities must be finite and normalized over labels."
    )
  return payload


def candidate_logps(model, payload, *, pad_id=0, eos_id=0, chunk_size=0):
  """Normalize model probabilities over the provided, distinct label tokens.

  Repeated independent one-token completions preserve next-token alignment
  without relying on a model-specific logits API. The reference scorer uses
  exactly this function with the immutable initial Proposer model.
  A positive chunk_size skips the prompt's full-vocabulary LM head and projects
  only the final hidden position; callers must propagate their memory profile.
  """
  batch, count = payload.completion_ids.shape
  prompts = jnp.repeat(payload.prompt_ids, count, axis=0)
  mask = jnp.repeat(payload.prompt_mask, count, axis=0)
  labels = jnp.asarray(payload.completion_ids).reshape(batch * count, 1)
  graph, state = nnx.split(model)
  values = common.compute_per_token_logps(
      graph,
      state,
      prompts,
      labels,
      pad_id,
      eos_id,
      stop_gradient=False,
      temperature=1.0,
      chunk_size=chunk_size,
      token_mask=jnp.concatenate([mask, jnp.ones_like(labels)], axis=1),
  ).reshape(batch, count)
  return jax.nn.log_softmax(values.astype(jnp.float32), axis=-1)


def pair_objective(
    current, reference, signed_advantages, *, beta=1.0, gap_clip=4.0
):
  """One +a/-a pair per row; exact KL(current || initial) over all labels."""
  reference = jax.lax.stop_gradient(reference)
  advantages = jax.lax.stop_gradient(signed_advantages)
  # Source clips the reference-relative log-odds gap in the advantageous
  # direction. The signed sum divided by |a| gives sign(a) * gap.
  magnitude = jnp.max(jnp.abs(advantages), axis=-1)
  directed_gap = jnp.sum(advantages * (current - reference), axis=-1)
  clipped = (
      (gap_clip > 0) & (magnitude > 0) & (directed_gap >= gap_clip * magnitude)
  )
  policy = -jnp.sum(advantages * current, axis=-1)
  policy = jnp.where(jax.lax.stop_gradient(clipped), 0.0, policy)
  kl = jnp.sum(jnp.exp(current) * (current - reference), axis=-1)
  return policy + beta * kl, kl


@function_registry.register_policy_loss_fn("r2orch_proposer_pair")
def proposer_pair_loss(
    model, train_example, algo_config, pad_id, eos_id, **kwargs
):
  """Registered native loss; every row is one list, not one selected token.

  Host-side admission must ensure distinct labels, exactly one opposite-sign
  pair (or all zero), finite normalized reference logps and no padded rows.
  """
  if train_example.ref_per_token_logps is None:
    raise ValueError("The Proposer objective requires a frozen reference.")
  current = candidate_logps(
      model,
      train_example,
      pad_id=pad_id,
      eos_id=eos_id,
      chunk_size=kwargs.get(
          "compute_logps_chunk_size",
          getattr(algo_config, "logps_chunk_size", 0),
      ),
  )
  losses, kl = pair_objective(
      current,
      train_example.ref_per_token_logps,
      train_example.advantages,
      beta=algo_config.beta,
      gap_clip=getattr(algo_config, "gap_clip", 4.0),
  )
  count = jnp.asarray(losses.size, jnp.float32)
  return sft_utils.LossOutput(
      primary_loss=sft_utils.WeightedMetric(
          jnp.sum(losses), count, min_denom=1.0
      ),
      aux_metrics={
          "proposer/kl": sft_utils.WeightedMetric(
              jnp.sum(kl), count, min_denom=1.0
          )
      },
  )
