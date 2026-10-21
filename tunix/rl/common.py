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
"""Common RL helper classes and functions."""

import functools
import math
import os
import re
import inspect
from typing import Any, Iterable, Sequence

from absl import logging
import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.custom_derivatives import SymbolicZero  # pylint: disable=g-importing-member
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import jax.tree_util as jtu
import numpy as np
from tunix.sft import utils

make_causal_attn_mask = utils.make_causal_attn_mask
build_positions_from_mask = utils.build_positions_from_mask


class RepeatIterable(Iterable[Any]):
  """An iterable that processes a list of rollout batches.

  For each rollout batch, it shuffles its contents, slices it into mini-batches,
  and yields them sequentially before moving to the next rollout batch. This
  entire process is repeated for a specified number of epochs.
  """

  def __init__(
      self,
      data: list[Any],
      repeat: int,
      mini_batch_size: int | None = None,
      shuffle: bool = False,
      key: jnp.ndarray | None = None,
  ):
    self._data = data

    self.repeat = repeat
    self.mini_batch_size = mini_batch_size

    self.shuffle = shuffle
    self.key = key if key is not None else jax.random.PRNGKey(0)

    # Maintain a private, mutable `mini_batch_size`, for simpler code.
    self._mini_batch_size = mini_batch_size

  def _shuffle_and_slice_one_batch(self, rollout_batch: Any):
    """A generator that shuffles and slices a single rollout batch."""
    leaves, _ = jtu.tree_flatten(rollout_batch)
    rollout_batch_size = leaves[0].shape[0]

    if self.mini_batch_size is None:
      self._mini_batch_size = rollout_batch_size

    if rollout_batch_size % self._mini_batch_size != 0:
      raise ValueError(
          "Each rollout batch's size must be divisible by `mini_batch_size`."
      )
    num_mini_batches = rollout_batch_size // self._mini_batch_size

    # Shuffle indices.
    if self.shuffle:
      self.key, _ = jax.random.split(self.key)
      shuffled_indices = jax.random.permutation(self.key, rollout_batch_size)
    else:
      shuffled_indices = jnp.arange(rollout_batch_size)

    # Slice the rollout batch into mini-batches.
    for i in range(num_mini_batches):
      start = i * self._mini_batch_size  # pyrefly: ignore[unsupported-operation]
      end = start + self._mini_batch_size  # pyrefly: ignore[unsupported-operation]
      batch_indices = shuffled_indices[start:end]

      mini_batch = jtu.tree_map(
          lambda leaf, indices=batch_indices: leaf[indices], rollout_batch
      )
      yield mini_batch

  def __iter__(self):
    """The main generator for the iterable."""
    for _ in range(self.repeat):
      for rollout_batch in self._data:
        yield from self._shuffle_and_slice_one_batch(rollout_batch)


ArrayType = jax.Array | np.ndarray


@flax.struct.dataclass(frozen=True)
class TrainExample:
  prompt_ids: ArrayType
  prompt_mask: ArrayType
  completion_ids: ArrayType
  completion_mask: ArrayType
  advantages: ArrayType
  ref_per_token_logps: ArrayType | None
  old_per_token_logps: ArrayType | None
  segment_ids: ArrayType | None = None
  segment_positions: ArrayType | None = None
  # Static (JIT-compile-time) upper bound on segments per packed row, including
  # the padding bucket (segment 0). Set by `pack_sequences` to
  # ``max_token_budget + 1`` -- a pack of ``budget`` tokens holds at most
  # ``budget`` segments (each >= 1 token), so this never overflows. Kept
  # ``pytree_node=False`` so it is a static Python int (a fixed value every
  # step -> the segment-aware loss compiles once, no per-step recompilation).
  num_segments: int | None = flax.struct.field(default=None, pytree_node=False)
  is_update_step: ArrayType | None = None
  # Truncated importance-sampling correction weights for off-policy
  # correction between the rollout sampler and the trainer. Per-token,
  # detached, multiplied into the policy-gradient loss BEFORE aggregation
  # to dampen positions where the trainer's recomputed log-probability
  # diverges from the rollout sampler's. ``None`` disables the correction.
  sampler_is_weights: ArrayType | None = None
  # Per-token log-probabilities the rollout engine reported for the tokens it
  # generated -- the *behaviour* policy. Distinct from ``old_per_token_logps``,
  # which is whatever policy the surrogate ratio is taken against: the two
  # coincide in the common case but diverge when the ratio is taken against a
  # trainer recompute or pinned to 1.0. Kept separate so a correction can
  # always reference the policy that actually produced the samples. ``None``
  # when the rollout engine does not return log-probabilities.
  rollout_per_token_logps: ArrayType | None = None
  # `[B]`, 1.0 where the rollout engine exhausted the response budget without
  # emitting an end-of-sequence token, i.e. the sample is a truncated prefix of
  # a trajectory rather than a trajectory. ``None`` when the engine reports no
  # truncation verdict.
  overlong: ArrayType | None = None
  # `[B, P + C, num_layers, top_k]` MoE experts the rollout routed through,
  # laid out over the same `[prompt | completion]` padding as the token ids.
  # When set, a model that accepts `forced_routed_experts` replays these
  # instead of re-running its router. `-1` marks a slot to leave to the router.
  routed_experts: ArrayType | None = None

  def to_jax_array(self) -> "TrainExample":
    """Returns a copy of the batch with all array fields moved to JAX array."""
    return jax.tree.map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
        self,
    )


def compute_kl_divergence(
    per_token_logps: jax.Array,
    ref_per_token_logps: jax.Array,
    method: str = "low_var_kl",
    clamp_value: float | None = None,
) -> jax.Array:
  """Compute per token KL divergence between trained and reference policy.

  Based on `method`, we compute one of three kinds of KL divergence:
  - "kl": Unbiased, high-variance estimator. Simple Forward KL:
    `logp - ref_logp`.
  - "mse_kl": Biased, low-variance estimator. Squared log-difference:
    `0.5 * (logp - ref_logp)^2`.
  - "low_var_kl": Unbiased, low-variance estimator. J. Schulman low-variance
    approx: `(r - 1) - log r`, where `r = q/p = exp(ref_logp - logp)`.

  Args:
    per_token_logps: Per token log probabilities from the trained policy.
    ref_per_token_logps: Per token log probabilities from the reference policy.
    method: KL penalty method. Defaults to "low_var_kl".
    clamp_value: Optional symmetric clamp applied to the returned KL, i.e.
      `clip(kl, -clamp_value, +clamp_value)`. `None` (default) disables the
      clamp and preserves prior behavior. Set to a positive float (e.g.
      `10000.0`) to bound rare outliers — useful when the trained policy briefly
      drifts far from the reference and the `low_var_kl` estimator's `exp(diff)`
      term can overflow fp32 / saturate bf16.

  Returns:
    KL divergence.
  """
  per_token_logps = per_token_logps.astype(jnp.float32)
  if ref_per_token_logps is not None:
    ref_per_token_logps = ref_per_token_logps.astype(jnp.float32)
    valid = jnp.isfinite(per_token_logps) & jnp.isfinite(ref_per_token_logps)
    per_token_logps = jnp.where(valid, per_token_logps, 0.0)
    ref_per_token_logps = jnp.where(valid, ref_per_token_logps, 0.0)

  if method == "kl":
    kl = per_token_logps - ref_per_token_logps
  elif method == "mse_kl":
    kl = 0.5 * jnp.square(per_token_logps - ref_per_token_logps)
  elif method == "low_var_kl":
    diff = jnp.clip(ref_per_token_logps - per_token_logps, -20.0, 20.0)
    kl = jnp.exp(diff) - diff - 1
  else:
    raise ValueError(
        "`method` must be one of 'kl', 'mse_kl', 'low_var_kl'. Received:"
        f" {method}"
    )

  if clamp_value is not None:
    kl = jnp.clip(kl, -clamp_value, clamp_value)
  return kl


def selective_log_softmax(logits: jax.Array, input_ids: jax.Array) -> jax.Array:
  """Compute the log probablity based on the input ids.

  Args:
    logits: Logits from the model.
    input_ids: Input ids to get logits.

  Returns:
    Selected log probabilities.
  """
  logits_f32 = jnp.where(jnp.isfinite(logits), logits.astype(jnp.float32), 0.0)
  target_logits = jnp.take_along_axis(
      logits_f32, input_ids[..., None], axis=-1
  ).squeeze(-1)
  normalizer = jax.nn.logsumexp(logits_f32, axis=-1)
  return target_logits - normalizer


# TODO(tsbao): remove this once old callsite is cleaned up.
@nnx.jit(static_argnames=("logits_to_keep",))
def get_per_token_logps(
    model: nnx.Module,
    input_tokens: jax.Array,
    positions: jax.Array,
    attn_mask: jax.Array,
    logits_to_keep: int,
    images: jax.Array | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Computes the per-token log probabilities."""
  kwargs = {} if images is None else {"images": images}
  logits, _ = model(
      input_tokens,
      positions=positions,
      attention_mask=attn_mask,
      cache=None,
      **kwargs,
  )
  logits = logits[:, -logits_to_keep - 1 : -1, :]
  input_tokens = input_tokens[:, -logits_to_keep:]
  per_token_logps = selective_log_softmax(logits, input_tokens)
  return per_token_logps


# TODO(abheesht): This is computed 4 times - twice in `compute_per_token_logps`
# and twice in `compute_score`. We can factor this out and compute it just once.
@functools.partial(jax.jit, static_argnames=("pad_id", "eos_id"))
def process_ids(
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    segment_ids: jax.Array | None = None,
    segment_positions: jax.Array | None = None,
    token_mask: jax.Array | None = None,
):
  """Processes prompt and completion ids.

  Args:
    prompt_tokens: jax.Array token IDs for prompt. If sequence packing is
      enabled, prompt_tokens will be empty (shape [B, 0]), because prompts and
      completions are already concatenated into completion_tokens.
    completion_tokens: jax.Array token IDs for completion. If sequence packing
      is enabled, completion_tokens functions as a unified 1D buffer holding
      pre-concatenated mixed prompt and completion boundaries padded
      sequentially.
    pad_id: pad token identifier.
    eos_id: end of sequence identifier.
    completion_mask: optional attention weights mapping completion sequences.
    segment_ids: optional 1D sequential document identifiers used for packing.
    segment_positions: optional 1D local position indices used for packing.
    token_mask: Explicit `[B, prompt_len + completion_len]` valid positions;
      distinct from a sampled-token loss mask. Cannot accompany packed segments.
  """
  prompt_completion_ids = jnp.concat([prompt_tokens, completion_tokens], axis=1)

  if segment_ids is not None:
    if token_mask is not None:
      raise ValueError("Use packed segment metadata or token_mask, not both")
    # Positions are either provided or must be computed correctly (assumed
    # provided for packed).
    if segment_positions is None:
      raise ValueError(
          "segment_positions must be explicitly provided for packed sequences. "
      )
    attn_mask = None  # Relies on segment_ids inside the model
    # Packed callers supply their own segment_ids that already separate
    # distinct documents in the buffer; no need for an additional padding
    # mask here.
    return prompt_completion_ids, segment_positions, attn_mask, None

  if token_mask is None:
    prompt_mask = prompt_tokens != pad_id
    completion_mask = completion_tokens != pad_id
    prompt_completion_mask = jnp.concatenate(
        [prompt_mask, completion_mask], axis=-1
    )
  else:
    if token_mask.shape != prompt_completion_ids.shape:
      raise ValueError("token_mask must match the full prompt/completion shape")
    prompt_completion_mask = token_mask.astype(jnp.bool_)
  positions = build_positions_from_mask(prompt_completion_mask)
  attn_mask = make_causal_attn_mask(prompt_completion_mask)

  # 1-D per-position non-pad mask for the full prompt+completion sequence.
  # Used as ``segment_ids`` by attention kernels that cannot consume the 2-D
  # ``attn_mask`` directly (e.g. pallas splash attention takes only a causal
  # mask kernel-side and respects per-position segment ids to suppress
  # cross-segment attention). With pad=0 and real=1, a real position never
  # attends to a pad position regardless of where padding lives in the
  # sequence (typically left-padded for prompt-side alignment).
  input_seg_ids = prompt_completion_mask.astype(jnp.int32)

  return prompt_completion_ids, positions, attn_mask, input_seg_ids


# Marks a router-replay slot the trainer must leave to the model's own router.
# `-1` and not `0`, because expert 0 is a real expert.
UNSET_ROUTED_EXPERT = -1


def align_routed_experts(
    routed_experts: Sequence[np.ndarray | None] | None,
    completion_lengths: Sequence[int],
    prompt_width: int,
    completion_width: int,
) -> np.ndarray | None:
  """Lays rollout routing out over the padded `[prompt | completion]` stream.

  The rollout reports `[prompt_len + completion_len, num_layers, top_k]` per
  generation, while the trainer sees prompts left-padded to `prompt_width` and
  completions right-padded to `completion_width`. Routing has to be padded the
  same way, or every replayed expert detaches from the token it was captured
  for -- a silent wrong-answer bug rather than a crash.

  Args:
    routed_experts: One array per generation, or None. Replay is all-or-nothing:
      if any row is missing, the whole batch falls back to the model's own
      router rather than mixing replayed and fresh rows.
    completion_lengths: Unpadded completion length per generation. The prompt /
      completion split cannot be inferred from the widths alone, since both
      sides are padded.
    prompt_width: Padded prompt length.
    completion_width: Padded completion length.

  Returns:
    `[B, prompt_width + completion_width, num_layers, top_k]`, or None.
  """
  if not routed_experts or any(r is None for r in routed_experts):
    return None
  if len(routed_experts) != len(completion_lengths):
    raise ValueError(
        f"got {len(routed_experts)} routing rows but"
        f" {len(completion_lengths)} completion lengths"
    )

  rows = []
  for routed, completion_len in zip(routed_experts, completion_lengths):
    routed = np.asarray(routed, dtype=np.int16)
    if routed.ndim != 3:
      raise ValueError(
          "routed_experts must be [length, num_layers, top_k]; got shape"
          f" {routed.shape}"
      )
    # Prompts are left-padded, so an over-long one keeps its tail; completions
    # are right-padded, so an over-long one keeps its head.
    split = max(routed.shape[0] - completion_len, 0)
    kept_prompt_start = max(split - prompt_width, 0)
    kept_completion_end = split + min(routed.shape[0] - split, completion_width)
    prompt_part = routed[kept_prompt_start:split]
    completion_part = routed[split:kept_completion_end]
    row = np.full(
        (prompt_width + completion_width,) + routed.shape[1:],
        UNSET_ROUTED_EXPERT,
        dtype=np.int16,
    )
    row[prompt_width - prompt_part.shape[0] : prompt_width] = prompt_part
    row[prompt_width : prompt_width + completion_part.shape[0]] = (
        completion_part
    )
    rows.append(row)
  return np.stack(rows)


@functools.cache
def _call_contains_by_type(target_cls: type[Any], target_arg: str) -> bool:
  """Determines if a class' call function contains a target argument and caches the result."""

  try:
    sig = inspect.signature(target_cls.__call__)
    return (target_arg in sig.parameters) or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
  except Exception:
    return False


def model_call_contains(model, target_arg: str) -> bool:
  """Determines if a model's call function contains a target argument"""

  target_obj = model.transformer if hasattr(model, "transformer") else model

  return _call_contains_by_type(type(target_obj), target_arg)


def pop_moe_overflow(model: nnx.Module) -> jax.Array:
  """Pops the `moe_has_overflow` intermediates MoE layers sowed on `model`.

  Args:
    model: The model after a forward pass.

  Returns:
    Scalar bool, True if any sowed flag (scanned layers stack them) is True.
  """
  flags = nnx.pop(
      model, nnx.All(nnx.Intermediate, nnx.PathContains("moe_has_overflow"))
  ).to_pure_dict()
  leaves = [jnp.any(f) for f in jax.tree.leaves(flags)]
  return jnp.any(jnp.stack(leaves)) if leaves else jnp.array(False)


def wants_moe_audit(model: Any) -> bool:
  """True for a MaxText model configured with log_moe_drops=true (MoE drop audit).

  The trainer hands the loss a `TunixMaxTextAdapter`, which has no `config` of
  its own: the MaxText model, and its config, is `.base`. Both are checked.
  """
  for m in (model, getattr(model, "base", None)):
    if bool(getattr(getattr(m, "config", None), "log_moe_drops", False)):
      return True
  return False


def _natural_key(path_str: str) -> list[Any]:
  return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path_str)]


def pop_moe_audit(model: nnx.Module) -> dict[str, jax.Array] | None:
  """Pops the MoE drop audit MaxText's RoutedMoE sowed on `model` (log_moe_drops).

  Returns `{"dropped": int32[L], "max_ratio": float32[L]}`, one entry per MoE layer call in natural path
  order (a scanned stack contributes its stacked values in order), or None if nothing was sown. The last entry
  of each sow tuple is this forward's. MaxText's training engine reads it from the loss aux under
  `_moe_audit` and logs it as MOE_DROP_AUDIT.
  """
  found = {"moe_dropped_assignments": [], "moe_max_load_ratio": []}
  for path, var in nnx.state(model, nnx.Intermediate).flat_state():
    name = str(path[-1]) if path else ""
    if name in found:
      sown = var.get_value() if hasattr(var, "get_value") else var
      latest = jnp.asarray(sown[-1] if isinstance(sown, tuple) else sown)
      found[name].append(("/".join(str(p) for p in path), latest))
  nnx.pop(
      model,
      nnx.All(
          nnx.Intermediate,
          nnx.Any(
              nnx.PathContains("moe_dropped_assignments"),
              nnx.PathContains("moe_max_load_ratio"),
          ),
      ),
  )
  if not found["moe_dropped_assignments"]:
    return None
  out = {}
  for name, key, dtype in (
      ("moe_dropped_assignments", "dropped", jnp.int32),
      ("moe_max_load_ratio", "max_ratio", jnp.float32),
  ):
    entries = sorted(found[name], key=lambda e: _natural_key(e[0]))
    out[key] = jnp.concatenate(
        [jnp.ravel(v).astype(dtype) for _, v in entries]
    )
  return out


@functools.partial(
    jax.jit,
    static_argnames=(
        "pad_id",
        "eos_id",
        "stop_gradient",
        "return_entropy",
        "temperature",
        "chunk_size",
        "return_moe_overflow",
        "return_moe_audit",
    ),
)
def compute_per_token_logps(
    graphdef,
    state,
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    images: jax.Array | None = None,
    stop_gradient: bool = True,
    return_entropy: bool = False,
    segment_ids: jax.Array | None = None,
    segment_positions: jax.Array | None = None,
    temperature: float = 1.0,
    chunk_size: int = 0,
    routed_experts: jax.Array | None = None,
    token_mask: jax.Array | None = None,
    return_moe_overflow: bool = False,
    return_moe_audit: bool = False,
) -> jax.Array | tuple[jax.Array, ...]:
  """Computes the per-token log probabilities.

  Args:
    graphdef: Flax NNX GraphDef.
    state: Flax NNX State.
    prompt_tokens: jax.Array token IDs for prompt. If sequence packing is
      enabled, prompt_tokens will be empty (shape [B, 0]), because prompts and
      completions are already concatenated into completion_tokens.
    completion_tokens: jax.Array token IDs for completion. If sequence packing
      is enabled, completion_tokens functions as a unified 1D buffer holding
      pre-concatenated mixed prompt and completion boundaries padded
      sequentially.
    pad_id: pad token identifier.
    eos_id: end of sequence identifier.
    images: optional images array.
    stop_gradient: whether to stop gradient.
    return_entropy: whether to return per token entropy.
    segment_ids: optional 1D sequential document identifiers used for packing.
    segment_positions: optional 1D local position indices used for packing.
    temperature: temperature used for rollout.
    chunk_size: If not 0, computes the log probabilities in sequence chunks of
      this size. With TUNIX_LOGPS_SHARDED_HEAD on and packed rows
      (`segment_ids` set), the sharded head, when it engages (see
      `maybe_compute_sharded_chunked_logps`), instead has each device scan its
      own block of the sequence in chunks of `min(chunk_size, block length)`
      positions.
    routed_experts: Optional `[B, T, num_layers, top_k]` MoE experts captured
      during rollout. Forwarded to the model as `forced_routed_experts` -- the
      name MaxText's adapter uses -- so it replays this routing instead of
      re-running its router. Ignored by models that do not accept the kwarg.
    token_mask: Optional explicit valid positions for prompt plus completion.
    return_moe_overflow: whether to also return a scalar bool that is True if
      any MoE layer sowed a True `moe_has_overflow` intermediate (MaxText's
      RoutedMoE dropped tokens) during the forward pass. The model is merged
      inside this jit, so these intermediates are otherwise lost.

  Returns:
    per_token_logps: jax.Array token-level logarithmic values.
      Without sequence packing, returns log probs for completion tokens only,
      with shape `[B, completion_len]`. With sequence packing, returns log
      probs for the full packed sequence padded out (since prompts and
      completions of multiple sequences are concatenated), with shape `[B,
      FullSeqLen]`.
    entropy: optional per-token entropy jax.Array of shape matches
    per_token_logps,
      returned if return_entropy is True.
    moe_overflow: optional scalar bool, returned last if return_moe_overflow
      is True; False for models that sow no `moe_has_overflow`.
  """
  model = nnx.merge(graphdef, state)

  input_tokens, calculated_positions, attn_mask, input_seg_ids = process_ids(
      prompt_tokens,
      completion_tokens,
      pad_id,
      eos_id,
      segment_ids,
      segment_positions,
      token_mask,
  )

  model_kwargs = {
      "positions": calculated_positions,
      "cache": None,
      "attention_mask": attn_mask,
  }
  if chunk_size > 0:
    sig = inspect.signature(model.__call__)
    if "skip_lm_head" not in sig.parameters:
      raise ValueError("Model __call__ must accept skip_lm_head.")
    if not hasattr(model, "compute_final_logits") or not callable(
        getattr(model, "compute_final_logits")
    ):
      raise ValueError(
          "Model must contain a function called compute_final_logits."
      )
    model_kwargs["skip_lm_head"] = True

  # Pass through any segment ids so the model's attention kernel can respect
  # them only if the model signature accepts it: caller-provided packing ids take
  # precedence; otherwise we pass the per-position non-pad mask derived in
  # ``process_ids`` so flash-attention variants that lack a separate
  # padding-mask input still skip pad positions.
  if model_call_contains(model, "segment_ids"):
    seg_to_pass = segment_ids if segment_ids is not None else input_seg_ids
    if seg_to_pass is not None:
      # Guard all-zero dummy padding rows (from batch-level padding) so
      # flash-attention and GDN kernels never encounter a completely empty row.
      row_all_zero = jnp.all(seg_to_pass == 0, axis=-1)
      seg_to_pass = seg_to_pass.at[:, 0].set(
          jnp.where(row_all_zero, jnp.ones_like(seg_to_pass[:, 0]), seg_to_pass[:, 0])
      )
      model_kwargs["segment_ids"] = seg_to_pass
  if images is not None:
    model_kwargs["images"] = images
  # Router replay. `forced_routed_experts` is the model-side name (MaxText's
  # adapter); only models that advertise it can consume it, so everything else
  # is unaffected.
  if routed_experts is not None and model_call_contains(
      model, "forced_routed_experts"
  ):
    model_kwargs["forced_routed_experts"] = jnp.asarray(
        routed_experts, dtype=jnp.int32
    )

  outputs, _ = model(input_tokens, **model_kwargs)
  moe_overflow = pop_moe_overflow(model) if return_moe_overflow else None
  # log_moe_drops: the drop audit dict (or None), appended after the overflow flag.
  moe_audit = pop_moe_audit(model) if return_moe_audit else None

  def _result(*outs):
    if return_moe_overflow:
      outs += (moe_overflow,)
    if return_moe_audit:
      outs += (moe_audit,)
    return outs if len(outs) > 1 else outs[0]

  if segment_ids is not None:
    # Packed Mode: Evaluate the full sequence (mixed prompts + completions).
    # Since predicting token[i] requires logit[i-1], we skip the first token.
    # This shrinks the output shape to [Batch, FullSeqLen - 1]
    logits_to_keep = input_tokens.shape[1] - 1
  else:
    logits_to_keep = completion_tokens.shape[1]

  input_tokens_to_keep = input_tokens[:, -logits_to_keep:]

  if chunk_size > 0:
    out = None
    if segment_ids is not None:
      # Packed rows score every position, so the head can run on the
      # unsliced, still evenly sharded hidden states. Opt-in via
      # TUNIX_LOGPS_SHARDED_HEAD; None when off or when it falls back. It
      # returns [B, T], already aligned like the padded output below.
      out = maybe_compute_sharded_chunked_logps(
          model,
          outputs,
          input_tokens,
          temperature,
          chunk_size,
          return_entropy,
      )
    aligned = out is not None
    if out is None:
      hidden_state = outputs[:, -logits_to_keep - 1 : -1, :]
      out = compute_chunked_logps(
          model,
          hidden_state,
          input_tokens_to_keep,
          temperature,
          chunk_size,
          return_entropy,
      )
    if return_entropy:
      per_token_logps, per_token_entropy = out
    else:
      per_token_logps = out

    if segment_ids is not None and not aligned:
      per_token_logps = jnp.pad(
          per_token_logps, ((0, 0), (1, 0)), constant_values=0.0
      )
      if return_entropy:
        per_token_entropy = jnp.pad(
            per_token_entropy, ((0, 0), (1, 0)), constant_values=0.0  # pyrefly: ignore[unbound-name]
        )

    if segment_ids is not None:
      # Zero out transitions from padding (segment_ids == 0) into a real segment
      # (segment_ids > 0), such as at `t = shift` after leading padding under
      # `pad_position="start"` or after an inter-segment alignment gap. Under
      # `pad_position="end"` the first segment starts at `t = 0`, which is
      # already zero-padded above; masking `0 -> >0` transitions ensures the
      # first token after padding has the exact same 0.0 log-prob and zero
      # backward cotangent into `hidden_states[:, t - 1]`.
      after_pad = (segment_ids[:, :-1] == 0) & (segment_ids[:, 1:] > 0)
      after_pad_mask = jnp.pad(
          after_pad, ((0, 0), (1, 0)), constant_values=False
      )
      per_token_logps = jnp.where(after_pad_mask, 0.0, per_token_logps)
      if return_entropy:
        per_token_entropy = jnp.where(
            after_pad_mask, 0.0, per_token_entropy  # pyrefly: ignore[unbound-name]
        )

    if stop_gradient:
      per_token_logps = jax.lax.stop_gradient(per_token_logps)
      if return_entropy:
        per_token_entropy = jax.lax.stop_gradient(per_token_entropy)  # pyrefly: ignore[unbound-name]

    if return_entropy:
      return _result(per_token_logps, per_token_entropy)  # pyrefly: ignore[unbound-name]
    return _result(per_token_logps)
  else:
    logits = outputs[:, -logits_to_keep - 1 : -1, :]
    if temperature != 0.0 and temperature != 1.0:
      logits /= temperature

    per_token_logps = selective_log_softmax(logits, input_tokens_to_keep)

    if segment_ids is not None:
      # Pad the front with 0.0 to make shape back to [Batch, FullSeqLen]. This
      # aligns indices (logp[i] matches token[i]) and avoids mask slicing downstream.
      per_token_logps = jnp.pad(
          per_token_logps, ((0, 0), (1, 0)), constant_values=0.0
      )
      logits = jnp.pad(logits, ((0, 0), (1, 0), (0, 0)), constant_values=0.0)
      after_pad = (segment_ids[:, :-1] == 0) & (segment_ids[:, 1:] > 0)
      after_pad_mask = jnp.pad(
          after_pad, ((0, 0), (1, 0)), constant_values=False
      )
      per_token_logps = jnp.where(after_pad_mask, 0.0, per_token_logps)

    if stop_gradient:
      per_token_logps = jax.lax.stop_gradient(per_token_logps)
      logits = jax.lax.stop_gradient(logits)

    if return_entropy:
      entropy = compute_entropy_from_logits(logits)
      return _result(per_token_logps, entropy)
    return _result(per_token_logps)


def sampler_trainer_agreement(
    rollout_per_token_logps: ArrayLike | None,
    trainer_per_token_logps: ArrayLike | None,
    completion_mask: ArrayLike | None,
    sampler_is: str | None = None,
    sampler_is_threshold: float = 2.0,
    seq_logprob_error_threshold: float | None = None,
    segment_ids: ArrayLike | None = None,
):
  """Sampler-vs-trainer agreement metrics, sequence error masking, and TIS weights.

  Shared by the RL orchestrator (``rl_program``), the agentic GRPO learner, and
  the standard GRPO learner. The unpacked and packed paths differ only in
  whether ``segment_ids`` is supplied.

  Args:
    rollout_per_token_logps: per-token logps recorded by the sampler at rollout
      time, or None (in which case no metrics/weights are produced).
    trainer_per_token_logps: per-token logps recomputed under the trainer's
      (actor) weights, or None.
    completion_mask: assistant-vs-env mask (1 for assistant-generated tokens, 0
      for env-injected tokens) scoping the comparison to model-emitted positions.
    sampler_is: if ``"token"``, also builds truncated per-token
      importance-sampling weights; otherwise no weights are returned.
    sampler_is_threshold: clamp applied to the importance-sampling weights.
    seq_logprob_error_threshold: if set, sequences (or packed segments) whose
      mean multiplicative probability error ``exp(|trainer_logp - rollout_logp|)``
      exceeds this threshold are masked out in ``filtered_completion_mask`` (and
      in ``sampler_is_weights``).
    segment_ids: optional ``[B, L]`` packing segment IDs (``1..K`` for packed
      trajectories, ``0`` for padding). When provided, sequence-level error
      gating operates per segment rather than per row.

  Returns:
    ``(metrics, sampler_is_weights, filtered_completion_mask)``, where
    ``filtered_completion_mask`` is ``completion_mask`` with any sequences
    exceeding ``seq_logprob_error_threshold`` masked to 0 (or the unchanged
    ``completion_mask`` when ``seq_logprob_error_threshold`` is None).
  """
  metrics = {}
  sampler_is_weights = None
  filtered_completion_mask = completion_mask
  if rollout_per_token_logps is None or trainer_per_token_logps is None:
    return metrics, sampler_is_weights, filtered_completion_mask

  if completion_mask is None:
    raise ValueError(
        "Completion mask is required for sampler-trainer agreement metrics."
    )

  orig_mask_dtype = jnp.asarray(completion_mask).dtype
  rollout_per_token_logps = np.asarray(rollout_per_token_logps)
  trainer_per_token_logps = np.asarray(trainer_per_token_logps)
  completion_mask = np.asarray(completion_mask)

  if (
      rollout_per_token_logps.shape != trainer_per_token_logps.shape
      or rollout_per_token_logps.shape != completion_mask.shape
  ):
    raise ValueError(
        "Shape mismatch: `rollout_per_token_logps`"
        f" ({rollout_per_token_logps.shape}), `trainer_per_token_logps`"
        f" ({trainer_per_token_logps.shape}), and `completion_mask`"
        f" ({completion_mask.shape}) must all have the same shape."
    )
  # ``completion_mask`` is the assistant-vs-env mask built upstream (1 for
  # assistant-generated tokens, 0 for env-injected tokens), and already
  # correctly scopes the comparison to model-emitted positions. We
  # deliberately do NOT additionally drop positions where the rollout logprob
  # equals exactly 0.0 -- that value can legitimately occur for near-certain
  # tokens and excluding them removes the most consistent positions from the
  # statistic, inflating the per-position mean.
  mask = jnp.asarray(completion_mask, dtype=jnp.bool_)
  mask_f = mask.astype(jnp.float32)
  mask_sum = jnp.maximum(mask_f.sum(), 1.0)
  diff = jnp.abs(rollout_per_token_logps - trainer_per_token_logps)
  diff_mean = float((diff * mask_f).sum() / mask_sum)
  diff_max = float(jnp.where(mask, diff, 0.0).max())
  # Multiplicative probability error: exp(min(|trainer_logp - rollout_logp|, 20))
  token_mult_err = jnp.exp(jnp.minimum(diff, 20.0))
  mult_err_mean = float((token_mult_err * mask_f).sum() / mask_sum)
  mult_err_max = float(jnp.where(mask, token_mult_err, 0.0).max())
  # Probability-space diff is more representative than logp_diff for
  # confidence agreement: logp can diverge arbitrarily for very
  # low-probability tokens whose contribution to the ratio is negligible.
  rp = jnp.exp(rollout_per_token_logps)
  tp = jnp.exp(trainer_per_token_logps)
  prob_diff = jnp.abs(rp - tp)
  prob_diff_mean = float((prob_diff * mask_f).sum() / mask_sum)
  prob_diff_max = float(jnp.where(mask, prob_diff, 0.0).max())
  rp_flat, tp_flat, mf = rp.reshape(-1), tp.reshape(-1), mask_f.reshape(-1)
  rp_mean = (rp_flat * mf).sum() / mask_sum
  tp_mean = (tp_flat * mf).sum() / mask_sum
  rp_d = (rp_flat - rp_mean) * mf
  tp_d = (tp_flat - tp_mean) * mf
  cov = (rp_d * tp_d).sum() / mask_sum
  rp_var = (rp_d * rp_d).sum() / mask_sum
  tp_var = (tp_d * tp_d).sum() / mask_sum
  pearson = float(cov / jnp.sqrt(jnp.maximum(rp_var * tp_var, 1e-12)))
  metrics.update({
      "sampler_trainer/logp_diff_mean": (diff_mean, np.mean),
      "sampler_trainer/logp_diff_max": (diff_max, np.max),
      "sampler_trainer/mult_prob_error_mean": (mult_err_mean, np.mean),
      "sampler_trainer/mult_prob_error_max": (mult_err_max, np.max),
      "sampler_trainer/prob_diff_mean": (prob_diff_mean, np.mean),
      "sampler_trainer/prob_diff_max": (prob_diff_max, np.max),
      "sampler_trainer/probs_pearson_corr": (pearson, np.mean),
  })
  logging.info(
      "sampler-trainer: logp_diff=(%.5f,%.5f) mult_err=(%.5f,%.5f)"
      " prob_diff=(%.5f,%.5f) pearson=%.5f",
      diff_mean,
      diff_max,
      mult_err_mean,
      mult_err_max,
      prob_diff_mean,
      prob_diff_max,
      pearson,
  )

  if seq_logprob_error_threshold is not None:
    if segment_ids is None:
      seq_tok_cnt = jnp.sum(mask_f, axis=-1, keepdims=True)
      seq_mult_err = jnp.sum(
          token_mult_err * mask_f, axis=-1, keepdims=True
      ) / jnp.maximum(seq_tok_cnt, 1.0)
      active_seq = seq_tok_cnt > 0
      keep_seq = (seq_mult_err <= seq_logprob_error_threshold) & active_seq
      num_active = jnp.maximum(jnp.sum(active_seq.astype(jnp.float32)), 1.0)
      num_masked = jnp.sum((~keep_seq & active_seq).astype(jnp.float32))
      keep_token_mask = keep_seq
    else:
      seg_ids_jnp = jnp.maximum(
          jnp.asarray(segment_ids, dtype=jnp.int32), 0
      )
      if seg_ids_jnp.shape != completion_mask.shape:
        raise ValueError(
            f"Shape mismatch: `segment_ids` ({seg_ids_jnp.shape}) must match "
            f"`completion_mask` ({completion_mask.shape})."
        )
      num_segments = int(jnp.max(seg_ids_jnp)) + 1
      seg_err_sum = segmented_sum(
          token_mult_err * mask_f,
          seg_ids_jnp,
          num_segments,
      )
      seg_tok_cnt = segmented_count(seg_ids_jnp, num_segments, mask=mask_f)
      seg_mult_err = seg_err_sum / jnp.maximum(seg_tok_cnt, 1.0)
      active_seg = (seg_tok_cnt > 0).at[:, 0].set(False)
      keep_seg = (seg_mult_err <= seq_logprob_error_threshold) & active_seg
      num_active = jnp.maximum(jnp.sum(active_seg.astype(jnp.float32)), 1.0)
      num_masked = jnp.sum((~keep_seg & active_seg).astype(jnp.float32))
      keep_token_mask = jnp.take_along_axis(keep_seg, seg_ids_jnp, axis=1)

    filtered_completion_mask = jnp.where(
        keep_token_mask, jnp.asarray(completion_mask), 0
    ).astype(orig_mask_dtype)
    completion_mask = filtered_completion_mask
    masked_frac = float(num_masked / num_active)
    masked_count = float(num_masked)
    metrics.update({
        "sampler_trainer/seq_error_masked_frac": (masked_frac, np.mean),
        "sampler_trainer/seq_error_masked_count": (masked_count, np.sum),
    })
    logging.info(
        "sampler_trainer/seq_error_masking: masked=%.0f/%.0f (%.4f) at"
        " threshold=%.2f",
        masked_count,
        float(num_active),
        masked_frac,
        seq_logprob_error_threshold,
    )

  # Truncated importance-sampling weights: per-token trainer-vs-sampler log
  # ratio, masked to assistant tokens, clamped at the threshold, detached.
  # The policy loss picks these up via ``train_example.sampler_is_weights``.
  if sampler_is == "token":
    asst_mask_f = jnp.asarray(completion_mask, dtype=jnp.float32)
    log_ratio = trainer_per_token_logps - rollout_per_token_logps
    log_ratio = jnp.clip(log_ratio, min=-20.0, max=20.0)
    sampler_is_weights = jax.lax.stop_gradient(
        jnp.minimum(jnp.exp(log_ratio), sampler_is_threshold) * asst_mask_f
    )
    is_mask_sum = jnp.maximum(asst_mask_f.sum(), 1.0)
    is_mean = float((sampler_is_weights * asst_mask_f).sum() / is_mask_sum)
    is_max = float(jnp.where(asst_mask_f > 0, sampler_is_weights, 0.0).max())
    frac_clipped = float(
        (
            (jnp.exp(log_ratio) > sampler_is_threshold)
            & (asst_mask_f > 0)
        )
        .astype(jnp.float32)
        .sum()
        / is_mask_sum
    )
    metrics.update({
        "sampler_is/weight_mean": (is_mean, np.mean),
        "sampler_is/weight_max": (is_max, np.max),
        "sampler_is/frac_clipped_at_threshold": (frac_clipped, np.mean),
    })
    logging.info(
        "sampler_is: weight_mean=%.4f weight_max=%.4f frac_clipped=%.4f"
        " (threshold=%.2f)",
        is_mean,
        is_max,
        frac_clipped,
        sampler_is_threshold,
    )
  return metrics, sampler_is_weights, filtered_completion_mask


def compute_chunked_logps(
    model,
    hidden_states,
    target_ids,
    temperature,
    chunk_size,
    return_entropy,
):
  """Computes per-token log probabilities in sequence chunks to save VRAM.

  Args:
      model: The actor model (needs to expose `lm_head`)
      hidden_states: [Batch, SeqLen, HiddenDim]
      target_ids:    [Batch, SeqLen]
      chunk_size:    Number of tokens to process at a time per sequence.

  Returns:
     per_token_logps: [Batch, SeqLen]
  """
  batch_size, seq_len, hidden_dim = hidden_states.shape
  hidden_states = jnp.where(jnp.isfinite(hidden_states), hidden_states, 0.0)

  # 1. Pad the sequence dimension if it's not perfectly divisible by chunk_size
  pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
  if pad_len > 0:
    # Pad sequence dimension (axis 1)
    hidden_states = jnp.pad(hidden_states, ((0, 0), (0, pad_len), (0, 0)))
    target_ids = jnp.pad(target_ids, ((0, 0), (0, pad_len)))

  padded_seq_len = seq_len + pad_len
  num_chunks = padded_seq_len // chunk_size

  # 2. Reshape into chunks: [Batch, NumChunks, ChunkSize, ...]
  hs_reshaped = hidden_states.reshape(
      batch_size, num_chunks, chunk_size, hidden_dim
  )
  ids_reshaped = target_ids.reshape(batch_size, num_chunks, chunk_size)

  # 3. Swap B, T axes to make it time-major for jax.lax.scan
  hs_scannable = jnp.swapaxes(hs_reshaped, 0, 1)
  ids_scannable = jnp.swapaxes(ids_reshaped, 0, 1)

  @nnx.remat
  def logp_step(carry, xs):
    hs_chunk, ids_chunk = xs

    # Project to vocabulary for just this chunk
    # Peak memory: [Batch, ChunkSize, VocabSize]
    logits_chunk = model.compute_final_logits(hs_chunk).astype(jnp.float32)

    if temperature != 0.0 and temperature != 1.0:
      logits_chunk /= temperature

    logps_chunk = selective_log_softmax(logits_chunk, ids_chunk)

    if return_entropy:
      entropy_chunk = compute_entropy_from_logits(logits_chunk)
      return None, (logps_chunk, entropy_chunk)
    else:
      return None, logps_chunk

  # 4. Scan over the NumChunks dimension
  _, scanned_out = jax.lax.scan(
      logp_step, init=None, xs=(hs_scannable, ids_scannable)
  )

  if return_entropy:
    logps_chunked, entropy_chunked = scanned_out
    # 5. Swap back to batch-major and flatten the sequence dimension
    logps_reshaped = jnp.swapaxes(logps_chunked, 0, 1)
    entropy_reshaped = jnp.swapaxes(entropy_chunked, 0, 1)
    # 6. Slice off any padding we added initially.
    per_token_logps = logps_reshaped.reshape(batch_size, padded_seq_len)[
        :, :seq_len
    ]
    per_token_entropy = entropy_reshaped.reshape(batch_size, padded_seq_len)[
        :, :seq_len
    ]
    return per_token_logps, per_token_entropy
  else:
    logps_chunked = scanned_out
    # 5. Swap back to batch-major and flatten the sequence dimension
    logps_reshaped = jnp.swapaxes(logps_chunked, 0, 1)
    # 6. Slice off any padding we added initially.
    per_token_logps = logps_reshaped.reshape(batch_size, padded_seq_len)[
        :, :seq_len
    ]
    return per_token_logps


# Sharded chunked log-prob head, opt-in via TUNIX_LOGPS_SHARDED_HEAD.
#
# Under GSPMD, `compute_chunked_logps` pays per chunk: an all-gather of every
# head weight (a [4096, 248320] vocab table is 2 GB in bf16) in the forward and
# again in the rematerialized backward, and an all-reduce of the full-size table
# gradient. And because `lax.scan` iterates over the sequence axis, a sequence
# sharded over a context-parallel axis is gathered, so every context rank
# computes every chunk. The sharded head runs the same per-chunk math inside
# `shard_map`: each head parameter is all-gathered once, in the forward, and the
# backward reuses that copy (so it stays live from the forward to the
# backward); each device scans only the tokens it holds; the table gradient is
# accumulated per device in float32 and reduce-scattered once; and chunks are
# read and written with dynamic slices rather than reshapes.
SHARDED_HEAD_ENV = "TUNIX_LOGPS_SHARDED_HEAD"
# Logical names of the hidden states' batch and length axes. When the active
# flax logical-axis rules (MaxText's) map them, the head uses that layout.
_LOGICAL_BATCH_AXIS = "activation_batch"
_LOGICAL_LENGTH_AXIS = "activation_length"


def sharded_head_mode() -> str | None:
  """Returns None (off), "local" or "fused", from TUNIX_LOGPS_SHARDED_HEAD.

  Case and surrounding whitespace are ignored. Off: unset, empty, 0, false, off,
  no. On: local, 1, true, on, yes. On, with each chunk scored by the fused
  log-softmax described above `_target_hit`: fused.

  Raises:
    ValueError: The variable has any other value.
  """
  value = os.environ.get(SHARDED_HEAD_ENV, "").strip().lower()
  if value in ("", "0", "false", "off", "no"):
    return None
  if value in ("1", "true", "on", "yes", "local"):
    return "local"
  if value == "fused":
    return "fused"
  raise ValueError(
      f"{SHARDED_HEAD_ENV}={value!r}: expected local, 1, true, on or yes (on),"
      " fused (on, with fused scoring), or empty, 0, false, off or no (off)."
  )


# Per-chunk reduce-scatter of the sharded head's parameter gradients, opt-in via
# TUNIX_LOGPS_HEAD_GRAD_SCATTER (it needs TUNIX_LOGPS_SHARDED_HEAD on).
#
# Off, the head's backward adds every chunk's parameter gradient into a
# full-size float32 accumulator per device and reduce-scatters it once, after
# the chunk loop. For a [4096, 248320] vocabulary table that accumulator is
# 3.8 GiB per device, and XLA may keep it live past the decoder's backward and
# lower the reduce-scatter as an all-reduce plus a slice. On, each chunk's
# gradient is reduce-scattered inside the chunk loop and each device keeps only
# its float32 shard (1 / devices of each parameter):
#   chunk: reduce-scatter the gradient in the dtype the head's vjp returns it
#     in (the gather dtype: bfloat16 for MaxText's vocabulary table), then add
#     the shard to the float32 accumulator. The cross-device sum then runs in
#     that dtype, so the gradient changes in its low bits.
#   chunk_f32: cast the gradient to float32 before the reduce-scatter: the same
#     float32 sums as off in another order, at twice the wire bytes of chunk
#     for a bfloat16 gradient.
#   chunk_unroll: chunk with the chunk loop unrolled, so that the scheduler can
#     overlap one chunk's reduce-scatter with the next chunk's matmuls.
# Like the head's switch, it is read when the caller is traced.
HEAD_GRAD_SCATTER_ENV = "TUNIX_LOGPS_HEAD_GRAD_SCATTER"
HEAD_GRAD_SCATTER_MODES = ("chunk", "chunk_f32", "chunk_unroll")


def head_grad_scatter_mode() -> str | None:
  """Returns None (off), "chunk", "chunk_f32" or "chunk_unroll".

  From TUNIX_LOGPS_HEAD_GRAD_SCATTER; case and surrounding whitespace are
  ignored. Off: unset, empty or off.

  Raises:
    ValueError: The variable has any other value.
  """
  value = os.environ.get(HEAD_GRAD_SCATTER_ENV, "").strip().lower()
  if value in ("", "off"):
    return None
  if value in HEAD_GRAD_SCATTER_MODES:
    return value
  raise ValueError(
      f"{HEAD_GRAD_SCATTER_ENV}={value!r}: expected"
      f" {', '.join(HEAD_GRAD_SCATTER_MODES)} (on), or empty or off (off)."
  )


# Fused scoring for the sharded head: TUNIX_LOGPS_SHARDED_HEAD=fused.
#
# With "local" the head scores each chunk with `selective_log_softmax` (and
# `compute_entropy_from_logits`), and its backward differentiates that with
# `jax.vjp`. At a 248,320-entry vocabulary two parts of it cost more than the
# arithmetic:
#   * The target logit is a `take_along_axis` gather. On TPU, XLA writes an f32
#     copy of the [chunk, vocab] logits for SparseCore to gather the targets
#     from, and the transpose of the gather is a scatter-add: an f32
#     [chunk, vocab] cotangent, a flattened copy of it, a sort of the indices, a
#     SparseCore scatter and a copy back to bf16, per chunk.
#   * The backward recomputes the logsumexp's max and exp-sum, and the entropy
#     recomputes the same reductions, plus one more exp pass, in the forward.
# "fused" scores a chunk in one pass over its logits (`_head_logps_stats`): the
# target logit by a one-hot select-sum, the max and exp-sum once, the entropy
# from the same exponentials, and the max and exp-sum kept for the backward
# (8 bytes per token). When only the log-probs have a cotangent, as in GRPO,
# whose loss stops the entropy's gradient, the backward forms the logits'
# cotangent in one elementwise pass over the recomputed logits
# (`_head_logps_cotangent`): no f32 [chunk, vocab] intermediate, no scatter and
# no reductions. An entropy cotangent takes the "local" backward.
# The log-probs and the logits' cotangent are the f32 operations of the "local"
# path in the same order. The exp-sum, though, is reduced in a different fusion
# (with the entropy's sum, when the entropy is returned), and XLA may order
# that reduction differently, so the log-probs and the gradients can differ
# from "local" in their last bits. The entropy is computed by another formula
# and differs in its last bits.


def _target_hit(
    ids: jax.Array, shape: tuple[int, ...]
) -> tuple[jax.Array, jax.Array]:
  """One-hot of `ids` over the last axis of `shape`, as take_along_axis reads it.

  Args:
    ids: [...] target ids.
    shape: Shape [..., V] of the logits the ids index.

  Returns:
    (hit, valid): `hit[..., v]` is True where v is the target, a negative id
    counting from the end as in take_along_axis; `valid` is False for an id
    outside [-V, V), which take_along_axis's default "fill" mode reads as NaN
    and whose scatter, in its transpose, drops the cotangent.
  """
  vocab = shape[-1]
  ids = jnp.where(ids < 0, ids + vocab, ids)  # take_along_axis wraps negatives
  hit = (
      jax.lax.broadcasted_iota(ids.dtype, shape, len(shape) - 1)
      == ids[..., None]
  )
  return hit, (ids >= 0) & (ids < vocab)


def _head_logps_stats(
    logits: jax.Array, ids: jax.Array, with_entropy: bool
) -> tuple[jax.Array, jax.Array | None, jax.Array, jax.Array]:
  """`selective_log_softmax` (and the entropy) of a chunk, keeping max and sum.

  Args:
    logits: [..., V] logits, as `selective_log_softmax` receives them.
    ids: [...] target ids.
    with_entropy: Also compute the entropy of `compute_entropy_from_logits`.

  Returns:
    (logps, entropy, m, s): `selective_log_softmax(logits, ids)` by the same f32
    operations; the entropy of `compute_entropy_from_logits(logits)` up to f32
    rounding, or None; and the logsumexp's per-token max `m` and exp-sum `s`
    (f32 [...]) for `_head_logps_cotangent`.
  """
  x = jnp.where(jnp.isfinite(logits), logits.astype(jnp.float32), 0.0)
  # jax.nn.logsumexp(x, axis=-1), op for op (jax/_src/ops/special.py, b=None),
  # with its max and exp-sum kept. The max is a stop_gradient there too.
  amax = jnp.max(x, axis=-1, initial=-jnp.inf)
  m = jax.lax.stop_gradient(
      jax.lax.select(jnp.isfinite(amax), amax, jax.lax.full_like(amax, 0))
  )
  shifted = x - m[..., None]
  e = jnp.exp(shifted)
  s = jnp.abs(jnp.sum(e, axis=-1))
  log_s = jnp.log(s)
  # The target logit as a select-sum over the one-hot. It has one non-zero term,
  # so it is exactly the gathered value, and its transpose is a select instead
  # of take_along_axis's scatter-add.
  hit, valid = _target_hit(ids, x.shape)
  target = jnp.sum(jnp.where(hit, x, 0.0), axis=-1)
  target = jnp.where(valid, target, jnp.nan)
  logps = target - (log_s + m)
  entropy = None
  if with_entropy:
    # The entropy from the same exponentials, so the whole forward is one pass
    # over the logits: with p = e / s, -sum(p * log p) = log(s) - sum(e * (x -
    # m)) / s. compute_entropy_from_logits reaches the same value through a
    # second log_softmax and a second exp pass. The entropy is a metric here:
    # GRPO's loss stops its gradient, and an entropy with a cotangent takes the
    # autodiff backward.
    entropy = log_s - jnp.sum(e * shifted, axis=-1) / s
  return logps, entropy, m, s


def _head_logps_cotangent(
    logits: jax.Array,
    ids: jax.Array,
    g: jax.Array,
    m: jax.Array,
    s: jax.Array,
) -> jax.Array:
  """Cotangent of `logits` for `selective_log_softmax(logits, ids)` and `g`.

  What `jax.vjp` of `selective_log_softmax` computes, as one elementwise
  expression. With x the finite-masked f32 logits, autodiff adds the transpose
  of the gather (a scatter-add of g into zeros) to the transpose of the
  logsumexp (sub -> -g, log -> divide by s, abs -> unchanged for s > 0,
  reduce_sum -> broadcast, exp -> multiply by exp(x - m)) and masks the sum
  with isfinite. Here the scatter is a select on the one-hot, which puts the
  same g at the target and zeros elsewhere; the other operations are the same
  f32 operations in the same order, with the max `m` and exp-sum `s` that
  `_head_logps_stats` kept.

  Args:
    logits: [..., V] logits, as `selective_log_softmax` received them.
    ids: [...] target ids.
    g: [...] cotangent of the log-probs.
    m: [...] max from `_head_logps_stats`.
    s: [...] exp-sum from `_head_logps_stats`.

  Returns:
    f32 [..., V] cotangent of `logits`.
  """
  finite = jnp.isfinite(logits)
  x = jnp.where(finite, logits.astype(jnp.float32), 0.0)
  hit, _ = _target_hit(ids, x.shape)
  g = g.astype(jnp.float32)
  d_x = jnp.where(hit, g[..., None], 0.0) + (-g / s)[..., None] * jnp.exp(
      x - m[..., None]
  )
  return jnp.where(finite, d_x, 0.0)


def _mesh_axes(entry) -> tuple[str, ...]:
  if entry is None:
    return ()
  if isinstance(entry, str):
    return (entry,)
  return tuple(entry)


def _token_partition(mesh, batch: int, seq_len: int):
  """Mesh axes sharding the [batch, seq] dims of the hidden states.

  Every mesh axis of size > 1 must shard one of the two, so that every device
  holds distinct tokens and the gradient sum runs over all devices. Tries the
  layout the active flax logical-axis rules give `activation_batch` and
  `activation_length` (an error while reading them skips this step), then one
  greedy pass over the mesh axes in mesh order: each axis of size > 1 shards
  the batch if the batch still divides, else the sequence if it still divides,
  else the pass fails.

  Returns:
    (batch_axes, seq_axes, source), with source "logical-axis-rules" or
    "mesh-order"; or None if both steps fail. The greedy pass can miss a
    layout that exists.
  """
  sizes = dict(mesh.shape)
  nontrivial = {a for a, n in sizes.items() if n > 1}

  def valid(b, s):
    if set(b) & set(s) or not set(b + s) <= set(sizes):
      return False
    if not nontrivial <= set(b + s):
      return False
    return (
        batch % math.prod(sizes[a] for a in b) == 0
        and seq_len % math.prod(sizes[a] for a in s) == 0
    )

  rules = flax.linen.get_logical_axis_rules()
  if rules:
    try:
      spec = flax.linen.logical_to_mesh_axes(
          (_LOGICAL_BATCH_AXIS, _LOGICAL_LENGTH_AXIS), rules
      )
      b, s = _mesh_axes(spec[0]), _mesh_axes(spec[1])
      if valid(b, s):
        return b, s, "logical-axis-rules"
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  b, s = [], []
  for a in mesh.axis_names:
    n = sizes[a]
    if n == 1:
      continue
    if batch % (math.prod(sizes[x] for x in b) * n) == 0:
      b.append(a)
    elif seq_len % (math.prod(sizes[x] for x in s) * n) == 0:
      s.append(a)
    else:
      return None
  return tuple(b), tuple(s), "mesh-order"


def _use_dtypes(jaxpr, var) -> set[Any]:
  """dtypes `var` is consumed in: a convert's target, else its own dtype.

  Recurses into an equation's inner jaxpr (its `jaxpr` or `call_jaxpr` param)
  when that jaxpr has one input per equation input, assuming they line up one
  to one. Known gap: a pass-through output records no use. If the inner jaxpr
  returns `var` unchanged as one of its outputs, a later use of that output is
  missed.
  """
  out = set()
  for eqn in jaxpr.eqns:
    for k, v in enumerate(eqn.invars):
      if v is not var:
        continue
      if eqn.primitive.name == "convert_element_type":
        out.add(jnp.dtype(eqn.params["new_dtype"]))
        continue
      inner = eqn.params.get("jaxpr", eqn.params.get("call_jaxpr"))
      inner = getattr(inner, "jaxpr", inner)
      if inner is not None and len(getattr(inner, "invars", ())) == len(
          eqn.invars
      ):
        out |= _use_dtypes(inner, inner.invars[k])
      else:
        out.add(jnp.dtype(var.aval.dtype))
  return out


def _head_parameters(graphdef, treedef, abstract_leaves, hidden_struct):
  """Leaves `compute_final_logits` reads, mapped to the dtype to gather in.

  A leaf whose every use is a cast to one narrower float dtype (MaxText casts an
  fp32 vocab table to bf16 before the dot) is gathered after that cast, which is
  what XLA already does for the gather inside `compute_chunked_logps`.

  Returns:
    {index into `abstract_leaves`: dtype to gather that leaf in}, for every
    leaf `compute_final_logits` reads that `_use_dtypes` sees. A leaf read only
    through an output of an inner call (for example an identity `jax.jit`,
    `jax.checkpoint` or `custom_jvp`) is missed; see `_use_dtypes`.
  """

  def head(leaves, hs):
    model = nnx.merge(graphdef, jax.tree.unflatten(treedef, leaves))
    return model.compute_final_logits(hs)

  jaxpr = jax.make_jaxpr(head)(list(abstract_leaves), hidden_struct).jaxpr
  out = {}
  for i, var in enumerate(jaxpr.invars[: len(abstract_leaves)]):
    dtypes = _use_dtypes(jaxpr, var)
    if not dtypes:
      continue
    own = jnp.dtype(abstract_leaves[i].dtype)
    dtype = next(iter(dtypes)) if len(dtypes) == 1 else own
    narrower = (
        jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize < own.itemsize
    )
    out[i] = dtype if narrower else own
  return out


def maybe_compute_sharded_chunked_logps(
    model,
    hidden_states,
    input_tokens,
    temperature,
    chunk_size,
    return_entropy,
):
  """`compute_chunked_logps` for packed rows, with a shard_map'ed vocab head.

  Off unless TUNIX_LOGPS_SHARDED_HEAD turns it on (see `sharded_head_mode`).
  The switch is read when the caller is traced and is not part of any jit
  cache key, so a change after the first trace takes effect only on a retrace.
  With "fused", chunks are scored by `_head_logps_stats`, and a backward in
  which only the log-probs have a cotangent forms the logits' cotangent with
  `_head_logps_cotangent` (see the comment above `_target_hit`).

  Returns None, after logging
  "[logps-head] sharded head off, using compute_chunked_logps: <reason>", when:
    * no mesh is set with `jax.set_mesh` (a legacy `with mesh:` sets none);
    * `_token_partition` finds no token layout;
    * `compute_final_logits` reads no model state;
    * a head parameter has no dimension divisible by the number of devices
      (a scalar parameter never has one).
  Some other unsupported inputs raise instead of falling back, for example, on
  a mesh with Explicit axes, or an input whose sharding differs from the head's
  `in_specs` (inside `jax.shard_map`). Sequence axes are used in mesh order, so
  logical-axis rules that list them in another order reshard the input.

  Args:
    model: The merged model; must expose `compute_final_logits`.
    hidden_states: [B, T, H], the model output for every position.
    input_tokens: [B, T] token ids; position t is scored from position t - 1.
    temperature: Sampling temperature the logits are divided by (0 or 1: none).
    chunk_size: Sequence positions per scan step on each device, capped at the
      length of the device's block of the sequence.
    return_entropy: Whether to also return the per-token entropy.

  Returns:
    None, or the packed-row result of `compute_per_token_logps`: per-token
    log-probs [B, T] whose index j scores `input_tokens[:, j]` from
    `hidden_states[:, j - 1]` (index 0 is 0), plus the entropy [B, T] if
    `return_entropy`.

  With TUNIX_LOGPS_HEAD_GRAD_SCATTER on (see `head_grad_scatter_mode`), the
  backward reduce-scatters each chunk's parameter gradient inside the chunk
  loop instead of accumulating full-size float32 gradients, and logs one
  "[logps-head] table-gradient scatter <mode>: ..." warning per trace.

  Raises:
    ValueError: TUNIX_LOGPS_SHARDED_HEAD or TUNIX_LOGPS_HEAD_GRAD_SCATTER has an
      unrecognized value, or TUNIX_LOGPS_HEAD_GRAD_SCATTER is on while
      TUNIX_LOGPS_SHARDED_HEAD is off.
  """
  mode = sharded_head_mode()
  grad_scatter = head_grad_scatter_mode()
  if mode is None:
    if grad_scatter is not None:
      raise ValueError(
          f"{HEAD_GRAD_SCATTER_ENV}={grad_scatter} needs the sharded head: set"
          f" {SHARDED_HEAD_ENV} to local or fused, or unset"
          f" {HEAD_GRAD_SCATTER_ENV}."
      )
    return None

  def fallback(reason):
    logging.warning(
        "[logps-head] sharded head off, using compute_chunked_logps: %s", reason
    )
    if grad_scatter is not None:
      logging.warning(
          "[logps-head] %s=%s has no effect: the sharded head is off",
          HEAD_GRAD_SCATTER_ENV,
          grad_scatter,
      )
    return None

  mesh = jax.sharding.get_abstract_mesh()
  if mesh.empty:
    return fallback("no mesh in context (jax.set_mesh)")
  batch, seq_len, hidden_dim = hidden_states.shape
  partition = _token_partition(mesh, batch, seq_len)
  if partition is None:
    return fallback(f"no token layout for {batch}x{seq_len} on {mesh.shape}")
  batch_axes, seq_axes, layout_source = partition
  # `jax.lax.ppermute` numbers the sequence shards in mesh order, so the specs
  # list the sequence axes in mesh order too. A rule that lists them in another
  # order then costs a reshard on entry instead of shifting tokens across the
  # wrong shards.
  seq_axes = tuple(a for a in mesh.axis_names if a in seq_axes)
  gather_axes = tuple(a for a in mesh.axis_names if mesh.shape[a] > 1)
  num_devices = math.prod(mesh.shape[a] for a in gather_axes)
  local_batch = batch // math.prod(mesh.shape[a] for a in batch_axes)
  local_len = seq_len // math.prod(mesh.shape[a] for a in seq_axes)
  chunk = min(chunk_size, local_len)

  graphdef, state = nnx.split(model)
  leaves, treedef = jax.tree.flatten(state)
  abstract = [
      jax.ShapeDtypeStruct(jnp.shape(x), jnp.result_type(x)) for x in leaves
  ]
  head = _head_parameters(
      graphdef,
      treedef,
      abstract,
      jax.ShapeDtypeStruct(
          (local_batch, chunk, hidden_dim), hidden_states.dtype
      ),
  )
  if not head:
    return fallback("compute_final_logits reads no model state")
  head_index = sorted(head)
  params = [leaves[i] for i in head_index]
  gather_dtypes = [head[i] for i in head_index]
  gather_dims = []
  for p in params:
    dim = next((d for d, n in enumerate(p.shape) if n % num_devices == 0), None)
    if dim is None:
      return fallback(
          f"head parameter {p.shape} not divisible by {num_devices}"
      )
    gather_dims.append(dim)
  param_dtypes = [jnp.result_type(p) for p in params]
  num_out = 2 if return_entropy else 1
  fused = mode == "fused"
  logging.info(
      "[logps-head] sharded head %s: tokens batch=%s seq=%s (%s), %dx%d local"
      " tokens in %d steps of %d; gathers %s over %s",
      mode,
      batch_axes,
      seq_axes,
      layout_source,
      local_batch,
      local_len,
      -(-local_len // chunk),  # ceil(local_len / chunk) scan steps
      chunk,
      [(p.shape, str(d)) for p, d in zip(params, gather_dtypes)],
      gather_axes,
  )
  grad_shard_shapes = [
      tuple(n // num_devices if k == d else n for k, n in enumerate(p.shape))
      for p, d in zip(params, gather_dims)
  ]
  if grad_scatter is not None:
    logging.warning(
        "[logps-head] table-gradient scatter %s: each of the %d chunks'"
        " parameter gradients is reduce-scattered over %s inside the backward's"
        " chunk loop (wire %s), accumulated in float32 shards %s%s",
        grad_scatter,
        -(-local_len // chunk),
        gather_axes,
        "float32"
        if grad_scatter == "chunk_f32"
        else [str(d) for d in gather_dtypes],
        grad_shard_shapes,
        "; chunk loop unrolled" if grad_scatter == "chunk_unroll" else "",
    )

  def merged(full):
    # Only the head leaves carry values. The others stay ShapeDtypeStruct
    # placeholders, which compute_final_logits never reads (_head_parameters
    # found the leaves it reads).
    values = list(abstract)
    for i, x in zip(head_index, full):
      values[i] = x
    return nnx.merge(graphdef, jax.tree.unflatten(treedef, values))

  def chunk_logits(full, hs):
    # The f32 logits a chunk is scored from.
    logits = merged(full).compute_final_logits(hs).astype(jnp.float32)
    if temperature != 0.0 and temperature != 1.0:
      logits /= temperature
    return logits

  def chunk_outputs(full, hs, ids, with_entropy):
    logits = chunk_logits(full, hs)
    logps = selective_log_softmax(logits, ids)
    if with_entropy:
      return logps, compute_entropy_from_logits(logits)
    return (logps,)

  def gather(p, dim, dtype):
    p = p.astype(dtype)
    if not gather_axes:
      return p
    return jax.lax.all_gather(p, gather_axes, axis=dim, tiled=True)

  def scatter(g, dim):
    if not gather_axes:
      return g
    return jax.lax.psum_scatter(
        g, gather_axes, scatter_dimension=dim, tiled=True
    )

  # TUNIX_LOGPS_HEAD_GRAD_SCATTER on: the chunk loop's carry is each
  # parameter's float32 shard, and each chunk's gradient is reduce-scattered
  # (in float32 for chunk_f32, else in the dtype it arrives in) and added to it.
  def chunk_grad_acc_init():
    return tuple(jnp.zeros(s, jnp.float32) for s in grad_shard_shapes)

  def chunk_grad_acc_add(acc, d_full):
    out = []
    for a, g, d in zip(acc, d_full, gather_dims):
      if grad_scatter == "chunk_f32":
        g = g.astype(jnp.float32)
      # The barrier keeps XLA's while-loop all-reduce code motion from moving
      # the reduce-scatter after the loop, which would bring back the full-size
      # accumulator (and, with the cast below, fails a RET_CHECK on TPU).
      shard = jax.lax.optimization_barrier(scatter(g, d))
      out.append(a + shard.astype(jnp.float32))
    return tuple(out)

  chunk_scan_kwargs = (
      {"unroll": True} if grad_scatter == "chunk_unroll" else {}
  )

  def chunk_outputs_fused(full, hs, ids, with_stats):
    # `chunk_outputs` for "fused", plus each token's max and exp-sum when the
    # backward will need them.
    logps, entropy, m, s = _head_logps_stats(
        chunk_logits(full, hs), ids, return_entropy
    )
    outs = (logps, entropy) if return_entropy else (logps,)
    return outs + (m, s) if with_stats else outs

  def scan_forward(full, h, t, with_stats=False):
    def step(bufs, i):
      start = i * chunk
      hs = jax.lax.dynamic_slice_in_dim(h, start, chunk, axis=1)
      ids = jax.lax.dynamic_slice_in_dim(t, start, chunk, axis=1)
      if fused:
        outs = chunk_outputs_fused(full, hs, ids, with_stats)
      else:
        outs = chunk_outputs(full, hs, ids, return_entropy)
      bufs = tuple(
          jax.lax.dynamic_update_slice_in_dim(b, o, start, axis=1)
          for b, o in zip(bufs, outs)
      )
      return bufs, None

    num_bufs = num_out + (2 if with_stats else 0)
    bufs = tuple(jnp.zeros(t.shape, jnp.float32) for _ in range(num_bufs))
    bufs, _ = jax.lax.scan(step, bufs, jnp.arange(h.shape[1] // chunk))
    return bufs

  @jax.custom_vjp
  def head_local(h, t, *ps):
    full = [
        gather(p, d, dt) for p, d, dt in zip(ps, gather_dims, gather_dtypes)
    ]
    return scan_forward(full, h, t)

  def head_local_fwd(h, t, *ps):
    # With symbolic_zeros=True the inputs arrive as CustomVJPPrimal wrappers,
    # hence `.value`. Their `.perturbed` flags are ignored, so the backward
    # builds the f32 accumulator even for a frozen head. The gathered
    # parameters are the residuals: the backward reuses them instead of
    # gathering again.
    h, t = h.value, t.value
    ps = [p.value for p in ps]
    full = [
        gather(p, d, dt) for p, d, dt in zip(ps, gather_dims, gather_dtypes)
    ]
    if fused:
      # Also keep the max and exp-sum of every token (f32 [B, T] each).
      bufs = scan_forward(full, h, t, with_stats=True)
      return bufs[:num_out], (h, t, tuple(full), bufs[num_out:])
    return scan_forward(full, h, t), (h, t, tuple(full))

  def bwd_fused(h, t, full, stats, d_logps):
    # The "local" backward's token chunks, with the logits' cotangent formed in
    # one pass (`_head_logps_cotangent`) from the forward's max and exp-sum
    # instead of by autodiff of `selective_log_softmax` (a scatter-add and f32
    # [chunk, vocab] copies). The table and hidden-state cotangents still come
    # from `jax.vjp` of `compute_final_logits` and are accumulated as there.
    m_all, s_all = stats
    scaled = temperature != 0.0 and temperature != 1.0

    def step(carry, i):
      acc, dh = carry
      start = i * chunk
      hs = jax.lax.dynamic_slice_in_dim(h, start, chunk, axis=1)
      ids = jax.lax.dynamic_slice_in_dim(t, start, chunk, axis=1)
      g = jax.lax.dynamic_slice_in_dim(d_logps, start, chunk, axis=1)
      m = jax.lax.dynamic_slice_in_dim(m_all, start, chunk, axis=1)
      s = jax.lax.dynamic_slice_in_dim(s_all, start, chunk, axis=1)
      raw, vjp = jax.vjp(
          lambda full_, hs_: merged(full_).compute_final_logits(hs_),
          list(full),
          hs,
      )
      # `chunk_logits` on `raw`, the logits' cotangent, then the transposes of
      # the temperature division (a division) and of `astype(float32)` (a
      # cast).
      logits = raw.astype(jnp.float32)
      if scaled:
        logits /= temperature
      d_logits = _head_logps_cotangent(logits, ids, g, m, s)
      if scaled:
        d_logits /= temperature
      # An identity that pins one materialized cotangent. Without it XLA:TPU
      # writes the cotangent for the hidden-state matmul and recomputes it (exp
      # included) inside the table-gradient matmul's operand fusion; with it
      # the cotangent becomes the logits matmul's epilogue and both gradient
      # matmuls read it (in a deviceless TPU compile of the head at chunk
      # 1024: 0.47 GiB fewer temporaries and a third fewer transcendentals).
      d_logits = jax.lax.optimization_barrier(d_logits.astype(raw.dtype))
      d_full, d_hs = vjp(d_logits)
      if grad_scatter is None:
        acc = tuple(a + gg.astype(jnp.float32) for a, gg in zip(acc, d_full))
      else:
        acc = chunk_grad_acc_add(acc, d_full)
      dh = jax.lax.dynamic_update_slice_in_dim(
          dh, d_hs.astype(dh.dtype), start, axis=1
      )
      return (acc, dh), None

    if grad_scatter is None:
      acc = tuple(jnp.zeros(f.shape, jnp.float32) for f in full)
      (acc, dh), _ = jax.lax.scan(
          step, (acc, jnp.zeros_like(h)), jnp.arange(h.shape[1] // chunk)
      )
      acc = [scatter(a, d) for a, d in zip(acc, gather_dims)]
    else:
      (acc, dh), _ = jax.lax.scan(
          step,
          (chunk_grad_acc_init(), jnp.zeros_like(h)),
          jnp.arange(h.shape[1] // chunk),
          **chunk_scan_kwargs,
      )
    return (dh, None, *[a.astype(dt) for a, dt in zip(acc, param_dtypes)])

  def head_local_bwd(res, cts):
    if fused:
      h, t, full, stats = res
    else:
      h, t, full = res
    d_logps = cts[0]
    # Without return_entropy the primal has one output, so no entropy
    # cotangent: treat it as a symbolic zero.
    d_entropy = cts[1] if return_entropy else SymbolicZero(None)
    use_logps = not isinstance(d_logps, SymbolicZero)
    use_entropy = not isinstance(d_entropy, SymbolicZero)
    if not (use_logps or use_entropy):
      shard_shapes = [
          tuple(
              n // num_devices if k == d else n for k, n in enumerate(f.shape)
          )
          for f, d in zip(full, gather_dims)
      ]
      zeros = [jnp.zeros(s, dt) for s, dt in zip(shard_shapes, param_dtypes)]
      return (jnp.zeros_like(h), None, *zeros)
    if fused and use_logps and not use_entropy:
      # An entropy with a cotangent (a loss that differentiates the entropy)
      # takes the autodiff backward below, which is the "local" backward
      # exactly.
      return bwd_fused(h, t, full, stats, d_logps)

    def step(carry, i):
      acc, dh = carry
      start = i * chunk
      hs = jax.lax.dynamic_slice_in_dim(h, start, chunk, axis=1)
      ids = jax.lax.dynamic_slice_in_dim(t, start, chunk, axis=1)
      cot = []
      if use_logps:
        cot.append(jax.lax.dynamic_slice_in_dim(d_logps, start, chunk, axis=1))
      if use_entropy:
        cot.append(
            jax.lax.dynamic_slice_in_dim(d_entropy, start, chunk, axis=1)
        )

      def outputs_fn(full_, hs_):
        # Exactly the outputs that have a cotangent in `cot`, in that order.
        outs = chunk_outputs(full_, hs_, ids, use_entropy)
        return outs if use_logps else outs[1:]

      _, vjp = jax.vjp(outputs_fn, list(full), hs)
      d_full, d_hs = vjp(tuple(cot))
      if grad_scatter is None:
        acc = tuple(a + g.astype(jnp.float32) for a, g in zip(acc, d_full))
      else:
        acc = chunk_grad_acc_add(acc, d_full)
      dh = jax.lax.dynamic_update_slice_in_dim(
          dh, d_hs.astype(dh.dtype), start, axis=1
      )
      return (acc, dh), None

    if grad_scatter is None:
      acc = tuple(jnp.zeros(f.shape, jnp.float32) for f in full)
      (acc, dh), _ = jax.lax.scan(
          step, (acc, jnp.zeros_like(h)), jnp.arange(h.shape[1] // chunk)
      )
      acc = [scatter(a, d) for a, d in zip(acc, gather_dims)]
    else:
      (acc, dh), _ = jax.lax.scan(
          step,
          (chunk_grad_acc_init(), jnp.zeros_like(h)),
          jnp.arange(h.shape[1] // chunk),
          **chunk_scan_kwargs,
      )
    d_params = [a.astype(dt) for a, dt in zip(acc, param_dtypes)]
    return (dh, None, *d_params)

  head_local.defvjp(head_local_fwd, head_local_bwd, symbolic_zeros=True)

  seq_shards = math.prod(mesh.shape[a] for a in seq_axes)

  def neighbour(x, step):
    """`x` of the sequence shard `step` places later; zeros past either end."""
    if seq_shards == 1:
      return jnp.zeros_like(x)
    perm = [
        (i, i - step) for i in range(seq_shards) if 0 <= i - step < seq_shards
    ]
    return jax.lax.ppermute(x, seq_axes, perm)

  def local(h, tok, *ps):
    # Position t predicts token t + 1; a shard's last position predicts the
    # next shard's first token. The sequence's last position has no next
    # token: its target is token id 0 (`neighbour` returns zeros past the last
    # shard) and its score is dropped by the output shift below. Both one-token
    # shifts are explicit collective-permutes, so neither depends on how XLA
    # partitions a shift of a sequence-sharded array.
    t = jnp.concatenate([tok[:, 1:], neighbour(tok[:, :1], 1)], axis=1)
    pad = -h.shape[1] % chunk  # up to the next multiple of chunk
    if pad:
      h = jnp.pad(h, ((0, 0), (0, pad), (0, 0)))
      t = jnp.pad(t, ((0, 0), (0, pad)))
    outs = [o[:, : o.shape[1] - pad] for o in head_local(h, t, *ps)]
    # Index j holds token j's log-prob (scored at position j - 1); index 0 is 0.
    return tuple(
        jnp.concatenate([neighbour(o[:, -1:], -1), o[:, :-1]], axis=1)
        for o in outs
    )

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  token_spec = P(batch_axes or None, seq_axes or None)
  in_specs = (
      P(batch_axes or None, seq_axes or None, None),
      token_spec,
      *[
          P(*[gather_axes if k == d else None for k in range(p.ndim)])
          for p, d in zip(params, gather_dims)
      ],
  )
  # Zero non-finite hidden states, as compute_chunked_logps does.
  hidden_states = jnp.where(jnp.isfinite(hidden_states), hidden_states, 0.0)
  outs = jax.shard_map(
      local,
      mesh=mesh,
      in_specs=in_specs,
      out_specs=(token_spec,) * num_out,
      # The scan carries start as jnp.zeros, which the vma check types as the
      # same on every device, while each chunk's outputs vary by device, so
      # with the check lax.scan rejects the carry. Unchecked is still exact:
      # every mesh axis of size > 1 is in every in and out spec, so the
      # transpose adds no psum or division over a nontrivial axis.
      check_vma=False,
  )(hidden_states, input_tokens, *params)
  return outs if return_entropy else outs[0]


@nnx.jit(static_argnames=("pad_id", "eos_id", "stop_gradient"))
def compute_score(
    model,
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    stop_gradient: bool = True,
    segment_ids: jax.Array | None = None,
    segment_positions: jax.Array | None = None,
):
  """Computes reward using the provided model."""
  (
      prompt_completion_ids,
      calculated_positions,
      attn_mask,
      input_seg_ids,
  ) = process_ids(
      prompt_tokens,
      completion_tokens,
      pad_id,
      eos_id,
      segment_ids,
      segment_positions,
  )

  has_segment_ids = model_call_contains(model, "segment_ids")
  model_kwargs = {"positions": calculated_positions, "cache": None}
  if has_segment_ids and segment_ids is not None:
    model_kwargs["segment_ids"] = segment_ids
  else:
    model_kwargs["attention_mask"] = attn_mask
    if has_segment_ids and input_seg_ids is not None:
      model_kwargs["segment_ids"] = input_seg_ids

  out = model(prompt_completion_ids, **model_kwargs)
  per_token_scores = out[0] if isinstance(out, tuple) else out
  # The model returns a tensor of shape [B, T, 1]. We squeeze the last
  # dimension to get a tensor of shape [B, T].
  per_token_scores = jnp.squeeze(per_token_scores, axis=-1)

  if stop_gradient:
    per_token_scores = jax.lax.stop_gradient(per_token_scores)

  return per_token_scores


def np_make_completion_mask(
    completion_ids: np.ndarray, eos_tok: int = 0
) -> np.ndarray:
  """Numpy version of make_completion_mask which executes on CPU.

  Args:
    completion_ids: Completion ids with shape [B, T].
    eos_tok: EOS token id.

  Returns:
    Completion mask.
  """
  is_eos = completion_ids == eos_tok
  seq_len = is_eos.shape[1]

  first_eos_idx = np.argmax(is_eos, axis=1)
  any_eos = np.any(is_eos, axis=1)
  eos_idx = np.where(any_eos, first_eos_idx, seq_len)
  sequence_indices = np.arange(seq_len)

  return (sequence_indices < eos_idx[:, None] + 1).astype(np.int32)


def make_completion_mask(
    completion_ids: jax.Array, eos_tok: int = 0
) -> jax.Array:
  """Create completion mask based on the EOS token.

  Args:
    completion_ids: Completion ids with shape [B, T].
    eos_tok: EOS token id.

  Returns:
    Completion mask.
  """
  is_eos = completion_ids == eos_tok
  eos_idx = jnp.full((is_eos.shape[0],), is_eos.shape[1], dtype=jnp.int32)

  any_eos = jnp.any(is_eos, axis=1)
  eos_idx = jax.lax.select(any_eos, jnp.argmax(is_eos, axis=1), eos_idx)

  sequence_indices = jnp.arange(is_eos.shape[1])[None, :]
  sequence_indices = jnp.broadcast_to(
      sequence_indices, (is_eos.shape[0], is_eos.shape[1])
  )
  return (sequence_indices <= eos_idx[:, None]).astype(jnp.int32)


def pad_to_length(
    x: jax.Array,
    target_length: int,
    pad_value: int = 0,
    left=False,
    axis: int = 0,
) -> jax.Array:
  """Pads a JAX array to a specified target length along a given axis.

  Args:
      x: The JAX array to pad.
      target_length: The desired length of the padded array.
      pad_value: The value to use for padding (default: 0).
      left: If True, add padding tokens to the left of the array.
      axis: The axis along which to pad (default: 0).

  Returns:
      A new JAX array that is padded to the target length along the specified
      axis. Return original array if it is already longer than the target
      length.
  """
  length = x.shape[axis]
  if length >= target_length:
    return x

  padding_shape = list(x.shape)
  padding_shape[axis] = target_length - length
  padding = jnp.full(padding_shape, pad_value, dtype=x.dtype)

  if left:
    return jnp.concatenate([padding, x], axis=axis)
  else:
    return jnp.concatenate([x, padding], axis=axis)


def segmented_sum(
    values: jax.Array,
    segment_ids: jax.Array,
    num_segments: int,
) -> jax.Array:
  """Per-row segment sum: `[B, T]` values grouped by `[B, T]` ids -> `[B, S]`.

  `jax.ops.segment_sum` is 1-D, so this `vmap`s it across the batch. Segment 0
  is the padding bucket by the `pack_sequences` convention. `indices_are_sorted`
  is left False on purpose: our ids look like `[1,1,2,2,3,0,0]` (real segments
  ascending, then trailing pad 0), which is non-monotonic, so the sorted-scan
  fast path would be an invalid promise (undefined behaviour on TPU) -- we take
  correctness over a marginal perf hint on this tiny `[B, S]` reduction.

  Args:
    values: `[batch_size, sequence_len]` values to group and sum per segment.
    segment_ids: `[batch_size, sequence_len]` segment id per position (0 = pad).
    num_segments: static upper bound on segment ids (padding bucket included).
      Must be a Python int because the output width `S` is static under JIT.

  Returns:
    `[batch_size, num_segments]` per-row per-segment sums.
  """
  return jax.vmap(
      functools.partial(
          jax.ops.segment_sum,
          num_segments=num_segments,
          indices_are_sorted=False,
      )
  )(values, segment_ids)


def segmented_count(
    segment_ids: jax.Array,
    num_segments: int,
    mask: jax.Array | None = None,
) -> jax.Array:
  """Per-row per-segment token count -> `[B, S]` (float32).

  Counts positions per segment. With `mask` (typically `completion_mask`), only
  `mask != 0` positions are counted, so the count matches the denominator of a
  per-segment token mean (scored tokens only). Segment 0 is the padding bucket.

  Args:
    segment_ids: `[batch_size, sequence_len]` segment id per position (0 = pad).
    num_segments: static upper bound on segment ids (padding bucket included).
    mask: optional `[batch_size, sequence_len]`; when given, only `mask != 0`
      positions are counted.

  Returns:
    `[batch_size, num_segments]` per-row per-segment counts (float32).
  """
  if mask is None:
    ones = jnp.ones_like(segment_ids, dtype=jnp.float32)
  else:
    ones = mask.astype(jnp.float32)
  return segmented_sum(ones, segment_ids, num_segments)


def aggregate_loss(
    per_token_loss: jax.Array,
    completion_mask: jax.Array,
    loss_agg_mode: str,
    segment_ids: jax.Array | None = None,
    num_segments: int | None = None,
    **kwargs: Any,
) -> utils.WeightedMetric:
  """Aggregate loss based on the loss aggregation mode.

  Args:
      per_token_loss: Per token loss.[batch_size, sequence_len]
      completion_mask: Completion mask.[batch_size, sequence_len]
      loss_agg_mode: Loss aggregation mode.
    segment_ids: optional [batch_size, sequence_len] packing segment ids. When
      provided, per-sequence reductions operate on segments rather than rows (a
      packed row holding K segments contributes K separate "sequences"); segment
      0 is the padding bucket and is excluded. Pairs with num_segments. When
      None, the per-row branch below runs unchanged.
    num_segments: static upper bound on segments per row (padding bucket
      included). Required when segment_ids is not None.

  Returns:
      Aggregated loss.
  """

  # Sever forward and reverse-mode autodiff paths on masked tokens before any
  # multiplicative reduction so 0.0 * Inf = NaN cannot propagate from padding.
  per_token_loss = jnp.where(
      completion_mask > 0, per_token_loss.astype(jnp.float32), 0.0
  )

  if segment_ids is not None:
    return _aggregate_loss_segmented(
        per_token_loss,
        completion_mask,
        loss_agg_mode,
        segment_ids,
        num_segments,
        **kwargs,
    )

  if loss_agg_mode == "token-mean":
    # sum all the token loss, and average by total number of completion tokens
    # in the batch
    unreduced_sum = (per_token_loss * completion_mask).sum()
    denominator = completion_mask.sum()
    min_denom = 1.0
  elif loss_agg_mode == "sequence-mean-token-mean":
    seq_mask = completion_mask.sum(axis=-1)  # per-sequence token count
    seq_loss = ((per_token_loss * completion_mask).sum(axis=-1)) / jnp.clip(
        seq_mask, min=1.0
    )
    unreduced_sum = seq_loss.sum()
    # Count non-empty rows so unpacked [N_max, R] padding rows do not inflate the
    # denominator (matches seq-mean-token-sum). For a fully-populated batch this
    # equals shape[0], so the non-packed path is unchanged.
    denominator = (seq_mask > 0).sum()
    min_denom = 1.0
  elif loss_agg_mode == "sequence-mean-token-scale":
    # Look up custom normalization factor, default to max response length.
    norm = _check_get_norm(kwargs, per_token_loss.shape[-1])

    # Scale by maximum response length instead of actual response length.
    seq_loss = (per_token_loss * completion_mask).sum(axis=-1) / jnp.clip(
        norm, min=1e-6
    )
    unreduced_sum = seq_loss.sum()
    denominator = (completion_mask.sum(axis=-1) > 0).sum()
    min_denom = 1.0
  elif loss_agg_mode == "seq-mean-token-sum":
    # 1) sum token losses within each sequence
    # 2) average only across sequences that have at least one valid token
    seq_loss = (per_token_loss * completion_mask).sum(axis=-1)
    seq_mask = (completion_mask.sum(axis=-1) > 0).astype(jnp.float32)
    unreduced_sum = (seq_loss * seq_mask).sum()
    denominator = seq_mask.sum()
    min_denom = 1e-6
  elif loss_agg_mode == "sequence-mean-token-sum-norm":
    # Get custom normalization factor from kwargs, default to batch size.
    norm = _check_get_norm(kwargs, per_token_loss.shape[0])
    unreduced_sum = (per_token_loss * completion_mask).sum()
    denominator = norm
    min_denom = 1e-6
  else:
    raise ValueError(
        f"Unsupported loss aggregation mode: {loss_agg_mode}. Supported modes:"
        " 'token-mean', 'sequence-mean-token-mean',"
        " 'sequence-mean-token-scale', 'seq-mean-token-sum',"
        " 'sequence-mean-token-sum-norm'."
    )
  return utils.WeightedMetric(
      jnp.asarray(unreduced_sum, dtype=jnp.float32),
      jnp.asarray(denominator, dtype=jnp.float32),
      min_denom=min_denom,
  )


def _aggregate_loss_segmented(
    per_token_loss: jax.Array,
    completion_mask: jax.Array,
    loss_agg_mode: str,
    segment_ids: jax.Array,
    num_segments: int | None,
    **kwargs: Any,
) -> utils.WeightedMetric:
  """Segment-aware `aggregate_loss` for packed batches.

  A packed row holds K sequences separated by `segment_ids` (real ids 1..K;
  0 = padding bucket). Per-sequence reductions group by SEGMENT instead of by
  row, so the K packed sequences each contribute separately -- the per-segment
  mirror of `aggregate_loss`'s per-row branch. The padding bucket (segment 0)
  is excluded from both numerator and denominator, so dummy/padding segments
  never dilute the loss. Returns the same `WeightedMetric` as the per-row
  branch, so no downstream (LossOutput / gradient) change is needed.

  Args:
    per_token_loss: Per token loss. [batch_size, sequence_len]
    completion_mask: Completion mask. [batch_size, sequence_len]
    loss_agg_mode: Loss aggregation mode (same set as `aggregate_loss`).
    segment_ids: [batch_size, sequence_len] packing segment ids (0 = padding).
    num_segments: static upper bound on segments per row (padding bucket
      included). Required -- raises ValueError if None.
    **kwargs: mode-specific extras, e.g. `norm` for the token-scale and
      token-sum-norm modes.

  Returns:
    Aggregated loss as a `WeightedMetric` (division deferred).
  """
  if num_segments is None:
    raise ValueError(
        "num_segments must be provided when segment_ids is not None."
    )

  masked_loss = per_token_loss * completion_mask
  # [B, S]: per-segment loss sum and scored-token count.
  l_seg = segmented_sum(masked_loss, segment_ids, num_segments)
  c_seg = segmented_count(segment_ids, num_segments, mask=completion_mask)
  # Active = segments with >=1 scored token; zero the padding bucket (seg 0) so
  # dummy/padding never dilutes numerator or denominator (denom = n_act).
  a_seg = (c_seg > 0).astype(jnp.float32)
  a_seg = a_seg.at[:, 0].set(0.0)
  n_act = a_seg.sum()

  if loss_agg_mode == "token-mean":
    # Segment-agnostic: the completion mask already handles everything.
    unreduced_sum = masked_loss.sum()
    denominator = completion_mask.sum()
    min_denom = 1.0
  elif loss_agg_mode == "sequence-mean-token-mean":
    per_seg_mean = jnp.where(a_seg > 0, l_seg / jnp.clip(c_seg, min=1.0), 0.0)
    unreduced_sum = per_seg_mean.sum()
    denominator = n_act
    min_denom = 1.0
  elif loss_agg_mode == "sequence-mean-token-scale":
    # Fail-loud: the default row-width norm is the whole pack budget, which
    # would dilute every segment -- require an explicit norm.
    if "norm" not in kwargs:
      raise ValueError(
          "sequence-mean-token-scale under sequence packing requires an"
          " explicit 'norm' (e.g. max_response_length); the default row-width"
          " norm is the pack budget, which dilutes every sequence."
      )
    norm = _check_get_norm(kwargs, per_token_loss.shape[-1])
    per_seg_scaled = jnp.where(a_seg > 0, l_seg / jnp.clip(norm, min=1e-6), 0.0)
    unreduced_sum = per_seg_scaled.sum()
    denominator = n_act
    min_denom = 1.0
  elif loss_agg_mode == "seq-mean-token-sum":
    unreduced_sum = jnp.where(a_seg > 0, l_seg, 0.0).sum()
    denominator = n_act
    min_denom = 1e-6
  elif loss_agg_mode == "sequence-mean-token-sum-norm":
    # Default norm = active segment count (per-segment analog of row count).
    norm = _check_get_norm(kwargs, n_act)
    unreduced_sum = masked_loss.sum()
    denominator = norm
    min_denom = 1e-6
  else:
    raise ValueError(
        f"Unsupported loss aggregation mode: {loss_agg_mode}. Supported modes:"
        " 'token-mean', 'sequence-mean-token-mean',"
        " 'sequence-mean-token-scale', 'seq-mean-token-sum',"
        " 'sequence-mean-token-sum-norm'."
    )
  return utils.WeightedMetric(
      jnp.asarray(unreduced_sum, dtype=jnp.float32),
      jnp.asarray(denominator, dtype=jnp.float32),
      min_denom=min_denom,
  )


def reduced_loss_agg(
    per_token_loss: jax.Array,
    completion_mask: jax.Array,
    loss_agg_mode: str,
    segment_ids: jax.Array | None = None,
    num_segments: int | None = None,
    **kwargs: Any,
) -> jax.Array:
  """Eager per-sequence loss reduction (the pre-unreduced form).

  Divides immediately and returns a scalar, unlike `aggregate_loss` which
  defers division into a `WeightedMetric`. Used only as the `reduced_pg_loss`
  logging metric to guard against regression; it never feeds the gradient.
  Numerically equal to `aggregate_loss(...).compute()` (asserted in
  common_test.py::CommonTest.test_reduced_equals_unreduced_compute) -- kept as
  an INDEPENDENT eager form (this divides eagerly; `aggregate_loss` defers into
  a WeightedMetric) so the two computation paths cross-check each other.

  Segment-aware: when `segment_ids` is given it reduces per SEGMENT via an
  independent eager branch (`_reduced_loss_agg_segmented`, NOT a call into
  `aggregate_loss`), so the metric is packing-invariant -- the same sequences
  give the same value packed or unpacked.

  Args:
    per_token_loss: Per token loss. [batch_size, sequence_len]
    completion_mask: Completion mask. [batch_size, sequence_len]
    loss_agg_mode: Loss aggregation mode.
    segment_ids: optional [batch_size, sequence_len] packing segment ids; when
      given, reduce per segment (segment 0 = padding bucket, excluded).
    num_segments: static upper bound on segments per row; required with
      segment_ids.
    **kwargs: Mode-specific extras (e.g. `norm`).

  Returns:
    A scalar reduced loss.
  """
  per_token_loss = jnp.where(
      completion_mask > 0, per_token_loss.astype(jnp.float32), 0.0
  )

  if segment_ids is not None:
    return _reduced_loss_agg_segmented(
        per_token_loss,
        completion_mask,
        loss_agg_mode,
        segment_ids,
        num_segments,
        **kwargs,
    )

  if loss_agg_mode == "token-mean":
    return (per_token_loss * completion_mask).sum() / jnp.clip(
        completion_mask.sum(), min=1.0
    )
  elif loss_agg_mode == "sequence-mean-token-mean":
    seq_mask = completion_mask.sum(axis=-1)
    seq_loss = (per_token_loss * completion_mask).sum(axis=-1) / jnp.clip(
        seq_mask, min=1.0
    )
    return seq_loss.mean()
  elif loss_agg_mode == "sequence-mean-token-scale":
    norm = _check_get_norm(kwargs, per_token_loss.shape[-1])
    seq_loss = (per_token_loss * completion_mask).sum(axis=-1) / jnp.clip(
        norm, min=1e-6
    )
    return seq_loss.mean()
  elif loss_agg_mode == "seq-mean-token-sum":
    seq_loss = (per_token_loss * completion_mask).sum(axis=-1)
    seq_mask = (completion_mask.sum(axis=-1) > 0).astype(jnp.float32)
    return (seq_loss * seq_mask).sum() / jnp.clip(seq_mask.sum(), min=1e-6)
  elif loss_agg_mode == "sequence-mean-token-sum-norm":
    norm = _check_get_norm(kwargs, per_token_loss.shape[0])
    return (per_token_loss * completion_mask).sum() / jnp.clip(norm, min=1e-6)
  else:
    raise ValueError(
        f"Unsupported loss aggregation mode: {loss_agg_mode}. Supported modes:"
        " 'token-mean', 'sequence-mean-token-mean',"
        " 'sequence-mean-token-scale', 'seq-mean-token-sum',"
        " 'sequence-mean-token-sum-norm'."
    )


def _reduced_loss_agg_segmented(
    per_token_loss: jax.Array,
    completion_mask: jax.Array,
    loss_agg_mode: str,
    segment_ids: jax.Array,
    num_segments: int | None,
    **kwargs: Any,
) -> jax.Array:
  """Eager per-SEGMENT reduction (scalar): the packed mirror of the per-row

  branch of `reduced_loss_agg`.

  Deliberately INDEPENDENT of `_aggregate_loss_segmented` (which defers division
  into a WeightedMetric) -- it divides eagerly -- so the two forms cross-check.
  Shares only the `segmented_sum`/`segmented_count` primitives. Each mode's
  `jnp.clip(denom, min=X)` uses the SAME `X` as the matching WeightedMetric
  `min_denom` (1.0 for the mean modes, 1e-6 for the sum modes) so the eager and
  deferred forms produce the same scalar.

  Args:
    per_token_loss: Per token loss. [batch_size, sequence_len]
    completion_mask: Completion mask. [batch_size, sequence_len]
    loss_agg_mode: Loss aggregation mode (same set as `reduced_loss_agg`).
    segment_ids: [batch_size, sequence_len] packing segment ids (0 = padding).
    num_segments: static upper bound on segments per row (padding bucket
      included). Required -- raises ValueError if None.
    **kwargs: mode-specific extras, e.g. `norm` for the token-scale and
      token-sum-norm modes.

  Returns:
    The reduced scalar loss.
  """
  if num_segments is None:
    raise ValueError(
        "num_segments must be provided when segment_ids is not None."
    )

  masked = per_token_loss * completion_mask
  l_seg = segmented_sum(masked, segment_ids, num_segments)
  c_seg = segmented_count(segment_ids, num_segments, mask=completion_mask)
  a_seg = (c_seg > 0).astype(jnp.float32)
  a_seg = a_seg.at[:, 0].set(0.0)
  n_act = a_seg.sum()

  if loss_agg_mode == "token-mean":
    return masked.sum() / jnp.clip(completion_mask.sum(), min=1.0)
  elif loss_agg_mode == "sequence-mean-token-mean":
    per_seg_mean = l_seg / jnp.clip(c_seg, min=1.0)
    return (per_seg_mean * a_seg).sum() / jnp.clip(n_act, min=1.0)
  elif loss_agg_mode == "sequence-mean-token-scale":
    if "norm" not in kwargs:
      raise ValueError(
          "sequence-mean-token-scale under sequence packing requires an"
          " explicit 'norm' (e.g. max_response_length); the default row-width"
          " norm is the pack budget, which dilutes every sequence."
      )
    norm = _check_get_norm(kwargs, per_token_loss.shape[-1])
    per_seg_scaled = l_seg / jnp.clip(norm, min=1e-6)
    return (per_seg_scaled * a_seg).sum() / jnp.clip(n_act, min=1.0)
  elif loss_agg_mode == "seq-mean-token-sum":
    return (l_seg * a_seg).sum() / jnp.clip(n_act, min=1e-6)
  elif loss_agg_mode == "sequence-mean-token-sum-norm":
    norm = _check_get_norm(kwargs, n_act)
    return masked.sum() / jnp.clip(norm, min=1e-6)
  else:
    raise ValueError(
        f"Unsupported loss aggregation mode: {loss_agg_mode}. Supported modes:"
        " 'token-mean', 'sequence-mean-token-mean',"
        " 'sequence-mean-token-scale', 'seq-mean-token-sum',"
        " 'sequence-mean-token-sum-norm'."
    )


def _metric_scalar(value: Any) -> Any:
  """A WeightedMetric reduced to its scalar via compute(); scalars pass through.

  Metric aux values may be `WeightedMetric` (deferred division) or plain
  scalars. This normalizes either to a scalar for reduction / logging.
  """
  return value.compute() if isinstance(value, utils.WeightedMetric) else value


def mean_of_means(values: Iterable[Any]) -> np.ndarray:
  """Reducer: mean of per-micro-batch values (a WeightedMetric-safe np.mean).

  Each value is reduced to a scalar first (WeightedMetric via compute(), i.e.
  divided per micro-batch), then averaged. For plain scalars this is exactly
  np.mean, so it is a safe drop-in replacement.
  """
  return np.mean([np.asarray(_metric_scalar(v)) for v in values])


def global_weighted_mean(values: Iterable[utils.WeightedMetric]) -> float:
  """Reducer: global sum(S)/sum(d) across per-micro-batch WeightedMetric values.

  Sums numerators and denominators before dividing (token-weighted / global),
  as opposed to `mean_of_means` which averages per-micro-batch means. Equal to
  `mean_of_means` when the denominator is constant across micro-batches; they
  diverge otherwise (e.g. sequence packing).
  """
  total_sum = sum(float(v.unreduced_sum) for v in values)
  total_denom = sum(float(v.denominator) for v in values)
  return total_sum / total_denom if total_denom != 0 else 0.0


def compute_entropy_from_logits(logits: jax.Array) -> jax.Array:
  """Computes the entropy of a distribution given its logits.

  Args:
    logits: Logits as returned by the model. Of shape `[batch_size, seq_len,
      emb_dim]`.

  Returns:
    A JAX array of shape `[batch_size, seq_len]`, containing the entropy values.
  """
  logits_f32 = jnp.where(jnp.isfinite(logits), logits.astype(jnp.float32), 0.0)
  log_probs = jax.nn.log_softmax(logits_f32, axis=-1)
  safe_log_probs = jnp.where(jnp.isfinite(log_probs), log_probs, 0.0)
  probs = jnp.where(jnp.isfinite(log_probs), jnp.exp(safe_log_probs), 0.0)
  return -jnp.sum(probs * safe_log_probs, axis=-1)


def _check_get_norm(arguments: dict[str, Any], default: Any) -> Any:
  """Get custom normalization factor from kwargs with a default value.

  Args:
      arguments: The arguments dictionary.
      default: The default value to use if no 'norm' key is found.

  Returns:
      The normalization factor.

  Raises:
      ValueError: If the 'norm' key is present but has an invalid value or type.
  """
  if "norm" in arguments:
    try:
      norm = float(arguments["norm"])
    except (ValueError, TypeError):
      raise ValueError(
          f"Invalid 'norm' value: {arguments['norm']}. Must be a positive"
          " number."
      )
    if norm <= 0:
      raise ValueError(
          f"Invalid 'norm' value: {norm}. Must be a positive number."
      )
    return norm
  return default
