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

"""Simple utils used by RL algorithms."""

import dataclasses
from itertools import chain  # pylint: disable=g-importing-member
import operator
from typing import Any, Iterator, Mapping, Optional, Sequence, TypeVar

from absl import logging
from flax import nnx
from flax.nnx import filterlib
from flax.nnx import statelib
import jax
from jax import tree_util
import jax.numpy as jnp
import jaxtyping
import numpy as np
from tunix.rl import common

ExampleT = TypeVar("ExampleT")
Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding

_OPTIONAL_PER_TOKEN_KEYS = (
    "ref_per_token_logps",
    "old_per_token_logps",
    "returns",
    "old_values",
    "sampler_is_weights",
)


def is_positive_integer(value: int | None, name: str):
  """Checks if the value is a positive integer.

  Accepts Python ``int`` and numpy integer scalars (e.g. ``np.int64``).
  Explicitly rejects ``bool``, which is a subclass of ``int`` in Python but
  is not semantically an integer in this context.
  """
  if value is None:
    return
  if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
    raise ValueError(f"{name} must be a positive integer. Got: {value}")
  if value <= 0:
    raise ValueError(f"{name} must be a positive integer. Got: {value}")


def check_divisibility(
    small_size,
    big_size,
    small_size_name,
    big_size_name,
):
  """Checks if big_size is a multiple of small_size."""
  if big_size % small_size != 0:
    raise ValueError(
        f"{big_size_name} must be a multiple of {small_size_name}."
    )


def to_flat_dict(
    tree: jaxtyping.PyTree | statelib.State,
) -> tuple[dict[tuple[str, ...], jaxtyping.Array], jaxtyping.PyTreeDef]:
  if isinstance(tree, statelib.State):
    tree = nnx.to_pure_dict(tree)
  flattened, tree_def = jax.tree.flatten_with_path(tree)
  return {tuple(k.key for k in keys): v for keys, v in flattened}, tree_def


def get_pytree_mesh_info(tree: jaxtyping.PyTree) -> Mesh | None:
  """Returns the mesh info for the pytree."""
  mesh_info = set()

  def _get_mesh_info(leaf: jaxtyping.PyTree):
    if isinstance(leaf, jax.Array):
      if hasattr(leaf, "sharding") and leaf.sharding:
        sharding = leaf.sharding
        if isinstance(sharding, NamedSharding):
          mesh_info.add(sharding.mesh)
    return leaf

  jax.tree_util.tree_map(_get_mesh_info, tree)
  if len(mesh_info) > 1:
    raise ValueError(
        f"All leaves of the pytree must have the same mesh. Found: {mesh_info}"
    )
  return mesh_info.pop() if mesh_info else None


def _is_same_state(s1: jaxtyping.PyTree, s2: jaxtyping.PyTree) -> bool:
  """Returns whether two states refer to the same Params."""
  return np.all(  # pyrefly: ignore[bad-return]
      jax.tree.map(
          lambda x, y: x is y,
          jax.tree_util.tree_leaves(s1),
          jax.tree_util.tree_leaves(s2),
      )
  )


def is_sharing_weights(
    m1: Optional[nnx.Module],
    m2: Optional[nnx.Module],
) -> bool:
  """Returns whether two models are sharing same copy of weights."""
  if m1 is None or m2 is None:
    return False

  s1 = nnx.state(m1)
  s2 = nnx.state(m2)
  return _is_same_state(s1, s2)


def is_sharing_backbone(
    m1: nnx.Module,
    m2: nnx.Module,
) -> bool:
  """Returns whether two models are sharing same copy of backbone."""
  s1 = nnx.state(m1, filterlib.Not(nnx.LoRAParam))
  s2 = nnx.state(m2, filterlib.Not(nnx.LoRAParam))
  return _is_same_state(s1, s2)


def chunk_slices_by_size(stop: int, step: int):
  """Yields slices `slice(...)` for samples before `stop`, chunked by `step`.

  The last chunk is allowed to be smaller than `step`.

  Args:
    stop: The total number of samples.
    step: The maximum size of each chunk.
  """
  i = 0
  while i < stop:
    yield slice(i, min(i + step, stop))
    i += step


def get_batch_slice(tree: Any, batch_slice: slice) -> Any:
  """Slices array-like leaves of a PyTree along the first dimension.

  Args:
    tree: The PyTree to slice.
    batch_slice: The slice to apply.

  Returns:
    A PyTree with sliced leaves.
  """

  def apply_slice(x: Any) -> Any:
    if x is None:
      return None
    # Apply slice if the leaf is an array with at least one dimension.
    if hasattr(x, "ndim") and hasattr(x, "shape") and x.ndim >= 1:
      return x[batch_slice]
    else:
      return x

  return jax.tree_util.tree_map(
      apply_slice, tree, is_leaf=lambda node: node is None
  )


def merge_micro_batches(batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
  """Merges micro-batch dictionaries into a single batch.

  Concatenates values from a list of micro-batch dicts. Values are concatenated
  along the batch dimension.

  Args:
    batches: List of micro-batch dictionaries.

  Returns:
    A dictionary with merged batch data.
  """
  if not batches:
    return {}

  merged = {}
  first_batch, *_ = batches
  for key in first_batch.keys():
    all_values = [item[key] for item in batches]
    first_value, *_ = all_values

    if isinstance(first_value, list):
      merged[key] = list(chain.from_iterable(all_values))
    else:
      merged[key] = tree_util.tree_map(
          lambda *xs: np.concatenate([np.atleast_1d(x) for x in xs]),
          *all_values,
      )

  return merged


def put_params_on_memory_kind(
    params: jaxtyping.PyTree,
    memory_kind: str,
) -> jaxtyping.PyTree:
  """Puts params on the given memory kind."""
  if memory_kind not in ["device", "pinned_host", "unpinned_host"]:
    raise ValueError(
        "memory_kind must be one of device, pinned_host, or "
        f"unpinned_host. Received: {memory_kind}."
    )
  if not jax.tree_util.tree_leaves(params):
    logging.debug(
        "put_params_on_memory_kind received an empty parameter tree. "
        "Skipping device transfer."
    )
    return params
  original_shardings = jax.tree.map(lambda x: x.sharding, params)
  logging.debug("original_shardings: %s", original_shardings)
  is_on_device = jax.tree_util.tree_reduce(
      operator.or_,
      jax.tree.map(lambda x: x.memory_kind == "device", original_shardings),
  )
  if (is_on_device and memory_kind == "device") or (
      not is_on_device and memory_kind == "pinned_host"
  ):
    logging.info(
        "Params are already on the requested memory kind: %s", memory_kind
    )
    return params

  def _get_new_sharding(x):
    if isinstance(x, jax.NamedSharding):
      return jax.NamedSharding(x.mesh, x.spec, memory_kind=memory_kind)
    else:
      return x.with_memory_kind(memory_kind)

  new_shardings = jax.tree.map(_get_new_sharding, original_shardings)
  params_on_memory_kind = jax.device_put(
      params,
      new_shardings,
  )
  shardings = jax.tree.map(lambda x: x.sharding, params_on_memory_kind)
  logging.debug("params_on_memory_kind shardings: %s", shardings)
  return params_on_memory_kind

def create_critic_model(
    actor_model: nnx.Module, seed: int = 0, rngs: nnx.Rngs = None, lm_head_to_replace: str = "lm_head"  # pyrefly: ignore[bad-function-definition]
) -> nnx.Module:
  """Creates a critic model from an actor model."""

  if rngs is None:
    rngs=nnx.Rngs(seed)

  g, state = nnx.split(actor_model)
  # TODO(tsbao): if actor model is a LoRA model, then we can potentially share
  # backbone of base weights with critic model. Do it later as an optimization.
  copied_state = jax.tree.map(jnp.copy, state)
  critic_model = nnx.merge(g, copied_state)
  lm_head = getattr(critic_model, lm_head_to_replace)
  hidden_dim = (
      lm_head.shape[0] if hasattr(lm_head, "shape") else lm_head.in_features
  )
  new_head = nnx.Linear(
          in_features=hidden_dim,
          out_features=1,
          use_bias=False,
          rngs=rngs,
      )

  # If Qwix is active for the model, also assign qwix_path for the new head
  if hasattr(critic_model, "qwix_path"):
    new_head.qwix_path = getattr(lm_head, "qwix_path", (lm_head_to_replace,))  # pyrefly: ignore[missing-attribute]
  setattr(
      critic_model,
      lm_head_to_replace,
      new_head
  )

  return critic_model


class TransformerWithScoreHead(nnx.Module):
  def __init__(self, transformer: nnx.Module, rngs: nnx.Rngs):
    """Initializes the transformer with a score head.

    Args:
      transformer: The transformer backbone.
      rngs: The random number generator.
    """
    if hasattr(transformer, 'embed_dim'):
      embed_dim = transformer.embed_dim
    elif hasattr(transformer.config, 'embed_dim'):  # pyrefly: ignore[missing-attribute]
      embed_dim = transformer.config.embed_dim  # pyrefly: ignore[missing-attribute]
    else:
      raise ValueError("Could not determine embed dim for the transformer.")

    self.transformer = transformer
    self.score = nnx.Linear(
        in_features=embed_dim,
        out_features=1,
        use_bias=False,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.normal(),
            transformer.config.shd_config.score_weight_d1,  # pyrefly: ignore[missing-attribute]
        ),
        rngs=rngs,
    )

  def __call__(self, *args, **kwargs):
    self.transformer(*args, **kwargs, output_hidden_states=True)
    hidden_states = nnx.pop(self.transformer, nnx.Intermediate)[
        'all_hidden_states'
    ].value[-1]
    score = self.score(hidden_states)
    return score


def get_partition_spec(
    sharding: jax.sharding.Sharding,
) -> jax.sharding.PartitionSpec:
  """Returns the partition spec for the given sharding."""
  if isinstance(sharding, jax.sharding.NamedSharding):
    return sharding.spec
  else:
    return jax.sharding.PartitionSpec()


def unpad_train_example(example: Any) -> list[dict[str, Any]]:
  """Unpads a TrainExample or RLTrainerPayload into a list of item dictionaries."""
  # TODO(noghabi): Skip padding and unpadding directly in the learner.
  res = []

  p_ids_val = getattr(example, "prompt_ids", None)
  p_ids = (
      np.asarray(p_ids_val)
      if p_ids_val is not None
      else np.zeros((1, 0), dtype=np.int32)
  )
  p_mask_val = getattr(example, "prompt_mask", None)
  p_mask = (
      np.asarray(p_mask_val)
      if p_mask_val is not None
      else np.ones_like(p_ids, dtype=np.float32)
  )

  c_ids_val = getattr(example, "completion_ids", None)
  c_ids = (
      np.asarray(c_ids_val)
      if c_ids_val is not None
      else np.zeros((1, 0), dtype=np.int32)
  )
  c_mask_val = getattr(example, "completion_mask", None)
  if c_mask_val is None:
    c_mask_val = getattr(example, "action_mask", None)
  if c_mask_val is None:
    c_mask_val = getattr(example, "loss_mask", None)
  c_mask = (
      np.asarray(c_mask_val)
      if c_mask_val is not None
      else np.ones_like(c_ids, dtype=np.float32)
  )

  is_1d = (
      (p_ids.ndim == 1 and p_ids.size > 0)
      or (c_ids.ndim == 1 and c_ids.size > 0)
      or (p_mask.ndim == 1 and p_mask.size > 0)
      or (c_mask.ndim == 1 and c_mask.size > 0)
  )

  if p_ids.ndim == 1:
    p_ids = p_ids[np.newaxis, :]
  elif p_ids.ndim == 0:
    p_ids = p_ids.reshape(1, -1)

  if p_mask.ndim == 1:
    p_mask = p_mask[np.newaxis, :]
  elif p_mask.ndim == 0:
    p_mask = p_mask.reshape(1, -1)

  if c_ids.ndim == 1:
    c_ids = c_ids[np.newaxis, :]
  elif c_ids.ndim == 0:
    c_ids = c_ids.reshape(1, -1)

  if c_mask.ndim == 1:
    c_mask = c_mask[np.newaxis, :]
  elif c_mask.ndim == 0:
    c_mask = c_mask.reshape(1, -1)

  batch_size = max(p_ids.shape[0], c_ids.shape[0], 1)

  adv_val = getattr(example, "advantages", None)
  if adv_val is None:
    adv = np.zeros((batch_size, c_ids.shape[1]), dtype=np.float32)
  else:
    adv = np.asarray(adv_val)
    if adv.ndim == 0:
      adv = adv.reshape(1)
    elif adv.ndim == 1 and is_1d and adv.size > 1:
      adv = adv[np.newaxis, :]

  ref_logps = None
  has_ref = getattr(example, "ref_per_token_logps", None) is not None
  if has_ref:
    ref_logps = np.asarray(example.ref_per_token_logps)
    if ref_logps.ndim == 1:
      ref_logps = ref_logps[np.newaxis, :]

  old_logps = None
  has_old = getattr(example, "old_per_token_logps", None) is not None
  if has_old:
    old_logps = np.asarray(example.old_per_token_logps)
    if old_logps.ndim == 1:
      old_logps = old_logps[np.newaxis, :]

  returns_np = None
  returns_val = getattr(example, "returns", None)
  has_returns = returns_val is not None
  if has_returns:
    returns_np = np.asarray(returns_val)
    if returns_np.ndim == 1:
      returns_np = returns_np[np.newaxis, :]

  old_values_np = None
  old_values_val = getattr(example, "old_values", None)
  has_old_values = old_values_val is not None
  if has_old_values:
    old_values_np = np.asarray(old_values_val)
    if old_values_np.ndim == 1:
      old_values_np = old_values_np[np.newaxis, :]

  sampler_is_weights_np = None
  sampler_is_weights_val = getattr(example, "sampler_is_weights", None)
  has_sampler_is_weights = sampler_is_weights_val is not None
  if has_sampler_is_weights:
    sampler_is_weights_np = np.asarray(sampler_is_weights_val)
    if sampler_is_weights_np.ndim == 1:
      sampler_is_weights_np = sampler_is_weights_np[np.newaxis, :]

  policy_version_np = None
  policy_version_val = getattr(example, "policy_version", None)
  has_policy_version = policy_version_val is not None
  if has_policy_version:
    policy_version_np = np.asarray(policy_version_val)
    if policy_version_np.ndim == 0:
      policy_version_np = policy_version_np.reshape(1)


  for i in range(batch_size):
    p_len = int(np.sum(p_mask[i])) if i < p_mask.shape[0] else 0
    if is_1d or getattr(example, "is_unpadded", False):
      c_len = c_ids[i].shape[0] if i < c_ids.shape[0] else 0
    else:
      c_len = int(np.sum(c_mask[i])) if i < c_mask.shape[0] else 0

    p_ids_i = p_ids[i] if i < p_ids.shape[0] else np.zeros(0, dtype=np.int32)
    p_mask_i = p_mask[i] if i < p_mask.shape[0] else np.zeros(0, dtype=np.float32)
    c_ids_i = c_ids[i] if i < c_ids.shape[0] else np.zeros(0, dtype=np.int32)
    c_mask_i = c_mask[i] if i < c_mask.shape[0] else np.zeros(0, dtype=np.float32)

    prompt_seq_len = p_ids_i.shape[0]
    completion_seq_len = c_ids_i.shape[0]

    if adv.ndim == 2:
      adv_is_per_token = True
      adv_row = adv[i] if i < adv.shape[0] else np.zeros(0, dtype=np.float32)
      if adv_row.shape[0] == prompt_seq_len + completion_seq_len:
        adv_i = adv_row[prompt_seq_len : prompt_seq_len + c_len]
      elif adv_row.shape[0] >= c_len:
        adv_i = adv_row[:c_len]
      else:
        adv_i = np.pad(
            adv_row,
            (0, max(0, c_len - adv_row.shape[0])),
            constant_values=0.0,
        )
    elif adv.ndim == 1 and adv.shape[0] == batch_size and not is_1d:
      adv_is_per_token = False
      adv_i = adv[i]
    else:
      adv_is_per_token = False
      adv_i = float(adv.reshape(-1)[0]) if adv.size > 0 else 0.0

    def _slice_feat(arr_2d: np.ndarray | None) -> np.ndarray | None:
      if arr_2d is None:
        return None
      row = arr_2d[i] if i < arr_2d.shape[0] else np.zeros(0, dtype=np.float32)
      if row.shape[0] == prompt_seq_len + completion_seq_len:
        return row[prompt_seq_len : prompt_seq_len + c_len]
      elif row.shape[0] >= c_len:
        return row[:c_len]
      else:
        return np.pad(
            row, (0, max(0, c_len - row.shape[0])), constant_values=0.0
        )

    policy_version_slice = (
        policy_version_np[i : i + 1] if policy_version_np is not None else None
    )

    # `policy_version` is per-row: row `i` of the input maps to scalar
    # `policy_version_np[i]`. We slice with `i:i+1` to keep a 1-D shape so that
    # `pack_sequences` can stack scalars from multiple items unambiguously.
    item = {
        "prompt_ids": p_ids_i[-p_len:] if p_len > 0 else p_ids_i[:0],
        "prompt_mask": p_mask_i[-p_len:] if p_len > 0 else p_mask_i[:0],
        "completion_ids": c_ids_i[:c_len],
        "completion_mask": c_mask_i[:c_len],
        "advantages": adv_i,
        "adv_is_per_token": adv_is_per_token,
        "ref_per_token_logps": _slice_feat(ref_logps),
        "old_per_token_logps": _slice_feat(old_logps),
        "returns": _slice_feat(returns_np),
        "old_values": _slice_feat(old_values_np),
        "sampler_is_weights": _slice_feat(sampler_is_weights_np),
        "policy_version": policy_version_slice,
    }
    res.append(item)
  return res



def compute_pack_size(mesh: jax.sharding.Mesh) -> int:
  """Packed rows per batch = product of the "fsdp"/"dp" mesh axes (1 if neither)."""
  if "fsdp" not in mesh.shape and "dp" not in mesh.shape:
    logging.warning(
        "Sequence packing: mesh has no 'fsdp'/'dp' axis; pack_size=1."
        " Axes: %s.",
        dict(mesh.shape),
    )
  return mesh.shape.get("fsdp", 1) * mesh.shape.get("dp", 1)


def validate_packing_budget(
    max_token_budget: int,
    max_prompt_length: int,
    max_response_length: int,
) -> None:
  """Fails at learner init if a maximal sequence cannot fit one packed row.

  Sequences are truncated to `max_prompt_length + max_response_length`, so that
  sum is the longest row `pack_sequences` can ever be handed. A smaller budget
  is not a degraded mode -- it raises mid-run the first time a maximal sequence
  shows up, which can be hours in. Check it up front instead.

  Args:
    max_token_budget: The configured `max_seq_token_per_tpu`.
    max_prompt_length: Rollout's prompt cap.
    max_response_length: Rollout's generation cap.

  Raises:
    ValueError: if the budget cannot hold one maximal sequence.
  """
  longest = max_prompt_length + max_response_length
  if max_token_budget < longest:
    raise ValueError(
        f"max_seq_token_per_tpu={max_token_budget} is smaller than the longest"
        f" possible sequence (max_prompt_length {max_prompt_length} +"
        f" max_response_length {max_response_length} = {longest}); packing"
        " would fail once a maximal sequence appears. Set"
        f" max_seq_token_per_tpu >= {longest}."
    )


def _ceildiv(a: int, b: int) -> int:
  return -(-a // b)


def _item_tokens(item: Mapping[str, Any]) -> int:
  return len(item["prompt_ids"]) + len(item["completion_ids"])


def _fill_one_chunk(
    items: Sequence[Mapping[str, Any]],
    pack_size: int,
    budget: int,
    max_segments: int,
) -> tuple[list[list[Mapping[str, Any]]], list[Mapping[str, Any]]]:
  """Fills ONE chunk of `pack_size` fixed-capacity bins, first-fit-decreasing.

  Sorts the items by token length descending and greedily places each into the
  first bin with room, where a bin has room only if it stays within both the
  token `budget` AND `max_segments` sequences (so the loss's static
  `num_segments = max_segments + 1` buckets never overflow). Items that fit no
  bin are returned as `leftover` (in their original order) for a later chunk.

  Returns (bins, leftover): `bins` is exactly `pack_size` lists (some may be
  empty); `leftover` are the items that did not fit.
  """
  bins: list[list[Mapping[str, Any]]] = [[] for _ in range(pack_size)]
  loads = [0] * pack_size
  leftover: list[Mapping[str, Any]] = []
  for item in sorted(items, key=_item_tokens, reverse=True):
    n = _item_tokens(item)
    for b in range(pack_size):
      if loads[b] + n <= budget and len(bins[b]) < max_segments:
        bins[b].append(item)
        loads[b] += n
        break
    else:
      leftover.append(item)
  return bins, leftover


def pack_sequences(
    item_iterator: Iterator[Sequence[ExampleT]],
    max_token_budget: int,
    sequences_per_update: int,
    pad_id: int = 0,
    pack_size: int = 1,
    max_segments_per_packed_row: int | None = None,
) -> Iterator[list[ExampleT]]:
  """FFD-packs sequences into [pack_size, max_token_budget] chunks, streaming.

  A chunk is emitted as soon as buffered sequences fill it (so training can
  overlap rollout); a mini-batch's last chunk has is_update_step=True.
  Colocated producers enqueue a whole mini-batch at once, so packing sees the
  full set (~global FFD); under streaming, chunk composition follows arrival
  order.

  Args:
    item_iterator: Stream of lists of TrainExamples (any granularity).
    max_token_budget: Max tokens per packed row (= max_seq_token_per_tpu).
    sequences_per_update: Sequences per mini-batch/update (= mini_batch_size *
      num_generations). Required: the update boundary is a mini-batch property,
      and the producer's list granularity does not carry it (a colocated
      producer enqueues a whole mini-batch, a streaming one a micro-batch), so
      it cannot be inferred from the input stream.
    pad_id: Padding vocabulary id.
    pack_size: Rows per chunk (= fsdp * dp); each chunk is [pack_size,
      max_token_budget].

  Yields:
    Single-element lists, each one [pack_size, max_token_budget] TrainExample.

  Raises:
    ValueError: empty mini-batch at an update boundary, a sequence longer than
      max_token_budget, a mid-mini-batch stream end, or a boundary inside an
      input example.
  """

  # Real segments per row are bounded by the token budget (each segment >= 1
  # token). `None` uses that safe bound so `num_segments = budget + 1` never
  # overflows; a smaller override shrinks the loss buckets and is enforced by
  # the raise in `_flush_pack`.
  effective_max_segments = (
      max_segments_per_packed_row
      if max_segments_per_packed_row is not None
      else max_token_budget
  )

  def _flush_pack(pack_items, example_cls, first_item) -> common.TrainExample:
    first_item = first_item or {}
    has_policy_version = first_item.get("policy_version") is not None
    kwargs = {}
    tracked_per_token_keys = []

    if first_item.get("ref_per_token_logps") is not None:
      tracked_per_token_keys.append("ref_per_token_logps")
    if first_item.get("old_per_token_logps") is not None:
      tracked_per_token_keys.append("old_per_token_logps")
    if first_item.get("returns") is not None:
      tracked_per_token_keys.append("returns")
    if first_item.get("old_values") is not None:
      tracked_per_token_keys.append("old_values")
    if first_item.get("sampler_is_weights") is not None:
      tracked_per_token_keys.append("sampler_is_weights")

    if not pack_items:
      p_ids_arr = jnp.zeros((1, 0), dtype=jnp.int32)
      p_mask_arr = jnp.zeros((1, 0), dtype=jnp.int32)
      c_ids_arr = jnp.full((1, max_token_budget), pad_id, dtype=jnp.int32)
      c_mask_arr = jnp.zeros((1, max_token_budget), dtype=jnp.int32)
      adv_arr = jnp.zeros((1, max_token_budget), dtype=jnp.float32)
      seg_arr = jnp.zeros((1, max_token_budget), dtype=jnp.int32)
      pos_arr = jnp.zeros((1, max_token_budget), dtype=jnp.int32)

      kwargs.update(
          prompt_ids=p_ids_arr,
          prompt_mask=p_mask_arr,
          completion_ids=c_ids_arr,
          completion_mask=c_mask_arr,
          advantages=adv_arr,
          ref_per_token_logps=None,
          old_per_token_logps=None,
          segment_ids=seg_arr,
          segment_positions=pos_arr,
      )
      for k in tracked_per_token_keys:
        kwargs[k] = jnp.zeros((1, max_token_budget), dtype=jnp.float32)
      if has_policy_version:
        kwargs["policy_version"] = first_item["policy_version"]

      fields = getattr(example_cls, "__dataclass_fields__", None)
      if fields is not None:
        if "loss_mask" in fields and "loss_mask" not in kwargs:
          kwargs["loss_mask"] = c_mask_arr
        if "action_mask" in fields and "action_mask" not in kwargs:
          kwargs["action_mask"] = c_mask_arr
        valid_kwargs = {k: v for k, v in kwargs.items() if k in fields}
        return example_cls(**valid_kwargs)
      return example_cls(**kwargs)  # pytype: disable=wrong-keyword-args

    # `len(pack_items)` is the real segment count of this row. It cannot exceed
    # the token budget (each segment >= 1 token), so the default bound never
    # trips; a too-small `max_segments_per_packed_row` override does, and we
    # fail loud rather than let `segment_sum` silently drop the overflow.
    if len(pack_items) > effective_max_segments:
      raise ValueError(
          f"pack_sequences: a packed row has {len(pack_items)} segments, "
          f"exceeding max_segments_per_packed_row={effective_max_segments}; "
          "increase it (or leave it None for the budget-derived safe default)."
      )

    current_tokens = sum(
        len(it["prompt_ids"]) + len(it["completion_ids"]) for it in pack_items
    )
    pad_len = max_token_budget - current_tokens

    packed_c_ids = []
    packed_c_mask = []
    packed_adv = []
    packed_segment_ids = []
    packed_positions = []

    per_token_feature_buffers = {k: [] for k in tracked_per_token_keys}

    for i, item in enumerate(pack_items, start=1):
      p_ids = item["prompt_ids"]
      c_ids = item["completion_ids"]
      seq_len = len(p_ids) + len(c_ids)

      packed_c_ids.extend([p_ids, c_ids])
      packed_c_mask.extend([np.zeros_like(p_ids), item["completion_mask"]])

      if item["adv_is_per_token"]:
        packed_adv.extend([
            np.zeros_like(p_ids, dtype=np.float32),
            item["advantages"],
        ])
      else:
        packed_adv.extend([
            np.zeros_like(p_ids, dtype=np.float32),
            np.full(len(c_ids), item["advantages"], dtype=np.float32),
        ])

      for k in tracked_per_token_keys:
        per_token_feature_buffers[k].extend([
            np.zeros_like(p_ids, dtype=np.float32),
            item[k],
        ])

      packed_segment_ids.append(np.full(seq_len, i, dtype=np.int32))
      packed_positions.append(np.arange(seq_len, dtype=np.int32))

    def _pad(arr_list, val, length):
      arr = np.concatenate(arr_list) if arr_list else np.array([])
      return np.pad(arr, (0, length), constant_values=val)

    p_ids_arr = jnp.zeros((1, 0), dtype=jnp.int32)
    p_mask_arr = jnp.zeros((1, 0), dtype=jnp.int32)

    c_ids_arr = jnp.array(_pad(packed_c_ids, pad_id, pad_len))[None, :]
    c_mask_arr = jnp.array(_pad(packed_c_mask, 0, pad_len))[None, :]
    adv_arr = jnp.array(_pad(packed_adv, 0.0, pad_len))[None, :]
    seg_arr = jnp.array(_pad(packed_segment_ids, 0, pad_len))[None, :]
    pos_arr = jnp.array(_pad(packed_positions, 0, pad_len))[None, :]

    per_token_features = {}
    for k in tracked_per_token_keys:
      per_token_features[k] = jnp.array(
          _pad(per_token_feature_buffers[k], 0.0, pad_len)
      )[None, :]

    kwargs = dict(
        prompt_ids=p_ids_arr,
        prompt_mask=p_mask_arr,
        completion_ids=c_ids_arr,
        completion_mask=c_mask_arr,
        advantages=adv_arr,
        ref_per_token_logps=per_token_features.get("ref_per_token_logps"),
        old_per_token_logps=per_token_features.get("old_per_token_logps"),
        segment_ids=seg_arr,
        segment_positions=pos_arr,
    )
    for k in tracked_per_token_keys:
      if k not in kwargs:
        kwargs[k] = per_token_features[k]

    if has_policy_version:
      kwargs["policy_version"] = pack_items[0]["policy_version"]

    fields = getattr(example_cls, "__dataclass_fields__", None)
    if fields is not None:
      if "loss_mask" in fields and "loss_mask" not in kwargs:
        kwargs["loss_mask"] = c_mask_arr
      if "action_mask" in fields and "action_mask" not in kwargs:
        kwargs["action_mask"] = c_mask_arr
      valid_kwargs = {k: v for k, v in kwargs.items() if k in fields}
      return example_cls(**valid_kwargs)

    return example_cls(**kwargs)  # pytype: disable=wrong-keyword-args

  chunk_capacity = pack_size * max_token_budget

  def _emit(chunk):
    """Merges one chunk (pack_size bins) into a [pack_size, budget] example."""
    chunk_examples = [
        _flush_pack(bin_items, example_cls, first_item_for_dummy)
        for bin_items in chunk
    ]
    if dataclasses.is_dataclass(chunk_examples[0]) and not hasattr(
        type(chunk_examples[0]), "_flax_dataclass"
    ):
      merged_fields = {}
      for f in dataclasses.fields(chunk_examples[0]):
        vals = [getattr(ex, f.name) for ex in chunk_examples]
        if all(v is None for v in vals):
          merged_fields[f.name] = None
        elif any(v is None for v in vals):
          merged_fields[f.name] = None
        elif isinstance(vals[0], dict):
          merged_fields[f.name] = vals[0]
        else:
          try:
            merged_fields[f.name] = jnp.concatenate(vals, axis=0)
          except Exception:
            merged_fields[f.name] = vals[0]
      return example_cls(**merged_fields)

    return jax.tree.map(
        lambda first_x, *rest_xs: None
        if first_x is None
        else jnp.concatenate((first_x, *rest_xs), axis=0),
        *chunk_examples,
    )

  def _mark(merged, is_update):
    # `num_segments = effective_max_segments + 1` (+1 = padding bucket) is a
    # static upper bound, fixed every step so the segment-aware loss compiles
    # once. Set here (not per bin) so every emitted chunk carries it.
    kwargs = dict(is_update_step=jnp.array([is_update], dtype=jnp.bool_))
    if hasattr(merged, "num_segments"):
      kwargs["num_segments"] = effective_max_segments + 1  # pyrefly: ignore[bad-assignment]
    if hasattr(merged, "replace"):
      return [merged.replace(**kwargs)]
    elif dataclasses.is_dataclass(merged):
      dc_kwargs = {}
      if hasattr(merged, "is_update_step"):
        dc_kwargs["is_update_step"] = kwargs["is_update_step"]
      if hasattr(merged, "num_segments"):
        dc_kwargs["num_segments"] = kwargs["num_segments"]
      if dc_kwargs:
        return [dataclasses.replace(merged, **dc_kwargs)]
      return [merged]
    return [merged]

  # See the docstring: buffer sequences, emit a chunk once it holds a chunk's
  # worth of tokens, and mark the mini-batch's last chunk as the update.
  buffered = []  # unpacked sequences
  received = 0  # sequences received this mini-batch (incl. emitted)
  tokens_in_mini = 0  # for the dummy_ratio log
  chunks_in_mini = 0
  example_cls = common.TrainExample
  first_item_for_dummy = None

  def _final_flush():
    nonlocal buffered, received, tokens_in_mini, chunks_in_mini
    if not buffered and chunks_in_mini == 0:
      raise ValueError(
          "pack_sequences reached an update boundary with an empty mini-batch;"
          " no packed example would be produced, dropping a gradient update."
      )
    while buffered:
      chunk, buffered = _fill_one_chunk(
          buffered, pack_size, max_token_budget, effective_max_segments
      )
      chunks_in_mini += 1
      yield _mark(_emit(chunk), not buffered)  # last chunk (empty leftover).
    total_cap = chunks_in_mini * chunk_capacity
    logging.info(
        "pack_sequences: %d seqs -> %d chunks, dummy_ratio=%.3f",
        received,
        chunks_in_mini,
        1.0 - tokens_in_mini / total_cap if total_cap else 0.0,
    )
    received = 0
    tokens_in_mini = 0
    chunks_in_mini = 0

  for item_list in item_iterator:
    for example in item_list:
      example_cls = type(example)
      for item in unpad_train_example(example):
        n = _item_tokens(item)
        if n > max_token_budget:
          raise ValueError(
              f"pack_sequences: a single sequence has {n} tokens, exceeding"
              f" max_token_budget {max_token_budget}; increase the budget."
          )
        if first_item_for_dummy is None:
          first_item_for_dummy = item
        buffered.append(item)
        received += 1
        tokens_in_mini += n

    if received > sequences_per_update:
      raise ValueError(
          "pack_sequences: mini-batch boundary falls inside an input example"
          f" (received {received} sequences, expected {sequences_per_update}"
          " per update)."
      )
    if received == sequences_per_update:
      yield from _final_flush()
    else:
      # Not the boundary yet: emit whole chunks eagerly, keep the remainder.
      while sum(_item_tokens(it) for it in buffered) >= chunk_capacity:
        chunk, buffered = _fill_one_chunk(
            buffered, pack_size, max_token_budget, effective_max_segments
        )
        chunks_in_mini += 1
        yield _mark(_emit(chunk), False)

  if buffered or received:
    raise ValueError("pack_sequences stream ended mid-mini-batch.")


VERIFY_UPDATE_PARAMS_KEY = "VERIFY_UPDATE_PARAMS_SRC_TO_TGT_MODULE_NAME"
