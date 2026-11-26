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

from itertools import chain  # pylint: disable=g-importing-member
import math
import operator
import time
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence
import zlib

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
from tunix.rl import packing

Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding

_OPTIONAL_PER_TOKEN_KEYS = (
    "ref_per_token_logps",
    "old_per_token_logps",
    "returns",
    "old_values",
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
    rngs = nnx.Rngs(seed)

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
  setattr(critic_model, lm_head_to_replace, new_head)

  return critic_model


class TransformerWithScoreHead(nnx.Module):

  def __init__(self, transformer: nnx.Module, rngs: nnx.Rngs):
    """Initializes the transformer with a score head.

    Args:
      transformer: The transformer backbone.
      rngs: The random number generator.
    """
    if hasattr(transformer, "embed_dim"):
      embed_dim = transformer.embed_dim
    elif hasattr(transformer.config, "embed_dim"):  # pyrefly: ignore[missing-attribute]
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
        "all_hidden_states"
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


def unpad_train_example(example: common.TrainExample) -> list[dict[str, Any]]:
  """Unpads a TrainExample into a list of dictionaries with numpy arrays."""
  # TODO(noghabi): Skip padding and unpadding directly in the learner.
  res = []
  batch_size = example.prompt_ids.shape[0]

  p_ids = np.asarray(example.prompt_ids)
  p_mask = np.asarray(example.prompt_mask)
  c_ids = np.asarray(example.completion_ids)
  c_mask = np.asarray(example.completion_mask)
  # With exact token continuity the valid completion length is explicit (it
  # may include tokens equal to the pad id), otherwise it is the loss-mask span.
  completion_valid = getattr(example, "completion_attention_mask", None)
  if isinstance(completion_valid, (jax.Array, np.ndarray, list, tuple)):
    completion_valid = np.asarray(completion_valid)
  else:
    completion_valid = None
  adv = np.asarray(example.advantages)
  adv_is_per_token = adv.ndim == 2

  has_ref = example.ref_per_token_logps is not None
  if has_ref:
    ref_logps = np.asarray(example.ref_per_token_logps)
  has_old = example.old_per_token_logps is not None
  if has_old:
    old_logps = np.asarray(example.old_per_token_logps)

  returns_val = getattr(example, "returns", None)
  has_returns = returns_val is not None
  if has_returns:
    returns_np = np.asarray(returns_val)

  old_values_val = getattr(example, "old_values", None)
  has_old_values = old_values_val is not None
  if has_old_values:
    old_values_np = np.asarray(old_values_val)

  policy_version_val = getattr(example, "policy_version", None)
  has_policy_version = policy_version_val is not None
  if has_policy_version:
    policy_version_np = np.asarray(policy_version_val)

  for i in range(batch_size):
    p_len = int(np.sum(p_mask[i]))
    c_len = int(
        np.sum(c_mask[i] if completion_valid is None else completion_valid[i])
    )

    # `policy_version` is per-row: row `i` of the input maps to scalar
    # `policy_version_np[i]`. We slice with `i:i+1` to keep a 1-D shape so that
    # `pack_sequences` can stack scalars from multiple items unambiguously.
    item = {
        "prompt_ids": p_ids[i, -p_len:] if p_len > 0 else p_ids[i, :0],
        "prompt_mask": p_mask[i, -p_len:] if p_len > 0 else p_mask[i, :0],
        "completion_ids": c_ids[i, :c_len],
        "completion_mask": c_mask[i, :c_len],
        "advantages": adv[i, :c_len] if adv_is_per_token else adv[i],
        "adv_is_per_token": adv_is_per_token,
        "ref_per_token_logps": ref_logps[i, :c_len] if has_ref else None,  # pyrefly: ignore[unbound-name]
        "old_per_token_logps": old_logps[i, :c_len] if has_old else None,  # pyrefly: ignore[unbound-name]
        "returns": returns_np[i, :c_len] if has_returns else None,  # pyrefly: ignore[unbound-name]
        "old_values": old_values_np[i, :c_len] if has_old_values else None,  # pyrefly: ignore[unbound-name]
        "policy_version": (
            policy_version_np[i : i + 1] if has_policy_version else None  # pyrefly: ignore[unbound-name]
        ),
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


def _ceildiv(a: int, b: int) -> int:
  return -(-a // b)


def train_example_to_pack_items(
    example: common.TrainExample,
) -> list[packing.PackItem]:
  """Converts a TrainExample to a list of PackItems."""
  items = [
      packing.PackItem(
          prompt_ids=np.asarray(item["prompt_ids"], dtype=np.int32),
          completion_ids=np.asarray(item["completion_ids"], dtype=np.int32),
          completion_mask=np.asarray(item["completion_mask"], dtype=np.float32),
          advantages=(
              np.asarray(item["advantages"], dtype=np.float32)
              if item["adv_is_per_token"]
              else np.full(
                  len(item["completion_ids"]),
                  float(np.asarray(item["advantages"]).reshape(-1)[0]),
                  dtype=np.float32,
              )
          ),
          per_token={
              k: np.asarray(item[k], dtype=np.float32)
              for k in packing.PER_TOKEN_FIELDS
              if item.get(k) is not None
          },
          policy_version=(
              np.asarray(item["policy_version"])
              if item.get("policy_version") is not None
              else None
          ),
      )
      for item in unpad_train_example(example)
  ]
  return items


def pack_rows_to_train_examples(
    chunk: packing.PackedChunk,
    example_cls: type[Any],
    *,
    mask_dtype: Any,
    num_segments: int,
    is_update_step: bool,
) -> Any:
  """Converts a PackedChunk to a TrainExample."""
  n = len(chunk)
  kwargs: dict[str, Any] = dict(
      prompt_ids=jnp.zeros((n, 0), dtype=np.int32),
      prompt_mask=jnp.zeros((n, 0), dtype=mask_dtype),
      completion_ids=jnp.asarray(chunk.ids),
      completion_mask=jnp.asarray(chunk.completion_mask, dtype=mask_dtype),
      advantages=jnp.asarray(chunk.advantages),
      segment_ids=jnp.asarray(chunk.segment_ids),
      segment_positions=jnp.asarray(chunk.segment_positions),
      ref_per_token_logps=None,
      old_per_token_logps=None,
  )
  for name, val in chunk.per_token.items():
    kwargs[name] = jnp.asarray(val)
  versions = chunk.policy_versions
  if any(v is not None for v in versions):
    fallback = next(v for v in versions if v is not None)
    kwargs["policy_version"] = jnp.concatenate([
        jnp.asarray(v if v is not None else fallback).reshape(-1)
        for v in versions
    ])
  example = example_cls(**kwargs)
  replacements: dict[str, Any] = {
      "is_update_step": jnp.array([is_update_step], dtype=jnp.bool_)
  }
  if hasattr(example, "num_segments"):
    replacements["num_segments"] = num_segments
  return example.replace(**replacements)


def pack_sequences(
    item_iterator: Iterator[Sequence[common.TrainExample]],
    max_token_budget: int,
    sequences_per_update: int,
    pad_id: int = 0,
    pack_size: int = 1,
    max_segments_per_packed_row: int | None = None,
) -> Iterator[list[common.TrainExample]]:
  """FFD-packs sequences into [pack_size, max_token_budget] chunks, streaming.

  A chunk is emitted as soon as buffered sequences fill it (so training can
  overlap rollout); a mini-batch's last chunk has is_update_step=True.
  Colocated producers enqueue a whole mini-batch at once, so packing sees the
  full set (~global FFD); under streaming, chunk composition follows arrival
  order.

  Args:
    item_iterator: Stream of lists of TrainExamples (any granularity).
    max_token_budget: Max tokens per packed row (= max_seq_tokens_per_device).
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
  max_segments = packing.effective_max_segments(
      max_token_budget, max_segments_per_packed_row
  )

  num_segments = max_segments + 1
  # See the docstring: buffer sequences, emit a chunk once it holds a chunk's
  # worth of tokens, and mark the mini-batch's last chunk as the update.
  buffered: list[packing.PackItem] = []  # unpacked sequences
  received = 0  # sequences received this mini-batch (incl. emitted)
  tokens_in_mini = 0  # for the dummy_ratio log
  chunks_in_mini = 0
  chunk_capacity = pack_size * max_token_budget
  example_cls: type[Any] = common.TrainExample
  mask_dtype: Any = jnp.float32
  first_item_for_dummy: packing.PackItem | None = None

  def _emit_chunk(bins, is_update):
    """Packs one chunk's bins and conver them to a TrainExample."""
    real = [it for b in bins for it in b]
    if first_item_for_dummy is not None:
      real = real + [first_item_for_dummy]
    carried = packing.carried_per_token_fields(real)
    rows = packing.pack_chunk(
        bins, budget=max_token_budget, pad_id=pad_id, carried=carried
    )
    return [
        pack_rows_to_train_examples(
            rows,
            example_cls,
            mask_dtype=mask_dtype,
            num_segments=num_segments,
            is_update_step=is_update,
        )
    ]

  def _take_chunk():
    nonlocal buffered, chunks_in_mini
    bins, buffered = packing.fill_one_chunk(
        buffered,
        pack_size=pack_size,
        budget=max_token_budget,
        max_segments=max_segments,
    )
    chunks_in_mini += 1
    return bins

  def _final_flush():
    nonlocal buffered, received, tokens_in_mini, chunks_in_mini
    if not buffered and chunks_in_mini == 0:
      raise ValueError(
          "pack_sequences reached an update boundary with an empty mini-batch;"
          " no packed example would be produced, dropping a gradient update."
      )
    while buffered:
      bins = _take_chunk()
      yield _emit_chunk(bins, not buffered)
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
      if getattr(example, "completion_mask", None) is not None:
        mask_dtype = np.asarray(example.completion_mask).dtype
      for item in train_example_to_pack_items(example):
        n = item.num_tokens
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
      while sum(it.num_tokens for it in buffered) >= chunk_capacity:
        bins = _take_chunk()
        yield _emit_chunk(bins, False)

  if buffered or received:
    raise ValueError("pack_sequences stream ended mid-mini-batch.")


# ------------------------------------------------------------------------------
# Per-host (multi-controller) sequence packing.
#
# Under the per-host learner (TUNIX_PER_HOST_LEARNER=1) every process packs its
# own rollouts only. `pack_sequences` would then emit a data-dependent number of
# chunks per process, and the processes would run different numbers of train
# steps (mismatched collectives). `pack_sequences_per_host` packs a process's
# whole mini-batch at once and agrees on the packed shape with one small
# allgather, so every process runs the same number of identically shaped steps.
# ------------------------------------------------------------------------------

# Columns of the int32 header every process contributes to the allgather.
_PH_ERR, _PH_ROWS, _PH_SEQS, _PH_TOKENS, _PH_LOSS_TOKENS, _PH_HOLES, _PH_SIG = (
    range(7)
)
_PH_NUM_COLS = 7
_PH_ERR_EXCEPTION = 1
_PH_ERR_TOO_LONG = 2
_PH_ERR_HOLES = 3
_PH_ERR_NO_PROMPT = 4
_PH_ERR_MASK_PAST_END = 5
_PH_ERR_UNSUPPORTED = 6
_PH_ERR_FIELDS = 7
_PH_ERROR_NAMES = {
    _PH_ERR_EXCEPTION: "unexpected error",
    _PH_ERR_TOO_LONG: "sequence longer than max_seq_tokens_per_device",
    _PH_ERR_HOLES: "pad-valued token inside a completion",
    _PH_ERR_NO_PROMPT: "empty prompt",
    _PH_ERR_MASK_PAST_END: "completion_mask set on padding",
    _PH_ERR_UNSUPPORTED: "unsupported TrainExample content",
    _PH_ERR_FIELDS: "sequences disagree on per-token fields",
}


class _PerHostPackError(Exception):
  """A per-host packing failure, reported to every process as an error code."""

  def __init__(self, code: int, detail: str):
    super().__init__(f"{_PH_ERROR_NAMES[code]}: {detail}")
    self.code = code


def per_host_pack_unit(data_axis_size: int, leading_dim_factor: int) -> int:
  """Smallest process-local row count that shards evenly over the data axis.

  `sharding_utils.shard_input` builds the global batch from process-local rows
  (`jax.make_array_from_process_local_data`) and falls back to a replicated
  sharding unless `local_rows * leading_dim_factor` divides by the data axis
  size (`leading_dim_factor` is the process count with
  TUNIX_PER_HOST_SAFE_LOGPS=1, else 1). Replicated process-local rows would
  silently train on the wrong data, so every packed chunk holds a multiple of
  this many rows per process.

  Args:
    data_axis_size: Product of the actor-mesh sizes of `data_sharding_axis`.
    leading_dim_factor: The factor `shard_input` applies to local rows.

  Returns:
    The row unit.
  """
  if data_axis_size <= 0 or leading_dim_factor <= 0:
    raise ValueError(
        f"data_axis_size={data_axis_size} and"
        f" leading_dim_factor={leading_dim_factor} must be positive."
    )
  return data_axis_size // math.gcd(data_axis_size, leading_dim_factor)


def per_host_pack_plan(
    max_local_rows: int,
    unit: int,
    max_rows_per_shard: int = 0,
    row_multiple: int = 1,
) -> tuple[int, int]:
  """Returns `(num_chunks, local_rows_per_chunk)` for one packed mini-batch.

  Every process calls this with the same allgathered `max_local_rows`, so every
  process derives the same plan.

  Args:
    max_local_rows: The most packed rows any process needs.
    unit: Row unit from `per_host_pack_unit`.
    max_rows_per_shard: If > 0, the most rows per data shard
      (`local_rows_per_chunk // unit`) in one train step; the mini-batch is then
      split into equal chunks whose gradients are accumulated. 0 = one step.
    row_multiple: Rounds the rows per data shard up to a multiple of this, so
      fewer distinct shapes are compiled. Must divide `max_rows_per_shard`.

  Returns:
    The number of train steps and the process-local rows of each.
  """
  if unit <= 0 or max_local_rows < 0:
    raise ValueError(
        f"per_host_pack_plan: unit={unit} must be > 0 and"
        f" max_local_rows={max_local_rows} >= 0."
    )
  if max_rows_per_shard < 0 or row_multiple <= 0:
    raise ValueError(
        f"per_host_pack_plan: max_rows_per_shard={max_rows_per_shard} must be"
        f" >= 0 and row_multiple={row_multiple} > 0."
    )
  if max_rows_per_shard and max_rows_per_shard % row_multiple:
    raise ValueError(
        f"per_host_pack_plan: row_multiple={row_multiple} must divide"
        f" max_rows_per_shard={max_rows_per_shard}."
    )
  rows_per_shard = max(1, _ceildiv(max_local_rows, unit))
  num_chunks = (
      _ceildiv(rows_per_shard, max_rows_per_shard) if max_rows_per_shard else 1
  )
  chunk_rows = _ceildiv(rows_per_shard, num_chunks)
  chunk_rows = _ceildiv(chunk_rows, row_multiple) * row_multiple
  return num_chunks, unit * chunk_rows


def _host_array(x: Any) -> np.ndarray:
  """Process-local host copy of a TrainExample leaf."""
  if isinstance(x, jax.Array):
    if not x.is_fully_addressable:
      # Never device_get a global array here: the other processes are not in
      # the same call, so the transfer would hang.
      raise _PerHostPackError(
          _PH_ERR_UNSUPPORTED, "a TrainExample leaf is a global array"
      )
    x = jax.device_get(x)
  return np.asarray(x)


def train_example_to_exact_pack_items(
    example: common.TrainExample, pad_id: int
) -> tuple[list[packing.PackItem], int]:
  """Unpads a TrainExample into PackItems whose packed loss is the unpacked one.

  The unpacked forward attends to exactly the tokens that are not `pad_id`
  (`common.process_ids`), while a packed segment attends to every token it
  holds. So here a completion ends at its last non-pad token (or follows
  `completion_attention_mask` when the example has one): loss-mask zeros before
  that point (e.g. rejected tokens) stay in the segment as context, as they do
  unpacked. A pad-valued token before that point (a "hole") has no packed
  equivalent; holes are counted so that the caller can refuse to pack.

  Args:
    example: A process-local TrainExample (left-padded prompts, right-padded
      completions).
    pad_id: The rollout pad id.

  Returns:
    `(items, num_holes)`.

  Raises:
    _PerHostPackError: if the example cannot be packed exactly.
  """
  if example.routed_experts is not None:
    raise _PerHostPackError(_PH_ERR_UNSUPPORTED, "routed_experts")
  if (
      example.old_topk_token_ids is not None
      or example.old_topk_logps is not None
  ):
    # Score Centering's `[B, C, K]` sampler top-k: packed rows do not carry it.
    raise _PerHostPackError(
        _PH_ERR_UNSUPPORTED, "old_topk_token_ids / old_topk_logps"
    )
  prompt_ids = _host_array(example.prompt_ids)
  prompt_mask = _host_array(example.prompt_mask).astype(bool)
  completion_ids = _host_array(example.completion_ids)
  completion_mask = _host_array(example.completion_mask)
  # Mirror `common.process_ids`: with an explicit completion_attention_mask the
  # forward attends to prompt_mask | completion_attention_mask, else to every
  # token that is not `pad_id`.
  valid = getattr(example, "completion_attention_mask", None)
  if isinstance(valid, (jax.Array, np.ndarray)):
    valid = _host_array(valid).astype(bool)
    prompt_valid = prompt_mask
  else:
    valid = completion_ids != pad_id
    prompt_valid = prompt_ids != pad_id
  advantages = _host_array(example.advantages).astype(np.float32)
  per_token = {
      name: _host_array(getattr(example, name)).astype(np.float32)
      for name in packing.PER_TOKEN_FIELDS
      if getattr(example, name, None) is not None
  }
  versions = getattr(example, "policy_version", None)
  if versions is not None:
    versions = _host_array(versions).reshape(-1)
  positions = np.arange(completion_ids.shape[1])
  lengths = np.where(valid, positions[None, :] + 1, 0).max(axis=1, initial=0)
  inside = positions[None, :] < lengths[:, None]
  holes = int(np.sum(inside & ~valid))
  if np.any((completion_mask != 0) & ~inside):
    raise _PerHostPackError(
        _PH_ERR_MASK_PAST_END, "a loss-mask token lies past the last valid one"
    )
  items = []
  for i in range(completion_ids.shape[0]):
    prompt = prompt_ids[i][prompt_valid[i]]
    if not prompt.size:
      # Its first completion token would be scored from the previous segment.
      raise _PerHostPackError(_PH_ERR_NO_PROMPT, f"row {i}")
    n = int(lengths[i])
    items.append(
        packing.PackItem(
            prompt_ids=prompt.astype(np.int32),
            completion_ids=completion_ids[i, :n].astype(np.int32),
            completion_mask=completion_mask[i, :n].astype(np.float32),
            advantages=(
                advantages[i, :n]
                if advantages.ndim == 2
                else np.full((n,), advantages[i], np.float32)
            ),
            per_token={name: arr[i, :n] for name, arr in per_token.items()},
            policy_version=None if versions is None else versions[i : i + 1],
        )
    )
  return items, holes


def _process_allgather_rows(row: np.ndarray) -> np.ndarray:
  """Gathers one int32 row from every JAX process, in process order.

  A collective: every process must call it at the same point of the program,
  from the thread that issues the other multi-process computations.

  Args:
    row: This process's 1-D int32 row.

  Returns:
    `[process_count, len(row)]`.
  """
  from jax.experimental import multihost_utils  # pylint: disable=g-import-not-at-top

  row = np.asarray(row, dtype=np.int32)
  with jax.transfer_guard("allow"):
    gathered = multihost_utils.process_allgather(row)
  return np.asarray(gathered).reshape(-1, row.shape[0])


def _pad_rows(arr: np.ndarray, rows: int, fill: Any = 0) -> np.ndarray:
  """Appends `fill` rows to `arr` up to `rows` rows."""
  extra = rows - arr.shape[0]
  if extra < 0:
    raise ValueError(f"_pad_rows: {arr.shape[0]} rows > {rows}.")
  if not extra:
    return arr
  pad = np.full((extra,) + arr.shape[1:], fill, dtype=arr.dtype)
  return np.concatenate([arr, pad], axis=0)


def _pack_mini_batch_per_host(
    examples: Sequence[common.TrainExample],
    *,
    budget: int,
    pad_id: int,
    unit: int,
    max_segments: int,
    max_rows_per_shard: int,
    row_multiple: int,
    allgather_fn: Callable[[np.ndarray], np.ndarray],
    process_index: int,
    step: int,
) -> list[common.TrainExample]:
  """Packs one process-local mini-batch into the globally agreed chunks."""
  t0 = time.perf_counter()
  num_segments = max_segments + 1
  example_cls: type[Any] = type(examples[0])
  mask_dtype = np.dtype(np.float32)
  applied = False
  err, detail = 0, ""
  items: list[packing.PackItem] = []
  holes = 0
  packed = None
  signature = 0
  # Phase A (local): every failure becomes an error code and nothing raises,
  # so no process skips the allgather below and leaves the others hanging.
  try:
    mask_dtype = _host_array(examples[0].completion_mask).dtype
    # Static flag of the unpacked examples; the packed chunks keep it.
    applied = bool(examples[0].sampler_agreement_applied)
    for example in examples:
      example_items, example_holes = train_example_to_exact_pack_items(
          example, pad_id
      )
      items.extend(example_items)
      holes += example_holes
    if holes:
      raise _PerHostPackError(
          _PH_ERR_HOLES,
          f"{holes} token(s); the unpacked forward treats them as padding. Add"
          " the pad id to the rollout stop tokens, or disable packing.",
      )
    longest = max((item.num_tokens for item in items), default=0)
    if longest > budget:
      raise _PerHostPackError(_PH_ERR_TOO_LONG, f"{longest} > {budget}")
    try:
      carried = packing.carried_per_token_fields(items)
    except ValueError as e:
      raise _PerHostPackError(_PH_ERR_FIELDS, str(e)) from e
    bins, leftover = packing.fill_one_chunk(
        items,
        pack_size=max(1, len(items)),
        budget=budget,
        max_segments=max_segments,
    )
    if leftover:
      raise _PerHostPackError(
          _PH_ERR_EXCEPTION, f"{len(leftover)} sequence(s) fit no row"
      )
    packed = packing.pack_chunk(
        [b for b in bins if b], budget=budget, pad_id=pad_id, carried=carried
    )
    signature = zlib.crc32(
        repr((
            budget,
            num_segments,
            tuple(carried),
            mask_dtype.str,
            example_cls.__name__,
            applied,
        )).encode()
    ) & 0x7FFFFFFF
  except _PerHostPackError as e:
    err, detail = e.code, str(e)
  except Exception as e:  # pylint: disable=broad-exception-caught
    err, detail = _PH_ERR_EXCEPTION, repr(e)[:300]
  rows_used = 0 if packed is None else len(packed)
  tokens = sum(item.num_tokens for item in items)
  loss_tokens = int(
      round(sum(float(item.completion_mask.sum()) for item in items))
  )
  header = np.array(
      [err, rows_used, len(items), tokens, loss_tokens, holes, signature],
      dtype=np.int32,
  )
  t1 = time.perf_counter()
  table = np.asarray(allgather_fn(header), dtype=np.int64).reshape(
      -1, _PH_NUM_COLS
  )
  t2 = time.perf_counter()
  failed = np.flatnonzero(table[:, _PH_ERR])
  if failed.size:
    reasons = {
        int(p): _PH_ERROR_NAMES.get(int(table[p, _PH_ERR]), "unknown")
        for p in failed
    }
    logging.error(
        "[TUNIX-PH-PACK] FAILED step=%d proc=%d local=%s header_table=%s",
        step,
        process_index,
        detail or "ok",
        table.tolist(),
    )
    raise RuntimeError(
        f"pack_sequences_per_host: packing failed on process(es) {reasons};"
        f" this process ({process_index}): {detail or 'ok'}. Every process"
        " raises here."
    )
  if len(set(table[:, _PH_SIG].tolist())) != 1:
    raise RuntimeError(
        "pack_sequences_per_host: processes disagree on the packed layout"
        f" (budget, segments, per-token fields, dtypes): {table.tolist()}"
    )
  assert packed is not None
  # Phase B (local, deterministic): the same plan on every process.
  num_chunks, rows_per_chunk = per_host_pack_plan(
      int(table[:, _PH_ROWS].max()), unit, max_rows_per_shard, row_multiple
  )
  total_rows = num_chunks * rows_per_chunk
  ids = _pad_rows(packed.ids, total_rows, fill=pad_id)
  completion_mask = _pad_rows(packed.completion_mask, total_rows).astype(
      mask_dtype
  )
  advantages = _pad_rows(packed.advantages, total_rows)
  segment_ids = _pad_rows(packed.segment_ids, total_rows)
  segment_positions = _pad_rows(packed.segment_positions, total_rows)
  per_token = {
      name: _pad_rows(arr, total_rows) for name, arr in packed.per_token.items()
  }
  versions = list(packed.policy_versions) + [None] * (total_rows - rows_used)
  policy_version = None
  if any(v is not None for v in versions):
    fallback = next(v for v in versions if v is not None)
    policy_version = np.concatenate([
        np.asarray(v if v is not None else fallback).reshape(-1)
        for v in versions
    ])
  chunks = []
  for c in range(num_chunks):
    rows = slice(c * rows_per_chunk, (c + 1) * rows_per_chunk)
    kwargs: dict[str, Any] = dict(
        prompt_ids=np.zeros((rows_per_chunk, 0), dtype=np.int32),
        prompt_mask=np.zeros((rows_per_chunk, 0), dtype=mask_dtype),
        completion_ids=ids[rows],
        completion_mask=completion_mask[rows],
        advantages=advantages[rows],
        segment_ids=segment_ids[rows],
        segment_positions=segment_positions[rows],
        ref_per_token_logps=None,
        old_per_token_logps=None,
    )
    for name, arr in per_token.items():
      kwargs[name] = arr[rows]
    if policy_version is not None:
      kwargs["policy_version"] = policy_version[rows]
    chunk = example_cls(**kwargs)
    # A 0-d flag: `shard_input` replicates it on any mesh, and the trainer
    # reads it with `.item()`.
    chunks.append(
        chunk.replace(
            is_update_step=np.asarray(c == num_chunks - 1),
            num_segments=num_segments,
            sampler_agreement_applied=applied,
        )
    )
  t3 = time.perf_counter()
  nproc = table.shape[0]
  glob_pad_frac = 1.0 - float(table[:, _PH_TOKENS].sum()) / float(
      nproc * total_rows * budget
  )
  logging.info(
      "[TUNIX-PH-PACK] step=%d proc=%d/%d seqs=%d rows_used=%d chunks=%d"
      " rows_per_chunk=%d rows_per_shard=%d tokens=%d loss_tokens=%d"
      " pad_frac=%.4f | glob_seqs=%d glob_rows_used=%d max_rows_used=%d"
      " glob_rows=%d glob_pad_frac=%.4f | pack_ms=%.1f allgather_ms=%.1f"
      " build_ms=%.1f",
      step,
      process_index,
      nproc,
      len(items),
      rows_used,
      num_chunks,
      rows_per_chunk,
      rows_per_chunk // unit,
      tokens,
      loss_tokens,
      1.0 - tokens / float(total_rows * budget),
      int(table[:, _PH_SEQS].sum()),
      int(table[:, _PH_ROWS].sum()),
      int(table[:, _PH_ROWS].max()),
      nproc * total_rows,
      glob_pad_frac,
      (t1 - t0) * 1e3,
      (t2 - t1) * 1e3,
      (t3 - t2) * 1e3,
  )
  if process_index == 0:
    # One summary line per step: K = packed rows per data shard the fullest
    # process needs; steps = train steps (chunks) this mini-batch runs.
    logging.info(
        "Per-host packing: step=%d K=%d steps=%d rows_per_shard_per_step=%d"
        " glob_rows_used=%d glob_rows=%d pad_frac=%.4f pack_ms=%.1f"
        " allgather_ms=%.1f build_ms=%.1f",
        step,
        max(1, _ceildiv(int(table[:, _PH_ROWS].max()), unit)),
        num_chunks,
        rows_per_chunk // unit,
        int(table[:, _PH_ROWS].sum()),
        nproc * total_rows,
        glob_pad_frac,
        (t1 - t0) * 1e3,
        (t2 - t1) * 1e3,
        (t3 - t2) * 1e3,
    )
  return chunks


def pack_sequences_per_host(
    item_iterator: Iterator[Sequence[common.TrainExample]],
    max_token_budget: int,
    sequences_per_update: int,
    *,
    pad_id: int,
    data_axis_size: int,
    leading_dim_factor: int,
    max_segments_per_packed_row: int | None = None,
    max_rows_per_shard: int = 0,
    row_multiple: int = 1,
    allgather_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    process_index: int | None = None,
    first_step: int = 0,
) -> Iterator[list[common.TrainExample]]:
  """Per-host learner counterpart of `pack_sequences`, with an agreed shape.

  Under TUNIX_PER_HOST_LEARNER=1 each process packs only its own sequences, so
  the number of packed rows differs between processes. Per mini-batch
  (`sequences_per_update` process-local sequences) this:

  1. Packs locally, FFD, into rows of `max_token_budget` tokens: one segment
     per sequence, `[prompt | completion]`, positions restarting at 0.
     Failures become an error code; nothing raises yet.
  2. Allgathers one int32 header per process (a collective, once per
     mini-batch). Any failure or disagreement raises the same error on every
     process.
  3. Derives the same plan on every process from the gathered table
     (`per_host_pack_plan`): `num_chunks` train steps of `rows_per_chunk`
     process-local rows each, a multiple of the row unit. Unused rows are dummy
     rows (segment 0, every mask 0) that add nothing to the loss or to its
     denominators. The last chunk has `is_update_step=True`.

  Every process therefore runs the same number of train steps on identically
  shaped and structured inputs, and one optimizer update still covers one
  global mini-batch.

  Args:
    item_iterator: Stream of lists of process-local TrainExamples.
    max_token_budget: Tokens per packed row (= max_seq_tokens_per_device).
    sequences_per_update: Process-local sequences per mini-batch.
    pad_id: The rollout pad id (marks invalid tokens, fills padding).
    data_axis_size: Product of the actor-mesh sizes of `data_sharding_axis`.
    leading_dim_factor: The factor `shard_input` applies to process-local rows
      (the process count with TUNIX_PER_HOST_SAFE_LOGPS=1, else 1).
    max_segments_per_packed_row: As in `pack_sequences`.
    max_rows_per_shard: As in `per_host_pack_plan` (0 = one step).
    row_multiple: As in `per_host_pack_plan`.
    allgather_fn: Gathers a 1-D int32 row from every process into
      `[process_count, len]`; defaults to `multihost_utils.process_allgather`.
    process_index: This process's index, for logs; defaults to JAX's.
    first_step: Step number of the first mini-batch, for logs.

  Yields:
    Single-element lists, each holding one packed TrainExample.

  Raises:
    ValueError: a mini-batch boundary inside an input example, or a stream
      that ends mid-mini-batch.
    RuntimeError: packing failed on any process (raised on every process).
  """
  if max_token_budget <= 0 or sequences_per_update <= 0:
    raise ValueError(
        f"max_token_budget={max_token_budget} and"
        f" sequences_per_update={sequences_per_update} must be positive."
    )
  unit = per_host_pack_unit(data_axis_size, leading_dim_factor)
  per_host_pack_plan(0, unit, max_rows_per_shard, row_multiple)  # Validates.
  max_segments = packing.effective_max_segments(
      max_token_budget, max_segments_per_packed_row
  )
  if max_segments <= 0:
    raise ValueError(f"max_segments_per_packed_row={max_segments} must be > 0.")
  if allgather_fn is None:
    allgather_fn = _process_allgather_rows
  if process_index is None:
    process_index = jax.process_index()
  buffered: list[common.TrainExample] = []
  received = 0
  step = first_step
  for example_list in item_iterator:
    for example in example_list:
      buffered.append(example)
      received += int(np.shape(example.completion_ids)[0])
    if received > sequences_per_update:
      raise ValueError(
          "pack_sequences_per_host: mini-batch boundary falls inside an input"
          f" example (received {received} sequences, expected"
          f" {sequences_per_update} per update)."
      )
    if received < sequences_per_update:
      continue
    chunks = _pack_mini_batch_per_host(
        buffered,
        budget=max_token_budget,
        pad_id=pad_id,
        unit=unit,
        max_segments=max_segments,
        max_rows_per_shard=max_rows_per_shard,
        row_multiple=row_multiple,
        allgather_fn=allgather_fn,
        process_index=process_index,
        step=step,
    )
    buffered, received = [], 0
    step += 1
    for chunk in chunks:
      yield [chunk]
  if buffered or received:
    raise ValueError("pack_sequences_per_host: stream ended mid-mini-batch.")


VERIFY_UPDATE_PARAMS_KEY = "VERIFY_UPDATE_PARAMS_SRC_TO_TGT_MODULE_NAME"
