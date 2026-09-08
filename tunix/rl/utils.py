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
import operator
from typing import Any, Iterator, Mapping, Optional, Sequence

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
    c_len = int(np.sum(c_mask[i]))

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
    rows: list[packing.PackedRow],
    example_cls: type[Any],
    *,
    mask_dtype: Any,
    num_segments: int,
    is_update_step: bool,
) -> Any:
  """Converts a list of PackedRows to a list of TrainExamples."""
  n = len(rows)
  stack = lambda attr: jnp.asarray(np.stack([getattr(r, attr) for r in rows]))
  kwargs: dict[str, Any] = dict(
      prompt_ids=jnp.zeros((n, 0), dtype=np.int32),
      prompt_mask=jnp.zeros((n, 0), dtype=mask_dtype),
      completion_ids=stack("ids"),
      completion_mask=jnp.asarray(
          np.stack([r.completion_mask for r in rows]).astype(mask_dtype),
          copy=False,
      ),
      advantages=stack("advantages"),
      segment_ids=stack("segment_ids"),
      segment_positions=stack("segment_positions"),
      ref_per_token_logps=None,
      old_per_token_logps=None,
  )
  for name in rows[0].per_token:
    kwargs[name] = jnp.asarray(np.stack([r.per_token[name] for r in rows]))
  versions = [r.policy_version for r in rows]
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


VERIFY_UPDATE_PARAMS_KEY = "VERIFY_UPDATE_PARAMS_SRC_TO_TGT_MODULE_NAME"
