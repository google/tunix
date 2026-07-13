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

import heapq
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

Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding

_OPTIONAL_PER_TOKEN_KEYS = (
    "ref_per_token_logps",
    "old_per_token_logps",
    "returns",
    "old_values",
)


def is_positive_integer(value: int | None, name: str):
  """Checks if the value is positive."""
  if value is not None and (not isinstance(value, int) or value <= 0):
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
    actor_model: nnx.Module, seed: int = 0, lm_head_to_replace: str = "lm_head"
) -> nnx.Module:
  """Creates a critic model from an actor model."""
  g, state = nnx.split(actor_model)
  # TODO(tsbao): if actor model is a LoRA model, then we can potentially share
  # backbone of base weights with critic model. Do it later as an optimization.
  copied_state = jax.tree.map(jnp.copy, state)
  critic_model = nnx.merge(g, copied_state)
  lm_head = getattr(critic_model, lm_head_to_replace)
  hidden_dim = (
      lm_head.shape[0] if hasattr(lm_head, "shape") else lm_head.in_features
  )
  setattr(
      critic_model,
      lm_head_to_replace,
      nnx.Linear(
          in_features=hidden_dim,
          out_features=1,
          use_bias=False,
          rngs=nnx.Rngs(seed),
      ),
  )
  return critic_model


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
        "ref_per_token_logps": ref_logps[i, :c_len] if has_ref else None,
        "old_per_token_logps": old_logps[i, :c_len] if has_old else None,
        "returns": returns_np[i, :c_len] if has_returns else None,
        "old_values": old_values_np[i, :c_len] if has_old_values else None,
        "policy_version": (
            policy_version_np[i : i + 1] if has_policy_version else None
        ),
    }
    res.append(item)
  return res


def _karmarkar_karp(vals, k):
  """Partition indices [0, len(vals)) into k groups with balanced sums.

  Karmarkar-Karp largest-differencing heuristic: repeatedly merge the two most
  imbalanced arrangements, pairing each arrangement's largest bucket with the
  other's smallest, until one arrangement (the k groups) remains. Returns k
  lists of indices into vals; groups may be empty when k > len(vals).
  """

  class _Set:  # one bucket (one pack): running sum + which sequences it holds
    def __init__(self):
      self.sum = 0
      self.items = []

    def add(self, idx, val):
      self.items.append((idx, val))
      self.sum += val

    def merge(self, other):
      for idx, val in other.items:
        self.items.append((idx, val))
        self.sum += val

    def __lt__(self, other):
      return self.sum < other.sum

  class _State:  # one arrangement of k buckets, kept sorted by sum descending
    def __init__(self, idx, val, k):
      self.k = k
      self.sets = [_Set() for _ in range(k)]
      self.sets[0].add(idx, val)
      self.sets.sort(reverse=True)

    @property
    def spread(self):
      return self.sets[0].sum - self.sets[-1].sum

    def merge(self, other):
      # Pair our i-th largest bucket with other's i-th smallest (big + small).
      for i in range(self.k):
        self.sets[i].merge(other.sets[self.k - 1 - i])
      self.sets.sort(reverse=True)

    def __lt__(self, other):
      # heapq is a min-heap; invert so the largest-spread state pops first.
      return self.spread > other.spread

    def partitions(self):
      return [[idx for idx, _ in s.items] for s in self.sets]

  pq = []
  for idx, val in enumerate(vals):
    heapq.heappush(pq, _State(idx, val, k))
  while len(pq) > 1:
    a = heapq.heappop(pq)
    b = heapq.heappop(pq)
    a.merge(b)
    heapq.heappush(pq, a)
  return pq[0].partitions()


def _group_by_version(valid):
  """Group (item, tokens) pairs by policy_version so a pack never mixes versions.

  Returns groups in first-seen version order. When no item carries a
  policy_version (synchronous training) everything lands in a single group.
  """
  groups = {}
  order = []
  for item, tokens in valid:
    v = item.get("policy_version")
    key = None if v is None else int(np.asarray(v).item())
    if key not in groups:
      groups[key] = []
      order.append(key)
    groups[key].append((item, tokens))
  return [groups[key] for key in order]


def _balanced_pack(items_with_tokens, max_token_budget, num_packs):
  """Split (item, tokens) pairs into KK-balanced packs, each <= max_token_budget.

  Balances by token count via Karmarkar-Karp. Picks the pack count k as the
  smallest multiple of num_packs whose average pack fits the budget, then bumps
  k by num_packs until every pack fits (a static-shape capacity guard the
  balancing itself does not enforce).
  """
  tokens = [t for _, t in items_with_tokens]
  # TODO: balancing by token count ignores attention's O(L^2) cost, so packs
  # can still be attention-imbalanced when sequence lengths vary. For long
  # context, switch to a length-squared-aware workload.
  workloads = tokens
  n = len(items_with_tokens)

  k = -(-sum(tokens) // max_token_budget)  # ceil(total / budget)
  k = -(-k // num_packs) * num_packs  # round up to a num_packs multiple
  k = min(k, n)

  while True:
    groups = _karmarkar_karp(workloads, k)
    if k >= n or all(
        sum(tokens[i] for i in g) <= max_token_budget for g in groups
    ):
      break
    k = min(n, k + num_packs)
  return [[items_with_tokens[i][0] for i in g] for g in groups]


def pack_sequences(
    item_iterator: Iterator[Sequence[common.TrainExample]],
    max_token_budget: int,
    pad_id: int = 0,
    target_items_per_update: int | None = None,
    num_packs: int = 1,
    packing_strategy: str = "kk",
) -> Iterator[list[common.TrainExample]]:
  """Packs a stream of TrainExamples into 1D sequences up to a token budget and stacks them into 2D batches of shape [num_packs, max_token_budget].

  Args:
    item_iterator: Stream of lists of TrainExamples.
    max_token_budget: The maximum number of tokens in a packed sequence.
    pad_id: The vocabulary id used for padding.
    target_items_per_update: Accumulate items over microbatches, and set `is_update_step=True` when reaching threshold.
    num_packs: Group into chunks of size num_packs. Emits batches of shape [num_packs, sequence_length].
    packing_strategy: "kk" (default) balances sequences across packs with
      Karmarkar-Karp; "first_fit" fills each pack greedily in arrival order.

  Yields:
    A stream of packed sequences, stacked in chunks of [num_packs, ...].
  """
  if packing_strategy not in ("kk", "first_fit"):
    raise ValueError(
        f"packing_strategy must be 'kk' or 'first_fit', got {packing_strategy!r}"
    )
  accumulated_items = 0

  def _flush_pack(pack_items, example_cls, first_item) -> common.TrainExample:
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
      return example_cls(**kwargs)  # pytype: disable=wrong-keyword-args

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

    if has_policy_version:
      versions = [
          int(np.asarray(item["policy_version"]).item()) for item in pack_items
      ]
      assert all(v == versions[0] for v in versions), (
          "pack_sequences invariant violation: heterogeneous policy_versions"
          f" within a single pack: {versions}"
      )

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
        ref_per_token_logps=None,
        old_per_token_logps=None,
        segment_ids=seg_arr,
        segment_positions=pos_arr,
    )
    for k in tracked_per_token_keys:
      kwargs[k] = per_token_features[k]

    if has_policy_version:
      kwargs["policy_version"] = pack_items[0]["policy_version"]

    return example_cls(**kwargs)  # pytype: disable=wrong-keyword-args

  for item_list in item_iterator:
    accumulated_items += 1

    all_unpadded_items = []
    example_cls = common.TrainExample
    for example in item_list:
      example_cls = type(example)
      all_unpadded_items.extend(unpad_train_example(example))

    if not all_unpadded_items:
      continue

    first_item, *_ = all_unpadded_items

    if packing_strategy == "first_fit":
      packs = []
      current_pack_items = []
      current_tokens = 0

      def _version_changed(item: Mapping[str, Any]) -> bool:
        if not current_pack_items:
          return False
        a = current_pack_items[0].get("policy_version")
        b = item.get("policy_version")
        if a is None or b is None:
          return False
        return int(np.asarray(a).item()) != int(np.asarray(b).item())

      for item in all_unpadded_items:
        tokens = len(item["prompt_ids"]) + len(item["completion_ids"])

        if tokens > max_token_budget:
          logging.warning(
              "Skipping single sequence with length %d exceeding budget %d",
              tokens, max_token_budget,
          )
          continue

        if current_tokens + tokens > max_token_budget or _version_changed(item):
          packs.append(current_pack_items)
          current_pack_items = []
          current_tokens = 0

        current_pack_items.append(item)
        current_tokens += tokens

      if current_pack_items:
        packs.append(current_pack_items)
    else:  # "kk": Karmarkar-Karp balanced packing (default)
      valid = []
      for item in all_unpadded_items:
        tokens = len(item["prompt_ids"]) + len(item["completion_ids"])
        if tokens > max_token_budget:
          logging.warning(
              "Skipping single sequence with length %d exceeding budget %d",
              tokens, max_token_budget,
          )
          continue
        valid.append((item, tokens))

      packs = []
      for group in _group_by_version(valid):
        packs.extend(_balanced_pack(group, max_token_budget, num_packs))

    if not packs:
      continue

    while len(packs) % num_packs != 0:
      packs.append([])

    chunks = [packs[i : i + num_packs] for i in range(0, len(packs), num_packs)]

    for chunk_idx, chunk in enumerate(chunks):
      chunk_examples = []
      for p in chunk:
        chunk_examples.append(_flush_pack(p, example_cls, first_item))

      merged_example = jax.tree.map(
          lambda first_x, *rest_xs: None
          if first_x is None
          else jnp.concatenate((first_x, *rest_xs), axis=0),
          *chunk_examples,
      )

      is_update = False
      if target_items_per_update and accumulated_items >= target_items_per_update:
        if chunk_idx == len(chunks) - 1:
          is_update = True
          accumulated_items = 0

      merged_example = merged_example.replace(
          is_update_step=jnp.array([is_update], dtype=jnp.bool_)
      )

      yield [merged_example]



VERIFY_UPDATE_PARAMS_KEY = "VERIFY_UPDATE_PARAMS_SRC_TO_TGT_MODULE_NAME"
