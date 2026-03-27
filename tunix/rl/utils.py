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
from typing import Any, Iterator, List, Optional

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
  return np.all(
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


def merge_micro_batches(batches: List[dict[str, Any]]) -> dict[str, Any]:
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

  for key in batches[0].keys():
    all_values = [item[key] for item in batches]

    if isinstance(all_values[0], list):
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
  logging.info("original_shardings: %s", original_shardings)
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
  logging.info("params_on_memory_kind shardings: %s", shardings)
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

    item = {
        "prompt_ids": p_ids[i, :p_len],
        "prompt_mask": p_mask[i, :p_len],
        "completion_ids": c_ids[i, :c_len],
        "completion_mask": c_mask[i, :c_len],
        "advantages": adv[i, :c_len] if adv_is_per_token else adv[i],
        "adv_is_per_token": adv_is_per_token,
        "ref_per_token_logps": ref_logps[i, :c_len] if has_ref else None,
        "old_per_token_logps": old_logps[i, :c_len] if has_old else None,
        "returns": returns_np[i, :c_len] if has_returns else None,
        "old_values": old_values_np[i, :c_len] if has_old_values else None,
        "policy_version": policy_version_np if has_policy_version else None,
    }
    res.append(item)
  return res


def pack_sequences(
    item_iterator: Iterator[list[common.TrainExample]],
    max_token_budget: int,
    pad_id: int = 0,
) -> Iterator[list[common.TrainExample]]:
  """Packs a stream of TrainExamples into 1D sequences up to a token budget."""
  buffer = []
  current_tokens = 0
  example_cls = common.TrainExample

  def _flush_buffer() -> list[common.TrainExample]:
    nonlocal buffer, current_tokens
    if not buffer:
      return []

    pad_len = max_token_budget - current_tokens

    packed_c_ids = []
    packed_c_mask = []
    packed_adv = []
    packed_ref_logps = []
    packed_old_logps = []
    packed_returns = []
    packed_old_values = []
    packed_segment_ids = []
    packed_positions = []

    has_ref = buffer[0].get("ref_per_token_logps") is not None
    has_old = buffer[0].get("old_per_token_logps") is not None
    has_returns = buffer[0].get("returns") is not None
    has_old_values = buffer[0].get("old_values") is not None
    has_policy_version = buffer[0].get("policy_version") is not None

    for i, item in enumerate(buffer, start=1):
      p_ids = item["prompt_ids"]
      c_ids = item["completion_ids"]
      seq_len = len(p_ids) + len(c_ids)

      packed_c_ids.extend([p_ids, c_ids])
      packed_c_mask.extend([np.zeros_like(p_ids), item["completion_mask"]])

      # Expand advantage to shape [c_len] to match completion length
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

      if has_ref:
        packed_ref_logps.extend([
            np.zeros_like(p_ids, dtype=np.float32),
            item["ref_per_token_logps"],
        ])
      if has_old:
        packed_old_logps.extend([
            np.zeros_like(p_ids, dtype=np.float32),
            item["old_per_token_logps"],
        ])
      if has_returns:
        packed_returns.extend([
            np.zeros_like(p_ids, dtype=np.float32),
            item["returns"],
        ])
      if has_old_values:
        packed_old_values.extend([
            np.zeros_like(p_ids, dtype=np.float32),
            item["old_values"],
        ])

      packed_segment_ids.append(np.full(seq_len, i, dtype=np.int32))
      packed_positions.append(np.arange(seq_len, dtype=np.int32))

    def _pad(arr_list, val, length):
      arr = np.concatenate(arr_list) if arr_list else np.array([])
      return np.pad(arr, (0, length), constant_values=val)

    # Empty prompt arrays
    p_ids_arr = jnp.zeros((1, 0), dtype=jnp.int32)
    p_mask_arr = jnp.zeros((1, 0), dtype=jnp.int32)

    # Pad all lists by pad_len
    c_ids_arr = jnp.array(_pad(packed_c_ids, pad_id, pad_len))[None, :]
    c_mask_arr = jnp.array(_pad(packed_c_mask, 0, pad_len))[None, :]
    adv_arr = jnp.array(_pad(packed_adv, 0.0, pad_len))[None, :]
    seg_arr = jnp.array(_pad(packed_segment_ids, 0, pad_len))[None, :]
    pos_arr = jnp.array(_pad(packed_positions, 0, pad_len))[None, :]

    ref_arr = (
        jnp.array(_pad(packed_ref_logps, 0.0, pad_len))[None, :]
        if has_ref
        else None
    )
    old_arr = (
        jnp.array(_pad(packed_old_logps, 0.0, pad_len))[None, :]
        if has_old
        else None
    )
    ret_arr = (
        jnp.array(_pad(packed_returns, 0.0, pad_len))[None, :]
        if has_returns
        else None
    )
    old_val_arr = (
        jnp.array(_pad(packed_old_values, 0.0, pad_len))[None, :]
        if has_old_values
        else None
    )

    kwargs = dict(
        prompt_ids=p_ids_arr,
        prompt_mask=p_mask_arr,
        completion_ids=c_ids_arr,
        completion_mask=c_mask_arr,
        advantages=adv_arr,
        ref_per_token_logps=ref_arr,
        old_per_token_logps=old_arr,
        segment_ids=seg_arr,
        positions=pos_arr,
    )
    if has_returns:
      kwargs["returns"] = ret_arr
    if has_old_values:
      kwargs["old_values"] = old_val_arr
    if has_policy_version:
      kwargs["policy_version"] = buffer[0]["policy_version"]

    packed_example = example_cls(**kwargs)  # pytype: disable=wrong-keyword-args

    buffer.clear()
    current_tokens = 0
    return [packed_example]

  for item_list in item_iterator:
    for example in item_list:
      example_cls = type(example)
      unpadded_items = unpad_train_example(example)
      for item in unpadded_items:
        tokens = len(item["prompt_ids"]) + len(item["completion_ids"])

        # If a single item is strictly larger than budget, we skip or truncate.
        # Ideally, budget > max_prompt_length + max_response_length.
        if tokens > max_token_budget:
          logging.warning(
              "Skipping single sequence with length %d exceeding budget %d",
              tokens,
              max_token_budget,
          )
          continue

        if current_tokens + tokens > max_token_budget:
          yield _flush_buffer()

        buffer.append(item)
        current_tokens += tokens

  if buffer:
    yield _flush_buffer()


VERIFY_UPDATE_PARAMS_KEY = "VERIFY_UPDATE_PARAMS_SRC_TO_TGT_MODULE_NAME"
