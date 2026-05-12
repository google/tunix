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
from typing import Any, Iterator, List, Mapping, Optional, Sequence

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


def pack_sequences(
    item_iterator: Iterator[Sequence[common.TrainExample]],
    max_token_budget: int,
    pad_id: int = 0,
    num_packs: int = 1,
) -> Iterator[list[common.TrainExample]]:
  """Packs a stream of TrainExamples into 1D sequences up to a token budget.

  Optionally stacks ``num_packs`` packs along the batch axis to produce a
  shardable batch of shape ``[num_packs, max_token_budget]``.

  Args:
    item_iterator: An iterator that yields lists of TrainExamples (one
      micro-batch per element).
    max_token_budget: The maximum number of tokens allowed per packed sequence.
    pad_id: The token ID to use for padding. Defaults to 0.
    num_packs: When ``> 1``, packs are accumulated and concatenated along the
      batch axis so each yielded batch contains ``num_packs`` sequences of
      shape ``[num_packs, max_token_budget]``. Set this to the FSDP mesh axis
      size to ensure the yielded batch is shardable by JAX. Trailing packs are
      padded with dummy zero-mask data when the total number of packs is not a
      multiple of ``num_packs``. Defaults to 1.

  Yields:
    A list of size 1 containing a single TrainExample whose arrays are stacked
    along the batch axis to shape ``[num_packs, max_token_budget]``.
  """
  buffer = []
  current_tokens = 0
  example_cls = common.TrainExample
  pending_packs: List[common.TrainExample] = []

  def _build_pack() -> common.TrainExample | None:
    """Builds a single packed TrainExample from the current buffer."""
    nonlocal buffer, current_tokens
    if not buffer:
      return None

    # TODO(noghabi): Pad to the next power of 2 instead of user defined
    # max_token_budget if the seq is short. This will incur an additional
    # compilation on trainer side, but also will result in faster compute.
    pad_len = max_token_budget - current_tokens

    packed_c_ids = []
    packed_c_mask = []
    packed_adv = []
    packed_segment_ids = []
    packed_positions = []

    tracked_per_token_keys = [
        k for k in _OPTIONAL_PER_TOKEN_KEYS if buffer[0].get(k) is not None
    ]
    per_token_feature_buffers = {k: [] for k in tracked_per_token_keys}
    has_policy_version = buffer[0].get("policy_version") is not None

    if has_policy_version:
      # Homogeneity invariant: every item in a single pack must share the same
      # policy_version (enforced by the version-aware flush below). If this
      # fires it indicates a bug in the buffering logic, not in caller data.
      versions = [int(np.asarray(item["policy_version"]).item()) for item in buffer]
      assert all(v == versions[0] for v in versions), (
          f"pack_sequences invariant violation: heterogeneous policy_versions"
          f" within a single pack: {versions}"
      )

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

    # Empty prompt arrays
    p_ids_arr = jnp.zeros((1, 0), dtype=jnp.int32)
    p_mask_arr = jnp.zeros((1, 0), dtype=jnp.int32)

    # Pad all lists by pad_len
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
      kwargs["policy_version"] = buffer[0]["policy_version"]

    packed_example = example_cls(**kwargs)  # pytype: disable=wrong-keyword-args

    buffer.clear()
    current_tokens = 0
    return packed_example

  def _stack_pending() -> list[common.TrainExample]:
    """Pad ``pending_packs`` with dummies and concatenate along batch axis."""
    if not pending_packs:
      return []
    # Pad with all-zero dummy packs so total count is a multiple of num_packs.
    # Dummy packs have completion_mask=0 everywhere, so they contribute 0 to
    # the loss/denominator. policy_version is inherited from the template so
    # downstream rollout-version bookkeeping does not see spurious sentinels.
    template = pending_packs[0]
    while len(pending_packs) % num_packs != 0:
      dummy = jax.tree.map(
          lambda x: jnp.zeros_like(x) if x is not None else None, template
      )
      if getattr(template, "policy_version", None) is not None:
        dummy = dataclasses.replace(
            dummy, policy_version=template.policy_version
        )
      pending_packs.append(dummy)
    stacked = jax.tree.map(
        lambda first_x, *rest: None
        if first_x is None
        else jnp.concatenate((first_x, *rest), axis=0),
        *pending_packs,
    )
    pending_packs.clear()
    return [stacked]

  def _emit() -> list[common.TrainExample]:
    """Builds a pack from the current buffer and yields when ready."""
    pack = _build_pack()
    if pack is None:
      return []
    if num_packs == 1:
      return [pack]
    pending_packs.append(pack)
    if len(pending_packs) >= num_packs:
      return _stack_pending()
    return []

  def _version_changed(item: dict[str, Any]) -> bool:
    """True if `item` carries a `policy_version` different from `buffer[0]`."""
    if not buffer:
      return False
    a = buffer[0].get("policy_version")
    b = item.get("policy_version")
    if a is None or b is None:
      return False
    return int(np.asarray(a).item()) != int(np.asarray(b).item())

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

        # Version-aware flush: a pack must contain items from a single
        # `policy_version` (so downstream off-policy filtering and per-row
        # version bookkeeping are unambiguous).
        if current_tokens + tokens > max_token_budget or _version_changed(item):
          out = _emit()
          if out:
            yield out

        buffer.append(item)
        current_tokens += tokens

  # Flush any remaining buffered items, then any pending packs.
  out = _emit()
  if out:
    yield out
  out = _stack_pending()
  if out:
    yield out


VERIFY_UPDATE_PARAMS_KEY = "VERIFY_UPDATE_PARAMS_SRC_TO_TGT_MODULE_NAME"
