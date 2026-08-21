"""Observer-only processed prompt logprobs for the P57 stock arm.

The pinned TPU runner's stock prompt-logprob helper derives next-token IDs with
``roll(input_ids, -1)`` over the whole DP-packed buffer.  At a request or padded
DP boundary that can name a token from a different row.  The stock helper also
scores raw logits even when decode reports processed logprobs.  P57 uses this
module only for the post-rollout B observer: rollout sampling, model forward,
training, and optimizer state never call it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.sample.sampling import (
    PromptLogprobsAsyncData,
    PromptLogprobsReqSnap,
    _SAMPLING_EPS,
    _apply_sampling_transforms,
    _jax_logprobs_copy_to_host_async,
    compute_and_gather_logprobs,
)
from tpu_inference.layers.jax.sample.sampling_metadata import (
    TPUSupportedSamplingMetadata,
)
from tpu_inference.utils import device_array


_ANNOUNCED = False


def _expand_sampling_params(
    input_batch: Any,
    scheduler_output: Any,
    req_ids_dp: Optional[Dict[int, List[str]]],
    dp_size: int,
    padded_num_tokens: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
  if req_ids_dp is None:
    raise ValueError("P57 processed prompt observer requires req_ids_dp")
  if dp_size <= 0 or padded_num_tokens <= 0:
    raise ValueError(
        "invalid P57 prompt dimensions: "
        f"dp={dp_size}, tokens={padded_num_tokens}"
    )
  if padded_num_tokens % dp_size:
    raise ValueError(
        f"P57 prompt rows {padded_num_tokens} not divisible by dp_size {dp_size}"
    )

  rows_per_dp = padded_num_tokens // dp_size
  temperatures = np.ones((padded_num_tokens,), dtype=np.float32)
  top_ks = np.full((padded_num_tokens,), -1, dtype=np.int32)
  top_ps = np.ones((padded_num_tokens,), dtype=np.float32)
  populated = 0
  for dp_rank in range(dp_size):
    local_offset = 0
    for req_id in req_ids_dp.get(dp_rank, []):
      if req_id not in scheduler_output.num_scheduled_tokens:
        raise KeyError(f"scheduled-token count missing for request {req_id}")
      if req_id not in input_batch.req_id_to_index:
        raise KeyError(f"input-batch slot missing for request {req_id}")
      num_scheduled = int(scheduler_output.num_scheduled_tokens[req_id])
      if num_scheduled < 0 or local_offset + num_scheduled > rows_per_dp:
        raise ValueError(
            f"request {req_id} overflows dp rank {dp_rank}: "
            f"offset={local_offset}, scheduled={num_scheduled}, "
            f"capacity={rows_per_dp}"
        )
      req_index = input_batch.req_id_to_index[req_id]
      start = dp_rank * rows_per_dp + local_offset
      stop = start + num_scheduled
      temperatures[start:stop] = input_batch.temperature_cpu[req_index]
      top_ks[start:stop] = input_batch.top_k_cpu[req_index]
      top_ps[start:stop] = input_batch.top_p_cpu[req_index]
      local_offset += num_scheduled
      populated += num_scheduled

  expected = sum(int(v) for v in scheduler_output.num_scheduled_tokens.values())
  if populated != expected:
    raise ValueError(
        f"P57 prompt sampling rows incomplete: expanded={populated}, "
        f"scheduled={expected}"
    )
  return temperatures, top_ks, top_ps, populated


def _expand_absolute_target_ids(
    requests: Dict[str, Any],
    scheduler_output: Any,
    req_ids_dp: Optional[Dict[int, List[str]]],
    dp_size: int,
    padded_num_tokens: int,
) -> np.ndarray:
  """Build targets from absolute request positions, never packed-row roll."""
  if req_ids_dp is None:
    raise ValueError("P57 processed prompt observer requires req_ids_dp")
  if dp_size <= 0 or padded_num_tokens <= 0:
    raise ValueError(
        "invalid P57 target dimensions: "
        f"dp={dp_size}, tokens={padded_num_tokens}"
    )
  if padded_num_tokens % dp_size:
    raise ValueError(
        f"P57 target rows {padded_num_tokens} not divisible by dp_size {dp_size}"
    )

  rows_per_dp = padded_num_tokens // dp_size
  target_ids = np.zeros((padded_num_tokens,), dtype=np.int32)
  populated = 0
  for dp_rank in range(dp_size):
    local_offset = 0
    for req_id in req_ids_dp.get(dp_rank, []):
      if req_id not in scheduler_output.num_scheduled_tokens:
        raise KeyError(f"scheduled-token count missing for request {req_id}")
      if req_id not in requests:
        raise KeyError(f"request state missing for request {req_id}")
      num_scheduled = int(scheduler_output.num_scheduled_tokens[req_id])
      if num_scheduled < 0 or local_offset + num_scheduled > rows_per_dp:
        raise ValueError(
            f"request {req_id} overflows target rows on dp rank {dp_rank}: "
            f"offset={local_offset}, scheduled={num_scheduled}, "
            f"capacity={rows_per_dp}"
        )
      req_state = requests[req_id]
      start_idx = int(req_state.num_computed_tokens)
      start = dp_rank * rows_per_dp + local_offset
      for rel_idx in range(num_scheduled):
        next_idx = start_idx + rel_idx + 1
        if next_idx < req_state.num_prompt_tokens:
          target_ids[start + rel_idx] = int(req_state.get_token_id(next_idx))
      local_offset += num_scheduled
      populated += num_scheduled

  expected = sum(int(v) for v in scheduler_output.num_scheduled_tokens.values())
  if populated != expected:
    raise ValueError(
        f"P57 prompt target rows incomplete: expanded={populated}, "
        f"scheduled={expected}"
    )
  return target_ids


@jax.jit
def _process_prompt_logits(
    logits: jax.Array,
    metadata: TPUSupportedSamplingMetadata,
) -> jax.Array:
  processed = _apply_sampling_transforms(logits, metadata)
  is_greedy = metadata.temperature < _SAMPLING_EPS
  return jnp.where(jnp.expand_dims(is_greedy, axis=-1), logits, processed)


def compute_processed_prompt_logprobs(
    *,
    mesh: Any,
    full_logits: jax.Array,
    input_batch: Any,
    requests: Dict[str, Any],
    scheduler_output: Any,
    req_ids_dp: Optional[Dict[int, List[str]]],
    dp_size: int,
    max_logprobs: int,
) -> PromptLogprobsAsyncData:
  """Return processed B values while preserving each absolute target token."""
  global _ANNOUNCED
  rows = int(full_logits.shape[0])
  temperatures, top_ks, top_ps, populated = _expand_sampling_params(
      input_batch, scheduler_output, req_ids_dp, dp_size, rows
  )
  token_sharding = NamedSharding(
      mesh, PartitionSpec(ShardingAxisName.ATTN_DATA)
  )
  metadata = TPUSupportedSamplingMetadata(
      temperature=device_array(mesh, temperatures, sharding=token_sharding),
      top_k=device_array(mesh, top_ks, sharding=token_sharding),
      top_p=device_array(mesh, top_ps, sharding=token_sharding),
      do_sampling=True,
      logprobs=True,
  )
  target_ids = device_array(
      mesh,
      _expand_absolute_target_ids(
          requests, scheduler_output, req_ids_dp, dp_size, rows
      ),
      sharding=token_sharding,
  )
  tensors = compute_and_gather_logprobs(
      _process_prompt_logits(full_logits, metadata), target_ids, max_logprobs
  )
  tensors = _jax_logprobs_copy_to_host_async(tensors)

  padded_tokens_per_dp = rows // dp_size
  req_snaps: List[PromptLogprobsReqSnap] = []
  if req_ids_dp:
    for dp_rank, req_id_list in req_ids_dp.items():
      dp_token_offset = dp_rank * padded_tokens_per_dp
      local_token_offset = 0
      for req_id in req_id_list:
        num_scheduled = int(scheduler_output.num_scheduled_tokens[req_id])
        if req_id in input_batch.num_prompt_logprobs:
          req_state = requests[req_id]
          start_idx = int(req_state.num_computed_tokens)
          num_remaining = req_state.num_prompt_tokens - (start_idx + 1)
          req_snaps.append(
              PromptLogprobsReqSnap(
                  req_id=req_id,
                  req_state=req_state,
                  req_offset=dp_token_offset + local_token_offset,
                  start_idx=start_idx,
                  num_logits=(
                      num_scheduled
                      if num_scheduled <= num_remaining
                      else num_remaining
                  ),
                  is_last_chunk=num_scheduled > num_remaining,
                  num_k=input_batch.num_prompt_logprobs[req_id],
              )
          )
        local_token_offset += num_scheduled

  if not _ANNOUNCED:
    print(
        "[P57.STOCK_OBSERVER] PROCESSED_PROMPT_LOGPROBS_PASS "
        f"rows={rows} populated={populated} "
        "targets=absolute-request-history treatment=observer-only",
        flush=True,
    )
    _ANNOUNCED = True
  return PromptLogprobsAsyncData(tensors=tensors, req_snaps=req_snaps)
