# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functional building blocks for the canonical Qwen3 engine adapter.

The rollout weight-sync utility mutates the destination engine state.  That is
correct for serving, but it cannot be the differentiable path used by the
trainer loss.  This module applies the same mapping transforms without writing
the target state and returns leaves in the target engine state's flat order.

The pure weight-map helpers implement the A1 contract.  The live adapter adds
the separately admitted model/cache/metadata contract and reuses the engine's
exact processed-logprob call boundary; neither layer mutates serving state.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import os
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp

from tunix.generate import utils as generate_utils
from tunix.rl import canonical_logsoftmax


class FunctionalMappingError(ValueError):
  """Raised when a trainer-to-engine weight map is not a bijection."""


def _make_canonical_compute_and_gather(gather_logprobs, mesh):
  """Builds the one shared rollout/trainer logprob function object."""

  def local_log_softmax(logits):
    return canonical_logsoftmax.log_softmax(logits)

  try:
    mapped_log_softmax = jax.shard_map(
        local_log_softmax,
        mesh=mesh,
        in_specs=jax.sharding.PartitionSpec(None, None),
        out_specs=jax.sharding.PartitionSpec(None, None),
        check_vma=False,
    )
  except TypeError:
    mapped_log_softmax = jax.shard_map(
        local_log_softmax,
        mesh=mesh,
        in_specs=jax.sharding.PartitionSpec(None, None),
        out_specs=jax.sharding.PartitionSpec(None, None),
        check_rep=False,
    )

  def compute_and_gather(logits, next_tokens, max_logprobs):
    logprobs = mapped_log_softmax(logits)
    return gather_logprobs(logprobs, next_tokens, max_logprobs)

  return jax.jit(compute_and_gather, static_argnames=("max_logprobs",))


def _install_shared_logprob_pipeline(
    runner,
    *,
    stock_compute_and_gather,
    gather_logprobs,
    runner_module=None,
    sampling_module=None,
):
  """Installs one default-off canonical scorer at both live lookup sites."""
  if os.environ.get(canonical_logsoftmax.ENV, "") != "1":
    return getattr(
        runner, "_canonical_compute_and_gather_logprobs", stock_compute_and_gather
    )
  if runner_module is None:
    runner_module = importlib.import_module(type(runner).__module__)
  if sampling_module is None:
    sampling_module = importlib.import_module(
        "tpu_inference.layers.jax.sample.sampling"
    )

  canonical = getattr(
      runner_module, "_canonical_logsoftmax_compute_and_gather", None
  )
  runner_stock = getattr(
      runner_module,
      "_canonical_stock_compute_and_gather_logprobs",
      stock_compute_and_gather,
  )
  current_runner = getattr(
      runner_module, "compute_and_gather_logprobs", None
  )
  if current_runner not in (runner_stock, canonical):
    raise FunctionalMappingError(
        "refusing to overwrite an unknown runner logprob implementation"
    )
  current_sampling = getattr(
      sampling_module, "compute_and_gather_logprobs", None
  )
  if current_sampling not in (stock_compute_and_gather, canonical):
    raise FunctionalMappingError(
        "refusing to overwrite an unknown sampling logprob implementation"
    )
  if canonical is None:
    canonical = _make_canonical_compute_and_gather(gather_logprobs, runner.mesh)
    runner_module._canonical_logsoftmax_compute_and_gather = canonical
    runner_module._canonical_stock_compute_and_gather_logprobs = runner_stock

  runner_module.compute_and_gather_logprobs = canonical
  sampling_module.compute_and_gather_logprobs = canonical
  runner._canonical_compute_and_gather_logprobs = canonical
  if not (
      runner_module.compute_and_gather_logprobs
      is sampling_module.compute_and_gather_logprobs
      is runner._canonical_compute_and_gather_logprobs
  ):
    raise FunctionalMappingError("shared canonical logprob identity check failed")
  print(
      "[CANON_ADAPTER] shared canonical logprob pipeline installed "
      "runner_sampling_adapter_same_object=True stages=partial,combine,normalize",
      flush=True,
  )
  return canonical


def _make_processed_target_logprob_vjp(compute_and_gather, max_logprobs):
  """Keeps the exact engine primal while supplying the analytic logp VJP."""

  def exact_value(logits, token_ids):
    return compute_and_gather(
        logits, token_ids, max_logprobs
    ).logprobs[:, 0]

  @jax.custom_vjp
  def target_logprobs(logits, token_ids):
    return exact_value(logits, token_ids)

  def forward(logits, token_ids):
    return exact_value(logits, token_ids), (logits, token_ids)

  def backward(residual, cotangent):
    print(
        "[PATHTRACE] CANON_PROCESSED_LOGPROB_VJP backward",
        flush=True,
    )
    logits, token_ids = residual
    probabilities = jax.nn.softmax(logits, axis=-1)
    selected = jax.nn.one_hot(
        token_ids, logits.shape[-1], dtype=logits.dtype
    )
    d_logits = (selected - probabilities) * cotangent[:, None]
    return d_logits, None

  target_logprobs.defvjp(forward, backward)
  return target_logprobs


@dataclasses.dataclass(frozen=True)
class FunctionalEngineLeaves:
  """Mapped engine leaves and their stable target paths."""

  paths: tuple[str, ...]
  leaves: tuple[jax.Array, ...]
  source_to_target: tuple[tuple[str, str], ...]


@dataclasses.dataclass(frozen=True)
class MappingManifestEntry:
  """One shape-only source-to-target mapping attestation."""

  source_path: str
  target_path: str
  source_shape: tuple[int, ...]
  source_dtype: str
  target_shape: tuple[int, ...]
  target_dtype: str
  mapped_shape: tuple[int, ...]
  mapped_dtype: str


@dataclasses.dataclass(frozen=True)
class MappingManifest:
  """A materialization-free inventory of the real mapping contract."""

  entries: tuple[MappingManifestEntry, ...]
  target_paths: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class LiveEngineContract:
  """JSON-safe attestation for a live in-process engine runner."""

  implementation_id: str
  mapping_entries: int
  target_path_sha256: str
  state_leaves: int
  mesh_shape: tuple[tuple[str, int], ...]
  kv_caches: int
  model_fn: str
  compute_logits_fn: str


class Qwen3EngineForwardAdapter:
  """Differentiable fixed-M Qwen3 forward backed by the live engine module."""

  is_engine_module = True
  supports_value_and_grad = True

  def __init__(
      self,
      *,
      sampler: Any,
      sampling_kwargs: Mapping[str, Any] | None = None,
  ):
    try:
      runner = sampler._model_runner  # pylint: disable=protected-access
    except (AttributeError, RuntimeError) as exc:
      raise FunctionalMappingError("rollout has no live model runner") from exc
    if os.environ.get("CANON_RPA_VJP2", "") != "1":
      raise FunctionalMappingError("canonical adapter requires CANON_RPA_VJP2=1")
    if os.environ.get("CANON_VJP2_MAX_SEQS", "") != "1":
      raise FunctionalMappingError(
          "canonical adapter executes one sequence per model_fn call; "
          "CANON_VJP2_MAX_SEQS must be explicitly 1"
      )
    bucket = int(os.environ.get("CANON_LOGPROB_M", "0"))
    if bucket != 256 or os.environ.get("MIN_TOKEN_BUCKET", "") != "256":
      raise FunctionalMappingError(
          "canonical adapter requires CANON_LOGPROB_M=MIN_TOKEN_BUCKET=256"
      )
    sampling_kwargs = dict(sampling_kwargs or {})
    top_k = sampling_kwargs.get("top_k", 0)
    top_p = sampling_kwargs.get("top_p", 1.0)
    if top_k not in (None, 0, -1) or top_p not in (None, 1.0):
      raise FunctionalMappingError(
          "canonical adapter currently admits only neutral top-k/top-p; "
          f"got top_k={top_k!r}, top_p={top_p!r}"
      )
    required = (
        "state",
        "model_fn",
        "compute_logits_fn",
        "mesh",
        "kv_caches",
        "layer_name_to_kvcache_index",
        "is_first_rank",
        "is_last_rank",
        "vllm_config",
        "max_num_reqs",
        "block_size",
        "model_config",
    )
    missing = [name for name in required if not hasattr(runner, name)]
    if missing:
      raise FunctionalMappingError(
          f"live runner is missing adapter attributes: {missing}"
      )
    if not runner.kv_caches:
      raise FunctionalMappingError("live runner exposes no paged kv caches")
    cache0 = runner.kv_caches[0]
    if cache0.ndim != 5 or int(cache0.shape[1]) != int(runner.block_size):
      raise FunctionalMappingError(
          f"unexpected KV-cache contract: {cache0.shape} block={runner.block_size}"
      )

    from tpu_inference.layers.common.attention_metadata import (  # pylint: disable=g-import-not-at-top
        AttentionMetadata,
    )
    from vllm.forward_context import set_forward_context  # pylint: disable=g-import-not-at-top
    from tpu_inference.layers.jax.sample.sampling import (  # pylint: disable=g-import-not-at-top
        compute_and_gather_logprobs,
        gather_logprobs,
        sample,
    )
    from tpu_inference.layers.jax.sample.sampling_metadata import (  # pylint: disable=g-import-not-at-top
        TPUSupportedSamplingMetadata,
    )

    self.implementation_id = (
        f"{type(runner).__module__}.{type(runner).__qualname__}:"
        "qwen3-canonical-m256-vjp2"
    )
    self._runner = runner
    self._engine_state_contract = runner.state
    self._key_mappings = getattr(sampler, "to_hf_key_mappings", None) or {}
    self._transpose_keys = getattr(sampler, "to_hf_transpose_keys", None)
    self._hook_fns = getattr(sampler, "to_hf_hook_fns", None)
    self._tp_size = int(
        getattr(sampler, "args", {}).get("tensor_parallel_size", 1)
    )
    self._bucket = bucket
    self._max_model_len = int(runner.model_config.max_model_len)
    runner_vocab_size = getattr(runner, "vocab_size", None)
    if runner_vocab_size is None:
      runner_vocab_size = runner.model_config.get_vocab_size()
    self._vocab_size = int(runner_vocab_size)
    self._max_num_reqs = int(runner.max_num_reqs)
    self._block_size = int(runner.block_size)
    self._blocks_per_req = (
        int(runner.model_config.max_model_len) + self._block_size - 1
    ) // self._block_size
    self._cache_shape = (self._blocks_per_req,) + tuple(cache0.shape[1:])
    self._cache_dtype = cache0.dtype
    self._cache_sharding = cache0.sharding
    self._input_sharding = jax.sharding.NamedSharding(
        runner.mesh,
        jax.sharding.PartitionSpec(
            ("data", "attn_dp", "attn_dp_expert"),
        ),
    )
    self._metadata_cls = getattr(
        runner, "_canonical_attention_metadata_cls", AttentionMetadata
    )
    self._set_forward_context = getattr(
        runner, "_canonical_set_forward_context", set_forward_context
    )
    self._sample = getattr(runner, "_canonical_sample", sample)
    self._sampling_metadata_cls = getattr(
        runner,
        "_canonical_sampling_metadata_cls",
        TPUSupportedSamplingMetadata,
    )
    self._compute_and_gather_logprobs = _install_shared_logprob_pipeline(
        runner,
        stock_compute_and_gather=compute_and_gather_logprobs,
        gather_logprobs=gather_logprobs,
    )
    self._max_logprobs = int(runner.model_config.max_logprobs)
    self._processed_target_logprobs = _make_processed_target_logprob_vjp(
        self._compute_and_gather_logprobs, self._max_logprobs
    )
    print(
        "[CANON_ADAPTER] processed-logprob custom VJP installed "
        f"m={self._bucket} max_logprobs={self._max_logprobs}",
        flush=True,
    )
    self._static_kv_indices = tuple(
        runner.layer_name_to_kvcache_index.items()
    )

  def _engine_array(self, value):
    return jax.lax.with_sharding_constraint(value, self._input_sharding)

  def _fresh_caches(self):
    return [
        jax.lax.with_sharding_constraint(
            jnp.zeros(self._cache_shape, self._cache_dtype),
            self._cache_sharding,
        )
        for _ in self._runner.kv_caches
    ]

  def _one_sequence(
      self,
      engine_leaves,
      prompt,
      completion,
      prompt_valid,
      completion_valid,
      pad_id,
      temperature,
      *,
      return_diagnostics=False,
  ):
    """Runs one packed sequence as cache-carried fixed-M engine chunks.

    The static trainer widths may exceed M=256, but every engine call keeps the
    admitted M=256 program.  Prediction targets are indexed from the packed
    *full* sequence so the final row of one chunk predicts the first token of
    the next chunk rather than wrapping within the chunk.
    """
    full = jnp.concatenate((prompt, completion), axis=0)
    valid = jnp.concatenate((prompt_valid, completion_valid), axis=0)
    n_real = jnp.sum(valid, dtype=jnp.int32)
    num_chunks = (full.shape[0] + self._bucket - 1) // self._bucket
    padded_width = num_chunks * self._bucket
    order = jnp.nonzero(valid, size=padded_width, fill_value=0)[0]
    packed_active = jnp.arange(padded_width, dtype=jnp.int32) < n_real
    packed_ids = jnp.where(
        packed_active, full[order], jnp.asarray(0, full.dtype)
    )
    next_ids = jnp.concatenate(
        (packed_ids[1:], jnp.zeros((1,), packed_ids.dtype)), axis=0
    )

    block_tables = jnp.zeros(
        (self._max_num_reqs, self._blocks_per_req), jnp.int32
    )
    block_tables = block_tables.at[0].set(
        jnp.arange(self._blocks_per_req, dtype=jnp.int32)
    )
    request_distribution = jnp.asarray((0, 0, 1), jnp.int32)
    prompt_len = jnp.sum(prompt_valid, dtype=jnp.int32)
    completion_ordinal = jnp.cumsum(completion_valid, dtype=jnp.int32) - 1
    token_positions = prompt_len + completion_ordinal
    source_rows = jnp.clip(token_positions - 1, 0, padded_width - 1)

    sampling_metadata = self._sampling_metadata_cls(
        temperature=self._engine_array(
            jnp.full((self._bucket,), temperature, jnp.float32)
        ),
        top_k=self._engine_array(
            jnp.full((self._bucket,), -1, jnp.int32)
        ),
        top_p=self._engine_array(
            jnp.ones((self._bucket,), jnp.float32)
        ),
        do_sampling=True,
        logprobs=True,
    )
    block_tables_flat = self._engine_array(block_tables.reshape(-1))
    request_distribution = self._engine_array(request_distribution)
    caches = self._fresh_caches()
    chunk_logps = []
    chunk_entropies = []
    if return_diagnostics:
      completion_width = completion.shape[0]
      raw_rows = jnp.zeros(
          (completion_width, self._vocab_size), jnp.float32
      )
      processed_rows = jnp.zeros_like(raw_rows)
      diagnostic_target_ids = jnp.zeros((completion_width,), jnp.int32)
      raw_targets = jnp.zeros((completion_width,), jnp.float32)
      processed_targets = jnp.zeros((completion_width,), jnp.float32)

    print(
        "[PATHTRACE] CANON_ADAPTER_FIXED_M_CHUNKS "
        f"static_width={full.shape[0]} chunks={num_chunks} M={self._bucket}",
        flush=True,
    )
    for chunk_index in range(num_chunks):
      chunk_start = chunk_index * self._bucket
      rows = jnp.arange(self._bucket, dtype=jnp.int32)
      q_len = jnp.clip(n_real - chunk_start, 0, self._bucket)
      kv_len = jnp.minimum(n_real, chunk_start + self._bucket)
      chunk_ids = packed_ids[chunk_start : chunk_start + self._bucket]
      chunk_targets = next_ids[chunk_start : chunk_start + self._bucket]
      positions = jnp.where(rows < q_len, chunk_start + rows, 0)
      query_start = jnp.zeros((self._max_num_reqs + 1,), jnp.int32)
      query_start = query_start.at[1:].set(q_len)
      seq_lens = jnp.zeros((self._max_num_reqs,), jnp.int32)
      seq_lens = seq_lens.at[0].set(kv_len)
      chunk_ids = self._engine_array(chunk_ids)
      chunk_targets = self._engine_array(chunk_targets)
      positions = self._engine_array(positions)
      metadata = self._metadata_cls(
          input_positions=positions,
          block_tables=block_tables_flat,
          seq_lens=self._engine_array(seq_lens),
          query_start_loc=self._engine_array(query_start),
          request_distribution=request_distribution,
      )
      metadata.padded_num_reqs = self._max_num_reqs

      def run_nonempty(active_caches):
        with self._set_forward_context(None, self._runner.vllm_config):
          next_caches, hidden, _, _ = self._runner.model_fn(
              engine_leaves,
              active_caches,
              chunk_ids,
              metadata,
              None,
              positions,
              self._static_kv_indices,
              None,
              None,
              bool(self._runner.is_first_rank),
              bool(self._runner.is_last_rank),
          )
        logits = self._runner.compute_logits_fn(
            engine_leaves, hidden, None
        ).astype(jnp.float32)
        if logits.shape != (self._bucket, self._vocab_size):
          raise FunctionalMappingError(
              "canonical logits shape does not match the admitted fixed-M "
              f"contract: {logits.shape} != "
              f"{(self._bucket, self._vocab_size)}"
          )
        _, processed_logits = self._sample(
            jax.random.PRNGKey(0),
            self._runner.mesh,
            logits,
            sampling_metadata,
        )
        target_logprobs = self._processed_target_logprobs(
            processed_logits, chunk_targets
        )
        normalized = jax.nn.log_softmax(processed_logits, axis=-1)
        probabilities = jnp.exp(normalized)
        entropy_rows = -jnp.sum(
            jnp.where(probabilities > 0, probabilities * normalized, 0.0),
            axis=-1,
        )
        if not return_diagnostics:
          return next_caches, (target_logprobs, entropy_rows)

        local_rows = jnp.clip(
            source_rows - chunk_start, 0, self._bucket - 1
        )
        belongs = (
            completion_valid
            & (source_rows >= chunk_start)
            & (source_rows < chunk_start + self._bucket)
        )
        row_mask = belongs[:, None]
        selected_raw = jnp.take(logits, local_rows, axis=0)
        selected_processed = jnp.take(
            processed_logits, local_rows, axis=0
        )
        selected_target_ids = jnp.take(chunk_targets, local_rows, axis=0)
        selected_raw_targets = jnp.take_along_axis(
            selected_raw, selected_target_ids[:, None], axis=-1
        )[:, 0]
        selected_processed_targets = jnp.take_along_axis(
            selected_processed, selected_target_ids[:, None], axis=-1
        )[:, 0]
        return next_caches, (
            target_logprobs,
            entropy_rows,
            jnp.where(row_mask, selected_raw, 0.0),
            jnp.where(row_mask, selected_processed, 0.0),
            jnp.where(belongs, selected_target_ids, 0),
            jnp.where(belongs, selected_raw_targets, 0.0),
            jnp.where(belongs, selected_processed_targets, 0.0),
        )

      def skip_empty(inactive_caches):
        zero_rows = jnp.zeros((self._bucket,), jnp.float32)
        if not return_diagnostics:
          return inactive_caches, (zero_rows, zero_rows)
        zero_action_rows = jnp.zeros(
            (completion.shape[0], self._vocab_size), jnp.float32
        )
        zero_actions = jnp.zeros((completion.shape[0],), jnp.float32)
        return inactive_caches, (
            zero_rows,
            zero_rows,
            zero_action_rows,
            zero_action_rows,
            jnp.zeros((completion.shape[0],), jnp.int32),
            zero_actions,
            zero_actions,
        )

      caches, chunk_output = jax.lax.cond(
          q_len > 0, run_nonempty, skip_empty, caches
      )
      chunk_logps.append(chunk_output[0])
      chunk_entropies.append(chunk_output[1])
      if return_diagnostics:
        raw_rows = raw_rows + chunk_output[2]
        processed_rows = processed_rows + chunk_output[3]
        diagnostic_target_ids = diagnostic_target_ids + chunk_output[4]
        raw_targets = raw_targets + chunk_output[5]
        processed_targets = processed_targets + chunk_output[6]

    target_logprobs = jnp.concatenate(chunk_logps, axis=0)
    entropy_rows = jnp.concatenate(chunk_entropies, axis=0)
    logps = jnp.take(target_logprobs, source_rows, axis=0)
    entropy = jnp.take(entropy_rows, source_rows, axis=0)
    zeros = jnp.zeros_like(logps)
    masked_logps = jnp.where(completion_valid, logps, zeros)
    masked_entropy = jnp.where(completion_valid, entropy, zeros)
    if not return_diagnostics:
      return masked_logps, masked_entropy

    row_mask = completion_valid[:, None]
    diagnostics = {
        "target_ids": jnp.where(
            completion_valid, diagnostic_target_ids, 0
        ),
        "raw_rows": jnp.where(row_mask, raw_rows, 0.0),
        "processed_rows": jnp.where(row_mask, processed_rows, 0.0),
        "raw_targets": jnp.where(completion_valid, raw_targets, 0.0),
        "processed_targets": jnp.where(
            completion_valid, processed_targets, 0.0
        ),
        "implied_log_normalizers": jnp.where(
            completion_valid, processed_targets - logps, 0.0
        ),
    }
    return masked_logps, masked_entropy, diagnostics

  def compute_per_token_logps(
      self,
      *,
      graphdef,
      state,
      prompt_tokens,
      completion_tokens,
      pad_id,
      eos_id,
      images=None,
      stop_gradient=True,
      return_entropy=False,
      segment_ids=None,
      segment_positions=None,
      temperature=1.0,
      chunk_size=0,
      prompt_mask=None,
      completion_mask=None,
  ):
    """Runs the real engine program with trainer weights and fresh caches."""
    del graphdef, eos_id, segment_positions
    if images is not None:
      raise FunctionalMappingError("canonical Qwen3 adapter is text-only")
    if segment_ids is not None:
      raise FunctionalMappingError(
          "canonical Qwen3 adapter does not yet admit sequence packing"
      )
    if chunk_size:
      raise FunctionalMappingError(
          "canonical engine adapter owns its fixed-M chunking; chunk_size must be 0"
      )
    if prompt_tokens.ndim != 2 or completion_tokens.ndim != 2:
      raise FunctionalMappingError("prompt/completion tokens must be rank 2")
    if prompt_tokens.shape[0] != completion_tokens.shape[0]:
      raise FunctionalMappingError("prompt/completion batch sizes differ")
    if prompt_mask is None:
      prompt_mask = prompt_tokens != pad_id
    else:
      prompt_mask = jnp.asarray(prompt_mask, dtype=jnp.bool_)
    if completion_mask is None:
      completion_mask = completion_tokens != pad_id
    else:
      completion_mask = jnp.asarray(completion_mask, dtype=jnp.bool_)
    if prompt_mask.shape != prompt_tokens.shape:
      raise FunctionalMappingError("prompt mask shape differs from prompt tokens")
    if completion_mask.shape != completion_tokens.shape:
      raise FunctionalMappingError(
          "completion mask shape differs from completion tokens"
      )
    if (
        prompt_tokens.shape[1] + completion_tokens.shape[1]
        > self._max_model_len
    ):
      raise FunctionalMappingError(
          "one sequence exceeds the live engine max-model-length contract: "
          f"{prompt_tokens.shape[1]}+{completion_tokens.shape[1]}"
      )

    model_config = self._runner.model_config
    mapped = map_trainer_state_to_engine_leaves(
        trainer_state=state,
        engine_state_contract=self._engine_state_contract,
        key_mappings=self._key_mappings,
        transpose_keys=self._transpose_keys,
        key_mapping_hook_fns=self._hook_fns,
        num_kv_heads=model_config.get_total_num_kv_heads(),
        head_dim=model_config.get_head_size(),
        tp_size=self._tp_size,
    )

    def body(rows):
      prompt, completion, prompt_valid, completion_valid = rows
      return self._one_sequence(
          mapped.leaves,
          prompt,
          completion,
          prompt_valid,
          completion_valid,
          pad_id,
          temperature,
      )

    logps, entropy = jax.lax.map(
        body, (prompt_tokens, completion_tokens, prompt_mask, completion_mask)
    )
    if stop_gradient:
      logps = jax.lax.stop_gradient(logps)
      entropy = jax.lax.stop_gradient(entropy)
    if return_entropy:
      return logps, entropy
    return logps

  def compute_per_token_diagnostics(
      self,
      *,
      graphdef,
      state,
      prompt_tokens,
      completion_tokens,
      pad_id,
      eos_id,
      images=None,
      segment_ids=None,
      segment_positions=None,
      temperature=1.0,
      chunk_size=0,
      prompt_mask=None,
      completion_mask=None,
  ):
    """Diagnostic-only forward that exports already-live action-logit rows."""
    del graphdef, eos_id, segment_positions
    if os.environ.get("CANON_L3_A3_DIAG", "") != "1":
      raise FunctionalMappingError(
          "compute_per_token_diagnostics requires CANON_L3_A3_DIAG=1"
      )
    if images is not None:
      raise FunctionalMappingError("canonical Qwen3 adapter is text-only")
    if segment_ids is not None:
      raise FunctionalMappingError(
          "canonical Qwen3 adapter does not yet admit sequence packing"
      )
    if chunk_size:
      raise FunctionalMappingError(
          "canonical engine adapter owns its fixed-M chunking; chunk_size must be 0"
      )
    if prompt_tokens.ndim != 2 or completion_tokens.ndim != 2:
      raise FunctionalMappingError("prompt/completion tokens must be rank 2")
    if prompt_tokens.shape[0] != completion_tokens.shape[0]:
      raise FunctionalMappingError("prompt/completion batch sizes differ")
    if prompt_mask is None:
      prompt_mask = prompt_tokens != pad_id
    else:
      prompt_mask = jnp.asarray(prompt_mask, dtype=jnp.bool_)
    if completion_mask is None:
      completion_mask = completion_tokens != pad_id
    else:
      completion_mask = jnp.asarray(completion_mask, dtype=jnp.bool_)
    if prompt_mask.shape != prompt_tokens.shape:
      raise FunctionalMappingError("prompt mask shape differs from prompt tokens")
    if completion_mask.shape != completion_tokens.shape:
      raise FunctionalMappingError(
          "completion mask shape differs from completion tokens"
      )
    if (
        prompt_tokens.shape[1] + completion_tokens.shape[1]
        > self._max_model_len
    ):
      raise FunctionalMappingError(
          "one sequence exceeds the live engine max-model-length contract: "
          f"{prompt_tokens.shape[1]}+{completion_tokens.shape[1]}"
      )

    model_config = self._runner.model_config
    mapped = map_trainer_state_to_engine_leaves(
        trainer_state=state,
        engine_state_contract=self._engine_state_contract,
        key_mappings=self._key_mappings,
        transpose_keys=self._transpose_keys,
        key_mapping_hook_fns=self._hook_fns,
        num_kv_heads=model_config.get_total_num_kv_heads(),
        head_dim=model_config.get_head_size(),
        tp_size=self._tp_size,
    )

    def body(rows):
      prompt, completion, prompt_valid, completion_valid = rows
      return self._one_sequence(
          mapped.leaves,
          prompt,
          completion,
          prompt_valid,
          completion_valid,
          pad_id,
          temperature,
          return_diagnostics=True,
      )

    return jax.lax.map(
        body, (prompt_tokens, completion_tokens, prompt_mask, completion_mask)
    )


def _flat_path(path: Sequence[Any]) -> str:
  return ".".join(str(part) for part in path)


def _mapping_pairs(*, trainer_state, engine_state_contract, key_mappings):
  target_flat = list(engine_state_contract.flat_state())
  source_to_target_contract = generate_utils.build_flat_dict(
      target_flat, dict(key_mappings)
  )
  source_paths = tuple(
      _flat_path(path)
      for path, _ in trainer_state.flat_state()
      if "rng" not in _flat_path(path)
  )
  missing_source = sorted(
      path for path in source_paths if path not in source_to_target_contract
  )
  if missing_source:
    raise FunctionalMappingError(
        "trainer leaves missing canonical engine mappings: "
        f"{missing_source}"
    )
  unrolled = generate_utils._unroll_scanned_layers(  # pylint: disable=protected-access
      trainer_state, source_to_target_contract
  )
  return target_flat, unrolled


def _transform_value(
    value,
    *,
    source_path,
    target_param,
    transpose_keys,
    key_mapping_hook_fns,
    rollout_engine,
    shape_kwargs,
):
  target_value = getattr(target_param, "value", target_param)
  value = generate_utils._apply_transpose(  # pylint: disable=protected-access
      value, source_path, transpose_keys, rollout_engine
  )
  if key_mapping_hook_fns and source_path in key_mapping_hook_fns:
    value = key_mapping_hook_fns[source_path](value)
  value = generate_utils._align_shape(  # pylint: disable=protected-access
      value,
      target_value.shape,
      source_path,
      rollout_engine,
      **shape_kwargs,
  )
  return generate_utils._apply_dtype_cast(  # pylint: disable=protected-access
      value, target_value.dtype, source_path
  )


def inspect_trainer_state_to_engine_contract(
    *,
    trainer_state: Any,
    engine_state_contract: Any,
    key_mappings: Mapping[str, tuple[str, tuple[str | None, ...]]],
    transpose_keys: Mapping[str, tuple[int, ...]] | None = None,
    key_mapping_hook_fns: Mapping[str, Any] | None = None,
    rollout_engine: str = "vllm_jax",
    **shape_kwargs: Any,
) -> MappingManifest:
  """Checks the real mapping inventory without allocating mapped weights."""
  target_flat, unrolled = _mapping_pairs(
      trainer_state=trainer_state,
      engine_state_contract=engine_state_contract,
      key_mappings=key_mappings,
  )
  entries_by_target: dict[str, MappingManifestEntry] = {}
  for (source_path, target_path), (source_value, target_param) in unrolled.items():
    if target_path in entries_by_target:
      raise FunctionalMappingError(
          f"canonical engine target written more than once: {target_path}"
      )
    target_value = getattr(target_param, "value", target_param)
    source_spec = jax.ShapeDtypeStruct(source_value.shape, source_value.dtype)
    mapped_spec = jax.eval_shape(
        lambda value: _transform_value(
            value,
            source_path=source_path,
            target_param=target_param,
            transpose_keys=transpose_keys,
            key_mapping_hook_fns=key_mapping_hook_fns,
            rollout_engine=rollout_engine,
            shape_kwargs=shape_kwargs,
        ),
        source_spec,
    )
    entry = MappingManifestEntry(
        source_path=source_path,
        target_path=target_path,
        source_shape=tuple(source_value.shape),
        source_dtype=str(source_value.dtype),
        target_shape=tuple(target_value.shape),
        target_dtype=str(target_value.dtype),
        mapped_shape=tuple(mapped_spec.shape),
        mapped_dtype=str(mapped_spec.dtype),
    )
    if (
        entry.mapped_shape != entry.target_shape
        or entry.mapped_dtype != entry.target_dtype
    ):
      raise FunctionalMappingError(
          "abstract mapped leaf does not match engine contract: "
          f"{entry}"
      )
    entries_by_target[target_path] = entry

  target_paths = tuple(_flat_path(path) for path, _ in target_flat)
  missing_target = sorted(
      path for path in target_paths if path not in entries_by_target
  )
  if missing_target:
    raise FunctionalMappingError(
        "canonical engine mapping is not target-complete: "
        f"missing={missing_target}"
    )
  return MappingManifest(
      entries=tuple(entries_by_target[path] for path in target_paths),
      target_paths=target_paths,
  )


def inspect_live_engine_contract(
    *, sampler: Any, trainer_state: Any
) -> LiveEngineContract:
  """Fail-closed A1b/A2 inspection without materializing mapped weights."""
  if os.environ.get("CANON_RPA_VJP2", "") != "1":
    raise FunctionalMappingError(
        "canonical engine contract requires CANON_RPA_VJP2=1"
    )
  try:
    runner = sampler._model_runner  # pylint: disable=protected-access
  except (AttributeError, RuntimeError) as exc:
    raise FunctionalMappingError("rollout has no live model runner") from exc

  required = (
      "state",
      "state_leaves",
      "model_fn",
      "compute_logits_fn",
      "mesh",
      "kv_caches",
      "layer_name_to_kvcache_index",
      "is_first_rank",
      "is_last_rank",
  )
  missing = [name for name in required if not hasattr(runner, name)]
  if missing:
    raise FunctionalMappingError(
        f"live tpu_inference runner is missing contract attributes: {missing}"
    )
  if not callable(runner.model_fn) or not callable(runner.compute_logits_fn):
    raise FunctionalMappingError("engine model_fn/compute_logits_fn is not callable")
  if not isinstance(runner.mesh, jax.sharding.Mesh):
    raise FunctionalMappingError("engine runner mesh is not a jax.sharding.Mesh")
  if not runner.kv_caches:
    raise FunctionalMappingError("engine runner exposes no paged kv caches")

  model_config = getattr(runner, "model_config", None)
  if model_config is None:
    raise FunctionalMappingError("engine runner exposes no model_config")
  manifest = inspect_trainer_state_to_engine_contract(
      trainer_state=trainer_state,
      engine_state_contract=runner.state,
      key_mappings=getattr(sampler, "to_hf_key_mappings", None) or {},
      transpose_keys=getattr(sampler, "to_hf_transpose_keys", None),
      key_mapping_hook_fns=getattr(sampler, "to_hf_hook_fns", None),
      num_kv_heads=model_config.get_total_num_kv_heads(),
      head_dim=model_config.get_head_size(),
      tp_size=getattr(sampler, "args", {}).get("tensor_parallel_size", 1),
  )

  runner_leaves = tuple(runner.state_leaves)
  state_leaves = tuple(jax.tree_util.tree_leaves(runner.state))
  if len(runner_leaves) != len(state_leaves):
    raise FunctionalMappingError(
        "runner.state_leaves length disagrees with runner.state: "
        f"{len(runner_leaves)} != {len(state_leaves)}"
    )
  for index, (declared, actual) in enumerate(zip(runner_leaves, state_leaves)):
    if declared.shape != actual.shape or declared.dtype != actual.dtype:
      raise FunctionalMappingError(
          "runner.state_leaves order/contract mismatch at index "
          f"{index}: {(declared.shape, declared.dtype)} != "
          f"{(actual.shape, actual.dtype)}"
      )

  path_digest = hashlib.sha256(
      "\n".join(manifest.target_paths).encode("utf-8")
  ).hexdigest()
  return LiveEngineContract(
      implementation_id=(
          f"{type(runner).__module__}.{type(runner).__qualname__}:qwen3"
      ),
      mapping_entries=len(manifest.entries),
      target_path_sha256=path_digest,
      state_leaves=len(state_leaves),
      mesh_shape=tuple(
          (str(name), int(size)) for name, size in runner.mesh.shape.items()
      ),
      kv_caches=len(runner.kv_caches),
      model_fn=getattr(runner.model_fn, "__name__", type(runner.model_fn).__name__),
      compute_logits_fn=getattr(
          runner.compute_logits_fn,
          "__name__",
          type(runner.compute_logits_fn).__name__,
      ),
  )


def map_trainer_state_to_engine_leaves(
    *,
    trainer_state: Any,
    engine_state_contract: Any,
    key_mappings: Mapping[str, tuple[str, tuple[str | None, ...]]],
    transpose_keys: Mapping[str, tuple[int, ...]] | None = None,
    key_mapping_hook_fns: Mapping[str, Any] | None = None,
    rollout_engine: str = "vllm_jax",
    **shape_kwargs: Any,
) -> FunctionalEngineLeaves:
  """Purely maps trainer parameters into the engine state's leaf contract.

  ``engine_state_contract`` is inspected only for target paths, shapes, dtypes,
  and order.  Its values are never assigned or returned.  All transforms of
  trainer values are JAX operations, so gradients flow back through casts,
  transposes, reshapes, padding, and repetition.

  The function fails closed on every non-RNG trainer leaf without a mapping,
  every engine target leaf not produced exactly once, and duplicate writes.
  That strictness is intentional for the canonical training path; serving's
  best-effort warning semantics are not sufficient evidence here.
  """
  target_flat, unrolled = _mapping_pairs(
      trainer_state=trainer_state,
      engine_state_contract=engine_state_contract,
      key_mappings=key_mappings,
  )
  mapped: dict[str, jax.Array] = {}
  provenance: list[tuple[str, str]] = []
  for (source_path, target_path), (value, target_param) in unrolled.items():
    if target_path in mapped:
      raise FunctionalMappingError(
          f"canonical engine target written more than once: {target_path}"
      )
    value = _transform_value(
        value,
        source_path=source_path,
        target_param=target_param,
        transpose_keys=transpose_keys,
        key_mapping_hook_fns=key_mapping_hook_fns,
        rollout_engine=rollout_engine,
        shape_kwargs=shape_kwargs,
    )
    mapped[target_path] = value
    provenance.append((source_path, target_path))

  target_paths = tuple(_flat_path(path) for path, _ in target_flat)
  missing_target = sorted(path for path in target_paths if path not in mapped)
  unexpected_target = sorted(path for path in mapped if path not in target_paths)
  if missing_target or unexpected_target:
    raise FunctionalMappingError(
        "canonical engine mapping is not target-complete: "
        f"missing={missing_target}, unexpected={unexpected_target}"
    )

  return FunctionalEngineLeaves(
      paths=target_paths,
      leaves=tuple(mapped[path] for path in target_paths),
      source_to_target=tuple(sorted(provenance)),
  )
