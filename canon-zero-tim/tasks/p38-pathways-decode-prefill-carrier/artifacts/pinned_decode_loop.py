# Copyright 2026 Google LLC
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

import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from vllm.v1.outputs import LogprobsTensors

from tpu_inference.layers.common.attention_metadata import AttentionMetadata


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "current_tokens",
        "active_mask",
        "attn_metadata",
        "step_counter",
    ],
    meta_fields=[],
)
@dataclass
class TpuSamplingState:
    # (batch_size,)
    current_tokens: jax.Array
    # (batch_size,)
    active_mask: jax.Array
    # AttentionMetadata PyTree
    attn_metadata: AttentionMetadata
    # scalar
    step_counter: jax.Array


@functools.partial(
    jax.jit,
    static_argnames=["eos_token_id", "padding_token_id", "dp_size", "pad_len"])
def _update_loop_state(
    next_tokens: jax.Array,
    active_mask: jax.Array,
    input_positions: jax.Array,
    seq_lens: jax.Array,
    eos_token_id: tuple[int, ...],
    padding_token_id: int,
    dp_size: int,
    pad_len: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    eos_arr = jnp.atleast_1d(jnp.array(eos_token_id, dtype=jnp.int32))
    is_eos = jnp.any(next_tokens[:, None] == eos_arr[None, :], axis=-1)
    new_active_mask = jnp.logical_and(active_mask, jnp.logical_not(is_eos))
    next_input_ids = jnp.where(new_active_mask, next_tokens, padding_token_id)
    increment = new_active_mask.astype(jnp.int32)
    new_positions = input_positions + increment

    # Fix DP padding bug by exposing DP dimension as 2D before padding
    increment_2d = increment.reshape(dp_size, -1)
    seq_lens_2d = seq_lens.reshape(dp_size, -1)
    if pad_len > 0:
        padded_increment_2d = jnp.pad(increment_2d, ((0, 0), (0, pad_len)))
    else:
        padded_increment_2d = increment_2d[:, :seq_lens_2d.shape[1]]
    new_seq_lens = (seq_lens_2d + padded_increment_2d).ravel()

    # Compute the token to record in generated_tokens
    step_record_tokens = jnp.where(active_mask, next_tokens, padding_token_id)

    any_hit_eos = jnp.any(jnp.logical_and(active_mask, is_eos))

    return new_active_mask, next_input_ids, new_positions, new_seq_lens, step_record_tokens, any_hit_eos


@functools.partial(jax.jit, static_argnums=(1, ))
def _split_rngs(rng, static_size, dynamic_size):
    all_rngs = jax.random.split(rng, static_size + 1)
    # Keep the per-step keys as an array (not a Python tuple): the decode loop
    # is a lax.while_loop, so step keys are indexed by a *traced* step counter,
    # which requires array indexing.
    return all_rngs[:static_size], all_rngs[dynamic_size]


@functools.partial(
    jax.jit,
    static_argnames=(
        "model_fn",
        "compute_logits_fn",
        "sample_fn",
        "mesh",
        "static_max_decode_steps",
        "eos_token_id",
        "padding_token_id",
        "dp_size",
        "pad_len",
        "has_experts",
        "expert_shape",
        "expert_dtype",
        "layer_name_to_kvcache_index",
        "is_first_rank",
        "is_last_rank",
        "max_logprobs",
        "logprobs_mode",
    ),
    donate_argnames=("kv_caches", ),
    # Hoisted here from the model's step_fun: JAX forbids compiler_options on
    # a nested jit, so they must live on this top-level loop jit instead.
    compiler_options={
        "xla_tpu_all_gather_collective_matmul_mode": "post_spmd_conservative",
        "xla_tpu_reduce_scatter_collective_matmul_mode":
        "post_spmd_conservative",
        "xla_tpu_use_minor_sharding_for_major_trivial_input": "true"
    },
)
def _decode_core(
    *,
    state,
    kv_caches,
    step_rngs,
    sampling_metadata,
    inputs_embeds,
    lora_metadata,
    intermediate_tensors,
    block_tables,
    query_start_loc,
    request_distribution,
    mamba_state_indices,
    current_tokens,
    active_mask,
    input_positions,
    seq_lens,
    model_fn,
    compute_logits_fn,
    sample_fn,
    mesh,
    max_decode_steps,
    static_max_decode_steps,
    eos_token_id,
    padding_token_id,
    dp_size,
    pad_len,
    has_experts,
    expert_shape,
    expert_dtype,
    layer_name_to_kvcache_index,
    is_first_rank,
    is_last_rank,
    max_logprobs,
    logprobs_mode,
):
    has_logprobs = False if sampling_metadata is None else sampling_metadata.logprobs

    def _run_one_step(step_idx, ct, am, pos, sl, kvc):
        step_rng = step_rngs[step_idx]
        attn_metadata = AttentionMetadata(
            input_positions=pos,
            block_tables=block_tables,
            seq_lens=sl,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            mamba_state_indices=mamba_state_indices,
        )
        kvc, hidden_states, _, expert_indices_step = model_fn(
            state,
            kvc,
            ct,
            attn_metadata,
            inputs_embeds,
            attn_metadata.input_positions,
            layer_name_to_kvcache_index,
            lora_metadata,
            intermediate_tensors,
            is_first_rank,
            is_last_rank,
        )
        logits = compute_logits_fn(state, hidden_states, None)
        logits = logits.astype(jnp.float32)
        next_tokens, processed_logits = sample_fn(step_rng, mesh, logits,
                                                  sampling_metadata)
        (new_active_mask, next_input_ids, new_positions, new_seq_lens,
         step_record_tokens, any_hit_eos) = _update_loop_state(
             next_tokens,
             am,
             pos,
             sl,
             eos_token_id,
             padding_token_id,
             dp_size,
             pad_len,
         )

        lp_ids_step = None
        lp_val_step = None
        lp_ranks_step = None
        if has_logprobs:
            logprobs_logits = (processed_logits if logprobs_mode
                               == "processed_logprobs" else logits)
            from tpu_inference.layers.jax.sample.sampling import \
                compute_and_gather_logprobs
            step_logprobs = compute_and_gather_logprobs(
                logprobs_logits, next_tokens, max_logprobs)
            lp_ids_step = step_logprobs.logprob_token_ids
            lp_val_step = step_logprobs.logprobs
            lp_ranks_step = step_logprobs.selected_token_ranks

        return (next_input_ids, new_active_mask, new_positions, new_seq_lens,
                kvc, step_record_tokens, expert_indices_step, lp_ids_step,
                lp_val_step, lp_ranks_step, any_hit_eos)

    batch_size = current_tokens.shape[0]
    token_buffer = jnp.full((static_max_decode_steps, batch_size),
                            padding_token_id,
                            dtype=current_tokens.dtype)
    expert_buffer = None
    if has_experts:
        expert_buffer = jnp.zeros((static_max_decode_steps, ) + expert_shape,
                                  dtype=expert_dtype)

    logprob_token_ids_buffer = None
    logprobs_buffer = None
    selected_token_ranks_buffer = None
    if has_logprobs:
        logprob_token_ids_buffer = jnp.zeros(
            (static_max_decode_steps, batch_size, max_logprobs + 1),
            dtype=jnp.int32)
        logprobs_buffer = jnp.zeros(
            (static_max_decode_steps, batch_size, max_logprobs + 1),
            dtype=jnp.float32)
        selected_token_ranks_buffer = jnp.zeros(
            (static_max_decode_steps, batch_size), dtype=jnp.int32)

    def _pack(i, ct, am, pos, sl, kvc, tb, eb, lp_ids, lp_val, lp_ranks, eos):
        base = [i, ct, am, pos, sl, kvc, tb]
        if has_experts:
            base.append(eb)
        if has_logprobs:
            base.extend([lp_ids, lp_val, lp_ranks])
        base.append(eos)
        return tuple(base)

    def _unpack(carry):
        i, ct, am, pos, sl, kvc, tb = carry[:7]
        idx = 7
        eb = None
        if has_experts:
            eb = carry[idx]
            idx += 1

        lp_ids = None
        lp_val = None
        lp_ranks = None
        if has_logprobs:
            lp_ids = carry[idx]
            lp_val = carry[idx + 1]
            lp_ranks = carry[idx + 2]
            idx += 3

        eos = carry[idx]
        return i, ct, am, pos, sl, kvc, tb, eb, lp_ids, lp_val, lp_ranks, eos

    def cond_fn(carry):
        i = carry[0]
        eos_flag = carry[-1]
        not_done = i < max_decode_steps
        return jnp.logical_and(not_done, jnp.logical_not(eos_flag))

    def body_fn(carry):
        (i, ct, am, pos, sl, kvc, tb, eb, lp_ids_buf, lp_val_buf, lp_ranks_buf,
         eos_flag) = _unpack(carry)
        (next_ct, new_mask, new_pos, new_sl, kvc, rec_tokens, experts,
         lp_ids_step, lp_val_step, lp_ranks_step,
         hit) = _run_one_step(i, ct, am, pos, sl, kvc)
        tb = tb.at[i].set(rec_tokens)
        if has_experts:
            eb = eb.at[i].set(experts)
        if has_logprobs:
            lp_ids_buf = lp_ids_buf.at[i].set(lp_ids_step)
            lp_val_buf = lp_val_buf.at[i].set(lp_val_step)
            lp_ranks_buf = lp_ranks_buf.at[i].set(lp_ranks_step)
        return _pack(i + 1, next_ct, new_mask, new_pos, new_sl, kvc, tb, eb,
                     lp_ids_buf, lp_val_buf, lp_ranks_buf,
                     jnp.logical_or(eos_flag, hit))

    init_carry = _pack(
        jnp.array(0, dtype=jnp.int32),
        current_tokens,
        active_mask,
        input_positions,
        seq_lens,
        kv_caches,
        token_buffer,
        expert_buffer if has_experts else None,
        logprob_token_ids_buffer,
        logprobs_buffer,
        selected_token_ranks_buffer,
        jnp.array(False),
    )
    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    (step_idx_final, current_tokens, active_mask, positions, seq_lens,
     kv_caches, token_buffer, expert_buffer, lp_ids_buffer, lp_val_buffer,
     lp_ranks_buffer, _) = _unpack(final_carry)

    return (step_idx_final, current_tokens, active_mask, positions, seq_lens,
            kv_caches, token_buffer, expert_buffer, lp_ids_buffer,
            lp_val_buffer, lp_ranks_buffer)


def continue_decode(
    state: dict,
    model_fn: Callable,
    compute_logits_fn: Callable,
    sample_fn: Callable,
    init_state: TpuSamplingState,
    kv_caches: Any,
    max_decode_steps: int,
    static_max_decode_steps: int,
    eos_token_id: tuple[int, ...],
    padding_token_id: int,
    rng: jax.Array,
    *,
    mesh: Any,
    sampling_metadata: Any,
    inputs_embeds: jax.Array | None = None,
    layer_name_to_kvcache_index: tuple[tuple[str, int], ...] = (),
    lora_metadata: dict | None = None,
    intermediate_tensors: dict[str, jax.Array] | None = None,
    is_first_rank: bool = True,
    is_last_rank: bool = True,
    dp_size: int = 1,
    collect_expert_indices: bool = False,
    max_logprobs: int = 0,
    logprobs_mode: str = "raw",
) -> tuple[jax.Array, Any, TpuSamplingState, jax.Array, jax.Array | None,
           Optional["LogprobsTensors"]]:
    """Run the TPU decode loop as one fused, kv-cache-donating program.

    Args:
      state: Model state dict (weights; passed through, not donated).
      model_fn: Stable model forward callable.
      compute_logits_fn: Stable logits callable.
      sample_fn: Stable sampling callable with signature
        (rng, mesh, logits, sampling_metadata) -> (next_tokens, _). Must be a
        stable object (not a per-call closure) so the jit cache persists;
        per-call sampling data is threaded via `sampling_metadata`.
      init_state: Initial TpuSamplingState.
      kv_caches: KV caches. Donated into the fused loop and returned updated.
      max_decode_steps: Max steps to run (static loop bound).
      static_max_decode_steps: Static maximum steps for RNG splitting.
      eos_token_id: EOS token ID(s).
      padding_token_id: Padding token ID.
      rng: RNG key.
      mesh: Device mesh (static; stable runner object).
      sampling_metadata: Per-call sampling metadata pytree (traced).
      inputs_embeds: Optional input embeddings.
      layer_name_to_kvcache_index: Mapping from layer name to KV cache index.
      lora_metadata: Optional LoRA metadata.
      intermediate_tensors: Optional intermediate tensors.
      is_first_rank: Whether this is the first PP rank.
      is_last_rank: Whether this is the last PP rank.
      dp_size: Data parallel size.
      collect_expert_indices: Whether model_fn returns routed-expert indices
        (caller derives this from
        vllm_config.model_config.enable_return_routed_experts). When True the
        expert-indices shape is discovered via jax.eval_shape (no execution)
        to presize the accumulation buffer.
      max_logprobs: Minimum number of logprobs to retain per token.
      logprobs_mode: Logprobs mode from model config ("raw" or "processed_logprobs").

    Returns:
      Tuple of (generated_tokens, final_kv_caches, final_state, final_rng,
      all_expert_indices, logprobs_tensors). generated_tokens is a fixed-size
      (max_decode_steps, batch_size) array and all_expert_indices, when not
      None, is (max_decode_steps, ...); rows beyond final_state.step_counter
      are padding (early EOS exit may stop before max_decode_steps), so the
      caller must trim with final_state.step_counter.
    """

    batch_size = init_state.current_tokens.shape[0]
    seq_lens_size = init_state.attn_metadata.seq_lens.shape[0]
    pad_len = (seq_lens_size - batch_size) // dp_size

    step_rngs, current_rng = _split_rngs(rng, static_max_decode_steps,
                                         max_decode_steps)

    attn = init_state.attn_metadata

    # Discover the per-step expert-indices shape without executing a step.
    # Gated by the caller's config flag so the abstract trace is skipped for
    # the common non-MoE path. eval_shape does no execution/compile/HBM work
    # and does not consume the (donatable) kv_caches.
    has_experts = False
    expert_shape = None
    expert_dtype = None
    if collect_expert_indices:

        def _model_experts_only(current_tokens, input_positions, seq_lens,
                                kv_caches):
            am = AttentionMetadata(
                input_positions=input_positions,
                block_tables=attn.block_tables,
                seq_lens=seq_lens,
                query_start_loc=attn.query_start_loc,
                request_distribution=attn.request_distribution,
                mamba_state_indices=attn.mamba_state_indices,
            )
            _, _, _, experts = model_fn(
                state,
                kv_caches,
                current_tokens,
                am,
                inputs_embeds,
                am.input_positions,
                layer_name_to_kvcache_index,
                lora_metadata,
                intermediate_tensors,
                is_first_rank,
                is_last_rank,
            )
            return experts

        expert_struct = jax.eval_shape(
            _model_experts_only,
            init_state.current_tokens,
            attn.input_positions,
            attn.seq_lens,
            kv_caches,
        )
        if expert_struct is not None:
            has_experts = True
            expert_shape = tuple(expert_struct.shape)
            expert_dtype = expert_struct.dtype

    (step_counter, current_tokens, active_mask, positions, seq_lens, kv_caches,
     token_buffer, expert_buffer, lp_ids_buffer, lp_val_buffer,
     lp_ranks_buffer) = _decode_core(
         state=state,
         kv_caches=kv_caches,
         step_rngs=step_rngs,
         sampling_metadata=sampling_metadata,
         inputs_embeds=inputs_embeds,
         lora_metadata=lora_metadata,
         intermediate_tensors=intermediate_tensors,
         block_tables=attn.block_tables,
         query_start_loc=attn.query_start_loc,
         request_distribution=attn.request_distribution,
         mamba_state_indices=attn.mamba_state_indices,
         current_tokens=init_state.current_tokens,
         active_mask=init_state.active_mask,
         input_positions=attn.input_positions,
         seq_lens=attn.seq_lens,
         model_fn=model_fn,
         compute_logits_fn=compute_logits_fn,
         sample_fn=sample_fn,
         mesh=mesh,
         max_decode_steps=max_decode_steps,
         static_max_decode_steps=static_max_decode_steps,
         eos_token_id=eos_token_id,
         padding_token_id=padding_token_id,
         dp_size=dp_size,
         pad_len=pad_len,
         has_experts=has_experts,
         expert_shape=expert_shape,
         expert_dtype=expert_dtype,
         layer_name_to_kvcache_index=layer_name_to_kvcache_index,
         is_first_rank=is_first_rank,
         is_last_rank=is_last_rank,
         max_logprobs=max_logprobs,
         logprobs_mode=logprobs_mode,
     )

    final_state = TpuSamplingState(
        current_tokens=current_tokens,
        active_mask=active_mask,
        attn_metadata=AttentionMetadata(
            input_positions=positions,
            block_tables=attn.block_tables,
            seq_lens=seq_lens,
            query_start_loc=attn.query_start_loc,
            request_distribution=attn.request_distribution,
            mamba_state_indices=attn.mamba_state_indices,
        ),
        step_counter=step_counter.astype(jnp.int32),
    )

    all_expert_indices = expert_buffer if has_experts else None

    logprobs_tensors = None
    if sampling_metadata is not None and sampling_metadata.logprobs:
        logprobs_tensors = LogprobsTensors(
            logprob_token_ids=lp_ids_buffer,
            logprobs=lp_val_buffer,
            selected_token_ranks=lp_ranks_buffer,
        )

    return (token_buffer, kv_caches, final_state, current_rng,
            all_expert_indices, logprobs_tensors)
