# Copyright 2025 Google LLC
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

"""vLLM rollout worker with Tunix sampler."""

import dataclasses
from typing import Any, Dict, Optional, Tuple

from flax import nnx
import jax
import jaxtyping
import numpy as np
import os
from tunix.generate import mappings
from tunix.generate import utils as generate_utils
from tunix.generate import vllm_sampler
from tunix.rl.rollout import base_rollout
# vllm_sampler already hard-imports vllm at module scope, so these add no new dependency.
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams


class VllmRollout(base_rollout.BaseRollout):
  """vLLM rollout worker."""

  def __init__(
      self,
      model: Any,
      tokenizer: Any,
      cache_config_or_size: base_rollout.CacheConfig | int,
      mesh: jax.sharding.Mesh,
      rollout_config: base_rollout.RolloutConfig,
  ):
    mapping_config = mappings.MappingConfig.build(
        mapping_obj=rollout_config.rollout_mapping_config,
        model=model,
        backend="vllm_jax",
    )
    self._sampler = vllm_sampler.VllmSampler(
        tokenizer=tokenizer,
        config=vllm_sampler.VllmConfig(
            server_mode=rollout_config.rollout_vllm_server_mode,
            server_mode_submission_threshold=(
              rollout_config.rollout_vllm_server_mode_submission_threshold
            ),
            server_mode_submission_timeout_s=(
              rollout_config.rollout_vllm_server_mode_submission_timeout_s
            ),
            mapping_config=mapping_config,
            return_logprobs=rollout_config.return_logprobs,
            init_with_random_weights=rollout_config.rollout_vllm_init_with_random_weights,
            tpu_backend_type=rollout_config.rollout_vllm_tpu_backend_type,
            additional_config=rollout_config.rollout_vllm_additional_config,
            enable_dp_attention=rollout_config.rollout_vllm_enable_dp_attention,
            hbm_utilization=rollout_config.rollout_vllm_hbm_utilization,
            lora_config=rollout_config.rollout_vllm_lora_config,
            mesh=mesh,
            tensor_parallel_size=rollout_config.tensor_parallel_size,
            data_parallel_size=rollout_config.data_parallel_size,
            expert_parallel_size=rollout_config.expert_parallel_size,
            delete_dst_buffers=rollout_config.rollout_vllm_delete_dst_buffers,
            reshard_chunk_size=rollout_config.rollout_vllm_reshard_chunk_size,
            engine_kwargs={
                "model": rollout_config.rollout_vllm_model_version,
                "max_model_len": cache_config_or_size,
                "async_scheduling": (
                    rollout_config.rollout_vllm_async_scheduling
                ),
                "max_num_batched_tokens": (
                    rollout_config.rollout_vllm_max_num_batched_tokens
                ),
                "max_num_seqs": rollout_config.rollout_vllm_max_num_seqs,
                "hf_config_path": rollout_config.rollout_vllm_hf_config_path,
                "max_logprobs": (
                    1
                ),  # We only need the logprobs of the sampled tokens
                "logprobs_mode": rollout_config.rollout_vllm_logprobs_mode,
                **rollout_config.rollout_vllm_kwargs,
            },
            sampling_kwargs=rollout_config.rollout_vllm_sampling_kwargs,
        ),
    )
    self._rollout_sampling_kwargs = dict(
        rollout_config.rollout_vllm_sampling_kwargs
    )
    self._last_sampling_transforms: dict[str, Any] | None = None
    self._last_prefill_rescore_provenance: dict[str, Any] | None = None
    self._last_grouped_prefill_rescore_provenance: dict[str, Any] | None = None
    state = nnx.state(model)
    self._sampler.load_checkpoint(state)
    self._canonical_engine_contract = None
    self._canonical_engine_adapter = None
    if os.environ.get("CANON_ENGINE_MODULE_C", "") == "1":
      from tunix.rl import canonical_qwen3_adapter  # pylint: disable=g-import-not-at-top

      self._canonical_engine_contract = (
          canonical_qwen3_adapter.inspect_live_engine_contract(
              sampler=self._sampler, trainer_state=state
          )
      )
      print(
          "[CANON_ADAPTER] live engine contract "
          f"{dataclasses.asdict(self._canonical_engine_contract)}",
          flush=True,
      )
      from tunix.rl import canonical_forward  # pylint: disable=g-import-not-at-top

      adapter_sampling_kwargs = {
          "temperature": rollout_config.temperature,
          "top_k": rollout_config.top_k,
          "top_p": rollout_config.top_p,
      }
      adapter_sampling_kwargs.update(self._rollout_sampling_kwargs)
      adapter = canonical_qwen3_adapter.Qwen3EngineForwardAdapter(
          sampler=self._sampler,
          sampling_kwargs=adapter_sampling_kwargs,
      )
      self._canonical_engine_adapter = adapter
      canonical_forward.register(adapter)
      print(
          "[CANON_ADAPTER] differentiable engine adapter registered "
          f"{canonical_forward.attestation()}",
          flush=True,
      )

  def canonical_engine_contract_attestation(self) -> dict[str, Any]:
    """Returns the admitted A1b/A2 contract; never fabricates one."""
    if self._canonical_engine_contract is None:
      raise RuntimeError(
          "canonical engine contract was not inspected; set "
          "CANON_ENGINE_MODULE_C=1 before constructing the rollout"
      )
    return dataclasses.asdict(self._canonical_engine_contract)

  def run_p28_segmented_forward_gate(self):
    """Runs the default-off P28 forward-only depth-boundary probe."""
    if self._canonical_engine_adapter is None:
      raise RuntimeError(
          "P28 segmented forward requires the canonical engine adapter"
      )
    return self._canonical_engine_adapter.run_p28_segmented_forward_gate()

  def run_p28_block_vjp_gate(self, *, layer_index=0):
    """Runs the default-off P28 one-real-layer VJP probe."""
    if self._canonical_engine_adapter is None:
      raise RuntimeError("P28 block VJP requires the canonical engine adapter")
    return self._canonical_engine_adapter.run_p28_block_vjp_gate(
        layer_index=layer_index
    )

  def run_p28_full_chain_gate(self):
    """Runs the default-off P28 36-layer staged-pullback capacity gate."""
    if self._canonical_engine_adapter is None:
      raise RuntimeError("P28 full chain requires the canonical engine adapter")
    return self._canonical_engine_adapter.run_p28_full_chain_gate()

  @property
  def mesh(self) -> jax.sharding.Mesh:
    return self._sampler.mesh

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates samples from the model."""
    effective_sampling = {
        "temperature": rollout_config.temperature,
        "top_p": rollout_config.top_p,
        "top_k": rollout_config.top_k,
    }
    # VllmSampler applies config-level sampling kwargs and then call-level kwargs after the
    # explicit arguments, so record parameters in exactly the same precedence order.
    effective_sampling.update(self._rollout_sampling_kwargs)
    effective_sampling.update(kwargs)
    self._last_sampling_transforms = effective_sampling
    self.output = self._sampler(
        input_strings=prompts,
        max_generation_steps=rollout_config.max_tokens_to_generate,
        max_prompt_length=rollout_config.max_prompt_length,
        temperature=rollout_config.temperature,
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        seed=rollout_config.seed,
        echo=False,
        pad_output=True,
        **kwargs,
    )

    return base_rollout.RolloutOutput(
        text=self.output.text,
        logits=None,
        tokens=self.output.tokens,
        left_padded_prompt_tokens=self.output.padded_prompt_tokens,
        logprobs=self.output.logprobs,
        prompt_lengths=self.output.prompt_lengths,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    """Returns per-token log probabilities from the rollout policy.

    NOTE: these are the logprobs CACHED during decode (`S_decode`). They are *not* a
    re-score: every value here was produced by the incremental q=1 decode program. Do not
    pass them where `S_prefill` is expected -- see `get_prefill_rescore_logps`.
    """
    if self.output.logprobs is None:
      return jax.numpy.empty((0,))
    return jax.numpy.array(self.output.logprobs)

  def get_prefill_rescore_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      reset_prefix_cache: bool = True,
      processed: bool = True,
      completion_lengths: np.ndarray | None = None,
      diagnostic_arm: str | None = None,
  ) -> np.ndarray:
    """Engine-native re-score of already-generated tokens (`S_prefill`).

    Re-submits prompt+completion to the SAME engine as one prompt with
    `prompt_logprobs=0`, so vLLM's own prefill forward scores every token, and returns the
    completion slice. With ``processed=True`` the engine-side
    ``CANON_PROMPT_PROCESSED_LOGPROBS`` path must be enabled and reuses decode's exact
    temperature/top-k/top-p processor. This is the only honest source for the processed
    `S_decode == S_prefill` boundary: aliasing `get_per_token_logps()` there compares a tensor
    with itself and makes the gate vacuous.

    Returns float32 `[B, C]` aligned with `completion_tokens`; padded positions are 0.0.
    Pass ``completion_lengths`` whenever a real sampled token can equal ``pad_id`` at the
    trailing edge; without lengths the legacy right-pad-run heuristic cannot distinguish the
    two cases.
    """
    if processed:
      if os.environ.get("CANON_PROMPT_PROCESSED_LOGPROBS", "") != "1":
        raise RuntimeError(
            "processed S_prefill requires CANON_PROMPT_PROCESSED_LOGPROBS=1 in the "
            "engine process; refusing to label raw prompt logprobs as processed"
        )
      if self._last_sampling_transforms is None:
        raise RuntimeError(
            "processed S_prefill must follow generate(); no rollout sampling provenance "
            "is available"
        )

      unsupported_defaults = {
          "presence_penalty": 0.0,
          "frequency_penalty": 0.0,
          "repetition_penalty": 1.0,
          "min_tokens": 0,
          "logit_bias": None,
          "allowed_token_ids": None,
          "bad_words": None,
          "bad_words_token_ids": None,
      }
      active_unsupported = {}
      for key, neutral in unsupported_defaults.items():
        value = self._last_sampling_transforms.get(key, neutral)
        if value not in (neutral, None, (), [], {}):
          active_unsupported[key] = value
      if active_unsupported:
        raise NotImplementedError(
            "TPU processed-prefill gate supports only temperature/top-k/top-p; "
            f"active unsupported transforms: {sorted(active_unsupported)}"
        )

      temperature = float(self._last_sampling_transforms.get("temperature", 1.0))
      top_p_value = self._last_sampling_transforms.get("top_p", 1.0)
      top_k_value = self._last_sampling_transforms.get("top_k", 0)
      top_p = 1.0 if top_p_value is None else float(top_p_value)
      top_k = 0 if top_k_value is None else int(top_k_value)
    else:
      temperature, top_p, top_k = 0.0, 1.0, 0
    pad = self.pad_id()
    prompts = np.atleast_2d(np.asarray(prompt_tokens))
    comps = np.atleast_2d(np.asarray(completion_tokens))
    if prompts.shape[0] != comps.shape[0]:
      raise ValueError(f"batch mismatch: {prompts.shape[0]} prompts vs {comps.shape[0]}")
    lengths = None
    if completion_lengths is not None:
      lengths = np.asarray(completion_lengths, dtype=np.int64).reshape(-1)
      if lengths.shape[0] != comps.shape[0]:
        raise ValueError(
            f"completion-length mismatch: {lengths.shape[0]} lengths for "
            f"{comps.shape[0]} rows"
        )
      if np.any(lengths < 0) or np.any(lengths > comps.shape[1]):
        raise ValueError(
            f"completion lengths outside [0,{comps.shape[1]}]: {lengths.tolist()}"
        )

    seqs, meta = [], []
    for row_index, (row_p, row_c) in enumerate(zip(prompts, comps)):
      # Prompts are LEFT padded and completions RIGHT padded: strip only the pad RUN at the
      # respective end, never every occurrence -- pad_id can be a legitimate interior token.
      lead = 0
      while lead < row_p.shape[0] and int(row_p[lead]) == pad:
        lead += 1
      p = [int(t) for t in row_p[lead:]]
      if lengths is None:
        trail = row_c.shape[0]
        while trail > 0 and int(row_c[trail - 1]) == pad:
          trail -= 1
      else:
        trail = int(lengths[row_index])
      c = [int(t) for t in row_c[:trail]]
      if not p:
        raise ValueError(
            "empty prompt after stripping padding: position 0 has no predictor, so its "
            "logprob does not exist and the re-score would be silently misaligned"
        )
      seqs.append(p + c)
      meta.append((len(p), trail))

    # Otherwise a cached prefix makes the 'prefill' partly a cache read, so the re-score is
    # no longer the same forward the gate claims to be comparing.  The sampler's driver-mode
    # path performs wait-idle + reset + request submission atomically, preventing another
    # continuous-batching producer from racing between the reset and this score request.
    if diagnostic_arm not in (None, "A", "B"):
      raise ValueError(f"unsupported P35 diagnostic arm: {diagnostic_arm!r}")
    previous_arm = os.environ.get("CANON_P35_ARM")
    if diagnostic_arm is not None:
      if os.environ.get("CANON_P35_ENVELOPE", "") != "1":
        raise RuntimeError(
            "a P35 diagnostic arm requires CANON_P35_ENVELOPE=1"
        )
      os.environ["CANON_P35_ARM"] = diagnostic_arm
    try:
      outputs = self._sampler.generate_request_outputs(
          [TokensPrompt(prompt_token_ids=s) for s in seqs],
          SamplingParams(
              max_tokens=1,
              temperature=temperature,
              top_p=top_p,
              top_k=top_k,
              prompt_logprobs=0,
              detokenize=False,
          ),
          reset_prefix_cache=reset_prefix_cache,
      )
    finally:
      if diagnostic_arm is not None:
        if previous_arm is None:
          os.environ.pop("CANON_P35_ARM", None)
        else:
          os.environ["CANON_P35_ARM"] = previous_arm

    self._last_prefill_rescore_provenance = {
        "processed": bool(processed),
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "batch_size": len(seqs),
        "sequence_lengths": tuple(len(s) for s in seqs),
        "diagnostic_arm": diagnostic_arm,
        "reset_prefix_cache": bool(reset_prefix_cache),
    }

    out = np.zeros(comps.shape, np.float32)
    for i, (out_i, seq, (n_p, n_c)) in enumerate(zip(outputs, seqs, meta)):
      plp = out_i.prompt_logprobs
      if plp is None or len(plp) != len(seq):
        raise RuntimeError(
            f"row {i}: engine returned {None if plp is None else len(plp)} prompt logprobs "
            f"for {len(seq)} tokens; cannot align the re-score"
        )
      if n_c:
        out[i, :n_c] = np.asarray(
            generate_utils.get_logprobs_from_vllm_output(seq[n_p:], plp[n_p:]),
            np.float32,
        )
    return out if np.asarray(completion_tokens).ndim > 1 else out[0]

  # Consumed by the alignment gate to refuse a `S_prefill` that is really a decode alias.
  get_prefill_rescore_logps.is_real_rescore = True
  get_prefill_rescore_logps.is_processed_rescore = True

  def get_grouped_prefill_rescore_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      *,
      completion_lengths: np.ndarray,
      group_size: int,
      processed: bool = True,
      source_row_indices: np.ndarray | None = None,
      diagnostic_arm: str | None = None,
  ) -> np.ndarray:
    """Re-scores fixed request groups through the native serving envelope.

    This diagnostic primitive changes only request submission grouping. Every
    group still executes through ``get_prefill_rescore_logps`` with a fresh
    prefix-cache reset. It is not used by normal rollout or training paths.
    """
    prompts = np.atleast_2d(np.asarray(prompt_tokens))
    completions = np.atleast_2d(np.asarray(completion_tokens))
    lengths = np.asarray(completion_lengths, dtype=np.int64).reshape(-1)
    if group_size <= 0:
      raise ValueError(f"group_size must be positive, got {group_size}")
    if (
        prompts.shape[0] != completions.shape[0]
        or prompts.shape[0] != lengths.size
    ):
      raise ValueError(
          "grouped prefill inputs have different row counts: "
          f"prompts={prompts.shape[0]} completions={completions.shape[0]} "
          f"lengths={lengths.size}"
      )
    if prompts.shape[0] == 0 or prompts.shape[0] % group_size:
      raise ValueError(
          "grouped prefill requires a nonempty exact number of groups: "
          f"rows={prompts.shape[0]} group_size={group_size}"
      )
    if source_row_indices is None:
      source_rows = np.arange(prompts.shape[0], dtype=np.int64)
    else:
      source_rows = np.asarray(source_row_indices, dtype=np.int64).reshape(-1)
      if source_rows.size != prompts.shape[0]:
        raise ValueError(
            "grouped prefill source-row count differs from request rows: "
            f"{source_rows.size} vs {prompts.shape[0]}"
        )
      if np.unique(source_rows).size != source_rows.size:
        raise ValueError("grouped prefill source-row indices contain duplicates")

    outputs = []
    provenance = []
    for start in range(0, prompts.shape[0], group_size):
      stop = start + group_size
      outputs.append(
          self.get_prefill_rescore_logps(
              prompts[start:stop],
              completions[start:stop],
              completion_lengths=lengths[start:stop],
              reset_prefix_cache=True,
              processed=processed,
              diagnostic_arm=diagnostic_arm,
          )
      )
      provenance.append(dict(self._last_prefill_rescore_provenance or {}))
    self._last_grouped_prefill_rescore_provenance = {
        "group_size": int(group_size),
        "groups": len(outputs),
        "rows": int(prompts.shape[0]),
        "source_row_indices": tuple(int(value) for value in source_rows),
        "diagnostic_arm": diagnostic_arm,
        "group_provenance": tuple(provenance),
    }
    return np.concatenate(outputs, axis=0)

  get_grouped_prefill_rescore_logps.is_real_rescore = True
  get_grouped_prefill_rescore_logps.is_processed_rescore = True

  def p35_grouped_prefill_contract(self) -> dict[str, Any]:
    """Returns the observed grouped-rescore provenance for P35 attestation."""
    if self._last_grouped_prefill_rescore_provenance is None:
      raise RuntimeError("P35 grouped rescore has not executed")
    return dict(self._last_grouped_prefill_rescore_provenance)

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    self._sampler.update_params(params, filter_types)

  def attest_canonical_engine_weights(self, trainer_state) -> dict[str, Any]:
    """Compares mapped trainer-anchor leaves with live engine leaves bitwise."""
    if self._canonical_engine_adapter is None:
      raise RuntimeError(
          "canonical weight attestation requires the registered engine adapter"
      )
    return self._canonical_engine_adapter.attest_exact_live_weights(
        trainer_state
    )

  def canonical_p35_adapter_contract(self) -> dict[str, Any]:
    """Returns the registered adapter's runtime P35 envelope contract."""
    if self._canonical_engine_adapter is None:
      raise RuntimeError("P35 adapter contract requires canonical engine C")
    return self._canonical_engine_adapter.p35_envelope_contract_attestation()

  def p35_exact_input_replay(
      self,
      trainer_state,
      records,
      *,
      full_prompt_tokens,
      full_completion_tokens,
      full_prompt_mask,
      full_completion_mask,
      selected_row_indices,
      temperature,
  ) -> dict[str, Any]:
    """Replays captured B tensors with live and mapped leaves."""
    if self._canonical_engine_adapter is None:
      raise RuntimeError("P35.3 exact replay requires canonical engine C")
    return self._canonical_engine_adapter.p35_exact_input_replay(
        trainer_state,
        records,
        full_prompt_tokens=full_prompt_tokens,
        full_completion_tokens=full_completion_tokens,
        full_prompt_mask=full_prompt_mask,
        full_completion_mask=full_completion_mask,
        selected_row_indices=selected_row_indices,
        pad_id=self.pad_id(),
        eos_id=self.eos_id(),
        temperature=temperature,
    )

  def pad_id(self) -> int:
    return self._sampler.tokenizer.pad_id()

  def eos_id(self) -> int:
    return self._sampler.tokenizer.eos_id()

  def model(self) -> nnx.Module:
    return self._sampler.transformer
