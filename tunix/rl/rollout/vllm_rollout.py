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
import hashlib
from typing import Any, Dict, Optional, Tuple

from flax import nnx
import jax
import jax.numpy as jnp
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


def _validated_vllm_seed_route(
    rollout_config: base_rollout.RolloutConfig,
) -> tuple[Any, Any]:
  """Returns request/engine seeds after enforcing the JAX seed contract."""
  request_seed = rollout_config.seed
  engine_seed = rollout_config.rollout_vllm_kwargs.get("seed")
  if rollout_config.rollout_vllm_tpu_backend_type == "jax":
    if request_seed is not None:
      raise ValueError(
          "vLLM JAX does not support per-request seed; set "
          "rollout_vllm_kwargs['seed'] for the global engine seed"
      )
    if engine_seed is not None and (
        not isinstance(engine_seed, int) or isinstance(engine_seed, bool)
    ):
      raise ValueError("vLLM JAX global engine seed must be an integer")
  return request_seed, engine_seed


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
    _, engine_seed = _validated_vllm_seed_route(rollout_config)
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
    if (
        rollout_config.rollout_vllm_tpu_backend_type == "jax"
        and engine_seed is not None
    ):
      print(
          "[VLLM.JAX_SEED] PASS "
          f"engine_seed={engine_seed} request_seed=none scope=engine-global",
          flush=True,
      )
    self._rollout_sampling_kwargs = dict(
        rollout_config.rollout_vllm_sampling_kwargs
    )
    self._last_sampling_transforms: dict[str, Any] | None = None
    self._recorded_sampling_transforms: dict[str, Any] | None = None
    self._recorded_sampling_source: str | None = None
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
          trainer_state=state,
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
      prompts: list[str] | list[list[int]],
      rollout_config: base_rollout.RolloutConfig,
      request_timeout_s: float | None = None,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates samples from the model."""
    if getattr(self, "_recorded_sampling_transforms", None) is not None:
      raise RuntimeError(
          "live generate cannot follow recorded sampling provenance"
      )
    request_seed, _ = _validated_vllm_seed_route(rollout_config)
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
        seed=request_seed,
        echo=False,
        pad_output=True,
        request_timeout_s=request_timeout_s,
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

  def set_recorded_sampling_transforms(
      self, transforms: dict[str, Any], *, source_identity: str
  ) -> None:
    """Primes processed re-score from one signed recorded-rollout source."""
    required_env = {
        "CANON_DEEPSWE_ONEHOST_SMOKE": "1",
        "CANON_P58_Q4_TP4_ZERO_ADMISSION": "1",
        "CANON_P58_Q4_TP4_SHORT_BACKWARD": "1",
        "CANON_P58_Q4_TP4_TRAJECTORY_REPLAY": "1",
    }
    wrong = {
        key: os.environ.get(key)
        for key, expected in required_env.items()
        if os.environ.get(key) != expected
    }
    expected_source = (
        "p58s22lr3_20260829t2256z@"
        "16c224aa80eb6b3a544be19f693c0542ab4b0dcb:rows7,0x2:B2G2"
    )
    expected_transforms = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    }
    if wrong:
      raise RuntimeError(
          f"recorded sampling provenance used outside signed replay: {wrong}"
      )
    if source_identity != expected_source:
      raise ValueError("recorded sampling source identity changed")
    if transforms != expected_transforms:
      raise ValueError(
          f"recorded sampling transforms changed: {transforms}"
      )
    if (
        getattr(self, "_last_sampling_transforms", None) is not None
        or getattr(self, "_recorded_sampling_transforms", None) is not None
    ):
      raise RuntimeError("sampling provenance was already initialized")
    self._recorded_sampling_transforms = dict(transforms)
    self._recorded_sampling_source = source_identity
    print(
        "[P58.23.REPLAY] SAMPLING_PROVENANCE_PASS "
        "temperature=1.0 top_p=1.0 top_k=0 "
        f"source={source_identity}",
        flush=True,
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
    processed-logprob path must be enabled: the canonical engine for zero-TIM,
    or the separately signed observer-only stock path for P58 native. Both
    reuse decode's exact temperature/top-k/top-p processor. This is the only honest source for the processed
    `S_decode == S_prefill` boundary: aliasing `get_per_token_logps()` there compares a tensor
    with itself and makes the gate vacuous.

    Returns float32 `[B, C]` aligned with `completion_tokens`; padded positions are 0.0.
    Pass ``completion_lengths`` whenever a real sampled token can equal ``pad_id`` at the
    trailing edge; without lengths the legacy right-pad-run heuristic cannot distinguish the
    two cases.
    """
    canonical_processed = (
        os.environ.get("CANON_PROMPT_PROCESSED_LOGPROBS", "") == "1"
    )
    p58_stock_observer_requested = (
        os.environ.get("CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER", "") == "1"
    )
    p58_native_requested = (
        os.environ.get("CANON_P58_TIM_ARM", "") == "native"
        or os.environ.get("CANON_P58_ONEHOST_XPROF_ARM", "") == "native"
    )
    p58_onehost_stock_observer = (
        os.environ.get("CANON_P58_ONEHOST_XPROF_ARM", "") == "native"
        and os.environ.get("CANON_DEEPSWE_ONEHOST_SMOKE", "0") == "1"
        and os.environ.get("CANON_DEEPSWE_ONEHOST_STAGE", "")
        == "backward-no-commit"
        and os.environ.get("CANON_DEEPSWE_ONEHOST_NO_COMMIT", "0") == "1"
        and os.environ.get("CANON_P58_DEEPSWE_TIM", "0") == "0"
    )
    p58_stock_observer_signed = (
        p58_native_requested
        and (
            (
                os.environ.get("CANON_P34_DEEPSWE", "") == "1"
                and os.environ.get("CANON_P58_DEEPSWE_TIM", "") == "1"
                and os.environ.get("CANON_P58_TIM_ADMITTED", "") == "1"
                and os.environ.get("CANON_PROFILE_FILE", "")
                == "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env"
            )
            or p58_onehost_stock_observer
        )
        and os.environ.get("CANON_ENGINE_MODULE_C", "") == "0"
        and os.environ.get("CANON_PROMPT_PROCESSED_LOGPROBS", "") == "0"
        and p58_stock_observer_requested
    )
    if p58_native_requested and not p58_stock_observer_signed:
      raise RuntimeError(
          "P58 native S_prefill requires the signed stock prompt observer "
          "while every canonical processed-logprob switch remains off"
      )
    if p58_stock_observer_requested and not p58_stock_observer_signed:
      raise RuntimeError(
          "P58 native stock prompt observer used outside its signed arm"
      )
    if processed and not canonical_processed and not p58_stock_observer_signed:
      raise RuntimeError(
          "processed S_prefill requires either the canonical processed "
          "engine path or the signed P58 native stock observer"
      )
    pad = self.pad_id()
    prompts = np.atleast_2d(np.asarray(prompt_tokens))
    comps = np.atleast_2d(np.asarray(completion_tokens))
    if prompts.shape[0] != comps.shape[0]:
      raise ValueError(
          f"batch mismatch: {prompts.shape[0]} prompts vs {comps.shape[0]}"
      )
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
            f"completion lengths outside [0,{comps.shape[1]}]: "
            f"{lengths.tolist()}"
        )

    seqs, meta = [], []
    for row_index, (row_p, row_c) in enumerate(zip(prompts, comps)):
      # Prompts are LEFT padded and completions RIGHT padded: strip only the
      # pad RUN at the respective end, never every occurrence -- pad_id can be
      # a legitimate interior token.
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
            "empty prompt after stripping padding: position 0 has no "
            "predictor, so its logprob does not exist and the re-score would "
            "be silently misaligned"
        )
      seqs.append(p + c)
      meta.append((len(p), trail))

    if diagnostic_arm not in (None, "A", "B"):
      raise ValueError(f"unsupported P35 diagnostic arm: {diagnostic_arm!r}")

    # There is no numerical S_prefill value to observe when the complete
    # batch has zero completion targets. In particular, an all-reset-timeout
    # DeepSWE batch never called generate(), so requiring decode sampling
    # provenance here would reject the preregistered compact-filter no-commit
    # path. Return padding zeros without invoking the engine and record that
    # fact explicitly; any non-empty row still requires real generation
    # provenance and a real engine re-score below.
    if all(completion_length == 0 for _, completion_length in meta):
      processor = (
          "p58-native-stock-observer"
          if p58_stock_observer_signed
          else "canonical-processed"
          if canonical_processed
          else "stock-raw"
      )
      self._last_prefill_rescore_provenance = {
          "processed": bool(processed),
          "processor": processor,
          "temperature": None,
          "top_p": None,
          "top_k": None,
          "batch_size": len(seqs),
          "sequence_lengths": tuple(len(s) for s in seqs),
          "completion_targets": 0,
          "diagnostic_arm": diagnostic_arm,
          "reset_prefix_cache": False,
          "engine_called": False,
          "skip_reason": "empty-completion-batch",
      }
      print(
          "[CANON_RESCORE] empty_completion_batch targets=0 "
          f"rows={len(seqs)} engine_called=0",
          flush=True,
      )
      out = np.zeros(comps.shape, np.float32)
      return out if np.asarray(completion_tokens).ndim > 1 else out[0]

    sampling_source = "unprocessed"
    if processed:
      live_sampling = getattr(self, "_last_sampling_transforms", None)
      recorded_sampling = getattr(self, "_recorded_sampling_transforms", None)
      if live_sampling is not None and recorded_sampling is not None:
        raise RuntimeError("live and recorded sampling provenance overlap")
      sampling_transforms = live_sampling or recorded_sampling
      if sampling_transforms is None:
        raise RuntimeError(
            "processed S_prefill must follow generate(); no rollout sampling provenance "
            "is available"
        )
      sampling_source = (
          "live-generate" if live_sampling is not None else "recorded-replay"
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
        value = sampling_transforms.get(key, neutral)
        if value not in (neutral, None, (), [], {}):
          active_unsupported[key] = value
      if active_unsupported:
        raise NotImplementedError(
            "TPU processed-prefill gate supports only temperature/top-k/top-p; "
            f"active unsupported transforms: {sorted(active_unsupported)}"
        )

      temperature = float(sampling_transforms.get("temperature", 1.0))
      top_p_value = sampling_transforms.get("top_p", 1.0)
      top_k_value = sampling_transforms.get("top_k", 0)
      top_p = 1.0 if top_p_value is None else float(top_p_value)
      top_k = 0 if top_k_value is None else int(top_k_value)
    else:
      temperature, top_p, top_k = 0.0, 1.0, 0
    # Otherwise a cached prefix makes the 'prefill' partly a cache read, so the re-score is
    # no longer the same forward the gate claims to be comparing.  The sampler's driver-mode
    # path performs wait-idle + reset + request submission atomically, preventing another
    # continuous-batching producer from racing between the reset and this score request.
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
        "processor": (
            "p58-native-stock-observer"
            if p58_stock_observer_signed
            else "canonical-processed"
            if canonical_processed
            else "stock-raw"
        ),
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "sampling_provenance": sampling_source,
        "recorded_sampling_source": getattr(
            self, "_recorded_sampling_source", None
        ),
        "batch_size": len(seqs),
        "sequence_lengths": tuple(len(s) for s in seqs),
        "diagnostic_arm": diagnostic_arm,
        "reset_prefix_cache": bool(reset_prefix_cache),
        "num_cached_tokens": tuple(
            int(output.num_cached_tokens or 0) for output in outputs
        ),
    }
    m15_b_contract = (
        os.environ.get("CANON_APC_M15_TARGET_DEBUG", "") in ("off", "on")
        or (
            os.environ.get("CANON_M15_TOKEN_CONTINUITY", "") == "exact"
            and os.environ.get("CANON_P38_ONEHOST_REHEARSAL", "0") == "1"
            and os.environ.get("CANON_P57_WORKLOAD_CANDIDATE", "") == "m15"
        )
    )
    if m15_b_contract:
      cached_tokens = self._last_prefill_rescore_provenance[
          "num_cached_tokens"
      ]
      if not reset_prefix_cache or any(cached_tokens):
        raise RuntimeError(
            "M15 APC B arm is not an independent full-reset judge: "
            f"reset_prefix_cache={reset_prefix_cache!r} "
            f"num_cached_tokens={cached_tokens!r}"
        )
      print(
          "[CAN" "ON_APC_M15_B_CONTRACT] reset_prefix_cache=True "
          "all_num_cached_tokens_zero=True",
          flush=True,
      )

    # A short prompt-logprob array is an engine accounting failure, not a
    # numerical one: the TPU runner accumulates prompt logprobs across
    # prefill chunks and emits them once, on the chunk it marks last, so a
    # request whose registration is lost before that chunk yields the
    # processor's bare seed (length one) with no tensors at all.  Retry the
    # affected rows alone -- a single-row batch is scheduled differently and
    # the value, if produced, is the same deterministic prefill -- and keep
    # the fail-closed length check on the retry.  Rows that answered
    # correctly the first time are never resubmitted.
    outputs = list(outputs)
    short_rows = [
        i
        for i, (out_i, seq) in enumerate(zip(outputs, seqs))
        if out_i.prompt_logprobs is None
        or len(out_i.prompt_logprobs) != len(seq)
    ]
    if short_rows:
      print(
          "[RESCORE.RETRY] rows="
          + ",".join(
              f"{i}:{None if outputs[i].prompt_logprobs is None else len(outputs[i].prompt_logprobs)}"
              f"/{len(seqs[i])}"
              for i in short_rows
          ),
          flush=True,
      )
      for i in short_rows:
        retry = self._sampler.generate_request_outputs(
            [TokensPrompt(prompt_token_ids=seqs[i])],
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
        outputs[i] = retry[0]
      self._last_prefill_rescore_provenance["retried_rows"] = tuple(short_rows)

    out = np.zeros(comps.shape, np.float32)
    for i, (out_i, seq, (n_p, n_c)) in enumerate(zip(outputs, seqs, meta)):
      plp = out_i.prompt_logprobs
      if plp is None or len(plp) != len(seq):
        raise RuntimeError(
            f"row {i}: engine returned {None if plp is None else len(plp)} prompt logprobs "
            f"for {len(seq)} tokens; cannot align the re-score"
            + (" (persisted after a single-row retry)" if i in short_rows else "")
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

  def run_p3_apc_boundary_probe(self) -> dict[str, Any]:
    """Compares real APC-hit decode against the full-reset B rescore.

    This is a default-off, forward-only Phase 3 discriminator.  It changes no
    engine arithmetic.  A primes the real vLLM prefix cache, decodes exactly 16
    tokens, and reads the production sampled-token logprobs.  B invokes the
    production processed-prefill rescore with ``reset_prefix_cache=True`` on
    those exact A-returned token IDs.  A intentionally does not request prompt
    logprobs because pinned vLLM makes such requests skip prefix-cache reads.
    """
    if not os.environ.get("CANON_P3_APC_BOUNDARY_REPORT", ""):
      raise RuntimeError(
          "P3 APC boundary probe requires CANON_P3_APC_BOUNDARY_REPORT"
      )
    if os.environ.get("CANON_P38_PRECHECK_ONLY", "") != "1":
      raise RuntimeError("P3 APC boundary probe is restricted to gate-only runs")

    prefix_lengths = (
        1535,
        1536,
        1537,
        1685,
        1686,
        1687,
        1788,
        1792,
        2047,
        2048,
        2049,
    )
    target_length = 16
    sampling = {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 0,
    }
    dirty_page_raw = os.environ.get("CANON_P3_APC_DIRTY_PAGE", "")
    if dirty_page_raw not in ("", "0", "1"):
      raise RuntimeError(
          "CANON_P3_APC_DIRTY_PAGE must be absent, empty, 0, or 1"
      )
    dirty_page_enabled = dirty_page_raw == "1"
    if dirty_page_enabled and os.environ.get(
        "CANON_VLLM_ENABLE_PREFIX_CACHING", "0"
    ) != "1":
      raise RuntimeError("P3 dirty-page control requires APC enabled")
    dirty_page_target_prefix = prefix_lengths[0]
    dirty_page_control = None

    def sampling_params(
        *, max_tokens: int, sampled_logprobs: int | None, ignore_eos: bool
    ) -> SamplingParams:
      return SamplingParams(
          max_tokens=max_tokens,
          temperature=sampling["temperature"],
          top_p=sampling["top_p"],
          top_k=sampling["top_k"],
          logprobs=sampled_logprobs,
          prompt_logprobs=None,
          detokenize=False,
          ignore_eos=ignore_eos,
      )

    cases = []
    for prefix_length in prefix_lengths:
      prefix_tokens = np.arange(
          1000, 1000 + prefix_length, dtype=np.int32
      )

      prime_output = self._sampler.generate_request_outputs(
          [TokensPrompt(prompt_token_ids=prefix_tokens.tolist())],
          sampling_params=sampling_params(
              max_tokens=1, sampled_logprobs=None, ignore_eos=True
          ),
          reset_prefix_cache=True,
      )[0]
      if dirty_page_enabled and prefix_length == dirty_page_target_prefix:
        dirty_page_control = self._p3_dirty_one_cached_page()
      cached_params = sampling_params(
          max_tokens=target_length, sampled_logprobs=1, ignore_eos=True
      )
      if cached_params.skip_reading_prefix_cache is not False:
        raise RuntimeError(
            "P3 APC A arm is not cache-readable: "
            f"skip_reading_prefix_cache={cached_params.skip_reading_prefix_cache}"
        )
      cached_output = self._sampler.generate_request_outputs(
          [TokensPrompt(prompt_token_ids=prefix_tokens.tolist())],
          sampling_params=cached_params,
          reset_prefix_cache=False,
      )[0]
      if len(cached_output.outputs) != 1:
        raise RuntimeError(
            "P3 APC cached arm returned an invalid output count: "
            f"prefix={prefix_length} got={len(cached_output.outputs)} expected=1"
        )
      completion = cached_output.outputs[0]
      target_tokens = np.asarray(completion.token_ids, dtype=np.int32)
      if target_tokens.size != target_length:
        raise RuntimeError(
            "P3 APC cached arm returned an invalid completion length: "
            f"prefix={prefix_length} got={target_tokens.size} "
            f"expected={target_length}"
        )
      cached_logps = np.asarray(
          generate_utils.get_logprobs_from_vllm_output(
              target_tokens.tolist(),
              completion.logprobs,
          ),
          dtype=np.float32,
      )
      if cached_logps.size != target_length:
        raise RuntimeError(
            "P3 APC cached arm returned an invalid sampled-logprob length: "
            f"prefix={prefix_length} got={cached_logps.size} "
            f"expected={target_length}"
        )

      # The direct cached request used the same processed sampling transforms;
      # record that real provenance before calling the unchanged production B.
      self._last_sampling_transforms = dict(sampling)
      full_logps = np.asarray(
          self.get_prefill_rescore_logps(
              prefix_tokens[None, :],
              target_tokens[None, :],
              reset_prefix_cache=True,
              processed=True,
              completion_lengths=np.asarray([len(target_tokens)], np.int64),
          )[0],
          dtype=np.float32,
      )
      b_provenance = dict(self._last_prefill_rescore_provenance or {})
      b_cached_tokens = tuple(b_provenance.get("num_cached_tokens", ()))
      if b_provenance.get("reset_prefix_cache") is not True:
        raise RuntimeError("P3 APC B arm did not attest reset_prefix_cache=True")
      if b_cached_tokens != (0,):
        raise RuntimeError(
            "P3 APC B arm was not a full recompute: "
            f"num_cached_tokens={b_cached_tokens}"
        )

      cached_bytes = cached_logps.view(np.uint8)
      full_bytes = full_logps.view(np.uint8)
      byte_diff = cached_bytes != full_bytes
      element_diff = cached_logps.view(np.uint32) != full_logps.view(np.uint32)
      element_indices = np.flatnonzero(element_diff)
      first_index = int(element_indices[0]) if element_indices.size else None
      input_sha = hashlib.sha256()
      input_sha.update(prefix_tokens.tobytes())
      input_sha.update(target_tokens.tobytes())
      case = {
          "prefix_length": prefix_length,
          "target_length": int(target_tokens.size),
          "target_tokens": target_tokens.tolist(),
          "target_sha256": hashlib.sha256(target_tokens.tobytes()).hexdigest(),
          "input_sha256": input_sha.hexdigest(),
          "prime_num_cached_tokens": int(prime_output.num_cached_tokens or 0),
          "a_num_cached_tokens": int(cached_output.num_cached_tokens or 0),
          "b_num_cached_tokens": int(b_cached_tokens[0]),
          "b_reset_prefix_cache": True,
          "finite": bool(
              np.isfinite(cached_logps).all() and np.isfinite(full_logps).all()
          ),
          "differing_bytes": int(np.count_nonzero(byte_diff)),
          "differing_elements": int(element_indices.size),
          "first_mismatch": (
              None
              if first_index is None
              else {
                  "target_index": first_index,
                  "a": float(cached_logps[first_index]),
                  "b": float(full_logps[first_index]),
              }
          ),
          "a_sha256": hashlib.sha256(cached_logps.tobytes()).hexdigest(),
          "b_sha256": hashlib.sha256(full_logps.tobytes()).hexdigest(),
      }
      cases.append(case)
      print(
          "[P3_APC_BOUNDARY_CASE] "
          f"prefix={prefix_length} a_cached={case['a_num_cached_tokens']} "
          f"b_cached={case['b_num_cached_tokens']} "
          f"differing_bytes={case['differing_bytes']}",
          flush=True,
      )

    return {
        "schema": "phase3-apc-boundary-probe-v2",
        "apc_enabled": os.environ.get(
            "CANON_VLLM_ENABLE_PREFIX_CACHING", "0"
        ) == "1",
        "topology": "DP1xTP4",
        "canonical_m": 256,
        "sampling": sampling,
        "token_source": "fixed-arange-prefix-v1:a-decode-completion-v1",
        "a_request_contract": {
            "max_tokens": target_length,
            "sampled_logprobs": 1,
            "prompt_logprobs": None,
            "skip_reading_prefix_cache": False,
            "ignore_eos": True,
        },
        "prefix_lengths": list(prefix_lengths),
        "cases": cases,
        "backward": 0,
        "optimizer_commits": 0,
        "claim": "G-A boundary reproduction only; not an APC certification",
        "dirty_page_control": {
            "enabled": dirty_page_enabled,
            "target_prefix_length": (
                dirty_page_target_prefix if dirty_page_enabled else None
            ),
            "page": dirty_page_control,
        },
    }

  def _p3_dirty_one_cached_page(self) -> dict[str, Any]:
    """Corrupts one real cached KV page for the Phase3 negative control.

    This hook is inaccessible unless the explicit diagnostic flag and the
    gate-only carrier are both enabled. It runs only while the in-process
    driver is idle, takes the first prefix-hash entry (the first logical block
    reused by A), and replaces that physical page in layer 0. B remains the
    unchanged full-reset recompute.
    """
    if os.environ.get("CANON_P3_APC_DIRTY_PAGE", "") != "1":
      raise RuntimeError("P3 dirty-page mutation was not explicitly enabled")
    if os.environ.get("CANON_P38_PRECHECK_ONLY", "") != "1":
      raise RuntimeError("P3 dirty-page mutation is restricted to gate-only")

    driver = self._sampler._driver  # pylint: disable=protected-access
    if driver is None:
      raise RuntimeError("P3 dirty-page mutation requires in-process vLLM")
    with driver._engine_lock:  # pylint: disable=protected-access
      if (
          driver._pending  # pylint: disable=protected-access
          or driver._submission_queue  # pylint: disable=protected-access
          or driver.llm_engine.has_unfinished_requests()
      ):
        raise RuntimeError("P3 dirty-page mutation requires an idle driver")
      engine_core = getattr(driver.llm_engine.engine_core, "engine_core", None)
      if engine_core is None:
        raise RuntimeError("P3 dirty-page mutation requires local EngineCore")
      scheduler = engine_core.scheduler
      manager = scheduler.kv_cache_manager
      if manager.num_kv_cache_groups != 1:
        raise RuntimeError(
            "P3 dirty-page mutation requires one KV-cache group, got "
            f"{manager.num_kv_cache_groups}"
        )
      cached_map = manager.block_pool.cached_block_hash_to_block._cache
      if not cached_map:
        raise RuntimeError("P3 dirty-page mutation found no cached prefix block")
      first_entry = next(iter(cached_map.values()))
      if isinstance(first_entry, dict):
        if not first_entry:
          raise RuntimeError("P3 dirty-page cache map contains an empty entry")
        block = next(iter(first_entry.values()))
      else:
        block = first_entry
      block_id = int(block.block_id)
      block_hash_tokens = block.block_hash_num_tokens
      if block_hash_tokens is None or int(block_hash_tokens) <= 0:
        raise RuntimeError("P3 dirty-page block has no logical token extent")

      runner = self._sampler._model_runner  # pylint: disable=protected-access
      if not runner.kv_caches:
        raise RuntimeError("P3 dirty-page mutation found no live KV cache")
      cache = runner.kv_caches[0]
      if len(cache.shape) != 5 or jnp.dtype(cache.dtype) != jnp.bfloat16:
        raise RuntimeError(
            "P3 dirty-page cache geometry drifted: "
            f"shape={cache.shape} dtype={cache.dtype}"
        )
      if not 0 <= block_id < int(cache.shape[0]):
        raise RuntimeError(
            f"P3 dirty-page block {block_id} exceeds cache {cache.shape}"
        )

      before = np.asarray(jax.device_get(cache[block_id]))
      before_bytes = before.tobytes()
      fill_value = 0.0 if any(before_bytes) else 1.0

      def replace_page(value):
        return value.at[block_id].set(
            jnp.asarray(fill_value, dtype=value.dtype)
        )

      dirty_cache = jax.jit(replace_page, donate_argnums=(0,))(cache)
      dirty_cache.block_until_ready()
      runner.kv_caches[0] = dirty_cache
      after = np.asarray(jax.device_get(dirty_cache[block_id]))
      after_bytes = after.tobytes()
      before_words = np.frombuffer(before_bytes, dtype=np.uint16)
      after_words = np.frombuffer(after_bytes, dtype=np.uint16)
      differing_bytes = int(np.count_nonzero(
          np.frombuffer(before_bytes, dtype=np.uint8)
          != np.frombuffer(after_bytes, dtype=np.uint8)
      ))
      differing_elements = int(np.count_nonzero(before_words != after_words))
      if differing_bytes <= 0 or differing_elements <= 0:
        raise RuntimeError("P3 dirty-page mutation did not change the page")
      result = {
          "layer_index": 0,
          "physical_block_id": block_id,
          "logical_token_extent": int(block_hash_tokens),
          "page_shape": [int(value) for value in after.shape],
          "page_dtype": str(after.dtype),
          "mutation": "fill-zero" if fill_value == 0.0 else "fill-one",
          "before_sha256": hashlib.sha256(before_bytes).hexdigest(),
          "after_sha256": hashlib.sha256(after_bytes).hexdigest(),
          "differing_bytes": differing_bytes,
          "differing_elements": differing_elements,
      }
      print(
          "[P3_APC_DIRTY_PAGE] "
          f"layer=0 block={block_id} tokens={block_hash_tokens} "
          f"mutation={result['mutation']} "
          f"differing_bytes={differing_bytes} "
          f"differing_elements={differing_elements}",
          flush=True,
      )
      return result

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

  def attest_exact_engine_weights(self, trainer_state) -> dict[str, Any]:
    """Attests live weights without changing the selected numerical arm."""
    p58_native_requested = (
        os.environ.get("CANON_P58_TIM_ARM", "") == "native"
    )
    p58_native_signed = (
        os.environ.get("CANON_P34_DEEPSWE", "") == "1"
        and p58_native_requested
        and os.environ.get("CANON_P58_DEEPSWE_TIM", "") == "1"
        and os.environ.get("CANON_P58_TIM_ADMITTED", "") == "1"
        and os.environ.get("CANON_ENGINE_MODULE_C", "") == "0"
    )
    if p58_native_requested:
      if not p58_native_signed or self._canonical_engine_adapter is not None:
        raise RuntimeError(
            "signed P58 native stock weight attestation forbids a registered "
            "canonical adapter and requires the exact admitted environment"
        )
      from tunix.rl import (  # pylint: disable=g-import-not-at-top
          canonical_qwen3_adapter,
      )

      return canonical_qwen3_adapter.attest_exact_live_engine_weights(
          sampler=self._sampler,
          trainer_state=trainer_state,
      )
    if self._canonical_engine_adapter is not None:
      return self._canonical_engine_adapter.attest_exact_live_weights(
          trainer_state
      )
    raise RuntimeError(
        "exact stock-engine weight attestation is admitted only for the "
        "signed P58 native arm"
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
